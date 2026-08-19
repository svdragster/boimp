//! CPU mip pyramid for a baked CROSS atlas.
//!
//! # Why this exists
//!
//! `cross_bindings::cross_single_sample` is ONE `textureLoad` with no filtering — the octahedral
//! path's `MAX_MIP_TAPS` software box filter was deliberately left out (see the module header in
//! `cross_bindings.wgsl`). That is the right call for cost and the wrong one for a minified
//! imposter: a 512-texel tile covering a handful of screen pixels is point-sampled, so each pixel
//! picks one arbitrary texel and picks a *different* one as soon as the camera moves. The result
//! sparkles, and it sparkles hardest exactly where cross imposters are used most — a whole forest
//! seen from far away.
//!
//! Hardware mips cannot fix it. The atlas is `Rg32Uint` with bit-packed props (see `shared.wgsl`
//! `pack_props`/`unpack_props`), so averaging raw texel bits is meaningless, integer textures are
//! not filterable, and WGSL has no `textureSampleLevel` for them. The pyramid therefore has to be
//! built in UNPACKED space and addressed explicitly with `textureLoad(t, coords, level)`.
//!
//! This is the plan `boimp::bindings` has carried as a comment ever since the octahedral profiling
//! ("PERF / PROPER FIX (not yet implemented)"), applied to the cross path first because it is
//! strictly easier here and strictly cheaper:
//!
//! * The octahedral atlas is a `grid_size²` grid (≈361 tiles) whose tile borders a naive halving
//!   would smear together. A cross atlas is a **2×2 grid of power-of-two tiles**, so halving the
//!   whole image halves each tile and the tile boundary stays exactly on the halfway texel at every
//!   level — no per-tile clamping, no bleed. See [`build_cross_mip_pyramid`] for the proof.
//! * The octahedral path would be *replacing* a 15-tap loop. The cross path is 1 tap today and
//!   stays 1 tap, so this buys the filtering for free at render time and costs only the usual
//!   +33% VRAM for the chain.
//!
//! # Where it runs
//!
//! On the CPU, off the bake's readback image, because the readback already happens — the trim
//! measurement in `cross_bake.rs` consumes exactly the same `Image`. Doing it on the GPU would mean
//! either a compute pass (`Rg32Uint` cannot take `STORAGE_BINDING` on WebGL2 — see `bake.rs`) or
//! threading per-mip render targets through the bake node, for a one-time cost that is already
//! paid asynchronously.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

/// Bytes per `Rg32Uint` texel.
const TEXEL_BYTES: usize = 8;

/// One texel's worth of material properties, unpacked.
///
/// Mirrors `UnpackedMaterialProps` in `shared.wgsl`. Deliberately a separate, private type rather
/// than something shared: it exists only to be averaged, and the *only* contract that matters is
/// that [`unpack_props`]/[`pack_props`] here stay bit-identical to their WGSL twins.
#[derive(Clone, Copy, Default)]
struct Props {
    rgba: Vec4,
    normal: Vec3,
    roughness: f32,
    metallic: f32,
    flags: u32,
    depth: f32,
}

// ---- bit layout, mirrored from shared.wgsl -----------------------------------------------------
// r: [0-4] r, [5-9] g, [10-14] b, [15-19] a, [20-23] roughness, [24-27] metallic, [28-31] flags
// g: [0-11] normal.x, [12-23] normal.y (octahedral uv), [24-31] depth

/// `pack_bits` from `shared.wgsl`: saturate, scale to the field's max, round, shift.
///
/// The `+ 0.5` then truncate is a round-to-nearest, and it is why this cannot be written as a
/// plain multiply-and-cast: WGSL's `u32()` truncates exactly like Rust's `as u32`, so the rounding
/// has to be explicit in both or the two drift by up to one quantisation step per field.
fn pack_bits(value: f32, offset: u32, count: u32) -> u32 {
    let mask = (1u32 << count) - 1;
    let quantised = (value.clamp(0.0, 1.0) * mask as f32 + 0.5) as u32;
    quantised.min(mask) << offset
}

/// `unpack_bits` from `shared.wgsl`.
fn unpack_bits(input: u32, offset: u32, count: u32) -> f32 {
    let mask = (1u32 << count) - 1;
    ((input >> offset) & mask) as f32 / mask as f32
}

/// `spherical_uv_from_normal` from `shared.wgsl` — octahedral encode.
fn spherical_uv_from_normal(dir: Vec3) -> Vec2 {
    let octant = Vec3::new(sign(dir.x), sign(dir.y), sign(dir.z));
    let sum = dir.dot(octant);
    if sum == 0.0 {
        return Vec2::splat(0.5);
    }
    let octahedron = dir / sum;
    let absolute = octahedron.abs();
    let folded = if octahedron.y < 0.0 {
        Vec2::new(
            octant.x * (1.0 - absolute.z),
            octant.z * (1.0 - absolute.x),
        )
    } else {
        Vec2::new(octahedron.x, octahedron.z)
    };
    (folded + Vec2::ONE) * 0.5
}

/// `spherical_normal_from_uv` from `shared.wgsl` — octahedral decode.
fn spherical_normal_from_uv(uv: Vec2) -> Vec3 {
    let x = uv.x * 2.0 - 1.0;
    let z = uv.y * 2.0 - 1.0;
    let y = 1.0 - x.abs() - z.abs();
    let n = if y < 0.0 {
        Vec3::new(sign(x) * (1.0 - z.abs()), y, sign(z) * (1.0 - x.abs()))
    } else {
        Vec3::new(x, y, z)
    };
    n.normalize_or_zero()
}

/// WGSL `sign()` semantics: `sign(0.0) == 0.0`. Rust's `f32::signum` returns 1.0 for +0.0, which
/// would fold the octahedral encode the wrong way on an axis-aligned normal.
fn sign(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn unpack_props(packed: [u32; 2]) -> Props {
    Props {
        rgba: Vec4::new(
            unpack_bits(packed[0], 0, 5),
            unpack_bits(packed[0], 5, 5),
            unpack_bits(packed[0], 10, 5),
            unpack_bits(packed[0], 15, 5),
        ),
        roughness: unpack_bits(packed[0], 20, 4).clamp(0.1, 0.9),
        metallic: unpack_bits(packed[0], 24, 4).clamp(0.1, 0.9),
        flags: packed[0] >> 28,
        normal: spherical_normal_from_uv(Vec2::new(
            unpack_bits(packed[1], 0, 12),
            unpack_bits(packed[1], 12, 12),
        )),
        depth: unpack_bits(packed[1], 24, 8) * 2.0 - 1.0,
    }
}

fn pack_props(props: Props) -> [u32; 2] {
    let normal_uv = spherical_uv_from_normal(props.normal);
    [
        pack_bits(props.rgba.x, 0, 5)
            | pack_bits(props.rgba.y, 5, 5)
            | pack_bits(props.rgba.z, 10, 5)
            | pack_bits(props.rgba.w, 15, 5)
            | pack_bits(props.roughness, 20, 4)
            | pack_bits(props.metallic, 24, 4)
            | (props.flags << 28),
        pack_bits(normal_uv.x, 0, 12)
            | pack_bits(normal_uv.y, 12, 12)
            | pack_bits(props.depth * 0.5 + 0.5, 24, 8),
    ]
}

/// `weighted_props` from `shared.wgsl`, weight fixed at 0.5.
///
/// The alpha weighting is the whole point and the reason a mip level cannot be a plain average:
/// colour must be weighted by coverage, or a texel that is 97% transparent contributes its
/// (arbitrary, usually black) RGB to the average and the canopy darkens as it recedes. Alpha and
/// depth are averaged straight, since those ARE the coverage.
fn blend_half(a: Props, b: Props) -> Props {
    let raw_wa = a.rgba.w * 0.5;
    let raw_wb = b.rgba.w * 0.5;
    let total = raw_wa + raw_wb;
    if total == 0.0 {
        // Fully transparent pair — matches the WGSL early-out, which returns a zeroed struct.
        return Props::default();
    }
    let wa = raw_wa / total;
    let wb = raw_wb / total;
    Props {
        rgba: (a.rgba.truncate() * wa + b.rgba.truncate() * wb)
            .extend(a.rgba.w * 0.5 + b.rgba.w * 0.5),
        roughness: a.roughness * wa + b.roughness * wb,
        metallic: a.metallic * wa + b.metallic * wb,
        normal: (a.normal * wa + b.normal * wb).normalize_or_zero(),
        depth: a.depth * wa + b.depth * wb,
        // Flags are categorical, so they cannot be averaged: take the majority-coverage one,
        // exactly as the WGSL does.
        flags: if wa < wb { b.flags } else { a.flags },
    }
}

/// Number of mip levels a cross atlas of this tile size supports.
///
/// Bounded by the TILE, not the image: the chain has to stop while each tile is still at least one
/// texel, because below that the 2×2 grid would collapse and neighbouring views would merge into
/// each other. `tile_size = 512` gives 10 levels (512, 256, .. 1).
pub fn cross_mip_levels(tile_size: u32) -> u32 {
    if tile_size == 0 {
        return 1;
    }
    tile_size.trailing_zeros() + 1
}

/// Build a mipped copy of a baked atlas.
///
/// `source` is a CPU-side `Rg32Uint` atlas — either the bake's readback (the same image
/// `cross_trim_from_atlas` measures) or one decoded straight out of a `.boimp` container — and
/// `tile_size` is the PER-TILE size, not the image width.
///
/// # Why halving the whole image is safe
///
/// The requirement is exactly one thing: **`tile_size` is a power of two, and the image is a whole
/// number of tiles on both axes.** Every tile boundary then sits at a multiple of `tile_size`,
/// which stays even at every level down to `log2(tile_size)`, so the 2×2 source block a
/// destination texel reads is always wholly inside one tile and never straddles a boundary. That
/// is what lets this be a plain image halving with no per-tile clamping.
///
/// Note the grid COUNT is irrelevant — a 19×19 octahedral atlas of 128-px tiles satisfies this
/// just as well as a 2×2 cross atlas of 512-px ones. What the octahedral path would additionally
/// need is a render-time LOD, which only the cross shader has.
///
/// Returns `None` (and warns) if the image is not a usable atlas, so a caller can fall back to the
/// unmipped original rather than lose the species.
pub fn build_cross_mip_pyramid(source: &Image, tile_size: u32) -> Option<Image> {
    let width = source.width();
    let height = source.height();

    if source.texture_descriptor.format != TextureFormat::Rg32Uint {
        warn!(
            "cross mips: atlas format is {:?}, expected Rg32Uint - not mipping",
            source.texture_descriptor.format
        );
        return None;
    }
    if tile_size == 0 || !tile_size.is_power_of_two() {
        warn!("cross mips: tile size {tile_size} is not a power of two - not mipping");
        return None;
    }
    // A whole number of tiles on both axes is the precondition the halving argument rests on.
    if width % tile_size != 0 || height % tile_size != 0 || width == 0 || height == 0 {
        warn!(
            "cross mips: atlas is {width}x{height}, not a whole number of {tile_size} tiles - not \
             mipping"
        );
        return None;
    }
    let Some(base) = source.data.as_ref() else {
        warn!("cross mips: atlas has no CPU data - not mipping");
        return None;
    };
    let expected = width as usize * height as usize * TEXEL_BYTES;
    if base.len() < expected {
        warn!(
            "cross mips: atlas data is {} bytes, expected {expected} - not mipping",
            base.len()
        );
        return None;
    }

    let levels = cross_mip_levels(tile_size);
    let mut data: Vec<u8> = Vec::with_capacity(expected * 4 / 3);
    data.extend_from_slice(&base[..expected]);

    let read = |src: &[u8], w: u32, x: u32, y: u32| -> [u32; 2] {
        let offset = ((y * w + x) as usize) * TEXEL_BYTES;
        [
            u32::from_le_bytes([
                src[offset],
                src[offset + 1],
                src[offset + 2],
                src[offset + 3],
            ]),
            u32::from_le_bytes([
                src[offset + 4],
                src[offset + 5],
                src[offset + 6],
                src[offset + 7],
            ]),
        ]
    };

    let mut prev_offset = 0usize;
    let mut prev_width = width;
    let mut prev_height = height;

    for _ in 1..levels {
        let dst_width = (prev_width / 2).max(1);
        let dst_height = (prev_height / 2).max(1);
        let src = &data[prev_offset..prev_offset + prev_width as usize * prev_height as usize * TEXEL_BYTES];

        let mut level = Vec::with_capacity(dst_width as usize * dst_height as usize * TEXEL_BYTES);
        for y in 0..dst_height {
            for x in 0..dst_width {
                let (sx, sy) = (x * 2, y * 2);
                let sx1 = (sx + 1).min(prev_width - 1);
                let sy1 = (sy + 1).min(prev_height - 1);
                // Reduce x-pairs then the y-pair, matching `imposter_blit.wgsl`'s order so the
                // CPU chain and the GPU multisample resolve agree on what "average" means.
                let top = blend_half(
                    unpack_props(read(src, prev_width, sx, sy)),
                    unpack_props(read(src, prev_width, sx1, sy)),
                );
                let bottom = blend_half(
                    unpack_props(read(src, prev_width, sx, sy1)),
                    unpack_props(read(src, prev_width, sx1, sy1)),
                );
                let packed = pack_props(blend_half(top, bottom));
                level.extend_from_slice(&packed[0].to_le_bytes());
                level.extend_from_slice(&packed[1].to_le_bytes());
            }
        }

        prev_offset = data.len();
        data.extend_from_slice(&level);
        prev_width = dst_width;
        prev_height = dst_height;
    }

    // `Image::new` asserts `data.len() == width * height * bpp`, i.e. exactly ONE mip level, so
    // the chain has to be attached after construction rather than passed in. Same shape as the
    // host's own `generate_mipmaps`, which extends `image.data` in place.
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data[..expected].to_vec(),
        TextureFormat::Rg32Uint,
        // RENDER_WORLD only: nothing reads this back again, and keeping the MAIN_WORLD copy would
        // double an already 11 MB asset per species.
        RenderAssetUsages::RENDER_WORLD,
    );
    image.data = Some(data);
    image.texture_descriptor.mip_level_count = levels;
    Some(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atlas(tile_size: u32, fill: impl Fn(u32, u32) -> [u32; 2]) -> Image {
        let dim = tile_size * 2;
        let mut data = Vec::with_capacity((dim * dim) as usize * TEXEL_BYTES);
        for y in 0..dim {
            for x in 0..dim {
                let texel = fill(x, y);
                data.extend_from_slice(&texel[0].to_le_bytes());
                data.extend_from_slice(&texel[1].to_le_bytes());
            }
        }
        Image::new(
            Extent3d {
                width: dim,
                height: dim,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rg32Uint,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    fn opaque(rgb: Vec3) -> [u32; 2] {
        pack_props(Props {
            rgba: rgb.extend(1.0),
            normal: Vec3::Y,
            roughness: 0.5,
            metallic: 0.1,
            flags: 0,
            depth: 0.0,
        })
    }

    #[test]
    fn pack_round_trips_through_unpack() {
        let props = Props {
            rgba: Vec4::new(0.5, 0.25, 0.75, 1.0),
            normal: Vec3::new(0.3, 0.8, -0.5).normalize(),
            roughness: 0.5,
            metallic: 0.3,
            flags: 2,
            depth: 0.25,
        };
        let back = unpack_props(pack_props(props));
        // 5-bit colour: one step is 1/31, so this is the precision the ATLAS has, not slack.
        assert!((back.rgba.x - props.rgba.x).abs() < 1.0 / 31.0);
        assert!((back.rgba.w - props.rgba.w).abs() < 1.0 / 31.0);
        assert!((back.depth - props.depth).abs() < 2.0 / 255.0);
        assert_eq!(back.flags, props.flags);
        assert!(back.normal.dot(props.normal) > 0.99, "normal {:?}", back.normal);
    }

    #[test]
    fn level_count_is_bounded_by_the_tile_not_the_image() {
        // 512 -> 10 levels: the last one has each TILE at 1x1 (image 2x2). Going further would
        // merge the side and top views into a single texel.
        assert_eq!(cross_mip_levels(512), 10);
        assert_eq!(cross_mip_levels(256), 9);
        assert_eq!(cross_mip_levels(1), 1);
    }

    #[test]
    fn a_uniform_atlas_survives_every_level_unchanged() {
        // The strongest correctness check available without a GPU: averaging a constant must give
        // that constant back at every level, which exercises pack -> unpack -> blend -> pack as a
        // closed loop and would catch any bit-offset or rounding drift.
        let colour = Vec3::new(0.5, 0.25, 0.75);
        let image = atlas(64, |_, _| opaque(colour));
        let mipped = build_cross_mip_pyramid(&image, 64).expect("should mip");
        let data = mipped.data.as_ref().unwrap();

        let mut offset = 0usize;
        let mut dim = 128u32;
        for level in 0..cross_mip_levels(64) {
            let texel = [
                u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]),
                u32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]),
            ];
            let props = unpack_props(texel);
            assert!(
                (props.rgba.truncate() - colour).length() < 1.5 / 31.0,
                "level {level} drifted to {:?}",
                props.rgba
            );
            assert!((props.rgba.w - 1.0).abs() < 1.0 / 31.0, "level {level} lost alpha");
            offset += (dim * dim) as usize * TEXEL_BYTES;
            dim /= 2;
        }
    }

    #[test]
    fn transparent_texels_do_not_darken_their_opaque_neighbours() {
        // The reason a mip level cannot be a plain average. Half the texels are transparent BLACK;
        // a straight mean would drag the colour to 50% grey-black, while coverage weighting must
        // keep the colour and halve only the alpha.
        let white = Vec3::ONE;
        let clear = pack_props(Props {
            rgba: Vec4::ZERO,
            normal: Vec3::Y,
            roughness: 0.5,
            metallic: 0.1,
            flags: 0,
            depth: 0.0,
        });
        let image = atlas(16, |x, _| if x % 2 == 0 { opaque(white) } else { clear });
        let mipped = build_cross_mip_pyramid(&image, 16).expect("should mip");
        let data = mipped.data.as_ref().unwrap();

        // First mip level starts after the 32x32 base.
        let offset = (32 * 32) as usize * TEXEL_BYTES;
        let props = unpack_props([
            u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]),
            u32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]),
        ]);
        assert!(
            props.rgba.truncate().length() > 1.5,
            "colour was dragged toward the transparent texels: {:?}",
            props.rgba
        );
        assert!(
            (props.rgba.w - 0.5).abs() < 1.5 / 31.0,
            "alpha should be the straight mean, got {}",
            props.rgba.w
        );
    }

    #[test]
    fn tile_borders_never_bleed() {
        // Tile 0 solid red, tile 1 solid blue. If the halving ever averaged across the vertical
        // tile boundary there would be a purple column at every level; the power-of-two argument
        // in the doc comment says there cannot be.
        let tile = 32u32;
        let image = atlas(tile, |x, _| {
            if x < tile {
                opaque(Vec3::X)
            } else {
                opaque(Vec3::Z)
            }
        });
        let mipped = build_cross_mip_pyramid(&image, tile).expect("should mip");
        let data = mipped.data.as_ref().unwrap();

        let mut offset = 0usize;
        let mut dim = tile * 2;
        for level in 0..cross_mip_levels(tile) {
            let half = dim / 2;
            let sample = |x: u32, y: u32| {
                let o = offset + ((y * dim + x) as usize) * TEXEL_BYTES;
                unpack_props([
                    u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]),
                    u32::from_le_bytes([data[o + 4], data[o + 5], data[o + 6], data[o + 7]]),
                ])
            };
            // Last texel of tile 0 and first of tile 1, on the boundary itself.
            let left = sample(half - 1, 0);
            let right = sample(half, 0);
            assert!(
                left.rgba.x > 0.9 && left.rgba.z < 0.1,
                "level {level}: tile 0 edge bled to {:?}",
                left.rgba
            );
            assert!(
                right.rgba.z > 0.9 && right.rgba.x < 0.1,
                "level {level}: tile 1 edge bled to {:?}",
                right.rgba
            );
            offset += (dim * dim) as usize * TEXEL_BYTES;
            dim /= 2;
        }
    }

    #[test]
    fn a_non_power_of_two_tile_is_refused_rather_than_smeared() {
        let image = atlas(24, |_, _| opaque(Vec3::ONE));
        assert!(build_cross_mip_pyramid(&image, 24).is_none());
    }
}

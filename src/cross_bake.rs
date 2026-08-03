//! Baking support for [`CrossImposter`], built entirely on top of the existing
//! [`ImposterBakeCamera`] — `bake.rs` needs no changes at all.
//!
//! The trick is `manual_camera_transforms`, which lets us replace the octahedral grid with any set
//! of views while reusing the bake's render graph, phases, pipelines and readback verbatim.

use bevy::prelude::*;

use crate::{
    bake::ImposterBakeCamera,
    cross::{CrossImposter, CrossImposterData, FULL_TILE},
    oct_coords::GridMode,
};

/// A bake camera configured to render the 2-tile cross atlas (one side view, one top view).
///
/// `center` must be the world position the bake camera entity is spawned at: `manual_camera_
/// transforms` are consumed as absolute world transforms (`bake.rs` uses them in place of the
/// generated ones and never re-centres them), so a mismatch bakes an off-centre model.
///
/// # Why `grid_size: 2` for two tiles
///
/// The bake's subview loop is `for y in 0..grid_size { for x in 0..grid_size { .. } }`, indexing
/// `manual_camera_transforms` by `y * grid_size + x`. There is no non-square grid, so the smallest
/// grid that holds 2 views is 2x2 = 4 tiles. Index 0 -> atlas tile (0,0) = side, index 1 -> atlas
/// tile (1,0) = top, which puts the two tiles we care about CONTIGUOUS IN THE TOP ROW — rows are
/// contiguous in the readback buffer, so cropping the strip out later is a plain `truncate`, and
/// the uncropped target can be sampled directly (see [`cross_imposter_from_bake_target`]).
///
/// Indices 2 and 3 are currently duplicates of the same two views. That wastes BAKE TIME (4 tiles
/// rendered to produce 2) but not VRAM at render time, since the bottom row is never sampled.
/// Phase 3 gets the second, perpendicular side view in slot 2 for free — the bake already pays for
/// it — which removes the "both verticals are the same picture" tell at 45 degrees of yaw.
pub fn cross_bake_camera(
    center: Vec3,
    radius: f32,
    tile_size: u32,
    multisample: u32,
) -> ImposterBakeCamera {
    // These MUST reproduce `normal_from_grid`'s convention exactly, because the mesh uvs in
    // `cross_imposter_mesh` are derived from that convention (see the uv comment there). In
    // particular `up` follows the `if n.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y }` rule, which is
    // why the top view uses +Z as up — looking straight down +Y, +Y is a degenerate up vector.
    let view = |normal: Vec3, up: Vec3| {
        GlobalTransform::from(
            Transform::from_translation(center + normal * radius).looking_at(center, up),
        )
    };
    let side = view(Vec3::Z, Vec3::Y);
    let top = view(Vec3::Y, Vec3::Z);

    ImposterBakeCamera {
        radius,
        grid_size: 2,
        tile_size,
        multisample,
        // Unused: `bake.rs` only consults `grid_mode` when `manual_camera_transforms` is `None`.
        // Left at the default rather than made an argument, so callers aren't misled into thinking
        // it changes the baked views.
        grid_mode: GridMode::Spherical,
        manual_camera_transforms: Some(vec![side, top, side, top]),
        ..Default::default()
    }
}

/// Phase 0 shortcut: build a [`CrossImposter`] that samples an [`ImposterBakeCamera::target`]
/// image directly, with no `.bcross` asset in between — the same way `examples/dynamic.rs` builds
/// its octahedral `Imposter` off the live bake target.
///
/// The bake target is a `2 * tile_size` square (the full 2x2 grid), NOT a cropped 2-tile strip, and
/// that works as-is: the shader's tile origin is `vec2(tile_id * packed_tile_size.x, 0)`, so with
///
/// * `packed_tile_offset = UVec2::ZERO`   (nothing has been trimmed)
/// * `packed_tile_size   = UVec2::splat(tile_size)`  (also the horizontal stride between tiles)
/// * `base_tile_size     = tile_size`
///
/// tile 0 addresses `[0, tile_size)` and tile 1 addresses `[tile_size, 2 * tile_size)` along x, both
/// in the top row. The unused bottom row is simply never sampled. Passing the target's true width
/// as `packed_tile_size.x` instead would be the classic mistake: it would stretch tile 0 across the
/// whole atlas and put tile 1 off the right-hand edge.
///
/// `indices` should be `DummyIndicesImage`'s handle — the bake target is never indexed, but bevy
/// hands out an rgba8 fallback for an unbound `u_int` texture, which fails bind-group validation.
///
/// `flags` are left at 0 (no `CROSS_ANGLE_FADE_FLAG`, etc.) so this is the clean Phase 0 perf
/// baseline; OR the `CROSS_*_FLAG` constants into the returned material's `data.flags` to opt in,
/// or use [`CrossImposterData::new`] directly for full control.
pub fn cross_imposter_from_bake_target(
    target: Handle<Image>,
    indices: Handle<Image>,
    center: Vec3,
    radius: f32,
    tile_size: u32,
    alpha_mode: AlphaMode,
) -> CrossImposter {
    CrossImposter {
        data: CrossImposterData::new(
            center,
            radius,
            tile_size,
            UVec2::ZERO,
            UVec2::splat(tile_size),
            false,
            false,
            false,
            false,
            1.0,
        ),
        pixels: target,
        indices,
        alpha_mode,
        // Nothing was allocated by us — the bake camera owns the target image.
        vram_bytes: 0,
    }
}

/// The two uv-space content boxes measured off a baked cross atlas, ready to drop into
/// [`crate::CrossLayout::side_trim`] / [`crate::CrossLayout::top_trim`].
///
/// A struct rather than a `(Rect, Rect)` because the two are not interchangeable and swapping them
/// renders a plausible-looking wrong thing: the vertical quads would take the canopy disc's box and
/// the canopy quad the tree's, both of which are perfectly valid rects that simply clip the wrong
/// picture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossTrim {
    /// Tile 0, the side view — what the VERTICAL quads sample.
    pub side: Rect,
    /// Tile 1, the top view — what the single horizontal canopy quad samples.
    pub top: Rect,
}

impl CrossTrim {
    /// No trim at all: both tiles declared full. The identity for the mesh builder (see
    /// [`crate::cross::FULL_TILE`]) and the honest answer whenever the measurement cannot be made —
    /// an unreadable readback, a mismatched image size, or a tile with nothing in it.
    pub const FULL: Self = Self {
        side: FULL_TILE,
        top: FULL_TILE,
    };
}

impl Default for CrossTrim {
    fn default() -> Self {
        Self::FULL
    }
}

/// Measures, per tile, the uv-space bounding box of everything in a baked cross atlas whose alpha
/// reaches `alpha_cutoff`.
///
/// Feed the result into [`crate::CrossLayout::side_trim`] / [`crate::CrossLayout::top_trim`] and the
/// quads shrink to the part of the bounding sphere that has anything in it — on this repo's two
/// sample atlases, from 100% of the square to 14-48% of it, with every deleted fragment one that
/// would have fetched a texel and discarded. See [`crate::CrossLayout::side_trim`] for why no
/// content can be lost, and for the one thing that does still change.
///
/// # `alpha_cutoff` is a CORRECTNESS parameter, not a quality knob
///
/// `cross_fragment.wgsl` discards `props.rgba.a * alpha < 0.01`. Passing anything at or below that
/// makes every texel this function excludes one the shader was already guaranteed to throw away, so
/// enabling the trim cannot lose content (see [`crate::CrossLayout::side_trim`] for what it CAN
/// still change, which is a resampling effect at extreme minification). Passing MORE than 0.01
/// stops being a lossless crop immediately: it starts cutting texels that
/// would have been drawn. The alpha channel is 5 bits (see below), so
/// the smallest representable non-zero value is 1/31 = 0.032 and everything in `(0, 0.032]` behaves
/// identically here — but the shader's threshold is what the argument has to be judged against, not
/// the quantisation.
///
/// # `pad_texels`
///
/// Expands the measured box outward, clamped to the tile. Pure insurance: the box is exact in
/// texels, but the quad corner it becomes travels through f32 interpolation and a `floor` in the
/// shader, so one texel of margin costs a fraction of a percent of the saving and removes the whole
/// class of "off by one at the silhouette edge" failures. 1 is a sensible default; 0 is legal and
/// exact.
///
/// # Layout of the buffer this reads
///
/// The bake target is a `2 * tile_size` square (the full 2x2 grid; see [`cross_bake_camera`]) and
/// the two tiles that matter are the TOP ROW — tile 0 over `x in [0, tile_size)` and tile 1 over
/// `[tile_size, 2 * tile_size)`, both `y in [0, tile_size)`. The bottom row is a by-design duplicate
/// and is not looked at.
///
/// Texels are `Rg32Uint`, 8 bytes each, and alpha lives in bits 15..19 of the LOW `u32` — 5 bits,
/// `/ 31.0` — exactly as `shared.wgsl`'s `pack_props` writes it and as the `--dump-atlas` decoder in
/// `examples/dynamic.rs` reads it back.
///
/// # Failure behaviour
///
/// Every failure returns [`CrossTrim::FULL`] and warns: no readback data, an image too small to
/// hold the strip, or a tile with nothing above the cutoff (an all-empty bake, which is worth a
/// warning in its own right). Never returns an inverted or zero-area rect — a rect the mesh builder
/// would turn into a degenerate quad is strictly worse than not trimming.
pub fn cross_trim_from_atlas(
    image: &Image,
    tile_size: u32,
    alpha_cutoff: f32,
    pad_texels: u32,
) -> CrossTrim {
    let Some(data) = image.data.as_ref() else {
        warn!("cross trim: baked atlas has no readback data, not trimming");
        return CrossTrim::FULL;
    };

    let (width, height) = (image.width(), image.height());
    // The strip we read is 2 tiles wide and 1 tall, wherever it sits inside the target. Checked
    // against the actual buffer length too, not just the declared size: `image.data` is the
    // readback, and a short buffer would otherwise be an index panic in a callback.
    let needed = (width as usize) * (tile_size as usize) * 8;
    if tile_size == 0 || width < tile_size * 2 || height < tile_size || data.len() < needed {
        warn!(
            "cross trim: atlas is {width}x{height} ({} bytes), too small for two {tile_size} tiles \
             - not trimming",
            data.len()
        );
        return CrossTrim::FULL;
    }

    let measure = |tile: u32, name: &str| {
        let origin = tile * tile_size;
        // Inverted on purpose: `min > max` is the "nothing found" state, and it cannot be confused
        // with a real one-texel box (which has min == max).
        let (mut x0, mut y0, mut x1, mut y1) = (tile_size, tile_size, 0u32, 0u32);
        for y in 0..tile_size {
            for x in 0..tile_size {
                let offset = ((y * width + origin + x) as usize) * 8;
                let packed = u32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                // 5 bits at 15, matching `pack_props` / the `--dump-atlas` decoder.
                let alpha = ((packed >> 15) & 0x1f) as f32 / 31.0;
                if alpha >= alpha_cutoff {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }

        if x0 > x1 || y0 > y1 {
            warn!(
                "cross trim: tile {tile} ({name}) has nothing above alpha {alpha_cutoff} - not \
                 trimming it"
            );
            return FULL_TILE;
        }

        // Outward pad, clamped to the tile: `saturating_sub` at the low end, `min` at the high end.
        let x0 = x0.saturating_sub(pad_texels);
        let y0 = y0.saturating_sub(pad_texels);
        let x1 = (x1 + pad_texels).min(tile_size - 1);
        let y1 = (y1 + pad_texels).min(tile_size - 1);

        // Texel INDICES to a uv rect on texel BOUNDARIES: texel i covers `[i, i+1)`, so the box that
        // contains texels x0..=x1 whole runs from `x0` to `x1 + 1`. Using `x1 / tile_size` instead
        // would put the rect edge through the middle of the last covered texel and clip a column of
        // real content - the classic off-by-one here, and one that looks like nothing at all until
        // it eats a tree's outermost leaves.
        let scale = 1.0 / tile_size as f32;
        Rect::new(
            x0 as f32 * scale,
            y0 as f32 * scale,
            (x1 + 1) as f32 * scale,
            (y1 + 1) as f32 * scale,
        )
    };

    CrossTrim {
        side: measure(0, "side"),
        top: measure(1, "top"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{asset::RenderAssetUsages, image::Image};

    /// Builds a `2 * tile` x `2 * tile` Rg32Uint readback with `covered` marked opaque, exactly as
    /// the bake would leave it: alpha in bits 15..19 of the low u32, 8 bytes per texel, everything
    /// else zero.
    ///
    /// `covered(tile, x, y)` is asked per TILE-LOCAL coordinate so a test can describe "a rectangle
    /// in tile 0, nothing in tile 1" without doing the strip arithmetic itself - which is the same
    /// arithmetic under test.
    fn atlas(tile: u32, covered: impl Fn(u32, u32, u32) -> bool) -> Image {
        let side = tile * 2;
        let mut data = vec![0u8; (side * side) as usize * 8];
        for y in 0..side {
            for x in 0..side {
                // Only the top row is meaningful; the bottom row is filled solid on purpose, so a
                // bug that reads past `tile_size` rows shows up as a box that runs to the bottom
                // edge instead of being silently absorbed by zeros.
                let opaque = if y < tile {
                    covered(x / tile, x % tile, y)
                } else {
                    true
                };
                if opaque {
                    let offset = ((y * side + x) as usize) * 8;
                    // alpha = 31/31 = 1.0
                    data[offset..offset + 4].copy_from_slice(&(0x1fu32 << 15).to_le_bytes());
                }
            }
        }
        Image::new(
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            wgpu::TextureDimension::D2,
            data,
            wgpu::TextureFormat::Rg32Uint,
            RenderAssetUsages::RENDER_WORLD,
        )
    }

    /// A known rectangle in each tile, at different places and different shapes, with no padding:
    /// the returned rect must be its exact texel-boundary box. Deliberately asymmetric in both axes
    /// and different per tile, so a transposed x/y or a tile-0 result leaking into tile 1 cannot
    /// pass.
    #[test]
    fn measures_the_exact_texel_boundaries() {
        // tile 0: x in 8..=23, y in 4..=59.  tile 1: x in 30..=61, y in 20..=35.
        let image = atlas(64, |tile, x, y| {
            if tile == 0 {
                (8..24).contains(&x) && (4..60).contains(&y)
            } else {
                (30..62).contains(&x) && (20..36).contains(&y)
            }
        });
        let trim = cross_trim_from_atlas(&image, 64, 0.01, 0);
        assert_eq!(
            trim.side,
            Rect::new(8.0 / 64.0, 4.0 / 64.0, 24.0 / 64.0, 60.0 / 64.0)
        );
        assert_eq!(
            trim.top,
            Rect::new(30.0 / 64.0, 20.0 / 64.0, 62.0 / 64.0, 36.0 / 64.0)
        );
    }

    /// One texel. The narrowest thing that still has to come back as a non-degenerate rect: the
    /// upper bound is `x1 + 1`, not `x1`, so a single covered texel is a 1/tile-wide box and not an
    /// empty one. (An empty one would build a zero-area quad that renders nothing.)
    #[test]
    fn a_single_texel_is_one_texel_wide() {
        let image = atlas(16, |tile, x, y| tile == 0 && x == 5 && y == 11);
        let trim = cross_trim_from_atlas(&image, 16, 0.01, 0);
        assert_eq!(
            trim.side,
            Rect::new(5.0 / 16.0, 11.0 / 16.0, 6.0 / 16.0, 12.0 / 16.0)
        );
        assert!(trim.side.width() > 0.0 && trim.side.height() > 0.0);
        // tile 1 had nothing at all -> full rect, never an inverted or empty one.
        assert_eq!(trim.top, FULL_TILE);
    }

    /// An empty tile falls back to the full rect rather than to whatever `(min, max)` the scan was
    /// left holding - which, being seeded inverted, would otherwise produce `Rect::new` swapping its
    /// arguments into a nonsense box.
    #[test]
    fn an_empty_tile_is_not_trimmed() {
        let image = atlas(16, |_, _, _| false);
        let trim = cross_trim_from_atlas(&image, 16, 0.01, 0);
        assert_eq!(trim, CrossTrim::FULL);
    }

    /// Content touching each of the four edges: the box must reach exactly 0.0 / 1.0 there and stop
    /// - no clamping artefact, no off-by-one that leaves a texel of margin the content needs.
    #[test]
    fn content_against_an_edge_reaches_the_border() {
        for (name, hit) in [
            ("left", (0u32, 8u32)),
            ("right", (15, 8)),
            ("top", (8, 0)),
            ("bottom", (8, 15)),
        ] {
            let image = atlas(16, move |tile, x, y| tile == 0 && (x, y) == hit);
            let r = cross_trim_from_atlas(&image, 16, 0.01, 0).side;
            assert_eq!(r.min.x, hit.0 as f32 / 16.0, "{name} min.x");
            assert_eq!(r.max.x, (hit.0 + 1) as f32 / 16.0, "{name} max.x");
            assert_eq!(r.min.y, hit.1 as f32 / 16.0, "{name} min.y");
            assert_eq!(r.max.y, (hit.1 + 1) as f32 / 16.0, "{name} max.y");
        }
    }

    /// Padding grows the box by exactly `pad_texels` on every side, and CLAMPS at the border rather
    /// than wrapping (`saturating_sub`) or running off the end (`min`). Checked with a pad far
    /// larger than the margin available, which is what a small tile plus a generous `--cross-trim-pad`
    /// looks like in practice.
    #[test]
    fn padding_expands_and_clamps() {
        let image = atlas(16, |tile, x, y| tile == 0 && (6..10).contains(&x) && y == 1);
        let padded = cross_trim_from_atlas(&image, 16, 0.01, 2).side;
        // x: 6..=9 grows to 4..=11 -> [4/16, 12/16]. y: 1..=1 grows to 0..=3 (clamped low).
        assert_eq!(padded, Rect::new(4.0 / 16.0, 0.0, 12.0 / 16.0, 4.0 / 16.0));

        // A pad bigger than the tile saturates to the whole tile in both directions.
        let flooded = cross_trim_from_atlas(&image, 16, 0.01, 999).side;
        assert_eq!(flooded, FULL_TILE);
    }

    /// The cutoff is compared against the DECODED 5-bit alpha, not against the raw bits. A texel at
    /// 1/31 (the smallest non-zero value the format can hold) counts at the shader's own 0.01
    /// threshold and does not at a deliberately high one.
    #[test]
    fn cutoff_is_applied_to_decoded_alpha() {
        let mut image = atlas(16, |_, _, _| false);
        let data = image.data.as_mut().unwrap();
        // tile 0, texel (3, 3), alpha = 1/31.
        let offset = ((3 * 32 + 3) as usize) * 8;
        data[offset..offset + 4].copy_from_slice(&(1u32 << 15).to_le_bytes());

        let kept = cross_trim_from_atlas(&image, 16, 0.01, 0).side;
        assert_eq!(
            kept,
            Rect::new(3.0 / 16.0, 3.0 / 16.0, 4.0 / 16.0, 4.0 / 16.0)
        );
        // 0.5 is what the BAKER thresholds at; 1/31 is well below it, so this tile reads as empty.
        assert_eq!(cross_trim_from_atlas(&image, 16, 0.5, 0).side, FULL_TILE);
    }

    /// A readback that is too small for the requested tile size returns FULL instead of indexing
    /// past the end of the buffer. This runs inside a bake callback, where a panic takes the app
    /// down with it.
    #[test]
    fn an_undersized_atlas_is_not_trimmed() {
        let image = atlas(16, |_, _, _| true);
        assert_eq!(cross_trim_from_atlas(&image, 64, 0.01, 0), CrossTrim::FULL);
    }
}

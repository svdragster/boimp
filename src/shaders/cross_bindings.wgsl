#define_import_path boimp::cross_bindings

#import boimp::shared::{
    UnpackedMaterialProps,
    unpack_props,
};

// ---------------------------------------------------------------------------------------------
// CROSS IMPOSTER bindings.
//
// This is the *cheap* counterpart to `boimp::bindings`. The octahedral path there has to derive a
// uv per fragment (camera-ray / sample-plane intersection), pick 1-3 tiles out of a grid_size^2
// atlas, and box-filter a MAX_MIP_TAPS^2 neighbourhood per tile - up to 15 `textureLoad`s per
// fragment plus derivatives. The cross path has none of that: the quad carries real interpolated
// UVs and a flat per-quad tile id, so the whole atlas addressing collapses to ONE `textureLoad`
// (two when indexed) with no derivatives and no loops. Everything omitted below - parallax,
// footprint/mip taps, tile blending, multisample - is omitted on purpose; see the per-site notes.
//
// Deliberately a SEPARATE module from `boimp::bindings` rather than a set of `#ifdef`s inside it:
// the two systems have different uniforms and different binding layouts, and a shared module would
// have to declare both, so every octahedral pipeline would carry the cross bindings and vice
// versa. Keeping them apart is what lets `CrossImposterPlugin` be usable without
// `ImposterRenderPlugin` at all.
// ---------------------------------------------------------------------------------------------

struct CrossImposterData {
    // xyz = bounding-sphere centre in the mesh's LOCAL space, w = radius (the same `radius` the
    // bake camera used). The vertex shader scales the unit (+-0.5) quad by `w * 2.0` and offsets
    // by `xyz`, exactly as the octahedral vertex shader does, so one bake drives both systems.
    //
    // Under MERGED this MUST be (0, 0, 0, 0.5): a merged patch mesh already holds final positions
    // (each tree scaled by its own radius, yawed and translated on the CPU), so the expression has
    // to degenerate to `position * 1.0 + 0`. Anything else rescales the entire forest. The per-tree
    // centre ANGLE_FADE needs comes from ATTRIBUTE_COLOR.xyz instead; see the MERGED block in
    // cross_vertex.wgsl.
    center_and_scale: vec4<f32>,
    // trim rect of the packed atlas, in texels. ONE rect shared by both tiles (see plan section 4)
    // so the fragment's coord math stays a single line; per-tile rects are a deferred optimisation.
    packed_tile_offset: vec2<u32>,
    packed_tile_size: vec2<u32>,
    // the untrimmed tile size the UVs were baked against; `uv * base_tile_size` is the texel
    // position within the *untrimmed* tile, which `packed_tile_offset` then shifts into the
    // trimmed atlas.
    base_tile_size: u32,
    flags: u32,
    alpha: f32,
    // self-illumination floor as a fraction of albedo (see the emissive-floor block in
    // cross_fragment.wgsl). 0 disables.
    emissive_floor: f32,
    // (start, end) of the |dot(view_dir, quad_normal)| band over which a quad fades out as it
    // turns edge-on. See the ANGLE_FADE block in cross_vertex.wgsl for why this exists and why
    // the defaults have to stay low. With the flag off, `fade` stays 1.0 everywhere.
    angle_fade: vec2<f32>,
    // PERF MEASUREMENT ladder, 0 = normal rendering. Mirrors `ImposterData::debug_level` so a
    // cross-vs-octahedral A/B compares like with like. A uniform and NOT a shader def, so stepping
    // the ladder never respecializes the pipeline and the levels stay directly comparable.
    debug_level: u32,
    // Highest mip level the bound atlas actually has, 0 when it has no chain at all. Set by the
    // host after `boimp::cross_mip::build_cross_mip_pyramid` builds one; see the LOD block in
    // cross_fragment.wgsl. Also a uniform and not a def, for the same reason as `debug_level`:
    // an atlas gaining mips must not fork the pipeline.
    max_lod: u32,
}

// the material bind group index moved from 2 (bevy 0.15) to 3 in 0.18; use the
// `MATERIAL_BIND_GROUP` shader def bevy injects so this tracks future moves too.
@group(#{MATERIAL_BIND_GROUP}) @binding(0)
var<uniform> cross_data: CrossImposterData;

@group(#{MATERIAL_BIND_GROUP}) @binding(1)
var cross_pixels: texture_2d<u32>;

#ifdef INDEXED_PIXELS
@group(#{MATERIAL_BIND_GROUP}) @binding(2)
var cross_indices: texture_2d<u32>;
#endif

// Varyings, defined here (not in either stage) so the vertex and fragment modules cannot drift
// apart: naga_oil compiles the two stages as separate modules, and a mismatched location list only
// shows up as a link failure at pipeline creation time, i.e. at runtime.
//
// `fade` is UNCONDITIONAL - it is 1.0 when ANGLE_FADE is off - rather than sitting behind an
// `#ifdef`. Guarding a varying location means both stages must agree on the def, which they do
// today, but it buys nothing: one extra f32 interpolant is far below the noise floor next to the
// three vec3s already here, and an unconditional field removes a whole class of "works until
// someone pushes the def to only one stage" bug.
struct CrossVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    // Per-tile uv in [0,1]^2, baked into the mesh to match the bake camera basis exactly
    // (see plan section 2.2). This single interpolated attribute is what replaces the octahedral
    // path's entire per-fragment `sample_uvs_unbounded` ray/plane intersection.
    @location(1) uv: vec2<f32>,
    // 0 = side tile, 1 = top tile. Constant per quad, hence @interpolate(flat): the fragment must
    // never see a fractional tile id, or `tile_origin` would land between tiles and the sample
    // would straddle the seam into the neighbouring view.
    @location(2) @interpolate(flat) tile: u32,
    // R^T of the instance rotation, split into columns because WGSL cannot pass a mat3x3 as a
    // varying. Used to rotate the baked LOCAL-space normals into world space in the fragment.
    @location(3) inverse_rotation_0c: vec3<f32>,
    @location(4) inverse_rotation_1c: vec3<f32>,
    @location(5) inverse_rotation_2c: vec3<f32>,
    // ANGLE_FADE weight in [0,1]; 1.0 = fully visible. Dithered (not blended) in the fragment.
    @location(6) fade: f32,
}

// ONE texture load (two when indexed), no filtering, no neighbourhood, no parallax.
//
// Adapted from `boimp::bindings::single_sample`, and deliberately NOT factored into a shared
// helper: that one reads `imposter_pixels`/`imposter_indices`, this one reads
// `cross_pixels`/`cross_indices`, and WGSL has no way to parameterise a module over a global
// binding. Duplicating ~10 lines is the cheaper of the two evils - but the INDEXED_PIXELS packing
// below MUST stay bit-identical to `single_sample`'s, because both consume atlases written by the
// same `pack_asset` code path. If you change one, change the other.
//
// `bounds_min`/`bounds_max` are the atlas texel rect of THIS tile. A coord outside it is zeroed
// rather than clamped: zeroing yields alpha 0 (unpack_rgba of 0 is all-zero), which the caller's
// alpha test then discards, so a fragment whose uv has drifted off the tile becomes a hole instead
// of smearing the neighbouring view's texels across the silhouette. With a trimmed 2-tile strip
// the side and top tiles are literally adjacent, so dropping this check would bleed the top view
// into the side view along one column of the seam.
//
// `level` is the mip level to fetch, which is why this takes it explicitly rather than deriving
// anything: `Rg32Uint` is not filterable and has no `textureSampleLevel`, so the level has to be
// an argument to `textureLoad`. `coords`/`bounds` are expected to be already scaled INTO that
// level by the caller.
fn cross_single_sample(coords: vec2<f32>, bounds_min: vec2<f32>, bounds_max: vec2<f32>, level: u32) -> UnpackedMaterialProps {
    let in_bounds = select(1u, 0u, any(coords < bounds_min) || any(coords >= bounds_max));

#ifdef INDEXED_PIXELS
    let pixel_dims = textureDimensions(cross_pixels);
    var index: u32;

    if pixel_dims.x * pixel_dims.y < 65536 {
        // The palette fits in 16 bits, so the index image stores TWO indices per texel: even x in
        // the low half, odd x in the high half. Hence the x*0.5 when addressing it and the parity
        // test when extracting. Must match the writer in `pack_asset` exactly.
        // Level 0 throughout the INDEXED path on purpose: `cross_pixels` is a PALETTE there, not a
        // spatial image, so a mip of it would average unrelated palette entries. An indexed atlas
        // simply does not get minification filtering; the host leaves `max_lod` at 0 for one.
        let index_pair = textureLoad(cross_indices, vec2<u32>(coords * vec2(0.5, 1.0)), 0).r;
        index = select(index_pair & 0xFFFF, index_pair >> 16, (u32(coords.x) & 1u) == 1u);
    } else {
        index = textureLoad(cross_indices, vec2<u32>(coords), 0).r;
    }

    let index_x = index % pixel_dims.x;
    let index_y = index / pixel_dims.x;

    let props = textureLoad(cross_pixels, vec2(index_x, index_y), 0).rg * vec2(in_bounds);
#else
    let props = textureLoad(cross_pixels, vec2<u32>(coords), level).rg * vec2(in_bounds);
#endif
    return unpack_props(props);
}

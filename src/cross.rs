//! "Cross" imposters: a fixed set of quads (one horizontal, plus a configurable number of vertical
//! plane ORIENTATIONS spread over 180 degrees, each optionally with a centre plane and an outer
//! pair) sampling a 2-tile atlas (one side view, one top view). See [`CrossLayout`].
//!
//! This is a SEPARATE system living alongside the octahedral `Imposter` in `render.rs`, aimed at
//! low-end/integrated GPUs. The octahedral path pays, per fragment, a camera-ray/sample-plane
//! intersection, an octahedral tile lookup and a 3-tile blend (up to 15 texture loads); the cross
//! path pays one interpolated uv and one texture load, because the uv is baked into the mesh.
//! Nothing here shares state with `render.rs` except `DummyIndicesImage` (see `CrossImposterPlugin`).

use std::f32::consts::PI;

use bevy::{
    asset::{load_internal_asset, uuid_handle, RenderAssetUsages},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

use crate::render::DummyIndicesImage;

pub const CROSS_BINDINGS_HANDLE: Handle<Shader> =
    uuid_handle!("77b27853-f35c-4dff-9a4d-0a628bb19adc");
pub const CROSS_VERTEX_HANDLE: Handle<Shader> =
    uuid_handle!("0a2370cd-a2f5-40f0-aefa-1b4d0976c4b4");
pub const CROSS_FRAGMENT_HANDLE: Handle<Shader> =
    uuid_handle!("c4cfce6e-2552-412c-bf98-2ce7157d853b");

// Deliberately a private flag block, NOT shared with `render.rs` — the two materials specialize
// independently and the bit values are free to diverge. Values are part of the CPU<->shader
// contract: changing one silently changes which `#ifdef` a baked material compiles with.

/// Pixels are stored deduplicated: `indices` holds an index per texel into the unique-pixel
/// `pixels` image. Costs a second texture load, which on a one-load shader is a 2x fetch
/// increase — measure before enabling on an iGPU. -> `INDEXED_PIXELS` (fragment only).
pub const CROSS_INDEXED_FLAG: u32 = 1;
/// Fade each quad out as it turns edge-on to the camera. Without this the horizontal quad seen
/// edge-on draws a hard bright one-pixel line through every instance — the classic cross-quad
/// tell. The vertex stage computes the fade from the quad normal, the fragment stage dithers it
/// away, so the def must go into BOTH stages. -> `ANGLE_FADE` (vertex + fragment).
pub const CROSS_ANGLE_FADE_FLAG: u32 = 2;
/// Flip the baked normal to face the viewer. The quads render with `cull_mode = None`, so half of
/// every instance is seen from its back face, where the baked normal points away from the camera
/// and the whole quad shades black. -> `TWO_SIDED_LIGHTING` (fragment only).
pub const CROSS_TWO_SIDED_FLAG: u32 = 4;
/// Replace `apply_pbr_lighting` with a Lambert term against the first directional light plus
/// ambient. Once the sampling cost is gone this is the dominant remaining fragment cost, and is
/// expected to be the single biggest iGPU lever. -> `CHEAP_LIGHTING` (fragment only).
pub const CROSS_CHEAP_LIGHTING_FLAG: u32 = 8;
// Bits 16 and 32 are RESERVED for WIND and MIP_TAP in Phase 2. Do not reuse them for anything
// else — assets/materials authored now must keep compiling to the same shader defs later.
/// The mesh is a MERGED patch built by [`cross_imposter_merged_mesh`]: N trees' worth of crosses
/// baked into one vertex buffer, already positioned/scaled/yawed, drawn as a single entity.
///
/// Two things break when the per-tree transform is baked into the vertices instead of living in the
/// entity `Transform`, and this flag is what repairs both from `ATTRIBUTE_COLOR` (see
/// [`CrossInstance`] and the `MERGED` block in `cross_vertex.wgsl`):
///
/// * the shader's notion of "this instance's centre" collapses to the patch origin, so `ANGLE_FADE`
///   would fade every tree as if it stood in the middle of the patch — wrong by a growing angle as
///   the camera approaches, and catastrophically wrong once it is inside the patch;
/// * the azimuth-derived quad normal (and the baked normal map's local frame) no longer matches the
///   geometry, because the geometry has been yawed and the azimuth in the vertex buffer has not.
///
/// You MUST also set `center_and_scale = (0, 0, 0, 0.5)` on the material — [`CrossImposterData::
/// set_merged`] does both together, precisely so the two cannot be set independently.
/// -> `MERGED` (vertex only).
pub const CROSS_MERGED_FLAG: u32 = 64;
/// Spend the [`CROSS_ANGLE_FADE_FLAG`] weight by SHRINKING the quad about the instance centre in
/// the vertex stage, instead of dithering it away per fragment. Requires
/// `CROSS_ANGLE_FADE_FLAG` — on its own the weight is always 1.0 and this does nothing.
///
/// The dither is the better mechanism when a temporal resolve is available and the wrong one when
/// it is not: its noise is keyed on `frag_coord`, so a half-faded quad is pinned to the screen
/// while the geometry slides across it and the surviving pixels are re-rolled every frame the
/// camera moves. Scaling the quad drives coverage to zero just as surely, continuously, and in
/// world space. The cost is that the imagery squeezes toward the centre rather than dissolving —
/// only resolvable if a quad can be seen fading while still near face-on, which the fade band is
/// meant to prevent. -> `ANGLE_SHRINK` (vertex + fragment; the fragment uses it only to compile
/// the dither OUT).
pub const CROSS_ANGLE_SHRINK_FLAG: u32 = 128;

pub struct CrossImposterPlugin;

impl Plugin for CrossImposterPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            CROSS_BINDINGS_HANDLE,
            "shaders/cross_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            CROSS_VERTEX_HANDLE,
            "shaders/cross_vertex.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            CROSS_FRAGMENT_HANDLE,
            "shaders/cross_fragment.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(MaterialPlugin::<CrossImposter>::default())
            // `CrossImposter` needs the same 1x1 R32Uint fallback as `Imposter` (bevy hands out an
            // *rgba8* fallback for an unbound `#[texture(sample_type = "u_int")]`, which fails
            // validation), but this plugin must stay usable without `ImposterRenderPlugin`.
            // `.after(crate::render::setup)` is the whole trick: if `ImposterRenderPlugin` is also
            // added — in EITHER order, since ordering constraints are resolved after all plugins
            // build — its `setup` runs first, and an exclusive system forces a command flush before
            // it runs, so the `contains_resource` check below sees the already-inserted resource.
            // If `ImposterRenderPlugin` is absent the constraint refers to an empty system set and
            // is silently vacuous. A plain `Commands`-based system could not do this: two deferred
            // inserts in the same schedule would both pass the guard and the second would clobber
            // the first, orphaning any handle already cloned out of it.
            .add_systems(
                Startup,
                ensure_dummy_indices_image.after(crate::render::setup),
            );
    }
}

/// Exclusive so that `Commands` queued by `render::setup` are already applied when we test for the
/// resource — see the comment at the call site.
fn ensure_dummy_indices_image(world: &mut World) {
    if world.contains_resource::<DummyIndicesImage>() {
        return;
    }

    let image = Image::new(
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        wgpu::TextureDimension::D2,
        vec![0, 0, 0, 0],
        wgpu::TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    let handle = world.resource_mut::<Assets<Image>>().add(image);
    world.insert_resource(DummyIndicesImage(handle));
}

/// Per-instance uniform. FIELD ORDER IS THE ABI — `cross_bindings.wgsl` declares the matching
/// struct member-for-member, and reordering here silently reinterprets the whole block.
#[derive(ShaderType, Clone, Copy, PartialEq, Debug)]
pub struct CrossImposterData {
    /// `xyz` = instance-local centre of the bounding sphere, `w` = its radius. The vertex shader
    /// scales the unit-space (+-0.5) mesh by `w * 2.0`, exactly like `ImposterData`, so the same
    /// `radius` off the bake camera drives both systems.
    pub center_and_scale: Vec4,
    /// Offset of the trimmed sub-rect inside a tile, in texels. `UVec2::ZERO` for an untrimmed
    /// bake target.
    pub packed_tile_offset: UVec2,
    /// Size of the stored (possibly trimmed) tile in texels. Also the atlas stride: the shader's
    /// tile origin is `vec2(tile_id * packed_tile_size.x, 0)`, i.e. the two tiles are a horizontal
    /// strip starting at the top-left of the image.
    pub packed_tile_size: UVec2,
    /// Size the tile was rendered at before trimming; uv in [0,1] is multiplied by this.
    pub base_tile_size: u32,
    /// `CROSS_*_FLAG` bits. Mirrored into `CrossImposterKey`, so every distinct value costs one
    /// pipeline variant.
    pub flags: u32,
    /// Extra multiplier on the sampled alpha.
    pub alpha: f32,
    /// Self-illumination floor as a fraction of albedo, added to emissive before lighting. Lifts
    /// the shaded side out of black under low-ambient host scenes. `0.0` (default) disables.
    pub emissive_floor: f32,
    /// `(start, end)` of the `|dot(view_dir, quad_normal)|` fade band used by `ANGLE_FADE`.
    ///
    /// `(0.20, 0.50)` by default: the horizontal canopy quad is fully hidden below ~11 degrees of
    /// camera elevation and fully shown above ~30, which is the range over which it actually turns
    /// into an edge-on sliver. `end` must still stay well below 0.707 - both verticals sit at that
    /// value at 45 degrees of yaw and are what carry the silhouette there, so fading them together
    /// visibly thins the forest.
    ///
    /// This was measured, not guessed. The original `(0.05, 0.30)` was effectively a NO-OP: a
    /// typical camera ~19 degrees above horizontal puts the canopy quad at `|dot| ~ 0.33`, just
    /// past the old upper edge, so `smoothstep` returned 1.0 and nothing faded. Screenshot diffs
    /// against fade-off, same seed and camera: `(0.05, 0.30)` changed 0.36% of pixels,
    /// `(0.20, 0.50)` changed 1.58%, and a deliberately absurd `(0.90, 1.00)` changed 22.7%.
    /// If you widen this, check it against a low, near-horizontal camera - that is where the
    /// artifact lives.
    pub angle_fade: Vec2,
    /// PERF MEASUREMENT ladder (`0` = normal rendering, the default), mirroring `ImposterData`'s so
    /// the two systems stay directly comparable. Deliberately a uniform and not a flag:
    /// `CrossImposterKey` keys on `flags` alone, so stepping this costs no respecialization and
    /// adds no pipeline variants.
    pub debug_level: u32,
    /// Highest mip level the bound atlas actually has; `0` (the default) means "no chain", and the
    /// shader then addresses level 0 exactly as it always did.
    ///
    /// Set this with [`CrossImposterData::set_mip_levels`] after building a pyramid with
    /// [`crate::cross_mip::build_cross_mip_pyramid`]. Leaving it at 0 with a mipped atlas bound is
    /// harmless (the chain is simply unused); setting it ABOVE the atlas's real level count is not,
    /// and would fetch a level that does not exist.
    ///
    /// A uniform, not a flag, for the same reason as `debug_level`: an atlas gaining mips must not
    /// fork the pipeline. It also completes the 16-byte block that `angle_fade` + `debug_level`
    /// left 4 bytes short, so this field is free.
    pub max_lod: u32,
}

impl CrossImposterData {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        center: Vec3,
        scale: f32,
        base_tile_size: u32,
        packed_tile_offset: UVec2,
        packed_tile_size: UVec2,
        indexed: bool,
        angle_fade: bool,
        two_sided: bool,
        cheap_lighting: bool,
        alpha: f32,
    ) -> Self {
        Self {
            center_and_scale: center.extend(scale),
            packed_tile_offset,
            packed_tile_size,
            base_tile_size,
            flags: if indexed { CROSS_INDEXED_FLAG } else { 0 }
                + if angle_fade { CROSS_ANGLE_FADE_FLAG } else { 0 }
                + if two_sided { CROSS_TWO_SIDED_FLAG } else { 0 }
                + if cheap_lighting {
                    CROSS_CHEAP_LIGHTING_FLAG
                } else {
                    0
                },
            alpha,
            emissive_floor: 0.0,
            // Set even when `CROSS_ANGLE_FADE_FLAG` is off, so flipping the flag on at runtime
            // produces a sane fade instead of a fully-discarded instance (`smoothstep(0,0,x)`).
            angle_fade: Vec2::new(0.20, 0.50),
            // Normal rendering; hosts step this for perf attribution only.
            debug_level: 0,
            // No pyramid until a host builds one and calls `set_mip_levels`.
            max_lod: 0,
        }
    }

    /// Point this material at a mipped atlas built by
    /// [`crate::cross_mip::build_cross_mip_pyramid`].
    ///
    /// `levels` is the atlas's total mip level count (what `cross_mip_levels` returns and what was
    /// written into `texture_descriptor.mip_level_count`), NOT the highest index — this converts.
    /// Passing 0 or 1 leaves minification filtering off, which is the correct result for an atlas
    /// with no chain.
    ///
    /// Never call this for an INDEXED atlas: `cross_pixels` is a palette there rather than a
    /// spatial image, so its mips would average unrelated palette entries. `cross_bindings.wgsl`
    /// pins the indexed path to level 0 regardless, but the intent belongs here too.
    pub fn set_mip_levels(&mut self, levels: u32) {
        if self.flags & CROSS_INDEXED_FLAG != 0 {
            self.max_lod = 0;
            return;
        }
        self.max_lod = levels.saturating_sub(1);
    }

    /// Switch this material over to the MERGED-mesh convention, for meshes built by
    /// [`cross_imposter_merged_mesh`].
    ///
    /// The vertex shader unconditionally evaluates `vertex.position * (scale * 2.0) + centre` from
    /// the material uniform. A merged mesh already carries FINAL patch-local positions — each tree
    /// scaled by its own radius, yawed and translated at build time — so that expression has to
    /// degenerate to the identity: `centre = 0` and `scale = 0.5` make it `position * 1.0 + 0`.
    ///
    /// This is a single call rather than two assignments because forgetting either half is silent
    /// and spectacular: leaving `scale` at the bake radius renders the whole patch blown up by
    /// `2 * radius` (a forest of overlapping skyscrapers), and leaving `centre` at the bake sphere
    /// centre shifts every patch by one tree's offset. It also ORs in [`CROSS_MERGED_FLAG`], since a
    /// merged mesh without the flag mis-fades and mis-lights every yawed tree.
    pub fn set_merged(&mut self) {
        self.center_and_scale = Vec4::new(0.0, 0.0, 0.0, 0.5);
        self.flags |= CROSS_MERGED_FLAG;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CrossImposterKey(u32);

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
#[bind_group_data(CrossImposterKey)]
pub struct CrossImposter {
    #[uniform(0)]
    pub data: CrossImposterData,
    #[texture(1, dimension = "2d", sample_type = "u_int")]
    pub pixels: Handle<Image>,
    // annoyingly we can't use an option here because bevy gives us an rgba8 fallback
    // Res<DummyIndicesImage> gives a default you can drop in
    #[texture(2, dimension = "2d", sample_type = "u_int")]
    pub indices: Handle<Image>,
    pub alpha_mode: AlphaMode,
    pub vram_bytes: usize,
}

impl From<&CrossImposter> for CrossImposterKey {
    fn from(value: &CrossImposter) -> Self {
        Self(value.data.flags)
    }
}

impl Material for CrossImposter {
    fn vertex_shader() -> ShaderRef {
        CROSS_VERTEX_HANDLE.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        CROSS_VERTEX_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        CROSS_FRAGMENT_HANDLE.into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        CROSS_FRAGMENT_HANDLE.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    fn specialize(
        _: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _: &bevy::mesh::MeshVertexBufferLayoutRef,
        key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        // MANDATORY, not a tweak: the cross is three fixed quads with no billboarding, so as the
        // camera orbits each one is seen from its back face for half of every revolution. With the
        // default back-face culling, half the cross simply vanishes and the instance flickers
        // between "tree" and "half a tree" every 180 degrees. Pair with `CROSS_TWO_SIDED_FLAG` so
        // the backfacing halves also shade correctly rather than going black.
        descriptor.primitive.cull_mode = None;

        let vert_defs = &mut descriptor.vertex.shader_defs;
        let frag_defs = &mut descriptor.fragment.as_mut().unwrap().shader_defs;

        if (key.bind_group_data.0 & CROSS_INDEXED_FLAG) != 0 {
            // second texture load: `pixels[indices[coord]]` instead of `pixels[coord]`
            frag_defs.push("INDEXED_PIXELS".into());
        }

        if (key.bind_group_data.0 & CROSS_ANGLE_FADE_FLAG) != 0 {
            // BOTH stages by design: the vertex shader computes the fade weight from the quad
            // normal (three verts' worth of ALU instead of per-fragment), the fragment shader
            // dithers the interpolated weight into a discard. Pushing it to only one stage gives a
            // silent varying mismatch, not a compile error.
            vert_defs.push("ANGLE_FADE".into());
            frag_defs.push("ANGLE_FADE".into());

            // Gated on ANGLE_FADE rather than tested independently: with no fade weight to spend
            // the shrink is a guaranteed no-op, and pushing the def anyway would fork the pipeline
            // into a second variant of an identical program. Reaches the fragment as well, where
            // its only job is to compile the dither out — the two ways of spending the weight are
            // alternatives, and running both would hide a quad at roughly the square of the
            // intended rate.
            if (key.bind_group_data.0 & CROSS_ANGLE_SHRINK_FLAG) != 0 {
                vert_defs.push("ANGLE_SHRINK".into());
                frag_defs.push("ANGLE_SHRINK".into());
            }
        }

        if (key.bind_group_data.0 & CROSS_TWO_SIDED_FLAG) != 0 {
            // flip the baked normal toward the viewer on back faces (see `cull_mode = None` above)
            frag_defs.push("TWO_SIDED_LIGHTING".into());
        }

        if (key.bind_group_data.0 & CROSS_CHEAP_LIGHTING_FLAG) != 0 {
            // single directional Lambert + ambient instead of `apply_pbr_lighting`
            frag_defs.push("CHEAP_LIGHTING".into());
        }

        if (key.bind_group_data.0 & CROSS_MERGED_FLAG) != 0 {
            // VERTEX ONLY. Everything `MERGED` changes - the per-tree centre used by ANGLE_FADE and
            // the per-tree yaw folded into the inverse-rotation varyings - is produced in the vertex
            // stage and reaches the fragment through the EXISTING varyings, so the fragment module
            // is byte-identical merged or not. Pushing it to the fragment too would only add a dead
            // def and a second pipeline variant of an identical program.
            vert_defs.push("MERGED".into());
        }

        Ok(())
    }
}

/// The whole tile: the "no trim" value of [`CrossLayout::side_trim`] / [`CrossLayout::top_trim`],
/// and what [`crate::cross_bake::cross_trim_from_atlas`] falls back to for an empty tile.
///
/// A named constant because the two derivations in [`cross_quads`] are only the f32 IDENTITY on
/// exactly these four values, and "the default is byte-identical to the pre-trim builder" is a
/// property of that identity — a rect written out as `(0.0, 0.0, 1.0, 1.0)` somewhere else is the
/// same numbers, but this way there is one place to look when asking whether it still is.
pub const FULL_TILE: Rect = Rect {
    min: Vec2::ZERO,
    max: Vec2::ONE,
};

/// WHICH quads a cross imposter is built from — the shape of the proxy, shared by
/// [`cross_imposter_mesh`], [`cross_imposter_merged_mesh`] and [`cross_quad_count`].
///
/// A config struct rather than positional parameters because every knob here is a small scalar of
/// the same type (two `f32`s, two `bool`s), so a call site is otherwise a row of unlabelled literals
/// whose meaning depends entirely on their order — and the two builders plus the counter must all be
/// passed the SAME set or they describe different meshes.
///
/// # The quad set
///
/// For each orientation `k` in `0..orientations`, azimuth `theta_k = PI * k / orientations` with
/// plane normal `(sin theta_k, 0, cos theta_k)`; the reference plane is the XY quad yawed by
/// `theta_k` about +Y. Emitted per orientation: the centre plane at offset 0 if `center_planes`, and
/// the pair at `+-side_offset` along the normal if `side_offset > 0.0`. Plus exactly ONE horizontal
/// quad. Total: `orientations * (center_planes + 2 * has_sides) + 1` — see [`cross_quad_count`].
///
/// Every vertical samples tile 0 (side) and carries the reference plane's uvs verbatim; the
/// horizontal samples tile 1 (top).
///
/// # Why an orientation COUNT
///
/// `orientations: 2` is the original perpendicular cross: `theta = 0` (normal +Z) and `theta = PI/2`
/// (normal +X). Its weakness is angular: a viewer at 45 degrees of azimuth is 45 degrees off BOTH
/// planes and the best `|dot|` any of them reaches is 0.707. `orientations: 3` (0, 60, 120 degrees)
/// caps the worst case at 30 degrees off and `|dot| ~ 0.87`, and a triangular lattice's three row
/// directions are at exactly those angles — which is what lets the outer planes sit ON real rows of
/// a hexagonally-placed cluster bake (`--cluster-hex` in `examples/dynamic.rs`).
///
/// [`Default`] is the original 3-quad cross: 2 orientations, centre planes only, no sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossLayout {
    /// Unit-space `y` of the single horizontal (canopy) quad, in `[-0.5, 0.5]` (0 = centre of the
    /// bounding sphere). The seam where it cuts the verticals is least visible inside dense canopy
    /// rather than at the silhouette edge, so trees usually want a small positive value.
    ///
    /// **It is a fraction of the BOUNDING SPHERE, not of the model** — the quad lands
    /// `horizontal_height * 2 * center_and_scale.w` above the sphere centre in world space. Those
    /// are the same thing only while the sphere is a tight fit around one model, and the case where
    /// they come apart is a CLUSTER bake: baking N models into one imposter grows the radius to
    /// cover the group's HORIZONTAL spread (`base_radius * sqrt(N)`-ish) while the models stay
    /// exactly as tall as they were, so a value tuned on a single tree lifts the canopy by the full
    /// inflation factor — ~3.3x at N = 16 — and it visibly detaches from the trunks. Nothing here
    /// can detect that: the builder sees only the unit-space number. If your radius carries
    /// horizontal inflation, scale the value you pass by `base_radius / inflated_radius` (see
    /// `examples/dynamic.rs`, which prints both), so one authored height means the same thing at
    /// every cluster size. `side_offset` needs the OPPOSITE treatment — see below.
    pub horizontal_height: f32,
    /// Number of vertical plane ORIENTATIONS, spread evenly over 180 degrees (a plane and its
    /// 180-degree rotation are the same plane, so the useful range is exactly one half-turn).
    /// `2` = the original square cross (0, 90 degrees); `3` = hex (0, 60, 120 degrees).
    ///
    /// `0` is legal and degenerates to the horizontal quad alone.
    pub orientations: u32,
    /// Unit-space offset of the OUTER plane pair, per orientation: two more planes at
    /// `+-side_offset` along that orientation's normal. `<= 0.0` = no outer planes, and that
    /// disabled path is byte-identical to a build that never had the feature.
    ///
    /// Unlike `horizontal_height` this wants NO cluster correction, and for a reason worth stating:
    /// these are offsets in the HORIZONTAL plane, and a cluster bake inflates the radius for exactly
    /// that reason — the group is horizontally wider. The planes have to span the inflated footprint
    /// to reach the outer rows of the cluster; normalising them back to one model's radius would pin
    /// all of them inside the centre cell and defeat the feature. So `side_offset` is honestly a
    /// fraction of WHATEVER the sphere spans, and is tuned per bake against the actual row positions
    /// rather than carried over from another cluster size.
    ///
    /// **Why this earns its overdraw only with a LATTICE-placed cluster.** With `--cluster N` the
    /// atlas holds N models at once, and the vertical planes are then arbitrary slices through that
    /// group: one plane through a random scatter agrees with the real tree positions nowhere, so
    /// intermediate view angles read as a flat cardboard cutout. Lay the cluster's trees out on a
    /// grid (or a triangular lattice, whose three row directions match `orientations: 3` exactly) and
    /// the planes can be put ON the rows, where they coincide with actual rows of trees — the
    /// silhouette then agrees with what was baked, from every direction the planes face. Rows with no
    /// plane of their own are not a problem: their trunks are already occluded by neighbouring
    /// foliage, so they drown in the noise.
    pub side_offset: f32,
    /// Include the plane at offset 0 for each orientation.
    ///
    /// Experimentally only the OUTER planes carry their weight in a dense cluster: interior rows are
    /// hidden behind foliage from every direction, while the outer rows are what define the
    /// silhouette. Dropping the centres buys back one quad per orientation to spend on more
    /// orientations at the same total quad count — `orientations: 3, center_planes: false` is 7
    /// quads, exactly as many as the square 3-plane-per-axis layout, with 60-degree coverage instead
    /// of 90.
    pub center_planes: bool,
    /// Fraction of the full quad width that the OUTER (`+-side_offset`) planes span, along their own
    /// in-plane horizontal axis. `1.0` = full width (the original behaviour, and the default, so
    /// every configuration authored before this field existed is byte-identical).
    ///
    /// **Why an outer plane is NARROWER than a centre one on a hex lattice.** Every vertical here is
    /// a slice through the baked cluster, and the honest width of that slice is the width of the ROW
    /// of trees it is sitting on. On a square grid every row has the same length, so the full-width
    /// quad is right for all of them and this stays `1.0`. On the triangular lattice
    /// `--cluster-hex` builds it is not: the axial cells satisfy `|q| <= R`, `|r| <= R`,
    /// `|q + r| <= R` and map to `x = pitch * (q + r/2)`, so the CENTRE row (`r = 0`, `q` free over
    /// `[-R, R]`) spans `+-pitch*R` while the OUTERMOST row (`r = R`, which forces `q` into
    /// `[-R, 0]`) spans only `+-pitch*R/2` — exactly half, because the edge of a hexagon is half its
    /// long diagonal. A full-width outer plane therefore draws trees well outside the hexagon's real
    /// footprint, and the horizontal canopy quad — which shows that footprint honestly from above —
    /// has nothing there to match them: trees visibly stick out at the sides with no bird's-eye
    /// counterpart.
    ///
    /// **CROP, do not squeeze.** The quad's half-extent becomes `0.5 * side_width` AND its u range
    /// is narrowed to `[0.5 - side_width/2, 0.5 + side_width/2]` about the centre, by the same
    /// factor. Both halves are mandatory and they only work as a pair: shrinking the quad alone
    /// would compress the whole cluster picture into a narrower plane (trees at the wrong world
    /// scale, thinner than their neighbours on the centre plane), and cropping the uv alone would
    /// stretch it. Together they keep every texel at its true world size and simply clip off the
    /// parts of the image that belong to rows this plane does not represent.
    ///
    /// Applies to the outer pairs ONLY. A centre plane sits on the widest row by construction and
    /// must keep its full width. Composes with [`Self::mirror_sides`] for free: the crop is
    /// symmetric about u = 0.5 and `uv.x -> 1 - uv.x` maps that interval onto itself, so a mirrored
    /// cropped plane shows the same clipped region, reversed.
    ///
    /// `<= 0.0` collapses the outer planes to zero-width slivers; nothing guards against it, because
    /// the only way to reach it is to ask for it.
    pub side_width: f32,
    /// UV-space bounding box of the content actually present in TILE 0 (the side view), as measured
    /// off the baked atlas by [`crate::cross_bake::cross_trim_from_atlas`]. Every VERTICAL quad is
    /// sized to this sub-rect of the bounding sphere instead of to the whole square, and its uvs are
    /// cropped to the same rect. `Rect::new(0.0, 0.0, 1.0, 1.0)` — the default — is "no trim" and is
    /// byte-identical to a build that never had the field.
    ///
    /// **Why this is free performance.** A cross quad is sized to the bake's bounding SPHERE, but
    /// the baked tile is a picture of a tree (tall and narrow) or of a canopy seen from above (a
    /// disc): measured on this repo's two atlases, `alpha >= 0.5` covers 9-17% of a tile and its
    /// bounding box only 14-48%. Everything outside that box rasterizes, fetches a texel and
    /// discards — and the fetch CANNOT be skipped, because colour and alpha share one `Rg32Uint`
    /// texel, so the shader has to load before it can know the fragment is empty. Shrinking the quad
    /// deletes those fragments outright.
    ///
    /// **Why no content is lost.** The bake discards `alpha < 0.5`
    /// (`standard_material_imposter_baker.wgsl`) and the render discards `props.rgba.a * alpha <
    /// 0.01` (`cross_fragment.wgsl`), so as long as the rect was measured with a cutoff at or below
    /// 0.01, every texel it drops was already guaranteed to be discarded by the shader. That part is
    /// exact, not statistical: the trim can only delete fragments that would have discarded.
    ///
    /// **It is nevertheless not a bit-identical frame, and here is the measurement.** 40 000
    /// clustered imposters, fixed seed, `--cross-angle-fade`: the untrimmed run reproduces itself at
    /// 0.00% / max 0, and the trimmed run differs from it by 0.34% of pixels. 89% of those pixels
    /// sit in a SIX-ROW band at the far horizon, where the whole forest saturates and an imposter is
    /// a few pixels across; 4135 of 4149 are content-to-content (a different foliage texel won),
    /// with 6 content-to-background against 8 background-to-content and the lighter/darker split at
    /// 2147/2002 — i.e. symmetric resampling noise, not a silhouette that moved. Shrinking a quad
    /// changes its screen-space rasterization and therefore, in the last bits, which texel a given
    /// pixel centre lands on; at one imposter per few pixels that is occasionally a visibly
    /// different leaf. Nothing above that band moves by more than a stray pixel per row.
    ///
    /// If you are chasing a diff LARGER than that, or one that is not confined to extreme
    /// minification, suspect the measurement instead: an off-by-one on the texel-to-uv boundary
    /// (the rect's upper edge is `x1 + 1`, not `x1`), a flipped v axis, or the pad clamping. Note
    /// that a v flip is nearly INVISIBLE on a typical bake, whose content is roughly centred in the
    /// tile — which is why the axis convention is pinned by a unit test against a deliberately
    /// lopsided rect rather than by looking at a frame.
    ///
    /// **Consequence: the mesh becomes asset-dependent.** Two models with different silhouettes want
    /// different rects, so a host that trims must build (and cache) one mesh per atlas rather than
    /// one mesh globally. Batching is unaffected — every instance of a given asset still shares that
    /// asset's mesh and material.
    ///
    /// **Interaction with [`Self::side_width`]**: `side_width` narrows about `x = 0`, the MODEL
    /// centre, not about the centre of the trimmed quad. That is deliberate: the crop exists to make
    /// an outer plane as wide as the row of a clustered bake it stands on, and those rows are
    /// centred on the model, so the fraction has to be taken about the model's axis or the planes
    /// stop landing on the rows. It also keeps `side_width`'s own "symmetric about u = 0.5"
    /// invariant, which is what lets [`Self::mirror_sides`] map the kept interval onto itself. The
    /// one case where the two disagree is a trim rect that does not straddle `u = 0.5` — content
    /// entirely off to one side — where narrowing about the model centre can walk the outer planes'
    /// uvs back OUTSIDE the measured box, sampling texels the trim declared empty. That is
    /// harmless (they are empty, so they discard) and it cannot happen for a bake centred on its
    /// own bounding sphere, which is every bake this system produces.
    pub side_trim: Rect,
    /// UV-space bounding box of the content actually present in TILE 1 (the top view). Applies to
    /// the single horizontal canopy quad exactly as [`Self::side_trim`] applies to the verticals —
    /// see there for the whole rationale, the pixel-identity argument and the caveats.
    ///
    /// Two rects and not one shared rect: the two views trim in completely different proportions (a
    /// side view of a tree is tall and narrow, a top view of the same tree is a wide disc), and
    /// collapsing them to their union would throw away most of the saving on both.
    pub top_trim: Rect,
    /// Flip `uv.x` on the OUTER planes. Every vertical samples the same side tile, and that tile
    /// shows the WHOLE cluster, so parallel copies of one image are parallel copies of a recognisable
    /// arrangement of trees — obvious repetition at exactly the viewing angles the outer planes exist
    /// to improve. Mirroring them breaks the pattern for free: no extra vertices, no extra tile, no
    /// extra draw, and (as with [`CrossInstance::mirror`]) it does not touch the baked normals or the
    /// winding, which on foliage is indistinguishable from correct. BOTH planes of a pair get the
    /// same flip, so they stay mirror images of the centre rather than of each other.
    pub mirror_sides: bool,
}

impl Default for CrossLayout {
    /// The original 3-quad cross — two perpendicular centre verticals plus the canopy quad — with
    /// the canopy height tuned by eye against a clustered forest.
    fn default() -> Self {
        Self {
            horizontal_height: 0.1125,
            orientations: 2,
            side_offset: 0.0,
            center_planes: true,
            // Full width: the outer planes are as wide as the centre ones, which is what a square
            // grid wants and what every configuration authored before `side_width` existed got.
            side_width: 1.0,
            // UNTRIMMED: quads span the whole bounding sphere and uvs span the whole tile. The two
            // derivations in `cross_quads` are the f32 identity on these values (`0.0 - 0.5` and
            // `0.5 - 1.0` are both exactly `-0.5`), so a default layout is bit-for-bit the mesh it
            // was before per-tile trim existed - which is what makes the feature safe to leave off.
            side_trim: FULL_TILE,
            top_trim: FULL_TILE,
            mirror_sides: false,
        }
    }
}

/// Builds the cross mesh: `4 * cross_quad_count(layout)` vertices and `6 *
/// cross_quad_count(layout)` indices — 12 / 18 for the default 3-quad cross. `TriangleList`.
///
/// Positions are in UNIT space (+-0.5) around the origin; the vertex shader scales them by
/// `center_and_scale.w * 2.0` and offsets by `center_and_scale.xyz`, so one mesh handle serves
/// every instance of every asset and bevy batches them into a single instanced draw.
///
/// See [`CrossLayout`] for what the quads are and [`cross_quads`] for how they are built.
///
/// # azimuth -> local normal
///
/// `ATTRIBUTE_UV_1` is `(tile_id, azimuth)` and the shader derives the quad's local normal from
/// BOTH halves — there is no quad id and no lookup table:
///
/// | quad | `uv_1` | local normal |
/// |---|---|---|
/// | horizontal | `(1, _)` | `(0, 1, 0)` |
/// | vertical, orientation `k` | `(0, theta_k)` | `(sin theta_k, 0, cos theta_k)` |
///
/// The horizontal quad is the ONLY quad that samples tile 1, so the tile id already identifies it
/// and the azimuth field is free to mean "azimuth" everywhere else. Encoding the ANGLE rather than
/// an index is what makes an arbitrary `orientations` expressible at all: the previous scheme was a
/// three-entry `quad_id -> normal` table hardcoded in the shader, which could not describe a fourth
/// direction without an edit on both sides of the CPU/GPU boundary.
///
/// The encoding is keyed on the PLANE, not on the individual quad — an outer plane is parallel to
/// its orientation's reference plane, so it genuinely has that normal, gets that azimuth, and
/// `ANGLE_FADE` fades a whole parallel family together, which is exactly right: they degenerate into
/// slivers at the same instant.
///
/// This is why the mesh carries NO `ATTRIBUTE_NORMAL`. The shader could not read it even if it
/// were there: bevy only binds the normal vertex attribute in the prepass under
/// `NORMAL_PREPASS_OR_DEFERRED_PREPASS`, so in a plain shadow/depth pass `vertex.normal` is not
/// declared at all and reading it is a compile error, not a wrong value. `UV_1` is declared by
/// every pipeline, which is what makes the encoding above the only workable one — see the
/// matching note in `cross_vertex.wgsl`. Nothing else wants the attribute either: the FRAGMENT's
/// shading normal comes from the ATLAS (`props.normal`, unpacked per texel), not from geometry, so
/// a per-vertex normal would be dead data in every pass.
///
/// Carrying it anyway is not free at forest scale: `Float32x3` = 12 bytes/vertex, taking the layout
/// from 40 to 28 bytes here and from 56 to 44 in [`cross_imposter_merged_mesh`] (~21% of merged
/// vertex memory). Merged meshes duplicate the vertex data per tree, so at 12 verts/tree and 30k
/// trees that is 30000 * 12 * 12 = ~4.3 MB of buffer that no shader stage ever loads. Both builders
/// drop it TOGETHER on purpose: differing layouts would make bevy specialize the same shader into
/// two pipeline variants (one with `VERTEX_NORMALS`, one without).
pub fn cross_imposter_mesh(layout: &CrossLayout) -> Mesh {
    let quads = cross_quads(layout);
    let vertex_count = quads.len() * 4;

    let mut positions: Vec<Vec3> = Vec::with_capacity(vertex_count);
    let mut uvs: Vec<Vec2> = Vec::with_capacity(vertex_count);
    let mut uvs_b: Vec<Vec2> = Vec::with_capacity(vertex_count);

    for quad in &quads {
        for corner in 0..4 {
            positions.push(quad.corners[corner]);
            uvs.push(quad.uvs[corner]);
            uvs_b.push(quad.uv_b);
        }
    }

    let indices: Vec<u32> = (0..quads.len() as u32)
        .flat_map(|q| QUAD_INDICES.map(|i| q * 4 + i))
        .collect();

    // 28 bytes/vertex: POSITION 12 + UV_0 8 + UV_1 8. No NORMAL and no COLOR — see the doc comment
    // for NORMAL; COLOR is `(centre.xyz, yaw)` and is read ONLY under `MERGED`, which by definition
    // never applies to this builder (one entity = one tree, so the `Transform` still carries both).
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    // `.x` = tile id (0 = side, 1 = top), `.y` = the plane's AZIMUTH in radians (unused on the
    // horizontal quad, which the tile id already identifies). Declaring UV_1 makes bevy set the
    // `VERTEX_UVS_B` shader def automatically. Both are constant per quad.
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, uvs_b)
    .with_inserted_indices(Indices::U32(indices))
}

/// One tree in a merged patch built by [`cross_imposter_merged_mesh`].
///
/// Everything here is baked into the vertex buffer at build time, which is exactly what replaces
/// the per-entity `Transform` a non-merged cross imposter would have carried.
#[derive(Clone, Copy, Debug)]
pub struct CrossInstance {
    /// Position of this tree relative to the patch entity's `Transform` (which is normally the
    /// chunk centroid, to keep the patch's bounding box tight and its float precision good).
    pub translation: Vec3,
    /// This tree's bounding-sphere RADIUS — the same quantity as `center_and_scale.w` and the same
    /// `radius` the bake camera used, NOT a diameter and NOT a unit-space multiplier. The builder
    /// applies the `* 2.0` itself, exactly as the vertex shader does for the non-merged path.
    pub scale: f32,
    /// Rotation about +Y, radians. Matches `Quat::from_rotation_y`, which is what the shader's
    /// `MERGED` block reconstructs from `ATTRIBUTE_COLOR.w`.
    pub yaw: f32,
    /// Mirror the tree by flipping `uv.x`. Free per-tree variation: it costs no extra vertices, no
    /// extra texture and no extra draw, and it does NOT touch the normals — a mirrored side view is
    /// still lit by the same baked normal map, which on foliage is indistinguishable from correct.
    /// Note that mirroring the uv does not flip the winding, so `cull_mode = None` is unaffected.
    pub mirror: bool,
}

/// Merges the cross of `instances.len()` trees into ONE mesh — `4 * cross_quad_count(layout)`
/// vertices and `6 * cross_quad_count(layout)` indices per instance — every tree already at its own
/// position, scale and yaw.
///
/// The whole [`CrossLayout`] is PER-PATCH, not per-tree: the point is that one quad set serves every
/// instance in the buffer, which is what makes the merge a merge. Two consequences worth stating:
///
/// * [`CrossLayout::horizontal_height`] carries the same trap as in [`cross_imposter_mesh`], and
///   here it is applied per tree against [`CrossInstance::scale`]: the quad lands
///   `horizontal_height * 2 * scale` above each tree's own centre. `scale` is a bounding-sphere
///   radius, so if that sphere was inflated to cover a CLUSTER's horizontal spread the canopy floats
///   by the inflation factor while the trees stay the same height — pre-divide by
///   `base_radius / inflated_radius` before calling. Mixing trees of genuinely different SIZES in
///   one patch is fine (the height is a fraction of each tree's own radius); mixing DIFFERENTLY
///   INFLATED radii is not, because a single pre-divided height cannot correct both.
/// * [`CrossLayout::mirror_sides`] composes with [`CrossInstance::mirror`] by XOR (both flip
///   `uv.x`), which is the useful behaviour — a mirrored tree still has its outer planes mirrored
///   RELATIVE to its centre planes, so the repetition-breaking survives.
///
/// # Why this exists (and why the octahedral cluster trick does not work here)
///
/// The octahedral path can bake a CLUSTER of N models into a single imposter, because its billboard
/// re-derives a view direction per fragment and the cluster's inter-model parallax is captured in
/// the atlas. A cross imposter has two pictures on three fixed planes, so a baked cluster collapses
/// into a flat wall of trees with no inter-tree parallax at all — and the cluster's radius grows as
/// `base_radius * sqrt(N)` while `tile_size` stays put, so per-tree texel density falls off a
/// cliff. Merging MESHES instead keeps every tree at full atlas resolution with its own correct
/// cross, and still collapses the patch to one draw.
///
/// # Contract with the material — read this before using it
///
/// The vertex shader always computes `vertex.position * (scale * 2.0) + centre` from the material
/// uniform, and the positions here are already final. The material must therefore be set to
/// `center_and_scale = (0, 0, 0, 0.5)` and carry [`CROSS_MERGED_FLAG`]; call
/// [`CrossImposterData::set_merged`] rather than assembling that by hand.
///
/// The mesh carries `ATTRIBUTE_COLOR = (translation.xyz, yaw)`, which is how the `MERGED` shader
/// path recovers the per-tree centre (for `ANGLE_FADE`'s view direction) and the per-tree yaw (to
/// rotate both the azimuth-derived quad normal and the baked normal map into the tree's frame).
/// `COLOR` is used because it is the only wide standard attribute bevy binds in BOTH the forward
/// and the prepass pipelines unconditionally — `ATTRIBUTE_NORMAL` is only bound in the prepass
/// under `NORMAL_PREPASS_OR_DEFERRED_PREPASS`, so a plain shadow pass would not even declare it.
/// That is also why the mesh carries no `ATTRIBUTE_NORMAL` at all: unreadable in the prepass,
/// redundant in the forward pass (the shading normal comes from the atlas), and at 12 bytes/vertex
/// the single largest piece of dead weight a merged patch could carry — see [`cross_imposter_mesh`]
/// for the arithmetic. Layout here is 44 bytes/vertex: POSITION 12 + UV_0 8 + UV_1 8 + COLOR 16.
///
/// # Batching
///
/// One patch = one entity = one draw, and every patch shares the material. Chunk sizes in the low
/// hundreds are the sweet spot: too small and you are back to per-entity draws, too large and the
/// patch's bounding box gets so big that frustum culling stops rejecting anything.
pub fn cross_imposter_merged_mesh(layout: &CrossLayout, instances: &[CrossInstance]) -> Mesh {
    let quads = cross_quads(layout);
    let count = instances.len();
    // Per-tree stride into the shared buffers. Derived from `quads.len()` rather than hardcoded to
    // 12/18, because a stale literal here would not fail to compile - it would silently point every
    // tree's indices at the previous tree's vertices.
    let verts_per_instance = quads.len() * 4;
    let indices_per_instance = quads.len() * QUAD_INDICES.len();

    let mut positions: Vec<Vec3> = Vec::with_capacity(count * verts_per_instance);
    let mut uvs: Vec<Vec2> = Vec::with_capacity(count * verts_per_instance);
    let mut uvs_b: Vec<Vec2> = Vec::with_capacity(count * verts_per_instance);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(count * verts_per_instance);
    let mut indices: Vec<u32> = Vec::with_capacity(count * indices_per_instance);

    for (instance_index, instance) in instances.iter().enumerate() {
        // Same order the non-merged path applies it in, and it is not commutative: scale (about the
        // unit-space origin) -> yaw about +Y -> translate. Swapping the last two would orbit each
        // tree around the patch origin instead of spinning it in place.
        let rotation = Quat::from_rotation_y(instance.yaw);
        let size = instance.scale * 2.0;
        let base = (instance_index * verts_per_instance) as u32;
        // Per-tree centre + yaw, replicated to every vertex. Constant over the whole instance,
        // so the interpolator hands the shader back exactly what was written.
        let color = [
            instance.translation.x,
            instance.translation.y,
            instance.translation.z,
            instance.yaw,
        ];

        for quad in &quads {
            for corner in 0..4 {
                positions.push(rotation * (quad.corners[corner] * size) + instance.translation);
                let uv = quad.uvs[corner];
                uvs.push(if instance.mirror {
                    Vec2::new(1.0 - uv.x, uv.y)
                } else {
                    uv
                });
                uvs_b.push(quad.uv_b);
                colors.push(color);
            }
        }

        indices
            .extend((0..quads.len() as u32).flat_map(|q| QUAD_INDICES.map(|i| base + q * 4 + i)));
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, uvs_b)
    // (centre.xyz, yaw). Declaring COLOR makes bevy set `VERTEX_COLORS` in both the forward and
    // the prepass pipelines; the shader guards its read with that def, so a merged mesh built
    // without this attribute degrades to the unmerged fade behaviour instead of failing to
    // compile.
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

/// Two triangles over the 4 corners of one quad, `TriangleList`.
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// One quad of a cross, in UNIT space (+-0.5) around the origin.
///
/// Deliberately carries no `normal`: neither builder emits `Mesh::ATTRIBUTE_NORMAL`, and the one
/// consumer of a quad normal (`ANGLE_FADE`) rebuilds it in the vertex shader from `uv_b`, because
/// the prepass does not declare `vertex.normal`. A field here would be a copy of a value the shader
/// derives anyway, that nothing reads and nothing keeps honest.
struct CrossQuad {
    corners: [Vec3; 4],
    uvs: [Vec2; 4],
    /// `ATTRIBUTE_UV_1` = (tile id, azimuth in radians). See the encoding table on
    /// [`cross_imposter_mesh`].
    uv_b: Vec2,
}

/// THE single source of truth for the cross's corner positions, uvs and `(tile, azimuth)` pairs,
/// shared by [`cross_imposter_mesh`] and [`cross_imposter_merged_mesh`].
///
/// Deliberately not duplicated into the two builders: the uv convention below is dictated by the
/// bake camera and a divergence between two copies of it would not fail to compile, it would render
/// a mirrored or transposed tile in one code path only — the single most likely bug in this system
/// and the hardest to spot in a forest.
///
/// # Emission order — LOAD-BEARING
///
/// All the centre planes first (ascending orientation), then the horizontal, then the outer planes
/// (inner loop over orientations, outer loop over `-side_offset` then `+side_offset`). That is
/// exactly the order the original hardcoded V0 / V1 / H / V0- / V1- / V0+ / V1+ builder used, so a
/// [`CrossLayout::default`]-shaped layout — and equally a `side_offset > 0.0` one at
/// `orientations: 2` — emits the identical vertex order, index list and values it always has.
/// Neither builder knows the order: both derive their strides from `quads.len()`.
///
/// # Reference plane and azimuth
///
/// Orientation `k` has azimuth `theta = PI * k / orientations` and its reference plane is V0 (the XY
/// quad) yawed by `theta` about +Y, which carries V0's +Z normal to `(sin theta, 0, cos theta)`.
/// At `orientations: 2` that is `theta = 0 -> (0,0,1)` and `theta = PI/2 -> (1,0,0)`: the ORIGINAL
/// V0/V1 pair, bit for bit — see `snap_axis` for the one f32 detail that makes "bit for bit" true
/// rather than "to within 4e-8".
///
/// Every vertical carries the reference plane's uvs VERBATIM rather than recomputing them from its
/// own corners. That is the relationship V1 always had to V0 and the reason is unchanged: all
/// verticals sample the one side tile, so a yawed plane shows a rotated-viewpoint copy of the same
/// picture (intentional, and invisible on foliage), while recomputing uvs from the geometry would
/// squash the tile to a degenerate line for any plane that spans no x. An OUTER plane is a
/// translation of its reference plane along the normal, so it spans no new axis either and copies
/// the same uvs — optionally mirrored (see [`CrossLayout::mirror_sides`]) and optionally CROPPED to
/// the middle `side_width` of them (see [`CrossLayout::side_width`], and note that the crop is the
/// one case where an outer plane's uvs are not literally the centre plane's).
///
/// # Reference quads and the trim
///
/// V0 and H are not the fixed `+-0.5` squares they read as: they span
/// [`CrossLayout::side_trim`] / [`CrossLayout::top_trim`] mapped back through the uv conventions
/// above, so a quad is only as big as the part of its tile that has anything in it. Every
/// downstream consumer — the yaws, the `side_width` crop, `mirror_sides`, the outer-plane offsets,
/// both builders — is written against those two arrays and inherits the trim for free.
fn cross_quads(layout: &CrossLayout) -> Vec<CrossQuad> {
    let h = layout.horizontal_height;

    // The uv convention is FORCED by the bake camera and must match it exactly; a mirrored or
    // transposed tile is the single most likely early bug here. The bake places the camera at
    // `center + normal * radius` looking at `center` with `up`, and the sample basis is
    // `r = cross(n, -up)`, `u = cross(r, n)`, `uv = (dot(v, r), dot(v, u)) + 0.5`.
    //
    // side view (n = +Z, up = +Y):  r = (1, 0, 0),  u = (0, -1, 0)  => uv = (0.5 + x, 0.5 - y)
    // top view  (n = +Y, up = +Z):  r = (-1, 0, 0), u = (0, 0, -1)  => uv = (0.5 - x, 0.5 - z)
    //
    // (the `up` values come from `normal_from_grid`: `Vec3::Z` when `|n.y| > 0.99`, else `Vec3::Y`)
    let v0_uv = |p: Vec3| Vec2::new(0.5 + p.x, 0.5 - p.y);
    let h_uv = |p: Vec3| Vec2::new(0.5 - p.x, 0.5 - p.z);

    // PER-TILE TRIM. The reference quads below span the trim rect rather than the whole +-0.5
    // square, and their uvs then fall out of running the SAME two closures over the narrowed
    // corners — the identical trick `side_width` uses further down, and for the identical reason:
    // both closures are affine, so a corner range and its uv range crop together by construction
    // and there is no second cropping expression to drift out of sync with the convention above.
    //
    // Inverting the two conventions to turn a uv rect back into corner coordinates:
    //
    //   side:  u = 0.5 + x  =>  x = u - 0.5        v = 0.5 - y  =>  y = 0.5 - v
    //   top:   u = 0.5 - x  =>  x = 0.5 - u        v = 0.5 - z  =>  z = 0.5 - v
    //
    // Both v axes are FLIPPED, so the rect's `min.y` (the TOP row of the image) becomes the MAXIMUM
    // y/z corner and vice versa. Getting that pair the wrong way round is the one mistake here that
    // still renders something plausible — a roughly symmetric tree looks fine upside down in its
    // own bounding box — so the corner ranges are written `(0.5 - max, 0.5 - min)` with the swap
    // visible rather than hidden inside a helper.
    //
    // At `FULL_TILE` every one of these is the f32 identity: `0.0 - 0.5` and `0.5 - 1.0` are both
    // exactly `-0.5`, `1.0 - 0.5` and `0.5 - 0.0` are both exactly `0.5`, and the uvs come back as
    // exactly 0.0/1.0. That is what `shipped_layouts_are_byte_identical` pins.
    let (side_x0, side_x1) = (layout.side_trim.min.x - 0.5, layout.side_trim.max.x - 0.5);
    let (side_y0, side_y1) = (0.5 - layout.side_trim.max.y, 0.5 - layout.side_trim.min.y);
    let (top_x0, top_x1) = (0.5 - layout.top_trim.max.x, 0.5 - layout.top_trim.min.x);
    let (top_z0, top_z1) = (0.5 - layout.top_trim.max.y, 0.5 - layout.top_trim.min.y);

    // V0: XY plane, normal +Z. CCW when viewed from +Z. The reference plane of orientation 0, and
    // the shape every other vertical is a yaw (plus a translation) of.
    //
    // WINDING is preserved by the trim: the corner ORDER is written in terms of the ranges as
    // (lo, lo) -> (hi, lo) -> (hi, hi) -> (lo, hi), and a `Rect` always has `min <= max`, so
    // `x0 <= x1` and `y0 <= y1` hold for ANY rect and the cross product of the first two edges keeps
    // pointing along +Z. It is checked by a test rather than left to show up in a frame precisely
    // because it CANNOT show up in a frame: `specialize` sets `cull_mode = None`, and bevy runs a
    // material's `specialize` for the prepass pipelines too, so an inverted quad renders and
    // depth-writes exactly like a correct one until the day somebody turns culling back on.
    let v0: [Vec3; 4] = [
        Vec3::new(side_x0, side_y0, 0.0),
        Vec3::new(side_x1, side_y0, 0.0),
        Vec3::new(side_x1, side_y1, 0.0),
        Vec3::new(side_x0, side_y1, 0.0),
    ];

    let side_uvs = v0.map(v0_uv);

    // The outer planes' reference shape: V0 narrowed about x = 0 by `side_width` along the in-plane
    // horizontal axis (V0's x, which every yaw keeps horizontal). See `CrossLayout::side_width` for
    // why an outer plane on a hex lattice is genuinely half the width of a centre one.
    //
    // About x = 0, i.e. the MODEL centre — NOT about the centre of the (possibly trimmed) V0. The
    // two differ once `side_trim` is not centred on the tile, and the model centre is the right one:
    // `side_width` exists to make an outer plane as wide as the row of a clustered bake it stands
    // on, and those rows are centred on the model's axis, so a fraction taken about anything else
    // stops landing on them. It also keeps the kept u interval symmetric about u = 0.5, which is the
    // property `mirror_sides` below relies on. See `CrossLayout::side_trim` for the one degenerate
    // case (a trim rect that does not straddle the centre) and why it cannot arise here.
    //
    // The uvs come from running the SAME `v0_uv` over the narrowed corners, which is the whole trick
    // and is why there is no separate crop expression to keep in sync: `v0_uv` is affine in x, so
    // scaling the corners' x by `w` scales the u interval about u = 0.5 by exactly the same factor
    // (untrimmed, `+-0.5` -> `[0.5 - w/2, 0.5 + w/2]`). Texel-per-world-unit is therefore unchanged and the
    // plane is a CROP of the side tile, not a squeezed copy of it. At `side_width == 1.0` the
    // multiply is the f32 identity and both arrays are bit-for-bit what they were before the field
    // existed — which is what keeps the default layouts on a 0.00% screenshot diff.
    let v0_outer: [Vec3; 4] = v0.map(|p| Vec3::new(p.x * layout.side_width, p.y, p.z));
    let outer_uvs = v0_outer.map(v0_uv);

    // Same picture, flipped horizontally — see `CrossLayout::mirror_sides`. Both planes of a pair
    // share this one uv set, so each is a mirror of the CENTRE plane and they stay identical to each
    // other; that is the arrangement that reads as "two different rows of trees" rather than "one
    // row rendered three times". Applied AFTER the crop, and order does not matter: the cropped
    // interval is symmetric about u = 0.5, so `1 - u` maps it onto itself and a mirrored cropped
    // plane shows the same clipped region, reversed. (A `side_trim` that is not itself centred on
    // the tile breaks that symmetry, so the mirrored planes then show a slightly different slice —
    // still valid texels, still the same tile; see `CrossLayout::side_trim`.)
    let outer_uvs = if layout.mirror_sides {
        outer_uvs.map(|uv| Vec2::new(1.0 - uv.x, uv.y))
    } else {
        outer_uvs
    };

    // H — XZ plane at y = horizontal_height, normal +Y, tile 1 (top). The only tile-1 quad, which is
    // exactly how the shader tells it apart from the verticals without a quad id.
    //
    // Corner order chosen so the winding is CCW when viewed from +Y, and — as for V0 above — that
    // survives `top_trim` because the order is expressed as (x lo, z hi) -> (x hi, z hi) ->
    // (x hi, z lo) -> (x lo, z lo). The z pairing looks inverted next to V0's and is not: the top
    // view's u axis is ALSO flipped (`u = 0.5 - x`), so both of this quad's axes run backwards
    // relative to the side view's, and swapping either one alone would flip the normal to -Y.
    let hq: [Vec3; 4] = [
        Vec3::new(top_x0, h, top_z1),
        Vec3::new(top_x1, h, top_z1),
        Vec3::new(top_x1, h, top_z0),
        Vec3::new(top_x0, h, top_z0),
    ];

    // (azimuth, sin, cos) per orientation, evenly over 180 degrees. Computed once: the outer-plane
    // loop below needs the same numbers, and re-deriving them there would be a second place for the
    // `snap_axis` correction to be forgotten.
    let orientations: Vec<(f32, f32, f32)> = (0..layout.orientations)
        .map(|k| {
            let theta = PI * k as f32 / layout.orientations as f32;
            (theta, snap_axis(theta.sin()), snap_axis(theta.cos()))
        })
        .collect();

    // Yaw about +Y by (sin, cos), written out rather than going through `Quat::from_rotation_y`:
    // the quaternion path multiplies half-angle sines together and would NOT reproduce the original
    // hardcoded `(x, y, z) -> (z, y, -x)` bit for bit at theta = PI/2, which is the whole point of
    // `snap_axis`.
    let yaw = |p: Vec3, s: f32, c: f32| Vec3::new(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);

    let mut quads = Vec::with_capacity(cross_quad_count(layout));

    if layout.center_planes {
        for &(theta, s, c) in &orientations {
            quads.push(CrossQuad {
                corners: v0.map(|p| yaw(p, s, c)),
                uvs: side_uvs,
                uv_b: Vec2::new(0.0, theta),
            });
        }
    }

    quads.push(CrossQuad {
        corners: hq,
        uvs: hq.map(h_uv),
        // Azimuth is meaningless for a horizontal plane and the shader never reads it here (tile 1
        // short-circuits to the +Y normal), so it is 0 rather than a made-up angle.
        uv_b: Vec2::new(1.0, 0.0),
    });

    if layout.side_offset > 0.0 {
        for offset in [-layout.side_offset, layout.side_offset] {
            for &(theta, s, c) in &orientations {
                // A translation of the (possibly `side_width`-narrowed) reference plane ALONG ITS
                // OWN NORMAL: narrowing happens within the plane, so this still keeps that
                // normal and therefore that azimuth — which is what makes `ANGLE_FADE` fade a
                // parallel family together (they go edge-on at the same instant) and what let the
                // shader stay agnostic about how many planes share an orientation.
                let n = Vec3::new(s, 0.0, c);
                quads.push(CrossQuad {
                    corners: v0_outer.map(|p| yaw(p, s, c) + n * offset),
                    uvs: outer_uvs,
                    uv_b: Vec2::new(0.0, theta),
                });
            }
        }
    }

    quads
}

/// Snaps a `sin`/`cos` result that should be exactly 0 to exactly 0.
///
/// `f32` trig is not exact at the axes: `cos(FRAC_PI_2)` is `-4.371139e-8`, not `0.0`. Left alone,
/// `orientations: 2` would yaw the +X reference plane by an extra 2.5e-6 degrees, moving every
/// vertex of it by ~5e-7 world units and perturbing the `ANGLE_FADE` dot product in the last few
/// bits — invisible, but it would also mean the default layout no longer reproduces the original
/// hardcoded quad set BIT FOR BIT, and "byte-identical to before" is what makes the whole
/// generalisation safe to land behind a 0.00% screenshot diff.
///
/// The threshold only has to separate "trig noise around zero" from any real value the table can
/// produce. The smallest non-zero `|sin|`/`|cos|` here is `sin(PI/orientations)`, so 1e-6 is safe up
/// to ~3 million orientations.
fn snap_axis(v: f32) -> f32 {
    if v.abs() < 1e-6 {
        0.0
    } else {
        v
    }
}

/// Number of quads [`cross_imposter_mesh`] / [`cross_imposter_merged_mesh`] will emit per instance
/// for a given layout: `orientations * (centre + outer pair) + 1` for the horizontal.
///
/// 3 for [`CrossLayout::default`]; 7 for both the square 3-planes-per-axis layout
/// (`orientations: 2, side_offset > 0`) and the hex outer-only one (`orientations: 3,
/// side_offset > 0, center_planes: false`).
///
/// Exists so hosts can report vertex/triangle budgets (a merged patch is `4 * n` verts and `2 * n`
/// triangles per tree) without re-deriving the "> 0.0 means enabled" rule and drifting from it.
pub fn cross_quad_count(layout: &CrossLayout) -> usize {
    let per_orientation =
        usize::from(layout.center_planes) + if layout.side_offset > 0.0 { 2 } else { 0 };
    layout.orientations as usize * per_orientation + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-exact, not approximate: the point of these is that generalising the builder to N
    /// orientations did not move a single float of the layouts that already shipped. `f32 ==` would
    /// pass on `-0.0 == 0.0`, which is a different vertex buffer, so compare the bit patterns.
    fn assert_same(actual: &[CrossQuad], expected: &[([Vec3; 4], [Vec2; 4], Vec2)]) {
        assert_eq!(actual.len(), expected.len(), "quad count");
        for (i, (quad, (corners, uvs, uv_b))) in actual.iter().zip(expected).enumerate() {
            for c in 0..4 {
                assert_eq!(
                    quad.corners[c].to_array().map(f32::to_bits),
                    corners[c].to_array().map(f32::to_bits),
                    "quad {i} corner {c}"
                );
                assert_eq!(quad.uvs[c], uvs[c], "quad {i} uv {c}");
            }
            assert_eq!(quad.uv_b, *uv_b, "quad {i} uv_b");
        }
    }

    /// The five layouts the regression fingerprints below cover, named so a failure says WHICH
    /// configuration moved rather than just "index 2".
    fn reference_layouts() -> [(&'static str, CrossLayout); 5] {
        [
            ("default", CrossLayout::default()),
            (
                "square sides",
                CrossLayout {
                    side_offset: 0.25,
                    ..default()
                },
            ),
            (
                "cropped sides",
                CrossLayout {
                    side_offset: 0.25,
                    side_width: 0.6,
                    ..default()
                },
            ),
            (
                "mirrored cropped sides",
                CrossLayout {
                    side_offset: 0.25,
                    side_width: 0.6,
                    mirror_sides: true,
                    ..default()
                },
            ),
            (
                "hex outer-only",
                CrossLayout {
                    orientations: 3,
                    side_offset: 0.25,
                    center_planes: false,
                    ..default()
                },
            ),
        ]
    }

    /// FNV-1a over the mesh's PACKED vertex buffer and index buffer — i.e. over the exact bytes
    /// that reach the GPU, attribute order and padding included, not over a re-derived summary of
    /// them.
    ///
    /// A hash rather than a transcribed array because the interesting layouts are 28 verts x 7
    /// floats each: written out longhand they would be ~800 literals per layout, which nobody
    /// reviews and everybody regenerates. A single u64 is reviewable, and it fails on a one-ulp
    /// move in any float of any attribute — which is the property these tests exist to defend.
    fn mesh_fingerprint(mesh: &Mesh) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |bytes: &[u8]| {
            for b in bytes {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        eat(&mesh.create_packed_vertex_buffer_data());
        eat(mesh.get_index_buffer_bytes().unwrap());
        hash
    }

    /// THE regression bar for every future change to `cross_quads`: the exact GPU bytes of five
    /// shipped layouts, recorded from the builder as it stood before per-tile trim existed.
    ///
    /// These constants are a RECORD, not an expectation to be refreshed. If a change moves one of
    /// them it has changed the vertex buffer of a configuration somebody may already be shipping,
    /// and the question to answer is "why", not "what is the new number". The per-tile trim work
    /// that added [`CrossLayout::side_trim`] passes this untouched, which is what makes the default
    /// (full-rect) trim provably a no-op rather than merely a small one.
    #[test]
    fn shipped_layouts_are_byte_identical() {
        const EXPECTED: [u64; 5] = [
            0xcbc3_2cf3_b1d4_4aa7,
            0x8558_10ce_5afb_6207,
            0xa5f5_e0c7_526b_1c47,
            0xfc25_21f1_0c01_a447,
            0xf5f8_9eb6_66de_2767,
        ];
        for ((name, layout), expected) in reference_layouts().into_iter().zip(EXPECTED) {
            assert_eq!(
                mesh_fingerprint(&cross_imposter_mesh(&layout)),
                expected,
                "{name}: unmerged mesh bytes changed"
            );
        }
    }

    /// Same bar for the merged builder. Separate constants because a merged patch is a different
    /// vertex layout (POSITION + UV_0 + UV_1 + COLOR) carrying per-tree transforms, so a change
    /// that moved only the merged path would slip past the test above.
    ///
    /// Two instances, one of them yawed by a non-axis angle, because 90-degree yaws are visually
    /// identity here and would hide a rotation-order mistake.
    #[test]
    fn shipped_merged_layouts_are_byte_identical() {
        const EXPECTED: [u64; 5] = [
            0x60db_39c1_2b6d_90b6,
            0x6d16_53a0_f44e_369e,
            0xda27_b622_fb3c_f37e,
            0x3a61_714d_a9cd_f3fa,
            0xd3ff_3d1d_5c5c_baca,
        ];
        let instances = [
            CrossInstance {
                translation: Vec3::new(1.0, 2.0, 3.0),
                scale: 1.5,
                yaw: 0.7,
                mirror: false,
            },
            CrossInstance {
                translation: Vec3::new(-4.0, 0.0, 2.5),
                scale: 0.75,
                yaw: -2.1,
                mirror: true,
            },
        ];
        for ((name, layout), expected) in reference_layouts().into_iter().zip(EXPECTED) {
            assert_eq!(
                mesh_fingerprint(&cross_imposter_merged_mesh(&layout, &instances)),
                expected,
                "{name}: merged mesh bytes changed"
            );
        }
    }

    /// The pre-generalisation builder's V0/V1/H, transcribed by hand from the hardcoded version.
    /// Only `uv_b.y` differs by design: it used to be a quad ID (0, 1, 2) and is now the plane's
    /// azimuth in radians, which the shader turns back into the same three normals.
    #[test]
    fn default_layout_is_the_original_cross() {
        let h = 0.1125;
        let side_uvs = [
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ];
        let quads = cross_quads(&CrossLayout::default());
        assert_eq!(cross_quad_count(&CrossLayout::default()), 3);
        assert_same(
            &quads,
            &[
                // V0 - XY plane, +Z normal, azimuth 0
                (
                    [
                        Vec3::new(-0.5, -0.5, 0.0),
                        Vec3::new(0.5, -0.5, 0.0),
                        Vec3::new(0.5, 0.5, 0.0),
                        Vec3::new(-0.5, 0.5, 0.0),
                    ],
                    side_uvs,
                    Vec2::new(0.0, 0.0),
                ),
                // V1 - V0 yawed +90 deg: (x, y, z) -> (z, y, -x). +X normal, azimuth PI/2
                (
                    [
                        Vec3::new(0.0, -0.5, 0.5),
                        Vec3::new(0.0, -0.5, -0.5),
                        Vec3::new(0.0, 0.5, -0.5),
                        Vec3::new(0.0, 0.5, 0.5),
                    ],
                    side_uvs,
                    Vec2::new(0.0, PI / 2.0),
                ),
                // H - XZ plane at y = h, +Y normal, tile 1
                (
                    [
                        Vec3::new(-0.5, h, 0.5),
                        Vec3::new(0.5, h, 0.5),
                        Vec3::new(0.5, h, -0.5),
                        Vec3::new(-0.5, h, -0.5),
                    ],
                    [
                        Vec2::new(1.0, 0.0),
                        Vec2::new(0.0, 0.0),
                        Vec2::new(0.0, 1.0),
                        Vec2::new(1.0, 1.0),
                    ],
                    Vec2::new(1.0, 0.0),
                ),
            ],
        );
    }

    /// The square 7-quad layout, in the order the hardcoded builder appended them: centres, H, then
    /// the -offset pair and the +offset pair, each inner-looping over the orientations.
    #[test]
    fn square_side_planes_keep_their_order_and_values() {
        let layout = CrossLayout {
            horizontal_height: 0.0,
            side_offset: 0.25,
            ..default()
        };
        let quads = cross_quads(&layout);
        assert_eq!(quads.len(), cross_quad_count(&layout));
        assert_eq!(quads.len(), 7);
        // Only the corners are interesting here (uvs are copies of V0's and are covered above).
        let corners: Vec<[Vec3; 4]> = quads.iter().map(|q| q.corners).collect();
        for (i, &z) in [-0.25f32, 0.25].iter().enumerate() {
            // V0-parallel: XY plane displaced along +Z.
            assert_eq!(corners[3 + i * 2][0], Vec3::new(-0.5, -0.5, z));
            assert_eq!(corners[3 + i * 2][2], Vec3::new(0.5, 0.5, z));
            // V1-parallel: YZ plane displaced along +X.
            assert_eq!(corners[4 + i * 2][0], Vec3::new(z, -0.5, 0.5));
            assert_eq!(corners[4 + i * 2][2], Vec3::new(z, 0.5, -0.5));
        }
    }

    /// The layout this generalisation exists for: 3 orientations at 60 degrees, outer planes only.
    /// Same 7 quads as the square layout above, spent on angular coverage instead of on depth.
    #[test]
    fn hex_outer_only_layout() {
        let layout = CrossLayout {
            orientations: 3,
            side_offset: 0.25,
            center_planes: false,
            ..default()
        };
        assert_eq!(cross_quad_count(&layout), 7);
        let quads = cross_quads(&layout);
        assert_eq!(quads.len(), 7);
        // Quad 0 is the horizontal (tile 1); the other six are the +-0.25 pairs of 0/60/120 deg.
        assert_eq!(quads[0].uv_b.x, 1.0);
        for (i, quad) in quads[1..].iter().enumerate() {
            let theta = PI * (i % 3) as f32 / 3.0;
            assert_eq!(quad.uv_b, Vec2::new(0.0, theta));
            // Every corner sits on the plane `dot(p, n) = offset`, which is what the shader's
            // azimuth -> normal derivation assumes.
            let n = Vec3::new(theta.sin(), 0.0, theta.cos());
            let offset = if i < 3 { -0.25 } else { 0.25 };
            for corner in quad.corners {
                assert!(
                    (corner.dot(n) - offset).abs() < 1e-6,
                    "{corner:?} off plane"
                );
            }
        }
    }

    /// `side_width: 1.0` — the default — must be the f32 IDENTITY on the outer planes, not merely
    /// "full width to within an ulp". The crop is a multiply into V0's x and a `0.5 + x` into the
    /// uv, so the whole "existing configurations are byte-identical" guarantee rests on
    /// `-0.5 * 1.0 == -0.5` and `0.5 + (-0.5) == 0.0` exactly. Covered for both `mirror_sides`
    /// settings, since mirroring rewrites the same uvs.
    #[test]
    fn side_width_one_leaves_the_outer_planes_untouched() {
        assert_eq!(CrossLayout::default().side_width, 1.0);
        // V0's uvs, and after `1.0 - u` the same four values in the other order — which is what
        // makes the mirrored case checkable against one table.
        let full_uvs = [
            Vec2::new(0.0, 1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 0.0),
        ];
        for mirror in [false, true] {
            let layout = CrossLayout {
                orientations: 3,
                side_offset: 0.25,
                center_planes: false,
                mirror_sides: mirror,
                ..default()
            };
            let quads = cross_quads(&layout);
            assert_eq!(quads.len(), 7);
            // Quad 0 is the horizontal; 1..7 are the outer pairs.
            for (i, quad) in quads[1..].iter().enumerate() {
                for c in 0..4 {
                    let expected = if mirror {
                        Vec2::new(1.0 - full_uvs[c].x, full_uvs[c].y)
                    } else {
                        full_uvs[c]
                    };
                    assert_eq!(
                        quad.uvs[c].to_array().map(f32::to_bits),
                        expected.to_array().map(f32::to_bits),
                        "outer quad {i} uv {c}"
                    );
                }
                // Corner 0 -> 1 is the in-plane HORIZONTAL edge (V0's +x, yawed), so its length is
                // the plane's width: exactly 1.0 unit, i.e. the full +-0.5 the centre planes span.
                assert_eq!(
                    (quad.corners[1] - quad.corners[0]).length().to_bits(),
                    1.0f32.to_bits(),
                    "outer quad {i} width"
                );
            }
        }
    }

    /// A cropped outer plane: narrower geometry AND a proportionally narrower slice of the side
    /// tile, so the image keeps its world scale instead of being squeezed. Checks the two together
    /// via the uv-per-unit density, because either half alone would pass a positions-only or a
    /// uvs-only test while rendering a compressed or a stretched cluster.
    #[test]
    fn side_width_crops_the_outer_planes_without_squeezing() {
        let w = 0.6f32;
        let layout = CrossLayout {
            horizontal_height: 0.0,
            orientations: 2,
            side_offset: 0.25,
            side_width: w,
            ..default()
        };
        let quads = cross_quads(&layout);
        assert_eq!(quads.len(), 7);

        // Centre planes (0, 1) sit on the widest row and must be untouched by the crop.
        for quad in &quads[..2] {
            assert_eq!((quad.corners[1] - quad.corners[0]).length(), 1.0);
            assert_eq!(quad.uvs[0].x, 0.0);
            assert_eq!(quad.uvs[1].x, 1.0);
        }

        // Outer planes 3..7. Orientation 0 is the XY plane (normal +Z), so its corners are the ones
        // that can be written out literally; the rest are its yaws and are checked structurally.
        for (i, quad) in quads[3..].iter().enumerate() {
            let offset = if i < 2 { -0.25 } else { 0.25 };
            let width = (quad.corners[1] - quad.corners[0]).length();
            assert!((width - w).abs() < 1e-6, "outer quad {i} width {width}");
            // Height is NOT cropped: the row is shorter, not shorter-and-stubbier.
            assert!(((quad.corners[3] - quad.corners[0]).length() - 1.0).abs() < 1e-6);
            // uv span / world span == 1.0, the same texel density the full-width centre planes
            // have. This is the "crop, don't squeeze" invariant.
            let du = (quad.uvs[1].x - quad.uvs[0].x).abs();
            assert!((du - w).abs() < 1e-6, "outer quad {i} u span {du}");
            assert!((du / width - 1.0).abs() < 1e-5, "outer quad {i} squeezed");
            // Symmetric about the tile centre: it is the MIDDLE of the row that is kept.
            assert!((quad.uvs[0].x + quad.uvs[1].x - 1.0).abs() < 1e-6);
            if i % 2 == 0 {
                // orientation 0: V0 narrowed in x and pushed along +-Z.
                assert!((quad.corners[0] - Vec3::new(-0.5 * w, -0.5, offset)).length() < 1e-6);
                assert!((quad.corners[2] - Vec3::new(0.5 * w, 0.5, offset)).length() < 1e-6);
            } else {
                // orientation 1: the same quad yawed 90 deg, (x, y, z) -> (z, y, -x).
                assert!((quad.corners[0] - Vec3::new(offset, -0.5, 0.5 * w)).length() < 1e-6);
                assert!((quad.corners[2] - Vec3::new(offset, 0.5, -0.5 * w)).length() < 1e-6);
            }
        }
    }

    /// `mirror_sides` composes with a crop rather than fighting it: the kept interval is symmetric
    /// about u = 0.5, so `1 - u` maps it onto ITSELF and the mirrored plane shows the same clipped
    /// region in reverse — not a different, off-centre slice of the tile.
    #[test]
    fn mirrored_side_planes_show_the_same_cropped_region() {
        let base = CrossLayout {
            orientations: 3,
            side_offset: 0.25,
            center_planes: false,
            side_width: 0.6,
            ..default()
        };
        let plain = cross_quads(&base);
        let mirrored = cross_quads(&CrossLayout {
            mirror_sides: true,
            ..base
        });
        for (i, (a, b)) in plain[1..].iter().zip(&mirrored[1..]).enumerate() {
            // Geometry is untouched by mirroring - only uv.x moves.
            assert_eq!(a.corners, b.corners, "outer quad {i} corners");
            let range = |q: &CrossQuad| {
                let (lo, hi) = q.uvs.iter().fold((f32::MAX, f32::MIN), |(lo, hi), uv| {
                    (lo.min(uv.x), hi.max(uv.x))
                });
                (lo, hi)
            };
            let (alo, ahi) = range(a);
            let (blo, bhi) = range(b);
            assert!(
                (alo - blo).abs() < 1e-6 && (ahi - bhi).abs() < 1e-6,
                "outer quad {i}: mirrored u range ({blo}, {bhi}) != ({alo}, {ahi})"
            );
            // ...and it really is mirrored, not a no-op: the two ends swapped.
            assert!((a.uvs[0].x - b.uvs[1].x).abs() < 1e-6);
            assert!((a.uvs[1].x - b.uvs[0].x).abs() < 1e-6);
        }
    }

    /// Two deliberately LOPSIDED rects: nothing here is symmetric in x, in y, or between the two
    /// tiles, so a transposed axis, a dropped `0.5 -`, or a side rect leaking into the top quad
    /// cannot survive by coincidence. Reused by every trim test below so they all describe the same
    /// asset.
    fn lopsided_trim() -> CrossLayout {
        CrossLayout {
            horizontal_height: 0.0,
            // side: narrow, tall, and OFF-CENTRE in u (centre 0.475, not 0.5) - which is what makes
            // "`side_width` narrows about the model centre, not about the quad's centre" a
            // distinguishable claim rather than a tautology.
            side_trim: Rect::new(0.25, 0.05, 0.70, 0.95),
            // top: wide, short, pushed left and towards the bottom of the image.
            top_trim: Rect::new(0.10, 0.55, 0.85, 0.80),
            ..default()
        }
    }

    /// Every quad must wind CCW about its OWN normal — +Z yawed by the azimuth for a vertical, +Y
    /// for the horizontal — at any trim, because both triangles are emitted with the fixed
    /// `QUAD_INDICES` order and nothing downstream re-derives a facing.
    ///
    /// Asserted rather than observed on purpose: `specialize` sets `cull_mode = None` and bevy
    /// applies a material's `specialize` to the prepass pipelines too, so an inverted quad looks
    /// and depth-writes exactly like a correct one. A winding bug in here would sit invisible until
    /// someone re-enables culling or feeds these meshes to something that cares.
    #[test]
    fn trimmed_quads_keep_their_winding() {
        // Both builders' triangles, in the order they index the 4 corners.
        let triangles = [[0usize, 1, 2], [0, 2, 3]];
        for (name, layout) in [
            ("full", CrossLayout::default()),
            ("lopsided", lopsided_trim()),
            (
                "lopsided + sides + crop",
                CrossLayout {
                    orientations: 3,
                    side_offset: 0.25,
                    side_width: 0.6,
                    center_planes: false,
                    mirror_sides: true,
                    ..lopsided_trim()
                },
            ),
        ] {
            for (i, quad) in cross_quads(&layout).iter().enumerate() {
                // The same derivation the vertex shader does from `uv_b`: tile 1 is the horizontal,
                // everything else rebuilds `(sin, 0, cos)` from the azimuth.
                let expected = if quad.uv_b.x == 1.0 {
                    Vec3::Y
                } else {
                    Vec3::new(quad.uv_b.y.sin(), 0.0, quad.uv_b.y.cos())
                };
                for tri in triangles {
                    let (a, b, c) = (
                        quad.corners[tri[0]],
                        quad.corners[tri[1]],
                        quad.corners[tri[2]],
                    );
                    let n = (b - a).cross(c - b);
                    // Non-degenerate AND same side: a zero-area triangle would pass a sign test.
                    assert!(n.length() > 1e-4, "{name} quad {i} {tri:?} degenerate");
                    assert!(
                        n.normalize().dot(expected) > 0.99,
                        "{name} quad {i} {tri:?} winds away from {expected:?}"
                    );
                }
            }
        }
    }

    /// The trim is a CROP, exactly as `side_width` is: the quad shrinks to the rect AND the uvs
    /// shrink to the same rect, so texels keep their world size. Checked as a uv-span/world-span
    /// ratio, because either half alone would pass a positions-only or a uvs-only test while
    /// rendering a stretched or squeezed tree.
    ///
    /// Also pins the axis flips: `min.y` of the rect is the TOP of the image and must come out as
    /// the quad's MAXIMUM y (side) / z (top).
    #[test]
    fn trim_shrinks_the_quads_and_crops_the_uvs() {
        let layout = lopsided_trim();
        let (side, top) = (layout.side_trim, layout.top_trim);
        let quads = cross_quads(&layout);
        assert_eq!(quads.len(), 3);

        // V0 - the XY plane. Corner 0 is (x lo, y lo), corner 2 is (x hi, y hi).
        let v0 = quads[0].corners;
        assert!((v0[0] - Vec3::new(side.min.x - 0.5, 0.5 - side.max.y, 0.0)).length() < 1e-6);
        assert!((v0[2] - Vec3::new(side.max.x - 0.5, 0.5 - side.min.y, 0.0)).length() < 1e-6);
        // ...and its uvs are the rect's own corners, in the image's y-down sense.
        assert!((quads[0].uvs[0] - Vec2::new(side.min.x, side.max.y)).length() < 1e-6);
        assert!((quads[0].uvs[2] - Vec2::new(side.max.x, side.min.y)).length() < 1e-6);
        // No squeeze: one unit of world across the quad is still one unit of uv across the tile.
        assert!(((v0[1] - v0[0]).length() - side.width()).abs() < 1e-6);
        assert!(((v0[3] - v0[0]).length() - side.height()).abs() < 1e-6);

        // V1 is V0 yawed 90 degrees and carries V0's uvs verbatim - including the crop.
        assert_eq!(quads[1].uvs, quads[0].uvs);

        // H - the XZ plane. Same shape of check through the top view's doubly-flipped convention.
        let hq = quads[2].corners;
        assert!((hq[0] - Vec3::new(0.5 - top.max.x, 0.0, 0.5 - top.min.y)).length() < 1e-6);
        assert!((hq[2] - Vec3::new(0.5 - top.min.x, 0.0, 0.5 - top.max.y)).length() < 1e-6);
        assert!((quads[2].uvs[0] - Vec2::new(top.max.x, top.min.y)).length() < 1e-6);
        assert!((quads[2].uvs[2] - Vec2::new(top.min.x, top.max.y)).length() < 1e-6);
        assert!(((hq[1] - hq[0]).length() - top.width()).abs() < 1e-6);
        assert!(((hq[3] - hq[0]).length() - top.height()).abs() < 1e-6);

        // The two rects really are independent - a shared rect would have made this pass by
        // accident if the two views happened to trim alike.
        assert!((v0[1] - v0[0]).length() < (hq[1] - hq[0]).length());
    }

    /// `side_width` narrows about x = 0 (the model centre), NOT about the centre of the trimmed
    /// quad — see the note on `CrossLayout::side_trim`. With a trim rect that is off-centre in u
    /// those two are different numbers, which is what makes this checkable at all.
    #[test]
    fn side_width_narrows_a_trimmed_plane_about_the_model_centre() {
        let w = 0.5f32;
        let layout = CrossLayout {
            orientations: 2,
            side_offset: 0.25,
            side_width: w,
            ..lopsided_trim()
        };
        let quads = cross_quads(&layout);
        assert_eq!(quads.len(), 7);
        let trim = layout.side_trim;
        // Corners 3..7 are the outer planes; orientation 0 keeps its extents in x.
        let outer = quads[3].corners;
        // Model-centre narrowing: [x0, x1] -> [w*x0, w*x1], so u -> [0.5 + w*(min-0.5), ...].
        assert!((outer[0].x - w * (trim.min.x - 0.5)).abs() < 1e-6);
        assert!((outer[2].x - w * (trim.max.x - 0.5)).abs() < 1e-6);
        assert!((quads[3].uvs[0].x - (0.5 + w * (trim.min.x - 0.5))).abs() < 1e-6);
        // The fixture has to stay off-centre in u or the two candidate behaviours coincide and this
        // test proves nothing - say so here rather than leaving it to whoever edits the rect.
        let trim_centre = 0.5 * (trim.min.x + trim.max.x);
        assert!(
            (trim_centre - 0.5).abs() > 1e-3,
            "lopsided_trim's side rect must not be centred on u = 0.5"
        );
        // Narrowing about the TRIMMED QUAD's centre would have left `trim_centre` where it is.
        // Narrowing about the model centre scales the whole interval toward u = 0.5, so its centre
        // ends up `w` of the way in - a different number, which is the point.
        let kept_centre = 0.5 * (quads[3].uvs[0].x + quads[3].uvs[1].x);
        assert!((kept_centre - (0.5 + w * (trim_centre - 0.5))).abs() < 1e-6);
        assert!((kept_centre - trim_centre).abs() > 1e-3);
        // Height is untouched by `side_width`, trimmed or not.
        assert!(((outer[3] - outer[0]).length() - trim.height()).abs() < 1e-6);
        // Still a crop, not a squeeze: uv span / world span is 1 as always.
        let du = (quads[3].uvs[1].x - quads[3].uvs[0].x).abs();
        let dx = (outer[1] - outer[0]).length();
        assert!((du / dx - 1.0).abs() < 1e-5);
    }

    /// The merged builder inherits the trim through `cross_quads` with no code of its own, so the
    /// per-tree positions are the trimmed quad scaled/yawed/translated — nothing rebuilds a
    /// full-size square along the way.
    #[test]
    fn merged_meshes_inherit_the_trim() {
        let layout = lopsided_trim();
        let instances = [CrossInstance {
            translation: Vec3::new(10.0, 0.0, -5.0),
            scale: 2.0,
            yaw: 0.0,
            mirror: false,
        }];
        let mesh = cross_imposter_merged_mesh(&layout, &instances);
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("no positions");
        };
        // V0's first two corners, scaled by `2 * scale` and translated - the same expression the
        // non-merged path leaves to the vertex shader.
        let quads = cross_quads(&layout);
        for corner in 0..4 {
            let expected =
                quads[0].corners[corner] * (instances[0].scale * 2.0) + instances[0].translation;
            assert!((Vec3::from(positions[corner]) - expected).length() < 1e-5);
        }
    }
}

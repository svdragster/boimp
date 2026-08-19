// spawn a gltf and bake dynamic imposters every frame
// the gltf can be animated or moved, and the changes are reflected in every imposter.
// scene mgmt copied wholesale from bevy

use std::f32::consts::{FRAC_PI_4, PI};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bevy::{
    animation::AnimationTargetId,
    anti_alias::fxaa::Fxaa,
    camera::{
        primitives::{Aabb, Sphere},
        visibility::RenderLayers,
    },
    core_pipeline::prepass::{DepthPrepass, NormalPrepass},
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::entity::EntityHashMap,
    math::FloatOrd,
    platform::collections::HashMap,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
        view::screenshot::{save_to_disk, Screenshot},
        RenderPlugin,
    },
    world_serialization::InstanceId,
};
use boimp::{
    bake::BakeState,
    cross::{
        CROSS_ANGLE_FADE_FLAG, CROSS_ANGLE_SHRINK_FLAG, CROSS_CHEAP_LIGHTING_FLAG,
        CROSS_TWO_SIDED_FLAG,
    },
    cross_bake_camera, cross_imposter_from_bake_target, cross_imposter_merged_mesh,
    cross_imposter_mesh, cross_quad_count, cross_trim_from_atlas,
    render::{DummyIndicesImage, DITHER_FLAG},
    CrossImposter, CrossImposterPlugin, CrossInstance, CrossLayout, CrossTrim, GridMode, Imposter,
    ImposterBakeCamera, ImposterBakePlugin, ImposterData,
};
use camera_controller::{CameraController, CameraControllerPlugin};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[path = "helpers/camera_controller.rs"]
mod camera_controller;

// Example-only instrumentation for a one-shot bake (SPACE). `impost` stamps `inflight` when it
// spawns the bake camera; `report_bake_timing` ticks the frame count each frame and, once the
// camera reaches BakeState::Finished, prints the end-to-end wall latency, frames spanned, the GPU
// bake-pass time (from RenderDiagnosticsPlugin), and the atlas VRAM. Useful for judging whether a
// bake can be afforded at runtime in a real game.
#[derive(Resource, Default)]
struct BakeProbe {
    inflight: Option<BakeProbeState>,
}

struct BakeProbeState {
    start: Instant,
    frames: u32,
}

#[derive(Resource)]
struct BakeSettings {
    mode: GridMode,
    grid_size: u32,
    tile_size: u32,
    count: usize,
    multisample_source: u32,
    multisample_target: bool,
    mask: bool,
    a2c: bool,
    fxaa: bool,
    dither: bool,
    coverage: bool,
    fade: bool,
    cluster: usize,
    // --cluster-grid: lay the `--cluster` copies out on a ceil(sqrt(n)) x ceil(sqrt(n)) grid
    // centred on the origin instead of the default Vogel/sunflower disc.
    //
    // This is what makes `--cross-sides` worth its overdraw. The extra vertical planes are slices
    // through whatever the cluster bake captured; against a random scatter a slice coincides with
    // nothing, so intermediate view angles read as flat cardboard. Against a grid, the planes can
    // be put ON the rows - the printout at the bottom of `setup_scene_after_load` exists purely to
    // make that alignment tunable by eye - and the silhouette then agrees with the real tree
    // positions. Rows without a plane of their own are fine: their trunks are occluded by
    // neighbouring foliage anyway.
    cluster_grid: bool,
    // --cluster-hex: lay the `--cluster` copies out on a TRIANGULAR lattice filling a hexagonal
    // footprint, instead of the square grid above or the default Vogel disc.
    //
    // Two reasons it beats the square grid for this system, both about the outer planes:
    //
    //   * a triangular lattice has THREE row directions at 60 degrees rather than two at 90, which
    //     is exactly what `--cross-orientations 3` puts planes on. Worst-case viewing azimuth is
    //     then 30 degrees off the best-aligned plane (|dot| >= 0.87) instead of 45 (0.707);
    //   * the footprint is a hexagon rather than a square, so it wastes less of the bake's bounding
    //     SPHERE - ~1.57x the texel density per tree at the same radius, since a square's corners
    //     push the radius out by sqrt(2) for trees the planes never line up with anyway.
    //
    // Mutually exclusive with --cluster-grid; if both are given, hex wins and a warning says so.
    cluster_hex: bool,
    spacing: f32,
    ambient: f32,
    swap: bool,
    swap_distance: f32,
    // --swap-fade: dither-dissolve the real model across [swap_distance/band, swap_distance] of
    // radius (solid up close, fully gone at the swap distance) instead of popping it in. Spawn and
    // despawn keep the usual 1.25x hysteresis at the swap distance regardless of the band.
    swap_fade: bool,
    swap_fade_band: f32,
    // --bake-tiles-per-frame: cap on how many atlas tiles the bake renders per frame, spreading a
    // one-shot bake over multiple frames to cut the per-frame spike. usize::MAX = all in one frame.
    bake_tiles_per_frame: usize,
    // --cross: render the scene with the `CrossImposter` system (three fixed quads sampling a
    // 2-tile atlas) instead of the octahedral `Imposter`. Everything else - source model, count,
    // spacing, camera, lights, bake trigger - is left alone, so the two modes are an A/B of the
    // render path only. `--grid` and `--mode` are ignored here: the cross bake always uses a 2x2
    // grid of manual views.
    cross: bool,
    // --cross-height: unit-space y of the horizontal (canopy) quad, in [-0.5, 0.5]. As REQUESTED on
    // the command line - `impost` divides out `--cluster`'s horizontal radius inflation before
    // handing it to the mesh builders, so this field is not what the mesh is built from.
    cross_height: f32,
    // --cross-angle-fade / --cross-two-sided / --cross-cheap-lighting: `CROSS_*_FLAG` opt-ins.
    // All three are off by default so `--cross` alone measures the cheapest possible fragment,
    // which is the number the plan's phase-0 go/no-go decision is made on; turn them on one at a
    // time to price each quality feature against that baseline.
    cross_angle_fade: bool,
    // --cross-angle-shrink: spend the ANGLE_FADE weight by scaling the quad down in the vertex
    // stage instead of dithering it per fragment. Needs --cross-angle-fade to do anything. The
    // difference is only visible in MOTION: pan the camera with the dither and a half-faded quad
    // sparkles, because the noise is pinned to the screen and the geometry is not.
    cross_angle_shrink: bool,
    cross_two_sided: bool,
    cross_cheap_lighting: bool,
    // --cross-fade-start / --cross-fade-end: the |dot(view_dir, quad_normal)| band ANGLE_FADE
    // smoothsteps over, defaulting to the (0.20, 0.50) in `CrossImposterData::new`. Exposed
    // because the useful setting is scene- and camera-dependent, and because "I see no difference"
    // is otherwise ambiguous between "working and unnecessary" and "not running at all" - the
    // original (0.05, 0.30) default was a measured no-op at this example's start camera (~19 deg
    // above horizontal puts the canopy quad at |dot| ~ 0.33, just past the old band).
    // To settle that ambiguity, set (0.9, 1.0): everything but near-face-on quads is discarded,
    // which is unmistakable. Screenshot-diff vs --cross with no fade, same seed: 0.36% of pixels
    // changed at (0.05, 0.30), 1.58% at (0.20, 0.50), 22.7% at (0.90, 1.00).
    cross_fade_start: f32,
    cross_fade_end: f32,
    // --cross-sides <f>: unit-space offset of the OUTER plane pair, per orientation (two more quads
    // each). 0.0 = off, and off is a byte-identical no-op on the mesh, not merely an equivalent one.
    //
    // Only useful with a lattice-placed cluster (`--cluster n` plus `--cluster-grid` or
    // `--cluster-hex`): see the `cluster_grid`/`cluster_hex` notes above and `CrossLayout` in
    // src/cross.rs. Each orientation's pair is pure extra vertical overdraw, so on the iGPU this
    // system targets it is a quality-for-fill trade to be measured, not a free win. The alignment
    // printout at the bottom of `setup_scene_after_load` recommends a value for the current cluster.
    cross_sides: f32,
    // --cross-side-width <f>: fraction of the full quad width the OUTER planes span, cropping the
    // side tile by the same factor about its centre so the picture keeps its world scale. 1.0 (the
    // default) is the original full-width behaviour and a byte-identical no-op on the mesh.
    //
    // Exists because a hexagonal cluster's rows are NOT all the same length: the outermost row of a
    // `--cluster-hex` lattice is exactly half as long as the centre one (hexagon edge vs long
    // diagonal), so a full-width outer plane draws trees beyond the footprint the horizontal canopy
    // quad shows from above - they stick out at the sides with nothing overhead to match. A square
    // grid has equal-length rows and wants 1.0. The alignment printout computes the right value from
    // the actual cluster geometry; see `CrossLayout::side_width` in src/cross.rs for the derivation.
    cross_side_width: f32,
    // --cluster-spread: multiplier on the cluster's horizontal spread at BAKE time. <1 packs the
    // source copies closer (fewer ground gaps between canopies, and a tighter bounding sphere so
    // each tree gets more atlas texels); >1 spreads them out. Distinct from --spacing, which moves
    // the spawned imposters apart and has no effect on what is baked.
    cluster_spread: f32,
    // --hide-source: put the main camera on render layer 1 ONLY, so the source glTF cluster (which
    // lives on layer 0 and is never hidden) stops being drawn into the main pass. It is 1614 tris of
    // alpha-masked foliage per copy, x --cluster, sitting close and dead centre in the view - a large
    // CONSTANT fragment cost present in every configuration, which dilutes any imposter A/B measured
    // from main_opaque_pass_3d. The bake camera runs its own visibility pass over layer 0, so baking
    // is unaffected; you just lose the real-model reference next to the imposters.
    hide_source: bool,
    // --cross-orientations <n>: how many vertical plane DIRECTIONS, spread over 180 degrees. 2 (the
    // default) is the original perpendicular cross; 3 gives 0/60/120 degrees, matching the three row
    // directions of `--cluster-hex`'s triangular lattice.
    cross_orientations: u32,
    // --cross-no-center: drop the centre plane of every orientation, keeping only the --cross-sides
    // pairs. Established by experiment: interior rows of a dense cluster are hidden behind foliage
    // from every direction, while the outer rows are what define the silhouette. Dropping the
    // centres buys back one quad per orientation, which is what makes 3 orientations x 2 outer
    // planes + 1 horizontal cost exactly the 7 quads the square 3-planes-per-axis layout did.
    cross_no_center: bool,
    // --cross-mirror-sides: flip uv.x on the outer planes. All the verticals sample the one side
    // tile, which for a cluster bake shows the whole group, so parallel copies of that image are
    // copies of a recognisable arrangement. Mirroring the outer pairs breaks the repetition for
    // free. Exposed as a toggle because whether it helps is a judgement call.
    cross_mirror_sides: bool,
    // --cross-trim: size each quad to the BOUNDING BOX OF ITS TILE'S CONTENT instead of to the bake
    // sphere, cropping its uvs by the same factor (`CrossLayout::side_trim` / `top_trim`, measured
    // from the readback by `cross_trim_from_atlas`).
    //
    // A cross quad is a square as wide as the bounding sphere, but the picture in it is a tree
    // (tall, narrow) or a canopy seen from above (a disc): ~85% of every quad's fragments are pure
    // discards that still pay a full atlas fetch, because colour and alpha share one Rg32Uint texel
    // and the shader cannot know a fragment is empty until it has loaded it. Trimming deletes those
    // fragments outright.
    //
    // OPT-IN, and the default-off path is not merely equivalent but literally the same code, so the
    // untrimmed run stays the reference the trimmed one is judged against. Every texel the trim
    // removes was already below the fragment shader's own 0.01 alpha discard, so no content can be
    // lost; the measured frame diff at 40k clustered imposters is 0.34% of pixels, 89% of them in a
    // six-row band at the far horizon where an imposter is a few pixels across and shrinking a quad
    // reshuffles which texel a pixel centre lands on. See `CrossLayout::side_trim` for the full
    // measurement. (The untrimmed run reproduces itself at max 0, so that 0.34% is the trim, not
    // run-to-run noise - which is also why `--cluster`'s jitter had to be seeded, see
    // `setup_scene_after_load`.)
    cross_trim: bool,
    // --cross-trim-pad <n>: texels of margin around the measured box. Float-rounding insurance
    // only - the box is exact in texels, but it travels through f32 interpolation and a floor() in
    // the shader before it addresses one. 1 costs a fraction of a percent of the saving.
    cross_trim_pad: u32,
    // --cross-trim-cutoff <f>: alpha at or above which a texel counts as content. Defaults to 0.01,
    // which is `cross_fragment.wgsl`'s own discard threshold - that equality is what makes the trim
    // lossless, so raising this is not a quality knob, it is a decision to start cutting texels that
    // would have been drawn.
    cross_trim_cutoff: f32,
    // --cross-merge <n>: merge the spawned cross imposters into patches of n trees, one entity and
    // one draw per patch, instead of one entity per tree. 0 or 1 keeps the per-entity behaviour.
    //
    // This is the cross system's answer to `--cluster`, and it is NOT the same trick. `--cluster`
    // bakes N models into one atlas, which a cross imposter cannot use: two pictures on three fixed
    // planes turn a baked cluster into a flat wall of trees with no inter-tree parallax, and the
    // cluster radius grows as base_radius*sqrt(N) while tile_size stays fixed, so per-tree texel
    // density collapses. Merging MESHES keeps every tree at full atlas resolution with its own
    // correct cross and still collapses the patch to a single draw.
    cross_merge: usize,
    // --auto-bake: fire the one-shot bake+spawn (the SPACE path) automatically as soon as the
    // scene has loaded and `setup_scene_after_load` has computed the bounding sphere. Exists so a
    // scripted A/B run needs no window focus and no input automation.
    auto_bake: bool,
    // --exit-after <secs>: once the imposters have been spawned, run for this long, print the
    // machine-greppable summary block and exit(0).
    exit_after: Option<f32>,
    // --debug-level <n>: initial value of the fragment perf ladder, applied to the material at
    // spawn time. The 0-4 key bindings still work on top of it.
    debug_level: u32,
    // --dump-atlas <path>: write the baked atlas albedo to a PNG (same decode as planet.rs).
    dump_atlas: Option<String>,
    // --screenshot <path>: with --exit-after, grab the primary window one second before quitting,
    // so an unattended run leaves behind a picture of what it actually rendered.
    screenshot: Option<String>,
    // --seed <u64>: seed for the imposter PLACEMENT rng. Fixed by default, because the A/B this
    // example exists to run is only valid if both modes get the identical scene: with an unseeded
    // rng, 1000 randomly-placed trees produce a different silhouette, a different amount of screen
    // coverage and a different overdraw depth every launch, which lands squarely on top of the
    // effect being measured. The camera is at a fixed position, so this is not a small term.
    seed: u64,
    // --shadows: spawn the default directional light with shadow maps already on (the `U` key does
    // the same thing interactively). Needed unattended because a shadow-caster pass is the ONLY
    // thing in this example that compiles the cross fragment shader's `PREPASS_PIPELINE` entry
    // points - without it those two entry points are never specialized and never validated.
    shadows: bool,
    // --prepass: add DepthPrepass + NormalPrepass to the camera. The only way to compile the
    // imposter fragment shaders' `PREPASS_FRAGMENT` (struct-returning) entry point - the shadow
    // pass alone only reaches the void, depth-only one.
    prepass: bool,
}

// Drives `--auto-bake` / `--exit-after`. `spawned_at` is stamped by `impost` when it spawns the
// imposters, so the measurement window starts at spawn and never includes scene loading.
#[derive(Resource, Default)]
struct AutoRun {
    spawned_at: Option<Instant>,
    done: bool,
    shot: bool,
}

fn main() {
    println!(
        "press SPACE to bake the imposter once and spawn the imposters. press O to clear them.\n\
         press I to instead bake continuously every frame (for animated/moving sources).\n\
         press 0-4 to step the fragment-shader PERF LADDER on the live material: 0 = normal,\n\
         1 = skip lighting/post, 2 = skip detail fade (octahedral) / flat colour + zero atlas\n\
         fetches (cross), 3 = force single-tap sampling, 4 = flat colour + zero fetches.\n\
         the cross ladder stops at 2 and clamps; the delta between rungs is what attributes\n\
         frame time to a single stage, so measure with 0-4 rather than guessing."
    );

    App::new()
        // AmbientLight is configured in `setup` from CLI args (--ambient / --no-ambient)
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: bevy::window::PresentMode::Immediate,
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                // TEMP: request GPU timestamp queries so RenderDiagnosticsPlugin can
                // measure actual GPU pass time (CPU-vs-GPU-bound diagnosis).
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                        features: WgpuFeatures::TIMESTAMP_QUERY
                            | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                        ..Default::default()
                    })),
                    ..Default::default()
                })
                // examples accept an arbitrary `--source` path, which may be absolute / outside the
                // asset root; 0.18 forbids such paths by default.
                .set(AssetPlugin {
                    unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                    ..Default::default()
                }),
            CameraControllerPlugin,
            ImposterBakePlugin,
            // Added UNCONDITIONALLY, not behind `--cross`: the point of this example is an A/B of
            // the two render paths, and a plugin present in one mode but not the other would also
            // change the render graph, the schedule and the set of specialized pipelines - a
            // confound in exactly the measurement we are taking. `CrossImposterPlugin` costs one
            // idle `MaterialPlugin` when no `CrossImposter` exists, and is explicitly designed to
            // coexist with `ImposterBakePlugin`'s `ImposterRenderPlugin` in either order (see the
            // `DummyIndicesImage` comment in `src/cross.rs`).
            CrossImposterPlugin,
        ))
        // FrameTimeDiagnosticsPlugin keeps the manual `G` dump (dump_diagnostics) working.
        // LogDiagnosticsPlugin/RenderDiagnosticsPlugin are disabled for now: the former spams the
        // log every second, the latter adds GPU-timing spans we don't need while debugging baking.
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(RenderDiagnosticsPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(PreUpdate, setup_scene_after_load)
        .add_systems(
            Update,
            (
                scene_load_check,
                impost,
                update_lights,
                rotate,
                swap_old,
                swap_close,
                dress_real_models,
                toggle_dither,
                setup_anim_after_load,
                dump_diagnostics,
                print_fps,
                report_bake_timing,
                set_debug_level,
                spawn_trimmed_cross_imposters,
            ),
        )
        .add_systems(Update, auto_exit)
        .init_resource::<BakeProbe>()
        .init_resource::<AutoRun>()
        .run();
}

fn parse_scene(scene_path: String) -> (String, usize) {
    if scene_path.contains('#') {
        let gltf_and_scene = scene_path.split('#').collect::<Vec<_>>();
        if let Some((last, path)) = gltf_and_scene.split_last() {
            if let Some(index) = last
                .strip_prefix("Scene")
                .and_then(|index| index.parse::<usize>().ok())
            {
                return (path.join("#"), index);
            }
        }
    }
    (scene_path, 0)
}

#[derive(Resource, Debug)]
pub struct SceneHandle {
    pub gltf_handle: Handle<Gltf>,
    scene_index: usize,
    instance_id: Option<InstanceId>,
    pub is_loaded: bool,
    pub has_light: bool,
    pub sphere: Sphere,
    /// Radius of the SINGLE source model, before `--cluster`'s horizontal spread was added to
    /// `sphere.radius`. Equal to `sphere.radius` when `--cluster 1`.
    ///
    /// Kept because `sphere.radius` is not a measure of the model at all once a cluster is baked:
    /// it grows with the group's horizontal footprint while the trees stay exactly as tall as they
    /// were. Anything expressed as a fraction of the bounding sphere but MEANT as a fraction of the
    /// model's own height - `--cross-height` is the one such knob - has to divide the inflation
    /// back out, and this is the only place the un-inflated number survives. See the effective-height
    /// computation in `impost`.
    pub base_radius: f32,
}

impl SceneHandle {
    pub fn new(gltf_handle: Handle<Gltf>, scene_index: usize) -> Self {
        Self {
            gltf_handle,
            scene_index,
            instance_id: None,
            is_loaded: false,
            has_light: false,
            sphere: Sphere::default(),
            base_radius: 0.0,
        }
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let mut args = pico_args::Arguments::from_env();
    let grid_size = args.value_from_str("--grid").unwrap_or(15);
    let tile_size = args.value_from_str("--tile").unwrap_or(128);
    let mode = match args
        .value_from_str("--mode")
        .unwrap_or("h".to_owned())
        .chars()
        .next()
        .unwrap()
    {
        'h' => GridMode::Hemispherical,
        'H' => GridMode::Horizontal,
        's' => GridMode::Spherical,
        _ => {
            warn!("unrecognized mode, use [h]emispherical or [s]pherical. defaulting to hemispherical");
            GridMode::Hemispherical
        }
    };
    let count = args.value_from_str("--count").unwrap_or(1000);
    let scene_path = args
        .value_from_str("--source")
        .unwrap_or_else(|_| "models/FlightHelmet/FlightHelmet.gltf".to_string());
    let multisample_target = args.contains("--multisample-target");
    let multisample_source = args.value_from_str("--multisample-source").unwrap_or(1);
    let mask = args.contains("--mask");
    let a2c = args.contains("--a2c");
    let fxaa = args.contains("--fxaa");
    let dither = args.contains("--dither");
    let coverage = args.contains("--coverage");
    let fade = args.contains("--fade");
    let cluster = args.value_from_str("--cluster").unwrap_or(1);
    let cluster_hex = args.contains("--cluster-hex");
    let cluster_grid = args.contains("--cluster-grid");
    let spacing = args.value_from_str("--spacing").unwrap_or(1.0);
    let ambient_brightness: f32 = args.value_from_str("--ambient").unwrap_or(1000.0);
    let no_ambient = args.contains("--no-ambient");
    let swap = args.contains("--swap");
    let swap_distance: f32 = args.value_from_str("--swap-distance").unwrap_or(8.0);
    let swap_fade = args.contains("--swap-fade");
    let swap_fade_band: f32 = args.value_from_str("--swap-fade-band").unwrap_or(1.6);
    let bake_tiles_per_frame: usize = args
        .value_from_str("--bake-tiles-per-frame")
        .unwrap_or(usize::MAX);
    let cross = args.contains("--cross");
    // +0.1125 puts the canopy quad slightly above the bounding-sphere centre: the seam where it cuts
    // the two verticals is far less visible inside dense foliage than at the silhouette edge.
    let cross_height: f32 = args.value_from_str("--cross-height").unwrap_or(0.1125);
    let cross_angle_fade = args.contains("--cross-angle-fade");
    let cross_angle_shrink = args.contains("--cross-angle-shrink");
    let cross_two_sided = args.contains("--cross-two-sided");
    let cross_cheap_lighting = args.contains("--cross-cheap-lighting");
    let cross_fade_start: f32 = args.value_from_str("--cross-fade-start").unwrap_or(0.20);
    let cross_fade_end: f32 = args.value_from_str("--cross-fade-end").unwrap_or(0.50);
    let cross_merge: usize = args.value_from_str("--cross-merge").unwrap_or(0);
    // Off by default: the untrimmed path is the reference the 0.00%-diff claim is measured against,
    // so it has to stay the thing that runs unless asked otherwise.
    let cross_trim = args.contains("--cross-trim");
    let cross_trim_pad: u32 = args.value_from_str("--cross-trim-pad").unwrap_or(1);
    // 0.01 == the `a < 0.01` discard in cross_fragment.wgsl. Matching it exactly is what makes the
    // trim provably free rather than approximately free.
    let cross_trim_cutoff: f32 = args.value_from_str("--cross-trim-cutoff").unwrap_or(0.01);
    // 0.0 = off, and off must stay the default: the side planes are a straight overdraw increase
    // that only pays for itself against a `--cluster-grid` bake.
    let cross_sides: f32 = args.value_from_str("--cross-sides").unwrap_or(0.0);
    // 1.0 = full-width outer planes, the pre-crop behaviour, and a byte-identical no-op on the mesh.
    // Only a hex cluster has rows of differing lengths, so only it wants anything else.
    let cross_side_width: f32 = args.value_from_str("--cross-side-width").unwrap_or(1.0);
    let cluster_spread: f32 = args.value_from_str("--cluster-spread").unwrap_or(1.0);
    let hide_source = args.contains("--hide-source");
    // 2 = the original perpendicular cross, so the default layout is unchanged.
    let cross_orientations: u32 = args.value_from_str("--cross-orientations").unwrap_or(2);
    let cross_no_center = args.contains("--cross-no-center");
    let cross_mirror_sides = args.contains("--cross-mirror-sides");
    let auto_bake = args.contains("--auto-bake");
    let exit_after: Option<f32> = args.opt_value_from_str("--exit-after").unwrap_or(None);
    let debug_level: u32 = args.value_from_str("--debug-level").unwrap_or(0);
    let dump_atlas: Option<String> = args.opt_value_from_str("--dump-atlas").unwrap_or(None);
    let screenshot: Option<String> = args.opt_value_from_str("--screenshot").unwrap_or(None);
    let seed: u64 = args
        .value_from_str("--seed")
        .unwrap_or(0xB01A_B01A_B01A_B01A);
    let shadows = args.contains("--shadows");
    let prepass = args.contains("--prepass");

    let unused = args.finish();
    if !unused.is_empty() {
        println!("unrecognized arguments: {unused:?}");
        println!("args: \n--mode [h]emispherical or [s]pherical\n--grid n (grid size, default 15)\n--image n (image size, default 1024)\n--count n (number of imposters to spawn)\n--multisample-source <n> (to multisample when generating the imposter, try 8)\n--multisample-target (to multisample when rendering imposters)\n--mask (use AlphaMode::Mask instead of Blend, enabling early-Z)\n--a2c (use AlphaMode::AlphaToCoverage: MSAA anti-aliases the alpha-tested silhouette edges, no temporal pass; overrides --mask)\n--fxaa (enable FXAA screen-space anti-aliasing on the camera)\n--dither (static stochastic screen-space dither tile selection instead of the continuous blend; toggle at runtime with F)\n--coverage (coverage-preserving alpha for distant foliage: rescales+softens minified alpha so thin features keep density and feed A2C/MSAA fractional coverage; pair with --a2c)\n--fade (distance detail fade: as imposters minify, flatten the baked normal map, raise roughness, and desaturate albedo toward a smooth blob to kill far-away sparkle)\n--cluster n (bake n randomly-placed copies of the source model into a single imposter, default 1)\n--cluster-grid (lay the --cluster copies out on a ceil(sqrt(n)) x ceil(sqrt(n)) grid centred on the origin, lightly jittered, instead of the default Vogel/sunflower disc; the grid has ROWS, which is what --cross-sides' extra vertical planes can be lined up with)\n--cluster-hex (lay the --cluster copies out on a TRIANGULAR lattice filling a hexagon, instead of --cluster-grid's square grid; three row directions at 60 degrees instead of two at 90 - which is what --cross-orientations 3 puts planes on - and a rounder footprint that wastes less of the bake's bounding sphere. Clean sizes are the centered hexagonal numbers 1, 7, 19, 37, 61; anything else leaves the outer ring partial and says so)\n--spacing f (scales the gap between spawned imposters, default 1.0; <1 packs them closer, >1 spreads them out)\n--ambient f (ambient light brightness/fill, default 1000.0)\n--no-ambient (disable ambient fill, leaving only the directional light)\n--swap (swap each imposter for the real glTF model when the camera gets close, and back to the imposter when it moves away)\n--swap-distance f (camera distance, in multiples of the model radius, at which --swap kicks in, default 8.0)\n--swap-fade (dither-dissolve the real model across a distance band instead of popping it to/from the imposter; the imposter shows through the dither holes for a smooth cross-fade)\n--swap-fade-band f (width of the --swap-fade band: the real model is fully dithered away at swap-distance and solidifies as the camera closes to swap-distance/band, default 1.6)\n--bake-tiles-per-frame n (cap atlas tiles baked per frame to spread a one-shot bake over multiple frames and cut the per-frame hitch; grid_size^2 tiles total, default: all in one frame)\n--cross (render with the CrossImposter system - 3 fixed quads over a 2-tile side/top atlas - instead of the octahedral Imposter, for an A/B of the two render paths in the same scene; ignores --grid/--mode/--dither/--coverage/--fade/--multisample-target and is incompatible with --swap; defaults to AlphaMode::Mask)\n--cross-height f (unit-space height of the horizontal canopy quad in [-0.5, 0.5], default 0.1125; higher hides the seam with the vertical quads inside the canopy. Measured against a SINGLE tree's radius: with --cluster n the bounding sphere is inflated by the group's horizontal spread, so the value is divided by that inflation at spawn - the effective height is printed - and 0.15 means the same thing at every --cluster)\n--cross-angle-fade (fade each cross quad out as it turns edge-on, killing the bright horizontal line the canopy quad otherwise draws through every instance)\n--cross-two-sided (flip baked normals toward the viewer, so the back faces of the double-sided quads don't shade black)\n--cross-cheap-lighting (replace apply_pbr_lighting with a single directional Lambert + ambient; the biggest remaining iGPU lever once atlas sampling is gone)\n--auto-bake (fire the one-shot bake+spawn automatically once the scene has loaded, as if SPACE had been pressed - for unattended/scripted A/B runs)\n--exit-after f (after the imposters are spawned, run for f seconds, print the ===BOIMP SUMMARY=== block and exit 0)\n--debug-level n (initial fragment perf-ladder rung, applied at spawn; the 0-4 keys still work)\n--dump-atlas <path> (write the baked atlas albedo to this PNG)\n--screenshot <path> (with --exit-after, capture the window one second before quitting)\n--seed n (rng seed for imposter placement; fixed by default so two runs render the identical scene)\n--shadows (spawn the directional light with shadow maps on, same as pressing U; exercises the imposter prepass/shadow-caster shader path)\n--prepass (add a depth+normal prepass to the camera; exercises the imposter shaders' PREPASS_FRAGMENT entry point)\n--cross-fade-start f / --cross-fade-end f (the abs(dot(view,quad_normal)) band --cross-angle-fade smoothsteps over, default 0.20 / 0.50; set to 0.9 / 1.0 to make the fade unmistakable for debugging)\n--cross-merge n (merge the cross imposters into patches of n trees: one mesh, one entity and one draw per patch instead of per tree, with every tree at full atlas resolution. 0 or 1 = one entity per tree, the default)\n--cross-sides f (add an OUTER PAIR of vertical quads at unit-space +-f to every orientation, two more quads each. Default 0.0 = off. Pair with --cluster n plus --cluster-grid or --cluster-hex and line the planes up with the rows the alignment printout lists - it also prints the exact recommended value; every pair is straight vertical overdraw, so price it on the target GPU)\n--cross-side-width f (fraction of the full quad width the --cross-sides OUTER planes span, default 1.0 = full width. The side tile is CROPPED to the middle f of itself by the same factor, so the trees keep their world scale and only the parts belonging to other rows are clipped off. Wanted by --cluster-hex, whose outermost row is exactly half the length of its centre row (a hexagon's edge against its long diagonal), so full-width outer planes draw trees outside the footprint the canopy quad shows from above; a square grid's rows are all the same length and want 1.0. The alignment printout recommends the exact value)\n--cross-orientations n (number of vertical plane DIRECTIONS, spread over 180 degrees, default 2. 2 = the original perpendicular cross (0/90 deg); 3 = hex (0/60/120 deg), matching --cluster-hex's three row directions - worst-case viewing azimuth is then 30 deg off the best plane instead of 45)\n--cross-no-center (drop the centre plane of every orientation, keeping only the --cross-sides pairs. Interior rows of a dense cluster are hidden behind foliage anyway, and this is what makes --cross-orientations 3 --cross-no-center --cross-sides f cost the same 7 quads as the square 3-planes-per-axis layout)\n--cross-mirror-sides (flip uv.x on the outer planes, so parallel copies of the same side tile don't read as obvious repetition)\n--cross-trim (size every quad to the BOUNDING BOX OF ITS TILE'S CONTENT, measured off the baked atlas, and crop its uvs by the same factor, instead of spanning the whole bake sphere. ~85% of an untrimmed quad's fragments are pure discards that still pay a full atlas fetch - colour and alpha share one Rg32Uint texel, so the fetch cannot be skipped - and the trim deletes them. Every texel it removes was already below the fragment shader's own 0.01 alpha discard, so no content can be lost; the measured frame diff at 40k clustered imposters is 0.34% of pixels, 89% of them in a six-row band at the far horizon where an imposter is a few pixels wide and a smaller quad reshuffles which texel a pixel centre lands on. The measured rects and the resulting quad-area fractions are printed. Off by default, and off is the reference that claim is measured against; the spawn WAITS for the bake readback when it is on)\n--cross-trim-pad n (texels of margin around the --cross-trim boxes, default 1. Float-rounding insurance only: the box is exact in texels but travels through f32 interpolation and a floor() in the shader before it addresses one)\n--cross-trim-cutoff f (alpha at or above which a texel counts as content for --cross-trim, default 0.01 = cross_fragment.wgsl's own discard threshold. That equality is what makes the trim lossless, so raising it is not a quality knob - it is a decision to start cutting texels that would have been drawn)\n--cluster-spread f (multiplier on the cluster's horizontal spread when BAKING, default 1.0; <1 packs the source trees closer so there are fewer ground gaps between canopies AND the bounding sphere shrinks, giving every tree more atlas texels. --cluster 37 --cluster-spread 0.72 fits 37 trees in the footprint 19 had. Not the same as --spacing, which moves the spawned imposters and does not change the bake)\n--hide-source (put the main camera on render layer 1 only, so the source glTF cluster is not drawn in the main pass. It is alpha-masked foliage sitting close and centre-screen, a large constant cost in every config that dilutes imposter A/B measurements. Use it for any perf comparison)\n--source <path> (asset to load, default flight helmet)");
        std::process::exit(1);
    }

    if cross && swap {
        // `--swap` drives everything off the `SwapReal` component on the imposter entities, and the
        // cross-fade half of it needs `ImposterData::swap_fade`, which `CrossImposterData` has no
        // equivalent of. Rather than half-support it (real models popping in while the cross quads
        // stay solid behind them), cross mode simply never attaches `SwapReal`, so `swap_close` and
        // `dress_real_models` find nothing to do.
        warn!("--swap is ignored with --cross: swapping is only implemented for the octahedral imposter");
    }

    if cross_trim && !cross {
        // Silent no-op otherwise: the trim only sizes CROSS quads, and the octahedral billboard has
        // no equivalent (its uv is derived per fragment from the view ray, not baked into a mesh).
        warn!("--cross-trim does nothing without --cross: it sizes the cross quads, and the octahedral imposter has no mesh-baked uvs to crop");
    }

    if cluster_grid && cluster_hex {
        // One placement, not a blend of two. Hex wins because it is the more specific request: it
        // is the layout `--cross-orientations 3` is aligned against, and asking for both is far more
        // likely to be a leftover `--cluster-grid` on the command line than a deliberate choice.
        warn!("--cluster-grid and --cluster-hex are mutually exclusive; using --cluster-hex");
    }

    info!("settings: grid: {grid_size}, tile: {tile_size}, mode: {mode:?}");
    info!("Loading {}", scene_path);
    let (file_path, scene_index) = parse_scene(scene_path);

    commands.insert_resource(SceneHandle::new(asset_server.load(file_path), scene_index));
    commands.insert_resource(BakeSettings {
        mode,
        grid_size,
        tile_size,
        count,
        multisample_source,
        multisample_target,
        mask,
        a2c,
        fxaa,
        coverage,
        fade,
        dither,
        cluster,
        cluster_grid,
        cluster_hex,
        spacing,
        ambient: if no_ambient { 0.0 } else { ambient_brightness },
        swap,
        swap_distance,
        swap_fade,
        swap_fade_band,
        bake_tiles_per_frame,
        cross,
        cross_height,
        cross_angle_fade,
        cross_angle_shrink,
        cross_two_sided,
        cross_cheap_lighting,
        cross_fade_start,
        cross_fade_end,
        cross_merge,
        cross_trim,
        cross_trim_pad,
        cross_trim_cutoff,
        cross_sides,
        cross_side_width,
        cluster_spread,
        hide_source,
        cross_orientations,
        cross_no_center,
        cross_mirror_sides,
        auto_bake,
        exit_after,
        debug_level,
        dump_atlas,
        screenshot,
        seed,
        shadows,
        prepass,
    });
    // replaced with the real layout once the scene loads (see setup_scene_after_load); a
    // single identity entry keeps `swap_close`'s resource lookup valid until then.
    commands.insert_resource(ClusterLayout(vec![(Vec3::ZERO, Quat::IDENTITY)]));
}

fn scene_load_check(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut scenes: ResMut<Assets<WorldAsset>>,
    gltf_assets: Res<Assets<Gltf>>,
    mut scene_handle: ResMut<SceneHandle>,
    mut scene_spawner: ResMut<WorldInstanceSpawner>,
) {
    match scene_handle.instance_id {
        None => {
            if asset_server
                .load_state(&scene_handle.gltf_handle)
                .is_loaded()
            {
                let gltf = gltf_assets.get(&scene_handle.gltf_handle).unwrap();
                if gltf.scenes.len() > 1 {
                    info!(
                        "Displaying scene {} out of {}",
                        scene_handle.scene_index,
                        gltf.scenes.len()
                    );
                    info!("You can select the scene by adding '#Scene' followed by a number to the end of the file path (e.g '#Scene1' to load the second scene).");
                }

                let gltf_scene_handle =
                    gltf.scenes
                        .get(scene_handle.scene_index)
                        .unwrap_or_else(|| {
                            panic!(
                                "glTF file doesn't contain scene {}!",
                                scene_handle.scene_index
                            )
                        });
                let mut scene = scenes.get_mut(gltf_scene_handle).unwrap();

                let mut query = scene
                    .world
                    .query::<(Option<&DirectionalLight>, Option<&PointLight>)>();
                scene_handle.has_light =
                    query
                        .iter(&scene.world)
                        .any(|(maybe_directional_light, maybe_point_light)| {
                            maybe_directional_light.is_some() || maybe_point_light.is_some()
                        });

                let root = commands
                    .spawn((
                        Transform::from_scale(Vec3::splat(1.0)),
                        Visibility::default(),
                    ))
                    .id();
                scene_handle.instance_id =
                    Some(scene_spawner.spawn_as_child(gltf_scene_handle.clone(), root));

                info!("Spawning scene...");
            }
        }
        Some(instance_id) if !scene_handle.is_loaded => {
            if scene_spawner.instance_is_ready(instance_id) {
                info!("...done!");
                scene_handle.is_loaded = true;
            }
        }
        Some(_) => {}
    }
}

fn setup_anim_after_load(
    mut setup: Local<bool>,
    mut players: Query<&mut AnimationPlayer>,
    targets: Query<(Entity, &AnimationTargetId)>,
    parents: Query<&ChildOf>,
    scene_handle: Res<SceneHandle>,
    clips: Res<Assets<AnimationClip>>,
    gltf_assets: Res<Assets<Gltf>>,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    mut commands: Commands,
) {
    if scene_handle.is_loaded && !*setup {
        *setup = true;
    } else {
        return;
    }

    let gltf = gltf_assets.get(&scene_handle.gltf_handle).unwrap();
    let animations = &gltf.animations;
    if animations.is_empty() {
        return;
    }

    // copied wholesale from animation_plugin
    let animation_target_id_to_entity: HashMap<_, _> = targets
        .iter()
        .map(|(entity, target)| (*target, entity))
        .collect();

    let mut player_to_graph: EntityHashMap<(AnimationGraph, Vec<AnimationNodeIndex>)> =
        EntityHashMap::default();

    for (clip_id, clip) in clips.iter() {
        let mut ancestor_player = None;
        for target_id in clip.curves().keys() {
            // If the animation clip refers to entities that aren't present in
            // the scene, bail.
            let Some(&target) = animation_target_id_to_entity.get(target_id) else {
                continue;
            };

            // Find the nearest ancestor animation player.
            let mut current = Some(target);
            while let Some(entity) = current {
                if players.contains(entity) {
                    match ancestor_player {
                        None => {
                            // If we haven't found a player yet, record the one
                            // we found.
                            ancestor_player = Some(entity);
                        }
                        Some(ancestor) => {
                            // If we have found a player, then make sure it's
                            // the same player we located before.
                            if ancestor != entity {
                                // It's a different player. Bail.
                                ancestor_player = None;
                                break;
                            }
                        }
                    }
                }

                // Go to the next parent.
                current = parents.get(entity).ok().map(|child_of| child_of.parent());
            }
        }

        let Some(ancestor_player) = ancestor_player else {
            warn!(
                "Unexpected animation hierarchy for animation clip {:?}; ignoring.",
                clip_id
            );
            continue;
        };

        let Some(clip_handle) = asset_server.get_id_handle(clip_id) else {
            warn!("Clip {:?} wasn't loaded.", clip_id);
            continue;
        };

        let &mut (ref mut graph, ref mut clip_indices) =
            player_to_graph.entry(ancestor_player).or_default();
        let node_index = graph.add_clip(clip_handle, 1.0, graph.root);
        clip_indices.push(node_index);
    }

    for (player_entity, (graph, clips)) in player_to_graph {
        let Ok(mut player) = players.get_mut(player_entity) else {
            warn!("Animation targets referenced a nonexistent player. This shouldn't happen.");
            continue;
        };
        let graph = graphs.add(graph);
        player.play(clips[0]).repeat();
        commands
            .entity(player_entity)
            .insert(AnimationGraphHandle(graph));
    }
}

fn setup_scene_after_load(
    mut commands: Commands,
    mut setup: Local<bool>,
    mut scene_handle: ResMut<SceneHandle>,
    meshes: Query<(&GlobalTransform, Option<&Aabb>), With<Mesh3d>>,
    mut scene_spawner: ResMut<WorldInstanceSpawner>,
    gltf_assets: Res<Assets<Gltf>>,
    settings: Res<BakeSettings>,
) {
    if scene_handle.is_loaded && !*setup {
        *setup = true;
        // Find an approximate bounding box of the scene from its meshes
        if meshes.iter().any(|(_, maybe_aabb)| maybe_aabb.is_none()) {
            return;
        }

        let mut points = Vec::default();
        for entity in scene_spawner.iter_instance_entities(scene_handle.instance_id.unwrap()) {
            let Ok((transform, maybe_aabb)) = meshes.get(entity) else {
                continue;
            };
            println!("loaded mesh entity: {entity:?}");

            let aabb = maybe_aabb.unwrap();
            let corners = [
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(-1.0, -1.0, 1.0),
                Vec3::new(-1.0, 1.0, -1.0),
                Vec3::new(-1.0, 1.0, 1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(1.0, -1.0, 1.0),
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(1.0, 1.0, 1.0),
            ];
            points.extend(corners.iter().map(|c| {
                transform
                    .transform_point(Vec3::from(aabb.center) + (Vec3::from(aabb.half_extents) * *c))
            }));
        }

        let aabb = Aabb::enclosing(&points).unwrap();
        let base_radius = points
            .iter()
            .map(|p| FloatOrd((*p - Vec3::from(aabb.center)).length()))
            .max()
            .unwrap()
            .0;

        // When baking a cluster, place the extra copies evenly across a disc around
        // the original (which stays at the centre) and grow the captured sphere so the
        // whole group ends up inside the single baked imposter. The spread scales
        // with sqrt(count) so density stays roughly constant as the count grows.
        // `--cluster-spread` scales this. The sqrt(count) law keeps the trees-per-area constant as
        // the count grows, but the constant it holds it AT is derived from `base_radius` - the
        // bounding SPHERE radius, which for a tall thin model like a tree is dominated by its
        // HEIGHT, not by how wide its canopy is. So the default leaves visible ground gaps between
        // canopies. Shrinking the spread is the direct fix, and it pays twice: `radius` below
        // shrinks with it, so the same `tile_size` covers a smaller world area and every tree gets
        // more texels. Raising `--cluster` at a fixed spread is the same knob from the other end -
        // `--cluster 37 --cluster-spread 0.72` packs 37 trees into the footprint 19 had.
        let spread = if settings.cluster > 1 {
            base_radius * (settings.cluster as f32).sqrt() * 0.5 * settings.cluster_spread
        } else {
            0.0
        };
        // 1.15 leaves headroom for the per-copy placement jitter below.
        let radius = base_radius + spread * 1.15;
        let size = radius * 2.0;
        let sphere = Sphere {
            center: aabb.center,
            radius,
        };

        info!("sphere: {:?}", sphere);
        scene_handle.sphere = sphere;
        // The un-inflated radius, kept for the spawn path: `radius` above has absorbed the cluster's
        // HORIZONTAL spread, so it is the only remaining handle on how big one tree actually is.
        scene_handle.base_radius = base_radius;

        // Record every copy's placement (index 0 is the original at the centre, identity
        // rotation) so `--swap` can rebuild the exact same cluster as real models. The
        // placement uses random jitter/rotation, so it has to be captured here rather than
        // recomputed at swap time.
        let mut cluster_layout: Vec<(Vec3, Quat)> = vec![(Vec3::ZERO, Quat::IDENTITY)];
        // Built here (not inside the loop) so the alignment printout further down can report the
        // nominal, unjittered row coordinates - the thing `--cross-sides` has to be lined up with.
        let placement = ClusterPlacement::new(&settings, spread);
        if settings.cluster > 1 {
            let gltf = gltf_assets.get(&scene_handle.gltf_handle).unwrap();
            let gltf_scene_handle = gltf.scenes.get(scene_handle.scene_index).unwrap().clone();
            // SEEDED, like the imposter placement in `impost` and for a stronger version of the
            // same reason. This rng decides where the `--cluster` copies stand and how each one is
            // yawed, which is what gets BAKED INTO THE ATLAS - so with an unseeded `thread_rng` two
            // launches of the identical command line produce two different pictures on every quad,
            // and a screenshot diff between them lands at ~37% before any code change is involved.
            // That silently invalidates every A/B this example exists to run as soon as `--cluster`
            // is in play. `--seed` already promises "two runs render the identical scene"; this is
            // the other half of keeping that promise.
            //
            // Offset by one from the placement seed so the cluster jitter and the first few tree
            // positions are not literally the same draws - same guarantee, no shared structure.
            let mut rng = StdRng::seed_from_u64(settings.seed.wrapping_add(1));
            info!(
                "placing {} source copies (spread {spread:.2}, {})",
                settings.cluster,
                placement.describe(settings.cluster)
            );
            for i in 1..settings.cluster {
                // Every branch of `offset` draws exactly two rng values, in x-then-z order, BEFORE
                // the yaw draw below. Keeping the draw pattern identical is what leaves the
                // (unseeded, but otherwise untouched) Vogel path behaving exactly as it did.
                let offset = placement.offset(i, settings.cluster, spread, &mut rng);
                let rotation = Quat::from_rotation_y(rng.gen_range(0.0..=(PI * 2.0)));
                cluster_layout.push((offset, rotation));
                let root = commands
                    .spawn((
                        Transform::from_translation(offset).with_rotation(rotation),
                        Visibility::default(),
                    ))
                    .id();
                scene_spawner.spawn_as_child(gltf_scene_handle.clone(), root);
            }
        }
        commands.insert_resource(ClusterLayout(cluster_layout));

        if settings.cross {
            // `base_radius` as well as the sphere: the outer planes' recommended WIDTH is the
            // outermost row's span plus one tree's own half-width, and the inflated radius has
            // already lost track of how big one tree is.
            print_cross_alignment(&settings, &sphere, base_radius, &placement);
        }

        info!("Spawning a controllable 3D perspective camera");
        let mut projection = PerspectiveProjection::default();
        projection.far = projection.far.max(size * 10.0);
        // let projection = OrthographicProjection {
        //     scaling_mode: bevy::render::camera::ScalingMode::FixedVertical(10.0),
        //     ..Default::default()
        // };

        let walk_speed = size * 3.0;
        let camera_controller = CameraController {
            walk_speed,
            run_speed: 3.0 * walk_speed,
            ..default()
        };

        // Display the controls of the scene viewer
        info!("{}", camera_controller);
        info!("{:?}", *scene_handle);

        let camera = commands
            .spawn((
                Camera3d::default(),
                Projection::from(projection),
                Transform::from_translation(
                    Vec3::from(aabb.center) + size * Vec3::new(0.5, 0.25, 0.5),
                )
                .looking_at(Vec3::from(aabb.center), Vec3::Y),
                camera_controller,
                // ambient light is now a per-camera component (was a global resource pre-0.16)
                AmbientLight {
                    color: Color::WHITE,
                    brightness: settings.ambient,
                    affects_lightmapped_meshes: true,
                },
                // layers 0+1 normally: 0 is the source model, 1 the imposters. --hide-source drops
                // layer 0 so ONLY imposters are drawn - see `BakeSettings::hide_source`.
                if settings.hide_source {
                    RenderLayers::layer(1)
                } else {
                    RenderLayers::default().with(1)
                },
            ))
            .id();

        if settings.fxaa {
            info!("Enabling FXAA");
            commands.entity(camera).insert(Fxaa::default());
        }

        // See `BakeSettings::prepass`: this is what specializes the imposter shaders'
        // PREPASS_FRAGMENT entry point, which nothing else in this example reaches.
        if settings.prepass {
            info!("Enabling depth + normal prepass");
            commands
                .entity(camera)
                .insert((DepthPrepass, NormalPrepass));
        }

        // Spawn a default light if the scene does not have one
        if !scene_handle.has_light {
            info!("Spawning a directional light");
            commands.spawn((
                DirectionalLight {
                    shadow_maps_enabled: settings.shadows,
                    ..default()
                },
                Transform::from_xyz(1.0, 1.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
                RenderLayers::default().with(1),
            ));

            scene_handle.has_light = true;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn impost(
    mut commands: Commands,
    k: Res<ButtonInput<KeyCode>>,
    scene_handle: Res<SceneHandle>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<Imposter>>,
    mut cross_materials: ResMut<Assets<CrossImposter>>,
    cams: Query<Entity, With<ImposterBakeCamera>>,
    imposters: Query<Entity, With<MeshMaterial3d<Imposter>>>,
    // separate query, not an `Or<>`: the two materials are unrelated types, so there is no single
    // component to filter on. Both are read-only `Entity` queries, so they can't conflict.
    cross_imposters: Query<Entity, With<MeshMaterial3d<CrossImposter>>>,
    reals: Query<Entity, With<RealModel>>,
    settings: Res<BakeSettings>,
    dummy_indices: Res<DummyIndicesImage>,
    mut probe: ResMut<BakeProbe>,
    // guards against spawning a second batch of imposters while one is already live; reset by `O`.
    mut spawned: Local<bool>,
    // `--auto-bake` fires exactly once per process, even after an `O` clear.
    mut auto_fired: Local<bool>,
) {
    if k.just_pressed(KeyCode::KeyO) {
        for entity in cams.iter() {
            commands.entity(entity).despawn();
        }
        let mut cleared = 0;
        for entity in imposters.iter().chain(cross_imposters.iter()) {
            commands.entity(entity).despawn();
            cleared += 1;
        }
        // ...and cancel a `--cross-trim` spawn that is still waiting on the bake readback, or it
        // would quietly repopulate the forest a few frames after O emptied it.
        commands.remove_resource::<PendingCrossSpawn>();
        // drop any real models that `--swap` substituted in for nearby imposters
        for entity in reals.iter() {
            commands.entity(entity).despawn();
        }
        if *spawned || cleared > 0 {
            println!("cleared {cleared} imposters and stopped baking");
        }
        *spawned = false;
        probe.inflight = None;
        return;
    }

    // SPACE => bake a single time then leave the imposters static; I => keep re-baking every frame
    // (useful when the source model animates or moves). Either way we only spawn imposters once
    // until they're cleared with O.
    // `--auto-bake` takes exactly the SPACE branch - same `continuous = false`, same body below -
    // so an unattended run measures the identical code path an interactive one does. The gate is
    // `sphere.radius > 0.0`, which is the only externally visible signal that
    // `setup_scene_after_load` has run: it computes the bounding sphere (and spawns the camera and
    // light) a frame after `is_loaded` flips, and baking before that would bake a zero-radius
    // sphere. `Sphere::default()` has radius 0, so the test cannot false-positive.
    let auto_now = settings.auto_bake
        && !*auto_fired
        && scene_handle.is_loaded
        && scene_handle.sphere.radius > 0.0;

    let continuous = if k.just_pressed(KeyCode::Space) {
        false
    } else if k.just_pressed(KeyCode::KeyI) {
        true
    } else if auto_now {
        *auto_fired = true;
        println!("--auto-bake: triggering one-shot bake + spawn");
        false
    } else {
        return;
    };

    if *spawned {
        println!("imposters already spawned - press O to clear them first");
        return;
    }
    *spawned = true;

    {
        println!(
            "baking imposter ({})",
            if continuous { "every frame" } else { "once" }
        );
        let mut camera = if settings.cross {
            // The cross atlas is not a grid of octahedral directions but two hand-placed views
            // (side, top) fed in through `manual_camera_transforms`, so the camera has to come from
            // the helper rather than a literal - the mesh uvs in `cross_imposter_mesh` are derived
            // from exactly these two transforms, and any drift between them shows up as a mirrored
            // or transposed tile. `manual_camera_transforms` are absolute world transforms, so the
            // helper needs the world position we spawn the camera entity at (below), not the
            // instance-local origin the material's `center_and_scale` uses.
            let mut camera = cross_bake_camera(
                scene_handle.sphere.center.into(),
                scene_handle.sphere.radius,
                settings.tile_size,
                settings.multisample_source,
            );
            camera.continuous = continuous;
            camera.max_tiles_per_frame = settings.bake_tiles_per_frame;
            camera
        } else {
            ImposterBakeCamera {
                radius: scene_handle.sphere.radius,
                grid_size: settings.grid_size,
                tile_size: settings.tile_size,
                grid_mode: settings.mode,
                continuous,
                multisample: settings.multisample_source,
                max_tiles_per_frame: settings.bake_tiles_per_frame,
                ..Default::default()
            }
        };
        camera.init_target(&mut images);

        // --cross-trim: the ONLY moment the readback image exists is inside this callback (`bake.rs`
        // hands it over and drops it), so the trim has to be measured here - but the mesh it feeds
        // is built below, synchronously, several frames earlier. The measurement is therefore
        // dropped into a shared slot that `spawn_trimmed_cross_imposters` polls, and the spawn is
        // deferred until it lands. `None` when the flag is off, which is what keeps the untrimmed
        // path byte-for-byte the one that always ran.
        let trim_slot = (settings.cross && settings.cross_trim)
            .then(|| Arc::new(Mutex::new(None::<CrossTrim>)));

        // --dump-atlas: decode the raw Rg32Uint readback (8 bytes/texel of packed g-buffer) into an
        // sRGB PNG so the baked tiles can actually be looked at. Verbatim decode from planet.rs's
        // --dump-atlas; the low u32 holds 5 bits each of RGBA albedo (see shaders/shared.wgsl
        // pack_props). For --cross this writes the full 2x2 bake target: the TOP ROW is the two
        // tiles the material samples (0,0 = side, 1,0 = top), the bottom row is the by-design
        // duplicate of the same two views.
        //
        // There is exactly ONE callback slot and two features that want it, so they are COMPOSED
        // into a single closure. `set_callback` overwrites, so calling it twice would silently drop
        // whichever feature registered first.
        let dump_path = settings.dump_atlas.clone();
        if dump_path.is_some() || trim_slot.is_some() {
            let trim_slot = trim_slot.clone();
            let (tile_size, cutoff, pad) = (
                settings.tile_size,
                settings.cross_trim_cutoff,
                settings.cross_trim_pad,
            );
            camera.set_callback(move |image| {
                if let Some(slot) = trim_slot.as_ref() {
                    *slot.lock().unwrap() =
                        Some(cross_trim_from_atlas(&image, tile_size, cutoff, pad));
                }
                let Some(dump_path) = dump_path.as_ref() else {
                    return;
                };
                let (w, h) = (image.width(), image.height());
                let Some(data) = image.data.as_ref() else {
                    error!("baked atlas has no readback data");
                    return;
                };
                let mut rgba = Vec::with_capacity((w * h * 4) as usize);
                for texel in data.chunks_exact(8) {
                    let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                    let chan = |off: u32| ((packed >> off) & 0x1f) as f32 / 31.0;
                    let srgb = |l: f32| {
                        let s = if l <= 0.0031308 {
                            l * 12.92
                        } else {
                            1.055 * l.powf(1.0 / 2.4) - 0.055
                        };
                        (s.clamp(0.0, 1.0) * 255.0).round() as u8
                    };
                    rgba.push(srgb(chan(0)));
                    rgba.push(srgb(chan(5)));
                    rgba.push(srgb(chan(10)));
                    rgba.push((chan(15) * 255.0).round() as u8);
                }
                match image::RgbaImage::from_raw(w, h, rgba) {
                    Some(img) => match img.save(dump_path) {
                        Ok(()) => println!("wrote {w}x{h} atlas albedo to {dump_path}"),
                        Err(e) => error!("failed to write atlas to {dump_path}: {e}"),
                    },
                    None => error!("atlas buffer/size mismatch, not writing PNG"),
                }
            });
        }

        // Seeded (see `BakeSettings::seed`): identical placement in both modes is what makes the
        // cross-vs-octahedral frame times comparable at all.
        let mut rng = StdRng::seed_from_u64(settings.seed);
        let range = scene_handle.sphere.radius * (settings.count as f32).sqrt() * settings.spacing;
        let range = -range..=range;
        let offset = Vec3::X * 0.5;
        let rotate_range = 0.0..=(PI * 2.0);
        println!("spawning {} imposters", settings.count);
        let hemi_mult = if settings.mode != GridMode::Spherical {
            0.0
        } else {
            1.0
        };

        // All imposters here are identical (same atlas, same ImposterData; only the
        // per-entity Transform differs), so we share a single mesh handle and a single
        // material handle. Sharing the asset ids lets bevy batch/instance the draws into
        // (ideally) a single instanced draw call instead of one draw call per imposter.
        let alpha_mode = if settings.a2c {
            // MSAA converts the fragment alpha into a sub-pixel coverage mask, so the
            // alpha-tested tree silhouette gets anti-aliased without a temporal pass.
            // renders opaque (depth writes, early-Z, no sorting) like Mask.
            AlphaMode::AlphaToCoverage
        } else if settings.mask {
            AlphaMode::Mask(0.5)
        } else if settings.cross {
            // Blend is the default for the octahedral path (one quad, sorted, no depth writes) but
            // is pathological for the cross: three overlapping quads per instance means every
            // instance is 3x overdraw of a sorted, depth-write-less, un-early-Z'd fragment, and the
            // three quads of a single instance cannot even be sorted against each other correctly.
            // The resulting number would say nothing about the fragment cost we are here to
            // measure, so cross mode defaults to Mask (opaque pass, depth writes, early-Z).
            info!("--cross defaults to AlphaMode::Mask(0.5); pass --a2c for antialiased silhouettes, or --mask to make this explicit");
            AlphaMode::Mask(0.5)
        } else {
            AlphaMode::Blend
        };

        // Placement is generated once, up front, and consumed by both paths: identical transforms
        // (and an identical rng draw order) are what make the two modes an A/B of the render path
        // alone rather than of two differently-arranged scenes.
        let transforms: Vec<Transform> = (0..settings.count)
            .map(|_| {
                let translation = Vec3::new(
                    rng.gen_range(range.clone()),
                    rng.gen_range(range.clone()) * hemi_mult,
                    rng.gen_range(range.clone()),
                ) + offset;
                let rotation = Vec3::new(
                    rng.gen_range(rotate_range.clone()) * hemi_mult,
                    rng.gen_range(rotate_range.clone()),
                    rng.gen_range(rotate_range.clone()) * hemi_mult,
                );
                Transform::from_translation(translation + Vec3::from(scene_handle.sphere.center))
                    .with_rotation(Quat::from_euler(
                        EulerRot::XYZ,
                        rotation.x,
                        rotation.y,
                        rotation.z,
                    ))
            })
            .collect();

        if settings.cross {
            let spawn = CrossSpawn {
                target: camera.target.clone().unwrap(),
                indices: dummy_indices.0.clone(),
                radius: scene_handle.sphere.radius,
                base_radius: scene_handle.base_radius,
                alpha_mode,
                transforms,
            };
            if settings.cross_trim {
                // Wait for the bake. The mesh depends on the ATLAS now, and the atlas does not
                // exist yet - `init_target` allocated it, but nothing has rendered into it and the
                // readback is several frames out. Park everything the spawn needs in a resource and
                // let `spawn_trimmed_cross_imposters` finish the job when the measurement lands.
                //
                // Deliberately not "spawn now, rebuild the mesh later": that would put a frame or
                // two of untrimmed forest into the capture and make the 0.00%-diff check a race.
                commands.insert_resource(PendingCrossSpawn {
                    trim: trim_slot.expect("--cross-trim installs the slot above"),
                    spawn: Some(spawn),
                });
            } else {
                // The unchanged path: same call, `FULL` rects, which `cross_quads` turns back into
                // the exact +-0.5 quads and 0..1 uvs it always emitted.
                spawn_cross_imposters(
                    &mut commands,
                    &mut meshes,
                    &mut cross_materials,
                    &settings,
                    spawn,
                    CrossTrim::FULL,
                );
            }
        } else {
            let shared_mesh = Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(0.5))));
            let mut data = ImposterData::new(
                Vec3::ZERO,
                scene_handle.sphere.radius,
                settings.grid_size,
                settings.tile_size,
                UVec2::ZERO,
                UVec2::splat(settings.tile_size),
                settings.mode,
                settings.multisample_target,
                false,
                settings.dither,
                settings.coverage,
                settings.fade,
                1.0,
            );
            // --swap-fade: dither the imposter out over [solid, spawn] world distance so the real
            // model swapped in behind it shows through. The imposter billboard renders at the front
            // of the bounding sphere, so fading the imposter (rather than the model) is what lets
            // the model appear through the dither holes. Matches `swap_close`'s spawn/solid
            // thresholds.
            if settings.swap && settings.swap_fade {
                let spawn = settings.swap_distance * scene_handle.sphere.radius;
                let solid = spawn / settings.swap_fade_band;
                data.swap_fade = Vec2::new(solid, spawn);
            }
            // --debug-level: same clamp the 0-4 keys apply (the octahedral ladder tops out at 4).
            data.debug_level = settings.debug_level.min(4);
            let shared_material = MeshMaterial3d(materials.add(Imposter {
                data,
                pixels: camera.target.clone().unwrap(),
                indices: dummy_indices.0.clone(),
                alpha_mode,
                vram_bytes: 0,
                // Runtime-baked here, and the bake writes a single level: mips only ever come from
                // `ImposterLoaderSettings::generate_mips` on the ASSET path, and the octahedral
                // shader has no render-time LOD to spend them on regardless.
                mip_levels: 1,
            }));

            for transform in transforms {
                commands.spawn((
                    shared_mesh.clone(),
                    transform,
                    shared_material.clone(),
                    RenderLayers::layer(1),
                    SwapReal::default(),
                ));
            }
        }

        commands.spawn((
            camera,
            Transform::from_translation(scene_handle.sphere.center.into()),
        ));

        // start the bake-cost probe for one-shot bakes (continuous re-bakes every frame, so there
        // is no single completion to time - watch the per-frame GPU diagnostics for those).
        if !continuous {
            probe.inflight = Some(BakeProbeState {
                start: Instant::now(),
                frames: 0,
            });
        }
    }
}

/// Everything `spawn_cross_imposters` needs that `impost` computed and that would otherwise be
/// unreachable from a later frame: the bake target's handle, the placement list, the bounding-sphere
/// radii.
///
/// Bundled into a struct rather than passed as six arguments because it has to SURVIVE in a
/// resource across the wait for the trim, and a struct is the only shape that can do both jobs.
struct CrossSpawn {
    /// The bake camera's target image - the atlas the material samples. Cloned out of the camera
    /// before it is spawned, since the camera entity owns it from then on.
    target: Handle<Image>,
    /// `DummyIndicesImage`: the atlas is never indexed, but bevy hands out an rgba8 fallback for an
    /// unbound `u_int` texture, which fails bind-group validation.
    indices: Handle<Image>,
    /// Bounding-sphere radius of the bake, `--cluster` inflation included.
    radius: f32,
    /// The single model's radius, before that inflation - `--cross-height`'s correction factor.
    base_radius: f32,
    alpha_mode: AlphaMode,
    /// The seeded placement list, generated ONCE in `impost` so that trimmed, untrimmed, merged and
    /// octahedral runs at the same seed all render the identical forest.
    transforms: Vec<Transform>,
}

/// `--cross-trim`'s wait state: the spawn, held until the bake's readback callback drops a measured
/// [`CrossTrim`] into the shared slot.
///
/// An `Arc<Mutex<Option<_>>>` and not a channel because the producer is a `FnOnce` handed to
/// `ImposterBakeCamera::set_callback` (so it must be `Send + Sync + 'static` and cannot borrow the
/// world) and the consumer is an ordinary system polling once a frame. One slot, last write wins,
/// which is also the right semantics for a `--cross-trim` + continuous (`I`) bake.
#[derive(Resource)]
struct PendingCrossSpawn {
    trim: Arc<Mutex<Option<CrossTrim>>>,
    /// `Option` so the polling system can move it out of the `ResMut` before dropping the resource.
    spawn: Option<CrossSpawn>,
}

/// The measured trim, kept for the `===BOIMP SUMMARY===` block. Absent without `--cross-trim`, which
/// is what keeps the summary of an untrimmed run byte-identical to the one it printed before this
/// feature existed.
#[derive(Resource)]
struct CrossTrimReport(CrossTrim);

/// Polls for `--cross-trim`'s measurement and finishes the spawn `impost` parked.
///
/// Exists as a separate system purely because `impost` is at bevy's 16-`SystemParam` limit; there is
/// nothing here that could not have lived there. It is a no-op - one `Option<Res<_>>` check - in
/// every run that does not pass `--cross-trim`.
fn spawn_trimmed_cross_imposters(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut cross_materials: ResMut<Assets<CrossImposter>>,
    settings: Res<BakeSettings>,
    pending: Option<ResMut<PendingCrossSpawn>>,
) {
    let Some(mut pending) = pending else {
        return;
    };
    let Some(trim) = pending.trim.lock().unwrap().take() else {
        // bake still in flight
        return;
    };
    let Some(spawn) = pending.spawn.take() else {
        return;
    };
    commands.remove_resource::<PendingCrossSpawn>();

    // The number the feature exists for, so print it in the summary's key: value voice rather than
    // burying it in a log line: the quad area fraction IS the fill-rate saving, and a rect that
    // covers 90% of its tile means the bake, not the trim, is what wants looking at.
    let report = |name: &str, rect: Rect| {
        println!(
            "cross_trim_{name}: uv ({:.4}, {:.4})..({:.4}, {:.4})   quad area 100% -> {:.1}%",
            rect.min.x,
            rect.min.y,
            rect.max.x,
            rect.max.y,
            rect.width() * rect.height() * 100.0
        );
    };
    println!(
        "--cross-trim: measured from the baked atlas (cutoff {}, pad {} texels)",
        settings.cross_trim_cutoff, settings.cross_trim_pad
    );
    report("side", trim.side);
    report("top", trim.top);

    commands.insert_resource(CrossTrimReport(trim));
    spawn_cross_imposters(
        &mut commands,
        &mut meshes,
        &mut cross_materials,
        &settings,
        spawn,
        trim,
    );
}

/// The `--cross` half of `impost`'s spawn, lifted out of it so `--cross-trim` can run the SAME
/// code a few frames later without a second copy of it.
///
/// `--cross-trim` needs [`CrossTrim`] measured off the baked atlas, and that measurement only
/// arrives in the bake's readback callback, several frames after `impost` has already built the
/// mesh and spawned the trees. Rather than duplicate ~150 lines into a deferred path - where the
/// copy would quietly drift and the "trim off is unchanged" guarantee would stop meaning anything -
/// `impost` calls this immediately with [`CrossTrim::FULL`] and the deferred path
/// (`spawn_trimmed_cross_imposters`) calls it with the measured rects. The `trim` argument is the
/// ONLY difference between the two.
///
/// Takes `&mut` handles rather than `SystemParam`s because `impost` is already at bevy's
/// 16-parameter limit, so the deferred caller has to be a separate system with its own params.
fn spawn_cross_imposters(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    cross_materials: &mut Assets<CrossImposter>,
    settings: &BakeSettings,
    spawn: CrossSpawn,
    trim: CrossTrim,
) {
    let CrossSpawn {
        target,
        indices,
        radius,
        base_radius,
        alpha_mode,
        transforms,
    } = spawn;
    // Same sharing rule as the octahedral path in `impost` and for the same reason: one mesh
    // handle and one material handle across every instance, so bevy batches them into a
    // single instanced draw. It matters more here - the cross is 6 triangles to the
    // billboard's 2, so a per-instance draw call would swamp the fragment saving we are
    // trying to measure.
    let merge = settings.cross_merge.max(1);
    // `--cross-height` is a UNIT-space y that the vertex shader scales by `2 * radius`, so
    // the canopy quad ends up `h * 2 * radius` above the sphere centre. With `--cluster n`
    // that radius has been inflated by the group's HORIZONTAL spread (see
    // `setup_scene_after_load`: `base_radius + base_radius * sqrt(n) * 0.5 * 1.15`, i.e.
    // 3.3x at n = 16) while the trees are exactly as tall as they were - so a height tuned
    // on a single tree floats the canopy 3.3x too high, which is the "canopy detached from
    // the forest" artifact. Re-express the knob against the TREES' own vertical extent by
    // dividing the inflation back out, so `--cross-height 0.1125` means the same thing at
    // every `--cluster`. At `--cluster 1` the ratio is exactly 1.0 (spread is 0.0, so
    // `radius == base_radius` bit for bit) and this is a numeric no-op.
    //
    // Only the HEIGHT is corrected. `--cross-sides` is an offset in the horizontal plane,
    // which is the same plane the inflation happened in, so it is already in the right
    // units - see the note on it in `print_cross_grid_alignment`.
    let height_scale = if radius > 0.0 {
        base_radius / radius
    } else {
        1.0
    };
    let cross_height = settings.cross_height * height_scale;
    println!(
        "cross canopy height: --cross-height {} -> effective {:.3} (base_radius {:.1} / \
         radius {:.1})",
        settings.cross_height, cross_height, base_radius, radius
    );
    // One layout value for the mesh, the merged patches and the quad-count report - see
    // `cross_layout`. The height is corrected above and the trim comes from the caller (FULL unless
    // `--cross-trim` measured it); the rest is verbatim from the CLI.
    let layout = cross_layout(settings, cross_height, trim);
    let shared_mesh = Mesh3d(meshes.add(cross_imposter_mesh(&layout)));
    let mut material = cross_imposter_from_bake_target(
        target,
        indices,
        // instance-local: the per-entity Transform carries `sphere.center`, exactly as the
        // octahedral path in `impost` passes `Vec3::ZERO` to `ImposterData::new`.
        Vec3::ZERO,
        radius,
        settings.tile_size,
        alpha_mode,
    );
    // The helper deliberately returns flags = 0 (the clean phase-0 perf baseline); the
    // quality features are opt-in per run so each one can be priced against it.
    if settings.cross_angle_fade {
        material.data.flags |= CROSS_ANGLE_FADE_FLAG;
        // Overrides the (0.05, 0.30) default baked into `CrossImposterData::new`, so the
        // band can be swept - or pushed to an absurd (0.9, 1.0) - without a recompile.
        material.data.angle_fade = Vec2::new(settings.cross_fade_start, settings.cross_fade_end);
        if settings.cross_angle_shrink {
            material.data.flags |= CROSS_ANGLE_SHRINK_FLAG;
        }
        info!(
            "cross angle fade ON, band ({}, {}), spent as {}",
            settings.cross_fade_start,
            settings.cross_fade_end,
            if settings.cross_angle_shrink {
                "vertex shrink"
            } else {
                "fragment dither"
            }
        );
    } else if settings.cross_angle_shrink {
        // The flag is gated on ANGLE_FADE in `specialize`, so this combination is silently inert.
        // Say so rather than letting a run be attributed to a feature that never ran.
        warn!("--cross-angle-shrink ignored: it needs --cross-angle-fade to have a weight to spend");
    }
    if settings.cross_two_sided {
        material.data.flags |= CROSS_TWO_SIDED_FLAG;
    }
    if settings.cross_cheap_lighting {
        material.data.flags |= CROSS_CHEAP_LIGHTING_FLAG;
    }
    // --debug-level: same clamp the 0-4 keys apply (the cross ladder tops out at 2).
    material.data.debug_level = settings.debug_level.min(2);
    if merge > 1 {
        // Rewrites center_and_scale to (0,0,0,0.5) AND sets CROSS_MERGED_FLAG. Merged
        // meshes carry FINAL positions, so the shader's `position * (scale*2) + centre` has
        // to become the identity; doing this by hand and getting the 0.5 wrong renders the
        // whole forest scaled by 2*radius.
        material.data.set_merged();
    }
    let shared_material = MeshMaterial3d(cross_materials.add(material));
    // 3 for the default cross, 7 for either 7-quad layout. Reported (rather than assumed)
    // because every extra vertical plane is straight VERTICAL overdraw and the whole system
    // exists to be fill-cheap.
    let quads_per_instance = cross_quad_count(&layout);

    if merge > 1 {
        // Chunk the EXISTING placement transforms rather than generating new ones: the
        // whole point of the seeded layout is that a merged run and an unmerged run at the
        // same seed render the identical forest, so the two are a clean A/B of the batching
        // and nothing else.
        let mut warned_non_yaw = false;
        let mut patches = 0usize;
        let mut verts = 0usize;
        // ...but chunk them in SPATIAL order, not in the order the rng produced them.
        // Bevy frustum-culls per entity, so a patch is all-or-nothing: 64 randomly
        // scattered trees give a patch Aabb that spans the entire forest, every patch
        // overlaps the frustum, and the merged run draws all 1000 trees where the unmerged
        // run drew only the ~quarter inside the view. That is a measured regression, not a
        // theoretical one - merging a randomly-ordered list cost ~2 ms/frame here even
        // though it removed 984 draws. A Morton (Z-order) key over the ground plane makes
        // each chunk a compact cell and gives culling something to reject again.
        //
        // This reorders WHICH tree lands in WHICH patch; it does not touch any tree's
        // transform, so the merged/unmerged screenshot A/B is unaffected.
        let sorted = morton_sorted(&transforms);
        for chunk in sorted.chunks(merge) {
            // Centroid, not the first tree and not the origin: the patch entity's Transform
            // is the frame the merged positions are relative to, and a centroid keeps the
            // offsets small (float precision) and the auto-computed Aabb tight (frustum
            // culling still rejects whole patches).
            let centroid = chunk.iter().map(|t| t.translation).sum::<Vec3>() / chunk.len() as f32;
            let instances: Vec<CrossInstance> = chunk
                .iter()
                .map(|t| {
                    // YXZ so the FIRST component is the yaw. In hemispherical/horizontal
                    // mode the generated rotation is pure yaw and the other two are exactly
                    // zero; in spherical mode they are not, and a cross imposter has no way
                    // to represent a tilted tree (the merged mesh only bakes a +Y rotation),
                    // so say so instead of silently rendering an upright tree where a tilted
                    // one was asked for.
                    let (yaw, pitch, roll) = t.rotation.to_euler(EulerRot::YXZ);
                    if !warned_non_yaw && (pitch.abs() > 1e-4 || roll.abs() > 1e-4) {
                        warned_non_yaw = true;
                        warn!(
                            "--cross-merge only bakes the YAW of each placement (pitch/roll \
                             of {pitch:.3}/{roll:.3} rad dropped); use --mode h or H for a \
                             merged run that matches the unmerged one exactly"
                        );
                    }
                    CrossInstance {
                        translation: t.translation - centroid,
                        // The bounding-sphere RADIUS, i.e. what center_and_scale.w used to
                        // carry - the builder applies the *2.0 itself.
                        scale: radius,
                        yaw,
                        // false, not random: `--cross-merge` has to be verifiable against
                        // the unmerged run pixel for pixel, and mirroring would change every
                        // tree's silhouette. Flip it per tree in a real scene for free
                        // variation.
                        mirror: false,
                    }
                })
                .collect();
            verts += instances.len() * quads_per_instance * 4;
            let mesh = meshes.add(cross_imposter_merged_mesh(&layout, &instances));
            commands.spawn((
                Mesh3d(mesh),
                Transform::from_translation(centroid),
                shared_material.clone(),
                RenderLayers::layer(1),
            ));
            patches += 1;
        }
        println!(
            "--cross-merge {merge}: {} trees -> {patches} patches ({} trees/patch, \
             {quads_per_instance} quads/tree, {verts} verts, {patches} draws instead of {})",
            transforms.len(),
            merge,
            transforms.len()
        );
    } else {
        for transform in transforms {
            // no `SwapReal` here - see the `--cross`/`--swap` warning in `setup`.
            commands.spawn((
                shared_mesh.clone(),
                transform,
                shared_material.clone(),
                RenderLayers::layer(1),
            ));
        }
    }
}

/// How `--cluster`'s extra copies are laid out around the original, which stays at the origin.
///
/// One enum rather than a pile of `Option`s because the three layouts are alternatives, and because
/// they must all draw the SAME number of rng values in the same order (two, x then z, for the
/// jitter) - keeping them in one `offset` match is what makes that checkable at a glance instead of
/// being a comment three call sites away.
enum ClusterPlacement {
    /// The default: a Vogel/sunflower disc. Even coverage, no rows, and therefore nothing for
    /// `--cross-sides`' planes to line up with.
    Vogel,
    Grid(ClusterGrid),
    Hex(ClusterHex),
}

impl ClusterPlacement {
    fn new(settings: &BakeSettings, spread: f32) -> Self {
        // `cluster <= 1` means there are no copies to place at all; both lattices divide by a ring
        // count/side that is only non-zero once there is a second copy, so keep them out entirely.
        if settings.cluster <= 1 {
            return Self::Vogel;
        }
        if settings.cluster_hex {
            Self::Hex(ClusterHex::new(settings.cluster, spread))
        } else if settings.cluster_grid {
            Self::Grid(ClusterGrid::new(settings.cluster, spread))
        } else {
            Self::Vogel
        }
    }

    fn describe(&self, cluster: usize) -> String {
        match self {
            Self::Vogel => "vogel spiral".to_string(),
            Self::Grid(grid) => format!("{0}x{0} grid, pitch {1:.2}", grid.side, grid.pitch),
            Self::Hex(hex) => format!(
                "hex lattice, {} rings, pitch {:.2}, row spacing {:.2}{}",
                hex.rings,
                hex.pitch,
                hex.row_spacing(),
                if hex.full == cluster {
                    String::new()
                } else {
                    // Say it out loud: a partial outer ring means the outermost ROWS are only
                    // partly populated, which is exactly the thing `--cross-sides` is aligned to.
                    format!(
                        " - PARTIAL outer ring ({} of {} cells; the clean sizes are the centered \
                         hexagonal numbers 1, 7, 19, 37, 61)",
                        cluster - (hex.full - hex.ring_cells()),
                        hex.ring_cells()
                    )
                }
            ),
        }
    }

    /// Placement for cluster copy `copy_index` (1-based; 0 is the original at the origin). Exactly
    /// two rng draws on every path, x then z.
    fn offset(&self, copy_index: usize, cluster: usize, spread: f32, rng: &mut impl Rng) -> Vec3 {
        match self {
            // Vogel/sunflower spiral: evenly distribute the copies across the disc using the golden
            // angle, with a touch of jitter so it doesn't look mechanical. Radius grows as sqrt(i)
            // for uniform area density.
            Self::Vogel => {
                let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
                let jitter = spread * 0.1;
                let fi = copy_index as f32;
                let r = spread * (fi / (cluster - 1) as f32).sqrt();
                let angle = fi * golden_angle;
                Vec3::new(
                    r * angle.cos() + rng.gen_range(-jitter..=jitter),
                    0.0,
                    r * angle.sin() + rng.gen_range(-jitter..=jitter),
                )
            }
            Self::Grid(grid) => grid.offset(copy_index, rng),
            Self::Hex(hex) => hex.offset(copy_index, rng),
        }
    }
}

/// `--cluster-hex`: a triangular lattice of cluster-copy slots filling a HEXAGON centred on the
/// origin, with the hexagon's corners at `spread`.
///
/// Axial coordinates `|q| <= R, |r| <= R, |q+r| <= R` (a hexagon of `R` rings), mapped to the ground
/// plane by `x = pitch * (q + r/2)`, `z = pitch * r * sqrt(3)/2` - the standard pointy-corner-on-x
/// layout, in which all six corners of ring `R` sit at exactly `pitch * R` from the origin, so
/// `pitch = spread / R` makes the footprint span the available spread.
///
/// Why this rather than [`ClusterGrid`]: the lattice has THREE row directions at 60 degrees instead
/// of two at 90, and they are exactly the azimuths `--cross-orientations 3` puts planes on
/// (0/60/120 degrees, normals derived in `cross_quads`). A square grid can only ever serve two
/// orientations, and its corners inflate the bake's bounding SPHERE by sqrt(2) for trees no plane
/// faces. The hexagon is the roundest footprint a lattice can have, which is what buys the ~1.57x
/// texel density per tree at equal radius.
struct ClusterHex {
    /// Number of rings around the centre cell. `R` above.
    rings: i32,
    /// Distance between adjacent lattice points, in world units.
    pitch: f32,
    /// Cell centres for copies `1..cluster`, ordered by distance from the origin.
    cells: Vec<Vec3>,
    /// How many cells a FULL `rings`-ring hexagon holds: `3R^2 + 3R + 1`, i.e. the centered
    /// hexagonal numbers 1, 7, 19, 37, 61. Compared against `cluster` to detect a partial outer
    /// ring, which the log has to mention because it thins the outermost rows.
    full: usize,
}

impl ClusterHex {
    /// `cluster` must be > 1, which guarantees `rings >= 1` and a non-zero `pitch`.
    fn new(cluster: usize, spread: f32) -> Self {
        // Smallest R with 3R^2 + 3R + 1 >= cluster, i.e. enough rings to hold them all; the
        // outermost ring is then left partially filled if `cluster` is not one of those numbers.
        let mut rings = 1i32;
        while (3 * rings * rings + 3 * rings + 1) < cluster as i32 {
            rings += 1;
        }
        let full = (3 * rings * rings + 3 * rings + 1) as usize;
        // All six corners of ring R are at pitch*R, so this puts them exactly on `spread` - the same
        // contract ClusterGrid's outermost rows have, so the `1.15` bake headroom still covers the
        // jitter.
        let pitch = spread / rings as f32;

        let mut cells: Vec<Vec3> = Vec::with_capacity(full);
        for q in -rings..=rings {
            for r in -rings..=rings {
                if (q + r).abs() > rings {
                    continue;
                }
                cells.push(Vec3::new(
                    pitch * (q as f32 + r as f32 * 0.5),
                    0.0,
                    pitch * r as f32 * 3.0_f32.sqrt() * 0.5,
                ));
            }
        }
        // Nearest-first, so a `cluster` that is not a centered hexagonal number leaves the OUTERMOST
        // cells empty and the occupied region stays as round as the lattice allows - the bake
        // captures a bounding sphere, so a ragged far cell costs every other tree atlas resolution.
        cells.sort_by_key(|c| FloatOrd(c.length_squared()));
        // The original model is spawned at the world origin by the scene loader and cannot be moved,
        // and `ClusterLayout` index 0 is contractually `(ZERO, IDENTITY)`. A hex lattice always HAS
        // a cell exactly at the origin (unlike an even-sided square grid), so the original consumes
        // it perfectly rather than sitting half a cell off its slot.
        cells.remove(0);
        Self {
            rings,
            pitch,
            cells,
            full,
        }
    }

    /// Distance between adjacent parallel rows, the same for all three row directions:
    /// `pitch * sqrt(3)/2`. Rows are the lines of lattice points along one of the three lattice
    /// directions; a plane containing one is `row_spacing` from the plane containing the next.
    fn row_spacing(&self) -> f32 {
        self.pitch * 3.0_f32.sqrt() * 0.5
    }

    /// World distance from the origin to the OUTERMOST row of any of the three directions - the row
    /// `--cross-sides`' outer planes want to sit on. `|q|`, `|r|` and `|q+r|` all top out at
    /// `rings`, and each of those three quantities indexes one of the three row families, so by
    /// symmetry the three directions share this single number.
    fn outer_row(&self) -> f32 {
        self.row_spacing() * self.rings as f32
    }

    /// How far the OUTERMOST row extends along its own direction, from the cluster centre - the
    /// number `--cross-side-width` exists to honour, and the one place a hexagon stops behaving like
    /// a square grid.
    ///
    /// `pitch * R / 2`, i.e. exactly HALF the centre row's `pitch * R`. Take the `r = R` row: the
    /// hexagon constraints `|q| <= R` and `|q + r| <= R` collapse to `q` in `[-R, 0]`, and
    /// `x = pitch * (q + r/2)` then runs over `[-pitch*R/2, +pitch*R/2]`. The centre row (`r = 0`)
    /// has `q` free over `[-R, R]` and spans `[-pitch*R, +pitch*R]`. That is just the hexagon's edge
    /// being half its long diagonal, and it is why the outer planes have to be cropped while a
    /// square grid's do not (every row of a square grid is the full grid wide).
    ///
    /// Trees are not points: a plane wide enough to hold the row's outermost TREE needs one
    /// `base_radius` more than this on each side, which the caller adds since only it knows it.
    fn outer_row_half_extent(&self) -> f32 {
        self.pitch * self.rings as f32 * 0.5
    }

    /// Cells in the outermost ring alone: `6R` (or 1 for R = 0).
    fn ring_cells(&self) -> usize {
        (6 * self.rings) as usize
    }

    /// Cell centre for cluster copy `copy_index` (1-based), plus the same ~12% jitter
    /// [`ClusterGrid::offset`] applies, and for the same reason: a planted row rather than a
    /// spreadsheet, without blurring the rows the planes are aligned to.
    fn offset(&self, copy_index: usize, rng: &mut impl Rng) -> Vec3 {
        let jitter = self.pitch * 0.12;
        self.cells[copy_index - 1]
            + Vec3::new(
                rng.gen_range(-jitter..=jitter),
                0.0,
                rng.gen_range(-jitter..=jitter),
            )
    }
}

/// `--cluster-grid`: a `ceil(sqrt(n)) x ceil(sqrt(n))` grid of cluster-copy slots, centred on the
/// origin and spanning `+-spread`.
///
/// The point is not tidiness, it is that a grid has ROWS. `--cross-sides` adds vertical quads at
/// fixed offsets, and a quad is only believable from an intermediate angle if the silhouette it
/// draws matches trees that are actually there; against the default Vogel scatter it is an arbitrary
/// slice through noise. Line the planes up with these rows and it stops being arbitrary. Rows that
/// get no plane are not a problem - their trunks are hidden behind neighbouring foliage regardless.
struct ClusterGrid {
    /// Cells per side.
    side: usize,
    /// Distance between adjacent rows, in world units.
    pitch: f32,
    /// The nominal (unjittered) row coordinates along one axis, ascending. The grid is square, so
    /// x and z share them. This is the list `--cross-sides` has to be tuned against, which is why
    /// it is kept rather than recomputed.
    rows: Vec<f32>,
    /// Cell centres for copies `1..cluster`, ordered by distance from the origin.
    cells: Vec<Vec3>,
}

impl ClusterGrid {
    /// `cluster` must be > 1, which guarantees `side >= 2` and a non-zero `pitch`.
    fn new(cluster: usize, spread: f32) -> Self {
        let side = (cluster as f32).sqrt().ceil() as usize;
        let pitch = 2.0 * spread / (side - 1) as f32;
        let rows: Vec<f32> = (0..side)
            .map(|i| (i as f32 - (side - 1) as f32 * 0.5) * pitch)
            .collect();
        let mut cells: Vec<Vec3> = rows
            .iter()
            .flat_map(|&z| rows.iter().map(move |&x| Vec3::new(x, 0.0, z)))
            .collect();
        // Nearest-first, so that when `cluster < side*side` the cells left EMPTY are the outermost
        // corners. That keeps the occupied region roughly circular, which matters because the bake
        // captures a bounding SPHERE: filling row-major instead would leave a ragged final row and
        // waste the same atlas radius on fewer trees.
        cells.sort_by_key(|c| FloatOrd(c.length_squared()));
        // The original model is spawned at the world origin by the scene loader and cannot be moved,
        // and `ClusterLayout` index 0 is contractually `(ZERO, IDENTITY)`. So it CONSUMES the cell
        // nearest the origin rather than being placed into one. With an odd `side` that cell is
        // exactly the origin and the fit is perfect; with an even `side` there is no row at 0 at all,
        // the four central cells are equidistant, and the original ends up half a cell diagonal off
        // its nominal slot. One tree out of n, and the printout shows the real row coordinates either
        // way - but it is why an odd square (9, 25, 49) lines up best with the centre planes.
        cells.remove(0);
        Self {
            side,
            pitch,
            rows,
            cells,
        }
    }

    /// Cell centre for cluster copy `copy_index` (1-based; 0 is the original at the origin), plus a
    /// jitter of ~12% of the pitch so the result reads as a planted row rather than a spreadsheet.
    /// Small on purpose: push it much past ~15% and the rows stop being visually identifiable, which
    /// is the entire reason for the grid.
    fn offset(&self, copy_index: usize, rng: &mut impl Rng) -> Vec3 {
        let jitter = self.pitch * 0.12;
        self.cells[copy_index - 1]
            + Vec3::new(
                rng.gen_range(-jitter..=jitter),
                0.0,
                rng.gen_range(-jitter..=jitter),
            )
    }
}

/// The [`CrossLayout`] the `--cross-*` flags describe.
///
/// One place, because the two mesh builders, the quad-count report and the alignment printout must
/// all be describing the SAME quad set - a printout tuned against a layout the mesh does not have is
/// worse than no printout. `horizontal_height` is a parameter rather than read off `settings`
/// because the spawn path corrects it for the cluster's radius inflation first (see `impost`), and
/// the printout does not care about it at all. `trim` is a parameter for the same reason and a
/// stronger one: it is MEASURED off the baked atlas, so it does not exist at all until the bake
/// completes, and every caller that runs before then passes [`CrossTrim::FULL`].
fn cross_layout(settings: &BakeSettings, horizontal_height: f32, trim: CrossTrim) -> CrossLayout {
    CrossLayout {
        horizontal_height,
        orientations: settings.cross_orientations,
        side_offset: settings.cross_sides,
        side_width: settings.cross_side_width,
        center_planes: !settings.cross_no_center,
        side_trim: trim.side,
        top_trim: trim.top,
        mirror_sides: settings.cross_mirror_sides,
    }
}

/// Prints where the vertical planes land against the cluster's rows, in the cluster's own (world)
/// frame, so the two can be lined up by eye - plus, for a lattice placement, the `--cross-sides`
/// value that puts the outer planes exactly on the outermost rows, and the `--cross-side-width`
/// value that makes them as WIDE as those rows (which only differs from 1.0 on a hexagon, whose
/// outer row is half the length of its centre one).
///
/// The recommendation is printed, not applied. The best offset is a judgement call - it trades "the
/// plane sits on a dense row" against "the planes are far enough apart to give real depth" - and the
/// number that looks right also depends on `--cluster` and on how much of the sphere the foliage
/// fills. So the tool here is legibility plus one good starting point, not a solver.
fn print_cross_alignment(
    settings: &BakeSettings,
    sphere: &Sphere,
    base_radius: f32,
    placement: &ClusterPlacement,
) {
    // FULL, not the measured trim: this runs at scene-load time, before anything has been baked, and
    // nothing it prints depends on the trim anyway (quad COUNT and plane OFFSETS, both untouched by
    // how wide a quad ends up).
    let layout = cross_layout(settings, settings.cross_height, CrossTrim::FULL);
    // The mesh is authored in unit space (+-0.5) and the vertex shader scales it by `2 * radius`
    // about `sphere.center`, so a plane at unit offset `s` lands `s * 2 * radius` from the centre.
    // `sphere.center` is the bake sphere's centre, whereas the cluster copies are placed around the
    // world ORIGIN - hence the per-orientation `dot(centre, n)` below rather than a bare +-value.
    //
    // The INFLATED `sphere.radius` is the right scale here, unlike for `--cross-height` (which is
    // divided by the inflation at spawn - see `impost`). `--cluster` grows the radius because the
    // group is horizontally wider, and these planes have to span that same horizontal footprint to
    // reach the outer rows; scaling them by the single-tree radius instead would cap them at
    // `+-base_radius` and leave every row but the middle one unrepresented. `s` is therefore a
    // fraction of the CLUSTER, which is exactly what it has to be tuned as - and the `1.15`
    // placement headroom folded into `radius` is why `s` does not map to rows by any tidy closed
    // form, hence this printout.
    let half_extent = sphere.radius * 2.0;
    let fmt = |vals: Vec<f32>| {
        vals.iter()
            .map(|v| format!("{v:+.2}"))
            .collect::<Vec<_>>()
            .join(", ")
    };

    println!(
        "cross quads: {} per instance = {} orientation(s) x {} plane(s) + 1 horizontal \
         (--cross-orientations {}, --cross-sides {}{}{})",
        cross_quad_count(&layout),
        layout.orientations,
        usize::from(layout.center_planes) + if layout.side_offset > 0.0 { 2 } else { 0 },
        layout.orientations,
        layout.side_offset,
        if layout.center_planes {
            ""
        } else {
            ", no centre planes"
        },
        if layout.mirror_sides {
            ", outer planes mirrored"
        } else {
            ""
        }
    );

    // One line per orientation, measured ALONG THAT ORIENTATION'S OWN NORMAL: with 3 orientations
    // the planes are no longer axis-aligned, so "at x = ..." would only be meaningful for one of
    // them. `dot(p, n)` is the coordinate that a row of trees and a plane facing it must agree on,
    // whatever the azimuth.
    for k in 0..layout.orientations {
        let theta = PI * k as f32 / layout.orientations.max(1) as f32;
        let n = Vec3::new(theta.sin(), 0.0, theta.cos());
        let centre = Vec3::from(sphere.center).dot(n);
        let mut planes = Vec::new();
        if layout.center_planes {
            planes.push(centre);
        }
        if layout.side_offset > 0.0 {
            planes.insert(0, centre - layout.side_offset * half_extent);
            planes.push(centre + layout.side_offset * half_extent);
        }
        println!(
            "  orientation {k}: azimuth {:5.1} deg, normal (x {:+.3}, z {:+.3}) | planes at dot(p,n) = {}",
            theta.to_degrees(),
            n.x,
            n.z,
            fmt(planes)
        );
    }

    // The cluster copies are laid out around the world origin, so the rows need no re-centring -
    // and that mismatch against the sphere-centred planes above is exactly what has to be eyeballed.
    match placement {
        ClusterPlacement::Vogel => {
            println!(
                "  cluster rows: none (--cluster-grid / --cluster-hex off) - the outer planes are \
                 arbitrary slices through a random scatter and will read as cardboard"
            );
            print_cross_side_width_recommendation(
                None,
                base_radius,
                sphere.radius,
                layout.side_width,
                "a disc scatter has no rows to be shorter than each other",
            );
        }
        ClusterPlacement::Grid(grid) => {
            println!(
                "  square grid rows at x = {} (and the same at z)",
                fmt(grid.rows.clone())
            );
            print_cross_sides_recommendation(
                grid.rows.last().copied().unwrap_or(0.0).abs(),
                half_extent,
                layout.side_offset,
                "square grid outer row",
            );
            print_cross_side_width_recommendation(
                None,
                base_radius,
                sphere.radius,
                layout.side_width,
                "every row of a square grid is the full grid wide",
            );
        }
        ClusterPlacement::Hex(hex) => {
            // The three row families are indexed by |r|, |q+r| and |q| respectively, and each is
            // perpendicular to one of the azimuths `--cross-orientations 3` generates - which is the
            // entire reason `--cluster-hex` exists. All three top out at the same distance, so the
            // three lines below deliberately carry the same number: seeing it three times is what
            // makes "one --cross-sides value serves all three" obvious rather than assumed.
            for k in 0..3 {
                let theta = PI * k as f32 / 3.0;
                println!(
                    "  hex row direction {k}: azimuth {:5.1} deg, rows every {:.2}, outermost row \
                     at dot(p,n) = {:+.2} / {:+.2}",
                    theta.to_degrees(),
                    hex.row_spacing(),
                    -hex.outer_row(),
                    hex.outer_row()
                );
            }
            print_cross_sides_recommendation(
                hex.outer_row(),
                half_extent,
                layout.side_offset,
                "hex outermost row",
            );
            print_cross_side_width_recommendation(
                Some(hex.outer_row_half_extent()),
                base_radius,
                sphere.radius,
                layout.side_width,
                "hex outermost row",
            );
            if layout.orientations != 3 {
                warn!(
                    "--cluster-hex lays the trees out on THREE row directions (0/60/120 deg) but \
                     --cross-orientations is {}; only 3 puts a plane on every row family",
                    layout.orientations
                );
            }
        }
    }
}

/// The `--cross-sides` value that lands the outer planes on `outer_row` world units from the
/// cluster centre. Split out so the grid and hex branches cannot drift in how they compute it.
///
/// `s * 2 * radius` is where the vertex shader puts a plane at unit offset `s` (see
/// `print_cross_alignment`), so the inverse is simply `outer_row / (2 * radius)`. `half_extent` IS
/// that `2 * radius`.
///
/// Loud on purpose: this is the number the layout is tuned with, it changes with every `--cluster`
/// size, and there is no way to guess it from the command line.
fn print_cross_sides_recommendation(outer_row: f32, half_extent: f32, current: f32, what: &str) {
    println!(
        "  >>> RECOMMENDED --cross-sides {:.4} <<<  ({what} at {outer_row:.2} world / 2*radius \
         {half_extent:.2})",
        outer_row / half_extent
    );
    if current > 0.0 {
        println!(
            "      current --cross-sides {current} puts the outer planes at +-{:.2} world",
            current * half_extent
        );
    }
}

/// The `--cross-side-width` value that makes the outer planes as wide as the row they stand on,
/// instead of as wide as the whole bounding sphere.
///
/// `outer_row_half_extent` is how far that row's outermost CELL CENTRE sits from the cluster centre
/// along the row's own direction, or `None` when every row is the same length and the answer is
/// therefore the full-width `1.0`. A tree is not a point, so the plane has to reach one
/// `base_radius` further to contain the outermost tree's foliage; and the quad's own half-extent in
/// world units is `radius` (the mesh spans `+-0.5` and the shader scales by `2 * radius`), so the
/// fraction is `(row + base_radius) / radius`.
///
/// Loud in the same style as `print_cross_sides_recommendation`, and for the same reason: it is a
/// property of the cluster geometry that changes with every `--cluster` size and cannot be guessed
/// from the command line. Printed, not applied - how much of the outer row's foliage to keep is a
/// judgement call, and cropping harder than this is a legitimate way to tighten the silhouette.
fn print_cross_side_width_recommendation(
    outer_row_half_extent: Option<f32>,
    base_radius: f32,
    radius: f32,
    current: f32,
    what: &str,
) {
    match outer_row_half_extent {
        // Not "no recommendation": 1.0 IS the recommendation, and saying why keeps the flag from
        // looking like something a square-grid bake forgot to set.
        None => println!(
            "  >>> RECOMMENDED --cross-side-width 1.0000 <<<  ({what}, so the outer planes are as \
             wide as the centre ones. Only --cluster-hex differs: its outermost row is HALF the \
             centre row, a hexagon's edge against its long diagonal)"
        ),
        Some(row) => {
            let world = row + base_radius;
            println!(
                "  >>> RECOMMENDED --cross-side-width {:.4} <<<  ({what} reaches +-{row:.2} world \
                 (pitch*R/2, half the centre row), +{base_radius:.2} for the outermost tree's own \
                 half-width = +-{world:.2}, against a quad half-extent of {radius:.2})",
                world / radius
            );
        }
    }
    if (current - 1.0).abs() > 1e-6 {
        println!(
            "      current --cross-side-width {current} makes the outer planes +-{:.2} world, \
             showing the middle {:.0}% of the side tile",
            current * radius,
            current * 100.0
        );
    }
}

/// Reorders `transforms` along a Morton (Z-order) curve over the XZ plane, so that consecutive
/// entries are spatial neighbours and `--cross-merge`'s chunks come out as compact patches rather
/// than as forest-wide clouds. Purely a reordering: every transform is returned exactly once,
/// unchanged.
///
/// Y is ignored on purpose. The placement is a flat field (`hemi_mult` zeroes the y jitter outside
/// spherical mode), so a 3D key would spend a third of its bits on a constant.
fn morton_sorted(transforms: &[Transform]) -> Vec<Transform> {
    // 16 bits of x interleaved with 16 bits of z. Standard bit-spreading ladder.
    fn spread(mut v: u32) -> u32 {
        v = (v | (v << 8)) & 0x00FF00FF;
        v = (v | (v << 4)) & 0x0F0F0F0F;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        v
    }

    let (mut min, mut max) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for t in transforms {
        let p = Vec2::new(t.translation.x, t.translation.z);
        min = min.min(p);
        max = max.max(p);
    }
    // `max(1e-6)` guards the degenerate single-tree / zero-extent case, where the quantisation
    // would otherwise divide by zero and every key would come out NaN-ordered.
    let extent = (max - min).max(Vec2::splat(1e-6));

    let mut out: Vec<(u32, Transform)> = transforms
        .iter()
        .map(|t| {
            let p = (Vec2::new(t.translation.x, t.translation.z) - min) / extent;
            let q = (p * 65535.0).clamp(Vec2::ZERO, Vec2::splat(65535.0));
            (spread(q.x as u32) | (spread(q.y as u32) << 1), *t)
        })
        .collect();
    // Stable, so equal keys keep the rng's order and the result stays deterministic for a seed.
    out.sort_by_key(|(key, _)| *key);
    out.into_iter().map(|(_, t)| t).collect()
}

// Smoothed frame stats + the top GPU timestamp spans, in one place so `print_fps` and the
// `--exit-after` summary cannot report different numbers from different code.
struct FrameStats {
    fps: Option<f64>,
    frame_ms: Option<f64>,
    resolution: String,
    gpu: Vec<(String, f64)>,
}

fn collect_frame_stats(diagnostics: &DiagnosticsStore, windows: &Query<&Window>) -> FrameStats {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed());
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed());
    let resolution = windows
        .single()
        .map(|w| format!("{}x{}", w.physical_width(), w.physical_height()))
        .unwrap_or_default();
    // GPU pass timings from RenderDiagnosticsPlugin (timestamp queries). The top-level
    // "elapsed_gpu" is total GPU time; compare it to the wall frame time above: if GPU
    // time << frame time, you're CPU/submission-bound, not fill-bound.
    let mut gpu: Vec<(String, f64)> = diagnostics
        .iter()
        .filter(|d| d.path().as_str().ends_with("elapsed_gpu"))
        .filter_map(|d| Some((d.path().as_str().to_owned(), d.smoothed()?)))
        .filter(|(_, v)| *v > 0.01)
        .collect();
    gpu.sort_by(|a, b| b.1.total_cmp(&a.1));
    FrameStats {
        fps,
        frame_ms,
        resolution,
        gpu,
    }
}

// `--exit-after <secs>`: wait out the window that started when the imposters were spawned, print a
// single grep-able block (delimited by ===BOIMP SUMMARY=== / ===END BOIMP SUMMARY===) and quit.
// Reuses `collect_frame_stats`, i.e. exactly the numbers `print_fps` has been printing each second,
// so the summary is the last one-second-smoothed sample rather than a separately-derived figure.
#[allow(clippy::too_many_arguments)]
fn auto_exit(
    mut commands: Commands,
    settings: Res<BakeSettings>,
    mut auto: ResMut<AutoRun>,
    diagnostics: Res<DiagnosticsStore>,
    windows: Query<&Window>,
    imposters: Query<Entity, With<MeshMaterial3d<Imposter>>>,
    cross_imposters: Query<Entity, With<MeshMaterial3d<CrossImposter>>>,
    // Present only under `--cross-trim`; see the summary block below.
    trim: Option<Res<CrossTrimReport>>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(secs) = settings.exit_after else {
        return;
    };

    let live = if settings.cross {
        cross_imposters.iter().count()
    } else {
        imposters.iter().count()
    };

    // Start the measurement window the frame the imposters first exist, rather than having `impost`
    // stamp it: `impost` is already at bevy's 16-parameter SystemParam limit, and detecting the
    // spawn here is equivalent (the entities appear on the frame after the bake request either way)
    // while also covering an interactive SPACE run with `--exit-after` but no `--auto-bake`.
    let spawned_at = match auto.spawned_at {
        Some(t) => t,
        None => {
            if live == 0 {
                return;
            }
            let now = Instant::now();
            auto.spawned_at = Some(now);
            now
        }
    };

    let elapsed = spawned_at.elapsed().as_secs_f32();

    // Grab the window a second before quitting: the capture is a render-world readback that
    // resolves a frame or two later, and `save_to_disk` writes on a task pool thread, so firing it
    // in the same tick as `AppExit` would race the shutdown and usually produce no file.
    if let Some(path) = settings.screenshot.clone() {
        if !auto.shot && elapsed >= (secs - 1.0).max(0.0) {
            auto.shot = true;
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path.clone()));
            println!("screenshot queued -> {path}");
        }
    }

    if auto.done || elapsed < secs {
        return;
    }
    auto.done = true;

    let stats = collect_frame_stats(&diagnostics, &windows);
    let mode = if settings.cross {
        "cross"
    } else {
        "octahedral"
    };
    let alpha = if settings.a2c {
        "a2c"
    } else if settings.mask || settings.cross {
        "mask"
    } else {
        "blend"
    };
    let level = if settings.cross {
        settings.debug_level.min(2)
    } else {
        settings.debug_level.min(4)
    };

    println!("===BOIMP SUMMARY===");
    println!("mode: {mode}");
    println!("alpha_mode: {alpha}");
    // With --cross-merge, `live` is the number of PATCH entities, i.e. the draw-call-relevant
    // number, and the tree count is `settings.count`. Printed as separate lines so a diff of two
    // summary blocks shows the batching change directly rather than as a mysterious 1000 -> 16.
    let merge = if settings.cross {
        settings.cross_merge.max(1)
    } else {
        1
    };
    println!("imposters: {live} (requested {})", settings.count);
    println!("cross_merge: {merge}");
    println!(
        "entities: {live}   trees: {}   trees_per_entity: {merge}",
        if merge > 1 { settings.count } else { live }
    );
    println!("spacing: {}", settings.spacing);
    println!("tile_size: {}", settings.tile_size);
    println!(
        "grid_size: {}",
        if settings.cross {
            2
        } else {
            settings.grid_size
        }
    );
    println!("debug_level: {level}");
    // Only with `--cross-trim` - the resource does not exist otherwise, so an untrimmed run's
    // summary block is byte-identical to the one it printed before this feature landed, which is
    // what makes two summaries diffable across the change.
    if let Some(CrossTrimReport(trim)) = trim.as_deref() {
        // The quad-area fraction is the whole point of the flag: it is (modulo the quads' screen
        // orientation) the fraction of the fill this run pays compared with an untrimmed one.
        for (name, rect) in [("side", trim.side), ("top", trim.top)] {
            println!(
                "cross_trim_{name}: ({:.4}, {:.4})..({:.4}, {:.4})   quad_area: {:.1}%",
                rect.min.x,
                rect.min.y,
                rect.max.x,
                rect.max.y,
                rect.width() * rect.height() * 100.0
            );
        }
    }
    println!("measure_window_s: {secs}");
    println!("resolution: {}", stats.resolution);
    match (stats.fps, stats.frame_ms) {
        (Some(fps), Some(ms)) => println!("fps: {fps:.1}\nframe_ms: {ms:.3}"),
        _ => println!("fps: n/a\nframe_ms: n/a"),
    }
    if stats.gpu.is_empty() {
        println!("gpu: n/a (no TIMESTAMP_QUERY spans)");
    } else {
        for (path, v) in stats.gpu.iter().take(12) {
            // RenderDiagnosticsPlugin records the bake span once and then leaves it stale, so it
            // sorts to the top while contributing nothing to the steady-state frame. Flagged rather
            // than filtered, because a one-shot bake cost is worth reading - just not as frame time.
            let note = if path.contains("imposter_bake") {
                "   [one-shot bake, NOT per-frame]"
            } else {
                ""
            };
            println!("gpu: {v:8.3} ms  {path}{note}");
        }
    }
    println!("===END BOIMP SUMMARY===");

    exit.write(AppExit::Success);
}

fn update_lights(
    key_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut DirectionalLight)>,
    mut animate_directional_light: Local<bool>,
) {
    for (_, mut light) in &mut query {
        if key_input.just_pressed(KeyCode::KeyU) {
            light.shadow_maps_enabled = !light.shadow_maps_enabled;
        }
    }

    if key_input.just_pressed(KeyCode::KeyL) {
        *animate_directional_light = !*animate_directional_light;
    }
    if *animate_directional_light {
        for (mut transform, _) in &mut query {
            transform.rotation = Quat::from_euler(
                EulerRot::ZYX,
                0.0,
                time.elapsed_secs() * PI / 15.0,
                -FRAC_PI_4,
            );
        }
    }
}

#[derive(Component)]
pub struct Rotate;

// per-copy (offset, yaw) transforms of the baked cluster, captured at bake time so `--swap`
// can rebuild the identical cluster as real models. Index 0 is the original at the origin.
// For `--cluster 1` this is just a single identity entry.
#[derive(Resource, Default)]
struct ClusterLayout(Vec<(Vec3, Quat)>);

// attached to every spawned imposter quad; tracks the real glTF model entities currently
// substituted in for it by `--swap` (empty while the imposter itself is being shown). With a
// baked cluster there is one entity per cluster copy.
#[derive(Component, Default)]
struct SwapReal {
    reals: Vec<Entity>,
    // true once `dress_real_models` has instantiated and revealed the real copies; until then the
    // imposter stays visible (covering the gap), after which `swap_close` owns its visibility.
    dressed: bool,
}

// a real glTF model spawned by `swap_close` in place of a nearby imposter. Holds the imposter
// entity it stands in for (so it can be re-shown on swap-out) and lets `O` clear these models
// alongside the imposters.
#[derive(Component)]
struct RealModel {
    imposter: Entity,
}

// set on a `RealModel` once its scene's meshes have been moved onto the imposter render layer
// and it has been revealed; lets `dress_real_models` skip models it has already finished.
#[derive(Component)]
struct RealModelDressed;

fn rotate(mut q: Query<&mut Transform, With<Rotate>>, time: Res<Time>) {
    for mut t in q.iter_mut() {
        t.rotation = Quat::from_rotation_y(time.elapsed_secs());
    }
}

fn swap_old(key_input: Res<ButtonInput<KeyCode>>, mut imps: ResMut<Assets<Imposter>>) {
    if key_input.just_pressed(KeyCode::KeyP) {
        for a in imps.iter_mut() {
            a.1.data.flags ^= 2;
        }
    }
}

// --swap: substitute the real glTF model for any imposter the camera gets close to, and
// restore the imposter once it's far away again. Distance is measured to the imposter quad
// (its visual centre) and expressed as a multiple of the model radius, so the same value
// works regardless of how big the source model is. A small hysteresis band (swap in at
// `swap_distance`, back out at 1.25x that) keeps imposters near the boundary from flickering
// between the two representations frame to frame.
#[allow(clippy::too_many_arguments)]
fn swap_close(
    mut commands: Commands,
    settings: Res<BakeSettings>,
    scene_handle: Res<SceneHandle>,
    cluster: Res<ClusterLayout>,
    gltf_assets: Res<Assets<Gltf>>,
    camera: Query<&Transform, With<CameraController>>,
    mut imposters: Query<(Entity, &Transform, &mut Visibility, &mut SwapReal)>,
) {
    if !settings.swap {
        return;
    }
    let Ok(cam) = camera.single() else {
        return;
    };
    let Some(gltf) = gltf_assets.get(&scene_handle.gltf_handle) else {
        return;
    };
    let Some(scene) = gltf.scenes.get(scene_handle.scene_index).cloned() else {
        return;
    };

    let cam_pos = cam.translation;
    let center = Vec3::from(scene_handle.sphere.center);
    let radius = scene_handle.sphere.radius;
    // `--swap-distance` is the OUTER edge: a real copy is spawned when the camera gets this close,
    // and (while fading) it first appears fully dithered away so only the imposter shows. `despawn`
    // is the usual 1.25x hysteresis beyond it so copies near the boundary don't flicker.
    let spawn = settings.swap_distance * radius;
    let despawn = spawn * 1.25;
    // `solid` is the INNER edge of the fade band: at/below it the model is fully opaque and the
    // imposter is hidden. With `--swap-fade` the model dithers from invisible (at `spawn`) to solid
    // (at `solid = spawn / band`) as the camera approaches; without it `solid == despawn`, so the
    // imposter is simply hidden the whole time a (popped-in, opaque) real copy exists.
    let solid = if settings.swap_fade {
        spawn / settings.swap_fade_band
    } else {
        despawn
    };
    let (spawn2, despawn2, solid2) = (spawn * spawn, despawn * despawn, solid * solid);

    for (entity, xf, mut vis, mut swap) in imposters.iter_mut() {
        let d2 = xf.translation.distance_squared(cam_pos);
        if swap.reals.is_empty() {
            if d2 < spawn2 {
                swap.dressed = false;
                // The imposter billboards to face the camera and uses its transform rotation R
                // only to sample the octahedral atlas in direction `R * view_dir` - i.e. it shows
                // the baked content rotated by R. So a real copy carries the inverse rotation
                // Q = R^-1. The whole cluster was baked around sphere.center, so copy `i` (baked
                // at world offset, yaw) maps into the imposter frame as
                //   translation = xf.translation + Q * (offset - center),  rotation = Q * yaw.
                // (Copy 0 = origin/identity reduces to placing the single model at the quad.)
                let q = xf.rotation.inverse();
                for &(offset, yaw) in &cluster.0 {
                    let translation = xf.translation + q * (offset - center);
                    // Spawn hidden and on the imposter render layer (1). The bake camera captures
                    // the default layer (0), so a swapped-in model on layer 0 would be baked into
                    // the atlas and corrupt every imposter. `dress_real_models` moves the scene's
                    // meshes onto layer 1, reveals the model, and marks `dressed` once a real copy
                    // is actually ready - keeping it out of the bake and avoiding a gap (the
                    // imposter stays visible here until then; `swap_close` hides it after).
                    let real = commands
                        .spawn((
                            WorldAssetRoot(scene.clone()),
                            Transform::from_translation(translation).with_rotation(q * yaw),
                            Visibility::Hidden,
                            RenderLayers::layer(1),
                            RealModel { imposter: entity },
                        ))
                        .id();
                    swap.reals.push(real);
                }
            }
        } else if d2 > despawn2 {
            for real in swap.reals.drain(..) {
                commands.entity(real).despawn();
            }
            swap.dressed = false;
            *vis = Visibility::Visible;
        } else if swap.dressed {
            // Real copies are up and revealed: hide the imposter once the model is solid enough to
            // cover it (inside `solid`), otherwise keep it visible so it shows through the model's
            // dither holes across the fade band. Until `dressed`, leave the imposter alone - it is
            // still Visible from before the swap, covering the one or two frames the scene needs to
            // instantiate.
            *vis = if d2 < solid2 {
                Visibility::Hidden
            } else {
                Visibility::Visible
            };
        }
    }
}

// Bevy doesn't propagate `RenderLayers` down the hierarchy, and a `SceneRoot`'s meshes only
// appear (as children) a frame or two after the root is spawned - all on the default layer 0,
// which the bake camera captures. So once a swapped-in model's scene has instantiated, push
// every descendant onto the imposter layer (1), reveal the (always-solid) model, and mark its
// imposter `dressed` so `swap_close` takes over the imposter's visibility (it stayed visible until
// now, covering the instantiation gap). The cross-fade itself is done by dithering the imposter
// away in its own shader (see the `swap_fade` band on the shared Imposter material), not here.
fn dress_real_models(
    mut commands: Commands,
    mut roots: Query<
        (Entity, &mut Visibility, &RealModel),
        (Without<RealModelDressed>, Without<SwapReal>),
    >,
    children: Query<&Children>,
    untagged: Query<(), Without<RenderLayers>>,
    mut imposters: Query<&mut SwapReal>,
) {
    for (root, mut root_vis, real) in roots.iter_mut() {
        let descendants: Vec<Entity> = children.iter_descendants(root).collect();
        if descendants.is_empty() {
            // scene instance not spawned yet - try again next frame
            continue;
        }
        for &d in &descendants {
            if untagged.contains(d) {
                commands.entity(d).insert(RenderLayers::layer(1));
            }
        }
        *root_vis = Visibility::Visible;
        if let Ok(mut swap) = imposters.get_mut(real.imposter) {
            swap.dressed = true;
        }
        commands.entity(root).insert(RealModelDressed);
    }
}

// press F to flip stochastic dither tile selection on/off at runtime. flipping the
// flag changes the ImposterKey, so bevy re-specializes the pipeline with/without the
// DITHERED shader def.
fn toggle_dither(key_input: Res<ButtonInput<KeyCode>>, mut imps: ResMut<Assets<Imposter>>) {
    if key_input.just_pressed(KeyCode::KeyF) {
        for a in imps.iter_mut() {
            a.1.data.flags ^= DITHER_FLAG;
        }
        let on = imps
            .iter()
            .next()
            .is_some_and(|a| a.1.data.flags & DITHER_FLAG != 0);
        println!("dither: {}", if on { "on" } else { "off" });
    }
}

// press 0-4 to set the fragment-shader perf ladder (`debug_level`) on every live imposter
// material. This is the measurement tool the cross-imposter work hinges on: each rung strips one
// more stage off the fragment shader while leaving the vertex path, quad size and (up to the last
// rung) the alpha-test silhouette alone, so the frame-time delta between two rungs attributes cost
// to exactly one stage instead of to a guess.
//
//   octahedral (0-4): 1 = no lighting/post, 2 = + no detail fade, 3 = + forced single tap,
//                     4 = flat colour, zero atlas fetches (the only non-coverage-preserving rung)
//   cross      (0-2): 1 = no lighting/post, 2 = flat colour, zero atlas fetches
//
// The ladders are deliberately NOT the same length - the cross fragment has no detail fade and no
// multi-tap path to strip - so a key above a material's top rung is clamped rather than ignored,
// and the printout says so; silently doing nothing would read as "that stage costs zero".
//
// `debug_level` is a uniform, not a flag, so stepping it re-uploads the material without
// respecializing any pipeline - the measurement doesn't include a shader compile.
fn set_debug_level(
    key_input: Res<ButtonInput<KeyCode>>,
    settings: Res<BakeSettings>,
    mut imps: ResMut<Assets<Imposter>>,
    mut cross_imps: ResMut<Assets<CrossImposter>>,
) {
    const KEYS: [KeyCode; 5] = [
        KeyCode::Digit0,
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
    ];
    let Some(requested) = KEYS.iter().position(|k| key_input.just_pressed(*k)) else {
        return;
    };
    let requested = requested as u32;

    let (max, kind) = if settings.cross {
        (2, "cross")
    } else {
        (4, "octahedral")
    };
    let level = requested.min(max);

    if settings.cross {
        for a in cross_imps.iter_mut() {
            a.1.data.debug_level = level;
        }
    } else {
        for a in imps.iter_mut() {
            a.1.data.debug_level = level;
        }
    }

    if level == requested {
        println!("debug_level: {level} ({kind} ladder, 0-{max})");
    } else {
        println!(
            "debug_level: {level} ({kind} ladder only goes to {max}, clamped from {requested})"
        );
    }
}

// press G to dump every registered diagnostic, including the per-pass GPU timings
// Continuously print smoothed FPS / frame time once per second so different render
// modes (e.g. --mask vs --a2c) can be compared at a glance without pressing G.
fn print_fps(
    time: Res<Time>,
    diagnostics: Res<DiagnosticsStore>,
    windows: Query<&Window>,
    mut elapsed: Local<f32>,
) {
    *elapsed += time.delta_secs();
    if *elapsed < 1.0 {
        return;
    }
    *elapsed = 0.0;

    let stats = collect_frame_stats(&diagnostics, &windows);
    if let (Some(fps), Some(ms)) = (stats.fps, stats.frame_ms) {
        println!(
            "fps: {fps:6.1}  ({ms:5.2} ms/frame cpu+gpu wall)  [{}]",
            stats.resolution
        );
    }
    for (path, v) in stats.gpu.iter().take(8) {
        println!("    {v:7.3} ms  {path}");
    }
}

// Reports the cost of a one-shot bake once it finishes (see BakeProbe). Ticks a frame counter
// while a bake is in flight and, on BakeState::Finished, prints:
//   - end-to-end wall latency + frames spanned (the hitch/stream budget a game would feel),
//   - the GPU bake-pass time from RenderDiagnosticsPlugin's `render/imposter_bake/elapsed_gpu`
//     timestamp span (n/a if the backend lacks TIMESTAMP_QUERY),
//   - the atlas VRAM: (tile_size * grid_size)^2 texels at 8 bytes/texel (Rg32Uint).
// Note the GPU pass time lags the CPU by a couple of frames (timestamp readback), but it has
// resolved by the time the bake reports Finished, so the reading reflects this bake.
fn report_bake_timing(
    mut probe: ResMut<BakeProbe>,
    settings: Res<BakeSettings>,
    diagnostics: Res<DiagnosticsStore>,
    cams: Query<&ImposterBakeCamera>,
) {
    let Some(state) = probe.inflight.as_mut() else {
        return;
    };
    state.frames += 1;

    // the one-shot bake camera (skip a continuous one if somehow present). It is spawned via a
    // deferred command, so on the request frame it doesn't exist yet - just keep waiting. A manual
    // clear (O) cancels the probe explicitly, so we never need to abandon it here.
    let Some(cam) = cams.iter().find(|c| !c.continuous) else {
        return;
    };
    if cam.state != BakeState::Finished {
        return;
    }

    let wall_ms = state.start.elapsed().as_secs_f64() * 1000.0;
    let frames = state.frames;

    // GPU time of the bake render pass (milliseconds). RenderDiagnosticsPlugin records this span
    // once per bake and then leaves it stale, so the latest value is this bake's reading.
    let gpu = diagnostics
        .iter()
        .find(|d| {
            let p = d.path().as_str();
            p.contains("imposter_bake") && p.ends_with("elapsed_gpu")
        })
        .and_then(|d| d.value().or_else(|| d.smoothed()));
    let gpu_str = gpu
        .map(|v| format!("{v:.2} ms"))
        .unwrap_or_else(|| "n/a (no TIMESTAMP_QUERY?)".to_string());

    // atlas is Rg32Uint = 8 bytes/texel, side = tile_size * grid_size. `--cross` ignores
    // `--grid` entirely (`cross_bake_camera` hardcodes a 2x2 grid of manual views), so reporting
    // `settings.grid_size` here would overstate the cross atlas by ~50x at the default --grid 15
    // and make the VRAM half of the A/B meaningless.
    let grid_size = if settings.cross {
        2
    } else {
        settings.grid_size
    };
    let side = settings.tile_size * grid_size;
    let bytes = side as u64 * side as u64 * 8;

    println!(
        "bake cost: {wall_ms:.1} ms wall over {frames} frame(s) | gpu pass {gpu_str} | \
         atlas {side}x{side} Rg32Uint = {:.2} MiB",
        bytes as f64 / (1024.0 * 1024.0)
    );

    probe.inflight = None;
}

// recorded by RenderDiagnosticsPlugin (these are gated on the wgpu TIMESTAMP_QUERY
// feature; if your adapter/backend doesn't expose it, only CPU diagnostics appear).
fn dump_diagnostics(key_input: Res<ButtonInput<KeyCode>>, diagnostics: Res<DiagnosticsStore>) {
    if !key_input.just_pressed(KeyCode::KeyG) {
        return;
    }

    println!("--- diagnostics ---");
    let mut lines: Vec<(String, String)> = diagnostics
        .iter()
        .filter_map(|d| {
            let value = d.smoothed()?;
            Some((
                d.path().as_str().to_owned(),
                format!("{:>12.4}{}", value, d.suffix),
            ))
        })
        .collect();
    lines.sort();
    for (path, value) in lines {
        println!("  {path:<48} {value}");
    }
    println!("-------------------");
}

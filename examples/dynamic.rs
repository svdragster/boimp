// spawn a gltf and bake dynamic imposters every frame
// the gltf can be animated or moved, and the changes are reflected in every imposter.
// scene mgmt copied wholesale from bevy

use std::f32::consts::{FRAC_PI_4, PI};
use std::time::Instant;

use bevy::{
    animation::AnimationTargetId,
    anti_alias::fxaa::Fxaa,
    camera::{
        primitives::{Aabb, Sphere},
        visibility::RenderLayers,
    },
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
    ecs::entity::EntityHashMap,
    math::FloatOrd,
    platform::collections::HashMap,
    prelude::*,
    scene::InstanceId,
};
use boimp::{
    bake::BakeState,
    render::{DummyIndicesImage, DITHER_FLAG},
    GridMode, Imposter, ImposterBakeCamera, ImposterBakePlugin, ImposterData,
};
use camera_controller::{CameraController, CameraControllerPlugin};
use rand::{thread_rng, Rng};

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
}

fn main() {
    println!(
        "press SPACE to bake the imposter once and spawn the imposters. press O to clear them.\n\
         press I to instead bake continuously every frame (for animated/moving sources)."
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
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        features: WgpuFeatures::TIMESTAMP_QUERY
                            | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                        ..Default::default()
                    }),
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
            ),
        )
        .init_resource::<BakeProbe>()
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

    let unused = args.finish();
    if !unused.is_empty() {
        println!("unrecognized arguments: {unused:?}");
        println!("args: \n--mode [h]emispherical or [s]pherical\n--grid n (grid size, default 15)\n--image n (image size, default 1024)\n--count n (number of imposters to spawn)\n--multisample-source <n> (to multisample when generating the imposter, try 8)\n--multisample-target (to multisample when rendering imposters)\n--mask (use AlphaMode::Mask instead of Blend, enabling early-Z)\n--a2c (use AlphaMode::AlphaToCoverage: MSAA anti-aliases the alpha-tested silhouette edges, no temporal pass; overrides --mask)\n--fxaa (enable FXAA screen-space anti-aliasing on the camera)\n--dither (static stochastic screen-space dither tile selection instead of the continuous blend; toggle at runtime with F)\n--coverage (coverage-preserving alpha for distant foliage: rescales+softens minified alpha so thin features keep density and feed A2C/MSAA fractional coverage; pair with --a2c)\n--fade (distance detail fade: as imposters minify, flatten the baked normal map, raise roughness, and desaturate albedo toward a smooth blob to kill far-away sparkle)\n--cluster n (bake n randomly-placed copies of the source model into a single imposter, default 1)\n--spacing f (scales the gap between spawned imposters, default 1.0; <1 packs them closer, >1 spreads them out)\n--ambient f (ambient light brightness/fill, default 1000.0)\n--no-ambient (disable ambient fill, leaving only the directional light)\n--swap (swap each imposter for the real glTF model when the camera gets close, and back to the imposter when it moves away)\n--swap-distance f (camera distance, in multiples of the model radius, at which --swap kicks in, default 8.0)\n--swap-fade (dither-dissolve the real model across a distance band instead of popping it to/from the imposter; the imposter shows through the dither holes for a smooth cross-fade)\n--swap-fade-band f (width of the --swap-fade band: the real model is fully dithered away at swap-distance and solidifies as the camera closes to swap-distance/band, default 1.6)\n--bake-tiles-per-frame n (cap atlas tiles baked per frame to spread a one-shot bake over multiple frames and cut the per-frame hitch; grid_size^2 tiles total, default: all in one frame)\n--source <path> (asset to load, default flight helmet)");
        std::process::exit(1);
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
        spacing,
        ambient: if no_ambient { 0.0 } else { ambient_brightness },
        swap,
        swap_distance,
        swap_fade,
        swap_fade_band,
        bake_tiles_per_frame,
    });
    // replaced with the real layout once the scene loads (see setup_scene_after_load); a
    // single identity entry keeps `swap_close`'s resource lookup valid until then.
    commands.insert_resource(ClusterLayout(vec![(Vec3::ZERO, Quat::IDENTITY)]));
}

fn scene_load_check(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut scenes: ResMut<Assets<Scene>>,
    gltf_assets: Res<Assets<Gltf>>,
    mut scene_handle: ResMut<SceneHandle>,
    mut scene_spawner: ResMut<SceneSpawner>,
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
                let scene = scenes.get_mut(gltf_scene_handle).unwrap();

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
    mut scene_spawner: ResMut<SceneSpawner>,
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
        let spread = if settings.cluster > 1 {
            base_radius * (settings.cluster as f32).sqrt() * 0.5
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

        // Record every copy's placement (index 0 is the original at the centre, identity
        // rotation) so `--swap` can rebuild the exact same cluster as real models. The
        // placement uses random jitter/rotation, so it has to be captured here rather than
        // recomputed at swap time.
        let mut cluster_layout: Vec<(Vec3, Quat)> = vec![(Vec3::ZERO, Quat::IDENTITY)];
        if settings.cluster > 1 {
            let gltf = gltf_assets.get(&scene_handle.gltf_handle).unwrap();
            let gltf_scene_handle = gltf
                .scenes
                .get(scene_handle.scene_index)
                .unwrap()
                .clone();
            let mut rng = thread_rng();
            info!(
                "placing {} source copies (spread {spread:.2})",
                settings.cluster
            );
            // Vogel/sunflower spiral: evenly distribute the copies across the disc
            // using the golden angle, with a touch of jitter so it doesn't look
            // mechanical. Index 0 is the original at the centre, so the copies use
            // indices 1..cluster and radius grows as sqrt(i) for uniform area density.
            let golden_angle = PI * (3.0 - 5.0_f32.sqrt());
            let last = (settings.cluster - 1) as f32;
            let jitter = spread * 0.1;
            for i in 1..settings.cluster {
                let fi = i as f32;
                let r = spread * (fi / last).sqrt();
                let angle = fi * golden_angle;
                let offset = Vec3::new(
                    r * angle.cos() + rng.gen_range(-jitter..=jitter),
                    0.0,
                    r * angle.sin() + rng.gen_range(-jitter..=jitter),
                );
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

        let camera = commands.spawn((
            Camera3d::default(),
            Projection::from(projection),
            Transform::from_translation(Vec3::from(aabb.center) + size * Vec3::new(0.5, 0.25, 0.5))
                .looking_at(Vec3::from(aabb.center), Vec3::Y),
            camera_controller,
            // ambient light is now a per-camera component (was a global resource pre-0.16)
            AmbientLight {
                color: Color::WHITE,
                brightness: settings.ambient,
                affects_lightmapped_meshes: true,
            },
            RenderLayers::default().with(1), // we keep imposters off the primary renderlayer to avoid imposterception
        )).id();

        if settings.fxaa {
            info!("Enabling FXAA");
            commands.entity(camera).insert(Fxaa::default());
        }

        // Spawn a default light if the scene does not have one
        if !scene_handle.has_light {
            info!("Spawning a directional light");
            commands.spawn((
                DirectionalLight::default(),
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
    cams: Query<Entity, With<ImposterBakeCamera>>,
    imposters: Query<Entity, With<MeshMaterial3d<Imposter>>>,
    reals: Query<Entity, With<RealModel>>,
    settings: Res<BakeSettings>,
    dummy_indices: Res<DummyIndicesImage>,
    mut probe: ResMut<BakeProbe>,
    // guards against spawning a second batch of imposters while one is already live; reset by `O`.
    mut spawned: Local<bool>,
) {
    if k.just_pressed(KeyCode::KeyO) {
        for entity in cams.iter() {
            commands.entity(entity).despawn();
        }
        let mut cleared = 0;
        for entity in imposters.iter() {
            commands.entity(entity).despawn();
            cleared += 1;
        }
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
    let continuous = if k.just_pressed(KeyCode::Space) {
        false
    } else if k.just_pressed(KeyCode::KeyI) {
        true
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
        let mut camera = ImposterBakeCamera {
            radius: scene_handle.sphere.radius,
            grid_size: settings.grid_size,
            tile_size: settings.tile_size,
            grid_mode: settings.mode,
            continuous,
            multisample: settings.multisample_source,
            max_tiles_per_frame: settings.bake_tiles_per_frame,
            ..Default::default()
        };
        camera.init_target(&mut images);

        let mut rng = thread_rng();
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
        } else {
            AlphaMode::Blend
        };
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
        // --swap-fade: dither the imposter out over [solid, spawn] world distance so the real model
        // swapped in behind it shows through. The imposter billboard renders at the front of the
        // bounding sphere, so fading the imposter (rather than the model) is what lets the model
        // appear through the dither holes. Matches `swap_close`'s spawn/solid thresholds.
        if settings.swap && settings.swap_fade {
            let spawn = settings.swap_distance * scene_handle.sphere.radius;
            let solid = spawn / settings.swap_fade_band;
            data.swap_fade = Vec2::new(solid, spawn);
        }
        let shared_material = MeshMaterial3d(materials.add(Imposter {
            data,
            pixels: camera.target.clone().unwrap(),
            indices: dummy_indices.0.clone(),
            alpha_mode,
            vram_bytes: 0,
        }));

        for _ in 0..settings.count {
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
            commands.spawn((
                shared_mesh.clone(),
                Transform::from_translation(translation + Vec3::from(scene_handle.sphere.center))
                    .with_rotation(Quat::from_euler(
                        EulerRot::XYZ,
                        rotation.x,
                        rotation.y,
                        rotation.z,
                    )),
                shared_material.clone(),
                RenderLayers::layer(1),
                SwapReal::default(),
            ));
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

fn update_lights(
    key_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut DirectionalLight)>,
    mut animate_directional_light: Local<bool>,
) {
    for (_, mut light) in &mut query {
        if key_input.just_pressed(KeyCode::KeyU) {
            light.shadows_enabled = !light.shadows_enabled;
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
                            SceneRoot(scene.clone()),
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
        let on = imps.iter().next().is_some_and(|a| a.1.data.flags & DITHER_FLAG != 0);
        println!("dither: {}", if on { "on" } else { "off" });
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

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed());
    let frame_ms = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed());
    let res = windows
        .single()
        .map(|w| format!("{}x{}", w.physical_width(), w.physical_height()))
        .unwrap_or_default();
    if let (Some(fps), Some(ms)) = (fps, frame_ms) {
        println!("fps: {fps:6.1}  ({ms:5.2} ms/frame cpu+gpu wall)  [{res}]");
    }
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
    for (path, v) in gpu.iter().take(8) {
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

    // atlas is Rg32Uint = 8 bytes/texel, side = tile_size * grid_size
    let side = settings.tile_size * settings.grid_size;
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

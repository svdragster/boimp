// Scatter 2D imposters over a 3D planet, each oriented radially so it "stands up"
// from the surface in any direction.
//
// The interesting part is the per-instance orientation. An imposter is baked ONCE
// in the model's local space (here: an upright tree), then every instance is
// rotated by its own `Transform.rotation` at spawn time. The whole render path -
// the octahedral view lookup, the parallax sample plane and the lit normal - is
// carried through the instance's rotation matrix (`inv_rot` in the shaders), so a
// tree tilted to stand on the side or underside of the planet samples and lights
// itself correctly from every camera angle. Nothing about the bake is planet-aware;
// the planet is purely a spawn-time `Quat::from_rotation_arc(Vec3::Y, radial)`.
//
// Flow: load a tree gltf -> bake it once into an atlas -> despawn the source tree ->
// scatter imposters over a radius-10 sphere -> orbit camera.
//
// Controls: drag left mouse to orbit, scroll to zoom. The planet also auto-spins
// slowly when you're not dragging.

use std::f32::consts::{PI, TAU};

use bevy::{
    asset::LoadState,
    camera::{
        primitives::{Aabb, Sphere as BoundingSphere},
        visibility::RenderLayers,
    },
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    scene::InstanceId,
};
use boimp::{
    bake::BakeState, render::DummyIndicesImage, GridMode, Imposter, ImposterBakeCamera,
    ImposterBakePlugin, ImposterData,
};
use rand::{thread_rng, Rng};

const PLANET_RADIUS: f32 = 10.0;
// billboard half-size of each tree in world units (sphere radius of the imposter).
// 0.3 gives small trees (~0.6 units tall) so a few thousand read as real forests on
// the radius-10 planet.
const TREE_DISPLAY_RADIUS: f32 = 0.3;
// octahedral bake resolution. Spherical mode bakes the full sphere of views, which
// is what a planet needs - the camera sees trees from every side, including the
// horizon silhouette and (on the far limb) their undersides.
const GRID_SIZE: u32 = 15;
const TILE_SIZE: u32 = 64;
const BAKE_MODE: GridMode = GridMode::Spherical;

// display layer for the planet + imposters + light. The source tree we bake from
// lives on the default layer 0 (where ImposterBakeCamera captures it); keeping the
// display on layer 1 means the main camera never sees the source tree sitting at
// the origin while the bake runs.
const DISPLAY_LAYER: usize = 1;

fn main() {
    let mut args = pico_args::Arguments::from_env();
    let tree_count: usize = args.value_from_str("--trees").unwrap_or(4000);
    let forests: usize = args.value_from_str("--forests").unwrap_or(80);
    // surface radius of a forest patch, in world units.
    let forest_radius: f32 = args.value_from_str("--forest-radius").unwrap_or(2.0);
    let source: String = args
        .value_from_str("--source")
        .unwrap_or_else(|_| "models/Tree/scene.gltf".to_string());
    let unused = args.finish();
    if !unused.is_empty() {
        println!("unrecognized arguments: {unused:?}");
        println!("args:\n--trees n (total imposters to scatter, default 4000)\n--forests n (number of forest clusters, default 80)\n--forest-radius f (surface radius of a forest patch in world units, default 2.0)\n--source path (gltf to bake, default models/Tree/scene.gltf)");
        std::process::exit(1);
    }

    App::new()
        .add_plugins(
            // examples accept an arbitrary `--source` path, which may be outside the asset root
            DefaultPlugins.set(AssetPlugin {
                unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                ..Default::default()
            }),
        )
        .add_plugins(ImposterBakePlugin)
        .insert_resource(Config {
            tree_count,
            forests,
            forest_radius,
            source,
        })
        .insert_resource(ClearColor(Color::srgb(0.01, 0.01, 0.03)))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                load_and_spawn_scene,
                start_bake_when_ready,
                scatter_when_baked,
                orbit_camera,
                sun_follows_camera,
            )
                .chain(),
        )
        .run();
}

#[derive(Resource)]
struct Config {
    tree_count: usize,
    forests: usize,
    forest_radius: f32,
    source: String,
}

// drives the load -> bake -> scatter pipeline across frames.
#[derive(Resource)]
struct Pipeline {
    gltf: Handle<Gltf>,
    root: Entity,
    instance: Option<InstanceId>,
    phase: Phase,
    // populated once the source bounding box is known
    sphere: BoundingSphere,
    half_height: f32,
    // populated once the bake camera is spawned
    bake_cam: Option<Entity>,
    target: Option<Handle<Image>>,
}

#[derive(PartialEq)]
enum Phase {
    LoadScene,
    Baking,
    Done,
}

#[derive(Component)]
struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

// the directional light; its direction is slaved to the camera (a headlight) so the
// hemisphere facing the camera is always the lit one - no terminator swings through
// the view as the camera orbits.
#[derive(Component)]
struct Sun;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, config: Res<Config>) {
    // a root for the source tree scene so we can despawn the whole thing after baking.
    let root = commands
        .spawn((Transform::default(), Visibility::default()))
        .id();

    commands.insert_resource(Pipeline {
        gltf: asset_server.load(config.source.clone()),
        root,
        instance: None,
        phase: Phase::LoadScene,
        sphere: BoundingSphere::default(),
        half_height: 0.0,
        bake_cam: None,
        target: None,
    });

    // orbit camera + sun, both on the display layer so they ignore the source tree.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, PLANET_RADIUS * 2.5).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitCamera {
            yaw: 0.0,
            pitch: 0.4,
            distance: PLANET_RADIUS * 2.5,
        },
        // per-camera fill light (AmbientLight is a component in this Bevy version).
        // a generous ambient keeps the night side readable as the camera orbits.
        AmbientLight {
            brightness: 500.0,
            ..default()
        },
        RenderLayers::layer(DISPLAY_LAYER),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 3000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::default(),
        Sun,
        RenderLayers::layer(DISPLAY_LAYER),
    ));
}

// load the gltf, then spawn its first scene as a child of `root` (on the default
// layer 0 so the bake camera captures it).
fn load_and_spawn_scene(
    asset_server: Res<AssetServer>,
    gltfs: Res<Assets<Gltf>>,
    mut scene_spawner: ResMut<SceneSpawner>,
    mut pipeline: ResMut<Pipeline>,
) {
    if pipeline.phase != Phase::LoadScene || pipeline.instance.is_some() {
        return;
    }
    match asset_server.load_state(&pipeline.gltf) {
        LoadState::Loaded => {
            let gltf = gltfs.get(&pipeline.gltf).unwrap();
            let scene = gltf.scenes[0].clone();
            let root = pipeline.root;
            pipeline.instance = Some(scene_spawner.spawn_as_child(scene, root));
        }
        LoadState::Failed(_) => {
            error!("failed to load source gltf");
            std::process::exit(1);
        }
        _ => {}
    }
}

// once the scene's meshes are ready, measure its bounding sphere and kick off a
// one-shot bake centred on it.
fn start_bake_when_ready(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    scene_spawner: Res<SceneSpawner>,
    meshes: Query<(&GlobalTransform, &Aabb), With<Mesh3d>>,
    all_meshes: Query<(), With<Mesh3d>>,
    raw_aabbs: Query<(), (With<Mesh3d>, Without<Aabb>)>,
    mut pipeline: ResMut<Pipeline>,
) {
    if pipeline.phase != Phase::LoadScene {
        return;
    }
    let Some(instance) = pipeline.instance else {
        return;
    };
    if !scene_spawner.instance_is_ready(instance) {
        return;
    }
    // wait until every mesh has had its Aabb computed
    if all_meshes.iter().count() == 0 || raw_aabbs.iter().count() > 0 {
        return;
    }

    // enclose all the scene meshes to get a bounding sphere (and the up half-height,
    // used later to seat the trunk on the surface).
    let mut points = Vec::new();
    for entity in scene_spawner.iter_instance_entities(instance) {
        let Ok((transform, aabb)) = meshes.get(entity) else {
            continue;
        };
        for cx in [-1.0f32, 1.0] {
            for cy in [-1.0f32, 1.0] {
                for cz in [-1.0f32, 1.0] {
                    let corner = Vec3::from(aabb.center)
                        + Vec3::from(aabb.half_extents) * Vec3::new(cx, cy, cz);
                    points.push(transform.transform_point(corner));
                }
            }
        }
    }
    let aabb = Aabb::enclosing(points).unwrap();
    pipeline.sphere = BoundingSphere {
        center: aabb.center,
        radius: aabb.half_extents.length(),
    };
    pipeline.half_height = aabb.half_extents.y;

    let mut camera = ImposterBakeCamera {
        radius: pipeline.sphere.radius,
        grid_size: GRID_SIZE,
        tile_size: TILE_SIZE,
        grid_mode: BAKE_MODE,
        multisample: 8,
        continuous: false,
        ..default()
    };
    camera.init_target(&mut images);
    pipeline.target = Some(camera.target.clone().unwrap());

    let cam = commands
        .spawn((
            camera,
            Transform::from_translation(pipeline.sphere.center.into()),
        ))
        .id();
    pipeline.bake_cam = Some(cam);
    pipeline.phase = Phase::Baking;
    info!("baking tree imposter...");
}

// when the one-shot bake finishes, drop the source tree + bake camera and scatter
// the imposters over the planet.
#[allow(clippy::too_many_arguments)]
fn scatter_when_baked(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<Imposter>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    dummy_indices: Res<DummyIndicesImage>,
    cams: Query<&ImposterBakeCamera>,
    config: Res<Config>,
    mut pipeline: ResMut<Pipeline>,
) {
    if pipeline.phase != Phase::Baking {
        return;
    }
    let Some(cam_entity) = pipeline.bake_cam else {
        return;
    };
    // deferred spawn: the camera may not be queryable on the trigger frame
    let Ok(camera) = cams.get(cam_entity) else {
        return;
    };
    if camera.state != BakeState::Finished {
        return;
    }

    // the source tree and bake camera have done their job.
    commands.entity(pipeline.root).despawn();
    commands.entity(cam_entity).despawn();

    // the planet.
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(PLANET_RADIUS).mesh().ico(6).unwrap())),
        MeshMaterial3d(std_materials.add(StandardMaterial {
            base_color: Color::srgb(0.18, 0.32, 0.16),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::default(),
        RenderLayers::layer(DISPLAY_LAYER),
    ));

    // one shared mesh + material for every tree, so the draws batch/instance. Only
    // the per-entity Transform (position + radial rotation) differs. The display
    // scale is set via the ImposterData `scale` (center_and_scale.w), independent of
    // the radius we baked at - so we can size the trees to the planet here.
    let shared_mesh = Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(0.5))));
    let data = ImposterData::new(
        Vec3::ZERO,
        TREE_DISPLAY_RADIUS,
        GRID_SIZE,
        TILE_SIZE,
        UVec2::ZERO,
        UVec2::splat(TILE_SIZE),
        BAKE_MODE,
        false,
        false,
        false,
        false,
        false,
        1.0,
    );
    let shared_material = MeshMaterial3d(materials.add(Imposter {
        data,
        pixels: pipeline.target.clone().unwrap(),
        indices: dummy_indices.0.clone(),
        alpha_mode: AlphaMode::Mask(0.5),
        vram_bytes: 0,
    }));

    // lift the billboard centre so the trunk base (half_height below the model's
    // sphere centre, scaled into display units) sits on the planet surface.
    let base_lift = TREE_DISPLAY_RADIUS * pipeline.half_height / pipeline.sphere.radius;

    let mut rng = thread_rng();
    // plant one radially-oriented tree at a unit surface direction.
    let plant = |commands: &mut Commands, dir: Vec3, rng: &mut rand::rngs::ThreadRng| {
        let radial = dir.normalize();
        let translation = radial * (PLANET_RADIUS + base_lift);
        // stand the tree up along the surface normal: local +Y -> radial, plus a random
        // spin about the radial axis for variety (free - the octahedral lookup handles
        // arbitrary azimuth).
        let rotation =
            Quat::from_rotation_arc(Vec3::Y, radial) * Quat::from_rotation_y(rng.gen_range(0.0..TAU));
        commands.spawn((
            shared_mesh.clone(),
            shared_material.clone(),
            Transform::from_translation(translation).with_rotation(rotation),
            RenderLayers::layer(DISPLAY_LAYER),
        ));
    };

    // Cluster the trees into forest patches so the planet has dense woods and bare
    // ground between them, rather than an even global sprinkle. Each forest picks a
    // random centre on the sphere, then scatters its trees over a tangent-plane disc
    // (Vogel/sunflower spiral for even area density + jitter so it doesn't look
    // mechanical), projecting each back onto the sphere. ~15% are scattered fully at
    // random as lone trees between the forests.
    let golden_angle = PI * (3.0 - 5.0f32.sqrt());
    let loners = config.tree_count * 15 / 100;
    let clustered = config.tree_count - loners;
    let per_forest = (clustered / config.forests.max(1)).max(1);

    let mut planted = 0;
    for _ in 0..config.forests {
        if planted >= clustered {
            break;
        }
        let center = random_unit(&mut rng);
        let (tangent, bitangent) = tangent_basis(center);
        // vary forest size so they don't all look identical
        let radius = config.forest_radius * rng.gen_range(0.6..1.4);
        let jitter = radius * 0.15;
        let count = per_forest.min(clustered - planted);
        for j in 0..count {
            let f = (j as f32 + 0.5) / count as f32;
            let r = radius * f.sqrt();
            let angle = j as f32 * golden_angle;
            // tangent-plane offset in world units, converted to a unit-sphere offset by
            // dividing by the planet radius, then renormalised onto the surface.
            let ox = r * angle.cos() + rng.gen_range(-jitter..=jitter);
            let oz = r * angle.sin() + rng.gen_range(-jitter..=jitter);
            let dir = center + (tangent * ox + bitangent * oz) / PLANET_RADIUS;
            plant(&mut commands, dir, &mut rng);
        }
        planted += count;
    }

    // lone trees scattered uniformly over the whole sphere
    for _ in 0..(config.tree_count - planted) {
        let dir = random_unit(&mut rng);
        plant(&mut commands, dir, &mut rng);
    }

    pipeline.phase = Phase::Done;
    info!(
        "planted {} trees across {} forests (+{} lone)",
        config.tree_count,
        config.forests,
        config.tree_count - planted
    );
}

// a uniformly-distributed random point on the unit sphere.
fn random_unit(rng: &mut rand::rngs::ThreadRng) -> Vec3 {
    let y: f32 = rng.gen_range(-1.0..1.0);
    let theta = rng.gen_range(0.0..TAU);
    let r = (1.0 - y * y).max(0.0).sqrt();
    Vec3::new(r * theta.cos(), y, r * theta.sin())
}

// an orthonormal tangent/bitangent pair on the sphere surface at `normal` (unit).
fn tangent_basis(normal: Vec3) -> (Vec3, Vec3) {
    let reference = if normal.y.abs() > 0.99 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent = reference.cross(normal).normalize();
    let bitangent = normal.cross(tangent);
    (tangent, bitangent)
}

// point the sun the same way the camera looks (a headlight). A DirectionalLight
// shines along its local -Z, same as the camera's forward, so copying the camera's
// rotation makes the lit hemisphere track the view.
fn sun_follows_camera(
    cam: Query<&Transform, (With<OrbitCamera>, Without<Sun>)>,
    mut sun: Query<&mut Transform, With<Sun>>,
) {
    let Ok(cam) = cam.single() else {
        return;
    };
    for mut t in &mut sun {
        t.rotation = cam.rotation;
    }
}

fn orbit_camera(
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    mut q: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let mut drag = Vec2::ZERO;
    if buttons.pressed(MouseButton::Left) {
        for ev in motion.read() {
            drag += ev.delta;
        }
    } else {
        motion.clear();
    }
    let mut scroll = 0.0;
    for ev in wheel.read() {
        scroll += match ev.unit {
            MouseScrollUnit::Line => ev.y,
            MouseScrollUnit::Pixel => ev.y / 16.0,
        };
    }

    for (mut orbit, mut transform) in &mut q {
        orbit.yaw -= drag.x * 0.005;
        orbit.pitch = (orbit.pitch - drag.y * 0.005).clamp(-1.4, 1.4);
        // gentle idle spin so the planet turns when you're not interacting
        if drag == Vec2::ZERO {
            orbit.yaw += time.delta_secs() * 0.08;
        }
        orbit.distance =
            (orbit.distance - scroll * 1.5).clamp(PLANET_RADIUS + 1.5, PLANET_RADIUS * 6.0);

        let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, orbit.pitch, 0.0);
        *transform = Transform::from_translation(rot * Vec3::new(0.0, 0.0, orbit.distance))
            .looking_at(Vec3::ZERO, Vec3::Y);
    }
}

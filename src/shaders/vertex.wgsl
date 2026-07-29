#import bevy_pbr::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_view_to_world, direction_view_to_world, perspective_camera_near},
}

#ifdef WIND
// `globals.time` drives the sway animation. Guarded so the bake/no-wind pipeline
// doesn't pull in the view-bindings group it doesn't otherwise need.
#import bevy_pbr::mesh_view_bindings::globals;
#endif

#ifdef PREPASS_PIPELINE
    #import bevy_pbr::prepass_io::Vertex;
#else
    #import bevy_pbr::forward_io::Vertex;
#endif

#import boimp::shared::{ImposterVertexOut, normalize_or_zero};
#import boimp::bindings::{imposter_data, sample_uvs_unbounded, grid_weights, sample_positions_from_camera_dir};

// Cheap 3D hash -> [0, 1) (Dave Hoskins, hash13). Local copy of fragment.wgsl's
// `detail_hash13` (shaders can't share plain fns without a shared import); used
// only to derive a stable per-instance tilt angle below.
fn tilt_hash13(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Tilt magnitude of the imposter's depth plane (~8.5°): big enough to break the
// coplanarity that makes adjacent billboards pop, small enough not to distort
// occlusion against terrain/buildings. Rim depth excursion is only
// ± scale * TILT_DEPTH_STRENGTH, safely inside the bounding sphere because the
// plane pivots about its untilted front point (origin left unchanged below).
const TILT_DEPTH_STRENGTH: f32 = 0.15;

// Prevailing wind direction in WORLD space (see WIND below). Projected onto each
// tree's tangent plane so the sway stays horizontal. A normalized diagonal reads as
// a steady cross-breeze over the whole globe; it only vanishes exactly at the two
// antipodal points where it happens to be parallel to the local up.
const WIND_WORLD_DIR: vec3<f32> = vec3<f32>(0.8, 0.0, 0.6);

@vertex
fn vertex(vertex: Vertex) -> ImposterVertexOut {
    var out: ImposterVertexOut;

    var model = mesh_functions::get_world_from_local(vertex.instance_index);

    let center = imposter_data.center_and_scale.xyz;
    let scale = imposter_data.center_and_scale.w;

    let imposter_world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(center, 1.0)).xyz;
    let camera_world_position = position_view_to_world(vec3<f32>(0.0));

    // extract inverse rotation
    let inv_rot = transpose(mat3x3<f32>(
        normalize(model[0].xyz),
        normalize(model[1].xyz),
        normalize(model[2].xyz)
    ));
    // todo: we could pass the instance index instead, and extract in frag shader
    out.inverse_rotation_0c = inv_rot[0];
    out.inverse_rotation_1c = inv_rot[1];
    out.inverse_rotation_2c = inv_rot[2];
    out.base_world_position = imposter_world_position;
    // Deformation-free copy of the centre. Every vertex writes the same value, so
    // the fragment gets a per-primitive constant to pick the octahedral tile from
    // even when WIND shears `base_world_position` below (see shared.wgsl).
    out.view_anchor_world_position = imposter_world_position;

    let up = vec3<f32>(0.0, 1.0, 0.0);
    let back = direction_view_to_world(vec3<f32>(0.0, 0.0, 1.0));
    let right = cross(up, back);
    let up2 = cross(back, right);

    let view_matrix = transpose(mat3x3(
        normalize(right), 
        normalize(up2), 
        back
    ));
    out.world_position = imposter_world_position + (vertex.position * scale * 2.0) * view_matrix;

#ifdef WIND
    // Wind sway: lean the billboard's top toward a prevailing world-space wind,
    // pinned at the trunk base. The unit quad's `vertex.position.y` runs -0.5 (base,
    // seated on the ground) .. +0.5 (crown), so `height` is 0 at the base and 1 at
    // the top — the base never moves, the crown moves most. Applied to
    // `out.world_position` before both the frag-UV sampling and the clip projection
    // below, so the whole billboard shears coherently (a real geometric lean, not a
    // texture slide). The bake path never sets WIND, so this never runs during a bake.
    let wind_amp = imposter_data.wind.x;
    if wind_amp > 0.0 {
        let height = vertex.position.y + 0.5;
        // Local up on the planet (centred at the world origin); project the fixed
        // prevailing wind onto the tangent plane so sway is horizontal and never
        // shoves a tree into or out of the ground.
        let radial = normalize(imposter_world_position);
        let wind_tangent = normalize_or_zero(WIND_WORLD_DIR - radial * dot(WIND_WORLD_DIR, radial));
        // Per-instance phase, hashed from the shared imposter centre so all 4 corners
        // get the SAME phase; keeps neighbouring trees from swaying in lockstep.
        let phase = tilt_hash13(imposter_world_position) * 6.2831853;
        let freq = imposter_data.wind.y;
        // Two octaves → a natural, non-metronomic gust instead of a pure sine.
        let gust = sin(globals.time * freq + phase)
                 + 0.35 * sin(globals.time * freq * 2.7 + phase * 1.7);
        // Height-weighted sway: 0 at the base, full at the crown. Amplitude scales
        // with the billboard size so large and small trees lean by a similar fraction.
        let sway = wind_tangent * (gust * wind_amp * scale * height);
        // Displace BOTH the quad vertex AND the imposter's sample-plane origin. The
        // boimp fragment reconstructs the tree image from the camera ray relative to
        // `base_world_position` (a per-vertex varying), so moving only `world_position`
        // changes the silhouette coverage but NOT the sampled content — the image
        // wouldn't move at all. Shearing `base_world_position` (constant at the base,
        // +sway at the crown) is what actually leans the reconstructed canopy; moving
        // `world_position` by the same amount keeps that leaning crown inside the quad.
        // NOTE: `view_anchor_world_position` is deliberately left un-sheared - tile
        // selection must stay constant across the quad (see shared.wgsl).
        out.world_position += sway;
        out.base_world_position += sway;
    }
#endif

#ifndef VIEW_PROJECTION_ORTHOGRAPHIC
    // project the actual frag position to the furthest of the front plane of the imposter, and the camera near plane * 0.9
    let ray_direction = normalize(camera_world_position - out.world_position);
    let plane_normal = direction_view_to_world(vec3<f32>(0.0, 0.0, 1.0));

    // Normal of the imposter front plane. Normally this is the flat camera-facing
    // `plane_normal`, which makes every imposter a wall parallel to its neighbours:
    // as the camera orbits, two overlapping walls cross through coplanarity and the
    // whole overlap flips occlusion at once (visible "popping"). TILT_DEPTH gives
    // each instance a tiny position-hashed tilt so neighbouring planes are NOT
    // parallel — they intersect along a line and overlap resolves per-pixel across a
    // stable seam. Only this shove plane is tilted: `out.world_position` (which feeds
    // ALL texture/UV sampling in the fragment shader) is untouched, and no frag_depth
    // is written, so early-Z stays on. Non-orthographic branch only — the ortho path
    // is the bake and must never tilt.
#ifdef TILT_DEPTH
    // Per-instance angle hashed from imposter_world_position, which is identical for
    // all 4 vertices, so every corner shares one tilt — no crack between the two
    // triangles. Tilt in the view basis (right/up), pivoting about `back` (= plane_normal).
    let tilt_angle = tilt_hash13(imposter_world_position) * 6.2831853;
    let right_n = normalize(right);
    let up_n = normalize(up2);
    let front_plane_normal = normalize(back + TILT_DEPTH_STRENGTH * (cos(tilt_angle) * right_n + sin(tilt_angle) * up_n));
#else
    let front_plane_normal = plane_normal;
#endif

    // Plane ORIGIN stays on the untilted front point, so the plane only pivots there.
    let imposter_front_plane_origin = imposter_world_position + plane_normal * imposter_data.center_and_scale.w;
    let imposter_front_plane_distance = dot(imposter_front_plane_origin - out.world_position, front_plane_normal) / dot(ray_direction, front_plane_normal);

    let camera_near_plane_origin = camera_world_position - perspective_camera_near() * back;
    let camera_near_plane_distance = dot(camera_near_plane_origin - out.world_position, plane_normal) / dot(ray_direction, plane_normal);

    let plane_distance = min(camera_near_plane_distance * 0.9, imposter_front_plane_distance);

    let point_on_plane = out.world_position + plane_distance * 1.0 * ray_direction;
    out.position = position_world_to_clip(point_on_plane);
#else
    out.position = position_world_to_clip(out.world_position);
#endif

    return out;
}

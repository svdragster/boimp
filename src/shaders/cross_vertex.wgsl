#import bevy_pbr::{
    mesh_functions,
    view_transformations::{position_world_to_clip, position_view_to_world},
}

// `prepass_io::Vertex` and `forward_io::Vertex` declare the SAME attributes at DIFFERENT
// locations (uv is @location(2) in forward but @location(1) in prepass, uv_b is 3 vs 2, normal is
// 1 vs 3), because the prepass omits the attributes it doesn't need. Importing the wrong one
// compiles fine and then silently reads the wrong buffer - uvs arriving as normals. Same pattern
// as vertex.wgsl; keep the two branches in sync with each other, not with what looks tidy.
#ifdef PREPASS_PIPELINE
    #import bevy_pbr::prepass_io::Vertex;
#else
    #import bevy_pbr::forward_io::Vertex;
#endif

#import boimp::cross_bindings::{cross_data, CrossVertexOut};

// ---------------------------------------------------------------------------------------------
// CROSS IMPOSTER vertex stage.
//
// This is a PLAIN TRANSFORM and that is the entire point of the system. Compare vertex.wgsl,
// which for every corner builds a camera-facing billboard basis, shoves the quad forward onto the
// nearer of the imposter front plane and 0.9 * the camera near plane, and (under TILT_DEPTH)
// hashes a per-instance tilt - all so the FRAGMENT can reconstruct an arbitrary view direction by
// ray-marching a sample plane. Here the three quads are fixed in model space and carry baked UVs,
// so none of that machinery is needed:
//
//   * NO billboarding      - the quads are rigid geometry; the "view dependence" comes from which
//                            of the three you happen to be looking at, not from rotating them.
//   * NO near-plane shove  - nothing here is a proxy for a volume, so there is no front plane to
//                            push onto and no reason to survive the camera entering the sphere.
//   * NO tilt              - TILT_DEPTH exists to break coplanarity between neighbouring
//                            camera-facing walls. Cross quads are already non-parallel by
//                            construction (verticals spread over 180 deg plus a horizontal), so the
//                            popping it fixes cannot occur.
//   * NO frag_depth        - the quads rasterise at their true geometric depth already, so
//                            DEPTH_OFFSET's whole reason to exist is gone and early-Z stays on.
//
// If you find yourself adding any of the above back, you have rebuilt the octahedral path and
// should use it instead.
// ---------------------------------------------------------------------------------------------

@vertex
fn vertex(vertex: Vertex) -> CrossVertexOut {
    var out: CrossVertexOut;

    var model = mesh_functions::get_world_from_local(vertex.instance_index);

    let center = cross_data.center_and_scale.xyz;
    let scale = cross_data.center_and_scale.w;

    // Unit-space (+-0.5) quad -> the bake camera's bounding sphere. `scale` is the sphere RADIUS,
    // so the quad must span 2*radius; identical convention to vertex.wgsl, which is what lets one
    // bake feed both imposter systems.
    let local = vertex.position * (scale * 2.0) + center;
    let world_position = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(local, 1.0)).xyz;

    out.world_position = world_position;
    out.position = position_world_to_clip(world_position);

    // Extract the instance's inverse rotation, normalising the columns so a scaled instance
    // transform still yields a pure rotation. This is R^T (R = the instance rotation), passed to
    // the fragment as three columns; the fragment rotates baked LOCAL normals to world with
    // `n * inv_rot` (vec*mat = R.n). Writing `inv_rot * n` instead applies R^-1 and is only
    // correct at identity - see the comment on `pbr_input.N` in fragment.wgsl and on
    // `basis.normal` in bindings.wgsl for the full derivation of the multiplication order.
    var inv_rot = transpose(mat3x3<f32>(
        normalize(model[0].xyz),
        normalize(model[1].xyz),
        normalize(model[2].xyz)
    ));

    // Centre of THIS tree, in the mesh's local space. Non-merged: the material uniform, because one
    // entity is one tree. Overridden under MERGED below.
    var tree_center = center;

#ifdef MERGED
    // ------------------------------------------------------------------------------------------
    // MERGED - one mesh holds N trees, each already scaled/yawed/translated at build time
    // (`cross_imposter_merged_mesh`). The entity Transform is now the PATCH origin, not a tree, so
    // two things the rest of this shader depends on have silently changed meaning:
    //
    //   1. `cross_data.center_and_scale.xyz` is (0,0,0) - it has to be, or the scale/offset above
    //      would move the already-final positions - so "the instance centre" collapses to the patch
    //      origin. ANGLE_FADE would then fade every tree in the patch as if it stood in the middle
    //      of it. The error is the angle between "direction to patch centre" and "direction to this
    //      tree": invisible from far away, and total once the camera is inside the patch, which is
    //      precisely where a forest is looked at.
    //
    //   2. The per-tree yaw is baked into the POSITIONS but not into `model`, so both local->world
    //      normal paths are now short by that rotation: the azimuth -> quad normal derivation below
    //      and, in the fragment, the baked normal map. 90-degree yaws would not expose this (they
    //      are visually identity here - both verticals sample the same side tile), which is exactly
    //      why only arbitrary yaw gives real variation and arbitrary yaw is what desyncs it.
    //
    // Both are repaired from ATTRIBUTE_COLOR = (centre.xyz, yaw). COLOR and not NORMAL: bevy binds
    // COLOR in the forward AND prepass pipelines unconditionally, whereas the prepass only declares
    // `vertex.normal` under NORMAL_PREPASS_OR_DEFERRED_PREPASS - which is also why the mesh carries
    // no normal attribute at all (see the quad-normal note below). COLOR is carried ONLY by merged
    // meshes; the unmerged builder omits it, so `VERTEX_COLORS` is undefined there and this whole
    // block compiles out.
    //
    // NOTE this is deliberately NOT confined to ANGLE_FADE. Fixing only the fade would leave point
    // 2's second half - the fragment's baked normals - unrotated, and a merged forest would then
    // light differently from the identical unmerged one: every tree shaded as though it faced its
    // unyawed direction, which on a directional light is a per-tree brightness error, not a subtle
    // one. Folding the yaw into `inv_rot` fixes the fade and the fragment in one place.
    // ------------------------------------------------------------------------------------------
#ifdef VERTEX_COLORS
    tree_center = vertex.color.xyz;

    let yaw = vertex.color.w;
    let yaw_s = sin(yaw);
    let yaw_c = cos(yaw);
    // R_y(yaw), column-major: e_x -> (c, 0, -s), e_z -> (s, 0, c). Must match `Quat::from_rotation_y`
    // on the CPU, which is what actually rotated the positions - a sign flip here would light every
    // tree as its own mirror image while the geometry stayed put.
    let yaw_rot = mat3x3<f32>(
        vec3<f32>(yaw_c, 0.0, -yaw_s),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(yaw_s, 0.0, yaw_c)
    );
    // The tree's true rotation is R_model * R_yaw, so what the fragment needs is
    // (R_model * R_yaw)^T = R_yaw^T * R_model^T = transpose(yaw_rot) * inv_rot. Consumers are
    // unchanged: `n * inv_rot` still reads as "rotate a local normal to world", it just now
    // includes the yaw that used to live in the entity Transform.
    inv_rot = transpose(yaw_rot) * inv_rot;
#endif
#endif

    out.inverse_rotation_0c = inv_rot[0];
    out.inverse_rotation_1c = inv_rot[1];
    out.inverse_rotation_2c = inv_rot[2];

    // UV_0 = the per-tile uv baked to match the bake camera basis (plan section 2.2). Guarded
    // because bevy only declares `Vertex.uv` when the mesh actually carries ATTRIBUTE_UV_0; the
    // cross mesh always does, so the fallback is dead code that exists purely so a mesh missing
    // the attribute fails as a black imposter instead of a shader compile error.
    var uv = vec2<f32>(0.0);
#ifdef VERTEX_UVS_A
    uv = vertex.uv;
#endif
    out.uv = uv;

    // UV_1 = (tile_id, azimuth). tile_id picks the atlas tile (0 = side, 1 = top) and is a small
    // integer stored as a float, so round with `+ 0.5` rather than trusting an exact float compare.
    // `azimuth` is a genuine f32 angle in radians - NOT an id - and must not be rounded; it is only
    // read by ANGLE_FADE below.
    var tile_id = 0u;
    var azimuth = 0.0;
#ifdef VERTEX_UVS_B
    tile_id = u32(vertex.uv_b.x + 0.5);
    azimuth = vertex.uv_b.y;
#endif
    out.tile = tile_id;

    // Default: fully visible. Kept unconditional so the fragment never has to `#ifdef` its read.
    out.fade = 1.0;

#ifdef ANGLE_FADE
    // ------------------------------------------------------------------------------------------
    // ANGLE FADE - the one quality control this system cannot ship without.
    //
    // The signature cross-quad artifact is the HORIZONTAL quad seen edge-on: from near ground
    // level it collapses to a one-pixel-tall sliver of fully opaque canopy texels, and because it
    // is the same sliver on every instance, a whole forest grows a hard bright horizontal line
    // through it. The verticals do the same thing at 90 deg yaw, just less obtrusively. Fading a
    // quad out as it turns edge-on removes the sliver before it can become a line.
    //
    // WHY abs(): the quads render with cull_mode = None (without that, half the cross vanishes as
    // the camera orbits), so a quad viewed from behind is just as valid as one viewed from the
    // front. dot(view, normal) is -1 there, not +1; without abs() every back-facing quad would sit
    // permanently at fade 0 and the tree would lose half its density from every second direction.
    //
    // WHY the band is (0.20 .. 0.50) and not narrower: the fade has to cover the angles at which
    // a quad ACTUALLY degenerates into a sliver, which is a wider range than it first appears. A
    // camera 19 deg above horizontal - an utterly ordinary third-person view - already puts the
    // horizontal canopy quad at |dot| ~= 0.33. The original (0.05, 0.30) band ended just below
    // that, so smoothstep returned 1.0 and the feature was a measured no-op (0.36% of pixels
    // changed vs fade-off; the current band changes 1.58%). If you narrow this, you are very
    // likely turning the feature off without noticing.
    //
    // The upper limit is 0.707: with TWO orientations (the default square cross), 45 deg of yaw puts
    // BOTH verticals exactly there, and they are what carries the tree's silhouette at that angle.
    // Push `end` toward 0.707 and they dim together and the whole forest visibly thins. So the
    // usable window is roughly 0.3 .. 0.6 for `end`. More orientations RAISE that ceiling - with
    // three, the worst-case azimuth is 30 deg off the best-aligned plane, i.e. |dot| >= 0.87 - so a
    // band tuned for two orientations stays safe for three but no longer sits as close to the edge.
    //
    // Computed here, per vertex, and not per fragment: it depends only on the instance centre and
    // the quad's own normal, both of which are constant over the primitive, so a fragment-side
    // evaluation would be the same value computed thousands of times over.
    // ------------------------------------------------------------------------------------------

    // Local-space normal of this quad, DERIVED from UV_1 rather than read from `vertex.normal`.
    // bevy's `prepass_io::Vertex` only declares the normal under
    // NORMAL_PREPASS_OR_DEFERRED_PREPASS - so in the shadow pass (and in a plain depth prepass)
    // `vertex.normal` does not exist and reading it is a compile error, not a wrong value. UV_1 is
    // declared by every pipeline.
    //
    // Because this is the ONLY consumer of a quad normal anywhere in the cross path (the fragment's
    // shading normal comes from the atlas, not from geometry), the mesh builders emit no
    // Mesh::ATTRIBUTE_NORMAL at all - it would be 12 dead bytes on every vertex, and merged patches
    // duplicate the vertex data per tree.
    //
    // Two cases, both keyed on data the mesh already had to carry (keep in sync with `cross_quads`
    // in cross.rs):
    //
    //   * the horizontal quad is the ONLY quad that samples tile 1, so the tile id identifies it
    //     outright -> +Y;
    //   * every other quad is vertical, and `uv_b.y` is its plane's AZIMUTH in radians ->
    //     (sin, 0, cos), which is the +Z reference plane yawed by that angle about +Y.
    //
    // This replaced a hardcoded 3-entry `quad_id -> normal` table, and the reason is that the table
    // could not express an arbitrary number of orientations: adding a third plane direction (see
    // CrossLayout::orientations - the hex layout puts them at 0/60/120 deg) needed a new id, a new
    // vector and a matching edit on the CPU side. An angle needs none of that, and it is exactly
    // equivalent for the layouts that already shipped: azimuth 0 -> (0,0,1) = the old V0 entry,
    // PI/2 -> (1,0,0) = V1, tile 1 -> (0,1,0) = H.
    //
    // Note the normal is per-PLANE, not per-quad: an outer plane is a translation of its
    // orientation's reference plane along that normal, so it carries the same azimuth and fades in
    // lockstep with its parallel family - correct, since they go edge-on at the same instant.
    let n_vertical = vec3<f32>(sin(azimuth), 0.0, cos(azimuth));
    let n_horizontal = vec3<f32>(0.0, 1.0, 0.0);
    let local_quad_normal = select(n_vertical, n_horizontal, tile_id == 1u);

    // local -> world with R; inv_rot is R^T, so `v * inv_rot` is R.v (see the derivation above).
    // Under MERGED `inv_rot` already carries this tree's yaw, so this evaluates to
    // R_model * R_yaw * local_quad_normal - i.e. the local normal IS rotated by the per-tree yaw
    // before the instance rotation, which is the whole point: the table above describes the quad as
    // authored, and the merged mesh yawed it.
    let quad_normal_world = normalize(local_quad_normal * inv_rot);

    // View direction from the TREE CENTRE, not from this vertex: the fade must be one decision per
    // quad. Using the vertex position would make `fade` vary across the quad and turn the dithered
    // cutout in the fragment into a gradient wipe across each billboard. `tree_center` is the
    // material's centre normally and this tree's COLOR.xyz under MERGED - using the uniform there
    // would fade all N trees of a patch as if they stood on the patch origin.
    let instance_center_world = mesh_functions::mesh_position_local_to_world(model, vec4<f32>(tree_center, 1.0)).xyz;
    let camera_world_position = position_view_to_world(vec3<f32>(0.0));
    // NB: this is the perspective view vector. Under an orthographic projection the correct vector
    // is the fixed view axis, not centre->eye; ANGLE_FADE is never enabled on the (orthographic)
    // bake path, and the ortho error only shows at wide off-axis angles, so this is left simple.
    let view_dir = normalize(camera_world_position - instance_center_world);

    out.fade = smoothstep(cross_data.angle_fade.x, cross_data.angle_fade.y, abs(dot(view_dir, quad_normal_world)));
#endif

    return out;
}

#import bevy_pbr::{
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::{STANDARD_MATERIAL_FLAGS_UNLIT_BIT, STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT},
    view_transformations::{direction_view_to_world, position_view_to_world, position_world_to_clip}
}

#ifdef PREPASS_PIPELINE
 #import bevy_pbr::prepass_io::FragmentOutput;
#else
 #import bevy_pbr::forward_io::FragmentOutput;
#endif

#import boimp::shared::{ImposterVertexOut, unpack_pbrinput, weighted_props};
#import boimp::bindings::{imposter_data, sample_positions_from_camera_dir, sample_uvs_unbounded, sample_tile, sample_tile_material};

#ifdef COVERAGE_PRESERVE
// How aggressively to scale averaged alpha back up per texel of minification.
// Higher = denser/heavier distant foliage; too high re-introduces hard edges.
const COVERAGE_SCALE: f32 = 0.02;
// Half-width of the soft alpha cutoff band. Wider = softer silhouette (more for
// A2C/MSAA to dither), narrower = crisper.
const COVERAGE_BAND: f32 = 0.4;
#endif

#ifdef DETAIL_FADE
// Texels of minification over which detail fully fades to a smooth blob. Larger =
// detail survives further out; smaller = trees flatten sooner.
const DETAIL_FADE_TEXELS: f32 = 2.0;
// Max fraction of albedo saturation removed at full fade (1.0 = greyscale).
const DETAIL_DESAT: f32 = 0.4;

// --- World-space micro-noise (re-injects macro "clump" variance) ---
// The plain desaturate-toward-luma fade above kills sub-pixel sparkle but also
// flattens the forest into a foggy blob. We claw the *look* of detail back by
// modulating brightness with a band-limited value noise sampled in WORLD space.
// Because it is anchored to world coordinates it moves rigidly with the planet
// (no swimming when the camera orbits), and because it is band-limited value
// noise (not a per-texel hash) its maximum spatial frequency is bounded, so it
// cannot itself reintroduce aliasing - it replaces sub-pixel leaf chaos with
// stable, controllable macro-pixel clumps.
// Cells per world unit. Higher = smaller/tighter clumps. Tune to model scale:
// aim for a handful of clumps across one tree canopy.
const DETAIL_NOISE_SCALE: f32 = 0.2;
// Brightness multiplier range driven by the noise (1.0 = unchanged). Wider =
// more contrasty clumps. Applied as mix(MIN, MAX, noise).
const DETAIL_NOISE_MIN: f32 = 0.4;
const DETAIL_NOISE_MAX: f32 = 2.0;
// Skews the noise distribution. > 1 pushes more of the field toward 0 (dark),
// so bright clumps stay tight and the dark gaps between them grow. 1.0 = even.
const DETAIL_NOISE_BIAS: f32 = 0.6;

// Direction the shading normal flattens toward at full fade. Must be
// VIEW-INDEPENDENT, otherwise the canopy lighting swings bright/dark as the
// camera orbits (flattening toward the view vector makes the far side fall into
// the sun's shadow). World-up shades the blob like an upward-facing canopy -
// consistent from every azimuth. On a spherical planet, swap for the local
// up (normalize(base_world_position - planet_center)).
const DETAIL_FADE_UP: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

// --- High-variance dynamic contrast ("anti-fog") ---
// Linear spatial averaging pulls dark shadow gaps and bright leaf highlights
// toward a muddy mid-grey/green. Push the faded colour back away from that
// middle ground (gamma > 1 deepens darks, the gain lifts highlights) so the eye
// reads distinct clumps of trees instead of flat fog.
const DETAIL_CONTRAST: f32 = 2.0;
const DETAIL_CONTRAST_GAIN: f32 = 1.5;

// Cheap 3D hash -> [0, 1) (Dave Hoskins, hash13).
fn detail_hash13(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Band-limited 3D value noise in [0, 1): trilinear blend of hashed lattice
// corners with a smoothstep fade. Band-limiting is what keeps this from
// aliasing - the highest frequency present is fixed by the lattice spacing.
fn detail_value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = detail_hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = detail_hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = detail_hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = detail_hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = detail_hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = detail_hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = detail_hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = detail_hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}
#endif

// Static interleaved gradient noise in [0, 1) keyed only on screen position. The
// pattern is fixed per pixel (no temporal term), so it does not shimmer without a
// temporal resolve - at intermediate view angles the stochastic per-fragment tile
// selection shows up as a stable dither/stipple instead of a continuous cross-fade.
// Used both for DITHERED tile selection and the --swap-fade distance dissolve, so it
// is defined unconditionally.
fn dither_noise(frag_coord: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * frag_coord.x + 0.00583715 * frag_coord.y));
}

@fragment
fn fragment(in: ImposterVertexOut) -> FragmentOutput {
    var out: FragmentOutput;

    let inv_rot = mat3x3(
        in.inverse_rotation_0c,
        in.inverse_rotation_1c,
        in.inverse_rotation_2c,
    );

    let camera_world_position = position_view_to_world(vec3<f32>(0.0));
#ifdef VIEW_PROJECTION_ORTHOGRAPHIC
    let back_vec = direction_view_to_world(vec3<f32>(0.0, 0.0, 1.0));
#else
    let back_vec = camera_world_position - in.base_world_position;
#endif

    let back = normalize(back_vec);

    let samples = sample_positions_from_camera_dir(back * inv_rot);
    let weights = samples.tile_weights;

    // texel footprint of this fragment in the atlas, used as a mip/LOD level to
    // suppress minification sparkle on distant imposters. computed once here in
    // uniform control flow (valid for dpdx/dpdy) from a fixed reference tile, so
    // it is independent of the per-pixel dither tile choice made below.
    let ref_uv = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[0]).xy;
    let footprint = max(length(dpdx(ref_uv)), length(dpdy(ref_uv))) * f32(imposter_data.base_tile_size);

    // DETAIL_FADE amount for this fragment, computed once here in uniform control flow.
    // Passed into sample_tile_material so the minification box filter spends fewer taps
    // where detail is being faded out anyway, and reused for the fade effect below.
#ifdef DETAIL_FADE
    let mip_fade = clamp((footprint - 1.0) / DETAIL_FADE_TEXELS, 0.0, 1.0);
#else
    let mip_fade = 0.0;
#endif

#ifdef DITHERED
    // Screen-space dither (stochastic sampling). Rather than blending the 2-3
    // nearest octahedral tiles on every fragment - which leaves a permanent
    // "ghost"/double-image at intermediate view angles - each fragment picks
    // exactly ONE tile, chosen with probability equal to that tile's barycentric
    // weight. A temporal resolve (TAA) averages the dither back into a smooth
    // cross-fade. Bonus: only one tile is sampled, so this is also cheaper than
    // the 3-tile blend.
    let noise = dither_noise(in.position.xy);

#ifdef GRID_HORIZONTAL
    let chosen_index = select(samples.tile_indices[1], samples.tile_indices[0], noise < weights.x);
#else
    var chosen_index = samples.tile_indices[2];
    if noise < weights.x {
        chosen_index = samples.tile_indices[0];
    } else if noise < weights.x + weights.y {
        chosen_index = samples.tile_indices[1];
    }
#endif

    let uv = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, chosen_index);
    let props_final = sample_tile_material(uv, chosen_index, vec2(0.0), footprint, mip_fade);
#else
    let uv_a = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[0]);
    let uv_b = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[1]);

    let props_a = sample_tile_material(uv_a, samples.tile_indices[0], vec2(0.0), footprint, mip_fade);
    let props_b = sample_tile_material(uv_b, samples.tile_indices[1], vec2(0.0), footprint, mip_fade);

#ifndef GRID_HORIZONTAL
    let uv_c = sample_uvs_unbounded(in.base_world_position, in.world_position, inv_rot, samples.tile_indices[2]);
    let props_c = sample_tile_material(uv_c, samples.tile_indices[2], vec2(0.0), footprint, mip_fade);
#endif

    let props_ab = weighted_props(props_a, props_b, weights.x / max(weights.x + weights.y, 0.0001));
#ifndef GRID_HORIZONTAL
    let props_final = weighted_props(props_ab, props_c, (weights.x + weights.y) / (weights.x + weights.y + weights.z));
#else
    let props_final = props_ab;
#endif
#endif

    var coverage = props_final.rgba.a;
#ifdef COVERAGE_PRESERVE
    // Coverage-preserving alpha for distant alpha-tested foliage (the trick used
    // for Ghost of Tsushima's far trees). As an imposter minifies, the box-filter
    // above averages its thin leaf/branch alpha down toward zero, so foliage
    // thins out and flickers. Scale the averaged coverage back up in proportion
    // to the texel footprint to preserve visual density, then run it through a
    // soft cutoff so the silhouette hands A2C/MSAA a *fractional* coverage to
    // dither instead of a hard binary edge. `footprint` is computed above in
    // uniform control flow, so reading it here (under the dither branch) is safe.
    // NB: only has a visible effect with AlphaToCoverage (--a2c) or Blend - under
    // an opaque/Mask blend state the fractional coverage is discarded.
    let mip = max(footprint - 1.0, 0.0);
    let scaled_a = coverage * (1.0 + mip * COVERAGE_SCALE);
    let soft_a = smoothstep(0.5 - COVERAGE_BAND, 0.5 + COVERAGE_BAND, scaled_a);
    // ramp the effect in over the first texel of minification so near/medium
    // range (mip ~= 0) is left untouched
    coverage = mix(coverage, soft_a, clamp(mip, 0.0, 1.0));
#endif

    if coverage < 0.01 {
        discard;
        // out.color = vec4(0.0, 0.2, 0.0, 0.2);
        // return out;
    }

    // --swap-fade (examples/dynamic.rs): stochastically dissolve the imposter as the camera closes
    // in, so the real model swapped in behind it shows through the dither holes. swap_fade =
    // (near, far) world distances: full imposter at/beyond `far`, fully gone at/within `near`. The
    // imposter billboard sits at the *front* of the bounding sphere and so would always occlude the
    // real model - dithering the imposter (not the model) is what makes the cross-fade work with
    // depth. (0,0) disables it, which is the default for every non-swap path.
    if imposter_data.swap_fade.y > 0.0 {
        let cam_dist = distance(in.world_position, position_view_to_world(vec3<f32>(0.0)));
        let keep = clamp(
            (cam_dist - imposter_data.swap_fade.x)
                / max(imposter_data.swap_fade.y - imposter_data.swap_fade.x, 0.0001),
            0.0,
            1.0,
        );
        // offset the noise coords so this dither doesn't correlate with the DITHERED tile pick
        if dither_noise(in.position.xy + vec2<f32>(37.0, 17.0)) >= keep {
            discard;
        }
    }

// we can discard based on actual depth if we have the depth prepass data
#ifdef DEPTH_PREPASS
#ifndef PREPASS_PIPELINE
    let existing_depth_ndc = bevy_pbr::prepass_utils::prepass_depth(in.position, 0u);
    let imposted_ndc = position_world_to_clip(in.world_position + back * props_final.depth * imposter_data.center_and_scale.w);
    let imposter_depth_ndc = imposted_ndc.z / imposted_ndc.w;

    if imposter_depth_ndc < existing_depth_ndc {
        // out.color = vec4<f32>(0.0, 0.5, 0.0, 0.5);
        // return out;
        discard;
    }
#endif
#endif

    var pbr_input = unpack_pbrinput(props_final, in.position);
    pbr_input.N = inv_rot * normalize(pbr_input.N);
    pbr_input.world_normal = pbr_input.N;

    pbr_input.material.base_color.a = coverage * imposter_data.alpha;

#ifdef DETAIL_FADE
    // Distance detail fade. The box-filter above averages albedo spatially, but
    // two high-frequency sources still sparkle on far imposters: the per-texel
    // baked normals (each drives a different lighting/specular response) and the
    // raw albedo contrast between neighbouring texels. As the texel footprint
    // grows, blend all of it toward a smooth low-frequency blob:
    //   - flatten the normal toward the billboard facing direction, so the normal
    //     map stops modulating lighting at distance,
    //   - push roughness toward fully-rough (specular antialiasing), and
    //   - desaturate albedo toward its luminance to calm colour flicker.
    // `footprint` is computed in uniform control flow above, so this is safe.
    let fade = mip_fade; // same curve, computed once next to `footprint`
    // Flatten toward a view-INDEPENDENT direction (world-up), not `back`/the view
    // vector - otherwise the faded canopy faces the camera and the hemisphere
    // pointing away from the sun reads as one giant shadow when you orbit behind.
    pbr_input.N = normalize(mix(pbr_input.N, DETAIL_FADE_UP, fade));
    pbr_input.world_normal = pbr_input.N;
    pbr_input.material.perceptual_roughness =
        mix(pbr_input.material.perceptual_roughness, 1.0, fade);
    let luma = dot(pbr_input.material.base_color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    var faded_rgb = mix(pbr_input.material.base_color.rgb, vec3<f32>(luma), fade * DETAIL_DESAT);

    // Anti-fog contrast: push the muddy averaged colour away from mid-grey,
    // ramped in by `fade` so near imposters keep their true albedo.
    let high_contrast = pow(max(faded_rgb, vec3<f32>(0.0)), vec3<f32>(DETAIL_CONTRAST)) * DETAIL_CONTRAST_GAIN;
    faded_rgb = mix(faded_rgb, high_contrast, fade);

    // World-anchored, band-limited brightness clumps. Sampled from the fragment
    // world position so the pattern is locked to the planet and never swims.
    let detail_noise = pow(detail_value_noise(in.world_position * DETAIL_NOISE_SCALE), DETAIL_NOISE_BIAS);
    let noise_gain = mix(1.0, mix(DETAIL_NOISE_MIN, DETAIL_NOISE_MAX, detail_noise), fade);
    faded_rgb *= noise_gain;

    pbr_input.material.base_color = vec4<f32>(faded_rgb, pbr_input.material.base_color.a);
#endif

#ifdef PREPASS_PIPELINE
    #ifdef NORMAL_PREPASS
        out.normal = vec4<f32>(pbr_input.N, 0.0);
    #endif
    // we don't support MOTION_VECTOR or DEFERRED
    #ifdef DEPTH_CLAMP_ORTHO
        out.frag_depth = in.position.z;
    #endif
#else
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }

    pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;

    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    // out.color = clamp(out.color, vec4<f32>(0.2, 0.0, 0.0, 0.2), vec4<f32>(1.0, 1.0, 1.0, 0.7));

    return out;
}

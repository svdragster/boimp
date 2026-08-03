#import bevy_pbr::{
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_types::{PbrInput, STANDARD_MATERIAL_FLAGS_UNLIT_BIT, STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT},
}

#ifdef CHEAP_LIGHTING
// Only pulled in for the cheap path. `lights` gives the directional light array + the ambient
// term, `view` gives the exposure multiplier that `apply_pbr_lighting` applies to everything
// except emissive - without it the cheap path would be exposure-independent and would not match
// the rest of the scene at all.
#import bevy_pbr::mesh_view_bindings::{lights, view};
#endif

#ifdef PREPASS_PIPELINE
    #ifdef PREPASS_FRAGMENT
    // `prepass_io::FragmentOutput` is declared INSIDE `#ifdef PREPASS_FRAGMENT` in bevy 0.19
    // (prepass_io.wgsl:81-99), so the import has to sit behind the same def - see the dual entry
    // point at the bottom of this file for why we are compiled without it at all.
    #import bevy_pbr::prepass_io::FragmentOutput;
    #endif
#else
 // No DEPTH_OFFSET equivalent here, so the stock colour-only output is enough. The octahedral
 // fragment shader has to shadow bevy's FragmentOutput with a frag_depth-carrying copy because its
 // billboard rasterises at a flat camera-facing plane depth that has nothing to do with the
 // silhouette it draws. Cross quads are real geometry at real depths, so they occlude correctly
 // out of the box - and not writing frag_depth is what keeps early-Z alive, which matters a lot
 // more here (three overlapping quads per instance) than it does for a single billboard.
 //
 // Unconditional in bevy, unlike the prepass one, so no def guard is needed.
 #import bevy_pbr::forward_io::FragmentOutput;
#endif

#import boimp::shared::{unpack_pbrinput, UnpackedMaterialProps};
#import boimp::cross_bindings::{cross_data, cross_single_sample, CrossVertexOut};

// Static interleaved gradient noise in [0, 1) keyed only on screen position. The pattern is fixed
// per pixel (no temporal term), so it does not shimmer without a temporal resolve - the stochastic
// per-fragment decision below shows up as a stable dither/stipple instead of crawling. Copied
// verbatim from fragment.wgsl: naga_oil modules cannot share a plain fn without a shared import,
// and giving `boimp::shared` a screen-space noise helper would drag it into the bake shaders too.
fn dither_noise(frag_coord: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * frag_coord.x + 0.00583715 * frag_coord.y));
}

#ifdef CHEAP_LIGHTING
// 1/PI, the Lambertian normalisation factor. Filament's (and therefore bevy's) diffuse BRDF
// `Fd_Burley` reduces to exactly `1/PI` when the Schlick scatter terms are dropped, so keeping the
// factor here means switching CHEAP_LIGHTING on changes the *quality* of the shading without also
// changing its overall brightness by 3.14x - which would otherwise be mistaken for a bug.
const CHEAP_INV_PI: f32 = 0.3183098861837907;

// -------------------------------------------------------------------------------------------
// CHEAP_LIGHTING: Lambert + ambient against the FIRST directional light only.
//
// WHY this exists: this whole system trades geometry for fragment cost, and once the atlas
// sampling is down to one load (plan section 1), `apply_pbr_lighting` is what is left holding the
// fragment. On the reference 7900 XTX the octahedral perf ladder already prices all shading at
// ~0.42 ms (levels 1-2 in fragment.wgsl); on the iGPU this system targets, that share is
// relatively much larger, because the ALU:bandwidth ratio is worse and `apply_pbr_lighting` is
// almost pure ALU - Burley diffuse, GGX specular, the full clusterable-object loop, cascade
// selection and shadow-map PCF per light, environment map IBL. Foliage sees essentially none of
// that: it is rough, non-metallic, sub-pixel-detailed, and already carries baked shading in its
// albedo. Plan section 6.4 calls this the single biggest remaining iGPU lever.
//
// WHAT IT APPROXIMATES AWAY (all deliberate, all visible if you look for it):
//   * SHADOWS. No cascade lookup, no PCF, no shadow map sample. Cross imposters are a distance
//     LOD, so they are mostly beyond the last useful cascade anyway - but a cross imposter
//     standing in a near shadow will be lit as if it were not.
//   * SPECULAR, and therefore all Fresnel/GGX/env-map response. Foliage is nearly Lambertian; the
//     bigger loss is that metallic surfaces baked into a cross imposter go flat.
//   * ALL LIGHTS BUT ONE. Only `directional_lights[0]`. Point/spot/rect lights and the entire
//     clustered-light loop are skipped. `lights.directional_lights` is a fixed-size uniform array
//     so index 0 is always readable, but it is only *meaningful* when `n_directional_lights > 0`,
//     hence the guard. For the sun-lit outdoor scenes this system is for, light 0 is the sun.
//   * IBL / light probes / lightmaps / SSAO. Ambient collapses to a flat `lights.ambient_color`
//     term instead of `ambient_light()`'s EnvBRDF-approximated hemisphere, which drops the
//     view-dependent ambient specular. This is the closest available honest reading: the ambient
//     term is a plain uniform, so no fallback or reconstruction is needed.
//
// WHAT IT KEEPS, because dropping it would make the cheap path look like a different material
// rather than a cheaper one: the metallic diffuse split, `view.exposure` (applied to direct +
// ambient exactly as pbr_functions does at its "Total light" line), and the emissive term with
// pbr's `mix(1.0, exposure, emissive.a)` rule. Fog and tonemapping are NOT dropped either - they
// live in `main_pass_post_lighting_processing`, which still runs at the call site.
// -------------------------------------------------------------------------------------------
fn cheap_lighting(pbr: PbrInput) -> vec4<f32> {
    let alpha = pbr.material.base_color.a;
    // Same diffuse/specular energy split pbr_functions makes; on a metal the diffuse lobe
    // disappears, and without this a baked metallic texel would read brighter here than under
    // full PBR rather than just duller.
    let diffuse_color = pbr.material.base_color.rgb * (1.0 - pbr.material.metallic);

    var direct = vec3<f32>(0.0);
    if lights.n_directional_lights > 0u {
        // `direction_to_light` points FROM the surface TO the light (bevy's convention), so this
        // is a plain N.L with no negation.
        let ndotl = saturate(dot(pbr.N, lights.directional_lights[0].direction_to_light));
        direct = diffuse_color * (CHEAP_INV_PI * ndotl) * lights.directional_lights[0].color.rgb;
    }

    let ambient = diffuse_color * lights.ambient_color.rgb;

    // `unpack_pbrinput` only sets `material.emissive` when the baked texel carried the EMISSIVE
    // flag; otherwise it is the default (0,0,0,1) and this whole term is zero.
    let emissive = pbr.material.emissive.rgb * alpha * mix(1.0, view.exposure, pbr.material.emissive.a);

    return vec4<f32>(view.exposure * (direct + ambient) + emissive, alpha);
}
#endif

// The result of the shared per-fragment work below. `props` is the raw unpacked texel, untouched;
// `alpha` is the material-scaled alpha that survived the discard test. They are kept SEPARATE
// rather than folding `alpha` back into `props.rgba.a`, because `unpack_pbrinput` copies
// `props.rgba` into `material.emissive` as well, where the `.a` channel is the
// `mix(1.0, exposure, emissive.a)` exposure-weight - not a coverage value. Scaling it by
// `cross_data.alpha` would silently make instance fade-out dim the emissive exposure weighting too.
struct CrossSurface {
    props: UnpackedMaterialProps,
    alpha: f32,
}

// ----------------------------------------------------------------------------------------------
// Everything every entry point must do, in one place: atlas addressing, the single texture load,
// the ANGLE_FADE stipple discard and the alpha-test discard.
//
// This is shared by BOTH fragment entry points below - including the void, depth-only one - which
// is the whole point: the discards are what carve the alpha-tested silhouette, and a depth prepass
// or shadow pass that skipped them would write the depth of a solid rectangle. `discard` inside a
// helper is ordinary WGSL and terminates the invocation exactly as it would inline.
// ----------------------------------------------------------------------------------------------
fn cross_surface(in: CrossVertexOut) -> CrossSurface {
    // ------------------------------------------------------------------------------------------
    // Atlas addressing. The whole thing, start to finish.
    //
    // The atlas is a 2-tile horizontal strip: tile 0 = side view, tile 1 = top view, both trimmed
    // by ONE shared rect. `in.tile` is flat-interpolated, so `tile_origin` is exact and constant
    // over the quad. `in.uv` is a plain interpolated attribute; contrast `sample_uvs_unbounded`,
    // which intersects the camera ray with a per-tile sample plane per fragment just to obtain the
    // same two numbers.
    //
    // NOT DONE HERE, and each omission is a deliberate quality-for-speed trade:
    //   * NO parallax. The octahedral path re-fetches at `uv + depth * dd` to fake volume from a
    //     flat billboard. A cross quad is fixed geometry sampled straight on, so there is no
    //     view-dependent shift to correct - and adding one would double the fetch count, which is
    //     the entire budget of this shader.
    //   * NO footprint / mip taps. That needs dpdx/dpdy and an O(taps^2) loop. Distant cross
    //     imposters will therefore sparkle (plan section 6.2); the intended fixes are
    //     AlphaToCoverage + MSAA first, then a MIP_TAP 2x2 box, then a real filterable RGBA8
    //     atlas - not a return to per-fragment software mips.
    //   * NO tile blending or dithered tile choice. Which tile a fragment reads is a property of
    //     the quad it lives on, not of the view direction, so there is nothing to blend between.
    // ------------------------------------------------------------------------------------------
    let tile_origin = vec2<f32>(f32(in.tile) * f32(cross_data.packed_tile_size.x), 0.0);
    let bounds_min = tile_origin;
    let bounds_max = bounds_min + vec2<f32>(cross_data.packed_tile_size);
    let coords = tile_origin - vec2<f32>(cross_data.packed_tile_offset)
        + in.uv * f32(cross_data.base_tile_size);

    // The one texture load (two when indexed). Everything above this line is ALU on interpolants.
    let props = cross_single_sample(coords, bounds_min, bounds_max);

#ifdef ANGLE_FADE
    // ANGLE_FADE hands us a FRACTIONAL weight, but the alpha modes this system is meant to run
    // under - Mask and AlphaToCoverage - cannot blend one: Mask thresholds it away entirely, and
    // Blend is out of the question here because three overlapping quads per instance in a
    // depth-sorted transparent pass is exactly the pathological case (plan section 9.2). So spend
    // the weight as a per-pixel probability instead: stipple the quad out with the same static
    // interleaved-gradient noise fragment.wgsl uses. Stays in the opaque pass, keeps depth writes,
    // keeps early-Z, and resolves to a smooth fade under TAA/MSAA.
    //
    // `in.fade` is constant across the quad (computed from the instance centre in the vertex
    // stage), so this is a uniform threshold against a screen-space pattern - a dither, not a
    // gradient wipe.
    if dither_noise(in.position.xy) >= in.fade {
        discard;
    }
#endif

    // Alpha test. Must run in the PREPASS too: the depth prepass and the shadow pass have to see
    // the same cutout silhouette as the forward pass, or the imposter casts the shadow of a solid
    // rectangle and self-occludes against its own prepass depth.
    let a = props.rgba.a * cross_data.alpha;
    if a < 0.01 {
        discard;
    }

    var surface: CrossSurface;
    surface.props = props;
    surface.alpha = a;
    return surface;
}

// ==============================================================================================
// ENTRY POINTS.
//
// Two of them, mirroring bevy's own `pbr_prepass.wgsl` (0.19, lines 23-31 and 147-151), because
// `prepass_io::FragmentOutput` only EXISTS when `PREPASS_FRAGMENT` is defined.
//
// Bevy pushes `PREPASS_FRAGMENT` only for the normal/motion-vector/deferred prepasses (and for
// unclipped-depth emulation), but it still *builds a fragment shader at all* whenever the mesh key
// has `MAY_DISCARD` and the material supplies a prepass fragment shader (prepass/mod.rs:563).
// `CrossImposter` does supply one and is meant to run as `AlphaMode::Mask` / `AlphaToCoverage`,
// both of which set `MAY_DISCARD` - so a plain depth-only prepass, and every shadow-caster pass,
// compiles this file with `PREPASS_FRAGMENT` UNDEFINED. A single struct-returning entry point
// would fail pipeline specialisation there (this is `examples/dynamic.rs --mask` plus the `U`
// shadow toggle, not a hypothetical).
//
// The void entry point below is not a stub: the discards inside `cross_surface` are the entire
// reason bevy wants a fragment shader in a depth-only pass. They are what cuts the alpha-tested
// silhouette into the depth buffer, and therefore into the shadow map.
// ==============================================================================================

#ifdef PREPASS_PIPELINE

#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(in: CrossVertexOut) -> FragmentOutput {
    var out: FragmentOutput;

    // R^T of the instance rotation, reassembled from the three varying columns (see cross_vertex).
    let inv_rot = mat3x3(
        in.inverse_rotation_0c,
        in.inverse_rotation_1c,
        in.inverse_rotation_2c,
    );

    // PERF MEASUREMENT ladder level 2: flat output, ZERO atlas fetches, no discard. See the
    // forward entry point for the full rationale; here it just means "write the attachments
    // without touching the atlas". Deliberately not coverage-preserving.
    if cross_data.debug_level >= 2u {
    #ifdef NORMAL_PREPASS
        out.normal = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    #endif
    #ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
        out.frag_depth = in.position.z;
    #endif
        return out;
    }

    let surface = cross_surface(in);

    // Built here rather than just rotating the raw normal, so the normal this pass writes is
    // produced by exactly the same code as the forward pass's - including `V`, which
    // TWO_SIDED_LIGHTING needs and which only `unpack_pbrinput` computes. A prepass normal that
    // disagreed with the forward normal is a shading seam waiting to happen.
    var pbr_input = unpack_pbrinput(surface.props, in.position);

    // Baked normals are in the model's LOCAL frame; rotate to world with R. inv_rot = R^T, so
    // R.n = `n * inv_rot` (vec*mat). `inv_rot * n` would be R^-1.n instead - identical at identity
    // and visibly wrong for any rotated instance, e.g. trees standing radially on a planet, which
    // makes it a bug that only shows up in the scene you test last. Matches fragment.wgsl:333-337.
    pbr_input.N = normalize(surface.props.normal) * inv_rot;
    pbr_input.world_normal = pbr_input.N;

    pbr_input.material.base_color.a = surface.alpha;

#ifdef TWO_SIDED_LIGHTING
    // Identical viewer-facing flip to the forward path; see the comment there for why it exists
    // and why it tests against `V` rather than `@builtin(front_facing)`.
    if dot(pbr_input.N, pbr_input.V) < 0.0 {
        pbr_input.N = -pbr_input.N;
        pbr_input.world_normal = pbr_input.N;
    }
#endif

    #ifdef NORMAL_PREPASS
        out.normal = vec4<f32>(pbr_input.N, 0.0);
    #endif
    // we don't support MOTION_VECTOR or DEFERRED
    //
    // `frag_depth` is guarded by `UNCLIPPED_DEPTH_ORTHO_EMULATION` and nothing else: that is the
    // one and only def under which `prepass_io::FragmentOutput` actually declares the field in
    // bevy 0.19 (prepass_io.wgsl:96-98). The pre-0.19 spelling `DEPTH_CLAMP_ORTHO` no longer
    // exists anywhere in bevy - it was renamed at prepass/mod.rs:457 - so writing under it would
    // only ever assign a field the struct does not have.
    #ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
        out.frag_depth = in.position.z;
    #endif

    return out;
}
#else // PREPASS_FRAGMENT

// Depth-only prepass / shadow caster pass. No attachments to write - the only job is the alpha
// discard, exactly as bevy's own `pbr_prepass.wgsl` does with `prepass_alpha_discard`.
@fragment
fn fragment(in: CrossVertexOut) {
    // PERF MEASUREMENT ladder level 2: zero atlas fetches, and therefore no discard, so the quad
    // writes depth solid. Matches what level 2 does in the other two entry points.
    if cross_data.debug_level >= 2u {
        return;
    }

    // Called purely for its `discard` side effects; the sampled surface is unused here.
    _ = cross_surface(in);
}
#endif // PREPASS_FRAGMENT

#else // PREPASS_PIPELINE

@fragment
fn fragment(in: CrossVertexOut) -> FragmentOutput {
    var out: FragmentOutput;

    // R^T of the instance rotation, reassembled from the three varying columns (see cross_vertex).
    let inv_rot = mat3x3(
        in.inverse_rotation_0c,
        in.inverse_rotation_1c,
        in.inverse_rotation_2c,
    );

    // PERF MEASUREMENT ladder (`CrossImposterData::debug_level`; 0 = normal rendering). Ported
    // from fragment.wgsl on purpose: an A/B against the octahedral system is only meaningful if
    // the same rungs mean the same thing on both sides.
    //   1 = skip lighting and post-processing, emit the unpacked albedo
    //   2 = flat colour, zero atlas fetches
    let dbg = cross_data.debug_level;

    // PERF MEASUREMENT ladder level 2: flat colour, ZERO atlas fetches, no discard.
    //
    // This rung is deliberately NOT coverage-preserving, for exactly the reason fragment.wgsl's
    // level 4 is not: the alpha that carves the silhouette comes out of the SAME packed Rg32Uint
    // texel as the colour, so you cannot drop the fetch and keep the cutout. Everything below
    // level 2 keeps the fetch and therefore keeps screen coverage and overdraw byte-identical to
    // level 0; this rung trades that away to become the isolated fetch-cost probe, and the two
    // surviving hypotheses predict OPPOSITE directions for it:
    //
    //   * if the per-fragment atlas fetch is the cost, this collapses the pass even though every
    //     quad now draws solid;
    //   * if raw fragment count is the cost, this makes it WORSE - removing the alpha test roughly
    //     doubles the shaded area of each quad.
    //
    // Compare against LEVEL 1, not level 0: level 1 is already "one fetch, no shading", so 1 -> 2
    // is as close to an isolated fetch delta as this design allows.
    //
    // Placed BEFORE the first `cross_single_sample`, and before anything that could need a
    // derivative. Nothing in this shader uses dpdx/dpdy at all (that is a large part of the win
    // over the octahedral path, which needs them for `footprint`), so unlike fragment.wgsl there
    // is no uniformity-analysis hazard to reason about when branching here.
    if dbg >= 2u {
        // Mid-green, so a mis-set rung looks obviously wrong rather than plausible.
        out.color = vec4<f32>(0.15, 0.35, 0.12, 1.0);
        return out;
    }

    let surface = cross_surface(in);

    var pbr_input = unpack_pbrinput(surface.props, in.position);

    // Baked normals are in the model's LOCAL frame; rotate to world with R. inv_rot = R^T, so
    // R.n = `n * inv_rot` (vec*mat). `inv_rot * n` would be R^-1.n instead - identical at identity
    // and visibly wrong for any rotated instance, e.g. trees standing radially on a planet, which
    // makes it a bug that only shows up in the scene you test last. Matches fragment.wgsl:333-337.
    pbr_input.N = normalize(surface.props.normal) * inv_rot;
    pbr_input.world_normal = pbr_input.N;

    pbr_input.material.base_color.a = surface.alpha;

#ifdef TWO_SIDED_LIGHTING
    // The quads render with `primitive.cull_mode = None` - without that, half the cross disappears
    // as the camera orbits (plan section 2.1). The consequence is that roughly half of every cross
    // is being viewed from BEHIND, where the baked normal points away from the camera: N.L and
    // N.V both flip sign, the surface falls to the ambient floor, and a tree reads as lit from one
    // side and dead black from the other as you orbit it. Flipping the normal to face the viewer
    // is the standard two-sided-foliage fix and is physically defensible for a thin leaf-like
    // surface, which is what a cross imposter is pretending to be.
    //
    // Tested against V (the view vector), not against a facing builtin: `pbr_input.V` is already
    // computed by `unpack_pbrinput`, and unlike `@builtin(front_facing)` it is correct regardless
    // of the winding the mesh builder happened to emit.
    if dot(pbr_input.N, pbr_input.V) < 0.0 {
        pbr_input.N = -pbr_input.N;
        pbr_input.world_normal = pbr_input.N;
    }
#endif

    // PERF LADDER level >= 1: emit the already-unpacked albedo and skip lighting, the emissive
    // floor and fog/tonemap post-processing. Alpha is carried through untouched so the
    // blend/alpha-test state behaves exactly as at level 0 - the silhouette, and therefore screen
    // coverage and overdraw, are identical, which is what makes the timing delta attributable.
    // Deliberately ugly (no terminator, no specular): this is a ruler, not a look.
    if dbg >= 1u {
        out.color = pbr_input.material.base_color;
    } else {
        if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
#ifdef CHEAP_LIGHTING
            out.color = cheap_lighting(pbr_input);
#else
            out.color = apply_pbr_lighting(pbr_input);
#endif
        } else {
            out.color = pbr_input.material.base_color;
        }

        // Emissive floor: host scenes with very low ambient leave the shaded side of a baked
        // billboard near-black (its baked normals can all face away from the one light at some
        // view angles). Clamp the lit colour up to a fraction of the albedo so foliage never goes
        // pitch black. Done on the final HDR colour (before tonemapping) so it's independent of
        // exposure/emissive scaling. 0 = unchanged. Do NOT weight by coverage: every kept
        // (alpha-tested) fragment must get the same floor, else low-coverage silhouette texels are
        // lifted less than the body and read as a dark outline around the imposter.
        if cross_data.emissive_floor > 0.0 {
            let lift = pbr_input.material.base_color.rgb * cross_data.emissive_floor;
            out.color = vec4<f32>(max(out.color.rgb, lift), out.color.a);
        }

        pbr_input.material.flags |= STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT;

        // Kept even under CHEAP_LIGHTING: this is fog, tonemapping and debanding, and dropping it
        // would not make the imposter cheaper so much as make it a different colour from every
        // other surface in the scene.
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    } // end PERF LADDER `dbg >= 1u` guard

    return out;
}

#endif // PREPASS_PIPELINE

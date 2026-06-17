# boimp

boimp is the sound a mesh makes when its LODs pop. It's also a library for octahedral imposters in Bevy.

## Versions

| boimp | Bevy | Note |
| --- | --- | --- |
| 0.1.0 | 0.14 | Requires a slightly modified Bevy 0.14.2 (see Cargo.toml) |
| 0.2.0 | 0.15 | |
| 0.2.0 (`bevy-0.18.1` branch) | 0.18 | Current; uses the 0.18 required-components / GPU-driven render world |

Add the plugins you need: `ImposterBakePlugin` to generate imposters (it pulls in `ImposterRenderPlugin`), or just `ImposterRenderPlugin` to render pre-baked ones.

## Baking

Spawn an `ImposterBakeCamera` to capture everything within `radius` of its transform. It's a plain component (Bevy 0.18 has no bundles), so spawn it alongside a `Transform`:

```rs
commands.spawn((
    ImposterBakeCamera {
        radius: 10.0,             // how large an area to snapshot
        grid_size: 6,             // 6x6 separate snapshots
        tile_size: 512,           // 512x512 pixels per snapshot tile
        multisample: 8,           // samples to average over (^2, 8 -> 64 samples)
        grid_mode: GridMode::Spherical, // Spherical / Hemispherical / Horizontal
        ..default()
    },
    Transform::from_translation(Vec3::ZERO),
));
```

The camera bakes once and reports completion through its callback; set `continuous: true` to re-bake every frame (for animated or moving sources). By default it waits until every visible entity is fully renderable (pipelines compiled, mesh and material data on the GPU) before snapshotting.

To write the result to disk, attach the built-in save callback (and create a target image to read back into):

```rs
let mut camera = ImposterBakeCamera { /* .. */ ..default() };
camera.init_target(&mut images);
let save = camera.save_asset_callback("assets/boimps/output.boimp", true, true);
camera.set_callback(move |image| save(image));
commands.spawn((camera, Transform::from_translation(center)));
```

For anything to be produced, the materials in the area must implement `ImposterBakeMaterial`. This is provided for `StandardMaterial` (and for `Imposter` itself, so you can bake imposters of imposters); register other materials with `ImposterBakeMaterialPlugin::<M>`. The fragment shader is quite simple — see [the standard material version](src/shaders/standard_material_imposter_baker.wgsl).

## Rendering

Render a baked imposter as an `Imposter` material on a `Vec3::Z`-facing quad, using the 0.18 `Mesh3d` / `MeshMaterial3d` components:

```rs
commands.spawn((
    Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::splat(0.5)))),
    MeshMaterial3d(asset_server.load_with_settings::<Imposter, ImposterLoaderSettings>(
        "boimps/output.boimp",
        move |s| s.multisample = true,
    )),
));
```

Use a `Rectangle` or `Plane3d::new(Vec3::Z, Vec2::splat(0.5))` mesh. For dynamic or in-memory imposters, construct the `Imposter` material directly from a bake camera's `target` image and an `ImposterData` (see the `dynamic` example).

## Examples

### `dynamic`

Bakes the source glTF and spawns a large field of imposters from the result. Press `I` to start baking continuously every frame (so animated or moving sources stay in sync) and `O` to stop. `F` toggles dither at runtime. Defaults to the FlightHelmet.

Args:
- `--mode [s]pherical | [h]emispherical | [H]orizontal` : how snapshots are arranged (default hemispherical)
- `--grid <n>` : number of separate snapshots (^2) (default 15)
- `--tile <n>` : pixels per snapshot tile (default 128)
- `--count <n>` : number of imposters to spawn (default 1000)
- `--source <path>` : glTF to load (default FlightHelmet)
- `--cluster <n>` : bake `n` randomly-placed copies of the source into a single imposter (default 1)
- `--spacing <f>` : scales the gap between spawned imposters (default 1.0; <1 packs closer, >1 spreads out)
- `--multisample-source <n>` : samples to average over when baking (^2) (default 1)
- `--multisample-target` : average over nearby material pixels when rendering (default false)
- `--mask` : render with `AlphaMode::Mask` instead of `Blend`, enabling depth writes / early-Z (and stable ordering)
- `--a2c` : render with `AlphaMode::AlphaToCoverage` — MSAA anti-aliases the alpha-tested silhouette, no temporal pass (overrides `--mask`)
- `--fxaa` : enable FXAA on the camera
- `--dither` : static stochastic screen-space dither tile selection instead of the continuous blend (toggle at runtime with `F`)
- `--coverage` : coverage-preserving alpha for distant foliage (rescales and softens minified alpha so thin features keep density; pair with `--a2c`)
- `--fade` : distance detail fade — as imposters minify, flatten the baked normal, raise roughness and desaturate albedo toward a smooth blob to kill far-away sparkle
- `--ambient <f>` : ambient light brightness (default 1000.0)
- `--no-ambient` : disable ambient fill, leaving only the directional light

### `save_asset`

Loads a glTF, bakes an imposter and saves it to disk.

Args:
- `--mode [h]emispherical | [s]pherical` : how snapshots are arranged (default hemispherical)
- `--grid <n>` : number of separate snapshots (^2) (default 15)
- `--tile <n>` : pixels per snapshot tile (default 128)
- `--multisample-source <n>` : samples to average over when baking (^2) (default 8)
- `--source <path>` : glTF to load (default FlightHelmet)
- `--output <path>` : where to write the asset (default `assets/boimps/output.boimp`)
- `--no-shrink` : don't trim blank tile edges (shrinking often saves ~50% VRAM, no render cost)
- `--no-index` : don't deduplicate pixels into an indexed asset (indexing saves VRAM at the cost of one extra texture lookup at render time)

### `load_asset`

Loads a previously baked imposter and renders it with a fly camera.

Args:
- `--source <path>` : `.boimp` asset to load (default `boimps/output.boimp`)
- `--multisample` : average over nearby material pixels when rendering (default false)

## Known issues

Non-opaque materials aren't well supported. A single alpha-blend texture works fine, but multiple overlapping texture layers will take only the alpha of the front-most layer.

## Todo

- [ ] Integrate with visibility ranges
- [ ] Improve the asset format
- [x] Store and adjust for depths
- [ ] Maybe make the storage more configurable — currently 5 bit/channel color and alpha, 4 bit metallic and roughness, 4 bit flags (only the unlit flag is currently passed), 24 bit normal, 8 bit depth
- [ ] Maybe add an "image" mode that records the actual view rather than the material properties
- [x] Update to 0.15 and upstream
- [x] Update to 0.18
- [ ] Fix alpha issues
- [ ] Generate an atlas mip pyramid so minified imposters sample in one tap (currently a per-fragment box filter; `--fade` caps the tap count as a stopgap)
- [ ] Use vertex instancing to avoid needing a mesh

## License

boimp is free and open source. All code in this repository is dual-licensed under either:

- MIT License ([LICENSE-MIT](/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
</content>

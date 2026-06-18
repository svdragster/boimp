use bevy::{
    asset::{load_internal_asset, uuid_handle, RenderAssetUsages},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};
use wgpu::{Extent3d, TextureFormat};

use crate::{
    asset_loader::ImposterLoader,
    oct_coords::{GridMode, GRID_MASK},
};

pub const BINDINGS_HANDLE: Handle<Shader> = uuid_handle!("4c1df430-2854-44c7-ad81-7aa6cdbda096");
pub const FRAGMENT_HANDLE: Handle<Shader> = uuid_handle!("84f33ae9-a38d-4dbe-b09a-e3880e884ffe");
pub const SHARED_HANDLE: Handle<Shader> = uuid_handle!("8b0f60b1-9504-4e3c-a55a-998a53b9e447");
pub const VERTEX_HANDLE: Handle<Shader> = uuid_handle!("029aec6d-84bc-4772-a2e1-3002152e2dd2");

pub const RENDER_MULTISAMPLE_FLAG: u32 = 16;
pub const INDEXED_FLAG: u32 = 32;
pub const DITHER_FLAG: u32 = 64;
pub const COVERAGE_FLAG: u32 = 128;
pub const DETAIL_FADE_FLAG: u32 = 256;

pub struct ImposterRenderPlugin;

impl Plugin for ImposterRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            BINDINGS_HANDLE,
            "shaders/bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            FRAGMENT_HANDLE,
            "shaders/fragment.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, SHARED_HANDLE, "shaders/shared.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, VERTEX_HANDLE, "shaders/vertex.wgsl", Shader::from_wgsl);

        app.add_plugins(MaterialPlugin::<Imposter>::default())
            .register_asset_loader(ImposterLoader)
            .add_systems(Startup, setup);
    }
}

/// provides a fallback image for imposter indices, for use with dynamic imposting
#[derive(Resource)]
pub struct DummyIndicesImage(pub Handle<Image>);

pub fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        wgpu::TextureDimension::D2,
        vec![0, 0, 0, 0],
        TextureFormat::R32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    commands.insert_resource(DummyIndicesImage(images.add(image)));
}

#[derive(ShaderType, Clone, Copy, PartialEq, Debug)]
pub struct ImposterData {
    pub center_and_scale: Vec4,
    pub packed_tile_offset: UVec2,
    pub packed_tile_size: UVec2,
    pub grid_size: u32,
    pub base_tile_size: u32,
    pub flags: u32,
    pub alpha: f32,
    // (near, far) world distances for the `--swap` distance dither dissolve (examples/dynamic.rs):
    // the imposter is full at/beyond `far` and dithered fully away at/within `near`, so a real
    // model swapped in behind it shows through. `Vec2::ZERO` (the default) disables it.
    pub swap_fade: Vec2,
}

impl ImposterData {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        center: Vec3,
        scale: f32,
        grid_size: u32,
        base_tile_size: u32,
        packed_tile_offset: UVec2,
        packed_tile_size: UVec2,
        mode: GridMode,
        multisample: bool,
        indexed: bool,
        dither: bool,
        coverage: bool,
        fade: bool,
        alpha: f32,
    ) -> Self {
        Self {
            center_and_scale: center.extend(scale),
            grid_size,
            base_tile_size,
            packed_tile_offset,
            packed_tile_size,
            flags: mode.as_flags()
                + if multisample {
                    RENDER_MULTISAMPLE_FLAG
                } else {
                    0
                }
                + if indexed { INDEXED_FLAG } else { 0 }
                + if dither { DITHER_FLAG } else { 0 }
                + if coverage { COVERAGE_FLAG } else { 0 }
                + if fade { DETAIL_FADE_FLAG } else { 0 },
            alpha,
            swap_fade: Vec2::ZERO,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ImposterKey(u32);

#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
#[bind_group_data(ImposterKey)]
pub struct Imposter {
    #[uniform(0)]
    pub data: ImposterData,
    #[texture(1, dimension = "2d", sample_type = "u_int")]
    pub pixels: Handle<Image>,
    // annoyingly we can't use an option here because bevy gives us an rgba8 fallback
    // Res<DummyIndicesImage> gives a default you can drop in
    #[texture(2, dimension = "2d", sample_type = "u_int")]
    pub indices: Handle<Image>,
    pub alpha_mode: AlphaMode,
    pub vram_bytes: usize,
}

impl From<&Imposter> for ImposterKey {
    fn from(value: &Imposter) -> Self {
        Self(value.data.flags)
    }
}

impl Material for Imposter {
    fn vertex_shader() -> ShaderRef {
        VERTEX_HANDLE.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        VERTEX_HANDLE.into()
    }

    fn fragment_shader() -> ShaderRef {
        FRAGMENT_HANDLE.into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        FRAGMENT_HANDLE.into()
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
        let vert_defs = &mut descriptor.vertex.shader_defs;
        let frag_defs = &mut descriptor.fragment.as_mut().unwrap().shader_defs;

        if (key.bind_group_data.0 & RENDER_MULTISAMPLE_FLAG) != 0 {
            frag_defs.push("MATERIAL_MULTISAMPLE".into());
        }
        let grid_mode = match key.bind_group_data.0 & GRID_MASK {
            i if i == GridMode::Hemispherical.as_flags() => "GRID_HEMISPHERICAL",
            i if i == GridMode::Spherical.as_flags() => "GRID_SPHERICAL",
            i if i == GridMode::Horizontal.as_flags() => "GRID_HORIZONTAL",
            _ => panic!(),
        };
        vert_defs.push(grid_mode.into());
        frag_defs.push(grid_mode.into());

        if (key.bind_group_data.0 & INDEXED_FLAG) != 0 {
            // indexed
            frag_defs.push("INDEXED_PIXELS".into());
        }

        if (key.bind_group_data.0 & DITHER_FLAG) != 0 {
            // stochastic (screen-space dither) tile selection instead of the
            // continuous barycentric blend of the nearest octahedral tiles
            frag_defs.push("DITHERED".into());
        }

        if (key.bind_group_data.0 & COVERAGE_FLAG) != 0 {
            // coverage-preserving alpha for minified (distant) alpha-tested
            // foliage: rescale + soften the averaged alpha so thin features keep
            // their visual density and A2C/MSAA gets a fractional coverage to
            // dither. only meaningful with AlphaToCoverage / Blend.
            frag_defs.push("COVERAGE_PRESERVE".into());
        }

        if (key.bind_group_data.0 & DETAIL_FADE_FLAG) != 0 {
            // fade per-texel detail (normal map, albedo contrast, sharp specular)
            // toward a smooth low-frequency blob as the imposter minifies, to kill
            // distance sparkle that texture filtering alone can't reach.
            frag_defs.push("DETAIL_FADE".into());
        }

        Ok(())
    }
}

pub mod asset_loader;
pub mod bake;
pub mod cross;
pub mod cross_bake;
pub mod oct_coords;
pub mod render;

pub use asset_loader::ImposterLoaderSettings;
pub use bake::{ImposterBakeCamera, ImposterBakePlugin};
pub use cross::{
    cross_imposter_merged_mesh, cross_imposter_mesh, cross_quad_count, CrossImposter,
    CrossImposterData, CrossImposterPlugin, CrossInstance, CrossLayout, FULL_TILE,
};
pub use cross_bake::{
    cross_bake_camera, cross_imposter_from_bake_target, cross_trim_from_atlas, CrossTrim,
};
pub use oct_coords::GridMode;
pub use render::{Imposter, ImposterData, ImposterRenderPlugin};

//! One-off: run the CPU pyramid over a real committed `.boimp` cross atlas.
use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use boimp::cross_mip::{build_cross_mip_pyramid, cross_mip_levels};
use std::io::Read;

fn main() {
    let path = std::env::args().nth(1).expect("usage: verify_cross_mips <file.boimp>");
    let bytes = std::fs::read(&path).unwrap();
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(&bytes[..])).unwrap();

    let mut settings = String::new();
    zip.by_name("settings.txt").unwrap().read_to_string(&mut settings).unwrap();
    let parts: Vec<&str> = settings.trim().split(' ').collect();
    let grid: u32 = parts[0].parse().unwrap();
    let packed_tile: u32 = parts[6].parse().unwrap();

    let mut raw = Vec::new();
    zip.by_name("texture.png").unwrap().read_to_end(&mut raw).unwrap();
    let mut reader = image::ImageReader::new(std::io::Cursor::new(raw));
    reader.set_format(image::ImageFormat::Png);
    reader.no_limits();
    let pixels = reader.decode().unwrap().into_bytes();

    let size = packed_tile * grid;
    let img = Image::new(
        Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        TextureDimension::D2,
        pixels,
        TextureFormat::Rg32Uint,
        RenderAssetUsages::RENDER_WORLD,
    );
    println!("{path}\n  grid {grid}  tile {packed_tile}  image {size}x{size}");

    // Mean alpha over the whole atlas, per level. Alpha is averaged STRAIGHT by `weighted_props`,
    // so it must be preserved across every level - that is the check with real teeth.
    let mean_alpha = |data: &[u8], off: usize, dim: u32| -> f64 {
        let mut sum = 0.0;
        for i in 0..(dim * dim) as usize {
            let o = off + i * 8;
            let r = u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            sum += ((r >> 15) & 0x1f) as f64 / 31.0;
        }
        sum / (dim * dim) as f64
    };

    let mipped = build_cross_mip_pyramid(&img, packed_tile).expect("pyramid should build");
    let levels = mipped.texture_descriptor.mip_level_count;
    assert_eq!(levels, cross_mip_levels(packed_tile));
    let data = mipped.data.as_ref().unwrap();

    let mut off = 0usize;
    let mut dim = size;
    let mut expected_bytes = 0usize;
    let base_alpha = mean_alpha(data, 0, size);
    for level in 0..levels {
        let a = mean_alpha(data, off, dim);
        println!(
            "  level {level:2}: {dim:5}x{dim:<5}  mean alpha {a:.4}  (drift {:+.4})",
            a - base_alpha
        );
        // Two separate claims, because they fail for different reasons:
        //
        //  * coverage must never be LOST. That is the classic alpha-mipping failure - canopies
        //    thin out and eventually vanish with distance - and it is the direction that actually
        //    looks wrong. A small gain is harmless.
        //  * the gain must stay small. It comes from 5-bit alpha rounding half-up once per level,
        //    which is invisible in the mean until the texel count gets tiny; measured across all
        //    16 shipped atlases it is under 0.5% through level 6 and peaks around 2.6% at the
        //    2x2 level, where a whole cluster billboard is one screen pixel.
        assert!(
            a >= base_alpha - 0.005,
            "level {level} LOST coverage: {a} vs {base_alpha}"
        );
        assert!(
            a - base_alpha < 0.04,
            "level {level} gained too much coverage: {a} vs {base_alpha}"
        );
        expected_bytes += (dim * dim) as usize * 8;
        off += (dim * dim) as usize * 8;
        dim /= 2;
    }
    assert_eq!(data.len(), expected_bytes, "data length must match the declared chain");
    println!(
        "  OK: {levels} levels, {:.1} MB -> {:.1} MB (+{:.0}%)",
        (size * size * 8) as f64 / 1e6,
        data.len() as f64 / 1e6,
        (data.len() as f64 / (size * size * 8) as f64 - 1.0) * 100.0
    );
}

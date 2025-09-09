use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Debug)]
struct HeicImage {
    data: Vec<u8>,
    width: u32,
    height: u32,
    channels: u8,
}

fn load_heic<P: AsRef<Path>>(path: P) -> Result<HeicImage> {
    let path = path.as_ref();
    println!("Processing: {}", path.display());
    
    let ctx = libheif_rs::HeifContext::read_from_file(path)
        .with_context(|| format!("Failed to read HEIC file: {}", path.display()))?;
        
    let handle = ctx.primary_image_handle()
        .with_context(|| format!("Failed to get primary image handle from: {}", path.display()))?;
    
    let width = handle.width() as u32;
    let height = handle.height() as u32;
    
    println!("  - Dimensions: {}x{} ({} MP)", 
        width, height, 
        (width as f64 * height as f64) / 1_000_000.0
    );
    
    // Decode the image to RGB color space
    let image = handle.decode(
        libheif_rs::ColorSpace::Rgb(libheif_rs::RgbChroma::Rgb),
        None,
        false
    ).with_context(|| format!("Failed to decode HEIC file: {}", path.display()))?;
    
    // Get the raw pixel data
    if let Some(planes) = image.planes() {
        if let Ok(rgb_data) = planes.interleaved_rgb() {
            return Ok(HeicImage {
                data: rgb_data.to_vec(),
                width,
                height,
                channels: 3, // RGB
            });
        }
    }
    
    Err(anyhow::anyhow!("Failed to extract image data"))
}

fn process_heic_file(path: &Path) -> Result<()> {
    let img = load_heic(path)?;
    
    // Calculate basic statistics
    let size_mb = img.data.len() as f64 / (1024.0 * 1024.0);
    println!("  - Size: {:.2} MB", size_mb);
    
    // Example: Calculate average brightness
    let sum: u64 = img.data.iter().map(|&v| v as u64).sum();
    let avg_brightness = sum as f64 / img.data.len() as f64;
    println!("  - Avg brightness: {:.1}/255", avg_brightness);
    
    Ok(())
}

fn main() -> Result<()> {
    let input_dir = "test_images";
    
    println!("Processing HEIC files in: {}", input_dir);
    
    let mut processed = 0;
    let mut errors = 0;
    
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|e| e.to_str()).map_or(false, |ext| 
            ext.eq_ignore_ascii_case("heic") || ext.eq_ignore_ascii_case("heif")
        ) {
            match process_heic_file(&path) {
                Ok(_) => processed += 1,
                Err(e) => {
                    eprintln!("❌ Error processing {}: {}", path.display(), e);
                    errors += 1;
                }
            }
            println!(); // Add space between files
        }
    }
    
    println!("\nProcessing complete!");
    println!("✅ Successfully processed: {}", processed);
    if errors > 0 {
        println!("❌ Errors: {}", errors);
    }
    
    Ok(())
}

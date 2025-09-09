use anyhow::Result;
use libheif_rs::{
    ColorSpace, HeifContext, RgbChroma, LibHeif
};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <path_to_heic_file>", args[0]);
        return Ok(());
    }
    
    let path = &args[1];
    test_heic(path)
}

fn test_heic(path: &str) -> Result<()> {
    let path = Path::new(path);
    println!("Testing HEIC/HEIF support for: {}", path.display());

    // Try to load the image
    match load_heic(path) {
        Ok(Some(img)) => {
            println!("✅ Successfully loaded HEIC/HEIF image");
            println!("  - Dimensions: {}x{}", img.width(), img.height());
            
            // Save as PNG for verification
            let output_file = "test_output.png";
            img.save(output_file)?;
            println!("  - Saved as {}", output_file);
        },
        Ok(None) => {
            println!("❌ Failed to decode HEIC/HEIF image: No image data");
        },
        Err(e) => {
            println!("❌ Error loading HEIC/HEIF image: {}", e);
            println!("\nTroubleshooting tips:");
            println!("1. Make sure the file exists and is a valid HEIC/HEIF file");
            println!("2. Check that all required system libraries are installed");
        }
    }
    Ok(())
}

fn load_heic<P: AsRef<std::path::Path>>(path: P) -> Result<Option<image::DynamicImage>> {
    let libheif = LibHeif::new();
    let ctx = HeifContext::read_from_file(path.as_ref().to_str().ok_or_else(|| anyhow::anyhow!("Invalid file path"))?)?;
    let handle = ctx.primary_image_handle()?;
    
    // Decode the image to RGB color space
    let image = libheif.decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)?;
    
    // Get the planes
    let planes = image.planes();
    let width = image.width() as u32;
    let height = image.height() as u32;
    
    // Access the interleaved plane which contains the RGB or RGBA data
    if let Some(interleaved) = planes.interleaved {
        let data = interleaved.data;
        
        // Calculate bytes per pixel based on bits per pixel
        let bytes_per_pixel = (interleaved.bits_per_pixel + 7) / 8;
        
        // We expect 3 (RGB) or 4 (RGBA) bytes per pixel
        if bytes_per_pixel >= 3 {
            let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
            
            // Convert interleaved RGB/RGBA to RGBA
            for i in 0..(width * height) as usize {
                let base = i * bytes_per_pixel as usize;
                rgba_data.push(data[base]);     // R
                rgba_data.push(data[base + 1]); // G
                rgba_data.push(data[base + 2]); // B
                // If we have alpha, use it, otherwise use 255 (opaque)
                rgba_data.push(if bytes_per_pixel >= 4 { data[base + 3] } else { 255 });
            }
            
            // Create the image from the RGBA data
            let img = image::RgbaImage::from_raw(width, height, rgba_data)
                .map(image::DynamicImage::ImageRgba8);
            return Ok(img);
        }
    }
    
    Ok(None)
}


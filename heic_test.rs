use anyhow::Result;
use image::DynamicImage;
use std::path::Path;
use libheif_rs::{LibHeif, HeifContext, ColorSpace, RgbChroma};

fn main() -> Result<()> {
    let heic_path = "test.heic";
    
    // Try to load and process the HEIC file if it exists
    if Path::new(heic_path).exists() {
        match load_heic(heic_path) {
            Ok(Some(img)) => {
                println!("Successfully loaded HEIC file: {}x{}", img.width(), img.height());
                img.save("output.png")?;
                println!("Saved decoded HEIC to output.png");
            },
            Ok(None) => {
                println!("Failed to decode HEIC file (no error returned)");
            },
            Err(e) => {
                eprintln!("Error loading HEIC file: {}", e);
            }
        }
    } else {
        println!("No HEIC file found at {}", heic_path);
        println!("To test HEIC support, place a HEIC file named 'test.heic' in the current directory.");
        println!("You can create a test HEIC file using an online converter or a tool like ImageMagick:");
        println!("  magick convert -size 100x100 xc:red test.heic");
    }
    
    Ok(())
}

fn load_heic<P: AsRef<Path>>(path: P) -> Result<Option<DynamicImage>> {
    let libheif = LibHeif::new();
    let path_str = path.as_ref().to_str().ok_or_else(|| anyhow::anyhow!("Invalid file path"))?;
    let ctx = HeifContext::read_from_file(path_str)?;
    let handle = ctx.primary_image_handle()?;
    let _dimensions = (handle.width(), handle.height());
    
    // Decode directly to RGB with minimum quality loss
    let image = match libheif.decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error decoding HEIC: {}", e);
            return Ok(None);
        }
    };
    
    let planes = image.planes();
    println!("Available planes: y={:?}, cb={:?}, cr={:?}",
             planes.y.is_some(), 
             planes.cb.is_some(), 
             planes.cr.is_some());
    
    // Try to handle YCbCr format
    if let (Some(y_plane), Some(cb_plane), Some(cr_plane)) = (&planes.y, &planes.cb, &planes.cr) {
        let width = y_plane.width as u32;
        let height = y_plane.height as u32;
        
        // Convert YCbCr to RGB (simplified)
        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
        
        for i in 0..(width * height) as usize {
            let y = y_plane.data[i] as f32;
            let cb = cb_plane.data[i] as f32 - 128.0;
            let cr = cr_plane.data[i] as f32 - 128.0;
            
            // YCbCr to RGB conversion
            let r = (y + 1.402 * cr).max(0.0).min(255.0) as u8;
            let g = (y - 0.344136 * cb - 0.714136 * cr).max(0.0).min(255.0) as u8;
            let b = (y + 1.772 * cb).max(0.0).min(255.0) as u8;
            
            rgb_data.extend_from_slice(&[r, g, b]);
        }
        
        // Create RGB image
        let img = image::RgbImage::from_raw(width, height, rgb_data)
            .map(DynamicImage::ImageRgb8);
        
        return Ok(img);
    }
    
    // Fallback to just using Y plane as grayscale
    if let Some(y_plane) = &planes.y {
        let width = y_plane.width as u32;
        let height = y_plane.height as u32;
        let img = image::GrayImage::from_raw(width, height, y_plane.data.to_vec())
            .map(DynamicImage::ImageLuma8);
        
        return Ok(img);
    }
    
    Ok(None)
}

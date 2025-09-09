use anyhow::{Result, Context};
use libheif_rs::LibHeif;
use std::path::Path;

fn main() -> Result<()> {
    // Initialize libheif
    let libheif = LibHeif::new();
    println!("✅ libheif version: {}", libheif.version());
    
    // List of HEIC files to test (add your own test files here)
    let test_files = [
        "test.heic",
        "test2.heic",
        // Add more test files as needed
    ];
    
    let mut success = true;
    
    for file in &test_files {
        let path = Path::new(file);
        println!("\nTesting: {}", path.display());
        
        if !path.exists() {
            println!("❌ File does not exist");
            success = false;
            continue;
        }
        
        match test_heic_file(path) {
            Ok((width, height)) => {
                println!("✅ Success! Dimensions: {}x{} pixels", width, height);
            }
            Err(e) => {
                println!("❌ Error: {}", e);
                success = false;
                
                // Print troubleshooting tips
                println!("\nTroubleshooting tips:");
                println!("1. Make sure the file is a valid HEIC/HEIF file");
                println!("2. Install the latest HEIF Image Extensions from the Microsoft Store");
                println!("3. Install the latest Visual C++ Redistributable");
                println!("4. Check file permissions");
            }
        }
    }
    
    if success {
        println!("\n🎉 All tests passed! HEIC support is working correctly.");
    } else {
        println!("\n⚠️  Some tests failed. See above for details.");
    }
    
    Ok(())
}

fn test_heic_file(path: &Path) -> Result<(u32, u32)> {
    // Read the HEIC file
    let ctx = libheif_rs::HeifContext::read_from_file(path)
        .with_context(|| format!("Failed to read HEIC file: {}", path.display()))?;
    
    // Get the primary image handle
    let handle = ctx.primary_image_handle()
        .with_context(|| format!("Failed to get primary image from: {}", path.display()))?;
    
    let width = handle.width();
    let height = handle.height();
    
    println!("  - Format: {:?}", handle.format());
    println!("  - Has alpha: {}", handle.has_alpha_channel());
    println!("  - Bit depth: {}", handle.luma_bits_per_pixel());
    
    // Try to decode a small thumbnail first (faster test)
    if let Some(thumb) = handle.thumbnail() {
        println!("  - Thumbnail: {}x{}", thumb.width(), thumb.height());
    }
    
    Ok((width as u32, height as u32))
}

use image::{Rgba, RgbaImage};
use std::fs::create_dir_all;

fn main() -> std::io::Result<()> {
    // Create test_images directory if it doesn't exist
    create_dir_all("test_images")?;
    
    // Create a simple 2x2 test image
    let width = 2;
    let height = 2;
    let mut img = RgbaImage::new(width, height);
    
    // Set pixel colors (RGBA)
    img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));     // Red
    img.put_pixel(1, 0, Rgba([0, 255, 0, 255]));     // Green
    img.put_pixel(0, 1, Rgba([0, 0, 255, 255]));     // Blue
    img.put_pixel(1, 1, Rgba([255, 255, 0, 255]));   // Yellow
    
    // Save as PNG
    img.save("test_images/test.png")?;
    println!("✅ Created test.png in test_images/");
    
    // Create a README with instructions
    let readme = r#"Test Images for HEIC/HEIF Support
================================

This directory contains test images for verifying HEIC/HEIF support.

test.png - A simple 2x2 test image with primary colors

To test HEIC/HEIF support:
1. Convert test.png to HEIC format using an external tool
2. Save it as test.heic in this directory
3. Run the test_heic program to verify support
"#;
    
    std::fs::write("test_images/README.txt", readme)?;
    println!("✅ Created README.txt with instructions");
    
    Ok(())
}

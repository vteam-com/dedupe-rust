use image::{Rgba, RgbaImage};
use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    // Create a simple 2x2 test image
    let width = 2;
    let height = 2;
    let mut img = RgbaImage::new(width, height);
    
    // Set pixel colors (RGBA)
    img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));     // Red
    img.put_pixel(1, 0, Rgba([0, 255, 0, 255]));     // Green
    img.put_pixel(0, 1, Rgba([0, 0, 255, 255]));     // Blue
    img.put_pixel(1, 1, Rgba([255, 255, 0, 255]));   // Yellow
    
    // Save as PNG first
    img.save("test_images/test.png").unwrap();
    println!("✅ Created test.png");
    
    // Create a README with instructions
    let mut readme = File::create("test_images/README.txt")?;
    writeln!(readme, "Test Images for HEIC/HEIF Support")?;
    writeln!(readme, "================================")?;
    writeln!(readme, "")?;
    writeln!(readme, "This directory contains test images for verifying HEIC/HEIF support.")?;
    writeln!(readme, "")?;
    writeln!(readme, "test.png - A simple 2x2 test image with primary colors")?;
    writeln!(readme, "")?;
    writeln!(readme, "To test HEIC/HEIF support:")?;
    writeln!(readme, "1. Convert test.png to HEIC format using an external tool")?;
    writeln!(readme, "2. Save it as test.heic in this directory")?;
    writeln!(readme, "3. Run the test_heic program to verify support")?;
    
    println!("✅ Created README.txt with instructions");
    
    Ok(())
}

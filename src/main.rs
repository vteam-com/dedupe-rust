use anyhow::Result;
use clap::Parser;
use image::{DynamicImage, GenericImageView};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(author, version, about = "Duplicate Image Finder - Find and manage duplicate images")]
struct Args {
    /// Directory to scan for duplicate images (default: current directory)
    #[arg(short, long, default_value = ".")]
    directory: String,
    
    /// Number of pixels to use for quick hash (default: 100)
    #[arg(short, long, default_value_t = 20)]
    quick_pixels: usize,
    
    /// Similarity threshold (0.0-1.0) for deep comparison (default: 0.95)
    #[arg(short, long, default_value_t = 1.0)]
    threshold: f64,
    
    /// Number of images to process in each batch (default: 1000)
    #[arg(short, long, default_value_t = 1000)]
    batch_size: usize,
    
    /// Enable early termination for obviously different images
    #[arg(short = 'e', long, default_value_t = true)]
    early_termination: bool,
    
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Scanning directory: {}", args.directory);
    
    // Step 1: Find all image files with progress
    let image_files = find_image_files(&args.directory)?;
    if image_files.is_empty() {
        println!("No image files found in the specified directory.");
        return Ok(());
    }
    
    // Step 2: Quick scan with checksum
    let quick_hashes = quick_scan(&image_files, 10)?; // Use fixed sample size of 100
    
    // Step 3: Find potential duplicates based on quick hashes
    let potential_duplicates = find_potential_duplicates(quick_hashes);
    
    if potential_duplicates.is_empty() {
        println!("No potential duplicates found in quick scan.");
        return Ok(());
    }
    
    println!("Found {} potential duplicate groups. Starting deep comparison...", potential_duplicates.len());
    
    // Step 4: Deep comparison for potential duplicates
    let duplicates = find_duplicates(
        potential_duplicates, 
        args.threshold,
        args.batch_size,
        args.early_termination,
    )?;
    
    // Print results
    if duplicates.is_empty() {
        println!("No duplicates found.");
    } else {
        println!("\nFound {} duplicate groups:", duplicates.len());
        for (i, group) in duplicates.iter().enumerate() {
            println!("\nGroup {} ({} files):", i + 1, group.len());
            for path in group {
                println!("  {}", path.display());
            }
        }
    }
    
    Ok(())
}

fn find_image_files(directory: &str) -> Result<Vec<PathBuf>> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .tick_strings(&[ "⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"])
        .template("{spinner:.blue} {msg}")?);
    pb.set_message("Searching for image files...");
    
    let mut image_files = Vec::new();
    let extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff"];
    
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir()) {
            
        if let Some(ext) = entry.path().extension() {
            if let Some(ext_str) = ext.to_str() {
                if extensions.contains(&ext_str.to_lowercase().as_str()) {
                    image_files.push(entry.into_path());
                    pb.set_message(format!("Found {} image files...", image_files.len()));
                }
            }
        }
        pb.tick();
    }
    
    pb.finish_with_message(format!("Found {} image files", image_files.len()));
    Ok(image_files)
}

fn quick_scan(files: &[PathBuf], sample_size: usize) -> Result<Vec<(PathBuf, String)>> {
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("Quick scan [{elapsed_precise}]({eta}) [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")?);
    
    let results: Result<Vec<_>> = files.par_iter()
        .map(|path| {
            let result = compute_checksum(path, sample_size)
                .map(|hash| (path.clone(), hash));
            pb.inc(1);
            result
        })
        .collect();
    
    pb.finish_with_message("Quick scan complete");
    results
}

fn compute_checksum(path: &Path, sample_size: usize) -> Result<String> {
    let img = image::open(path)?;
    let (width, height) = img.dimensions();
    
    // Calculate step size to sample roughly sample_size pixels
    let total_pixels = (width * height) as usize;
    let step = (total_pixels / sample_size.max(1)).max(1) as u32;
    
    // Simple checksum using sampled pixels
    let mut checksum: u64 = 0;
    let mut count = 0;
    
    for y in (0..height).step_by(step as usize) {
        for x in (0..width).step_by(step as usize) {
            if x < width && y < height {
                let pixel = img.get_pixel(x, y);
                // Simple checksum: sum of RGB values with some bit shifting
                checksum = checksum.wrapping_add(
                    ((pixel[0] as u64) << 16) | 
                    ((pixel[1] as u64) << 8) | 
                    (pixel[2] as u64)
                );
                count += 1;
            }
        }
    }
    
    // Include image dimensions in the hash to catch differently sized images
    let dim_hash = (width as u64) << 32 | (height as u64);
    let final_hash = checksum.wrapping_mul(count as u64) ^ dim_hash;
    
    Ok(format!("{:016x}", final_hash))
}

fn find_potential_duplicates(hashes: Vec<(PathBuf, String)>) -> Vec<Vec<PathBuf>> {
    let mut hash_map: HashMap<String, Vec<PathBuf>> = HashMap::new();
    
    for (path, hash) in hashes {
        hash_map.entry(hash).or_default().push(path);
    }
    
    hash_map.into_values()
        .filter(|group| group.len() > 1)
        .collect()
}

fn find_duplicates(
    groups: Vec<Vec<PathBuf>>, 
    threshold: f64,
    batch_size: usize,
    early_termination: bool,
) -> Result<Vec<Vec<PathBuf>>> {
    let total_comparisons: usize = groups.iter()
        .map(|g| g.len().saturating_sub(1))
        .sum();
        
    if total_comparisons == 0 {
        return Ok(Vec::new());
    }
    
    println!("Performing deep comparison on {} potential groups...", groups.len());
    let pb = ProgressBar::new(total_comparisons as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {percent:>3}% | {msg}"
        )?
        .progress_chars("##-"),
    );
    
    // Process groups in batches to control memory usage
    let all_groups: Vec<Vec<PathBuf>> = groups.chunks(batch_size)
        .flat_map(|batch| {
            batch.par_iter()
                .filter_map(|group| {
                    if group.len() < 2 {
                        return None;
                    }
                    
                    // Load base image
                    let base_path = &group[0];
                    let base_img = match load_image(base_path) {
                        Ok(img) => img,
                        Err(_) => return None,
                    };
                    
                    pb.set_message(format!("Processing: {}", 
                        base_path.file_name().unwrap_or_default().to_string_lossy()
                    ));
                    
                    let mut current_group = vec![base_path.clone()];
                    
                    // Compare with other images in the group
                    for other_path in group.iter().skip(1) {
                        pb.inc(1);
                        
                        let other_img = match load_image(other_path) {
                            Ok(img) => img,
                            Err(_) => continue,
                        };
                        
                        pb.set_message(format!(
                            "Comparing: {} vs {}", 
                            base_path.file_name().unwrap_or_default().to_string_lossy(),
                            other_path.file_name().unwrap_or_default().to_string_lossy()
                        ));
                        
                        if compare_images(&base_img, &other_img, threshold, early_termination) {
                            current_group.push(other_path.clone());
                        }
                    }
                    
                    if current_group.len() > 1 {
                        Some(current_group)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    pb.finish_with_message("Deep comparison complete");
    Ok(all_groups)
}

fn load_image(path: &Path) -> Result<DynamicImage> {
    let (width, height) = image::image_dimensions(path)
        .map_err(|e| anyhow::anyhow!("Failed to get dimensions for {}: {}", path.display(), e))?;
    
    let file_size = std::fs::metadata(path)
        .map(|m| format!("{:.1} KB", m.len() as f64 / 1024.0))
        .unwrap_or_else(|_| "unknown".to_string());
    
    println!("    Loading: {} ({}x{}, {})", 
             path.file_name().unwrap_or_default().to_string_lossy(),
             width, height, file_size);
    
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to load image {}: {}", path.display(), e))?;
    
    Ok(img)
}

fn compare_images(img1: &DynamicImage, img2: &DynamicImage, threshold: f64, early_termination: bool) -> bool {
    // Get dimensions
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();
    
    // Calculate aspect ratios
    let aspect1 = w1 as f64 / h1 as f64;
    let aspect2 = w2 as f64 / h2 as f64;
    
    // Early termination for obviously different images
    if early_termination {
        // Check aspect ratios first (fast check)
        if (aspect1 - aspect2).abs() > 0.1 {
            return false;
        }
        
        // Check dimensions (very fast check)
        if (w1 as i32 - w2 as i32).abs() > 100 || (h1 as i32 - h2 as i32).abs() > 100 {
            return false;
        }
    }
    
    // Resize both images to the same dimensions (using the smaller dimensions)
    let target_size = 256; // Size to scale to for comparison
    let (target_w, target_h) = if w1 > h1 {
        (target_size, (target_size as f64 / aspect1) as u32)
    } else {
        ((target_size as f64 * aspect1) as u32, target_size)
    };
    
    let img1 = img1.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    let img2 = img2.resize_exact(target_w, target_h, image::imageops::FilterType::Lanczos3);
    
    // Convert to grayscale for comparison (more robust against color variations)
    let img1 = img1.to_luma8();
    let img2 = img2.to_luma8();
    
    let mut diff_sum = 0.0;
    let max_diff = 255.0 * (target_w * target_h) as f64;
    
    // Compare pixels
    for (p1, p2) in img1.pixels().zip(img2.pixels()) {
        let p1_val = p1.0[0] as f64;
        let p2_val = p2.0[0] as f64;
        diff_sum += (p1_val - p2_val).abs();
    }
    
    let similarity = 1.0 - (diff_sum / max_diff);
    
    println!("    Similarity: {:.2}% (threshold: {:.0}%)", 
             similarity * 100.0, 
             threshold * 100.0);
             
    similarity >= threshold
}

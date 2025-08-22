use anyhow::Result;
use clap::Parser;
use image::{DynamicImage, GenericImageView};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use thousands::Separable;

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
    
    // Step 2: Group files by extension
    let mut files_by_extension: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
    for file in &image_files {
        if let Some(ext) = file.extension().and_then(|e| e.to_str()) {
            files_by_extension.entry(ext.to_lowercase()).or_default().push(file.clone());
        }
    }
    
    // Convert to vector and sort by number of files (smallest to largest)
    let mut extensions: Vec<_> = files_by_extension.into_iter().collect();
    extensions.sort_by_key(|(_, files)| files.len());
    
    // Process each extension group separately (smallest first)
    let mut all_duplicates = Vec::new();
    let mut total_groups_processed = 0;
    let total_extensions = extensions.len();
    
    for (ext, ext_files) in extensions {
        println!("\nProcessing {} files with extension .{}...", ext_files.len(), ext);
        
        // Skip if we don't have enough files to find duplicates
        if ext_files.len() < 2 {
            continue;
        }
        
        // Step 3: Quick scan with checksum for this extension group
        let quick_hashes = quick_scan(&ext_files, args.quick_pixels)?;
        
        // Step 4: Find potential duplicates based on quick hashes
        let potential_duplicates = find_potential_duplicates(quick_hashes);
        
        if potential_duplicates.is_empty() {
            println!("  No potential duplicates found for .{} files.", ext);
            continue;
        }
        
        println!("  Found {} potential duplicate groups for .{} files. Starting deep comparison...", 
                potential_duplicates.len(), ext);
        
        // Step 5: Deep comparison for potential duplicates in this group
        let duplicates = find_duplicates(
            potential_duplicates, 
            args.threshold,
            args.batch_size,
            args.early_termination,
        )?;
        
        all_duplicates.extend(duplicates);
        total_groups_processed += 1;
        println!("  Completed processing .{} files ({} groups found so far, {}/{} extensions processed)",
                ext, all_duplicates.len(), total_groups_processed, total_extensions);
    }
    
    let duplicates = all_duplicates;
    
    // Print results
    if duplicates.is_empty() {
        println!("No duplicates found.");
    } else {
        // Sort groups by size (smallest first)
        let mut sorted_duplicates = duplicates;
        sorted_duplicates.sort_by_key(|group| group.len());
        
        println!("\nFound {} duplicate groups (sorted by size, smallest first):", sorted_duplicates.len().separate_with_commas());
        for (i, group) in sorted_duplicates.iter().enumerate() {
            println!("\nGroup {} ({} files):", (i + 1).separate_with_commas(), group.len().separate_with_commas());
            for path in group {
                println!("+  {}", path.display());
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
    let extensions = ["jpg", "jpeg", "png", "gif",  "tiff"];
    let mut extension_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir()) {
            
        if let Some(ext) = entry.path().extension() {
            if let Some(ext_str) = ext.to_str() {
                let ext_lower = ext_str.to_lowercase();
                if extensions.contains(&ext_lower.as_str()) {
                    *extension_counts.entry(ext_lower).or_insert(0) += 1;
                    image_files.push(entry.into_path());
                    pb.set_message(format!("Found {} image files...", image_files.len().separate_with_commas()));
                }
            }
        }
        pb.tick();
    }
    
    // Display the summary of found files by extension
    if !extension_counts.is_empty() {
        println!("\nFound image files by extension (sorted by count):");
        let mut extensions: Vec<_> = extension_counts.iter().collect();
        // Sort by count in descending order, then by extension name
        extensions.sort_by(|(a_ext, a_count), (b_ext, b_count)| 
            a_count.cmp(b_count).then_with(|| b_ext.cmp(a_ext))
        );
        
        for (ext, &count) in extensions {
            println!(">  .{}: {}", ext, count.separate_with_commas());
        }
        println!(); // Add an extra newline for better readability
    }
    
    pb.finish_with_message(format!("Found {} image files in total", image_files.len().separate_with_commas()));
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
    let img = match load_image(path)? {
        Some(img) => img,
        None => return Ok("ERROR".to_string()), // Return a special checksum for unloadable images
    };

    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return Ok("EMPTY".to_string());
    }

    // --- Updated: downscale first to avoid iterating over full-res pixels ---
    // We scale the image down to at most `sqrt(sample_size)` in each dimension.
    // This ensures we work on at most ~sample_size pixels, regardless of original size.
    let target_side = (sample_size as f64).sqrt().ceil() as u32;
    let resized = img.thumbnail(target_side, target_side);

    let (_w, _h) = resized.dimensions();
    let mut checksum: u64 = 0;
    let mut count = 0usize;

    // Iterate over all pixels of the reduced image
    for pixel in resized.to_rgb8().pixels() {
        checksum = checksum.wrapping_add(
            ((pixel[0] as u64) << 16)
                | ((pixel[1] as u64) << 8)
                | (pixel[2] as u64),
        );
        count += 1;
    }

    // Include original image dimensions in the hash
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

fn load_image(path: &Path) -> Result<Option<DynamicImage>> {
    match image::open(path) {
        Ok(img) => Ok(Some(img)),
        Err(e) => {
            eprintln!("⚠️  Warning: Could not load image {}: {}", path.display(), e);
            Ok(None)
        }
    }
}

fn compare_images(img1: &Option<DynamicImage>, img2: &Option<DynamicImage>, threshold: f64, early_termination: bool) -> bool {
    // If either image failed to load, they can't be compared
    if img1.is_none() || img2.is_none() {
        return false;
    }
    let img1 = img1.as_ref().unwrap();
    let img2 = img2.as_ref().unwrap();
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

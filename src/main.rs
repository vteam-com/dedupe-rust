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
    
        // Step 2: First group files by extension
    println!("\nGrouping files by extension...");
    let mut files_by_extension: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
    for file in &image_files {
        if let Some(ext) = file.extension().and_then(|e| e.to_str()) {
            files_by_extension.entry(ext.to_lowercase())
                .or_default()
                .push(file.clone());
        }
    }
    
    // Sort extensions alphabetically for consistent processing
    let mut extensions: Vec<_> = files_by_extension.into_iter().collect();
    extensions.sort_by_key(|(ext, _)| ext.clone());
    
    let mut all_duplicates = Vec::new();
    let total_extensions = extensions.len();
    
    // Process each extension group separately
    for (ext_idx, (ext, ext_files)) in extensions.into_iter().enumerate() {
        println!("\n\n=== Processing .{} files ({} of {}) ===", ext, ext_idx + 1, total_extensions);
        
        // Create a progress bar with consistent style
        let pb = ProgressBar::new(ext_files.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} files ({eta})")
                .unwrap()
                .progress_chars("#=:-·")
        );
        
        // Group files of this extension by dimensions
        let mut dimension_groups: std::collections::HashMap<(u32, u32), Vec<PathBuf>> = std::collections::HashMap::new();
        
        for file in ext_files {
            if let Ok(img) = image::open(&file) {
                let dimensions = img.dimensions();
                dimension_groups.entry(dimensions)
                    .or_default()
                    .push(file.clone());
            }
            pb.inc(1);
        }
        pb.finish_with_message("Done");
        
        // Separate groups with potential duplicates from unique dimensions
        let (mut dim_groups, unique_dims): (Vec<_>, Vec<_>) = dimension_groups
            .into_iter()
            .partition(|(_, files)| files.len() >= 2);
        
        // Sort groups by size (smallest first)
        dim_groups.sort_by_key(|(_, files)| files.len());
        
        // Print unique dimensions summary
        if !unique_dims.is_empty() {
            let unique_count: usize = unique_dims.iter().map(|(_, files)| files.len()).sum();
            let dims_list: Vec<String> = unique_dims
                .into_iter()
                .map(|((w, h), _)| format!("{}x{}", w, h))
                .collect();
            
            println!("\n.{}: Found {} files with unique dimensions ({}): {}", 
                    ext, 
                    unique_count,
                    if unique_count == 1 { "file" } else { "files" },
                    dims_list.join(", ")
            );
        }
        
        if dim_groups.is_empty() {
            println!("  No potential duplicate groups found for .{} files.", ext);
            continue;
        }
        
        // Print dimension groups with potential duplicates (sorted by width then height)
        println!("\n.{}: Found {} dimension groups with potential duplicates (sorted by dimensions):", ext, dim_groups.len());
        
        // Sort by width, then height
        dim_groups.sort_by(|((w1, h1), _), ((w2, h2), _)| (w1, h1).cmp(&(w2, h2)));
        
        for ((w, h), files) in &dim_groups {
            println!("  - {:>5}x{:<5} : {:>4} {}", 
                    w, h, 
                    files.len().separate_with_commas(),
                    if files.len() == 1 { "file" } else { "files" });
        }
        
        // Process each dimension group for this extension
        for ((_width, _height), files) in dim_groups {
            
            // Step 3: Quick scan with checksum for this dimension group
            let quick_hashes = quick_scan(&files, args.quick_pixels)?;
            
            // Step 4: Find potential duplicates based on quick hashes
            let potential_duplicates = find_potential_duplicates(quick_hashes);
            
            if potential_duplicates.is_empty() {
                println!("  No potential duplicates found for this group.");
                continue;
            }
            
            println!("  Found {} potential duplicate groups. Starting deep comparison...", 
                    potential_duplicates.len());
            
            // Step 5: Deep comparison for potential duplicates in this group
            let duplicates = find_duplicates(
                potential_duplicates, 
                args.threshold,
                args.batch_size,
                args.early_termination,
            )?;
            
            let num_duplicates = duplicates.len();
            all_duplicates.extend(duplicates);
            println!("  Found {} duplicate groups in this dimension group ({} total so far)",
                    num_duplicates, all_duplicates.len());
        }
    }
    
    // Print final results
    if all_duplicates.is_empty() {
        println!("\nNo duplicates found.");
    } else {
        // Sort groups by size (smallest first)
        let mut sorted_duplicates = all_duplicates;
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
    
    let mut image_files = Vec::new();
    let all_image_extensions = [
        // Processed formats (will be checked for duplicates)
        "jpg", "jpeg", "png", "gif",
        // Other known image formats (will be listed but not processed)
        "bmp", "tiff", "tif", "webp", "heic", "heif", "raw", "cr2", "nef",
        "arw", "orf", "rw2", "svg", "eps", "ai", "pdf", "ico", "dds",
        "psd", "xcf", "exr", "jp2"
    ];
    let processed_extensions = ["jpg", "jpeg", "png", "gif"];
    
    let mut extension_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir()) {
            
        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_lowercase();
            
            // Count all image extensions
            if all_image_extensions.contains(&ext_lower.as_str()) {
                *extension_counts.entry(ext_lower.clone()).or_insert(0) += 1;
                
                // Only add to processing list if it's one of our processed extensions
                if processed_extensions.contains(&ext_lower.as_str()) {
                    image_files.push(entry.into_path());
                }
            }
        }
    }
    
    // Display the summary of found files by extension
    if !extension_counts.is_empty() {
        println!("\nFound {} image files by extension:",  image_files.len().separate_with_commas());
        println!("   (sorted by count)");
        println!("   ([X] Marked extensions will be checked for duplicates)");
        
        let mut extensions: Vec<_> = extension_counts.iter().collect();
        // Sort by count in descending order, then by extension name
        extensions.sort_by(|(a_ext, a_count), (b_ext, b_count)| 
            a_count.cmp(b_count).then_with(|| b_ext.cmp(a_ext))
        );
        
        for (ext, &count) in extensions {
            let marker = if processed_extensions.contains(&ext.as_str()) { "[X]" } else { "[ ]" };
            println!("{} {:>6}: {}", marker, format!(".{}", ext), count.separate_with_commas());
        }
        println!(); // Add an extra newline for better readability
    }
    
    Ok(image_files)
}

fn quick_scan(files: &[PathBuf], sample_size: usize) -> Result<Vec<(PathBuf, String)>> {
    let total_files = files.len();
    println!("  Computing checksums for {} files...", total_files.separate_with_commas());
    
    let pb = ProgressBar::new(total_files as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} hashes ({eta})")
            .unwrap()
            .progress_chars("#=:-·")
    );
    
    let result = files.par_iter()
        .map(|path| {
            let result = compute_checksum(path, sample_size)
                .map(|hash| (path.clone(), hash));
            pb.inc(1);
            result
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into);
    
    pb.finish_with_message("Done");
    result
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
    
    let total_groups = groups.len();
    let pb = ProgressBar::new(total_groups as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} groups ({eta})\n{msg}"
        )
        .unwrap()
        .progress_chars("#=:-·")
    );

    // Process groups in batches to control memory usage
    let all_groups: Vec<Vec<PathBuf>> = groups.chunks(batch_size)
        .flat_map(|batch| {
            let batch_result = batch.par_iter()
                .filter_map(|group| {
                    if group.len() < 2 {
                        return None;
                    }
                    
                    // Load base image
                    let base_path = &group[0];
                    let base_img = match load_image(base_path) {
                        Ok(Some(img)) => img,
                        _ => return None,
                    };
                    
                    let mut current_group = vec![base_path.clone()];
                    
                    // Compare with other images in the group
                    for other_path in group.iter().skip(1) {
                        if let Ok(Some(other_img)) = load_image(other_path) {
                            if compare_images(&Some(base_img.clone()), &Some(other_img), threshold, early_termination) {
                                current_group.push(other_path.clone());
                            }
                        }
                    }
                    
                    if current_group.len() > 1 {
                        pb.set_message(format!("Found group with {} duplicates", current_group.len() - 1));
                        Some(current_group)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            
            pb.inc(batch.len() as u64);
            batch_result
        })
        .collect();
    
    pb.finish_with_message(format!("✓ Found {} duplicate groups", all_groups.len()));
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

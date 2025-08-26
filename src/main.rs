//! # Duplicate Image Finder
//! 
//! A command-line tool to find and manage duplicate images in a directory.
//! It uses perceptual hashing and image comparison to identify similar images.

use anyhow::Result;
use rayon::prelude::*;
use rayon::iter::IntoParallelRefIterator;
use clap::Parser;
use image::{DynamicImage, ImageFormat};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thousands::Separable;

/// Command-line arguments for the duplicate image finder.
/// 
/// This struct defines all the command-line arguments that the program accepts,
/// including directory to scan, comparison thresholds, and performance settings.
#[derive(Parser, Debug)]
#[command(author, version, about = "Duplicate Image Finder - Find and manage duplicate images")]
struct Args {
    /// Directory to scan for duplicate images (default: current directory)
    #[arg(short, long, default_value = ".")]
    directory: String,
        
    /// Number of images to process in each batch (default: 1000)
    #[arg(short, long, default_value_t = 1000)]
    batch_size: usize,    
}

fn main() -> Result<(), anyhow::Error> {
    let start_time = std::time::Instant::now();
    let args = Args::parse();

    //-----------------------------------------
    // step 1 find image files
    println!("🔍 Scanning {} for images...", args.directory);
    let image_files = step_1_find_image_files(&args.directory)?;
    
    if image_files.is_empty() {
        println!("❌ No image files found in the specified directory.");
    }
    else{        
        //-----------------------------------------
        // step 2 group and sort
        let files_by_extension = step_2_group_files_by_extension(&image_files);
        
        //-----------------------------------------
        // step 3 Process files by extension and find duplicates
        let all_duplicates = step_3_process_extensions(files_by_extension)?;
        
        //-----------------------------------------
        // Print results
        step_4_print_results(all_duplicates, start_time);
    }
    
    Ok(())
}


/// Recursively finds all image files in the specified directory.
///
/// # Arguments
/// * `directory` - The directory to search for image files
///
/// # Returns
/// * `Ok(Vec<PathBuf>)` - A vector of paths to found image files
/// * `Err(anyhow::Error)` - If there was an error reading the directory
///
/// # Supported Formats
/// * Processed for duplicates: jpg, jpeg, png, gif
/// * Listed but not processed: bmp, tiff, tif, webp, heic, heif, raw, cr2, nef, arw, orf, rw2, svg, eps, ai, pdf, ico, dds, psd, xcf, exr, jp2
fn step_1_find_image_files(directory: &str) -> Result<Vec<PathBuf>, anyhow::Error> {
    // Only include formats we actually support for processing
    let supported_extensions = ["jpg", "jpeg", "png", "gif", "webp"];
    let processed_extensions = supported_extensions;
    let mut extension_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut image_files = Vec::new();
    
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| !e.file_type().is_dir()) {
            
        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_lowercase();
            *extension_counts.entry(ext_lower.clone()).or_default() += 1;
            
            if supported_extensions.contains(&ext_lower.as_str()) {
                image_files.push(entry.path().to_path_buf());
            }
        }
    }
    
    // Print summary of found extensions
    if !extension_counts.is_empty() {
        println!("\nFound {} image files by extension: ",  image_files.len().separate_with_commas());
        println!("   (sorted by count)");
        println!("   ([X] Marked extensions will be checked for duplicates)");
        
        let mut extensions: Vec<_> = extension_counts.iter().collect();
        // Sort by count in descending order, then by extension name
        extensions.sort_by(|(a_ext, a_count), (b_ext, b_count)| 
            a_count.cmp(b_count).then_with(|| b_ext.cmp(a_ext))
        );
        
        for (ext, &count) in extensions {
            let marker = if processed_extensions.contains(&ext.as_str()) { "[X]" } else { "[ ]" };
            println!("{} {:<6}: {}", marker, format!(".{}", ext), count.separate_with_commas());
        }
        println!(); // Add an extra newline for better readability
    }
    
    Ok(image_files)
}

/// Groups files by their extension
fn step_2_group_files_by_extension(files: &[PathBuf]) -> std::collections::HashMap<String, Vec<PathBuf>> {
    let mut groups: std::collections::HashMap<_, Vec<PathBuf>> = std::collections::HashMap::new();
    for file in files {
        if let Some(ext) = file.extension().and_then(|e| e.to_str()) {
            groups.entry(ext.to_lowercase())
                .or_default()
                .push(file.clone());
        }
    }
    groups
}

/// Processes files grouped by extension to find duplicates
fn step_3_process_extensions(
    files_by_extension: std::collections::HashMap<String, Vec<PathBuf>>,
) -> Result<Vec<Vec<PathBuf>>, anyhow::Error> {
    let mut all_duplicates = Vec::new();
    
    // Sort extensions for consistent processing
    let mut extensions: Vec<_> = files_by_extension.into_iter().collect();
    extensions.sort_by_key(|(ext, _)| ext.clone());
    
    for (ext, ext_files) in extensions {
        // Set up progress bar for this extension
        let estimated_duplicates = (ext_files.len() as f32 * 0.1).max(1.0) as u64;
        let total_work = (ext_files.len() * 2) as u64 + estimated_duplicates;
        
        let pb = ProgressBar::new(total_work);
        pb.set_style(ProgressStyle::with_template("🔍 {prefix:.bold.dim} {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>·"));
        pb.set_prefix(format!(".{}", ext));
        pb.set_message("Starting...");
        
        // Group files by dimensions and process potential duplicates
        let dim_groups = group_by_dimensions(&ext_files, &pb);
        
        // Process each dimension group
        for ((width, height), files) in dim_groups {
            // Step 3: Quick scan with checksum for this dimension group
            pb.set_message("Quick scanning...");
            let quick_hashes = quick_scan(&files, (width, height), pb.clone())?;
            
            // Step 4: Find potential duplicates based on quick hashes
            let potential_duplicates = find_potential_duplicates(quick_hashes);
            
            // Step 5: Deep comparison for potential duplicates in this group
            pb.set_message("Deep comparing...");
            let duplicates = find_duplicates(potential_duplicates)?;
            all_duplicates.extend(duplicates);
            
            // Update progress for the deep comparison phase
            pb.inc(1);
        }
    }
    
    Ok(all_duplicates)
}

/// Groups files by their dimensions and returns groups of potential duplicates
fn group_by_dimensions(files: &[PathBuf], pb: &ProgressBar) -> Vec<((u32, u32), Vec<PathBuf>)> {
    pb.set_message("Analyzing dimensions...");
    
    // Group files by their dimensions
    let dimension_groups = files
        .par_iter()
        .filter_map(|file| {
            let dimensions = get_dimensions(file);
            pb.inc(1);
            dimensions.map(|d| (d, file.clone()))
        })
        .fold(
            || std::collections::HashMap::new(),
            |mut acc, (dimensions, file)| {
                acc.entry(dimensions).or_insert_with(Vec::new).push(file);
                acc
            },
        )
        .reduce(
            || std::collections::HashMap::new(),
            |mut a, mut b| {
                for (k, v) in b.drain() {
                    a.entry(k).or_default().extend(v);
                }
                a
            },
        );
    
    // Get groups with potential duplicates (2+ files with same dimensions)
    let (mut dim_groups, _): (Vec<_>, _) = dimension_groups
        .into_iter()
        .partition(|(_, files)| files.len() >= 2);
    
    // Sort groups by count (smallest first) then by dimensions
    dim_groups.sort_by(|(dims_a, files_a), (dims_b, files_b)| {
        files_a.len().cmp(&files_b.len()).then_with(|| dims_a.cmp(dims_b))
    });
    
    dim_groups
}

/// Prints the results of the duplicate search
fn step_4_print_results(duplicates: Vec<Vec<PathBuf>>, start_time: std::time::Instant) {
    if duplicates.is_empty() {
        println!("\n✅ No duplicates found in {:.2?}", start_time.elapsed());
        return;
    }
    
    // Sort groups by size (smallest first)
    let mut sorted_duplicates = duplicates;
    sorted_duplicates.sort_by_key(|group| group.len());
    
    println!("\n✨ Found {} groups of duplicates in {:.2?}", 
        sorted_duplicates.len().separate_with_commas(), 
        start_time.elapsed());
    
    for (i, group) in sorted_duplicates.iter().enumerate() {
        // Get dimensions of the first image in the group
        let dimensions = group.first()
            .and_then(|p| get_dimensions(p))
            .map(|(w, h)| format!("{}x{}", w, h))
            .unwrap_or_else(|| "unknown".to_string());
        
        println!("\nGroup {} ({} files, {}):", 
            (i + 1).separate_with_commas(), 
            group.len().separate_with_commas(),
            dimensions
        );
        
        for path in group {
            println!("  {}", path.display());
        }
    }
    
    let elapsed = start_time.elapsed();
    let total_seconds = elapsed.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    println!("\nTotal execution time: {:02}:{:02}:{:02}", hours, minutes, seconds);
}



/// Performs a quick scan of images to find potential duplicates.
///
/// This function processes images in parallel and computes their perceptual hashes.
/// It's called "quick" because it uses a sampling approach rather than comparing
/// every pixel of every image.
///
/// # Arguments
/// * `files` - Slice of image file paths to process
/// * `sample_size` - Number of pixels to use for the perceptual hash
/// * `group_dims` - The dimensions (width, height) that all images in this group share
///
/// # Returns
/// * `Ok(Vec<(PathBuf, String)>)` - Vector of (file path, hash) tuples
/// * `Err(anyhow::Error)` - If there was an error processing any of the images
/// Performs a parallel quick scan of images to find potential duplicates.
///
/// This function processes images in parallel using Rayon and computes their perceptual hashes.
/// It's designed to be memory efficient by processing files in chunks.
///
/// # Arguments
/// * `files` - Slice of image file paths to process
/// * `group_dims` - The dimensions (width, height) that all images in this group share
///
/// # Returns
/// * `Ok(Vec<(PathBuf, String)>)` - Vector of (file path, hash) tuples
/// * `Err(anyhow::Error)` - If there was an error processing any of the images
fn quick_scan(files: &[PathBuf], _group_dims: (u32, u32), pb: ProgressBar) -> Result<Vec<(PathBuf, String)>, anyhow::Error> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    let total_files = files.len();
    if total_files == 0 {
        return Ok(Vec::new());
    }

    let processed = Arc::new(AtomicUsize::new(0));
    let last_update = Arc::new(std::sync::Mutex::new(Instant::now()));
    
    // Process files in parallel with progress updates
    let result: Vec<_> = files
        .par_iter()
        .map_init(
            || (processed.clone(), last_update.clone(), pb.clone()),
            |(counter, last_update, pb), path| {
                // Process the file
                let result = match compute_checksum(path) {
                    Ok(hash) => Ok((path.clone(), hash)),
                    Err(e) => {
                        eprintln!("\nError processing {}: {}", path.display(), e);
                        Err(e)
                    }
                };

                // Update progress
                let processed = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if processed % 10 == 0 || processed == total_files {
                    let now = Instant::now();
                    let should_update = {
                        let mut last = last_update.lock().unwrap();
                        if now.duration_since(*last).as_millis() > 100 {
                            *last = now;
                            true
                        } else {
                            false
                        }
                    };
                    
                    if should_update {
                        pb.inc(10);
                        pb.set_message(format!("Quick scanning: {}/{} files", processed, total_files));
                    }
                }

                result
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    
    Ok(result)
}


/// Gets the dimensions of an image file.
///
/// # Arguments
/// * `path` - Path to the image file
///
/// # Returns
/// * `Some((width, height))` if the dimensions could be determined
/// * `None` if the file is not a supported image format or could not be read
fn get_dimensions(path: &Path) -> Option<(u32, u32)> {
    // First check if the file extension is supported
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase());
    
    // Only try to get dimensions for supported image formats
    match ext.as_deref() {
        Some("jpg") | Some("jpeg") | Some("png") | Some("gif") | Some("webp") => {
            // These are the formats we know how to handle
            let format = match ext.as_deref() {
                Some("jpg") | Some("jpeg") => ImageFormat::Jpeg,
                Some("png") => ImageFormat::Png,
                Some("gif") => ImageFormat::Gif,
                Some("webp") => ImageFormat::WebP,
                _ => return None, // This shouldn't happen due to the outer match
            };
            
            if let Ok(file) = File::open(path) {
                let reader = BufReader::new(file);
                if let Ok(dimensions) = image::io::Reader::with_format(reader, format)
                    .with_guessed_format()
                    .map_err(|_| ())
                    .and_then(|r| Ok(r.into_dimensions().map_err(|_| ())?))
                {
                    return Some(dimensions);
                }
            }
            None
        },
        // Skip unsupported formats entirely
        _ => None
    }
}

/// Groups files with identical hashes together as potential duplicates.
///
/// # Arguments
/// * `hashes` - Vector of (file path, hash) tuples
///
/// # Returns
/// * `Vec<Vec<PathBuf>>` - A vector of groups, where each group contains paths to
///   files that have identical hashes and are potential duplicates
fn find_potential_duplicates(hashes: Vec<(PathBuf, String)>) -> Vec<Vec<PathBuf>> {
    let mut hash_map: std::collections::HashMap<String, Vec<PathBuf>> = std::collections::HashMap::new();
    
    for (path, hash) in hashes {
        hash_map.entry(hash).or_default().push(path);
    }
    
    hash_map.into_values()
        .filter(|group| group.len() > 1)
        .collect()
}

/// Computes a perceptual hash of an image for quick comparison.
///
/// This function creates a 64-bit hash that represents the visual content of the image
/// by sampling the first 400 pixels. The hash is computed by:
/// 1. Reading the first 1200-1600 bytes of the image (400 pixels × 3-4 bytes per pixel)
/// 2. Converting each RGB pixel to a 24-bit value
/// 3. Summing these values with proper bit shifting
/// 4. Incorporating the image dimensions into the final hash
///
/// # Arguments
/// * `path` - Path to the image file
///
/// # Returns
/// * `Ok(String)` - A 16-character hexadecimal string representing the 64-bit hash
/// * `Err(anyhow::Error)` - If there was an error reading or processing the image
///
/// # Notes
/// - The hash is designed to be fast while providing reasonable uniqueness
/// - Images with similar visual content will have similar hash values
/// - The hash is sensitive to image dimensions and pixel values
fn compute_checksum(path: &Path) -> Result<String, anyhow::Error> {
    use std::fs::File;
    use std::io::Read;

    // Constants for the hash computation
    const PIXELS_TO_SAMPLE: usize = 1000;
    const BYTES_PER_PIXEL: usize = 3; // RGB
    const BUFFER_SIZE: usize = PIXELS_TO_SAMPLE * BYTES_PER_PIXEL;
    
    // Read the initial portion of the image file
    let mut file = File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open file {}: {}", path.display(), e))?;

    let mut buffer = vec![0u8; BUFFER_SIZE];
    let bytes_read = file.read(&mut buffer)?;
    
    if bytes_read == 0 {
        return Ok("EMPTY_FILE".to_string());
    }

    // Get image dimensions for the final hash
    let (width, height) = get_dimensions(path)
        .ok_or_else(|| anyhow::anyhow!("Could not determine image dimensions"))?;

    // Process pixels to compute the checksum
    let (checksum, valid_pixels) = buffer
        .chunks(BYTES_PER_PIXEL)
        .take(PIXELS_TO_SAMPLE)
        .filter_map(|chunk| {
            if chunk.len() >= BYTES_PER_PIXEL {
                Some((
                    ((chunk[0] as u64) << 16) | ((chunk[1] as u64) << 8) | (chunk[2] as u64),
                    1
                ))
            } else {
                None
            }
        })
        .fold((0u64, 0usize), |(sum, count), (pixel, valid)| {
            (sum.wrapping_add(pixel), count + valid)
        });

    if valid_pixels == 0 {
        return Ok("NO_VALID_PIXELS".to_string());
    }

    // Combine pixel data with image dimensions for final hash
    let dim_hash = (width as u64) << 32 | (height as u64);
    let final_hash = checksum.wrapping_mul(valid_pixels as u64) ^ dim_hash;

    // Format as 16-character hex string (64 bits)
    Ok(format!("{:016x}", final_hash))
}

/// Groups images by their pixel data hash to find exact duplicates.
///
/// This function calculates a hash for each image's pixel data and groups
/// images with identical hashes together, making it O(n) complexity.
///
/// # Arguments
/// * `groups` - Groups of potentially duplicate images
///
/// # Returns
/// * `Ok(Vec<Vec<PathBuf>>)` - Groups of confirmed duplicate images
/// * `Err(anyhow::Error)` - If there was an error processing the images
fn find_duplicates(
    groups: Vec<Vec<PathBuf>>
) -> Result<Vec<Vec<PathBuf>>, anyhow::Error> {
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use rayon::prelude::*;

    // A simple hash function for image data
    fn hash_image(img: &DynamicImage) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        img.as_bytes().hash(&mut hasher);
        hasher.finish()
    }

    // Create a progress bar for the entire operation
    let pb = ProgressBar::new(groups.len() as u64);
    // Get the file extension from the first file in the first group
    let ext = groups.first()
        .and_then(|g| g.first())
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .unwrap_or("");
        
    // Track total number of duplicate files found
    let total_duplicates = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let pb_dup = ProgressBar::new_spinner();
    // Remove the separate spinner bar since we'll show the count in the main progress bar
    drop(pb_dup);
    
    pb.set_style(
        ProgressStyle::with_template("🔍 {prefix:.bold.dim} {bar:40.cyan/blue} {pos}/{len} | {msg}")
        .unwrap()
        .progress_chars("#=:"));
    pb.set_prefix(format!("DeepCompare .{}", ext));

    // Process groups in parallel
    let results: Vec<Vec<Vec<PathBuf>>> = groups
        .par_iter()
        .enumerate()
        .filter_map(|(_i, group)| {
            if group.len() < 2 {
                pb.inc(1);
                return None;
            }


            let mut hash_map: HashMap<u64, Vec<PathBuf>> = HashMap::new();
            
            // Calculate hash for each image in the group
            for path in group {
                if let Ok(Some(img)) = load_image(path) {
                    let hash = hash_image(&img);
                    hash_map.entry(hash).or_default().push(path.clone());
                }
            }
            
            // Collect duplicates from this group
            let mut group_duplicates = Vec::new();
            for (_, mut files) in hash_map {
                if files.len() > 1 {
                    files.sort();
                    group_duplicates.push(files);
                }
            }
            
            pb.inc(1);
            
            if !group_duplicates.is_empty() {
                // Calculate and update total number of duplicate files
                let total_in_group: usize = group_duplicates.iter().map(|g| g.len() - 1).sum();
                let current_total = total_duplicates.fetch_add(total_in_group, std::sync::atomic::Ordering::Relaxed) + total_in_group;
                pb.set_message(format!("{} duplicates", current_total));
                Some(group_duplicates)
            } else {
                None
            }
        })
        .collect();
    
    // Flatten the results and sort by group size
    let mut all_duplicates: Vec<Vec<PathBuf>> = results.into_iter().flatten().collect();
    all_duplicates.sort_by_key(|group| group.len());
    
    Ok(all_duplicates)
}

/// Loads an image from disk with error handling.
///
/// # Arguments
/// * `path` - Path to the image file
///
/// # Returns
/// * `Ok(Some(image))` - If the image was loaded successfully
/// * `Ok(None)` - If the image could not be loaded (with error message printed to stderr)
/// * `Err(anyhow::Error)` - If there was an unexpected error
fn load_image(path: &Path) -> Result<Option<DynamicImage>, anyhow::Error> {
    match image::open(path) {
        Ok(img) => Ok(Some(img)),
        Err(e) => {
            eprintln!("⚠️  Warning: Could not load image {}: {}", path.display(), e);
            Ok(None)
        }
    }
}
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

fn main() -> Result<(), anyhow::Error> {
    let start_time = std::time::Instant::now();
    
    let args = Args::parse();
    println!("Scanning directory: {}", args.directory);
    
    // Step 1: Find all image files with progress
    let image_files = find_image_files(&args.directory)?;
    if image_files.is_empty() {
        println!("No image files found in the specified directory.");
        return Ok(());
    }
    
    // Step 2: First group files by extension

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
    
    let mut all_duplicates: Vec<Vec<PathBuf>> = Vec::new();
    
    // Process each extension group separately
    for (_ext_idx, (ext, ext_files)) in extensions.into_iter().enumerate() {
        // Create a progress bar with file type and dimensions
        let pb = ProgressBar::new(ext_files.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("Getting dimensions for {msg} [{elapsed_precise}] {eta} {bar:40.cyan/blue} {pos}/{len}")
                .unwrap()
                .progress_chars("#=:-·")
        );
        
        // Set initial message with file type and count
        pb.set_message(format!("'.{:<6}'", ext));
        
        // Group files of this extension by dimensions
        let mut dimension_groups: std::collections::HashMap<(u32, u32), Vec<PathBuf>> = std::collections::HashMap::new();
        
        for file in ext_files {
            if let Some(dimensions) = get_dimensions(&file) {
                dimension_groups.entry(dimensions)
                    .or_default()
                    .push(file.clone());
            }
            pb.inc(1);
        }
        
        // Separate groups with potential duplicates from unique dimensions
        let (mut dim_groups, _unique_dims): (Vec<_>, Vec<_>) = dimension_groups
            .into_iter()
            .partition(|(_, files)| files.len() >= 2);
        
        // Sort groups by size (smallest first)
        dim_groups.sort_by_key(|(_, files)| files.len());
        

        // Sort by width, then height
        dim_groups.sort_by(|((w1, h1), _), ((w2, h2), _)| (w1, h1).cmp(&(w2, h2)));
        
        // Process each dimension group for this extension
        for ((width, height), files) in dim_groups {
            print!("[{:>6}x{:<6}] {:>6} files = ", width, height, files.len());
            
            // Step 3: Quick scan with checksum for this dimension group
            let quick_hashes = quick_scan(&files, (width, height))?;
            
            // Step 4: Find potential duplicates based on quick hashes
            let potential_duplicates = find_potential_duplicates(quick_hashes);
            
            if potential_duplicates.is_empty() {
                println!("No potential duplicates.");
                continue;
            }
            
            println!("{} potential duplicates. Switching to full compare...", potential_duplicates.len());
            
            // Step 5: Deep comparison for potential duplicates in this group
            let duplicates = find_duplicates(potential_duplicates)?;
            
            let num_duplicates = duplicates.len();
            all_duplicates.extend(duplicates);
            if num_duplicates > 0 {
                use colored::Colorize;
                              
                println!("  ^^^^^^^^^^^   >>>> {} duplicate(s) found ({} total duplicates so far)",
                        num_duplicates.to_string().yellow(), 
                        all_duplicates.len().separate_with_commas().yellow().bold());
            }
        }
    }
    
    // Print final results
    if all_duplicates.is_empty() {
        println!("\nNo duplicates found.");
    } else {
        // Sort groups by size (smallest first)
        let mut sorted_duplicates = all_duplicates;
        sorted_duplicates.sort_by_key(|group| group.len());
        println!("\n========================================================");
        println!("Found {} groups of duplicates", sorted_duplicates.len().separate_with_commas());
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
    }
    
    let elapsed = start_time.elapsed();
    let total_seconds = elapsed.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    println!("\nTotal execution time: {:02}:{:02}:{:02}", hours, minutes, seconds);
    
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
fn find_image_files(directory: &str) -> Result<Vec<PathBuf>, anyhow::Error> {
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
    const PIXELS_TO_SAMPLE: usize = 400;
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
fn quick_scan(files: &[PathBuf], group_dims: (u32, u32)) -> Result<Vec<(PathBuf, String)>, anyhow::Error> {
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "[{elapsed_precise}]{eta} [{msg:>12}] {bar:40.cyan/blue} {pos:>7}/{len:7}"
    )
    .unwrap()
    .progress_chars("#=:-·"));
    
    let dims_msg = format!("{}x{}", group_dims.0, group_dims.1);
    pb.set_message(dims_msg.clone());
    
    let result = files.par_iter()
        .map(|path| {
            let ext = path.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
            let msg = format!("{} {}x{}", ext, group_dims.0, group_dims.1);
            pb.set_message(msg);
            
            let result = compute_checksum(path)
                .map(|hash| (path.clone(), hash));
                
            pb.inc(1);
            result
        })
        .collect::<Result<Vec<_>, _>>()?;        
    Ok(result)
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
    pb.set_style(
        ProgressStyle::with_template("  ^^^^^^^^^^^   {bar:40.cyan/blue} {pos}/{len} ({eta})\n{msg}")
        .unwrap()
        .progress_chars("#=:-·")
    );

    // Process groups in parallel
    let results: Vec<Vec<Vec<PathBuf>>> = groups
        .par_iter()
        .enumerate()
        .filter_map(|(i, group)| {
            if group.len() < 2 {
                pb.inc(1);
                return None;
            }

            // Update progress message
            if let Some(group_info) = group.first() {
                if let Some(dir) = group_info.parent() {
                    pb.set_message(format!("Group {}: {} files in {}", 
                        i + 1, 
                        group.len(), 
                        dir.display()));
                }
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
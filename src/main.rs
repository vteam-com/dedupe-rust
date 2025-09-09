//! # Duplicate Image Finder
//! 
//! A command-line tool to find and manage duplicate images in a directory.
//! It uses perceptual hashing and image comparison to identify similar images.

use anyhow::Result;
use clap::Parser;
use chrono::Local;
use image::DynamicImage;
use libheif_rs::{HeifContext, LibHeif, ColorSpace, RgbChroma};
use rayon::prelude::*;
use rayon::iter::IntoParallelRefIterator;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use serde::Serialize;
use thousands::Separable;

/// Formats a duration in a human-readable way
/// 
/// # Examples
/// ```
/// use std::time::Duration;
/// let duration = Duration::from_secs(125);
/// assert_eq!(format_duration(duration), "  2.1m");
/// ```
fn format_duration(duration: Duration) -> String {
    let elapsed_secs = duration.as_secs_f64();
    if elapsed_secs < 60.0 {
        format!("{:>5.1}s", elapsed_secs)
    } else if elapsed_secs < 3600.0 {
        format!("{:>5.1}m", elapsed_secs / 60.0)
    } else {
        format!("{:>5.1}h", elapsed_secs / 3600.0)
    }
}
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Debug, Serialize)]
struct DuplicateGroup {
    files: Vec<String>,
    dimensions: String,
}

#[derive(Debug, Serialize)]
struct Results {
    groups: Vec<DuplicateGroup>,
    total_groups: usize,
    execution_time: String,
}

mod dimensions;

// Re-export the get_dimensions function from the dimensions module
pub use dimensions::get_dimensions;

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
    
    /// Folders to exclude from scanning (can be specified multiple times)
    #[arg(long, value_name = "FOLDER")]
    exclude: Vec<String>,
}

fn main() -> Result<(), anyhow::Error> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    // Log the start of the application
    log::info!("Starting duplicate image finder");
    
    // Parse command line arguments
    let args = Args::parse();
    log::debug!("Command line arguments: {:?}", args);
    
    let start_time = Instant::now();
    
    //-----------------------------------------
    // step 1 find image files
    println!("🔍 Scanning {} for images...", args.directory);
    let image_files = step_1_find_image_files(&args.directory, &args.exclude)?;
    
    if image_files.is_empty() {
        println!("❌ No image files found in the specified directory.");
    }
    else{        
        //-----------------------------------------
        // step 2 group and sort
        let files_by_extension = step_2_group_files_by_extension(&image_files);
        
        //-----------------------------------------
        // step 3 Process files by extension and find duplicates
        let all_duplicates = step_3_process_extensions(files_by_extension, image_files.len(), start_time)?;
        
        // -----------------------------------------
        // Print results
        step_4_print_results(&all_duplicates, start_time, image_files.len());
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
/// * Processed for duplicates: jpg, jpeg, png, gif, webp, heic, heif
/// * Listed but not processed: bmp, tiff, tif, raw, cr2, nef, arw, orf, rw2, svg, eps, ai, pdf, ico, dds, psd, xcf, exr, jp2
fn step_1_find_image_files(directory: &str, exclude_folders: &[String]) -> Result<Vec<PathBuf>, anyhow::Error> {
    // Define supported image extensions
    let processed_extensions = ["bmp", "jpg", "jpeg", "png", "gif", "webp"]; //, "heic", "heif"];
    // All supported extensions (including detected but not processed)
    let _known_image_extensions = ["bmp", "jpg", "jpeg", "png", "gif", "webp", "heic", "heif", "tiff", "tif", "raw", "cr2", "nef", "arw", "orf", "rw2", "svg", "eps", "ai", "pdf", "ico", "dds", "psd", "xcf", "exr", "jp2"];
    let mut extension_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut image_files = Vec::new();
    
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_entry(move |e| {
            // Skip hidden directories and files
            let is_hidden = e.file_name()
                .to_str()
                .map(|s| s.starts_with('.'))
                .unwrap_or(false);
                
            // Skip excluded folders
            let is_excluded = e.path()
                .components()
                .any(|c| {
                    c.as_os_str()
                        .to_str()
                        .map(|s| exclude_folders.iter().any(|x| x == s))
                        .unwrap_or(false)
                });
                
            !is_hidden && !is_excluded
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
            let ext_lower = ext.to_lowercase();
            // Only add to image_files if it's in our processed_extensions list
            if processed_extensions.contains(&ext_lower.as_str()) {
                image_files.push(entry.path().to_path_buf());
            }
            // Still count all image extensions for the report
            *extension_counts.entry(ext_lower).or_insert(0) += 1;
        }
    }
    
    // Print summary of found extensions
    if !extension_counts.is_empty() {
        let total_count = extension_counts.values().sum::<usize>();
        let supported_count = image_files.len();
        
        println!("\nFound {} total files in directory", total_count.separate_with_commas());
        println!("Processing {} files with supported extensions:", supported_count.separate_with_commas());
        
        // Sort extensions by count in descending order, then by extension name
        let mut extensions: Vec<_> = extension_counts.iter().collect();
        extensions.sort_by(|(a_ext, a_count), (b_ext, b_count)| 
            b_count.cmp(a_count).then_with(|| a_ext.cmp(b_ext))
        );
        
        use colored::*;
        for (ext, &count) in extensions {
            let is_supported = processed_extensions.contains(&ext.as_str());
            let (marker, ext_display) = if is_supported {
                ("[PROCESSING]".green().bold(), format!(".{}", ext).green())
            } else {
                ("[ SKIPPING ]".dimmed(), format!(".{}", ext).dimmed())
            };
            println!("{} {:<6}: {} files", marker, ext_display, count.separate_with_commas());
        }
        
        if supported_count == 0 {
            println!("\n{} No files with supported extensions found in the specified directory.", 
                "Warning:".yellow().bold());
            println!("Supported extensions are: {}", 
                processed_extensions.iter().map(|e| format!(".{}", e)).collect::<Vec<_>>().join(", "));
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
    total_files: usize,
    start_time: Instant,
) -> Result<Vec<Vec<PathBuf>>, anyhow::Error> {
    let mut all_duplicates = Vec::new();
    
    // Sort extensions by number of files in ascending order
    let mut extensions: Vec<_> = files_by_extension.into_iter().collect();
    extensions.sort_by(|(_, a), (_, b)| a.len().cmp(&b.len()));

    
    // Track progress manually to avoid race conditions
    let mut processed_files = 0;
    
    for (ext, ext_files) in extensions {
        println!();
        println!("🔍 {} .{}", ext_files.len(), ext);
        
        // Group files by extension and dimensions
        let dim_groups = group_by_dimensions(&ext_files);
        
        // Process each dimension group
        for ((ext, width, height), files) in dim_groups {
            
            // Format elapsed time in a human-readable way
            let elapsed = start_time.elapsed();
            print!("[{}] \x1b[96m{:>6}\x1b[0m .{:<6} {:>7} x {:<7} ", 
                format_duration(elapsed), files.len(), ext, width, height);
            let quick_hashes = quick_scan(&files, width, height)?;
            
            // Step 4: Find potential duplicates based on quick hashes
            let potential_duplicates = find_potential_duplicates(quick_hashes);
            
            let color = if potential_duplicates.is_empty() { "\x1b[90m" } else { "\x1b[33m" };
            print!(" >> {}{:>6} \x1b[0m", color, potential_duplicates.len());
            
            // Step 5: Deep comparison for potential duplicates in this group
            let duplicates = find_duplicates(potential_duplicates)?;
            let num_duplicates = duplicates.len();
            all_duplicates.extend(duplicates.clone());
            
            // Update progress
            processed_files += files.len();
            
            let color = if num_duplicates == 0 { "\x1b[90m" } else { "" };
            let dupes_color = if num_duplicates > 0 { "\x1b[31m" } else { "" };
            println!(" >> {processed:>7}/{total:<7} {dupes_color} >> +{dupes}\x1b[0m{color} = {dupes_color}{total_dupes}\x1b[0m duplicates.\x1b[0m",
                   processed = processed_files.separate_with_commas(),
                   total = total_files.separate_with_commas(),
                   dupes = num_duplicates.separate_with_commas(),
                   total_dupes = all_duplicates.len().separate_with_commas(),
                   color = color,
                   dupes_color = dupes_color);
        }
    }
    
    Ok(all_duplicates)
}

/// Prints the results of the duplicate search and saves to a JSON file
fn step_4_print_results(duplicates: &[Vec<PathBuf>], start_time: Instant, total_files_found: usize) {
    let formatted_time = format!("{:.2?}", start_time.elapsed());
    
    if duplicates.is_empty() {
        let message = format!("\n✅ No duplicates found in {}", formatted_time);
        let results = Results {
            groups: Vec::new(),
            total_groups: 0,
            execution_time: formatted_time,
        };
        
        // Generate filename with timestamp
        let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("dedupe_{}.json", timestamp);
        
        // Save empty results to JSON file
        if let Ok(json) = serde_json::to_string_pretty(&results) {
            if let Err(e) = std::fs::write(&filename, &json) {
                eprintln!("Error writing to JSON file: {}", e);
            } else {
                println!("\nResults saved to: {}", filename);
            }
        }
        
        println!("{}", message);
        return;
    }

    let mut sorted_duplicates = duplicates.to_vec();
    sorted_duplicates.sort_by(|a, b| {
        let a_path = a.first().map(|p| p.as_path()).unwrap_or_else(|| Path::new(""));
        let b_path = b.first().map(|p| p.as_path()).unwrap_or_else(|| Path::new(""));
        a_path.cmp(b_path)
    });
    
    // Prepare the results
    let results = Results {
        groups: sorted_duplicates.iter()
            .map(|group| {
                let dimensions = group.first()
                    .and_then(|p| dimensions::get_dimensions(p))
                    .map(|(w, h)| format!("{}x{}", w, h))
                    .unwrap_or_else(|| "unknown".to_string());
                
                DuplicateGroup {
                    files: group.iter()
                        .map(|p| p.display().to_string())
                        .collect(),
                    dimensions,
                }
            })
            .collect(),
        total_groups: sorted_duplicates.len(),
        execution_time: formatted_time.clone(),
    };
    
    // Generate filename with timestamp
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let filename = format!("dedupe_{}.json", timestamp);
    
    if let Ok(json) = serde_json::to_string_pretty(&results) {
        if let Err(e) = std::fs::write(&filename, &json) {
            eprintln!("Error writing to JSON file: {}", e);
        } else {
            println!("\nResults saved to: {}", filename);
        }
    } else {
        eprintln!("Error serializing results to JSON");
    }
    
    // Calculate duplicate statistics
    let total_files_in_duplicate_groups = duplicates.iter().map(|group| group.len()).sum::<usize>();
    let total_unique_files = total_files_found - (total_files_in_duplicate_groups - duplicates.len());
    
    // Print summary to console
    println!("\n📊 Summary:");
    println!("  • {} files scanned", total_files_found.separate_with_commas());
    println!("  • {} unique files", total_unique_files.separate_with_commas());
    println!("  • {} duplicate files in {} groups",  duplicates.iter().map(|group| group.len()).sum::<usize>().separate_with_commas(),duplicates.len().separate_with_commas());
    println!("  • Complete in {}", format_duration(start_time.elapsed()).trim());
}

/// Groups files by their dimensions and returns groups of potential duplicates
fn group_by_dimensions(files: &[PathBuf]) -> Vec<((String, u32, u32), Vec<PathBuf>)> {
    println!("🔍 Analyzing image dimensions...");
    
    // Create a progress bar
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len}",
    ).unwrap()
    .progress_chars("#>-"));
    
    // Group files by their extension and dimensions
    let dimension_groups: std::collections::HashMap<(String, u32, u32), Vec<PathBuf>> = files
        .par_iter()
        .inspect(|_| {
            pb.inc(1);
        })
        .filter_map(|file| {
            let ext = file.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();
                
            // Get dimensions
            let dimensions = dimensions::get_dimensions(file);
            
            // Return (extension, dimensions, file) tuple
            dimensions.map(|d| ((ext, d.0, d.1), file.clone()))
        })
        .fold(
            || std::collections::HashMap::new(),
            |mut acc: std::collections::HashMap<_, Vec<_>>, ((ext, width, height), file)| {
                acc.entry((ext, width, height))
                    .or_default()
                    .push(file);
                acc
            },
        )
        .reduce(
            || std::collections::HashMap::new(),
            |mut a: std::collections::HashMap<_, Vec<_>>, mut b| {
                // Update progress bar during reduction
                pb.inc(0); // This just forces a refresh without incrementing
                for (k, v) in b.drain() {
                    a.entry(k).or_default().extend(v);
                }
                a
            },
        );
    
    // Finish the progress bar
    pb.finish_with_message(format!("Analyzed {} files", files.len()));
    
    // Get groups with potential duplicates (2+ files with same dimensions and extension)
    let mut dim_groups: Vec<_> = dimension_groups
        .into_iter()
        .filter(|(_, files)| files.len() > 1)
        .collect();
    
    // Sort the dimension groups by extension, then width, then height
    dim_groups.sort_by(|&((ref a_ext, a_w, a_h), _), &((ref b_ext, b_w, b_h), _)| {
        // First sort by extension
        a_ext.cmp(b_ext)
            // Then by width
            .then_with(|| a_w.cmp(&b_w))
            // Then by height
            .then_with(|| a_h.cmp(&b_h))
    });
    
    println!("✅ Found {} potential duplicate groups based on dimensions", dim_groups.len());
    dim_groups
}

/// Performs a quick scan of images to find potential duplicates.
///
/// This function processes images in parallel and computes their perceptual hashes.
/// It's called "quick" because it uses a sampling approach rather than comparing
/// every pixel of every image.
///
/// # Arguments
/// * `files` - Slice of image file paths to process
/// * `width` - The width of all images in this group
/// * `height` - The height of all images in this group
/// * `pb` - Progress bar for reporting progress
///
/// # Returns
/// * `Ok(Vec<(PathBuf, String)>)` - Vector of (file path, hash) tuples
/// * `Err(anyhow::Error)` - If there was an error processing any of the images
fn quick_scan(files: &[PathBuf], _width: u32, _height: u32) -> Result<Vec<(PathBuf, String)>, anyhow::Error> {
    use rayon::prelude::*;
    
    match files.len() {
        0 => return Ok(Vec::new()),
        1 => return Ok(vec![(files[0].clone(), compute_checksum(&files[0])?)]),
        _ => {}
    }
    
    // Process files in parallel
    let result: Vec<_> = files
        .par_iter()
        .map(|path| {
            // Process the file
            let result = match compute_checksum(path) {
                Ok(hash) => Ok((path.clone(), hash)),
                Err(e) => {
                    eprintln!("\nError processing {}: {}", path.display(), e);
                    Err(e)
                }
            };
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
/// images with identical hashes together. It ensures that only files with
/// the same extension are compared.
///
/// # Arguments
/// * `groups` - Groups of potentially duplicate images (already grouped by extension and dimensions)
///
/// # Returns
/// * `Ok(Vec<Vec<PathBuf>>)` - Groups of confirmed duplicate images
/// * `Err(anyhow::Error)` - If there was an error processing the images
fn find_duplicates(
    groups: Vec<Vec<PathBuf>>,    
) -> Result<Vec<Vec<PathBuf>>, anyhow::Error> {
    use std::collections::HashMap;
    use rayon::prelude::*;

    // A simple hash function for image data that includes the file extension
    fn hash_image(img: &DynamicImage, path: &Path) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Include file extension in the hash
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            ext.hash(&mut hasher);
        }
        img.as_bytes().hash(&mut hasher);
        hasher.finish()
    }
    
    // Process groups in parallel
    let results: Vec<Vec<Vec<PathBuf>>> = groups
        .par_iter()
        .enumerate()
        .filter_map(|(_i, group)| {
            if group.len() < 2 {
                // Update progress for skipped groups
                return None;
            }


            let mut hash_map: HashMap<u64, Vec<PathBuf>> = HashMap::new();
            
            // Calculate hash for each image in the group
            for path in group {
                if let Ok(Some(img)) = load_image(path) {
                    let hash = hash_image(&img, path);
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
            
            if group_duplicates.is_empty() {
                None
            } else {
                Some(group_duplicates)
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
    // Check if the file is a HEIC/HEIF image
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        if ext.eq_ignore_ascii_case("heic") || ext.eq_ignore_ascii_case("heif") {
            match load_heic(path) {
                Ok(img) => return Ok(img),
                Err(e) => {
                    eprintln!("⚠️  Warning: Could not load HEIC image {}: {}", path.display(), e);
                    return Ok(None);
                }
            }
        }
    }
    
    // Handle other image formats
    match image::open(path) {
        Ok(img) => Ok(Some(img)),
        Err(e) => {
            eprintln!("⚠️  Warning: Could not load image {}: {}", path.display(), e);
            Ok(None)
        }
    }
}

/// Loads a HEIC/HEIF image efficiently with better error handling
fn load_heic<P: AsRef<std::path::Path>>(path: P) -> Result<Option<image::DynamicImage>, anyhow::Error> {
    let path = path.as_ref();
    log::debug!("Loading HEIC file: {}", path.display());
    
    // Check file size first to avoid reading very large files
    let metadata = std::fs::metadata(path)?;
    if metadata.len() == 0 {
        log::debug!("HEIC file is empty: {}", path.display());
        return Ok(None);
    }
    
    // Check if we're on Windows 32-bit which has known issues
    #[cfg(windows)]
    if std::mem::size_of::<usize>() != 8 {
        log::warn!("HEIC loading on 32-bit Windows is not supported. Please use a 64-bit build.");
        return Ok(None);
    }
    
    let libheif = LibHeif::new();
    // LibHeif::new() doesn't return a Result, so we don't need to match on it
    // Any initialization errors will be caught when we try to use it
    
    let path_str = path.to_str().ok_or_else(|| anyhow::anyhow!("Invalid file path"))?;
    
    log::trace!("Reading HEIC context");
    let ctx = match HeifContext::read_from_file(path_str) {
        Ok(ctx) => ctx,
        Err(e) => {
            log::error!("Failed to read HEIC context: {}", e);
            return Err(e.into());
        }
    };
    
    log::trace!("Getting primary image handle");
    let handle = match ctx.primary_image_handle() {
        Ok(handle) => handle,
        Err(e) => {
            log::error!("Failed to get primary image handle: {}", e);
            return Err(e.into());
        }
    };
    
    // Get dimensions from metadata
    let width = handle.width() as u32;
    let height = handle.height() as u32;
    
    // Decode directly to RGB with minimum quality loss
    let image = libheif.decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)?;
    let planes = image.planes();
    
    // Try to get interleaved RGB data first (most efficient)
    if let Some(interleaved) = planes.interleaved {
        let data = interleaved.data;
        if data.len() >= (width * height * 3) as usize {
            // Convert RGB to RGBA by adding alpha channel
            let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
            for chunk in data.chunks(3) {
                rgba_data.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            
            if let Some(img) = image::RgbaImage::from_raw(width, height, rgba_data) {
                return Ok(Some(image::DynamicImage::ImageRgba8(img)));
            }
        }
    }
    
    // Fallback to separate planes if interleaved not available
    if let (Some(y_plane), Some(cb_plane), Some(cr_plane)) = (&planes.y, &planes.cb, &planes.cr) {
        // Convert YCbCr to RGB
        let mut rgb_data = Vec::with_capacity((width * height * 3) as usize);
        
        for i in 0..(width * height) as usize {
            let y = y_plane.data[i] as f32;
            let cb = cb_plane.data[i] as f32 - 128.0;
            let cr = cr_plane.data[i] as f32 - 128.0;
            
            // YCbCr to RGB conversion
            let r = (y + 1.402 * cr).clamp(0.0, 255.0) as u8;
            let g = (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0) as u8;
            let b = (y + 1.772 * cb).clamp(0.0, 255.0) as u8;
            
            rgb_data.extend_from_slice(&[r, g, b]);
        }
        
        // Convert RGB to RGBA
        let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
        for chunk in rgb_data.chunks(3) {
            rgba_data.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
        }
        
        if let Some(img) = image::RgbaImage::from_raw(width, height, rgba_data) {
            return Ok(Some(image::DynamicImage::ImageRgba8(img)));
        }
    }
    
    // Last resort: Use Y plane as grayscale if color conversion fails
    if let Some(y_plane) = &planes.y {
        let mut rgba_data = Vec::with_capacity((width * height * 4) as usize);
        for &y in y_plane.data.iter() {
            rgba_data.extend_from_slice(&[y, y, y, 255]);
        }
        
        if let Some(img) = image::RgbaImage::from_raw(width, height, rgba_data) {
            return Ok(Some(image::DynamicImage::ImageRgba8(img)));
        }
    }
    
    Ok(None)
}
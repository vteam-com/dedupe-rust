# Duplicate Image Finder

A high-performance command-line tool to find and manage duplicate images in your directories, written in Rust.

## Features

- 🚀 Blazing fast scanning using multi-threading with Rayon
- 🔍 Two-phase detection (quick hash + deep comparison)
- 📊 Detailed file statistics including extension-based grouping
- 🖼️ Supports multiple image formats (JPEG, PNG, GIF, BMP, TIFF)
- 📈 Progress tracking with `indicatif`
- ⚙️ Configurable similarity threshold (0.0-1.0)
- 🛠️ Early termination for faster processing
- 📋 Clean, organized output with file type breakdown

## Installation

### Prerequisites
- Rust and Cargo (Rust's package manager) installed

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/dedupe_rust.git
cd dedupe_rust

# Build in release mode
cargo build --release

# The binary will be available at ./target/release/duplicate_image_finder
```

## Usage

```bash
# Basic usage (scans current directory)
./duplicate_image_finder

# Scan a specific directory
./duplicate_image_finder -d /path/to/images

# Example output:
# Found image files by extension:
#   .jpg: 42 files
#   .png: 15 files
#   .gif: 3 files
#
# Found 60 image files in total
# Scanning for duplicates...
# Found 5 potential duplicate groups. Starting deep comparison...
# ...

# Adjust quick scan sensitivity (default: 20)
./duplicate_image_finder -p 30

# Set custom similarity threshold (0.0-1.0, default: 0.95)
# Lower values = more aggressive matching
./duplicate_image_finder -t 0.9

# Process images in batches (default: 1000)
# Useful for very large collections to manage memory usage
./duplicate_image_finder -b 500

# Disable early termination for more accurate results
# (slower but more thorough comparison)
./duplicate_image_finder --no-early-termination
```

### Options

- `-d, --directory <DIRECTORY>`: Directory to scan (default: current directory)
- `-p, --quick-pixels <QUICK_PIXELS>`: Number of pixels to use for quick hash (default: 20)
- `-t, --threshold <THRESHOLD>`: Similarity threshold (0.0-1.0) for deep comparison (default: 0.95)
- `-b, --batch-size <BATCH_SIZE>`: Number of images to process in each batch (default: 1000)
- `-e, --early-termination`: Enable early termination for obviously different images (default: true)
- `-h, --help`: Print help
- `-V, --version`: Print version

## How It Works

1. **File Discovery Phase**:
   - Recursively scans the specified directory for image files
   - Groups and counts files by their extensions (e.g., .jpg, .png)
   - Displays a summary of found files by type

2. **Quick Scan Phase**:
   - Computes a quick hash for each image using a configurable number of sample pixels
   - Groups images with identical hashes as potential duplicates
   - Provides progress feedback during scanning

3. **Deep Comparison Phase**:
   - For each group of potential duplicates, performs a pixel-by-pixel comparison
   - Uses configurable threshold (0.0-1.0) to determine similarity
   - Supports early termination for faster processing of obviously different images
   - Displays detailed comparison results

## Performance Tips

- Increase `--quick-pixels` for more accurate quick scanning (at the cost of speed)
- Adjust `--threshold` based on your needs (lower = more sensitive)
- Larger `--batch-size` can be faster but uses more memory
- Early termination (`-e`) provides significant speed improvements with minimal accuracy loss

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## TODO

- [ ] Add support for more image formats
- [ ] Implement a dry-run mode
- [ ] Add option to move/delete duplicates automatically
- [ ] Add more detailed progress reporting
- [ ] Support for video frame comparison

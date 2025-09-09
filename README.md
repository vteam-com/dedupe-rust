# Duplicate Image Finder

A high-performance command-line tool to find and manage duplicate images in your directories, written in Rust.

## Features

- 🚀 Blazing fast scanning using multi-threading with Rayon
- 🔍 Two-phase detection (quick hash + deep comparison)
- 📊 Detailed file statistics including extension-based grouping
- 🖼️ Supports multiple image formats (JPEG, PNG, GIF, BMP, TIFF, HEIC, HEIF, WebP)
- 📈 Real-time progress tracking with visual feedback
- 🎨 Color-coded console output for better readability
- 🔢 Batch processing for memory efficiency
- 📋 Clean, organized output with file type breakdown
- 🔍 Smart filtering by image dimensions before comparison
- 📝 Multiple output formats (Plain text and JSON)
- 🖥️ Cross-platform support (Windows, macOS, Linux)

## Installation

### Prerequisites
- Rust and Cargo (Rust's package manager) installed
- System dependencies for HEIC/HEIF support:
  - Windows: [vcpkg](https://vcpkg.io/) with `libheif`
  - macOS: `brew install libheif`
  - Linux (Debian/Ubuntu): `sudo apt-get install libheif-dev`

For detailed HEIC/HEIF testing instructions, see [HEIC_HEIF_TESTING.md](HEIC_HEIF_TESTING.md).

## Building from Source

### Prerequisites
- Rust and Cargo (Rust's package manager) installed
- System dependencies for HEIC/HEIF support

### Windows

1. Install build tools:
   ```powershell
   # Install Rust from https://rustup.rs/
   rustup install stable
   
   # Install vcpkg for HEIC/HEIF support
   git clone https://github.com/Microsoft/vcpkg.git
   .\vcpkg\bootstrap-vcpkg.bat
   .\vcpkg\vcpkg install libheif:x64-windows
   
   # Set environment variable
   $env:LIBHEIF_LIB_DIR = "path\to\vcpkg\installed\x64-windows\lib"
   $env:LIBHEIF_INCLUDE_DIR = "path\to\vcpkg\installed\x64-windows\include"
   ```

2. Build the release:
   ```powershell
   cargo build --release
   # Binary: .\target\release\dedupe_rust.exe
   ```

### macOS

1. Install dependencies:
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install required libraries
   brew install libheif pkg-config
   ```

2. Build the release:
   ```bash
   cargo build --release
   # Binary: ./target/release/dedupe_rust
   ```

### Linux (Debian/Ubuntu)

1. Install dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libheif-dev pkg-config
   ```

2. Build the release:
   ```bash
   cargo build --release
   # Binary: ./target/release/dedupe_rust
   ```

### Building a Release Package

To create a standalone release package:

```bash
# Install cargo-deb (for Debian/Ubuntu) or cargo-wix (for Windows)
cargo install cargo-deb  # Linux
cargo install cargo-wix  # Windows

# Build release
cargo build --release

# Create package
cargo deb  # For .deb package
# OR
cargo wix  # For Windows installer (requires WIX installed)
```

## Usage

### Basic Usage

```bash
# Scan current directory
./dedupe_rust

# Scan a specific directory
./dedupe_rust -d /path/to/images

# Windows
.\dedupe_rust.exe -d C:\Path\To\Images
```

### Development

For development and testing:

```bash
# Run with debug output
RUST_LOG=debug cargo run --release -- -d /path/to/images

# Windows (PowerShell)
$env:RUST_LOG="debug"
cargo run --release -- -d C:\Path\To\Images
```


### Command Line Options

```
USAGE:
    dedupe_rust [OPTIONS] --directory <DIRECTORY>

OPTIONS:
    -d, --directory <DIRECTORY>    Directory to scan (default: current directory)
    -b, --batch-size <BATCH_SIZE>  Number of images to process in each batch (default: 1000)
    -h, --help                     Print help
    -V, --version                  Print version
```

### Output Formats

The tool supports two output formats:

1. **Plain Text (Default)**
   - Human-readable format with grouped duplicates
   - Shows file paths and image dimensions
   - Example:
     ```
     ✨ Found 2 groups of duplicates in 1.23s
     
     Group 1 (3 files, 800x600):
       /path/to/image1.jpg
       /path/to/image2.jpg
       /path/to/image3.jpg
     
     Group 2 (2 files, 1920x1080):
       /path/to/image4.jpg
       /path/to/image5.jpg
     ```

2. **JSON**
   - Machine-readable format for programmatic use
   - Includes all information in a structured format
   - Example:
     ```json
     {
       "groups": [
         {
           "files": [
             "/path/to/image1.jpg",
             "/path/to/image2.jpg",
             "/path/to/image3.jpg"
           ],
           "dimensions": "800x600"
         },
         {
           "files": [
             "/path/to/image4.jpg",
             "/path/to/image5.jpg"
           ],
           "dimensions": "1920x1080"
         }
       ],
       "total_groups": 2,
       "execution_time": "00:00:01.23"
     }
     ```

### Sequential Diagram

![Sequential Diagram](docs/sequence_diagram.png)

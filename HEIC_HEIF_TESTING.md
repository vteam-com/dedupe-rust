# Testing HEIC/HEIF Support

This guide explains how to test HEIC/HEIF support in the duplicate image finder.

## Prerequisites

1. Install the required system dependencies:
   - On Windows: Install [vcpkg](https://vcpkg.io/en/getting-started.html) and run:
     ```
     vcpkg install libheif:x64-windows
     set VCPKG_ROOT=C:\path\to\vcpkg
     ```
   - On macOS: `brew install libheif`
   - On Linux (Debian/Ubuntu): `sudo apt-get install libheif-dev`

## Testing with a Sample HEIC File

1. Obtain a sample HEIC file or convert an existing image to HEIC format.

2. Build the test tool with HEIC support:
   ```
   cargo build --release --features=heic
   ```

3. Test the HEIC support by running:
   ```
   cargo run --release --bin test_heic_support -- path/to/your/image.heic
   ```

4. If successful, the script will:
   - Display the image dimensions
   - Save a PNG version as `test_output.png` for verification

## Troubleshooting

- If you see an error about missing libraries, ensure you've installed the system dependencies.
- On Windows, make sure the `VCPKG_ROOT` environment variable is set correctly.
- For other issues, check the error message for specific details.

## Implementation Details

The HEIC/HEIF support is implemented using the `libheif-rs` crate, which provides bindings to the `libheif` library. The implementation includes:

- HEIC/HEIF file detection
- Image decoding and conversion to a common format
- Integration with the existing image processing pipeline

## Building with HEIC Support

To build the application with HEIC support, use:

```
cargo build --release --features=heic
```

Or to run directly:

```
cargo run --release --features=heic -- <arguments>
```

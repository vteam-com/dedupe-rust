use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;
use anyhow::Result;
use libheif_rs::{LibHeif, HeifContext};
use png;

/// Gets the dimensions of an image file by delegating to format-specific functions.
pub fn get_dimensions(path: &Path) -> Option<(u32, u32)> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    // Handle formats with optimized dimension detection first
    match ext.as_str() {
        "jpg" | "jpeg" => get_jpeg_dimensions(path),
        "png" => get_png_dimensions(path),
        "gif" => get_gif_dimensions(path),
        "webp" => get_webp_dimensions(path),
        "heic" | "heif" => get_heic_dimensions(path).unwrap_or(None),
        _ => None,
    }
}

/// Gets the dimensions of a JPEG file by reading minimal header data.
fn get_jpeg_dimensions(path: &Path) -> Option<(u32, u32)> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return None,
    };
    
    let mut buffer = [0u8; 2];
    if file.read_exact(&mut buffer).is_err() || buffer != [0xFF, 0xD8] {
        return None; // Not a valid JPEG file
    }
    
    let mut buffer = [0u8; 4];
    loop {
        if file.read_exact(&mut buffer[0..2]).is_err() {
            return None;
        }
        
        if buffer[0] != 0xFF {
            continue;
        }
        
        while buffer[1] == 0xFF {
            if file.read_exact(&mut buffer[1..2]).is_err() {
                return None;
            }
        }
        
        // Check for SOF (Start of Frame) marker
        if (0xC0..=0xCF).contains(&buffer[1]) && buffer[1] != 0xC4 && buffer[1] != 0xC8 {
            if file.read_exact(&mut buffer[..4]).is_err() {
                return None;
            }
            let height = u16::from_be_bytes([buffer[0], buffer[1]]) as u32;
            let width = u16::from_be_bytes([buffer[2], buffer[3]]) as u32;
            return Some((width, height));
        } else {
            // Skip this marker section
            if file.read_exact(&mut buffer[..2]).is_err() {
                return None;
            }
            let len = u16::from_be_bytes([buffer[0], buffer[1]]) as u64 - 2;
            if file.seek(std::io::SeekFrom::Current(len as i64)).is_err() {
                return None;
            }
        }
    }
}

/// Gets the dimensions of a PNG file by reading just the header.
fn get_png_dimensions(path: &Path) -> Option<(u32, u32)> {
    use std::io::BufReader;
    
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return None,
    };
    
    let decoder = png::Decoder::new(BufReader::new(file));
    let reader = match decoder.read_info() {
        Ok(reader) => reader,
        Err(_) => return None,
    };
    
    Some((reader.info().width, reader.info().height))
}

/// Gets the dimensions of a GIF file by reading just the header.
fn get_gif_dimensions(path: &Path) -> Option<(u32, u32)> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return None,
    };
    
    let mut header = [0u8; 10];
    if file.read_exact(&mut header).is_err() {
        return None;
    }
    
    // Check GIF signature
    if &header[0..3] != b"GIF" {
        return None;
    }
    
    let width = u16::from_le_bytes([header[6], header[7]]) as u32;
    let height = u16::from_le_bytes([header[8], header[9]]) as u32;
    
    Some((width, height))
}

/// Gets the dimensions of a WebP file by reading the header.
fn get_webp_dimensions(path: &Path) -> Option<(u32, u32)> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return None,
    };
    
    let mut header = [0u8; 30];
    if file.read_exact(&mut header).is_err() {
        return None;
    }
    
    // Check WebP signature
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WEBP" {
        return None;
    }
    
    // Check for VP8 or VP8L/VP8X format
    if &header[12..16] == b"VP8 " && header[20] == 0x2A {
        // VP8 (lossy) format
        let width = u16::from_le_bytes([header[26], header[27] & 0x3F]) as u32;
        let height = u16::from_le_bytes([header[28], header[29] & 0x3F]) as u32;
        return Some((width, height));
    } else if &header[12..16] == b"VP8L" {
        // VP8L (lossless) format
        if header[20] != 0x2F {
            return None;
        }
        let b1 = header[21] as u32;
        let b2 = header[22] as u32;
        let b3 = header[23] as u32;
        let b4 = header[24] as u32;
        
        let width = (b1 | ((b2 & 0x3F) << 8)) + 1;
        let height = (((b2 >> 6) | (b3 << 2) | ((b4 & 0x03) << 10))) + 1;
        return Some((width, height));
    } else if &header[12..16] == b"VP8X" {
        // VP8X (extended) format
        let width = 1 + (((header[24] as u32) | ((header[25] as u32) << 8) | ((header[26] as u32) << 16)) & 0xFFFFFF);
        let height = 1 + (((header[27] as u32) | ((header[28] as u32) << 8) | ((header[29] as u32) << 16)) & 0xFFFFFF);
        return Some((width, height));
    }
    
    None
}

/// Gets the dimensions of a HEIC/HEIF image without fully decoding it
pub fn get_heic_dimensions<P: AsRef<Path>>(path: P) -> Result<Option<(u32, u32)>> {
    let _lib = LibHeif::new();
    let ctx = match HeifContext::read_from_file(path.as_ref().to_str().unwrap()) {
        Ok(ctx) => ctx,
        Err(_) => return Ok(None),
    };
    
    let handle = match ctx.primary_image_handle() {
        Ok(handle) => handle,
        Err(_) => return Ok(None),
    };
    
    Ok(Some((handle.width() as u32, handle.height() as u32)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_get_dimensions() {
        // You can add test cases here
    }
}

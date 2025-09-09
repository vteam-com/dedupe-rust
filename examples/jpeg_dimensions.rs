use std::path::Path;
#[path = "../src/dimensions.rs"]
mod dimensions;
use dimensions::get_jpeg_dimensions;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: cargo run --example jpeg_dimensions -- path/to/image.jpg");
        std::process::exit(1);
    }

    let path = Path::new(&args[1]);
    if let Some((width, height)) = get_jpeg_dimensions(path) {
        println!("Image dimensions: {} x {}", width, height);
    } else {
        println!("Could not determine image dimensions");
    }
}

use crate::app::calculate::ProgressMsg;

use image::imageops;
use serde::Deserialize;
use serde::Serialize;
use uuid::Uuid;

use std::error::Error;
use std::io::{Error as IoError, ErrorKind};

// pub(crate) fn save_result(
//     target: image::SourceImg,
//     base_name: String,
//     source: image::SourceImg,
//     assignments: Vec<usize>,
//     img: image::SourceImg,
// ) -> Result<String, Box<dyn Error>> {
//     let mut dir_name = base_name.clone();
//     let mut counter = 1;
//     while std::path::Path::new(&format!("./presets/{}", dir_name)).exists() {
//         dir_name = format!("{}_{}", base_name, counter);
//         counter += 1;
//     }
//     std::fs::create_dir_all(format!("./presets/{}", dir_name))?;
//     img.save(format!("./presets/{}/output.png", dir_name))?;
//     source.save(format!("./presets/{}/source.png", dir_name))?;
//     target.save(format!("./presets/{}/target.png", dir_name))?;
//     std::fs::write(
//         format!("./presets/{}/assignments.json", dir_name),
//         serialize_assignments(assignments),
//     )?;
//     Ok(dir_name)
// }

pub trait ProgressSink {
    fn send(&mut self, msg: ProgressMsg);
}
// Native-friendly adapter
impl ProgressSink for std::sync::mpsc::SyncSender<ProgressMsg> {
    fn send(&mut self, msg: ProgressMsg) {
        let _ = std::sync::mpsc::SyncSender::send(self, msg);
    }
}

// Allow using closures as progress sinks in WASM
impl<T> ProgressSink for T
where
    T: FnMut(crate::app::calculate::ProgressMsg),
{
    fn send(&mut self, msg: crate::app::calculate::ProgressMsg) {
        self(msg);
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn get_images(
    source: SourceImg,
    settings: &GenerationSettings,
) -> Result<(Vec<(u8, u8, u8)>, Vec<(u8, u8, u8)>, Vec<i64>), Box<dyn Error>> {
    validate_sidelen(settings.sidelen)?;
    let source = settings.source_crop_scale.apply(&source, settings.sidelen);
    let source_pixels = source
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();

    let (target, weights) = settings.get_target()?;
    let target_pixels = target
        .pixels()
        .map(|p| (p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    if source_pixels.len() != target_pixels.len() || weights.len() != target_pixels.len() {
        return Err(IoError::new(
            ErrorKind::InvalidData,
            "source, target, and weight image dimensions do not match",
        )
        .into());
    }
    Ok((source_pixels, target_pixels, weights))
}

/// Decode the raw image representation used by presets without relying on type
/// inference or `unwrap`. Older drawing presets stored RGBA bytes while normal
/// presets store RGB, so accepting both formats also keeps those presets usable.
pub(crate) fn rgb_image_from_raw(
    width: u32,
    height: u32,
    data: Vec<u8>,
) -> Result<SourceImg, Box<dyn Error>> {
    if width == 0 || height == 0 {
        return Err(
            IoError::new(ErrorKind::InvalidInput, "image dimensions must be non-zero").into(),
        );
    }

    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "image dimensions are too large"))?;
    let rgb_len = pixels
        .checked_mul(3)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "RGB image is too large"))?;
    let rgba_len = pixels
        .checked_mul(4)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "RGBA image is too large"))?;

    if data.len() == rgb_len {
        return image::ImageBuffer::from_raw(width, height, data).ok_or_else(|| {
            IoError::new(ErrorKind::InvalidData, "invalid RGB image buffer").into()
        });
    }
    if data.len() == rgba_len {
        let rgba = image::RgbaImage::from_raw(width, height, data)
            .ok_or_else(|| IoError::new(ErrorKind::InvalidData, "invalid RGBA image buffer"))?;
        return Ok(image::DynamicImage::ImageRgba8(rgba).to_rgb8());
    }

    Err(IoError::new(
        ErrorKind::InvalidData,
        format!(
            "invalid image byte length: expected {rgb_len} RGB or {rgba_len} RGBA bytes, got {}",
            data.len()
        ),
    )
    .into())
}

pub(crate) fn validate_sidelen(sidelen: u32) -> Result<(), Box<dyn Error>> {
    // Coordinates are currently stored as u16/i16 in the matching algorithms.
    // A conservative upper bound also prevents malformed persisted settings from
    // allocating unexpectedly large images.
    if !(1..=256).contains(&sidelen) {
        return Err(IoError::new(
            ErrorKind::InvalidInput,
            format!("resolution must be between 1 and 256, got {sidelen}"),
        )
        .into());
    }
    Ok(())
}

/// Convert the UI's resolution-independent structure value to the coordinate
/// system used by the heuristic. Spatial distance is squared before it is
/// weighted, so the weight must scale with the inverse square of resolution.
pub(crate) fn normalized_proximity_importance(value: i64, sidelen: u32) -> i64 {
    if value <= 0 || sidelen == 0 {
        return 0;
    }
    let base = 128_i128;
    let side = sidelen as i128;
    let numerator = value as i128 * base * base;
    let rounded = (numerator + side * side / 2) / (side * side);
    rounded.clamp(1, i64::MAX as i128) as i64
}

#[derive(Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct CropScale {
    pub x: f32,     // -1: all left, 0: center, 1: all right
    pub y: f32,     // -1: all top, 0: center, 1: all bottom
    pub scale: f32, // 1: fit within frame, >1: zoom in, <1: not allowed
}

impl CropScale {
    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            scale: 1.0,
        }
    }

    pub fn apply(&self, img: &SourceImg, sidelen: u32) -> SourceImg {
        let (w, h) = img.dimensions();

        let s = self.scale.max(1.0);

        let base_side = w.min(h) as f32;
        let mut crop_side = (base_side / s).floor().max(1.0);

        crop_side = crop_side.min(w as f32).min(h as f32);

        let max_x_off = (w as f32 - crop_side).max(0.0);
        let max_y_off = (h as f32 - crop_side).max(0.0);

        let xn = (self.x.clamp(-1.0, 1.0) + 1.0) * 0.5;
        let yn = (self.y.clamp(-1.0, 1.0) + 1.0) * 0.5;

        let x0 = (xn * max_x_off).floor() as u32;
        let y0 = (yn * max_y_off).floor() as u32;
        let cs = crop_side as u32;
        let cropped = imageops::crop_imm(img, x0, y0, cs, cs).to_image();

        if cs == sidelen {
            cropped
        } else {
            imageops::resize(&cropped, sidelen, sidelen, imageops::FilterType::Lanczos3)
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Algorithm {
    Optimal,
    Genetic,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GenerationSettings {
    pub id: Uuid,
    pub name: String,

    pub proximity_importance: i64,
    pub algorithm: Algorithm,

    pub sidelen: u32,
    custom_target: Option<(u32, u32, Vec<u8>)>,
    pub target_crop_scale: CropScale,
    pub source_crop_scale: CropScale,
}

pub type SourceImg = image::RgbImage;

impl GenerationSettings {
    pub fn default(id: Uuid, name: String) -> Self {
        Self {
            name,
            proximity_importance: 13, // 20
            algorithm: Algorithm::Genetic,
            id,
            // Browser-friendly default. Users can still raise this in advanced settings.
            sidelen: 64,
            custom_target: None,
            target_crop_scale: CropScale::identity(),
            source_crop_scale: CropScale::identity(),
        }
    }

    pub fn get_target(&self) -> Result<(SourceImg, Vec<i64>), Box<dyn std::error::Error>> {
        let target = self.get_raw_target();
        let target = self.target_crop_scale.apply(&target, self.sidelen);
        let weights = if self.custom_target.is_some() {
            vec![255; (self.sidelen * self.sidelen) as usize] // uniform weights
        } else {
            let target_weights =
                image::load_from_memory(include_bytes!("weights256.png"))?.to_rgb8();
            let target_weights = self.target_crop_scale.apply(&target_weights, self.sidelen);
            load_weights(target_weights)
        };

        Ok((target, weights))
    }

    pub(crate) fn get_raw_target(&self) -> SourceImg {
        if let Some((w, h, data)) = &self.custom_target {
            if let Ok(image) = rgb_image_from_raw(*w, *h, data.clone()) {
                return image;
            }
        }
        // This image is compiled into the binary and covered by build/tests. If
        // it ever becomes corrupt, return a valid one-pixel target rather than
        // panicking during app startup.
        image::load_from_memory(include_bytes!("target256.png"))
            .map(|image| image.to_rgb8())
            .unwrap_or_else(|_| SourceImg::from_pixel(1, 1, image::Rgb([0, 0, 0])))
    }

    pub fn clone_with_new_id(&self) -> Self {
        let mut new = self.clone();
        new.id = Uuid::new_v4();

        new.name = if let Some(v_pos) = self.name.rfind(" v") {
            let potential_version = &self.name[v_pos + 2..];
            if let Ok(version) = potential_version.parse::<u32>() {
                let base_name = &self.name[..v_pos];
                format!("{} v{}", base_name, version + 1)
            } else {
                format!("{} v2", self.name)
            }
        } else {
            format!("{} v2", self.name)
        };

        new
    }
}

pub fn load_weights(source: SourceImg) -> Vec<i64> {
    let (width, height) = source.dimensions();
    let mut weights = vec![0; (width * height) as usize];
    for (x, y, pixel) in source.enumerate_pixels() {
        let weight = pixel[0] as i64;
        weights[(y * width + x) as usize] = weight;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_image_decoder_accepts_rgb_and_rgba() {
        let rgb = rgb_image_from_raw(1, 1, vec![10, 20, 30]).unwrap();
        assert_eq!(rgb.get_pixel(0, 0).0, [10, 20, 30]);

        let rgba = rgb_image_from_raw(1, 1, vec![40, 50, 60, 7]).unwrap();
        assert_eq!(rgba.get_pixel(0, 0).0, [40, 50, 60]);
    }

    #[test]
    fn raw_image_decoder_rejects_bad_lengths_and_dimensions() {
        assert!(rgb_image_from_raw(0, 1, Vec::new()).is_err());
        assert!(rgb_image_from_raw(2, 2, vec![0; 5]).is_err());
    }

    #[test]
    fn proximity_weight_is_normalized_by_resolution_squared() {
        assert_eq!(normalized_proximity_importance(13, 128), 13);
        assert_eq!(normalized_proximity_importance(13, 64), 52);
        assert_eq!(normalized_proximity_importance(13, 256), 3);
        assert_eq!(normalized_proximity_importance(0, 64), 0);
    }
}

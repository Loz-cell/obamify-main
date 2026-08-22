use serde::{Deserialize, Serialize};

pub const MIN_PRESET_SIDE: u32 = 32;
pub const MAX_PRESET_SIDE: u32 = 512;
pub const MAX_PRESET_NAME_CHARS: usize = 64;
#[cfg(target_arch = "wasm32")]
pub const MAX_PERSISTED_PRESET_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Serialize, Deserialize)]
pub struct Preset {
    pub inner: UnprocessedPreset,
    pub assignments: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct UnprocessedPreset {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub source_img: Vec<u8>,
}

impl UnprocessedPreset {
    pub fn validate(&self) -> Result<(), String> {
        let name = self.name.trim();
        if name.is_empty() {
            return Err("preset name is empty".to_owned());
        }
        if name.chars().count() > MAX_PRESET_NAME_CHARS {
            return Err(format!(
                "preset name is longer than {MAX_PRESET_NAME_CHARS} characters"
            ));
        }
        if self.width == 0 || self.height == 0 {
            return Err("preset dimensions cannot be zero".to_owned());
        }
        if self.width > MAX_PRESET_SIDE || self.height > MAX_PRESET_SIDE {
            return Err(format!(
                "preset dimensions cannot exceed {MAX_PRESET_SIDE} pixels"
            ));
        }

        let expected_len = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| "preset dimensions overflow".to_owned())?;
        if self.source_img.len() != expected_len {
            return Err(format!(
                "preset RGB data has {} bytes; expected {expected_len}",
                self.source_img.len()
            ));
        }
        Ok(())
    }
}

impl Preset {
    pub fn validate(&self) -> Result<(), String> {
        self.inner.validate()?;
        if self.inner.width != self.inner.height {
            return Err("processed preset image must be square".to_owned());
        }
        if self.inner.width < MIN_PRESET_SIDE {
            return Err(format!(
                "processed preset side cannot be below {MIN_PRESET_SIDE} pixels"
            ));
        }
        let pixel_count = (self.inner.width as usize)
            .checked_mul(self.inner.height as usize)
            .ok_or_else(|| "preset dimensions overflow".to_owned())?;
        if self.assignments.len() != pixel_count {
            return Err(format!(
                "preset has {} assignments; expected {pixel_count}",
                self.assignments.len()
            ));
        }

        let mut seen = vec![false; pixel_count];
        for &source_index in &self.assignments {
            if source_index >= pixel_count {
                return Err(format!(
                    "preset assignment {source_index} is outside 0..{pixel_count}"
                ));
            }
            if std::mem::replace(&mut seen[source_index], true) {
                return Err(format!(
                    "preset assignment {source_index} occurs more than once"
                ));
            }
        }
        Ok(())
    }

    /// Estimate the JSON/local-storage representation without allocating a
    /// second serialized copy.
    pub fn estimated_storage_bytes(&self) -> usize {
        fn decimal_digits(mut value: usize) -> usize {
            let mut digits = 1;
            while value >= 10 {
                value /= 10;
                digits += 1;
            }
            digits
        }

        let image_bytes = self.inner.source_img.iter().fold(0usize, |total, value| {
            total.saturating_add(decimal_digits(*value as usize) + 1)
        });
        let assignment_bytes = self.assignments.iter().fold(0usize, |total, value| {
            total.saturating_add(decimal_digits(*value) + 1)
        });
        image_bytes
            .saturating_add(assignment_bytes)
            .saturating_add(self.inner.name.len())
            .saturating_add(256)
    }
}

/// Keep only structurally safe presets and cap the amount written to browser
/// storage. Newest presets are preferred; the first preset is retained when it
/// fits because it is normally the bundled fallback portrait.
#[cfg(target_arch = "wasm32")]
pub fn presets_for_storage(presets: &[Preset]) -> (Vec<Preset>, usize) {
    let mut valid: Vec<(usize, &Preset)> = presets
        .iter()
        .enumerate()
        .filter(|(_, preset)| preset.validate().is_ok())
        .collect();
    valid.sort_by_key(|(index, _)| {
        if *index == 0 {
            (0, std::cmp::Reverse(0))
        } else {
            (1, std::cmp::Reverse(*index))
        }
    });

    let mut used = 0usize;
    let mut selected = Vec::new();
    for (index, preset) in valid {
        let estimate = preset.estimated_storage_bytes();
        if used.saturating_add(estimate) <= MAX_PERSISTED_PRESET_BYTES {
            used = used.saturating_add(estimate);
            selected.push((index, (*preset).clone()));
        }
    }
    selected.sort_by_key(|(index, _)| *index);
    let omitted = presets.len().saturating_sub(selected.len());
    (
        selected.into_iter().map(|(_, preset)| preset).collect(),
        omitted,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_preset() -> Preset {
        let side = MIN_PRESET_SIDE;
        let pixels = (side * side) as usize;
        Preset {
            inner: UnprocessedPreset {
                name: "test".to_owned(),
                width: side,
                height: side,
                source_img: vec![0; pixels * 3],
            },
            assignments: (0..pixels).collect(),
        }
    }

    #[test]
    fn validates_well_formed_preset() {
        assert!(valid_preset().validate().is_ok());
    }

    #[test]
    fn rejects_duplicate_assignment() {
        let mut preset = valid_preset();
        preset.assignments[1] = 0;
        assert!(preset.validate().is_err());
    }

    #[test]
    fn rejects_wrong_rgb_length() {
        let mut preset = valid_preset();
        preset.inner.source_img.pop();
        assert!(preset.validate().is_err());
    }
}

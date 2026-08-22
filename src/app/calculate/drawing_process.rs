use crate::app::SeedColor;
use crate::app::calculate;
use crate::app::preset::UnprocessedPreset;

use std::error::Error;

use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::mpsc;

use super::ProgressMsg;

use super::GenerationSettings;

#[derive(Clone, Copy)]
pub struct PixelData {
    pub stroke_id: u32,
    pub last_edited: u32,
}
impl PixelData {
    pub(crate) fn init_canvas(frame_count: u32) -> Vec<PixelData> {
        vec![
            PixelData {
                stroke_id: 0,
                last_edited: frame_count
            };
            DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE
        ]
    }
}

pub(crate) struct DrawingWorkerEvent {
    pub drawing_id: u32,
    pub msg: ProgressMsg,
}

pub const DRAWING_CANVAS_SIZE: usize = 128;
const DRAWING_SWAPS_PER_PIXEL: usize = 2;
const DRAWING_MAX_SWAPS_PER_GENERATION: usize = 32_768;
const DRAWING_CANCEL_CHECK_INTERVAL: usize = 2_048;

use super::heuristic;

#[derive(Clone, Copy)]
pub(crate) struct DrawingPixel {
    pub(crate) src_x: u16,
    pub(crate) src_y: u16,
    pub(crate) h: i64, // current heuristic value
}

impl DrawingPixel {
    pub(crate) fn new(src_x: u16, src_y: u16, h: i64) -> Self {
        Self { src_x, src_y, h }
    }

    pub(crate) fn update_heuristic(&mut self, new_h: i64) {
        self.h = new_h;
    }

    #[inline(always)]
    pub(crate) fn calc_drawing_heuristic(
        &self,
        target_pos: (u16, u16),
        target_col: (u8, u8, u8),
        weight: i64,
        colors: &[SeedColor],
        proximity_importance: i64,
    ) -> i64 {
        heuristic(
            (self.src_x, self.src_y),
            target_pos,
            {
                let rgba =
                    colors[self.src_y as usize * DRAWING_CANVAS_SIZE + self.src_x as usize].rgba;
                (
                    (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                )
            },
            target_col,
            weight,
            proximity_importance,
        )
    }
}

pub(crate) const STROKE_REWARD: i64 = -10000000000;

type NeighbourMap = Vec<[Option<usize>; 4]>;

fn build_neighbour_map(side: usize) -> NeighbourMap {
    let mut map = vec![[None; 4]; side * side];
    for y in 0..side {
        for x in 0..side {
            let index = y * side + x;
            map[index] = [
                y.checked_sub(1).map(|ny| ny * side + x),
                x.checked_sub(1).map(|nx| y * side + nx),
                (x + 1 < side).then_some(y * side + x + 1),
                (y + 1 < side).then_some((y + 1) * side + x),
            ];
        }
    }
    map
}

fn source_index(pixel: DrawingPixel, sidelen: usize) -> Option<usize> {
    let x = pixel.src_x as usize;
    let y = pixel.src_y as usize;
    (x < sidelen && y < sidelen).then_some(y * sidelen + x)
}

fn stroke_reward(
    newpos: usize,
    oldpos: usize,
    pixel_data: &[PixelData],
    pixels: &[DrawingPixel],
    frame_count: u32,
    neighbours: &NeighbourMap,
) -> i64 {
    let Some(pixel) = pixels.get(oldpos).copied() else {
        return 0;
    };
    let Some(source_idx) = source_index(pixel, DRAWING_CANVAS_SIZE) else {
        return 0;
    };
    let Some(data) = pixel_data.get(source_idx).copied() else {
        return 0;
    };
    let stroke_id = data.stroke_id;
    if stroke_id == 0 || frame_count < data.last_edited {
        return 0;
    }

    let Some(neighbours) = neighbours.get(newpos) else {
        return 0;
    };
    for neighbour in neighbours.iter().flatten() {
        let Some(neighbour_pixel) = pixels.get(*neighbour).copied() else {
            continue;
        };
        let Some(neighbour_source) = source_index(neighbour_pixel, DRAWING_CANVAS_SIZE) else {
            continue;
        };
        if pixel_data
            .get(neighbour_source)
            .is_some_and(|neighbour_data| neighbour_data.stroke_id == stroke_id)
        {
            return STROKE_REWARD;
        }
    }
    0
}

#[allow(clippy::too_many_arguments)]
pub fn drawing_process_genetic(
    source: UnprocessedPreset,
    mut settings: GenerationSettings,
    tx: mpsc::SyncSender<DrawingWorkerEvent>,
    colors: Arc<std::sync::RwLock<Vec<SeedColor>>>,
    pixel_data: Arc<std::sync::RwLock<Vec<PixelData>>>,
    frame_count: u32,
    my_id: u32,
    current_id: Arc<AtomicU32>,
) -> Result<(), Box<dyn Error>> {
    // Drawing state, seed colors and PixelData are all defined on the fixed
    // 128x128 canvas. GenerationSettings::default is intentionally lower for
    // web transforms, so do not let that web default corrupt drawing indices.
    settings.sidelen = DRAWING_CANVAS_SIZE as u32;
    let source_img = calculate::util::rgb_image_from_raw(
        source.width,
        source.height,
        source.source_img.clone(),
    )?;
    let (source_pixels, target_pixels, weights) =
        calculate::util::get_images(source_img, &settings)?;
    let proximity_importance = calculate::util::normalized_proximity_importance(
        settings.proximity_importance,
        settings.sidelen,
    );
    let expected_pixels = DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE;
    if source_pixels.len() != expected_pixels {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "drawing source must resolve to a 128x128 RGB image",
        )
        .into());
    }

    let neighbours = build_neighbour_map(DRAWING_CANVAS_SIZE);
    let mut color_snapshot = colors
        .read()
        .map_err(|_| std::io::Error::other("drawing color state is unavailable"))?
        .clone();
    if color_snapshot.len() != expected_pixels {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "drawing color state has the wrong size",
        )
        .into());
    }

    let mut pixels = {
        source_pixels
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let x = (i as u32 % settings.sidelen) as u16;
                let y = (i as u32 / settings.sidelen) as u16;
                let mut p = DrawingPixel::new(x, y, 0);
                let h = p.calc_drawing_heuristic(
                    (x, y),
                    target_pixels[i],
                    weights[i],
                    &color_snapshot,
                    proximity_importance,
                );
                p.update_heuristic(h);
                p
            })
            .collect::<Vec<_>>()
    };

    let mut rng = frand::Rand::with_seed(12345);
    fn max_dist(age: u32) -> u32 {
        ((((DRAWING_CANVAS_SIZE / 4) as f32) * (0.99f32).powi(age.min(i32::MAX as u32) as i32 / 30))
            .round() as u32)
            .max(1)
    }

    let swaps_per_generation =
        (DRAWING_SWAPS_PER_PIXEL * pixels.len()).clamp(1, DRAWING_MAX_SWAPS_PER_GENERATION);
    let mut pixel_data_snapshot = vec![
        PixelData {
            stroke_id: 0,
            last_edited: frame_count,
        };
        expected_pixels
    ];
    let mut generation = 0_u32;
    let mut idle_generations = 0_u32;

    loop {
        if my_id != current_id.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tx.try_send(DrawingWorkerEvent {
                drawing_id: my_id,
                msg: ProgressMsg::Cancelled,
            });
            return Ok(());
        }
        {
            let current_colors = colors
                .read()
                .map_err(|_| std::io::Error::other("drawing color state is unavailable"))?;
            if current_colors.len() != expected_pixels {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "drawing color state changed to an invalid size",
                )
                .into());
            }
            color_snapshot.clone_from_slice(&current_colors);
        }
        {
            let current_pixel_data = pixel_data
                .read()
                .map_err(|_| std::io::Error::other("drawing pixel state is unavailable"))?;
            if current_pixel_data.len() != expected_pixels {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "drawing pixel state has the wrong size",
                )
                .into());
            }
            pixel_data_snapshot.clone_from_slice(&current_pixel_data);
        }
        let current_frame = frame_count.saturating_add(generation).max(
            pixel_data_snapshot
                .iter()
                .map(|data| data.last_edited)
                .max()
                .unwrap_or(frame_count),
        );
        let mut swaps_made = 0;

        for attempt in 0..swaps_per_generation {
            if attempt % DRAWING_CANCEL_CHECK_INTERVAL == 0
                && my_id != current_id.load(std::sync::atomic::Ordering::Relaxed)
            {
                let _ = tx.try_send(DrawingWorkerEvent {
                    drawing_id: my_id,
                    msg: ProgressMsg::Cancelled,
                });
                return Ok(());
            }
            let apos = rng.gen_range(0..pixels.len() as u64) as usize;
            let ax = apos as u16 % settings.sidelen as u16;
            let ay = apos as u16 / settings.sidelen as u16;

            //let stroke_id = pixel_data[apos].stroke_id as usize;
            let Some(source_a) = source_index(pixels[apos], DRAWING_CANVAS_SIZE) else {
                continue;
            };
            let max_dist_a =
                max_dist(current_frame.saturating_sub(pixel_data_snapshot[source_a].last_edited));

            let bx = (ax as i16 + rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let by = (ay as i16 + rng.gen_range(-(max_dist_a as i16)..(max_dist_a as i16 + 1)))
                .clamp(0, settings.sidelen as i16 - 1) as u16;
            let bpos = by as usize * settings.sidelen as usize + bx as usize;

            let Some(source_b) = source_index(pixels[bpos], DRAWING_CANVAS_SIZE) else {
                continue;
            };
            let max_dist_b =
                max_dist(current_frame.saturating_sub(pixel_data_snapshot[source_b].last_edited));
            if (bx as i32 - ax as i32).abs() > max_dist_b as i32
                || (by as i32 - ay as i32).abs() > max_dist_b as i32
            {
                continue;
            }

            let t_a = target_pixels[apos];
            let t_b = target_pixels[bpos];

            let old_a_h = pixels[apos].calc_drawing_heuristic(
                (ax, ay),
                t_a,
                weights[apos],
                &color_snapshot,
                proximity_importance,
            ) + stroke_reward(
                apos,
                apos,
                &pixel_data_snapshot,
                &pixels,
                current_frame,
                &neighbours,
            );
            let old_b_h = pixels[bpos].calc_drawing_heuristic(
                (bx, by),
                t_b,
                weights[bpos],
                &color_snapshot,
                proximity_importance,
            ) + stroke_reward(
                bpos,
                bpos,
                &pixel_data_snapshot,
                &pixels,
                current_frame,
                &neighbours,
            );

            let a_on_b_h = pixels[apos].calc_drawing_heuristic(
                (bx, by),
                t_b,
                weights[bpos],
                &color_snapshot,
                proximity_importance,
            ) + stroke_reward(
                bpos,
                apos,
                &pixel_data_snapshot,
                &pixels,
                current_frame,
                &neighbours,
            );

            let b_on_a_h = pixels[bpos].calc_drawing_heuristic(
                (ax, ay),
                t_a,
                weights[apos],
                &color_snapshot,
                proximity_importance,
            ) + stroke_reward(
                apos,
                bpos,
                &pixel_data_snapshot,
                &pixels,
                current_frame,
                &neighbours,
            );

            if old_a_h + old_b_h > a_on_b_h + b_on_a_h {
                // swap
                pixels.swap(apos, bpos);
                pixels[apos].update_heuristic(b_on_a_h);
                pixels[bpos].update_heuristic(a_on_b_h);
                swaps_made += 1;
            }
        }

        //println!("swaps made: {}", swaps_made);

        // let img = make_new_img(&source_pixels, &assignments, target.width());
        // if swaps_made < 10 || cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        //     let dir_name = save_result(target, base_name, source, assignments, img)?;
        //     tx.send(ProgressMsg::Done(PathBuf::from(format!(
        //         "./presets/{}",
        //         dir_name
        //     ))))?;
        //     return Ok(());
        // }
        // tx.send(ProgressMsg::UpdatePreview(img))?;
        if swaps_made > 0 {
            idle_generations = 0;
            let assignments = pixels
                .iter()
                .map(|p| p.src_y as usize * settings.sidelen as usize + p.src_x as usize)
                .collect::<Vec<_>>();
            let _ = tx.try_send(DrawingWorkerEvent {
                drawing_id: my_id,
                msg: ProgressMsg::UpdateAssignments(assignments),
            });
        } else {
            // Once the live canvas reaches a local optimum, back off instead of
            // burning an entire CPU core while waiting for the next brush edit.
            idle_generations = idle_generations.saturating_add(1);
            let backoff_ms = (1_u64 << idle_generations.min(9)).min(500);
            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
        }
        if my_id != current_id.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tx.try_send(DrawingWorkerEvent {
                drawing_id: my_id,
                msg: ProgressMsg::Cancelled,
            });
            return Ok(());
        }

        generation = generation.saturating_add(1);

        //max_dist = (max_dist as f32 * 0.99).max(4.0) as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_index_uses_the_pixels_source_coordinates() {
        let pixel = DrawingPixel::new(7, 3, 0);
        assert_eq!(source_index(pixel, DRAWING_CANVAS_SIZE), Some(391));
        assert_eq!(
            source_index(
                DrawingPixel::new(DRAWING_CANVAS_SIZE as u16, 0, 0),
                DRAWING_CANVAS_SIZE
            ),
            None
        );
    }

    #[test]
    fn neighbour_map_has_no_row_wrapping() {
        let map = build_neighbour_map(3);
        let corner = map[0].iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(corner, vec![1, 3]);
        let opposite_corner = map[8].iter().flatten().copied().collect::<Vec<_>>();
        assert_eq!(opposite_corner, vec![5, 7]);
    }

    #[test]
    fn stroke_reward_follows_source_pixels_after_reassignment() {
        let count = DRAWING_CANVAS_SIZE * DRAWING_CANVAS_SIZE;
        let mut pixels = (0..count)
            .map(|index| {
                DrawingPixel::new(
                    (index % DRAWING_CANVAS_SIZE) as u16,
                    (index / DRAWING_CANVAS_SIZE) as u16,
                    0,
                )
            })
            .collect::<Vec<_>>();
        pixels.swap(0, 5);
        pixels.swap(1, 6);

        let mut data = PixelData::init_canvas(0);
        data[5].stroke_id = 42;
        data[6].stroke_id = 42;
        let neighbours = build_neighbour_map(DRAWING_CANVAS_SIZE);
        assert_eq!(
            stroke_reward(0, 0, &data, &pixels, 0, &neighbours),
            STROKE_REWARD
        );
    }
}

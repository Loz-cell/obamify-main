use std::mem;

use image::ImageBuffer;

use crate::app::{SeedColor, SeedPos, calculate::util::rgb_image_from_raw, preset::Preset};

#[cfg(not(target_arch = "wasm32"))]
use crate::app::preset::UnprocessedPreset;

// const DST_FORCE: f32 = 0.2;
pub fn init_image(sidelen: u32, source: Preset) -> (u32, Vec<SeedPos>, Vec<SeedColor>, Sim) {
    let imgpath = square_image(
        rgb_image_from_raw(
            source.inner.width,
            source.inner.height,
            source.inner.source_img,
        )
        .unwrap_or_else(|_| image::RgbImage::from_pixel(1, 1, image::Rgb([0, 0, 0]))),
    );
    let assignments = source.assignments;

    let (seeds, colors, seeds_n) = init_colors(sidelen, imgpath);
    let mut sim = Sim::new(source.inner.name);
    sim.cells = vec![CellBody::new(0.0, 0.0, 0.0, 0.0, 0.0); seeds_n];

    sim.set_assignments(assignments, sidelen);
    for cell in &mut sim.cells {
        cell.dst_force = 0.13;
    }
    (seeds_n as u32, seeds, colors, sim)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn init_canvas(
    sidelen: u32,
    source: UnprocessedPreset,
) -> (u32, Vec<SeedPos>, Vec<SeedColor>, Sim) {
    let imgpath = square_image(
        rgb_image_from_raw(source.width, source.height, source.source_img)
            .unwrap_or_else(|_| image::RgbImage::from_pixel(1, 1, image::Rgb([0, 0, 0]))),
    );
    let assignments =
        (0..imgpath.width() as usize * imgpath.height() as usize).collect::<Vec<usize>>();

    let (seeds, colors, seeds_n) = init_colors(sidelen, imgpath);
    let mut sim = Sim::new(source.name);
    sim.cells = vec![CellBody::new(0.0, 0.0, 0.0, 0.0, 0.0); seeds_n];

    sim.set_assignments(assignments, sidelen);
    (seeds_n as u32, seeds, colors, sim)
}

fn square_image(source: image::RgbImage) -> image::RgbImage {
    let (width, height) = source.dimensions();
    if width == height {
        return source;
    }
    let side = width.min(height).max(1);
    let x = (width - side) / 2;
    let y = (height - side) / 2;
    image::imageops::crop_imm(&source, x, y, side, side).to_image()
}

fn init_colors(
    sidelen: u32,
    source: ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> (Vec<SeedPos>, Vec<SeedColor>, usize) {
    let mut seeds = Vec::new();
    let mut colors = Vec::new();

    let width = source.width() as usize;
    let height = source.height() as usize;

    debug_assert_eq!(width, height);

    let seeds_n = width * height;
    let pixelsize = sidelen as f32 / width as f32;

    for y in 0..height {
        for x in 0..width {
            let p = source.get_pixel(x as u32, y as u32);
            seeds.push(SeedPos {
                xy: [(x as f32 + 0.5) * pixelsize, (y as f32 + 0.5) * pixelsize],
            });
            colors.push(SeedColor {
                rgba: [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                    1.0,
                ],
            });
        }
    }
    (seeds, colors, seeds_n)
}

#[derive(Clone, Copy)]
pub struct CellBody {
    srcx: f32,
    srcy: f32,
    dstx: f32,
    dsty: f32,

    velx: f32,
    vely: f32,

    accx: f32,
    accy: f32,

    dst_force: f32,
    age: u32,
    stroke_id: u32,
}

const PERSONAL_SPACE: f32 = 0.95;
const MAX_VELOCITY: f32 = 6.0;
const ALIGNMENT_FACTOR: f32 = 0.8;

fn factor_curve(x: f32) -> f32 {
    (x * x * x).min(1000.0)
}

impl CellBody {
    fn new(srcx: f32, srcy: f32, dstx: f32, dsty: f32, dst_force: f32) -> Self {
        Self {
            srcx,
            srcy,
            dstx,
            dsty,
            dst_force,
            velx: 0.0,
            vely: 0.0,
            accx: 0.0,
            accy: 0.0,
            age: 0,
            stroke_id: 0,
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_age(&mut self, age: u32) {
        self.age = age;
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_dst_force(&mut self, force: f32) {
        self.dst_force = force;
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_stroke_id(&mut self, stroke_id: u32) {
        self.stroke_id = stroke_id;
    }

    fn update(&mut self, pos: &mut SeedPos) {
        self.velx += self.accx;
        self.vely += self.accy;

        self.accx = 0.0;
        self.accy = 0.0;

        self.velx *= 0.97;
        self.vely *= 0.97;

        // self.velx = self.velx.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        // self.vely = self.vely.clamp(-MAX_VELOCITY, MAX_VELOCITY);

        // pos.xy[0] += self.velx;
        // pos.xy[1] += self.vely;

        pos.xy[0] += self.velx.clamp(-MAX_VELOCITY, MAX_VELOCITY);
        pos.xy[1] += self.vely.clamp(-MAX_VELOCITY, MAX_VELOCITY);

        self.age += 1;
    }

    fn apply_dst_force(&mut self, pos: &SeedPos, sidelen: f32) {
        let elapsed = self.age as f32 / 60.0;
        let factor = if self.dst_force == 0.0 {
            0.1
        } else {
            factor_curve(elapsed * self.dst_force)
        };

        let dx = self.dstx - pos.xy[0];
        let dy = self.dsty - pos.xy[1];
        let dist = (dx * dx + dy * dy).sqrt();

        self.accx += (dx * dist * factor) / sidelen;
        self.accy += (dy * dist * factor) / sidelen;
    }

    fn apply_neighbour_force(&mut self, pos: &SeedPos, other: &SeedPos, pixel_size: f32) -> f32 {
        let dx = other.xy[0] - pos.xy[0];
        let dy = other.xy[1] - pos.xy[1];
        let dist = (dx * dx + dy * dy).sqrt();
        let personal_space = pixel_size * PERSONAL_SPACE;

        if dist > 0.0 && dist < personal_space {
            let weight = (1.0 / dist) * (personal_space - dist) / personal_space;
            self.accx -= dx * weight;
            self.accy -= dy * weight;
            weight
        } else if dist.abs() < f32::EPSILON {
            // if they are exactly on top of each other, push in a random direction
            let seed = (pos.xy[0].to_bits() as u64) ^ ((pos.xy[1].to_bits() as u64) << 32);
            let mut rng = frand::Rand::with_seed(seed);

            let r1 = rng.gen_range(0.0..1.0);
            let r2 = rng.gen_range(0.0..1.0);

            self.accx += (r1 - 0.5) * 0.1;
            self.accy += (r2 - 0.5) * 0.1;
            0.0
        } else {
            0.0
        }
    }

    fn apply_wall_force(&mut self, pos: &SeedPos, sidelen: f32, pixel_size: f32) {
        let personal_space = pixel_size * PERSONAL_SPACE * 0.5;

        if pos.xy[0] < personal_space {
            self.accx += (personal_space - pos.xy[0]) / personal_space;
        } else if pos.xy[0] > sidelen - personal_space {
            self.accx -= (pos.xy[0] - (sidelen - personal_space)) / personal_space;
        }

        if pos.xy[1] < personal_space {
            self.accy += (personal_space - pos.xy[1]) / personal_space;
        } else if pos.xy[1] > sidelen - personal_space {
            self.accy -= (pos.xy[1] - (sidelen - personal_space)) / personal_space;
        }
    }

    fn apply_stroke_attraction(&mut self, i: SeedPos, other_cell: SeedPos, weight: f32) {
        self.accx += (other_cell.xy[0] - i.xy[0]) * weight * 0.8;
        self.accy += (other_cell.xy[1] - i.xy[1]) * weight * 0.8;
    }
}

pub struct Sim {
    //elapsed_frames: u32,
    pub cells: Vec<CellBody>,
    name: String,
    reversed: bool,
    // Reused every frame. Keeping the buckets avoids thousands of small heap
    // allocations during a transformation animation.
    spatial_grid: Vec<Vec<usize>>,
    assignment_seen: Vec<bool>,
}

impl Sim {
    pub fn new(name: String) -> Self {
        Self {
            cells: Vec::new(),
            //elapsed_frames: 0,
            name,
            reversed: false,
            spatial_grid: Vec::new(),
            assignment_seen: Vec::new(),
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    // pub fn source_path(&self) -> PathBuf {
    //     self.source.clone()
    // }

    pub fn switch(&mut self) {
        for cell in &mut self.cells {
            mem::swap(&mut cell.srcx, &mut cell.dstx);
            mem::swap(&mut cell.srcy, &mut cell.dsty);
            cell.age = 0;
        }
        self.reversed = !self.reversed;
    }

    pub fn update(&mut self, positions: &mut [SeedPos], sidelen: u32) {
        if self.cells.is_empty() || positions.len() != self.cells.len() || sidelen == 0 {
            return;
        }
        let grid_side = (self.cells.len() as f32).sqrt().ceil() as usize;
        let grid_size = grid_side as f32;
        let pixel_size = sidelen as f32 / grid_size;
        //dbg!(grid_size, pixel_size);

        let bucket_count = grid_side * grid_side;
        self.spatial_grid.resize_with(bucket_count, Vec::new);
        for bucket in &mut self.spatial_grid {
            bucket.clear();
        }

        for (i, p) in positions.iter().enumerate() {
            let x = p.xy[0] / pixel_size;
            let y = p.xy[1] / pixel_size;

            let index = (y.floor().clamp(0.0, grid_size - 1.0) * grid_size) as usize
                + (x.floor().clamp(0.0, grid_size - 1.0) as usize);
            //
            self.spatial_grid[index].push(i);
        }

        for (i, cell) in self.cells.iter_mut().enumerate() {
            cell.apply_wall_force(&positions[i], sidelen as f32, pixel_size);
            cell.apply_dst_force(&positions[i], sidelen as f32);
        }

        for i in 0..self.cells.len() {
            let pos = positions[i].xy;
            let col = ((pos[0] / pixel_size).floor() as isize).clamp(0, grid_side as isize - 1);
            let row = ((pos[1] / pixel_size).floor() as isize).clamp(0, grid_side as isize - 1);
            let mut avg_xvel = 0.0;
            let mut avg_yvel = 0.0;
            let mut count = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let ncol = col + dx;
                    let nrow = row + dy;
                    if ncol < 0
                        || nrow < 0
                        || ncol >= grid_side as isize
                        || nrow >= grid_side as isize
                    {
                        continue;
                    }
                    let nindex = nrow as usize * grid_side + ncol as usize;
                    for other in self.spatial_grid[nindex].iter() {
                        if other == &i {
                            continue;
                        }
                        let other_cell = positions[*other];
                        let weight = self.cells[i].apply_neighbour_force(
                            &positions[i],
                            &other_cell,
                            pixel_size,
                        );

                        if self.cells[i].stroke_id != 0
                            && self.cells[i].stroke_id == self.cells[*other].stroke_id
                        {
                            // stronger attraction to same stroke
                            self.cells[i].apply_stroke_attraction(positions[i], other_cell, weight);
                        }

                        avg_xvel += self.cells[*other].velx * weight;
                        avg_yvel += self.cells[*other].vely * weight;
                        count += weight;
                    }
                }
            }

            if count > 0.0 {
                avg_xvel /= count;
                avg_yvel /= count;

                self.cells[i].accx += (avg_xvel - self.cells[i].velx) * ALIGNMENT_FACTOR;
                self.cells[i].accy += (avg_yvel - self.cells[i].vely) * ALIGNMENT_FACTOR;
            }
        }

        for (index, cell) in self.cells.iter_mut().enumerate() {
            cell.update(&mut positions[index]);
        }
    }

    pub fn set_assignments(&mut self, assignments: Vec<usize>, sidelen: u32) {
        if self.cells.is_empty() || sidelen == 0 {
            return;
        }
        let width = (self.cells.len() as f32).sqrt() as usize;
        if width * width != self.cells.len() {
            return;
        }
        self.assignment_seen.resize(self.cells.len(), false);
        self.assignment_seen.fill(false);
        let mut valid = assignments.len() == self.cells.len();
        if valid {
            for &source in &assignments {
                if source >= self.assignment_seen.len() || self.assignment_seen[source] {
                    valid = false;
                    break;
                }
                self.assignment_seen[source] = true;
            }
        }
        let pixelsize = sidelen as f32 / width as f32;

        for (dst_idx, assigned_src) in assignments
            .iter()
            .copied()
            .chain(std::iter::repeat(0))
            .take(self.cells.len())
            .enumerate()
        {
            // Invalid or corrupted persisted assignments fall back to identity
            // instead of indexing outside the cell array and panicking.
            let src_idx = if valid { assigned_src } else { dst_idx };
            let src_x = (src_idx % width) as f32;
            let src_y = (src_idx / width) as f32;
            let dst_x = (dst_idx % width) as f32;
            let dst_y = (dst_idx / width) as f32;
            let prev = self.cells[src_idx];

            self.cells[src_idx] = CellBody::new(
                (src_x + 0.5) * pixelsize,
                (src_y + 0.5) * pixelsize,
                (dst_x + 0.5) * pixelsize,
                (dst_y + 0.5) * pixelsize,
                prev.dst_force,
            );

            self.cells[src_idx].age = prev.age;
            self.cells[src_idx].stroke_id = prev.stroke_id;
        }
    }

    pub(crate) fn prepare_play(&mut self, positions: &mut [SeedPos], reverse: bool) {
        if self.reversed == reverse {
            for (i, cell) in self.cells.iter_mut().enumerate() {
                positions[i].xy[0] = cell.srcx;
                positions[i].xy[1] = cell.srcy;
                cell.age = 0;
            }
        } else {
            for (i, cell) in self.cells.iter().enumerate() {
                positions[i].xy[0] = cell.dstx;
                positions[i].xy[1] = cell.dsty;
            }
            self.switch();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four_cell_sim() -> Sim {
        let mut sim = Sim::new("test".into());
        sim.cells = vec![CellBody::new(0.0, 0.0, 0.0, 0.0, 0.1); 4];
        sim
    }

    #[test]
    fn invalid_assignments_fall_back_to_identity() {
        let mut sim = four_cell_sim();
        sim.set_assignments(vec![0, 0, 9], 4);

        for (index, cell) in sim.cells.iter().enumerate() {
            assert_eq!(cell.srcx, cell.dstx, "cell {index} x assignment");
            assert_eq!(cell.srcy, cell.dsty, "cell {index} y assignment");
        }
    }

    #[test]
    fn overlapping_cells_do_not_produce_nan_or_infinity() {
        let mut sim = four_cell_sim();
        sim.set_assignments(vec![0, 1, 2, 3], 4);
        let mut positions = vec![SeedPos { xy: [1.0, 1.0] }; 4];
        sim.update(&mut positions, 4);
        assert!(
            positions
                .iter()
                .flat_map(|position| position.xy)
                .all(f32::is_finite)
        );
    }

    #[test]
    fn spatial_grid_storage_is_reused_between_frames() {
        let mut sim = four_cell_sim();
        sim.set_assignments(vec![0, 1, 2, 3], 4);
        let mut positions = vec![SeedPos { xy: [1.0, 1.0] }; 4];
        sim.update(&mut positions, 4);
        let grid_ptr = sim.spatial_grid.as_ptr();
        sim.update(&mut positions, 4);
        assert_eq!(grid_ptr, sim.spatial_grid.as_ptr());
    }
}

// pub fn preset_path_to_name(source_dir: &Path) -> String {
//     source_dir
//         .file_stem()
//         .unwrap()
//         .to_string_lossy()
//         .into_owned()
// }

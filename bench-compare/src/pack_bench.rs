//! Rectangle packing: snugly against etagere, guillotiere, crunch and rectangle-pack.
//!
//! Every packer gets the same rectangles, sorted largest side first, and a fixed atlas width.
//! The atlas height is the lowest one each packer fits (binary search for the fixed-size
//! packers; farthing's `pack` reports its own height). Utilization = rect area / atlas area.

use std::collections::BTreeMap;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next() % (hi - lo + 1) as u64) as u32
    }
    fn log_range(&mut self, lo: u32, hi: u32) -> u32 {
        let t = (self.next() % 1_000_000) as f64 / 1e6;
        ((lo as f64).ln() + t * ((hi as f64).ln() - (lo as f64).ln())).exp().round() as u32
    }
}

fn datasets() -> Vec<(&'static str, u32, Vec<(u32, u32)>)> {
    let mut r = Rng(0x9e3779b97f4a7c15);
    let uniform = (0..500).map(|_| (r.range(8, 64), r.range(8, 64))).collect();
    let sprites = (0..300).map(|_| (r.log_range(16, 256), r.log_range(16, 256))).collect();
    let bars = (0..200)
        .map(|_| {
            let (a, b) = (r.range(8, 24), r.range(64, 200));
            if r.next() & 1 == 0 { (a, b) } else { (b, a) }
        })
        .collect();
    let few = (0..24).map(|_| (r.log_range(48, 512), r.log_range(48, 512))).collect();
    let mut frames: Vec<(u32, u32)> = vec![(256, 256); 24];
    frames.extend((0..40).map(|_| (r.log_range(32, 192), r.log_range(32, 192))));
    vec![
        ("uniform 500 (8–64 px)", 1024, uniform),
        ("sprites 300 (16–256 px, log)", 2048, sprites),
        ("bars 200 (1:3–1:25)", 1024, bars),
        ("few large 24 (48–512 px)", 1024, few),
        ("impostor frames + props 64", 2048, frames),
    ]
}

/// Lowest height in [lo, hi] for which `fits` holds (assumes it holds at `hi`).
fn min_height(lo: u32, hi: u32, fits: impl Fn(u32) -> bool) -> Option<u32> {
    if !fits(hi) {
        return None;
    }
    let (mut lo, mut hi) = (lo, hi);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if fits(mid) { hi = mid } else { lo = mid + 1 }
    }
    Some(hi)
}

fn etagere_fits(rects: &[(u32, u32)], w: u32, h: u32) -> bool {
    let mut a = etagere::AtlasAllocator::new(etagere::size2(w as i32, h as i32));
    rects.iter().all(|&(rw, rh)| a.allocate(etagere::size2(rw as i32, rh as i32)).is_some())
}

fn bucketed_fits(rects: &[(u32, u32)], w: u32, h: u32) -> bool {
    let mut a = etagere::BucketedAtlasAllocator::new(etagere::size2(w as i32, h as i32));
    rects.iter().all(|&(rw, rh)| a.allocate(etagere::size2(rw as i32, rh as i32)).is_some())
}

fn guillotiere_fits(rects: &[(u32, u32)], w: u32, h: u32) -> bool {
    let mut a = guillotiere::AtlasAllocator::new(guillotiere::size2(w as i32, h as i32));
    rects.iter().all(|&(rw, rh)| a.allocate(guillotiere::size2(rw as i32, rh as i32)).is_some())
}

fn crunch_fits(rects: &[(u32, u32)], w: u32, h: u32, rot: bool) -> bool {
    let rot = if rot { crunch::Rotation::Allowed } else { crunch::Rotation::None };
    let items = rects.iter().enumerate().map(|(i, &(rw, rh))| crunch::Item::new(i, rw as usize, rh as usize, rot));
    crunch::pack(crunch::Rect::of_size(w as usize, h as usize), items).is_ok()
}

fn rectangle_pack_fits(rects: &[(u32, u32)], w: u32, h: u32) -> bool {
    use rectangle_pack::{contains_smallest_box, pack_rects, volume_heuristic, GroupedRectsToPlace, RectToInsert, TargetBin};
    let mut place: GroupedRectsToPlace<usize, ()> = GroupedRectsToPlace::new();
    for (i, &(rw, rh)) in rects.iter().enumerate() {
        place.push_rect(i, None, RectToInsert::new(rw, rh, 1));
    }
    let mut bins = BTreeMap::new();
    bins.insert(0u8, TargetBin::new(w, h, 1));
    pack_rects(&place, &mut bins, &volume_heuristic, &contains_smallest_box).is_ok()
}

fn farthing_rects_fits(rects: &[(u32, u32)], w: u32, h: u32, rot: bool) -> bool {
    snugly::pack_rects(rects, w, h, 0, rot).is_some()
}

fn snugly(rects: &[(u32, u32)], w: u32, rot: bool) -> Option<u32> {
    let shapes: Vec<_> = rects.iter().map(|&(rw, rh)| snugly::ChartMask::rect(rw, rh)).collect();
    let opts = snugly::PackOptions {
        rotations: rot,
        ..snugly::PackOptions::width(w)
    };
    snugly::pack(&shapes, &opts).ok().map(|p| p.height)
}

pub fn pack_mode(_args: &[String]) {
    type Method = (&'static str, Box<dyn Fn(&[(u32, u32)], u32, u32, u32) -> Option<u32>>);
    let search = |f: fn(&[(u32, u32)], u32, u32) -> bool| -> Box<dyn Fn(&[(u32, u32)], u32, u32, u32) -> Option<u32>> {
        Box::new(move |r, w, lo, hi| min_height(lo, hi, |h| f(r, w, h)))
    };
    let methods: Vec<Method> = vec![
        ("snugly pack (rot)", Box::new(|r, w, _, _| snugly(r, w, true))),
        ("snugly pack", Box::new(|r, w, _, _| snugly(r, w, false))),
        ("snugly pack_rects fixed-size (rot)", search(|r, w, h| farthing_rects_fits(r, w, h, true))),
        ("crunch (rot)", search(|r, w, h| crunch_fits(r, w, h, true))),
        ("crunch", search(|r, w, h| crunch_fits(r, w, h, false))),
        ("rectangle-pack", search(rectangle_pack_fits)),
        ("etagere", search(etagere_fits)),
        ("etagere bucketed", search(bucketed_fits)),
        ("guillotiere", search(guillotiere_fits)),
    ];
    println!("| dataset | packer | atlas | utilization | ms |");
    println!("|---|---|---|---|---|");
    for (name, w, mut rects) in datasets() {
        rects.sort_by_key(|&(a, b)| std::cmp::Reverse((a.max(b), a * b)));
        let area: u64 = rects.iter().map(|&(a, b)| a as u64 * b as u64).sum();
        let lo = (area / w as u64) as u32;
        let hi = 8 * lo.max(512);
        for (m, f) in &methods {
            let t = Instant::now();
            let h = f(&rects, w, lo, hi);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            match h {
                Some(h) => println!("| {name} | {m} | {w}×{h} | {:.3} | {ms:.1} |", area as f64 / (w as f64 * h as f64)),
                None => println!("| {name} | {m} | — | failed | {ms:.1} |"),
            }
        }
    }
}

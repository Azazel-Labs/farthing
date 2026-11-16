//! Shrinking textures with a choice of filter, after the mip settings engines offer (Unreal's
//! *Mip Gen Settings*: Unfiltered, SimpleAverage, Sharpen0–10, Blur1–5).
//!
//! Color is filtered in linear light (sRGB bytes are decoded first), and tangent-space normal
//! maps as vectors that are renormalized afterwards. Filters with negative lobes (Kaiser,
//! Lanczos, sharpening) can overshoot; results are clamped to the storable range.

use super::Texture;
use glam::{Vec3, Vec4};
use rayon::prelude::*;

/// The filter that turns many source texels into one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Downsample {
    /// The source texel under each output texel's center, no filtering (Unreal: Unfiltered).
    /// Keeps hard-edged texels hard, and aliases.
    Nearest,
    /// The average of the source texels each output texel covers (Unreal: SimpleAverage).
    #[default]
    Box,
    /// A tent one output texel wide on each side: smoother than [`Box`](Self::Box), a little
    /// softer.
    Triangle,
    /// Kaiser-windowed sinc (half-width 3 output texels, α 4), the usual offline mip filter:
    /// sharper than [`Box`](Self::Box), with slight ringing at hard edges.
    Kaiser,
    /// Lanczos-3: the sharpest here, and the most ringing.
    Lanczos3,
    /// A Gaussian that blurs more with strength 1–5 (Unreal: Blur1–5).
    Blur(u8),
}

/// How [`Texture::downsample`] filters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DownsampleOptions {
    pub filter: Downsample,
    /// Unsharp masking after the resample, 0 (none) to 10 (strongest), in the spirit of
    /// Unreal's Sharpen0–10 (not its exact kernel).
    pub sharpen: u8,
    /// The texture tiles: filters reaching past an edge read the opposite edge. Off, they
    /// repeat the edge texel (atlases, whose charts never wrap).
    pub wrap: bool,
    /// A tangent-space normal map (xyz · 0.5 + 0.5): filtered as vectors, then renormalized.
    pub normal_map: bool,
}

impl Downsample {
    /// Half-width of the kernel in output texels (0: one source texel, `Nearest`).
    fn support(self) -> f32 {
        match self {
            Downsample::Nearest => 0.0,
            Downsample::Box => 0.5,
            Downsample::Triangle => 1.0,
            Downsample::Kaiser | Downsample::Lanczos3 => 3.0,
            Downsample::Blur(n) => 3.0 * blur_sigma(n),
        }
    }

    /// Kernel weight at `x` output texels from the output texel's center.
    fn weight(self, x: f32) -> f32 {
        let x = x.abs();
        match self {
            Downsample::Nearest => 1.0,
            Downsample::Box => 1.0,
            Downsample::Triangle => (1.0 - x).max(0.0),
            Downsample::Kaiser => {
                const ALPHA: f32 = 4.0;
                let t = x / 3.0;
                if t >= 1.0 {
                    0.0
                } else {
                    sinc(x) * bessel_i0(ALPHA * (1.0 - t * t).sqrt()) / bessel_i0(ALPHA)
                }
            }
            Downsample::Lanczos3 => {
                if x >= 3.0 {
                    0.0
                } else {
                    sinc(x) * sinc(x / 3.0)
                }
            }
            Downsample::Blur(n) => {
                let s = blur_sigma(n);
                (-0.5 * (x / s) * (x / s)).exp()
            }
        }
    }
}

/// Standard deviation of `Blur(n)`, in output texels: 0.75 at strength 1, 1.75 at 5.
fn blur_sigma(n: u8) -> f32 {
    0.5 + 0.25 * f32::from(n.clamp(1, 5))
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let p = std::f32::consts::PI * x;
        p.sin() / p
    }
}

/// Modified Bessel function of the first kind, order 0 (series; converges fast for |x| ≤ 4).
fn bessel_i0(x: f32) -> f32 {
    let (mut sum, mut term, q) = (1.0f32, 1.0f32, x * x / 4.0);
    for k in 1..32 {
        term *= q / (k * k) as f32;
        sum += term;
        if term < sum * 1e-7 {
            break;
        }
    }
    sum
}

/// The source texels and weights of one output texel along one axis.
struct Taps {
    start: Vec<u32>,
    len: Vec<u32>,
    index: Vec<u32>,
    weight: Vec<f32>,
}

/// Weights resampling `src` texels to `dst` along one axis.
fn taps(src: u32, dst: u32, filter: Downsample, wrap: bool) -> Taps {
    let scale = src as f32 / dst as f32;
    // Downsampling widens the kernel to the output texel's footprint; upsampling keeps it.
    let stretch = scale.max(1.0);
    let fetch = |i: i64| -> u32 {
        let n = i64::from(src);
        (if wrap {
            i.rem_euclid(n)
        } else {
            i.clamp(0, n - 1)
        }) as u32
    };
    let mut t = Taps {
        start: Vec::with_capacity(dst as usize),
        len: Vec::with_capacity(dst as usize),
        index: Vec::new(),
        weight: Vec::new(),
    };
    for o in 0..dst {
        let center = (o as f32 + 0.5) * scale; // in source texels
        let first = t.index.len();
        if filter == Downsample::Nearest {
            t.index.push(fetch(center.floor() as i64));
            t.weight.push(1.0);
        } else if filter == Downsample::Box {
            // Exact coverage of the footprint [center - r, center + r] by each source texel.
            let r = 0.5 * stretch;
            let (lo, hi) = (center - r, center + r);
            for i in lo.floor() as i64..hi.ceil() as i64 {
                let w = (hi.min(i as f32 + 1.0) - lo.max(i as f32)).max(0.0);
                if w > 0.0 {
                    t.index.push(fetch(i));
                    t.weight.push(w);
                }
            }
        } else {
            let r = filter.support() * stretch;
            for i in (center - r).floor() as i64..=(center + r).ceil() as i64 {
                let w = filter.weight((i as f32 + 0.5 - center) / stretch);
                if w != 0.0 {
                    t.index.push(fetch(i));
                    t.weight.push(w);
                }
            }
        }
        let sum: f32 = t.weight[first..].iter().sum();
        if sum.abs() > 1e-8 {
            t.weight[first..].iter_mut().for_each(|w| *w /= sum);
        } else {
            // A kernel that sums to nothing here: fall back to the nearest texel.
            t.index.truncate(first);
            t.weight.truncate(first);
            t.index.push(fetch(center.floor() as i64));
            t.weight.push(1.0);
        }
        t.start.push(first as u32);
        t.len.push((t.index.len() - first) as u32);
    }
    t
}

impl Taps {
    #[inline]
    fn apply(&self, o: usize, read: impl Fn(u32) -> Vec4) -> Vec4 {
        let (s, n) = (self.start[o] as usize, self.len[o] as usize);
        self.index[s..s + n]
            .iter()
            .zip(&self.weight[s..s + n])
            .fold(Vec4::ZERO, |acc, (&i, &w)| acc + read(i) * w)
    }
}

impl Texture {
    /// This texture resampled to `width × height` (each at least 1) with `opts.filter`.
    /// Usually smaller; a larger size interpolates with the same kernel. sRGB textures are
    /// filtered in linear light and stored as sRGB again; normal maps are renormalized.
    pub fn downsample(&self, width: u32, height: u32, opts: &DownsampleOptions) -> Texture {
        let (w, h) = (width.max(1), height.max(1));
        let (sw, sh) = (self.width.max(1), self.height.max(1));
        let lut = super::srgb_lut();
        let srgb = self.srgb && !opts.normal_map;
        let decode = |t: [u8; 4]| -> Vec4 {
            let f = |v: u8| f32::from(v) / 255.0;
            if opts.normal_map {
                Vec4::new(
                    f(t[0]) * 2.0 - 1.0,
                    f(t[1]) * 2.0 - 1.0,
                    f(t[2]) * 2.0 - 1.0,
                    f(t[3]),
                )
            } else if srgb {
                Vec4::new(
                    lut[t[0] as usize],
                    lut[t[1] as usize],
                    lut[t[2] as usize],
                    f(t[3]),
                )
            } else {
                Vec4::new(f(t[0]), f(t[1]), f(t[2]), f(t[3]))
            }
        };
        let src: Vec<Vec4> = self.texels.iter().map(|&t| decode(t)).collect();

        // Separable: rows to the new width, then columns to the new height.
        let tx = taps(sw, w, opts.filter, opts.wrap);
        let ty = taps(sh, h, opts.filter, opts.wrap);
        let mut rows = vec![Vec4::ZERO; w as usize * sh as usize];
        rows.par_chunks_mut(w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let line = &src[y * sw as usize..(y + 1) * sw as usize];
                for (x, o) in row.iter_mut().enumerate() {
                    *o = tx.apply(x, |i| line[i as usize]);
                }
            });
        drop(src);
        let mut out = vec![Vec4::ZERO; w as usize * h as usize];
        out.par_chunks_mut(w as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    *o = ty.apply(y, |i| rows[i as usize * w as usize + x]);
                }
            });
        drop(rows);

        if opts.sharpen > 0 {
            out = unsharp(&out, w, h, f32::from(opts.sharpen.min(10)) * 0.1, opts.wrap);
        }

        let texels = out
            .iter()
            .map(|&p| {
                if opts.normal_map {
                    let n = p.truncate().clamp(Vec3::splat(-1.0), Vec3::splat(1.0));
                    let n = n.try_normalize().unwrap_or(Vec3::Z);
                    let mut t = Texture::encode_normal(n.extend(0.0));
                    t[3] = super::unorm8(p.w);
                    t
                } else {
                    Texture::encode(p, srgb)
                }
            })
            .collect();
        Texture {
            width: w,
            height: h,
            srgb: self.srgb,
            texels,
        }
    }

    /// This texture with each side halved `times` times (each at least 1): what dropping
    /// that many top mips leaves, as a texture LOD bias does.
    pub fn halve(&self, times: u32, opts: &DownsampleOptions) -> Texture {
        if times == 0 {
            return self.clone();
        }
        let s = times.min(31);
        self.downsample(self.width >> s, self.height >> s, opts)
    }
}

/// `img + amount · (img − blur(img))`, with a 3 × 3 binomial blur.
fn unsharp(img: &[Vec4], w: u32, h: u32, amount: f32, wrap: bool) -> Vec<Vec4> {
    let (w, h) = (i64::from(w), i64::from(h));
    let at = |x: i64, y: i64| -> Vec4 {
        let (x, y) = if wrap {
            (x.rem_euclid(w), y.rem_euclid(h))
        } else {
            (x.clamp(0, w - 1), y.clamp(0, h - 1))
        };
        img[(y * w + x) as usize]
    };
    let mut out = vec![Vec4::ZERO; img.len()];
    out.par_chunks_mut(w as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as i64;
            for (x, o) in row.iter_mut().enumerate() {
                let x = x as i64;
                let mut blur = Vec4::ZERO;
                for (dy, wy) in [(-1, 1.0), (0, 2.0), (1, 1.0)] {
                    for (dx, wx) in [(-1, 1.0), (0, 2.0), (1, 1.0)] {
                        blur += at(x + dx, y + dy) * (wx * wy / 16.0);
                    }
                }
                let p = at(x, y);
                *o = p + (p - blur) * amount;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILTERS: [Downsample; 7] = [
        Downsample::Nearest,
        Downsample::Box,
        Downsample::Triangle,
        Downsample::Kaiser,
        Downsample::Lanczos3,
        Downsample::Blur(1),
        Downsample::Blur(5),
    ];

    fn tex(w: u32, h: u32, srgb: bool, f: impl Fn(u32, u32) -> [u8; 4]) -> Texture {
        let data: Vec<u8> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .flat_map(|(x, y)| f(x, y))
            .collect();
        Texture::from_rgba8(w, h, &data, srgb).unwrap()
    }

    #[test]
    fn a_flat_texture_stays_flat_with_every_filter() {
        let t = tex(37, 20, true, |_, _| [200, 90, 10, 255]);
        for filter in FILTERS {
            for sharpen in [0, 10] {
                for wrap in [false, true] {
                    let o = DownsampleOptions {
                        filter,
                        sharpen,
                        wrap,
                        ..Default::default()
                    };
                    let d = t.downsample(9, 7, &o);
                    assert_eq!((d.width, d.height), (9, 7));
                    assert!(
                        d.rgba8().chunks(4).all(|c| c == [200, 90, 10, 255]),
                        "{filter:?} sharpen {sharpen} wrap {wrap}"
                    );
                }
            }
        }
    }

    #[test]
    fn box_averages_srgb_in_linear_light() {
        // Black and white texels: the linear mean is 0.5, which sRGB stores as 188, not 128.
        let t = tex(2, 2, true, |x, y| {
            let v = if (x + y) % 2 == 0 { 0 } else { 255 };
            [v, v, v, 255]
        });
        let d = t.halve(1, &DownsampleOptions::default());
        assert_eq!(d.rgba8(), &[188, 188, 188, 255]);
        // Linear data averages as stored.
        let l = Texture::from_rgba8(2, 2, t.rgba8(), false).unwrap();
        assert_eq!(l.halve(1, &DownsampleOptions::default()).rgba8()[0], 128);
    }

    #[test]
    fn nearest_picks_texels_without_mixing() {
        let t = tex(4, 4, false, |x, y| [(x * 60) as u8, (y * 60) as u8, 0, 255]);
        let o = DownsampleOptions {
            filter: Downsample::Nearest,
            ..Default::default()
        };
        let d = t.halve(1, &o);
        let values: Vec<u8> = d.rgba8().chunks(4).map(|c| c[0]).collect();
        assert_eq!(values, [60, 180, 60, 180]);
    }

    #[test]
    fn normal_maps_come_out_unit_length() {
        // Alternating normals tilted ±45° about y average to a short vector along z.
        let t = tex(8, 8, false, |x, _| {
            if x % 2 == 0 {
                [218, 128, 218, 255]
            } else {
                [37, 128, 218, 255]
            }
        });
        for filter in FILTERS {
            let o = DownsampleOptions {
                filter,
                normal_map: true,
                ..Default::default()
            };
            for c in t.halve(2, &o).rgba8().chunks(4) {
                let n = Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 127.5 - 1.0;
                assert!((n.length() - 1.0).abs() < 0.02, "{filter:?}: {c:?}");
            }
        }
    }

    #[test]
    fn wrap_reads_the_opposite_edge() {
        // A white first column: clamped, the last output column never sees it; wrapped, it does.
        let t = tex(
            8,
            1,
            false,
            |x, _| if x == 0 { [255; 4] } else { [0, 0, 0, 255] },
        );
        let o = |wrap| DownsampleOptions {
            filter: Downsample::Triangle,
            wrap,
            ..Default::default()
        };
        let last = |d: Texture| d.rgba8()[4 * 3];
        assert_eq!(last(t.downsample(4, 1, &o(false))), 0);
        assert!(last(t.downsample(4, 1, &o(true))) > 0);
    }

    #[test]
    fn sharper_filters_keep_more_contrast() {
        // A soft-edged stripe pattern; contrast left after a 4× shrink.
        let t = tex(64, 4, false, |x, _| {
            let v = (128.0 + 127.0 * (x as f32 * 0.4).sin()) as u8;
            [v, v, v, 255]
        });
        let contrast = |filter, sharpen| {
            let d = t.downsample(
                16,
                1,
                &DownsampleOptions {
                    filter,
                    sharpen,
                    ..Default::default()
                },
            );
            let v: Vec<i32> = d.rgba8().chunks(4).map(|c| i32::from(c[0])).collect();
            v.iter().max().unwrap() - v.iter().min().unwrap()
        };
        let blur = contrast(Downsample::Blur(5), 0);
        let boxed = contrast(Downsample::Box, 0);
        let lanczos = contrast(Downsample::Lanczos3, 0);
        assert!(blur < boxed, "blur {blur} box {boxed}");
        assert!(boxed < lanczos, "box {boxed} lanczos {lanczos}");
        assert!(contrast(Downsample::Box, 10) > boxed);
    }

    #[test]
    fn halving_stops_at_one_texel() {
        let t = tex(8, 2, false, |_, _| [1, 2, 3, 4]);
        let d = t.halve(5, &DownsampleOptions::default());
        assert_eq!((d.width, d.height), (1, 1));
        assert_eq!(d.rgba8(), &[1, 2, 3, 4]);
    }
}

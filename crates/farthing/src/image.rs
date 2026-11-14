//! Images: [`Image`], linear float RGBA for work in progress (bakes, captures, metrics), and
//! [`Texture`], compact shared 8-bit RGBA for material textures as they are stored.

use crate::{Error, Result};
use glam::{Vec2, Vec4};
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// Row-major RGBA32F image. UV (0, 0) is the top-left corner, matching glTF.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Vec4>,
}

impl Image {
    pub fn new(width: u32, height: u32, fill: Vec4) -> Image {
        Image {
            width,
            height,
            pixels: vec![fill; (width * height) as usize],
        }
    }
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Vec4 {
        self.pixels[(y * self.width + x) as usize]
    }
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: Vec4) {
        let i = (y * self.width + x) as usize;
        self.pixels[i] = v;
    }
    #[inline]
    fn fetch(&self, x: i64, y: i64, wrap: bool) -> Vec4 {
        let (w, h) = (self.width as i64, self.height as i64);
        let (x, y) = if wrap {
            (x.rem_euclid(w), y.rem_euclid(h))
        } else {
            (x.clamp(0, w - 1), y.clamp(0, h - 1))
        };
        self.pixels[(y * w + x) as usize]
    }

    /// Bilinear sample at `uv` (texel centers at (i + 0.5) / size).
    pub fn sample_bilinear(&self, uv: Vec2, wrap: bool) -> Vec4 {
        let x = uv.x * self.width as f32 - 0.5;
        let y = uv.y * self.height as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self
            .fetch(x0, y0, wrap)
            .lerp(self.fetch(x0 + 1, y0, wrap), fx);
        let b = self
            .fetch(x0, y0 + 1, wrap)
            .lerp(self.fetch(x0 + 1, y0 + 1, wrap), fx);
        a.lerp(b, fy)
    }

    /// Copies `src` into this image at (x, y), clipping at the edges.
    pub fn blit(&mut self, src: &Image, x: u32, y: u32) {
        for sy in 0..src.height {
            let dy = y + sy;
            if dy >= self.height {
                break;
            }
            for sx in 0..src.width {
                let dx = x + sx;
                if dx >= self.width {
                    break;
                }
                self.set(dx, dy, src.get(sx, sy));
            }
        }
    }

    /// Box-filter downsample by an integer factor, averaging only pixels with alpha > 0.
    /// Output alpha is the covered fraction.
    pub fn downsample_coverage(&self, factor: u32) -> Image {
        if factor <= 1 {
            return self.clone();
        }
        let (w, h) = (self.width / factor, self.height / factor);
        let mut out = Image::new(w, h, Vec4::ZERO);
        for y in 0..h {
            for x in 0..w {
                let mut acc = Vec4::ZERO;
                let mut n = 0u32;
                for sy in 0..factor {
                    for sx in 0..factor {
                        let p = self.get(x * factor + sx, y * factor + sy);
                        if p.w > 0.0 {
                            acc += p.truncate().extend(1.0);
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    let rgb = acc.truncate() / n as f32;
                    out.set(x, y, rgb.extend(n as f32 / (factor * factor) as f32));
                }
            }
        }
        out
    }

    /// Pushes covered pixels (alpha > 0) outward into uncovered ones for `iterations` rings,
    /// so mip levels don't bleed background into charts. Dilated pixels get alpha 1 but are
    /// reported in the returned mask as `false` (not originally covered).
    pub fn dilate(&mut self, iterations: u32) -> Vec<bool> {
        let original: Vec<bool> = self.pixels.iter().map(|p| p.w > 0.0).collect();
        let mut covered = original.clone();
        let (w, h) = (self.width as i64, self.height as i64);
        for _ in 0..iterations {
            let mut changes = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) as usize;
                    if covered[i] {
                        continue;
                    }
                    let mut acc = Vec4::ZERO;
                    let mut n = 0.0;
                    for (dx, dy) in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (1, -1),
                        (-1, 1),
                        (1, 1),
                    ] {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < w && ny < h {
                            let j = (ny * w + nx) as usize;
                            if covered[j] {
                                acc += self.pixels[j];
                                n += 1.0;
                            }
                        }
                    }
                    if n > 0.0 {
                        changes.push((i, (acc / n).truncate().extend(1.0)));
                    }
                }
            }
            if changes.is_empty() {
                break;
            }
            for (i, v) in changes {
                self.pixels[i] = v;
                covered[i] = true;
            }
        }
        original
    }

    /// Writes an 8-bit RGBA PNG. With `srgb`, RGB is encoded linear→sRGB.
    pub fn save_png(&self, path: impl AsRef<Path>, srgb: bool) -> Result<()> {
        let mut buf = Vec::with_capacity(self.pixels.len() * 4);
        for p in &self.pixels {
            let rgb = if srgb {
                [
                    linear_to_srgb(p.x),
                    linear_to_srgb(p.y),
                    linear_to_srgb(p.z),
                ]
            } else {
                [p.x, p.y, p.z]
            };
            for c in rgb {
                buf.push((c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            buf.push((p.w.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        image::save_buffer(
            path.as_ref(),
            &buf,
            self.width,
            self.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| Error::Image(e.to_string()))
    }

    /// Loads a PNG. With `srgb`, RGB is decoded sRGB→linear.
    pub fn load_png(path: impl AsRef<Path>, srgb: bool) -> Result<Image> {
        let img = image::open(path.as_ref())
            .map_err(|e| Error::Image(e.to_string()))?
            .to_rgba8();
        Ok(Self::from_rgba8(
            img.width(),
            img.height(),
            img.as_raw(),
            srgb,
        ))
    }

    pub fn from_rgba8(width: u32, height: u32, data: &[u8], srgb: bool) -> Image {
        let pixels = data
            .chunks_exact(4)
            .map(|c| {
                let f = |v: u8| v as f32 / 255.0;
                let conv = |v: u8| if srgb { srgb_to_linear(f(v)) } else { f(v) };
                Vec4::new(conv(c[0]), conv(c[1]), conv(c[2]), f(c[3]))
            })
            .collect();
        Image {
            width,
            height,
            pixels,
        }
    }
}

/// A material texture as stored: 8-bit RGBA, a quarter of an [`Image`]'s memory (a 2048²
/// texture is 16 MB instead of 64 MB), behind an `Arc` so clones share the texels.
///
/// With `srgb` the RGB bytes are sRGB-encoded (glTF base colour) and sampling decodes them to
/// linear; alpha is always linear. Without it the bytes are linear unorm data (metallic-
/// roughness, normal maps with xyz in 0..1). Sampling decodes each texel before filtering, so
/// it returns exactly what the same lookups on [`Image::from_rgba8`]'s output would.
#[derive(Clone, PartialEq, Eq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub srgb: bool,
    texels: Arc<[[u8; 4]]>,
}

impl std::fmt::Debug for Texture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let enc = if self.srgb { ", sRGB" } else { "" };
        write!(f, "Texture({}x{}{enc})", self.width, self.height)
    }
}

fn srgb_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

fn unorm8(c: f32) -> u8 {
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

impl Texture {
    /// From tightly packed RGBA8 rows (`width · height · 4` bytes).
    pub fn from_rgba8(width: u32, height: u32, data: &[u8], srgb: bool) -> Result<Texture> {
        let need = width as usize * height as usize * 4;
        if data.len() != need {
            return Err(Error::InvalidInput(format!(
                "{width}x{height} RGBA8 texture needs {need} bytes, got {}",
                data.len()
            )));
        }
        Ok(Texture {
            width,
            height,
            srgb,
            texels: data.as_chunks::<4>().0.into(),
        })
    }

    /// Quantizes a linear [`Image`]; with `srgb` its RGB is encoded linear→sRGB.
    pub fn from_image(img: &Image, srgb: bool) -> Texture {
        let enc = |c: f32| unorm8(if srgb { linear_to_srgb(c) } else { c });
        Texture {
            width: img.width,
            height: img.height,
            srgb,
            texels: img
                .pixels
                .iter()
                .map(|p| [enc(p.x), enc(p.y), enc(p.z), unorm8(p.w)])
                .collect(),
        }
    }

    /// A tangent-space normal map from xyz in -1..1 (the bake convention), stored as glTF
    /// does: xyz · 0.5 + 0.5, alpha 1.
    pub fn from_signed_normals(img: &Image) -> Texture {
        let enc = |c: f32| unorm8(c * 0.5 + 0.5);
        Texture {
            width: img.width,
            height: img.height,
            srgb: false,
            texels: img
                .pixels
                .iter()
                .map(|p| [enc(p.x), enc(p.y), enc(p.z), 255])
                .collect(),
        }
    }

    /// The stored bytes, RGBA row-major.
    pub fn rgba8(&self) -> &[u8] {
        self.texels.as_flattened()
    }

    /// Bytes of texel storage (shared between clones).
    pub fn byte_size(&self) -> usize {
        self.texels.len() * 4
    }

    #[inline]
    fn decode(&self, t: [u8; 4]) -> Vec4 {
        let f = |v: u8| v as f32 / 255.0;
        if self.srgb {
            let lut = srgb_lut();
            Vec4::new(
                lut[t[0] as usize],
                lut[t[1] as usize],
                lut[t[2] as usize],
                f(t[3]),
            )
        } else {
            Vec4::new(f(t[0]), f(t[1]), f(t[2]), f(t[3]))
        }
    }

    /// Texel (x, y), decoded to linear.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Vec4 {
        self.decode(self.texels[(y * self.width + x) as usize])
    }

    #[inline]
    fn fetch(&self, x: i64, y: i64, wrap: bool) -> Vec4 {
        let (w, h) = (self.width as i64, self.height as i64);
        let (x, y) = if wrap {
            (x.rem_euclid(w), y.rem_euclid(h))
        } else {
            (x.clamp(0, w - 1), y.clamp(0, h - 1))
        };
        self.decode(self.texels[(y * w + x) as usize])
    }

    /// Bilinear sample at `uv` (texel centers at (i + 0.5) / size), decoded to linear.
    /// Transparent black for an empty texture.
    pub fn sample_bilinear(&self, uv: Vec2, wrap: bool) -> Vec4 {
        if self.texels.is_empty() {
            return Vec4::ZERO;
        }
        let x = uv.x * self.width as f32 - 0.5;
        let y = uv.y * self.height as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self
            .fetch(x0, y0, wrap)
            .lerp(self.fetch(x0 + 1, y0, wrap), fx);
        let b = self
            .fetch(x0, y0 + 1, wrap)
            .lerp(self.fetch(x0 + 1, y0 + 1, wrap), fx);
        a.lerp(b, fy)
    }

    /// The bytes with RGB encoded as sRGB (`srgb`) or linear, converting only if the stored
    /// encoding differs. What an exporter writes.
    pub fn rgba8_as(&self, srgb: bool) -> std::borrow::Cow<'_, [u8]> {
        if srgb == self.srgb {
            return self.rgba8().into();
        }
        let lut = srgb_lut();
        let conv = |v: u8| {
            if srgb {
                unorm8(linear_to_srgb(v as f32 / 255.0))
            } else {
                unorm8(lut[v as usize])
            }
        };
        self.texels
            .iter()
            .flat_map(|t| [conv(t[0]), conv(t[1]), conv(t[2]), t[3]])
            .collect::<Vec<u8>>()
            .into()
    }

    /// Decoded to a linear float [`Image`] (4× the memory).
    pub fn to_image(&self) -> Image {
        Image {
            width: self.width,
            height: self.height,
            pixels: self.texels.iter().map(|&t| self.decode(t)).collect(),
        }
    }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.max(0.0).powf(1.0 / 2.4) - 0.055
    }
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_roundtrip() {
        for i in 0..=20 {
            let v = i as f32 / 20.0;
            assert!((srgb_to_linear(linear_to_srgb(v)) - v).abs() < 1e-5);
        }
    }

    #[test]
    fn dilate_fills_ring() {
        let mut img = Image::new(5, 5, Vec4::ZERO);
        img.set(2, 2, Vec4::new(1.0, 0.0, 0.0, 1.0));
        img.dilate(1);
        assert_eq!(img.get(1, 1), Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(img.get(0, 0), Vec4::ZERO);
    }

    #[test]
    fn texture_samples_like_the_float_image() {
        let bytes: Vec<u8> = (0..4 * 7 * 5).map(|i| (i * 37 % 256) as u8).collect();
        for srgb in [false, true] {
            let t = Texture::from_rgba8(7, 5, &bytes, srgb).unwrap();
            let img = Image::from_rgba8(7, 5, &bytes, srgb);
            assert_eq!(t.to_image(), img);
            for uv in [
                Vec2::new(0.13, 0.71),
                Vec2::new(-0.4, 1.3),
                Vec2::new(0.5, 0.5),
            ] {
                for wrap in [false, true] {
                    assert_eq!(t.sample_bilinear(uv, wrap), img.sample_bilinear(uv, wrap));
                }
            }
            assert_eq!(Texture::from_image(&img, srgb).rgba8(), &bytes[..]);
            let other = t.rgba8_as(!srgb);
            let back = Texture::from_rgba8(7, 5, &other, !srgb).unwrap().to_image();
            let close = back.pixels.iter().zip(&img.pixels);
            assert!(close
                .into_iter()
                .all(|(a, b)| (*a - *b).abs().max_element() < 0.02));
        }
        assert!(Texture::from_rgba8(2, 2, &[0; 15], false).is_err());
    }

    #[test]
    fn bilinear_center_is_exact() {
        let mut img = Image::new(2, 2, Vec4::ZERO);
        img.set(1, 0, Vec4::ONE);
        assert_eq!(img.sample_bilinear(Vec2::new(0.75, 0.25), false), Vec4::ONE);
    }
}

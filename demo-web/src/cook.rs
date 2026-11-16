//! Texture compression with [mipple](https://github.com/Azazel-Labs/mipple): a level's textures
//! as they would ship, in GPU block formats with full mip chains, in KTX2 files with Zstd
//! supercompression. The viewer draws the blocks decoded back, so what it shows is what a GPU
//! would sample, and the stats are the real sizes.

use crate::pipeline::texture_bytes;
use farthing::image::Texture;
use farthing::io::ExportTextures;
use mipple::ktx2::PrepMode;
use mipple::metrics::{normal_angle_error_deg, normal_from_rg, psnr};
use mipple::pipeline::{decode_level, encode, prepare};
use mipple::{EncodeOptions, Format, GpuMode, RdoParams, Surface};
use serde::{Deserialize, Serialize};

/// Zstd level of the KTX2 files: what a shipping build would use.
pub const ZSTD_LEVEL: i32 = 19;

/// The platform the textures are cooked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureCompression {
    /// RGBA8 textures, as baked.
    #[default]
    Off,
    /// Desktop and console: BC7 base color and ORM, BC5 normals.
    Bc,
    /// Mobile and Apple GPUs: ASTC base color and ORM at the footprint, normals ASTC 4x4.
    Astc4x4,
    Astc6x6,
    Astc8x8,
    /// OpenGL ES 3 / Vulkan baseline: ETC2 base color and ORM, EAC RG11 normals.
    Etc2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Channel {
    BaseColor,
    Normal,
    Orm,
}

impl Channel {
    fn name(self) -> &'static str {
        match self {
            Channel::BaseColor => "base_color",
            Channel::Normal => "normal",
            Channel::Orm => "orm",
        }
    }
}

impl TextureCompression {
    /// A short name for the target.
    pub fn target(self) -> &'static str {
        match self {
            TextureCompression::Off => "off",
            TextureCompression::Bc => "Desktop (BC)",
            TextureCompression::Astc4x4 => "Mobile (ASTC 4x4)",
            TextureCompression::Astc6x6 => "Mobile (ASTC 6x6)",
            TextureCompression::Astc8x8 => "Mobile (ASTC 8x8)",
            TextureCompression::Etc2 => "Mobile (ETC2)",
        }
    }

    fn format(self, channel: Channel, has_alpha: bool) -> Option<Format> {
        use TextureCompression as C;
        Some(match (self, channel) {
            (C::Off, _) => return None,
            (C::Bc, Channel::Normal) => Format::Bc5,
            (C::Bc, _) => Format::Bc7,
            // Normals keep the finest footprint: coarse ones band visibly in lighting.
            (C::Astc4x4 | C::Astc6x6 | C::Astc8x8, Channel::Normal) => Format::Astc4x4,
            (C::Astc4x4, _) => Format::Astc4x4,
            (C::Astc6x6, _) => Format::Astc6x6,
            (C::Astc8x8, _) => Format::Astc8x8,
            (C::Etc2, Channel::Normal) => Format::EacRg11,
            (C::Etc2, Channel::BaseColor) if has_alpha => Format::Etc2Rgba,
            (C::Etc2, _) => Format::Etc2Rgb,
        })
    }
}

/// How hard mipple works: the target, effort and rate-distortion weight.
#[derive(Clone, Copy, Debug)]
pub struct CookOptions {
    pub target: TextureCompression,
    /// 0–9 (mipple's default is 5).
    pub effort: u8,
    /// Rate-distortion optimization weight; 0 = off. Higher trades quality for smaller files
    /// after Zstd (the GPU size does not change).
    pub lambda: f32,
}

/// One cooked texture.
#[derive(Clone, Debug, Serialize)]
pub struct CookedTexture {
    pub channel: &'static str,
    pub format: &'static str,
    pub width: u32,
    pub height: u32,
    pub mips: usize,
    /// The blocks of every mip level: what the GPU holds.
    pub gpu_bytes: usize,
    /// The KTX2 file with Zstd: what ships.
    pub ktx2_bytes: usize,
    /// Base color and ORM: PSNR of the top level over the format's channels, dB. `None` when
    /// the blocks decode to the exact texels.
    pub psnr: Option<f32>,
    /// Normals: mean angle between the source and decoded normals, degrees.
    pub normal_error: Option<f32>,
}

/// What cooking a level's textures cost and gave.
#[derive(Clone, Debug, Serialize)]
pub struct CookedInfo {
    pub target: &'static str,
    pub effort: u8,
    pub lambda: f32,
    pub zstd_level: i32,
    pub textures: Vec<CookedTexture>,
    /// GPU memory of the cooked textures (blocks, full mip chains).
    pub gpu_bytes: usize,
    /// The KTX2 files together.
    pub ktx2_bytes: usize,
    /// The same textures as RGBA8 with mips, for comparison.
    pub rgba_bytes: usize,
    /// The level as it ships: its compact GLB without textures plus the KTX2 files.
    pub ship_bytes: usize,
    /// Time to encode, decode, measure and write.
    pub ms: f64,
    /// The level whose textures these are (shared levels reuse the carrier's).
    pub level: usize,
}

/// A level's cooked textures: the stats and the KTX2 files (`textures/<channel>.ktx2`).
#[derive(Debug)]
pub struct Cooked {
    pub info: CookedInfo,
    pub files: Vec<(String, Vec<u8>)>,
}

fn surface(t: &Texture) -> Surface {
    Surface {
        width: t.width,
        height: t.height,
        pixels: t.rgba8().as_chunks::<4>().0.to_vec(),
    }
}

/// Cooks every texture of `tex` for `opts.target`. Returns `None` when compression is off or
/// there are no textures; otherwise the cooked files and stats (`ship_bytes` and `level` left
/// for the caller) and `tex` with each texture replaced by its decoded top level, for the
/// viewer. `stage` reports progress.
pub fn cook(
    tex: &ExportTextures,
    opts: &CookOptions,
    now: &dyn Fn() -> f64,
    stage: &mut dyn FnMut(&str),
) -> Result<Option<(Cooked, ExportTextures)>, String> {
    if opts.target == TextureCompression::Off {
        return Ok(None);
    }
    let t0 = now();
    let mut decoded = tex.clone();
    let mut textures = Vec::new();
    let mut files = Vec::new();
    let maps = [
        (Channel::BaseColor, &mut decoded.base_color),
        (Channel::Normal, &mut decoded.normal),
        (Channel::Orm, &mut decoded.metallic_roughness),
    ];
    for (channel, map) in maps {
        let several = map.len() > 1;
        for (&slot, t) in map.iter_mut() {
            // The exported materials are opaque unless alpha-tested, so alpha is data only in
            // an alpha-tested base color. Elsewhere it is set opaque, which spends no bits on
            // it and keeps mipple from bleeding color under the atlas's empty (alpha 0) texels.
            let cutoff = match channel {
                Channel::BaseColor => tex.alpha_cutoff.get(&slot).copied(),
                _ => None,
            };
            let mut src = surface(t);
            if cutoff.is_none() {
                for p in &mut src.pixels {
                    p[3] = 255;
                }
            }
            let has_alpha = src.has_alpha();
            let Some(format) = opts.target.format(channel, has_alpha) else {
                continue;
            };
            stage(&format!(
                "mipple: {} {}×{} → {}",
                channel.name().replace('_', " "),
                t.width,
                t.height,
                format.name()
            ));
            let normal = channel == Channel::Normal;
            let srgb = t.srgb && !normal;
            let eo = EncodeOptions {
                effort: opts.effort.min(mipple::format::MAX_EFFORT),
                srgb,
                normal_map: normal,
                // Alpha-tested base color keeps its coverage down the mips.
                alpha_cutoff: cutoff,
                rdo: RdoParams {
                    lambda: opts.lambda.clamp(0.0, 16.0),
                    ..RdoParams::default()
                },
                gpu: GpuMode::Off,
                ..EncodeOptions::new(format)
            };
            let (enc, _) = encode(&src, &eo);
            let mut back = decode_level(&enc, 0, normal);
            if normal && format == Format::EacRg11 {
                // BC5's decode rebuilds Z; EAC RG11's leaves it to the shader, as here.
                for p in &mut back.pixels {
                    let v = normal_from_rg(p[0], p[1]);
                    p[2] = ((v[2] + 1.0) * 127.5 + 0.5) as u8;
                }
            }
            let ktx2 = mipple::ktx2::write(
                &enc,
                Some(ZSTD_LEVEL),
                Some("farthing demo-web, encoded by mipple"),
                PrepMode::Off,
            )
            .map_err(|e| format!("KTX2: {e}"))?;
            let (score, normal_error) = if normal {
                (None, Some(normal_angle_error_deg(&src, &back) as f32))
            } else {
                // Measured against the top level mipple encoded, which has color bled under
                // transparent texels; the blocks' error, not the bleed's.
                let top = if has_alpha {
                    prepare(
                        &src,
                        &EncodeOptions {
                            mips: false,
                            ..eo.clone()
                        },
                    )
                    .swap_remove(0)
                    .into_owned()
                } else {
                    src.clone()
                };
                let db = psnr(&top, &back, &format.metric_channels(has_alpha));
                (db.is_finite().then_some(db as f32), None)
            };
            textures.push(CookedTexture {
                channel: channel.name(),
                format: format.name(),
                width: t.width,
                height: t.height,
                mips: enc.levels.len(),
                gpu_bytes: enc.levels.iter().map(|l| l.data.len()).sum(),
                ktx2_bytes: ktx2.len(),
                psnr: score,
                normal_error,
            });
            let suffix = if several {
                format!("_{slot}")
            } else {
                String::new()
            };
            files.push((format!("textures/{}{suffix}.ktx2", channel.name()), ktx2));
            *t = Texture::from_rgba8(t.width, t.height, back.pixels.as_flattened(), t.srgb)
                .map_err(|e| e.to_string())?;
        }
    }
    if textures.is_empty() {
        return Ok(None);
    }
    let info = CookedInfo {
        target: opts.target.target(),
        effort: opts.effort.min(mipple::format::MAX_EFFORT),
        lambda: opts.lambda.clamp(0.0, 16.0),
        zstd_level: ZSTD_LEVEL,
        gpu_bytes: textures.iter().map(|t| t.gpu_bytes).sum(),
        ktx2_bytes: textures.iter().map(|t| t.ktx2_bytes).sum(),
        rgba_bytes: texture_bytes(tex),
        textures,
        ship_bytes: 0,
        ms: now() - t0,
        level: 0,
    };
    Ok(Some((Cooked { info, files }, decoded)))
}

/// `t` without its textures (factors kept): the GLB that ships next to the KTX2 files.
pub fn bare(t: &ExportTextures) -> ExportTextures {
    ExportTextures {
        base_color: Default::default(),
        normal: Default::default(),
        metallic_roughness: Default::default(),
        ..t.clone()
    }
}

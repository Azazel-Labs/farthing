//! The `EXT_meshopt_compression` bitstreams: meshoptimizer's attribute (v0) and triangle (v1)
//! encoders, the octahedral filter's encoder, and decoders for every mode and filter, so
//! farthing writes compressed glTF and reads what gltfpack writes.

use crate::{Error, Result};

/// Elements per delta group.
const GROUP: usize = 16;

/// The triangle codec's `codeaux` table, meshoptimizer's (built from symbol frequencies).
const CODEAUX: [u8; 16] = [
    0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0, 0,
];

/// Rotations that put a matched edge first.
const ORDER: [[usize; 3]; 3] = [[0, 1, 2], [1, 2, 0], [2, 0, 1]];

fn bad(what: &str) -> Error {
    Error::InvalidInput(format!("EXT_meshopt_compression: {what}"))
}

fn zigzag8(v: u8) -> u8 {
    if v & 0x80 != 0 {
        !(v << 1)
    } else {
        v << 1
    }
}

fn unzigzag8(v: u8) -> u8 {
    if v & 1 != 0 {
        !(v >> 1)
    } else {
        v >> 1
    }
}

/// Elements per attribute block.
fn block_elements(stride: usize) -> usize {
    ((8192 / stride) & !15).min(256)
}

/// Mode 0 (`ATTRIBUTES`): `data` holds elements of `stride` bytes (a multiple of 4, at most
/// 256). Each byte position is delta coded against the previous element in groups of 16.
pub(crate) fn encode_vertex_buffer(data: &[u8], stride: usize) -> Vec<u8> {
    assert!(stride > 0 && stride <= 256 && stride.is_multiple_of(4));
    assert!(data.len().is_multiple_of(stride));
    let count = data.len() / stride;
    let first = if count > 0 {
        data[..stride].to_vec()
    } else {
        vec![0; stride]
    };
    let mut out = vec![0xa0];
    let mut last = first.clone();
    let mut deltas = [0u8; 256];
    let mut start = 0;
    while start < count {
        let n = (count - start).min(block_elements(stride));
        let padded = n.div_ceil(GROUP) * GROUP;
        for (k, prev) in last.iter_mut().enumerate() {
            for (i, d) in deltas[..n].iter_mut().enumerate() {
                let v = data[(start + i) * stride + k];
                *d = zigzag8(v.wrapping_sub(*prev));
                *prev = v;
            }
            deltas[n..padded].fill(0);
            encode_bytes(&mut out, &deltas[..padded]);
        }
        start += n;
    }
    // Tail: the baseline element, padded in front to 32 bytes.
    out.resize(out.len() + 32usize.saturating_sub(stride), 0);
    out.extend_from_slice(&first);
    out
}

/// One data block: 2 header bits per group, then each group in the smallest of its modes.
fn encode_bytes(out: &mut Vec<u8>, deltas: &[u8]) {
    let header = out.len();
    out.resize(header + (deltas.len() / GROUP).div_ceil(4), 0);
    for (g, d) in deltas.chunks_exact(GROUP).enumerate() {
        let size = |bits: usize| match bits {
            0 if d.iter().all(|&x| x == 0) => 0,
            0 => usize::MAX,
            8 => GROUP,
            b => {
                let sentinel = (1u8 << b) - 1;
                GROUP * b / 8 + d.iter().filter(|&&x| x >= sentinel).count()
            }
        };
        let (mode, bits) = [(0u8, 0usize), (1, 2), (2, 4), (3, 8)]
            .into_iter()
            .min_by_key(|&(_, b)| size(b))
            .unwrap();
        out[header + g / 4] |= mode << ((g % 4) * 2);
        match bits {
            0 => {}
            8 => out.extend_from_slice(d),
            b => {
                let sentinel = (1u8 << b) - 1;
                let per = 8 / b;
                for chunk in d.chunks_exact(per) {
                    let mut byte = 0u8;
                    for &x in chunk {
                        byte = (byte << b) | x.min(sentinel);
                    }
                    out.push(byte);
                }
                out.extend(d.iter().filter(|&&x| x >= sentinel));
            }
        }
    }
}

/// Decodes mode 0 into `count` elements of `stride` bytes.
pub(crate) fn decode_vertex_buffer(src: &[u8], count: usize, stride: usize) -> Result<Vec<u8>> {
    if stride == 0 || stride > 256 || !stride.is_multiple_of(4) {
        return Err(bad("attribute stride"));
    }
    let tail = stride.max(32);
    if src.len() < 1 + tail || src[0] != 0xa0 {
        return Err(bad("attribute header"));
    }
    let end = src.len() - tail;
    let mut last = src[src.len() - stride..].to_vec();
    let mut out = vec![0u8; count * stride];
    let mut pos = 1;
    let mut deltas = [0u8; 256];
    let mut start = 0;
    while start < count {
        let n = (count - start).min(block_elements(stride));
        let groups = n.div_ceil(GROUP);
        for (k, prev) in last.iter_mut().enumerate() {
            let header = pos;
            pos += groups.div_ceil(4);
            if pos > end {
                return Err(bad("attribute data truncated"));
            }
            for g in 0..groups {
                let mode = (src[header + g / 4] >> ((g % 4) * 2)) & 3;
                let d = &mut deltas[g * GROUP..(g + 1) * GROUP];
                pos = decode_group(src, pos, end, mode, d)?;
            }
            for (i, &d) in deltas[..n].iter().enumerate() {
                *prev = prev.wrapping_add(unzigzag8(d));
                out[(start + i) * stride + k] = *prev;
            }
        }
        start += n;
    }
    Ok(out)
}

fn decode_group(src: &[u8], mut pos: usize, end: usize, mode: u8, d: &mut [u8]) -> Result<usize> {
    let truncated = || bad("attribute data truncated");
    match mode {
        0 => d.fill(0),
        3 => {
            d.copy_from_slice(
                src.get(pos..pos + GROUP)
                    .filter(|_| pos + GROUP <= end)
                    .ok_or_else(truncated)?,
            );
            pos += GROUP;
        }
        _ => {
            let bits = if mode == 1 { 2 } else { 4 };
            let sentinel = (1u8 << bits) - 1;
            let packed = GROUP * bits / 8;
            if pos + packed > end {
                return Err(truncated());
            }
            let per = 8 / bits;
            for (i, x) in d.iter_mut().enumerate() {
                let byte = src[pos + i / per];
                *x = (byte >> (8 - bits * (i % per + 1))) & sentinel;
            }
            pos += packed;
            for x in d.iter_mut().filter(|x| **x == sentinel) {
                if pos >= end {
                    return Err(truncated());
                }
                *x = src[pos];
                pos += 1;
            }
        }
    }
    Ok(pos)
}

/// The triangle codec's recent edges and vertices.
struct Fifos {
    edges: [[u32; 2]; 16],
    edge_at: usize,
    verts: [u32; 16],
    vert_at: usize,
}

impl Fifos {
    fn new() -> Self {
        Fifos {
            edges: [[u32::MAX; 2]; 16],
            edge_at: 0,
            verts: [u32::MAX; 16],
            vert_at: 0,
        }
    }
    /// The `i`th most recent edge.
    fn edge_at(&self, i: usize) -> [u32; 2] {
        self.edges[self.edge_at.wrapping_sub(1 + i) & 15]
    }
    /// The `i`th most recent vertex.
    fn vert(&self, i: usize) -> u32 {
        self.verts[self.vert_at.wrapping_sub(1 + i) & 15]
    }
    /// The most recent edge of triangle (a, b, c), and which rotation of it starts there.
    fn find_edge(&self, a: u32, b: u32, c: u32) -> Option<(usize, usize)> {
        (0..16).find_map(|i| match self.edge_at(i) {
            e if e == [a, b] => Some((i, 0)),
            e if e == [b, c] => Some((i, 1)),
            e if e == [c, a] => Some((i, 2)),
            _ => None,
        })
    }
    fn find_vert(&self, v: u32) -> Option<usize> {
        (0..16).find(|&i| self.vert(i) == v)
    }
    fn push_edge(&mut self, a: u32, b: u32) {
        self.edges[self.edge_at] = [a, b];
        self.edge_at = (self.edge_at + 1) & 15;
    }
    fn push_vert(&mut self, v: u32) {
        self.verts[self.vert_at] = v;
        self.vert_at = (self.vert_at + 1) & 15;
    }
}

fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_varint(src: &[u8], pos: &mut usize, end: usize) -> Result<u32> {
    let mut v = 0u32;
    for shift in (0..35).step_by(7) {
        let byte = *src
            .get(*pos)
            .filter(|_| *pos < end)
            .ok_or_else(|| bad("index data truncated"))?;
        *pos += 1;
        v |= ((byte & 0x7f) as u32) << shift;
        if byte < 0x80 {
            return Ok(v);
        }
    }
    Err(bad("index varint too long"))
}

/// `index` as a zigzag delta from `last`.
fn write_index(out: &mut Vec<u8>, index: u32, last: u32) {
    let d = index.wrapping_sub(last);
    write_varint(out, (d << 1) ^ ((d as i32 >> 31) as u32));
}

fn read_index(src: &[u8], pos: &mut usize, end: usize, last: &mut u32) -> Result<u32> {
    let v = read_varint(src, pos, end)?;
    let delta = if v & 1 != 0 { !(v >> 1) } else { v >> 1 };
    *last = last.wrapping_add(delta);
    Ok(*last)
}

/// Mode 1 (`TRIANGLES`): a triangle list, most triangles in one byte when the list is ordered
/// for the vertex cache and its vertices by first use.
pub(crate) fn encode_index_buffer(indices: &[u32]) -> Vec<u8> {
    assert!(indices.len().is_multiple_of(3));
    let mut codes = Vec::with_capacity(indices.len() / 3);
    let mut data = Vec::new();
    let mut f = Fifos::new();
    let (mut next, mut last) = (0u32, 0u32);
    for t in indices.chunks_exact(3) {
        if let Some((fe, rot)) = f.find_edge(t[0], t[1], t[2]).filter(|&(fe, _)| fe < 15) {
            // A recent edge and a third vertex: recent, next, last ± 1 or explicit.
            let [a, b, c] = ORDER[rot].map(|k| t[k]);
            let mut fec = match f.find_vert(c) {
                Some(i) if (1..13).contains(&i) => i,
                _ if c == next => {
                    next += 1;
                    0
                }
                _ => 15,
            };
            if fec == 15 {
                if c.wrapping_add(1) == last {
                    fec = 13;
                    last = c;
                }
                if c == last.wrapping_add(1) {
                    fec = 14;
                    last = c;
                }
            }
            codes.push((fe << 4 | fec) as u8);
            if fec == 15 {
                write_index(&mut data, c, last);
                last = c;
            }
            if fec == 0 || fec >= 13 {
                f.push_vert(c);
            }
            f.push_edge(c, b);
            f.push_edge(a, c);
        } else {
            // No recent edge: three vertices, each recent, next or explicit.
            let rot = if t[1] == next {
                1
            } else if t[2] == next {
                2
            } else {
                0
            };
            let [a, b, c] = ORDER[rot].map(|k| t[k]);
            // 0, 1, 2 after other vertices restarts `next` (concatenated streams).
            let reset = [a, b, c] == [0, 1, 2] && next > 0;
            if reset {
                next = 0;
                f.verts = [u32::MAX; 16];
            }
            let (fb, fc) = (f.find_vert(b), f.find_vert(c));
            let mut take = |v: u32, fifo: Option<usize>| match fifo {
                Some(i) if i < 14 => i + 1,
                _ if v == next => {
                    next += 1;
                    0
                }
                _ => 15,
            };
            let fea = take(a, None);
            let feb = take(b, fb);
            let fec = take(c, fc);
            let aux = (feb << 4 | fec) as u8;
            match CODEAUX.iter().position(|&x| x == aux) {
                Some(i) if fea == 0 && i < 14 && !reset => codes.push(0xf0 | i as u8),
                _ => {
                    codes.push(if fea == 0 { 0xfe } else { 0xff });
                    data.push(aux);
                }
            }
            for (v, fe) in [(a, fea), (b, feb), (c, fec)] {
                if fe == 15 {
                    write_index(&mut data, v, last);
                    last = v;
                }
            }
            for (v, fe) in [(a, fea), (b, feb), (c, fec)] {
                if fe == 0 || fe == 15 {
                    f.push_vert(v);
                }
            }
            f.push_edge(b, a);
            f.push_edge(c, b);
            f.push_edge(a, c);
        }
    }
    let mut out = Vec::with_capacity(1 + codes.len() + data.len() + 16);
    out.push(0xe1);
    out.extend_from_slice(&codes);
    out.extend_from_slice(&data);
    out.extend_from_slice(&CODEAUX);
    out
}

/// Decodes mode 1 into `count` indices.
pub(crate) fn decode_index_buffer(src: &[u8], count: usize) -> Result<Vec<u32>> {
    let tris = count / 3;
    if !count.is_multiple_of(3) || src.len() < 1 + tris + 16 || src[0] & 0xf0 != 0xe0 {
        return Err(bad("triangle header"));
    }
    if src[0] & 15 != 1 {
        return Err(bad("unsupported triangle codec version"));
    }
    let end = src.len() - 16;
    let table = &src[end..];
    let mut pos = 1 + tris;
    let mut f = Fifos::new();
    let (mut next, mut last) = (0u32, 0u32);
    let mut out = Vec::with_capacity(count);
    for &code in &src[1..1 + tris] {
        let (hi, lo) = ((code >> 4) as usize, (code & 15) as usize);
        let tri = if hi < 15 {
            let [a, b] = f.edge_at(hi);
            let c = match lo {
                0 => {
                    next += 1;
                    next - 1
                }
                1..=12 => f.vert(lo),
                13 => {
                    last = last.wrapping_sub(1);
                    last
                }
                14 => {
                    last = last.wrapping_add(1);
                    last
                }
                _ => read_index(src, &mut pos, end, &mut last)?,
            };
            if lo == 0 || lo >= 13 {
                f.push_vert(c);
            }
            f.push_edge(c, b);
            f.push_edge(a, c);
            [a, b, c]
        } else {
            let aux = if lo < 14 {
                table[lo]
            } else {
                let x = *src
                    .get(pos)
                    .filter(|_| pos < end)
                    .ok_or_else(|| bad("index data truncated"))?;
                pos += 1;
                if x == 0 {
                    next = 0;
                }
                x
            };
            let (feb, fec) = ((aux >> 4) as usize, (aux & 15) as usize);
            let fea = if lo == 15 { 15 } else { 0 };
            // FIFO reads and `next` increments happen in a, b, c order, before any pushes.
            let mut pick = |fe: usize| -> Option<u32> {
                match fe {
                    0 => {
                        next += 1;
                        Some(next - 1)
                    }
                    15 => None,
                    i => Some(f.vert(i - 1)),
                }
            };
            let (a, b, c) = (pick(fea), pick(feb), pick(fec));
            let mut explicit = |v: Option<u32>| match v {
                Some(v) => Ok(v),
                None => read_index(src, &mut pos, end, &mut last),
            };
            let (a, b, c) = (explicit(a)?, explicit(b)?, explicit(c)?);
            f.push_vert(a);
            if feb == 0 || feb == 15 {
                f.push_vert(b);
            }
            if fec == 0 || fec == 15 {
                f.push_vert(c);
            }
            f.push_edge(b, a);
            f.push_edge(c, b);
            f.push_edge(a, c);
            [a, b, c]
        };
        out.extend_from_slice(&tri);
    }
    Ok(out)
}

/// Decodes mode 2 (`INDICES`, any index sequence) into `count` indices.
pub(crate) fn decode_index_sequence(src: &[u8], count: usize) -> Result<Vec<u32>> {
    if src.len() < 1 + count + 4 || src[0] != 0xd1 {
        return Err(bad("index sequence header"));
    }
    let end = src.len() - 4;
    let mut pos = 1;
    let mut last = [0u32; 2];
    (0..count)
        .map(|_| {
            let v = read_varint(src, &mut pos, end)?;
            let base = (v & 1) as usize;
            let delta = if v & 2 != 0 { !(v >> 2) } else { v >> 2 };
            last[base] = last[base].wrapping_add(delta);
            Ok(last[base])
        })
        .collect()
}

fn snorm(v: f32, bits: u32) -> i32 {
    let scale = ((1 << (bits - 1)) - 1) as f32;
    let v = v.clamp(-1.0, 1.0);
    (v * scale + if v >= 0.0 { 0.5 } else { -0.5 }) as i32
}

/// The octahedral filter's 8-bit input for unit vector `n` and fourth component `w`
/// (passed through; ±1 for a tangent's handedness).
pub(crate) fn encode_oct8(n: [f32; 3], w: f32) -> [u8; 4] {
    let l = n[0].abs() + n[1].abs() + n[2].abs();
    let s = if l == 0.0 { 0.0 } else { 1.0 / l };
    let (x, y, z) = (n[0] * s, n[1] * s, n[2] * s);
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let u = if z >= 0.0 {
        x
    } else {
        (1.0 - y.abs()) * sign(x)
    };
    let v = if z >= 0.0 {
        y
    } else {
        (1.0 - x.abs()) * sign(y)
    };
    [snorm(u, 8), snorm(v, 8), snorm(1.0, 8), snorm(w, 8)].map(|c| c as i8 as u8)
}

/// Applies the named filter to decoded attribute data in place.
pub(crate) fn apply_filter(filter: &str, data: &mut [u8], stride: usize) -> Result<()> {
    match filter {
        "NONE" => {}
        "OCTAHEDRAL" if stride == 4 => {
            for e in data.chunks_exact_mut(4) {
                let [x, y, z] = octahedral(
                    [e[0] as i8 as f32, e[1] as i8 as f32, e[2] as i8 as f32],
                    127.0,
                );
                e[..3].copy_from_slice(&[x as i8 as u8, y as i8 as u8, z as i8 as u8]);
            }
        }
        "OCTAHEDRAL" if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| i16::from_le_bytes([e[2 * i], e[2 * i + 1]]) as f32;
                let out = octahedral([c(0), c(1), c(2)], 32767.0);
                for (i, v) in out.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&(v as i16).to_le_bytes());
                }
            }
        }
        "QUATERNION" if stride == 8 => {
            for e in data.chunks_exact_mut(8) {
                let c = |i: usize| i16::from_le_bytes([e[2 * i], e[2 * i + 1]]);
                let range = std::f32::consts::FRAC_1_SQRT_2;
                let one = (c(3) | 3) as f32;
                let x = c(0) as f32 / one * range;
                let y = c(1) as f32 / one * range;
                let z = c(2) as f32 / one * range;
                let w = (1.0 - x * x - y * y - z * z).max(0.0).sqrt();
                let max = (c(3) & 3) as usize;
                let mut out = [0i16; 4];
                for (k, v) in [(1, x), (2, y), (3, z), (0, w)] {
                    out[(max + k) % 4] = (v * 32767.0).round() as i16;
                }
                for (i, v) in out.into_iter().enumerate() {
                    e[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        "EXPONENTIAL" if stride.is_multiple_of(4) => {
            for e in data.chunks_exact_mut(4) {
                let v = i32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                let m = (v << 8) >> 8;
                let f = (m as f32) * 2f32.powi(v >> 24);
                e.copy_from_slice(&f.to_le_bytes());
            }
        }
        _ => return Err(bad(&format!("filter {filter} with stride {stride}"))),
    }
    Ok(())
}

/// The octahedral filter: components in units of `one` (the third input) to a unit vector in
/// units of `max`.
fn octahedral(c: [f32; 3], max: f32) -> [f32; 3] {
    let (mut x, mut y) = (c[0] / c[2], c[1] / c[2]);
    let z = 1.0 - x.abs() - y.abs();
    let t = z.min(0.0);
    x += if x >= 0.0 { t } else { -t };
    y += if y >= 0.0 { t } else { -t };
    let len = (x * x + y * y + z * z).sqrt();
    [x, y, z].map(|v| (v / len * max).round())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg32;

    #[test]
    fn attributes_roundtrip() {
        let mut r = Pcg32::new(3);
        for (count, stride) in [
            (0, 4),
            (1, 4),
            (17, 8),
            (300, 12),
            (1000, 16),
            (40, 256),
            (700, 64),
        ] {
            // Smooth data with occasional jumps exercises every group mode.
            let mut data = vec![0u8; count * stride];
            for i in 1..count {
                for k in 0..stride {
                    let prev = data[(i - 1) * stride + k];
                    let step = match r.below(10) {
                        0 => r.below(256) as u8,
                        1..=3 => r.below(9) as u8,
                        _ => 0,
                    };
                    data[i * stride + k] = prev.wrapping_add(step);
                }
            }
            let enc = encode_vertex_buffer(&data, stride);
            assert_eq!(
                decode_vertex_buffer(&enc, count, stride).unwrap(),
                data,
                "{count}x{stride}"
            );
        }
    }

    #[test]
    fn triangles_roundtrip() {
        // A grid in cache order, a shuffled one, two streams concatenated (reset) and
        // strip-like and far-apart explicit indices.
        let n = 30u32;
        let mut grid = Vec::new();
        for y in 0..n {
            for x in 0..n {
                let a = y * (n + 1) + x;
                grid.extend_from_slice(&[a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        let mut r = Pcg32::new(9);
        let mut shuffled: Vec<[u32; 3]> =
            grid.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        for i in (1..shuffled.len()).rev() {
            shuffled.swap(i, r.below(i as u32 + 1) as usize);
        }
        let shuffled = shuffled.concat();
        let concat = [&grid[..], &grid[..]].concat();
        let wild: Vec<u32> = (0..300).map(|_| r.next_u32() >> r.below(32)).collect();
        let strip: Vec<u32> = (0..100u32).flat_map(|i| [i + 2, i + 1, i]).collect();
        for idx in [&grid, &shuffled, &concat, &wild, &strip, &vec![]] {
            // Triangles keep their order and winding, but may start at another vertex.
            let canon = |v: &[u32]| -> Vec<[u32; 3]> {
                v.chunks_exact(3)
                    .map(|t| {
                        let r = (0..3).min_by_key(|&k| t[k]).unwrap();
                        [t[r], t[(r + 1) % 3], t[(r + 2) % 3]]
                    })
                    .collect()
            };
            let enc = encode_index_buffer(idx);
            assert_eq!(
                canon(&decode_index_buffer(&enc, idx.len()).unwrap()),
                canon(idx)
            );
        }
        // Ordered input costs about a byte per triangle.
        assert!(encode_index_buffer(&grid).len() < grid.len() / 3 * 3 / 2);
    }

    /// The worked example in the extension's specification (4-bit sentinel group).
    #[test]
    fn spec_group_example() {
        let src = [0x17, 0x5f, 0xf0, 0xbc, 0x77, 0xa9, 0x21, 0x00, 0x34, 0xb5];
        let mut d = [0u8; 16];
        assert_eq!(
            decode_group(&src, 0, src.len(), 2, &mut d).unwrap(),
            src.len()
        );
        let got: Vec<i8> = d.iter().map(|&x| unzigzag8(x) as i8).collect();
        assert_eq!(
            got,
            [-1, -4, -3, 26, -91, 0, -6, 6, -4, -4, 5, -5, 1, -1, 0, 0]
        );
    }

    #[test]
    fn index_sequence_decodes() {
        // 5, 3 (baseline 0), 100 (baseline 1), 4 (baseline 0).
        let mut src = vec![0xd1];
        for (delta, base) in [(5i32, 0u32), (-2, 0), (100, 1), (1, 0)] {
            let z = ((delta << 1) ^ (delta >> 31)) as u32;
            write_varint(&mut src, z << 1 | base);
        }
        src.extend_from_slice(&[0; 4]);
        assert_eq!(decode_index_sequence(&src, 4).unwrap(), vec![5, 3, 100, 4]);
    }

    #[test]
    fn octahedral_roundtrip() {
        let mut r = Pcg32::new(1);
        for _ in 0..2000 {
            let v = glam::Vec3::new(r.f32() - 0.5, r.f32() - 0.5, r.f32() - 0.5);
            let n = v.normalize_or(glam::Vec3::Z);
            let mut e = encode_oct8(n.to_array(), -1.0).to_vec();
            apply_filter("OCTAHEDRAL", &mut e, 4).unwrap();
            let d =
                glam::Vec3::new(e[0] as i8 as f32, e[1] as i8 as f32, e[2] as i8 as f32) / 127.0;
            assert!(d.normalize().dot(n) > 0.999, "{n} -> {d}");
            assert_eq!(e[3] as i8, -127);
        }
    }
}

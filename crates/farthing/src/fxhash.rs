//! Small-key hasher (multiply-rotate, as in rustc's FxHash) for maps keyed by vertex, face and
//! corner ids or attribute bits, where SipHash's DoS resistance only costs time. Results never
//! depend on map iteration order (it is sorted wherever it matters), so the hasher does not
//! change output.

#[derive(Default, Clone, Copy)]
pub(crate) struct Fx(u64);

impl std::hash::Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Slices of integers hash as their bytes: take them a word at a time.
        let mut words = bytes.chunks_exact(8);
        for w in &mut words {
            self.write_u64(u64::from_le_bytes([
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7],
            ]));
        }
        let rest = words.remainder();
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.write_u64(u64::from_le_bytes(last));
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) type FxBuild = std::hash::BuildHasherDefault<Fx>;
pub(crate) type HashMap<K, V> = std::collections::HashMap<K, V, FxBuild>;
pub(crate) type HashSet<K> = std::collections::HashSet<K, FxBuild>;

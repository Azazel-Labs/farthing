//! Envelope (profile) Cholesky for the small symmetric positive-definite systems of chart
//! flattening. Rows are reordered with reverse Cuthill–McKee, which keeps a mesh Laplacian's
//! profile near O(n·√n); each row's band is stored contiguously, so factorization and solves are
//! dense dot products over contiguous memory.

/// LLᵀ factor of a permuted SPD matrix stored row-wise by envelope.
pub struct EnvelopeCholesky {
    n: usize,
    /// perm[new] = old index.
    perm: Vec<usize>,
    /// First stored column of each row and the offset of its band in `data` (band ends at the
    /// diagonal).
    first: Vec<usize>,
    offset: Vec<usize>,
    data: Vec<f64>,
}

/// Envelope entries above which callers should prefer an iterative solver.
pub const MAX_ENVELOPE: usize = 6_000_000;

impl EnvelopeCholesky {
    /// Factors the matrix given as entries `(row, col, value)` of one triangle (each
    /// off-diagonal pair listed once; duplicates are summed) with `n` unknowns. None if the envelope is too large or the
    /// matrix is not positive definite.
    pub fn factor(n: usize, entries: &[(usize, usize, f64)]) -> Option<EnvelopeCholesky> {
        if n == 0 {
            return None;
        }
        // Adjacency for the ordering (off-diagonal structure) as CSR: counting sort by row,
        // then each short row sorted and deduplicated in place (same CSR as a global sort of
        // all pairs, at a fraction of the cost).
        let mut count = vec![0usize; n + 1];
        for &(i, j, _) in entries {
            if i != j {
                count[i + 1] += 1;
                count[j + 1] += 1;
            }
        }
        for i in 0..n {
            count[i + 1] += count[i];
        }
        let mut fill = count.clone();
        let mut raw = vec![0u32; count[n]];
        for &(i, j, _) in entries {
            if i != j {
                raw[fill[i]] = j as u32;
                fill[i] += 1;
                raw[fill[j]] = i as u32;
                fill[j] += 1;
            }
        }
        let mut start = vec![0usize; n + 1];
        let mut nbrs: Vec<u32> = Vec::with_capacity(raw.len());
        for i in 0..n {
            let row = &mut raw[count[i]..count[i + 1]];
            row.sort_unstable();
            let mut last = u32::MAX;
            for &j in row.iter() {
                if j != last {
                    nbrs.push(j);
                    last = j;
                }
            }
            start[i + 1] = nbrs.len();
        }
        let adj = Csr { start, nbrs };
        let perm = rcm(&adj);
        let mut inv = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let mut first: Vec<usize> = (0..n).collect();
        for &(i, j, _) in entries {
            let (a, b) = (inv[i], inv[j]);
            let (r, c) = if a >= b { (a, b) } else { (b, a) };
            first[r] = first[r].min(c);
        }
        let mut offset = Vec::with_capacity(n + 1);
        let mut total = 0usize;
        for (i, &f) in first.iter().enumerate() {
            offset.push(total);
            total += i - f + 1;
        }
        if total > MAX_ENVELOPE {
            return None;
        }
        let mut data = vec![0.0f64; total];
        for &(i, j, v) in entries {
            let (a, b) = (inv[i], inv[j]);
            let (r, c) = if a >= b { (a, b) } else { (b, a) };
            data[offset[r] + c - first[r]] += v;
        }
        let mut f = EnvelopeCholesky {
            n,
            perm,
            first,
            offset,
            data,
        };
        if f.factorize() {
            Some(f)
        } else {
            None
        }
    }

    #[inline]
    fn row(&self, i: usize) -> &[f64] {
        &self.data[self.offset[i]..self.offset[i] + (i - self.first[i] + 1)]
    }

    fn factorize(&mut self) -> bool {
        for i in 0..self.n {
            let fi = self.first[i];
            for j in fi..=i {
                let fj = self.first[j];
                let k0 = fi.max(fj);
                // Σ_{k0 ≤ k < j} L[i][k]·L[j][k], both rows contiguous over k.
                let s = {
                    let ri = &self.data[self.offset[i] + (k0 - fi)..self.offset[i] + (j - fi)];
                    let rj = &self.data[self.offset[j] + (k0 - fj)..self.offset[j] + (j - fj)];
                    dot(ri, rj)
                };
                let idx = self.offset[i] + (j - fi);
                if j < i {
                    let djj = self.data[self.offset[j] + (j - fj)];
                    self.data[idx] = (self.data[idx] - s) / djj;
                } else {
                    let d = self.data[idx] - s;
                    if d <= 0.0 || !d.is_finite() {
                        return false;
                    }
                    self.data[idx] = d.sqrt();
                }
            }
        }
        true
    }

    /// Solves A x = b in place (b in original ordering).
    pub fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = (0..n).map(|i| b[self.perm[i]]).collect();
        // Forward: L z = y.
        for i in 0..n {
            let fi = self.first[i];
            let r = self.row(i);
            let s = dot(&r[..i - fi], &y[fi..i]);
            y[i] = (y[i] - s) / r[i - fi];
        }
        // Backward: Lᵀ x = z.
        for i in (0..n).rev() {
            let fi = self.first[i];
            let r = self.row(i);
            y[i] /= r[i - fi];
            let yi = y[i];
            for (k, &l) in r[..i - fi].iter().enumerate() {
                y[fi + k] -= l * yi;
            }
        }
        for i in 0..n {
            b[self.perm[i]] = y[i];
        }
    }
}

/// Dot product with four independent accumulators (vectorizes; fixed order, deterministic).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Reverse Cuthill–McKee ordering (per connected component, from a pseudo-peripheral node).
struct Csr {
    start: Vec<usize>,
    nbrs: Vec<u32>,
}

impl Csr {
    #[inline]
    fn of(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        self.nbrs[self.start[u]..self.start[u + 1]]
            .iter()
            .map(|&v| v as usize)
    }
    #[inline]
    fn degree(&self, u: usize) -> usize {
        self.start[u + 1] - self.start[u]
    }
}

fn rcm(adj: &Csr) -> Vec<usize> {
    let n = adj.start.len() - 1;
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let bfs_last = |start: usize, seen_mask: &[bool]| -> (usize, usize) {
        // Returns (farthest node, eccentricity) within start's component.
        let mut dist = vec![usize::MAX; n];
        let mut q = std::collections::VecDeque::from([start]);
        dist[start] = 0;
        let mut last = start;
        while let Some(u) = q.pop_front() {
            last = u;
            for v in adj.of(u) {
                if dist[v] == usize::MAX && !seen_mask[v] {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        (last, dist[last])
    };
    for s in 0..n {
        if seen[s] {
            continue;
        }
        // Pseudo-peripheral start: repeat BFS from the farthest node while the eccentricity grows.
        let (mut far, mut ecc) = bfs_last(s, &seen);
        for _ in 0..4 {
            let (f2, e2) = bfs_last(far, &seen);
            if e2 <= ecc {
                break;
            }
            far = f2;
            ecc = e2;
        }
        let start = far;
        let mut comp = Vec::new();
        let mut q = std::collections::VecDeque::from([start]);
        seen[start] = true;
        while let Some(u) = q.pop_front() {
            comp.push(u);
            let mut nb: Vec<usize> = adj.of(u).filter(|&v| !seen[v]).collect();
            nb.sort_by_key(|&v| (adj.degree(v), v));
            for v in nb {
                seen[v] = true;
                q.push_back(v);
            }
        }
        comp.reverse();
        order.extend(comp);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_grid_laplacian() {
        // 2D grid Laplacian + identity shift (SPD).
        let k = 12;
        let n = k * k;
        let mut e = Vec::new();
        for y in 0..k {
            for x in 0..k {
                let i = y * k + x;
                let mut deg = 1.0;
                for (dx, dy) in [(1i32, 0i32), (0, 1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < k as i32 && ny < k as i32 {
                        let j = ny as usize * k + nx as usize;
                        e.push((i, j, -1.0));
                    }
                }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx >= 0 && ny >= 0 && nx < k as i32 && ny < k as i32 {
                        deg += 1.0;
                    }
                }
                e.push((i, i, deg));
            }
        }
        let f = EnvelopeCholesky::factor(n, &e).unwrap();
        let xs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        // b = A x
        let mut b = vec![0.0; n];
        for &(i, j, v) in &e {
            b[i] += v * xs[j];
            if i != j {
                b[j] += v * xs[i];
            }
        }
        f.solve(&mut b);
        for i in 0..n {
            assert!((b[i] - xs[i]).abs() < 1e-9, "x[{i}] {} vs {}", b[i], xs[i]);
        }
    }
}

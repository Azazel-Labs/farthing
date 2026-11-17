//! Streaming pages for a [`ClusterDag`], independent of any file format.
//!
//! A renderer that streams cluster LOD loads the DAG in pages: fixed-size blocks of clusters
//! that are made resident as a whole. [`plan_pages`] decides which clusters go in which page
//! and what each page needs resident first; a [`PageLayout`] says how many bytes a page takes
//! in your format. The packed [`GpuClusterDag`](super::GpuClusterDag) is one such format (its
//! layout is [`FcldLayout`](super::gpu::FcldLayout)); write your own `PageLayout` to plan pages
//! for another, then write the bytes however you like.
//!
//! The rules the plan follows, which a streaming renderer relies on:
//!
//! * **A group's output never spans pages.** The clusters one [`ClusterGroup`](super::ClusterGroup)
//!   produced share their LOD error and bounds, so they become visible together. They are the
//!   unit of packing. Source-level clusters (produced by no group) are packed by the group that
//!   consumes them.
//! * **Coarse to fine.** Units go in level-descending order, so the roots are in the first
//!   pages and pages streamed in order always have a cluster's coarser replacement resident
//!   before the cluster itself.
//! * **Dependencies point back.** Each page lists the pages holding its clusters' parent
//!   groups' output (what is drawn instead of them), all with lower indices. A page may be made
//!   resident once its dependencies are.
//!
//! Pages are filled greedily: a unit joins the open page if the page still fits
//! [`PageLayout::max_page_bytes`] with it, and starts a new page otherwise.
//!
//! ```
//! use farthing::cluster::{Cluster, ClusterOptions, PageLayout, PageTally, build_cluster_dag, plan_pages};
//! # let mesh = farthing::Mesh::new(
//! #     vec![glam::Vec3::ZERO, glam::Vec3::X, glam::Vec3::Y],
//! #     vec![0, 1, 2],
//! # );
//!
//! /// 32-byte vertices, a 16-byte header per cluster, u16 indices, 64 KiB pages.
//! struct MyLayout;
//! impl PageLayout for MyLayout {
//!     fn cluster_bytes(&self, cluster: &Cluster, _vertices: usize) -> [usize; 4] {
//!         [16, 2 * cluster.indices.len(), 0, 0]
//!     }
//!     fn page_bytes(&self, t: &PageTally) -> usize {
//!         64 + 32 * t.vertices + t.bytes[0] + t.bytes[1].next_multiple_of(4)
//!     }
//!     fn max_page_bytes(&self) -> usize {
//!         64 * 1024
//!     }
//! }
//!
//! let dag = build_cluster_dag(&mesh, &ClusterOptions::default())?;
//! let plan = plan_pages(&dag, &MyLayout)?;
//! for page in &plan.pages {
//!     // page.clusters: DAG cluster indices, page.vertices: the source vertices they use,
//!     // page.dependencies: pages to load first.
//!     assert!(page.dependencies.iter().all(|&d| (d as usize) < plan.pages.len()));
//! }
//! # Ok::<(), farthing::Error>(())
//! ```

use super::{Cluster, ClusterDag};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

const NONE: u32 = u32::MAX;

/// Running totals of a page, for [`PageLayout::page_bytes`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageTally {
    pub clusters: usize,
    /// Distinct source vertices: a vertex shared by two clusters of the page counts once.
    pub vertices: usize,
    /// The clusters' own distinct vertex counts, summed: a vertex shared by two clusters of
    /// the page counts twice (what per-cluster vertex index lists take).
    pub cluster_vertices: usize,
    pub triangles: usize,
    /// [`PageLayout::cluster_bytes`] summed over the page's clusters, per section.
    pub bytes: [usize; 4],
}

/// How many bytes a page takes in a format.
pub trait PageLayout {
    /// What `cluster` adds to a page besides its share of the page's vertices, in up to four
    /// sections the format pads separately (say headers, vertex indices, triangles,
    /// materials). `vertices` is the cluster's distinct vertex count. Summed into
    /// [`PageTally::bytes`]. None by default.
    fn cluster_bytes(&self, cluster: &Cluster, vertices: usize) -> [usize; 4] {
        let _ = (cluster, vertices);
        [0; 4]
    }

    /// The size of a page with these totals, padding included. It must not shrink as
    /// clusters are added.
    fn page_bytes(&self, tally: &PageTally) -> usize;

    /// The largest page. Every group's output must fit in one page.
    fn max_page_bytes(&self) -> usize;
}

/// One planned page.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlannedPage {
    /// DAG cluster indices, in order: whole units (a group's output), coarse to fine.
    pub clusters: Vec<u32>,
    /// The distinct source vertices the clusters use, in first-use order (cluster by cluster,
    /// each cluster's in the order its triangles first use them).
    pub vertices: Vec<u32>,
    /// Pages holding the clusters this page's clusters are replaced by, ascending; all lower
    /// than this page.
    pub dependencies: Vec<u32>,
    /// The highest cluster level in the page.
    pub level: u32,
    pub tally: PageTally,
}

/// The pages of a DAG; see the [module docs](self).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PagePlan {
    pub pages: Vec<PlannedPage>,
    /// Each DAG cluster's page.
    pub cluster_page: Vec<u32>,
    /// Each DAG group's page: where the clusters it produced are (`u32::MAX` if it produced
    /// none).
    pub group_page: Vec<u32>,
}

impl PagePlan {
    /// The DAG cluster indices in page order (every cluster once).
    pub fn cluster_order(&self) -> impl Iterator<Item = u32> + '_ {
        self.pages.iter().flat_map(|p| p.clusters.iter().copied())
    }
}

/// One packing unit: clusters that always share a page.
struct Unit {
    level: u32,
    clusters: Vec<u32>,
}

/// Each cluster's distinct vertices in first-use order (the order [`Cluster::meshlet`] uses).
fn cluster_vertices(dag: &ClusterDag, nv: usize) -> Vec<Vec<u32>> {
    let mut stamp = vec![NONE; nv];
    dag.clusters
        .iter()
        .enumerate()
        .map(|(c, cl)| {
            let mut vs = Vec::new();
            for &v in &cl.indices {
                if std::mem::replace(&mut stamp[v as usize], c as u32) != c as u32 {
                    vs.push(v);
                }
            }
            vs
        })
        .collect()
}

/// Splits `dag` into streaming pages sized by `layout`; see the [module docs](self).
pub fn plan_pages<L: PageLayout + ?Sized>(dag: &ClusterDag, layout: &L) -> Result<PagePlan> {
    let invalid = |m: String| Err(Error::InvalidInput(m));
    let ng = dag.groups.len();
    let nc = dag.clusters.len();
    if dag.clusters.iter().any(|c| {
        (c.group != NONE && c.group as usize >= ng)
            || (c.parent_group != NONE && c.parent_group as usize >= ng)
    }) || dag
        .groups
        .iter()
        .any(|g| g.clusters.iter().any(|&c| c as usize >= nc))
    {
        return invalid("cluster group out of range".into());
    }
    let nv = dag
        .clusters
        .iter()
        .flat_map(|c| c.indices.iter())
        .max()
        .map_or(0, |&v| v as usize + 1);
    let verts = cluster_vertices(dag, nv);

    // Units: each group's output, then source-level clusters by consuming group; coarse to
    // fine.
    let mut units: Vec<Unit> = dag
        .groups
        .iter()
        .map(|g| Unit {
            level: g.level + 1,
            clusters: g.clusters.clone(),
        })
        .collect();
    let mut by_parent: Vec<Vec<u32>> = vec![Vec::new(); ng + 1];
    for (i, c) in dag.clusters.iter().enumerate() {
        if c.group == NONE {
            let k = if c.parent_group == NONE {
                ng
            } else {
                c.parent_group as usize
            };
            by_parent[k].push(i as u32);
        }
    }
    for list in by_parent {
        if !list.is_empty() {
            units.push(Unit {
                level: dag.clusters[list[0] as usize].level,
                clusters: list,
            });
        }
    }
    // Stable: within a level, groups keep their (spatially coherent) build order.
    units.sort_by_key(|u| std::cmp::Reverse(u.level));

    let cluster_bytes: Vec<[usize; 4]> = dag
        .clusters
        .iter()
        .zip(&verts)
        .map(|(c, v)| layout.cluster_bytes(c, v.len()))
        .collect();
    let add = |t: &mut PageTally, c: usize| {
        t.clusters += 1;
        t.cluster_vertices += verts[c].len();
        t.triangles += dag.clusters[c].indices.len() / 3;
        for (s, b) in t.bytes.iter_mut().zip(cluster_bytes[c]) {
            *s += b;
        }
    };
    let max = layout.max_page_bytes();
    let too_big = |n: usize| {
        invalid(format!(
            "a group's clusters need {n} bytes, more than the page size {max}"
        ))
    };

    // Greedy packing. `in_page[v]` is the page (+1) that holds source vertex `v`, `seen[v]`
    // the unit (+1) that last counted it.
    let mut in_page = vec![0u32; nv];
    let mut seen = vec![0u32; nv];
    let mut page_units: Vec<(Vec<usize>, PageTally)> = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    let mut tally = PageTally::default();
    for (u, unit) in units.iter().enumerate() {
        let page = page_units.len() as u32 + 1;
        let tag = u as u32 + 1;
        let mut grown = tally;
        for &c in &unit.clusters {
            for &v in &verts[c as usize] {
                let v = v as usize;
                if in_page[v] != page && seen[v] != tag {
                    seen[v] = tag;
                    grown.vertices += 1;
                }
            }
            add(&mut grown, c as usize);
        }
        if layout.page_bytes(&grown) > max {
            if open.is_empty() {
                return too_big(layout.page_bytes(&grown));
            }
            page_units.push((std::mem::take(&mut open), tally));
            tally = PageTally::default();
        }
        let page = page_units.len() as u32 + 1;
        for &c in &unit.clusters {
            for &v in &verts[c as usize] {
                let v = v as usize;
                if in_page[v] != page {
                    in_page[v] = page;
                    tally.vertices += 1;
                }
            }
            add(&mut tally, c as usize);
        }
        if layout.page_bytes(&tally) > max {
            return too_big(layout.page_bytes(&tally));
        }
        open.push(u);
    }
    if !open.is_empty() {
        page_units.push((open, tally));
    }

    let mut cluster_page = vec![NONE; nc];
    for (p, (us, _)) in page_units.iter().enumerate() {
        for &u in us {
            for &c in &units[u].clusters {
                cluster_page[c as usize] = p as u32;
            }
        }
    }
    if cluster_page.contains(&NONE) {
        return invalid("a cluster belongs to no group and is consumed by none".into());
    }
    let group_page: Vec<u32> = dag
        .groups
        .iter()
        .map(|g| {
            g.clusters
                .first()
                .map_or(NONE, |&c| cluster_page[c as usize])
        })
        .collect();

    let mut stamp = vec![NONE; nv];
    let pages = page_units
        .into_iter()
        .enumerate()
        .map(|(p, (us, tally))| {
            let clusters: Vec<u32> = us
                .iter()
                .flat_map(|&u| units[u].clusters.iter().copied())
                .collect();
            let mut vertices = Vec::with_capacity(tally.vertices);
            for &c in &clusters {
                for &v in &verts[c as usize] {
                    if std::mem::replace(&mut stamp[v as usize], p as u32) != p as u32 {
                        vertices.push(v);
                    }
                }
            }
            let mut dependencies: Vec<u32> = clusters
                .iter()
                .map(|&c| dag.clusters[c as usize].parent_group)
                .filter(|&g| g != NONE)
                .map(|g| group_page[g as usize])
                .filter(|&d| d != NONE && d != p as u32)
                .collect();
            dependencies.sort_unstable();
            dependencies.dedup();
            let level = clusters
                .iter()
                .map(|&c| dag.clusters[c as usize].level)
                .max()
                .unwrap_or(0);
            PlannedPage {
                clusters,
                vertices,
                dependencies,
                level,
                tally,
            }
        })
        .collect();
    Ok(PagePlan {
        pages,
        cluster_page,
        group_page,
    })
}

#[cfg(test)]
mod tests {
    use super::super::tests::blob;
    use super::super::{ClusterOptions, build_cluster_dag};
    use super::*;

    /// A plain layout: 16-byte vertices, 32 bytes per cluster, 4-byte triangles.
    struct Plain(usize);
    impl PageLayout for Plain {
        fn cluster_bytes(&self, c: &Cluster, _: usize) -> [usize; 4] {
            [32 + 4 * c.triangle_count(), 0, 0, 0]
        }
        fn page_bytes(&self, t: &PageTally) -> usize {
            16 + 16 * t.vertices + t.bytes[0]
        }
        fn max_page_bytes(&self) -> usize {
            self.0
        }
    }

    #[test]
    fn plan_follows_the_rules() {
        let m = blob(60);
        let dag = build_cluster_dag(&m, &ClusterOptions::default()).unwrap();
        for size in [24 * 1024, 64 * 1024] {
            let plan = plan_pages(&dag, &Plain(size)).unwrap();
            assert!(plan.pages.len() > 1);
            // Every cluster once, in the page the plan says.
            let mut seen = vec![false; dag.clusters.len()];
            for (p, page) in plan.pages.iter().enumerate() {
                assert!(Plain(size).page_bytes(&page.tally) <= size);
                assert_eq!(page.tally.clusters, page.clusters.len());
                assert_eq!(page.tally.vertices, page.vertices.len());
                for &c in &page.clusters {
                    assert!(!std::mem::replace(&mut seen[c as usize], true));
                    assert_eq!(plan.cluster_page[c as usize], p as u32);
                    // The page's vertex list covers the cluster.
                    let cl = &dag.clusters[c as usize];
                    assert!(cl.indices.iter().all(|v| page.vertices.contains(v)));
                }
                for &d in &page.dependencies {
                    assert!((d as usize) < p);
                }
            }
            assert!(seen.iter().all(|&s| s));
            for (c, cl) in dag.clusters.iter().enumerate() {
                let p = plan.cluster_page[c];
                // A group's output shares a page; the parent group's page is this one or a
                // dependency.
                if cl.group != NONE {
                    assert_eq!(plan.group_page[cl.group as usize], p);
                }
                if cl.parent_group != NONE {
                    let pp = plan.group_page[cl.parent_group as usize];
                    assert!(pp == p || plan.pages[p as usize].dependencies.contains(&pp));
                }
            }
            // Coarse to fine.
            let levels: Vec<u32> = plan
                .cluster_order()
                .map(|c| dag.clusters[c as usize].level)
                .collect();
            assert!(levels.windows(2).all(|w| w[0] >= w[1]));
        }
        assert!(plan_pages(&dag, &Plain(2048)).is_err());
    }
}

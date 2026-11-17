//! Editor visualizations: what the demo draws over a model to show what farthing did to it, as
//! opposed to how the model looks (`pbr.wgsl`).
//!
//! - **Wireframe**: every triangle edge.
//! - **UV seams**: edges where triangles meet in position but not in texture coordinates, so a
//!   reduction's seams can be checked against the original's.
//!
//! - **Splats**: a surfel cloud as opaque discs or blended Gaussians (`splats.wgsl`).
//!
//! The overlays are line segments drawn as screen-space quads of a constant pixel width
//! (`lines.wgsl`), over surfaces that the viewer's depth bias pushes back a little.

use std::collections::HashMap;

/// An overlay over the model's surfaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlay {
    Wireframe = 0,
    Seams = 1,
}

/// Every distinct edge of `indices`' triangles, as segments (six floats each).
pub fn wire_segments(positions: &[f32], indices: &[u32]) -> Vec<f32> {
    let mut edges: Vec<u64> = indices
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
        .filter(|(a, b)| a != b)
        .map(|(a, b)| (u64::from(a.min(b)) << 32) | u64::from(a.max(b)))
        .collect();
    edges.sort_unstable();
    edges.dedup();
    let mut out = Vec::with_capacity(edges.len() * 6);
    for e in edges {
        for v in [(e >> 32) as usize, (e & 0xffff_ffff) as usize] {
            out.extend_from_slice(&positions[3 * v..3 * v + 3]);
        }
    }
    out
}

/// The UV seams of a mesh as segments: edges whose two triangles share the positions but not
/// the texture coordinates. Open borders aren't seams. None without uvs.
pub fn seam_segments(positions: &[f32], uvs: &[f32], indices: &[u32]) -> Vec<f32> {
    let n = positions.len() / 3;
    if uvs.len() != 2 * n {
        return Vec::new();
    }
    // Vertices split along a seam share a position: number the distinct positions.
    let mut ids: HashMap<[u32; 3], u32> = HashMap::new();
    let pid: Vec<u32> = positions
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            let next = ids.len() as u32;
            *ids.entry([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()])
                .or_insert(next)
        })
        .collect();
    let same = |a: usize, b: usize| {
        a == b
            || ((uvs[2 * a] - uvs[2 * b]).abs() < 1e-5
                && (uvs[2 * a + 1] - uvs[2 * b + 1]).abs() < 1e-5)
    };
    // Edge (lower position id first) -> the vertices of its first triangle, and whether a later
    // triangle disagreed on its uvs.
    let mut first: HashMap<u64, (usize, usize, bool)> = HashMap::new();
    let mut order = Vec::new();
    for t in indices.as_chunks::<3>().0 {
        for e in 0..3 {
            let (mut a, mut b) = (t[e] as usize, t[(e + 1) % 3] as usize);
            if pid[a] == pid[b] {
                continue;
            }
            if pid[a] > pid[b] {
                (a, b) = (b, a);
            }
            let key = (u64::from(pid[a]) << 32) | u64::from(pid[b]);
            match first.get_mut(&key) {
                None => {
                    first.insert(key, (a, b, false));
                    order.push(key);
                }
                Some(f) => {
                    if !same(f.0, a) || !same(f.1, b) {
                        f.2 = true;
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for key in order {
        let (a, b, seam) = first[&key];
        if seam {
            out.extend_from_slice(&positions[3 * a..3 * a + 3]);
            out.extend_from_slice(&positions[3 * b..3 * b + 3]);
        }
    }
    out
}

/// The wireframe's and the seams' pipelines (in [`Overlay`] order): instanced quads, one per
/// segment, blended over the surfaces and depth-tested against them without writing depth.
pub(crate) fn line_pipelines(
    device: &wgpu::Device,
    groups: &[&wgpu::BindGroupLayout],
    format: wgpu::TextureFormat,
    depth: wgpu::TextureFormat,
    samples: u32,
) -> [wgpu::RenderPipeline; 2] {
    let source = format!(
        "{}\n{}",
        include_str!("../common.wgsl"),
        include_str!("lines.wgsl")
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lines"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let groups: Vec<_> = groups.iter().map(|g| Some(*g)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &groups,
        immediate_size: 0,
    });
    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
    let buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: 24,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &attributes,
    })];
    ["wire", "seam"].map(|which| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(which),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(&format!("vs_{which}")),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    })
}

/// The floats per splat in a splat cloud: center xyz, covariance xx xy xz yy yz zz, linear
/// RGBA, normal xyz.
pub const SPLAT_FLOATS: usize = 16;

/// The splat pipelines, discs then Gaussians: instanced quads, one per splat. Discs are opaque
/// and write depth; Gaussians blend and don't.
pub(crate) fn splat_pipelines(
    device: &wgpu::Device,
    groups: &[&wgpu::BindGroupLayout],
    format: wgpu::TextureFormat,
    depth: wgpu::TextureFormat,
    samples: u32,
) -> [wgpu::RenderPipeline; 2] {
    let source = format!(
        "{}\n{}",
        include_str!("../common.wgsl"),
        include_str!("splats.wgsl")
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("splats"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let groups: Vec<_> = groups.iter().map(|g| Some(*g)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &groups,
        immediate_size: 0,
    });
    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x4, 4 => Float32x3];
    let buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: (SPLAT_FLOATS * 4) as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &attributes,
    })];
    [false, true].map(|gauss| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(if gauss { "gaussians" } else { "discs" }),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_splat"),
                compilation_options: Default::default(),
                buffers: &buffers,
            },
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: Some(!gauss),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(if gauss { "fs_gauss" } else { "fs_disc" }),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit square of two triangles, its uvs split along the diagonal.
    fn square(split: bool) -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mut positions: Vec<f32> = p.iter().flatten().copied().collect();
        let mut uvs = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mut indices = vec![0, 1, 2, 0, 2, 3];
        if split {
            // The second triangle's diagonal vertices, again, with other uvs.
            positions.extend_from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
            uvs.extend_from_slice(&[0.5, 0.5, 0.9, 0.9]);
            indices = vec![0, 1, 2, 4, 5, 3];
        }
        (positions, uvs, indices)
    }

    #[test]
    fn wireframe_has_each_edge_once() {
        let (p, _, i) = square(false);
        assert_eq!(wire_segments(&p, &i).len(), 5 * 6);
    }

    #[test]
    fn seams_are_where_uvs_split() {
        let (p, uv, i) = square(false);
        assert!(seam_segments(&p, &uv, &i).is_empty());
        let (p, uv, i) = square(true);
        let s = seam_segments(&p, &uv, &i);
        assert_eq!(s, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    }
}

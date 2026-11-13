//! Scenes: meshes, instances, skeletons and opaque material slots.

use crate::math::Aabb;
use crate::mesh::Mesh;
use glam::Mat4;
use serde::{Deserialize, Serialize};

/// An opaque handle to a material owned by the [`crate::host::RenderHost`].
/// farthing never inspects what it means.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub struct MaterialId(pub u32);

/// A material slot referenced by mesh triangles (`Mesh::material_ids` index into
/// `Scene::materials`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialSlot {
    pub name: String,
    pub id: MaterialId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Joint {
    pub name: String,
    pub parent: Option<usize>,
    /// Inverse of the joint's world-space bind matrix.
    pub inverse_bind: Mat4,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Skeleton {
    pub name: String,
    pub joints: Vec<Joint>,
}

impl Skeleton {
    pub fn bind_world(&self, joint: usize) -> Mat4 {
        self.joints[joint].inverse_bind.inverse()
    }
    pub fn children(&self, joint: usize) -> impl Iterator<Item = usize> + '_ {
        self.joints
            .iter()
            .enumerate()
            .filter(move |(_, j)| j.parent == Some(joint))
            .map(|(i, _)| i)
    }
    /// Joints ordered so every parent comes before its children.
    pub fn topological_order(&self) -> Vec<usize> {
        let mut order = Vec::with_capacity(self.joints.len());
        let mut placed = vec![false; self.joints.len()];
        while order.len() < self.joints.len() {
            let before = order.len();
            for (i, j) in self.joints.iter().enumerate() {
                if !placed[i] && j.parent.is_none_or(|p| placed[p]) {
                    placed[i] = true;
                    order.push(i);
                }
            }
            if order.len() == before {
                // Cycle: append the rest in index order rather than loop forever.
                order.extend((0..self.joints.len()).filter(|&i| !placed[i]));
                break;
            }
        }
        order
    }
}

/// One placement of a mesh in the world.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub name: String,
    pub mesh: usize,
    /// World transform.
    pub transform: Mat4,
    pub skeleton: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<MaterialSlot>,
    pub instances: Vec<Instance>,
    pub skeletons: Vec<Skeleton>,
}

/// Where a triangle of a flattened mesh came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TriRef {
    pub instance: u32,
    pub triangle: u32,
}

impl Scene {
    /// A scene holding one mesh at the origin, with one material slot per distinct id.
    pub fn from_mesh(mesh: Mesh) -> Scene {
        let slots = mesh.material_ids.iter().copied().max().map_or(1, |m| m + 1);
        Scene {
            materials: (0..slots)
                .map(|i| MaterialSlot {
                    name: format!("material_{i}"),
                    id: MaterialId(i),
                })
                .collect(),
            meshes: vec![mesh],
            instances: vec![Instance {
                name: "mesh".into(),
                mesh: 0,
                transform: Mat4::IDENTITY,
                skeleton: None,
            }],
            skeletons: Vec::new(),
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.instances
            .iter()
            .map(|i| self.meshes[i.mesh].triangle_count())
            .sum()
    }

    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::EMPTY;
        for inst in &self.instances {
            for p in &self.meshes[inst.mesh].positions {
                b.extend(inst.transform.transform_point3(*p));
            }
        }
        b
    }

    /// Material host handle for a slot index (falls back to the slot index itself).
    pub fn material_id(&self, slot: u32) -> MaterialId {
        self.materials
            .get(slot as usize)
            .map_or(MaterialId(slot), |m| m.id)
    }

    /// Bakes every instance into one world-space mesh. Material ids stay slot indices.
    /// Returns the mesh and, per output triangle, the instance and triangle it came from.
    pub fn flatten(&self) -> (Mesh, Vec<TriRef>) {
        let mut out = Mesh::default();
        let mut refs = Vec::new();
        for (ii, inst) in self.instances.iter().enumerate() {
            let mut m = self.meshes[inst.mesh].clone();
            m.transform(inst.transform);
            refs.extend((0..m.triangle_count() as u32).map(|t| TriRef {
                instance: ii as u32,
                triangle: t,
            }));
            out.append(&m, 0);
        }
        (out, refs)
    }

    /// World-space triangles in flatten order (for building a BVH).
    pub fn world_triangles(&self) -> (Vec<[glam::Vec3; 3]>, Vec<TriRef>) {
        let mut tris = Vec::with_capacity(self.triangle_count());
        let mut refs = Vec::with_capacity(self.triangle_count());
        for (ii, inst) in self.instances.iter().enumerate() {
            let m = &self.meshes[inst.mesh];
            let flip = inst.transform.determinant() < 0.0;
            for t in 0..m.triangle_count() {
                let [a, b, c] = m.triangle_positions(t);
                let (a, b, c) = (
                    inst.transform.transform_point3(a),
                    inst.transform.transform_point3(b),
                    inst.transform.transform_point3(c),
                );
                tris.push(if flip { [a, c, b] } else { [a, b, c] });
                refs.push(TriRef {
                    instance: ii as u32,
                    triangle: t as u32,
                });
            }
        }
        (tris, refs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::cube;

    #[test]
    fn flatten_two_instances() {
        let mut s = Scene::from_mesh(cube());
        s.instances.push(Instance {
            name: "b".into(),
            mesh: 0,
            transform: Mat4::from_translation(glam::Vec3::X * 3.0),
            skeleton: None,
        });
        let (m, refs) = s.flatten();
        assert_eq!(m.triangle_count(), 24);
        assert_eq!(
            refs[12],
            TriRef {
                instance: 1,
                triangle: 0
            }
        );
        assert!((s.bounds().extent().x - 4.0).abs() < 1e-5);
    }

    #[test]
    fn skeleton_order() {
        let j = |p| Joint {
            name: String::new(),
            parent: p,
            inverse_bind: Mat4::IDENTITY,
        };
        let s = Skeleton {
            name: "s".into(),
            joints: vec![j(Some(2)), j(None), j(Some(1))],
        };
        assert_eq!(s.topological_order(), vec![1, 2, 0]);
    }
}

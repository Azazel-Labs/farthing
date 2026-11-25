//! Models in any format farthing reads, from memory or by path. glTF goes to the glTF importer
//! as it is; with the `formats` (and `usd`) features, polysmelt reads the other formats into
//! glTF first.

use super::gltf_import::{GltfMaterial, load_gltf_with};
use crate::{Error, Result, Scene};
use polysmelt::assets::{Assets, Memory};
use std::borrow::Cow;
use std::path::Path;

/// [`MODEL_EXTENSIONS`] for each combination of features, from one list of each.
macro_rules! model_extensions {
    (gltf: [$($g:literal),*], formats: [$($f:literal),*], usd: [$($u:literal),*]) => {
        /// The extensions (lowercase, without the dot) of the model formats [`load_model`] reads
        /// in this build: glTF always, the others with the `formats` and `usd` features. When a
        /// set of files holds more than one model, the one listed first is the model.
        #[cfg(not(feature = "formats"))]
        pub const MODEL_EXTENSIONS: &[&str] = &[$($g),*];
        /// The extensions (lowercase, without the dot) of the model formats [`load_model`] reads
        /// in this build: glTF always, the others with the `formats` and `usd` features. When a
        /// set of files holds more than one model, the one listed first is the model.
        #[cfg(all(feature = "formats", not(feature = "usd")))]
        pub const MODEL_EXTENSIONS: &[&str] = &[$($g,)* $($f),*];
        /// The extensions (lowercase, without the dot) of the model formats [`load_model`] reads
        /// in this build: glTF always, the others with the `formats` and `usd` features. When a
        /// set of files holds more than one model, the one listed first is the model.
        #[cfg(feature = "usd")]
        pub const MODEL_EXTENSIONS: &[&str] = &[$($g,)* $($f,)* $($u),*];
    };
}

// Scenes before the meshes they place (`.irr` before `.irrmesh`). Files that only hold motion
// (`.vmd`, `.md5anim`, `.vta`, `.vrma`, OGRE `.skeleton`) are a model's companions, not models.
model_extensions! {
    gltf: ["glb", "gltf"],
    formats: [
        "fbx", "obj", "dae", "zae", "kmz", "skp", "3mf", "amf", "3ds", "ply", "stl", "off", "dxf",
        "x3d", "x3dv", "x3dz", "wrl", "wrz", "vrm", "vox", "x", "lwo", "ac", "ms3d", "b3d", "md2",
        "md3", "md5mesh", "iqm", "smd", "pmx", "pmd", "mesh", "flt", "ogex", "irr", "irrmesh",
        "xgl", "zgl"
    ],
    usd: ["usdz", "usd", "usda", "usdc"]
}

/// Loads a model from memory, in any format of [`MODEL_EXTENSIONS`]. `name` is the model's path
/// among `files`, which hold it and the files it refers to (glTF buffers and images, MTL
/// libraries, textures, USD layers) by relative, `/`-separated path. A reference is looked up
/// next to the model, then anywhere by file name. Works without a file system (WebAssembly).
///
/// Textures come in as PNG or JPEG; with the `formats` feature, WebP, TGA, BMP, GIF, TIFF, PSD
/// and DDS ones are converted to PNG.
pub fn load_model(
    name: &str,
    files: impl IntoIterator<Item = (String, Vec<u8>)>,
) -> Result<(Scene, Vec<GltfMaterial>)> {
    let mut memory = Memory::new();
    for (path, bytes) in files {
        memory.insert(&path, bytes);
    }
    let bytes = memory
        .get(name)
        .ok_or_else(|| Error::InvalidInput(format!("`{name}` is not among the files")))?;
    let ext = extension(Path::new(name));
    if ext == "glb" || ext == "gltf" {
        let dir = Path::new(name).parent().unwrap_or(Path::new(""));
        return load_gltf_with(bytes, |uri| {
            let found = memory.find(uri, dir)?;
            memory.read(&found).map(Cow::into_owned)
        });
    }
    if !MODEL_EXTENSIONS.contains(&ext.as_str()) {
        return Err(unsupported(&ext));
    }
    #[cfg(feature = "formats")]
    {
        convert(
            polysmelt::Loader::new()
                .keep(polysmelt::document::Keep::MESHES)
                .read_with(bytes.clone(), name, &memory),
        )
    }
    #[cfg(not(feature = "formats"))]
    unreachable!("without `formats`, every extension listed is glTF")
}

/// [`super::load`] for the formats polysmelt reads: the files a model refers to are found from
/// its folder.
#[cfg(feature = "formats")]
pub(super) fn load_file(path: &Path, ext: &str) -> Result<Scene> {
    if !MODEL_EXTENSIONS.contains(&ext) {
        return Err(unsupported(ext));
    }
    let dir = path.parent().unwrap_or(Path::new(""));
    let doc = polysmelt::Loader::new()
        .keep(polysmelt::document::Keep::MESHES)
        .assets(polysmelt::assets::Folder::new(dir))
        .open(path);
    convert(doc).map(|(scene, _)| scene)
}

/// A document polysmelt read, as a scene: plain glTF (compression and quantization decoded),
/// written as a GLB and imported.
#[cfg(feature = "formats")]
fn convert(
    doc: polysmelt::Result<polysmelt::document::Document>,
) -> Result<(Scene, Vec<GltfMaterial>)> {
    let invalid = |e: polysmelt::Error| Error::InvalidInput(e.to_string());
    let mut doc = doc.map_err(invalid)?;
    doc.decode().map_err(invalid)?;
    super::load_gltf_slice(&doc.to_glb().map_err(invalid)?)
}

fn unsupported(ext: &str) -> Error {
    Error::InvalidInput(format!(
        "unsupported model format `{ext}` (this build reads {})",
        MODEL_EXTENSIONS.join(", ")
    ))
}

pub(super) fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glam::Vec3;
    use crate::io::{ExportTextures, glb_bytes};
    use crate::{Mesh, Scene};

    fn files(list: &[(&str, &[u8])]) -> Vec<(String, Vec<u8>)> {
        list.iter()
            .map(|(n, b)| (n.to_string(), b.to_vec()))
            .collect()
    }

    #[test]
    fn reads_glb_by_name() {
        let m = Mesh::new(vec![Vec3::ZERO, Vec3::X, Vec3::Y], vec![0, 1, 2]);
        let glb = glb_bytes(&Scene::from_mesh(m), &ExportTextures::default()).unwrap();
        let (scene, _) = load_model("pack/a.GLB", files(&[("pack/a.GLB", &glb)])).unwrap();
        assert_eq!(scene.triangle_count(), 1);
    }

    #[test]
    fn polysmelt_reads_every_extension_listed() {
        for ext in MODEL_EXTENSIONS {
            assert!(polysmelt::Format::from_extension(ext).is_some(), "{ext}");
        }
    }

    #[test]
    fn rejects_what_it_cannot_read() {
        let e = load_model("a.xyz", files(&[("a.xyz", b"")])).unwrap_err();
        assert!(e.to_string().contains("glb, gltf"), "{e}");
        assert!(load_model("missing.glb", files(&[])).is_err());
    }

    #[cfg(feature = "formats")]
    #[test]
    fn reads_off_and_collada() {
        let off = b"OFF
4 2 0
0 0 0
1 0 0
1 1 0
0 1 0
3 0 1 2
3 0 2 3
";
        let (scene, _) = load_model("a.off", files(&[("a.off", off)])).unwrap();
        assert_eq!(scene.triangle_count(), 2);
        let dae = br##"<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_geometries><geometry id="g"><mesh>
    <source id="p"><float_array id="pa" count="9">0 0 0 1 0 0 0 1 0</float_array>
      <technique_common><accessor source="#pa" count="3" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
    <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="s"><node>
    <instance_geometry url="#g"/>
  </node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>"##;
        let (scene, _) = load_model("m/a.DAE", files(&[("m/a.DAE", dae)])).unwrap();
        assert_eq!(scene.triangle_count(), 1);
    }

    #[cfg(feature = "formats")]
    #[test]
    fn reads_obj_with_its_material_and_a_tga_texture() {
        let obj = b"mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nusemtl skin\nf 1/1 2/2 3/3 4/4\n";
        let mtl = b"newmtl skin\nKd 1 1 1\nmap_Kd tex/skin.tga\n";
        let mut tga = std::io::Cursor::new(Vec::new());
        image::RgbaImage::from_pixel(2, 2, image::Rgba([200, 100, 50, 255]))
            .write_to(&mut tga, image::ImageFormat::Tga)
            .unwrap();
        let (scene, materials) = load_model(
            "pack/m.obj",
            files(&[
                ("pack/m.obj", obj),
                ("pack/m.mtl", mtl),
                ("pack/tex/skin.tga", tga.get_ref()),
            ]),
        )
        .unwrap();
        assert_eq!(scene.triangle_count(), 2);
        let texture = materials[0].base_color_texture.as_ref().map(|t| t.0.width);
        assert_eq!(
            texture,
            Some(2),
            "the TGA arrives as the base color texture"
        );
    }

    #[cfg(feature = "formats")]
    #[test]
    fn reads_ascii_stl() {
        let stl = b"solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
        let (scene, _) = load_model("t.stl", files(&[("t.stl", stl)])).unwrap();
        assert_eq!(scene.triangle_count(), 1);
    }

    #[cfg(feature = "usd")]
    #[test]
    fn reads_usda() {
        let usda = br#"#usda 1.0
def Mesh "quad"
{
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
}
"#;
        let (scene, _) = load_model("q.usda", files(&[("q.usda", usda)])).unwrap();
        assert_eq!(scene.triangle_count(), 2);
    }
}

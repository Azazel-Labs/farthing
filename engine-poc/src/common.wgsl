// What every viewer pass shares: the view (group 0) and the drawn model (group 1).

struct Frame {
    view_proj: mat4x4f,
    // World to camera.
    view: mat4x4f,
    eye: vec4f,
    // Width, height and their reciprocals, in target pixels.
    viewport: vec4f,
    wire: vec4f,
    seam: vec4f,
    // The wireframe's and the seams' widths, in target pixels.
    widths: vec4f,
    // The focal lengths in target pixels (x, y).
    focal: vec4f,
}

struct Model {
    model: mat4x4f,
    // The inverse transpose; normal[3].w is the sign of the model's determinant.
    normal: mat4x4f,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<uniform> obj: Model;

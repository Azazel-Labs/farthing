//! Browser demo for farthing: drop a glTF (or a zip of one), tune LOD options and watch the
//! chain build and bake, all in WebAssembly. The pipeline itself is in [`pipeline`] and runs
//! natively too; this file only adapts it to JavaScript.

pub mod cook;
mod gpu;
pub mod modes;
pub mod pipeline;

// Only the `readme` example draws with the engine; the library just carries the dependency.
#[cfg(all(feature = "readme", not(target_arch = "wasm32")))]
use farthing_poc_engine as _;

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::modes::ViewRequest;
    use crate::pipeline::{Buffer, Demo, Event, LodOptions, scene_payload};
    use js_sys::{Array, Float32Array, Function, Uint8Array, Uint32Array};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
    }

    fn typed(b: &Buffer) -> JsValue {
        match b {
            Buffer::F32(v) => Float32Array::from(v.as_slice()).into(),
            Buffer::U32(v) => Uint32Array::from(v.as_slice()).into(),
            Buffer::U8(v) => Uint8Array::from(v.as_slice()).into(),
        }
    }

    /// `[json, ...typed arrays]`.
    fn json_and_buffers(json: String, buffers: &[Buffer]) -> Array {
        let out = Array::new();
        out.push(&json.into());
        for b in buffers {
            out.push(&typed(b));
        }
        out
    }

    fn js_err(e: String) -> JsError {
        JsError::new(&e)
    }

    /// A loaded asset and the results of its last run.
    #[wasm_bindgen]
    pub struct Session(Demo);

    #[wasm_bindgen]
    impl Session {
        /// `files`: an array of `[name, Uint8Array]` pairs (dropped files; zips are expanded).
        #[wasm_bindgen(constructor)]
        pub fn new(files: Array) -> Result<Session, JsError> {
            let mut v = Vec::new();
            for f in files.iter() {
                let pair: Array = f
                    .dyn_into()
                    .map_err(|_| JsError::new("expected [name, bytes] pairs"))?;
                let name = pair.get(0).as_string().unwrap_or_default();
                v.push((name, Uint8Array::new(&pair.get(1)).to_vec()));
            }
            Demo::load(v).map(Session).map_err(js_err)
        }

        /// Scene statistics as JSON.
        pub fn info(&self) -> String {
            serde_json::to_string(&self.0.info()).unwrap_or_default()
        }

        /// The original as the viewer takes it, with the materials as farthing reads them:
        /// `[json, ...typed arrays]` (see `scene_payload`).
        #[wasm_bindgen(js_name = originalScene)]
        pub fn original_scene(&self) -> Result<Array, JsError> {
            let (json, buffers) = self.0.level_scene(0).map_err(js_err)?;
            Ok(json_and_buffers(json.to_string(), &buffers))
        }

        /// A level of the last run as a self-contained GLB: compact (as the viewer loads it) or
        /// plain glTF (for download).
        #[wasm_bindgen(js_name = levelGlb)]
        pub fn level_glb(&self, level: usize, compact: bool) -> Result<Vec<u8>, JsError> {
            self.0.level_glb(level, compact).map_err(js_err)
        }

        /// A level of the last run as it would ship with texture compression: a zip of its
        /// compact GLB, mipple's KTX2 files and a manifest.
        #[wasm_bindgen(js_name = levelCooked)]
        pub fn level_cooked(&self, level: usize) -> Result<Vec<u8>, JsError> {
            self.0.level_cooked(level).map_err(js_err)
        }

        /// A splat level of the last splat run as a Gaussian splatting PLY.
        #[wasm_bindgen(js_name = levelPly)]
        pub fn level_ply(&self, level: usize) -> Result<Vec<u8>, JsError> {
            self.0.level_ply(level).map_err(js_err)
        }

        /// The cut of the last cluster or HLOD run for a camera (`ViewRequest` JSON):
        /// `[json, ...typed arrays]`.
        pub fn select(&self, view: &str) -> Result<Array, JsError> {
            let view: ViewRequest =
                serde_json::from_str(view).map_err(|e| JsError::new(&e.to_string()))?;
            let (json, buffers) = self.0.select(&view).map_err(js_err)?;
            Ok(json_and_buffers(json, &buffers))
        }

        /// Runs `options.mode`. `options` is JSON (see `LodOptions`); `on_event(kind, json,
        /// payload)` is called with `"stage"`, `"level"` (payload = `[scene json, ...typed
        /// arrays]`, or null for the original), `"texture"`
        /// (RGBA8 bytes), and mode data (`"clusters"`, `"hlod_node"`, `"hlod_tree"`,
        /// `"splats"`; payload = an array of typed arrays).
        pub fn run(&mut self, options: &str, on_event: &Function) -> Result<(), JsError> {
            let opts: LodOptions =
                serde_json::from_str(options).map_err(|e| JsError::new(&e.to_string()))?;
            let now = js_sys::Date::now;
            let call = |kind: &str, json: String, payload: JsValue| {
                let _ = on_event.call3(&JsValue::NULL, &kind.into(), &json.into(), &payload);
            };
            let mut emit = |e: Event| match e {
                Event::Stage(s) => call("stage", s.to_owned(), JsValue::NULL),
                Event::Level { info, scene } => call(
                    "level",
                    serde_json::to_string(info).unwrap_or_default(),
                    scene.map_or(JsValue::NULL, |(s, t)| {
                        let (json, buffers) = scene_payload(s, t);
                        json_and_buffers(json.to_string(), &buffers).into()
                    }),
                ),
                Event::Texture {
                    level,
                    channel,
                    width,
                    height,
                    rgba,
                } => call(
                    "texture",
                    serde_json::json!({
                        "level": level,
                        "channel": channel,
                        "width": width,
                        "height": height,
                    })
                    .to_string(),
                    Uint8Array::from(rgba).into(),
                ),
                Event::Data {
                    kind,
                    json,
                    buffers,
                } => {
                    let arr = Array::new();
                    for b in &buffers {
                        arr.push(&typed(b));
                    }
                    drop(buffers);
                    call(kind, json, arr.into());
                }
            };
            self.0.run(&opts, &now, &mut emit).map_err(js_err)
        }
    }
}

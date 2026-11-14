//! Browser demo for farthing: drop a glTF (or a zip of one), tune LOD options and watch the
//! chain build and bake, all in WebAssembly. The pipeline itself is in [`pipeline`] and runs
//! natively too; this file only adapts it to JavaScript.

pub mod modes;
pub mod pipeline;

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::modes::ViewRequest;
    use crate::pipeline::{Buffer, Demo, Event, LodOptions};
    use js_sys::{Array, Float32Array, Function, Uint32Array, Uint8Array};
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

        /// The original as GLB, with the materials as farthing reads them.
        #[wasm_bindgen(js_name = originalGlb)]
        pub fn original_glb(&self) -> Result<Vec<u8>, JsError> {
            self.0.original_glb().map_err(js_err)
        }

        /// A level of the last run as a self-contained GLB.
        #[wasm_bindgen(js_name = levelGlb)]
        pub fn level_glb(&self, level: usize) -> Result<Vec<u8>, JsError> {
            self.0.level_glb(level).map_err(js_err)
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
            let out = Array::new();
            out.push(&json.into());
            for b in &buffers {
                out.push(&typed(b));
            }
            Ok(out)
        }

        /// Runs `options.mode`. `options` is JSON (see `LodOptions`); `on_event(kind, json,
        /// payload)` is called with `"stage"`, `"level"` (payload = GLB bytes), `"texture"`
        /// (RGBA8 bytes), and mode data (`"clusters"`, `"hlod_node"`, `"hlod_tree"`,
        /// `"splats"`; payload = an array of typed arrays).
        pub fn run(&mut self, options: &str, on_event: &Function) -> Result<(), JsError> {
            let opts: LodOptions =
                serde_json::from_str(options).map_err(|e| JsError::new(&e.to_string()))?;
            let now = js_sys::Date::now;
            let call = |kind: &str, json: String, payload: JsValue| {
                let _ = on_event.call3(&JsValue::NULL, &kind.into(), &json.into(), &payload);
            };
            let bytes = |b: &[u8]| -> JsValue { Uint8Array::from(b).into() };
            let mut emit = |e: Event| match e {
                Event::Stage(s) => call("stage", s.to_owned(), JsValue::NULL),
                Event::Level { info, glb } => call(
                    "level",
                    serde_json::to_string(info).unwrap_or_default(),
                    bytes(glb),
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
                    bytes(rgba),
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

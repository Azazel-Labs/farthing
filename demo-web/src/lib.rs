//! Browser demo for farthing: drop a glTF (or a zip of one), tune LOD options and watch the
//! chain build and bake, all in WebAssembly. The pipeline itself is in [`pipeline`] and runs
//! natively too; this file only adapts it to JavaScript.

pub mod pipeline;

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::pipeline::{Demo, Event, LodOptions};
    use js_sys::{Array, Function, Uint8Array};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn start() {
        console_error_panic_hook::set_once();
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
            Demo::load(v).map(Session).map_err(|e| JsError::new(&e))
        }

        /// Scene statistics as JSON.
        pub fn info(&self) -> String {
            serde_json::to_string(&self.0.info()).unwrap_or_default()
        }

        /// The original as GLB, with the materials as farthing reads them.
        #[wasm_bindgen(js_name = originalGlb)]
        pub fn original_glb(&mut self) -> Result<Vec<u8>, JsError> {
            self.0.original_glb().map_err(|e| JsError::new(&e))
        }

        /// A level of the last run as a self-contained GLB.
        #[wasm_bindgen(js_name = levelGlb)]
        pub fn level_glb(&mut self, level: usize) -> Result<Vec<u8>, JsError> {
            self.0.level_glb(level).map_err(|e| JsError::new(&e))
        }

        /// Builds the chain. `options` is JSON (see `LodOptions`); `on_event(kind, json, bytes)`
        /// is called with `"stage"`, `"level"` (bytes = GLB) and `"texture"` (bytes = RGBA8).
        pub fn run(&mut self, options: &str, on_event: &Function) -> Result<(), JsError> {
            let opts: LodOptions =
                serde_json::from_str(options).map_err(|e| JsError::new(&e.to_string()))?;
            let now = js_sys::Date::now;
            let call = |kind: &str, json: String, bytes: &[u8]| {
                let _ = on_event.call3(
                    &JsValue::NULL,
                    &kind.into(),
                    &json.into(),
                    &Uint8Array::from(bytes).into(),
                );
            };
            let mut emit = |e: Event| {
                match e {
                Event::Stage(s) => call("stage", s.to_owned(), &[]),
                Event::Level { info, glb } => {
                    call("level", serde_json::to_string(info).unwrap_or_default(), glb)
                }
                Event::Texture {
                    level,
                    channel,
                    width,
                    height,
                    rgba,
                } => call(
                    "texture",
                    serde_json::json!({ "level": level, "channel": channel, "width": width, "height": height })
                        .to_string(),
                    rgba,
                ),
            }
            };
            self.0
                .run(&opts, &now, &mut emit)
                .map_err(|e| JsError::new(&e))
        }
    }
}

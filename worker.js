const params = new URLSearchParams(self.location.search)
const scriptName = params.get("script") || "./obamify.js"
const scriptUrl = new URL(scriptName, self.location.href)

try {
  const obamifyModule = await import(scriptUrl.href)
  const wasmUrl = new URL(scriptUrl.href.replace(/\.js$/, "_bg.wasm"))

  await obamifyModule.default({ module_or_path: wasmUrl.href })

  // Call the worker entry explicitly. Depending on the wasm-bindgen/Trunk
  // version, importing a binary bundle does not reliably invoke Rust's main
  // function inside a module worker.
  if (typeof obamifyModule.worker_entry !== "function") {
    throw new Error("worker_entry export is missing")
  }
  obamifyModule.worker_entry()
} catch (e) {
  console.error("worker failed to initialize:", e)
  throw e
}

const CACHE_PREFIX = "portraitify-pwa-"
const CACHE_NAME = `${CACHE_PREFIX}v2`

function versionedAssetGroup(url) {
  const filename = url.pathname.split("/").pop() || ""
  if (/^obamify-[^/]+\.js$/.test(filename)) return "module-js"
  if (/^obamify-[^/]+_bg\.wasm$/.test(filename)) return "module-wasm"
  if (filename === "worker.js" && url.searchParams.has("script")) return "worker"
  return null
}

self.addEventListener("install", (event) => {
  // Activate a new worker immediately. Requests remain network-first, so an
  // older cached index can never pin a stale JS/WASM pair after deployment.
  event.waitUntil(self.skipWaiting())
})

self.addEventListener("activate", (event) => {
  event.waitUntil(
    Promise.all([
      caches.keys().then((keys) =>
        Promise.all(
          keys
            .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_NAME)
            .map((key) => caches.delete(key)),
        ),
      ),
      self.clients.claim(),
    ]),
  )
})

self.addEventListener("fetch", (event) => {
  const request = event.request
  const url = new URL(request.url)

  if (request.method !== "GET" || url.origin !== self.location.origin) {
    return
  }

  let cacheCopy = null
  const networkResponse = fetch(request).then((response) => {
    if (response.ok && response.type === "basic") {
      cacheCopy = response.clone()
    }
    return response
  })

  // waitUntil must be registered while the fetch event is still active. Keep
  // cache failures separate from the response so they never break a good load.
  event.waitUntil(
    networkResponse
      .then(async () => {
        if (cacheCopy) {
          const cache = await caches.open(CACHE_NAME)
          const group = versionedAssetGroup(url)
          if (group) {
            const oldRequests = await cache.keys()
            await Promise.all(
              oldRequests
                .filter((oldRequest) => {
                  const oldUrl = new URL(oldRequest.url)
                  return oldRequest.url !== request.url && versionedAssetGroup(oldUrl) === group
                })
                .map((oldRequest) => cache.delete(oldRequest)),
            )
          }
          await cache.put(request, cacheCopy)
        }
      })
      .catch(() => undefined),
  )

  event.respondWith(
    networkResponse.catch(async () => {
      const cached = await caches.match(request)
      if (cached) {
        return cached
      }

      if (request.mode === "navigate") {
        const scope = new URL("./", self.registration.scope)
        const fallback = await caches.match(scope)
        if (fallback) {
          return fallback
        }
      }

      return new Response("Portraitify is offline and this asset is not cached yet.", {
        status: 503,
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      })
    }),
  )
})

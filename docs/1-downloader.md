# Stage 1 — WMS Downloader

`wms_download.py` is the Stage 1 acquisition script of the Snitcher pipeline.
It downloads a layer of the Italian Cadastre WMS over a user-specified bbox
and writes a single georeferenced GeoTIFF (or a plain PNG).

## What it does

1. Takes a bbox in EPSG:4326 (lat/lng) and reprojects it to EPSG:25832 (meters).
2. Computes the mosaic pixel size from a user-supplied resolution in pixels per meter.
3. Splits the mosaic into ≤ 2048×2048 tiles, fetches them concurrently from the WMS,
   and pastes them into a single in-memory image.
4. Saves the result as a GeoTIFF with embedded CRS, geotransform, and provenance
   metadata — or as a plain PNG.

## Usage

```bash
pip install requests Pillow pyproj rasterio numpy

python wms_download.py \
  --bbox 45.468160,9.229277,45.471094,9.233311 \
  --layer fabbricati \
  --ppm 10 \
  --workers 8 \
  --format geotiff \
  --output milan_test.tif
```

| Flag        | Meaning                                                       | Default        |
|-------------|---------------------------------------------------------------|----------------|
| `--bbox`    | Area to download, as `lat_min,lng_min,lat_max,lng_max`        | required       |
| `--layer`   | WMS layer name (`fabbricati`, `CP.CadastralParcel`, …)        | `fabbricati`   |
| `--ppm`     | Output resolution in pixels per meter                         | required       |
| `--workers` | Concurrent tile downloads                                     | `4`            |
| `--format`  | `geotiff` or `png`                                            | `geotiff`      |
| `--output`  | Output path                                                   | `cadastre.tif` |

## Design decisions

### Working CRS: EPSG:25832, not EPSG:4326

EPSG:25832 (ETRS89 / UTM zone 32N) is metric. With a metric CRS, "pixels per
meter" maps directly to pixel dimensions, and tile bbox arithmetic is trivial.
EPSG:25832 is one of the CRSs the WMS explicitly supports and covers most of
Italy. Working in EPSG:4326 would mean degrees, which are not constant in
meters and would make the resolution definition meaningless.

### WMS 1.3.0 axis order

WMS 1.3.0 + EPSG:4326 uses lat,lon order (the classic gotcha). WMS 1.3.0 +
projected CRSs like EPSG:25832 uses easting,northing — which is the natural
`minx,miny,maxx,maxy`. The script uses natural order, which is correct here.

### Tiling and float-drift avoidance

The 2048×2048 cap is the per-request hard limit from the WMS docs. The mosaic
is split into a grid, with the last column/row truncated to whatever pixels
remain rather than padded.

Each tile's geographic bbox is computed from its pixel offset in the mosaic
using a single meters-per-pixel ratio (`width_m / total_w_px`). This
guarantees that adjacent tiles share exact edge coordinates — no seams from
float drift, no gaps, no overlap.

### Content-Type check

On error (rate limiting, bad parameters, etc.) the WMS returns a
`ServiceExceptionReport` XML with HTTP 200. Without an explicit
`Content-Type` check, PIL would silently choke on the XML. The script
raises a clear `RuntimeError` instead.

### GeoTIFF over PNG (default)

A GeoTIFF carries everything Stage 2 will need:

- **CRS** (EPSG:25832) and **affine transform** — standard GeoKeys, picked up
  automatically by QGIS, GDAL, rasterio.
- **Provenance tags** (custom GeoTIFF metadata, inspect with `gdalinfo`):
  source WMS URL and version, layer name, UTC acquisition timestamp, bbox in
  both CRSs, pixels-per-meter, license (CC BY 4.0), attribution.

PNG is kept as an option for quick visual inspection or for sharing outside
GIS tools.

### 4 bands (RGBA), not 3

`fabbricati` polygons sit on a transparent background. The alpha channel
distinguishes "white pixel = no building drawn" from "transparent = no data
here", which is useful for Stage 2 vectorization. Switching to RGB later
is a one-line change.

### LZW compression + internal tiling

LZW is lossless and roughly 3× smaller than uncompressed for sparse line art.
Internal 256×256 tiling (enabled when the mosaic is ≥ 256 px on a side) lets
Stage 2 read sub-windows efficiently without decoding the full raster.

### Concurrency: threads, not asyncio or processes

Tile fetching is I/O-bound — the bottleneck is waiting on the WMS server,
not CPU. Python threads release the GIL during socket I/O, so a
`ThreadPoolExecutor` delivers full concurrency with minimal code. `asyncio`
would mean swapping `requests` for `httpx`/`aiohttp` and rewriting the loop
— not worth the complexity here.

Three small but deliberate choices in the concurrency code:

- **Workers do only the HTTP call.** PIL paste operations stay on the main
  thread, consuming futures via `as_completed`. PIL isn't thread-safe for
  writes to the same image, and centralizing pastes on one thread sidesteps
  the issue without locking.
- **One shared `requests.Session`.** Safe for concurrent GETs and lets the
  underlying connection pool be reused across threads — fewer TCP handshakes,
  less TLS overhead.
- **`workers = min(workers, total_tiles)`.** No point spawning 16 threads
  for a 4-tile job.

## Known caveats

- **WMS rate limiting.** The docs mention "a maximum limit of simultaneous
  consultation requests" without giving a number. 4–8 workers is comfortable;
  pushing to 16+ on a large area may start triggering rejections, which
  surface as `RuntimeError` from the Content-Type check. If this becomes a
  problem, a small retry-with-backoff wrapper around `fetch_tile` is the
  right fix.
- **Actual resolution slightly exceeds requested ppm.** `total_w_px =
  ceil(width_m * ppm)` rounds up to an integer pixel count, so effective
  resolution is a hair higher than requested. The ppm value is honored as a
  lower bound and the tile bbox math stays self-consistent.
- **EPSG:25832 is UTM zone 32N.** It covers most of Italy well but distorts
  in the east (Puglia, eastern Sicily fall into zone 33N). For those areas,
  switching the working CRS to EPSG:25833 (or another supported metric CRS)
  is one constant change at the top of the script.

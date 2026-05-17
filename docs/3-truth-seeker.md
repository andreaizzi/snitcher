# Stage 3 — Truth Seeker (`3-truth-seeker.py`)

Downloads OpenBuildingMap (OBM) building footprints for a target area and saves them as a GeoPackage (or GeoJSON), ready for Stage 4 reconciliation against the Stage 2 cadastral polygons.

## Install

```bash
pip install requests rasterio geopandas
```

`geopandas` brings in `pandas`, `shapely`, `pyproj`, and `fiona` (GDAL bindings).

## Usage

```bash
python 3-truth-seeker.py (--bbox <lat_min,lng_min,lat_max,lng_max> | --from-geotiff <path>) [options]
```

### Arguments

| Flag             | Required | Default                  | Description                                              |
| ---------------- | -------- | ------------------------ | -------------------------------------------------------- |
| `--bbox`         | one of   | —                        | EPSG:4326 bbox as `lat_min,lng_min,lat_max,lng_max`      |
| `--from-geotiff` | one of   | —                        | Stage-1 GeoTIFF; bbox read from its `BBOX_EPSG_4326` tag |
| `--output`       | no       | `buildings_truth.gpkg`   | Output path                                              |
| `--format`       | no       | inferred from `--output` | `gpkg` or `geojson`                                      |
| `--cache-dir`    | no       | `~/.cache/snitcher/obm`  | On-disk cache for downloaded OBM tiles                   |

### Example

```bash
python 3-truth-seeker.py \
  --from-geotiff cadastre.tif \
  --output milan_truth.gpkg
```

## How it works

### 1. Resolve the bbox

If `--bbox` is given, it's parsed straight. If `--from-geotiff` is given, the script opens the GeoTIFF and reads its `BBOX_EPSG_4326` tag — the exact, untouched user input from Stage 1. If that tag is missing, it falls back to reprojecting `ds.bounds` to EPSG:4326.

### 2. Map bbox → OBM quadkey tiles

OBM is partitioned as **Bing-style quadkey tiles at zoom 6**. Eight of them cover all of Italy:

```
120221, 120230, 120223, 120232, 120233, 122001, 122010, 122011
```

For each bbox corner the script computes `(tile_x, tile_y)` via the standard Web Mercator formula, then encodes to a quadkey string. Every tile in the rectangular range is yielded; for typical Snitcher inputs (a town, a neighbourhood) this is exactly one tile.

### 3. Cache + download

For each quadkey:

- if `{cache_dir}/building.{qk}.gpkg` already exists, use it;
- otherwise download `building.{qk}.gpkg.bz2` from GFZ Data Services, decoding the bz2 stream chunk-by-chunk into a `.gpkg.part` file, then atomic-rename to `.gpkg` on completion.

The `.part` staging guarantees that a Ctrl-C or network drop never leaves a half-written `.gpkg` that the next run would mistake for a valid cache hit.

### 4. Spatial filter

For each cached tile, `geopandas.read_file(path, bbox=...)` runs the bbox query. Under the hood this hits GDAL's spatial filter, which uses the **R-tree index built into every GeoPackage**. The 1.9 GB file is never fully read — only features whose envelope intersects the bbox come back.

### 5. Merge and write

Per-tile results are concatenated. Each feature gets an extra `obm_tile` column carrying its source quadkey, so when a bbox straddles a tile boundary you can always tell which tile a row came from. Output is written as a single layer in EPSG:4326.

## Output

A single file in EPSG:4326, one feature per building. Geometry is the OBM footprint (Polygon or MultiPolygon). Attributes propagated from OBM include:

- `source` — `OSM`, `Google`, or `Microsoft`
- `height` — height code (`H:N`, `HBET:a-b`, `HHT:m`, etc.)
- `occupancy` — GEM occupancy code (`RES`, `COM`, `IND`, ...)
- `obm_tile` — added by this script: quadkey of the source tile

## Design choices

### Why download whole tiles instead of just the bbox?

GFZ Data Services only publishes whole tiles as `building.{qk}.gpkg.bz2`. There is no HTTP query API, and Range requests don't help: GeoPackage is a SQLite file, and random access into its B-tree is incompatible with streaming bz2 decompression.

Google Earth Engine *does* expose OBM with server-side `filterBounds()`, but it requires OAuth, an EE project, and an export–poll–download workflow. Too much overhead for an offline pipeline stage.

The compromise: pay the bandwidth once per tile (~700 MB compressed → ~1.9 GB on disk), then reuse the on-disk R-tree forever. After the first run, any bbox in that tile is an essentially free R-tree query. For Italy that's a one-time ~6 GB cache to cover the whole country, and most projects will only touch one or two tiles.

### Why GeoPackage as the default, not GeoJSON?

Stage 4 is a spatial join, and GeoPackage:

- ships with a built-in R-tree, so Stage 4 indexing is free;
- preserves attribute types (height numeric, occupancy string) instead of stringifying everything;
- is OBM's native format — no reprojection or serialisation introduces drift.

GeoJSON earns its keep only when you need a text-diffable file or an OSM-friendly drop. Stage 4 has no such requirement, so GPKG is the default; GeoJSON stays an opt-in via `--format geojson`.

### Why EPSG:4326 in the output, not the metric working CRS (25832)?

Three reasons:

- OBM ships in EPSG:4326; reprojecting at this stage just throws away precision.
- Stage 4 will reproject to a metric CRS *anyway* for IoU and Hausdorff metrics — that decision belongs to Stage 4, not here.
- EPSG:4326 is portable; the truth layer opens cleanly in QGIS or any downstream tool with no surprises.

### Why quadkey math by hand instead of a library?

`mercantile` and similar libraries would add a dependency for ~20 lines of stable, well-documented code (Bing Maps Tile System). The math is also self-validating: if the script ever returned a tile outside the published 8-tile Italy set for an Italian bbox, something would obviously be wrong.

### Why stamp `obm_tile` on every feature?

When a bbox crosses a tile boundary, OBM may contain near-duplicate buildings along the seam (OSM coverage on one side, Google on the other). Stage 4 reconciliation will collapse these by IoU, but having the source quadkey on each row makes debugging and provenance trivial.

## Notes & limits

- First run on a new tile downloads ~700 MB. Subsequent runs use the cache; no network needed.
- The decompressed `.gpkg` is ~1.9 GB per tile on disk. Plan accordingly.
- License is **ODbL v1.0**. Downstream use must preserve share-alike and attribution to OBM and its upstream sources (OSM, Google Open Buildings, Microsoft Global ML Building Footprints).
- A bbox entirely outside OBM coverage results in an empty output and a warning — no crash.
- When a bbox spans multiple tiles, expect occasional near-duplicate buildings along the seam; Stage 4 will handle them.

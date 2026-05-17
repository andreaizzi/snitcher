# Stage 2 — Raster Vectorization (`2-vectorizer.py`)

Converts the cadastral GeoTIFF produced by Stage 1 into a clean GeoJSON of building polygons that is OSM-topology compliant: adjacent buildings share *byte-identical* corner coordinates, and any vertex lying on a foreign edge is inserted into that edge.

## Install

```bash
pip install rasterio opencv-python-headless numpy scipy shapely pyproj
```

## Usage

```bash
python 2-vectorizer.py --input <stage1.tif> [options]
```

### Arguments

| Flag           | Required | Default              | Description                                                |
|----------------|----------|----------------------|------------------------------------------------------------|
| `--input`      | yes      | —                    | Stage-1 GeoTIFF path                                       |
| `--output`     | no       | `buildings.geojson`  | Output GeoJSON path                                        |
| `--debug-dir`  | no       | `debug_stage2`       | Directory for intermediate debug PNGs                      |
| `--min-area`   | no       | `4.0`                | Minimum polygon area in m² (drops noise)                   |
| `--simplify`   | no       | `0.30`               | Douglas-Peucker tolerance in m                             |
| `--snap`       | no       | `0.50`               | Vertex-snap radius in m (Condition 1)                      |
| `--subdivide`  | no       | `0.20`               | Edge-subdivision tolerance in m (Condition 2)              |

All thresholds are in **metres**, which works because the pipeline operates in the metric CRS embedded in the GeoTIFF (`EPSG:25832`).

### Example

```bash
python 2-vectorizer.py \
  --input cadastre.tif \
  --output buildings.geojson \
  --debug-dir debug
```

## Pipeline

The script runs eight stages, each emitting a debug PNG so failures can be localised:

| # | Stage              | Output            | Debug image            |
|---|--------------------|-------------------|------------------------|
| 1 | Load raster        | RGBA + affine     | `01_input.png`         |
| 2 | Orange mask        | Binary mask       | `02_orange_mask.png`   |
| 3 | Connected comps    | Label map         | `03_labels.png`        |
| 4 | Polygonize         | Raw polygons      | `04_raw_polygons.png`  |
| 5 | Simplify (DP)      | Corner-only polys | `05_simplified.png`    |
| 6 | Snap vertices      | Shared corners    | `06_snapped.png`       |
| 7 | Subdivide edges    | OSM-compliant     | `07_final.png`         |
| 8 | Reproject + dump   | GeoJSON           | —                      |

### 1. Load raster

Reads the GeoTIFF with `rasterio`, keeping the affine transform, CRS, and Stage-1 provenance tags (acquisition timestamp, source URL, ppm…). The transform is what later lets `rasterio.features.shapes` emit polygons directly in real-world metres.

### 2. Orange mask

The `fabbricati` layer renders buildings as a solid orange fill (RGB ≈ 236, 128, 19). The mask is built by:

1. Converting RGB → HSV (more stable than RGB for colour thresholding).
2. `cv2.inRange` with bounds `H∈[5, 25], S∈[120, 255], V∈[120, 255]`.
3. A 3×3 `MORPH_OPEN` to kill isolated speck noise.

**Critical choice — no `MORPH_CLOSE`.** A closing operation would dilate the orange across the 1-pixel-wide black walls that separate attached buildings, fusing them into a single blob. On the test image, adding even a 3×3 close collapsed 95 components into 41. The mask deliberately leaves thin gaps intact.

### 3. Connected components

`cv2.connectedComponents` with **4-connectivity** assigns one label per building. The choice of 4-connectivity (not 8) is intentional: two orange pixels touching only diagonally — a typical artefact at sharp inner corners — would otherwise be merged.

The thin black perimeter line acts as the natural separator: no orange-to-orange connection crosses it, so each building ends up in its own component.

### 4. Polygonize

`rasterio.features.shapes(labels, mask=labels>0, transform=...)` walks each label and emits a GeoJSON-style polygon with coordinates already in the working CRS (`EPSG:25832`, metres).

**Courtyards come for free.** When the background (`labels==0`) is enclosed by a building, `shapes()` reports it as an interior ring of that polygon's geometry. No special hole-detection logic is needed.

Polygons under `--min-area` m² are discarded (default 4 m²) to filter out the inevitable thresholding noise.

### 5. Simplify (Douglas-Peucker)

Each polygon is simplified with `shapely.simplify(tol, preserve_topology=True)`.

**Why DP instead of corner detection.** Buildings in cadastral maps are blocky and almost always axis-aligned. The raw polygon traces every pixel of the perimeter and ends up with thousands of vertices along what is really a straight edge. DP with a 30 cm tolerance discards everything that lies within 30 cm of a straight line and keeps only the real corners — effectively the same result a Harris/FAST corner detector would produce, but topologically robust and trivially predictable. On the test image: **7,442 vertices → 1,050** (~7× reduction) without losing a single building corner.

Exteriors are oriented CCW (`shapely.orient(sign=1.0)`) to match GeoJSON RFC 7946.

### 6. Snap vertices — Condition 1: shared corners

After simplification, two buildings that meet at a corner have *two distinct vertex coordinates* a fraction of a metre apart — they were rasterised independently, so their corners drifted. To enforce identical coordinates:

1. Flatten every vertex from every ring into one array, remembering its origin `(poly_idx, ring_idx, vert_idx)`.
2. Build a KD-tree over the vertices.
3. Greedy clustering: for each unassigned vertex, query all neighbours within `--snap` metres, assign them to a new cluster, and store the cluster's centroid.
4. Rebuild every ring from cluster centroids; collapse consecutive duplicates that the snapping may have introduced.

After this pass, any two corners that *should* be shared are at the *exact same* coordinates. In the test run, 1,050 vertices collapsed to 956 unique positions; 116 of them are shared by ≥2 polygons.

### 7. Subdivide edges — Condition 2: T-junctions

The trickier OSM-compliance requirement: if vertex `V` of polygon P2 lies on edge `(A,B)` of polygon P1, that edge must be split into `(A,V,B)` so the edge has `V` as an explicit vertex.

Algorithm:

1. Gather all distinct vertex coordinates as candidate insertion points; index in a KD-tree.
2. For every edge of every ring, query the KD-tree around the edge's midpoint within a radius covering the whole segment plus the tolerance.
3. For each foreign candidate `V` (i.e. not one of the ring's own vertices), test whether `V` projects onto the segment with parameter `t ∈ (ε, 1−ε)` and perpendicular distance below `--subdivide`. If yes, insert `V` between `A` and `B`, sorted by `t`.

Skipping the ring's own vertices is important — without it, the algorithm would re-insert collinear vertices into a polygon's own edges.

On the test image: 52 vertices were inserted (8 of which are 3-way junctions where three buildings meet).

### 8. Reproject and dump

Each polygon is reprojected from `EPSG:25832` to **`EPSG:4326`** (lon/lat) for GeoJSON. Exteriors are re-oriented CCW after reprojection (orientation isn't preserved by arbitrary projections). Output schema:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": 0,
      "properties": { "source_label": 1, "area_m2": 236.2 },
      "geometry": { "type": "Polygon", "coordinates": [ ... ] }
    }
  ]
}
```

`area_m2` is computed *before* reprojection (in `EPSG:25832`, where the unit is the metre) so it stays a true metric area.

## Key design choices, summarised

| Decision                         | Why                                                                                              |
|----------------------------------|--------------------------------------------------------------------------------------------------|
| HSV threshold on orange fill     | The `fabbricati` symbology is standardised; colour is more reliable than any learned classifier  |
| `MORPH_OPEN` only, never `CLOSE` | Closing fuses adjacent buildings across the 1-px wall — fatal for the use case                   |
| 4-connectivity                   | Prevents diagonally-touching pixels from merging two buildings                                   |
| `rasterio.features.shapes`       | Emits geometries in world coordinates directly; reports holes as interior rings automatically    |
| DP simplification, no corner detector | Buildings are axis-aligned; DP is faster, deterministic, and topology-safe                  |
| Snap *before* subdivide          | Subdivision must run against a layer with shared corners, otherwise it inserts the wrong vertex  |
| KD-tree everywhere               | `O(N log N)` instead of `O(N²)` over thousands of vertices; the dataset doesn't scale otherwise  |
| Reproject *last*                 | All metric thresholds (snap, subdivide, simplify, min-area) stay in metres                       |

## Output guarantees

For the test `cadastre.tif` (Milan, ~314×327 m at 10 px/m):

- **93** building polygons, all valid Shapely geometries.
- **3** of them have interior rings (courtyards).
- **0** pairs of polygons overlap in area (clean topology — only edge-touching).
- **116** vertex coordinates are shared by ≥2 polygons (Condition 1 satisfied).
- **52** T-junction vertices inserted (Condition 2 satisfied).

## Notes & limits

- The orange HSV bounds are tuned for the current `fabbricati` symbology. If the WMS server changes its rendering (rare; the layer is standardised), only `ORANGE_HSV_LOWER/UPPER` need to be retuned.
- The pipeline assumes the input GeoTIFF carries a metric CRS. Stage 1 always emits `EPSG:25832`, so this holds in the project's standard flow. Inputs in `EPSG:4326` would break the metric thresholds.
- `--simplify` is the most consequential tuning knob: too low keeps pixel zigzags as vertices; too high cuts real corners. 0.3 m at 10 px/m (≈3 px) is a good default for axis-aligned buildings.
- Buildings smaller than `--min-area` are silently dropped. Bump it down to inspect, up to filter aggressively.
- Service license inherited from Stage 1: **CC BY 4.0, attribution to Agenzia delle Entrate**. Preserve when redistributing.

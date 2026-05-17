# Stage 4 — Snitcher (`4-snitcher.py`)

Compares the cadastral building polygons from Stage 2 against the OpenBuildingMap (OBM) footprints from Stage 3 and classifies each polygon into one of four buckets — `matched`, `partial_match`, `cadastre_only`, `satellite_only` — producing a single file that's easy to drop into QGIS for visual inspection.

This is the stage that gives the project its name: it tells on buildings that exist in reality but not in the cadastre (and the reverse).

## Install

```bash
pip install geopandas shapely pyproj fiona pandas
```

`geopandas` pulls in `shapely`, `pyproj`, and `fiona` (GDAL bindings) transitively.

## Usage

```bash
python 4-snitcher.py --cadastre <stage2.geojson> --truth <stage3.gpkg> [options]
```

### Arguments

| Flag              | Required | Default              | Description                                                         |
| ----------------- | -------- | -------------------- | ------------------------------------------------------------------- |
| `--cadastre`      | yes      | —                    | Stage-2 GeoJSON of cadastral building polygons                      |
| `--truth`         | yes      | —                    | Stage-3 GeoPackage (or GeoJSON) of OBM building footprints          |
| `--output`        | no       | `discrepancies.gpkg` | Output file path                                                    |
| `--format`        | no       | inferred from path   | `gpkg` or `geojson`                                                 |
| `--metric-crs`    | no       | `EPSG:25832`         | CRS used for the geometry math (must be metric)                     |
| `--out-crs`       | no       | `EPSG:4326`          | CRS written to the output file                                      |
| `--iou-match`     | no       | `0.50`               | Cluster IoU at/above which a match is "matched"                     |
| `--iou-partial`   | no       | `0.10`               | Cluster IoU at/above which a match is "partial_match"               |
| `--overlap-ratio` | no       | `0.10`               | Min `intersection / min(area)` to draw an edge between two polygons |
| `--min-area`      | no       | `1.0`                | Drop input polygons smaller than this many m² (noise filter)        |

All tolerances are in **metres** or unit-less ratios.

### Example

```bash
python 4-snitcher.py \
  --cadastre buildings.geojson \
  --truth buildings_truth.gpkg \
  --output discrepancies.gpkg
```

## Pipeline

The script runs five stages.

| #   | Stage             | What it produces                                                    |
| --- | ----------------- | ------------------------------------------------------------------- |
| 1   | Load + reproject  | Two `GeoDataFrame`s in the metric CRS                               |
| 2   | Build match graph | Bipartite edges between cadastre and OBM polygons                   |
| 3   | Find clusters     | Connected components = groups of polygons "about the same building" |
| 4   | Classify clusters | Per-cluster IoU → `matched` / `partial_match` / dissolved           |
| 5   | Assemble + write  | Multi-layer GeoPackage (or merged GeoJSON) in EPSG:4326             |

### 1. Load + reproject

Both inputs are read with `geopandas.read_file`. The CRS of each is inspected — if either input lacks a CRS the script refuses to guess and exits, since a wrong assumption here silently corrupts everything downstream.

Both layers are reprojected to the same metric CRS (default `EPSG:25832`) so every threshold the user passes is in honest metres. Geographic CRSs like EPSG:4326 measure in degrees, which is meaningless for area math and varies wildly with latitude — reprojecting upfront avoids the entire class of "did I mean kilometres or millionths of a degree?" bugs.

Two cleanup steps run in this stage:

- **Invalid geometries are fixed with `buffer(0)`.** OBM occasionally ships polygons with self-intersections (Microsoft's ML-generated footprints especially) that would blow up `intersection()` calls downstream. `buffer(0)` is Shapely's idiomatic "make this valid" trick.
- **Polygons under `--min-area` (1 m² default) are dropped.** These are almost always rasterisation noise or detection artefacts and they pollute the match graph with spurious tiny overlaps.

### 2. Build match graph

This is the heart of the script. The output is a list of `(cad_idx, obm_idx)` edges — pairs of polygons "talking about the same physical building".

Two polygons get an edge if:

```
intersection_area / min(area_a, area_b) >= overlap_ratio
```

The `min(area)` denominator (instead of `union(area)` or just `area_a`) is what makes the matcher robust to the asymmetric case where a small cadastral unit sits mostly inside a large OBM roof — the small polygon's area dominates the ratio, so even a generous tolerance still produces an edge.

Performance: rather than the O(N×M) brute-force check, the script queries `obm_gdf.sindex` (a STRtree wrapped by GeoPandas) with `predicate="intersects"`, which prunes the candidate set down to a handful per cadastral polygon. The intersection-area computation only happens on real intersections.

### 3. Find clusters

This is where the N-to-M matching comes from. The terraced-houses problem — 4 cadastral units → 1 OBM roof, or 1 cadastral unit → 3 OBM roof patches — is the central reason this script is structured the way it is.

The solution is structural rather than special-cased. Treat the cadastre and OBM polygon sets together as nodes in a single graph (cadastre at indices `[0, n_cad)`, OBM at `[n_cad, n_cad+n_obm)`), with the edges from Stage 2. Then **connected components of this graph are exactly the matching clusters** we want — by definition, a connected component contains every polygon that overlaps any polygon in the cluster, transitively.

A 4-bedroom terraced row becomes one component containing 4 cadastre nodes and 1 OBM node. A complex with 3 cadastral units and 2 OBM patches becomes one component of 5 nodes. A standalone match becomes a 1+1 component. Isolated polygons (no edges) become singleton components.

Implementation note: connected components are computed with a tiny in-script **Union-Find** with path compression and union-by-rank. This is ~30 lines of code and removes the need to pull in `networkx` (~7 MB of dependencies) for one function call. Path compression keeps `find()` effectively O(α(N)) — constant for any realistic N.

The clusters are sorted into three groups:

- **Mixed clusters** — at least one polygon from each side. These will be classified by IoU.
- **Lonely cadastre polygons** — components made entirely of cadastre nodes (nothing on the OBM side touches them). These go straight to `cadastre_only`.
- **Lonely OBM polygons** — symmetric. These go straight to `satellite_only`.

### 4. Classify mixed clusters

For each mixed cluster, the script computes aggregate metrics on the **unions** of each side:

- `cad_union = unary_union(cluster.cad_polygons)`
- `obm_union = unary_union(cluster.obm_polygons)`
- `iou = intersection_area(cad_union, obm_union) / union_area(cad_union, obm_union)`

This is the crucial design choice that makes N-to-M work correctly. If you instead matched each cadastral polygon individually against its overlapping OBM polygons, three terraced houses would each get a low IoU score against the same roof (each one covers only ~⅓ of it) and the script would conclude they're all bad matches. The union-vs-union approach asks the right question: "do these polygons, collectively, describe the same physical footprint?"

Each cluster is classified by its aggregate IoU:

| IoU                       | Verdict         |
| ------------------------- | --------------- |
| `>= iou_match` (≥ 0.50)   | `matched`       |
| `>= iou_partial` (≥ 0.10) | `partial_match` |
| `< iou_partial`           | **dissolved**   |

A "dissolved" cluster is one where the polygons overlap enough to be linked by the edge threshold but not enough to credibly be the same building — typically a tiny corner of a cadastral building grazing a neighbouring OBM building. The cluster's polygons are not classified together: each cadastre polygon falls back to `cadastre_only` and each OBM polygon falls back to `satellite_only`. This is the right behaviour: if the union IoU is 3 %, there is no honest sense in which the cluster as a whole "matches", but each individual polygon still represents a real claim about a building somewhere.

Beyond IoU, every cluster also records:

- `cluster_centroid_dist_m` — distance between the cluster's centroids; small for well-aligned clusters, large when the cadastre is drawn in a different position than the actual roof.
- `cluster_hausdorff_m` — worst-case point-to-set distance; large when shape disagrees even if centroids align.
- `cluster_sym_diff_area_m2` — area covered by exactly one side; the absolute counterpart to IoU.
- `cluster_n_cad`, `cluster_n_obm` — sizes of each side. `(4, 1)` is a row of terraced houses; `(1, 3)` is a fragmented OBM detection.

These metrics are written onto every row in the cluster, so a QGIS user can immediately filter "partial matches with Hausdorff > 5 m" or "matches where the cadastre is half the area of the roof" without re-running anything.

### 5. Assemble + write output

The four output GeoDataFrames are built in the metric CRS, then reprojected to `--out-crs` (default EPSG:4326) on write.

**Why reproject back to EPSG:4326 at the end?** Three reasons: it's the universal interop CRS that opens cleanly in any tool, it matches what Stages 2 and 3 emit (so a downstream comparison is apples-to-apples), and reprojection is cheap. Areas in the output rows stay in m² because they were computed in the metric CRS *before* reprojection.

## Output

### GeoPackage (default)

Four layers, one per classification:

| Layer            | Contents                                                                         |
| ---------------- | -------------------------------------------------------------------------------- |
| `matched`        | Polygons in clusters with IoU ≥ `--iou-match`                                    |
| `partial_match`  | Polygons in clusters with `--iou-partial` ≤ IoU < `--iou-match`                  |
| `cadastre_only`  | Cadastre polygons with no credible OBM peer (candidate demolition / mapping gap) |
| `satellite_only` | OBM polygons with no credible cadastre peer (candidate **abusivo**)              |

Each row carries:

- `geometry` — the polygon (in `--out-crs`)
- `layer` — `cadastre` or `obm`, so you can tell which side a feature came from inside a `matched` / `partial_match` layer
- `area_m2` — polygon's own area in metres²
- `classification` — same as the layer name (redundant in GPKG mode, useful in GeoJSON mode)

For `matched` and `partial_match` rows, additional cluster-level fields are also propagated (`cluster_id`, `cluster_iou`, `cluster_n_cad`, `cluster_n_obm`, `cluster_cad_area_m2`, `cluster_obm_area_m2`, `cluster_centroid_dist_m`, `cluster_hausdorff_m`, `cluster_sym_diff_area_m2`). All rows in the same cluster share the same `cluster_id`, so a "select all features in this cluster" query is one SQL line.

Useful OBM attributes (`source`, `height`, `occupancy`, `obm_tile`) are propagated onto OBM rows under an `obm_` prefix.

### GeoJSON (opt-in)

GeoJSON has no multi-layer concept, so all four layers are flattened into one `FeatureCollection`. The `classification` property is how you split them back apart (QGIS: "Categorized" symbology by `classification`; mapshaper: `-filter`).

GeoPackage is the default because GIS visualisation benefits from per-classification styling, but GeoJSON earns its keep when you need a text-diffable file or want to drop the output into a non-GIS tool.

## Design choices, summarised

| Decision                                        | Why                                                                                      |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Reproject everything to a metric CRS upfront    | Tolerances in metres are honest; tolerances in degrees are not                           |
| Edges use `intersection / min(area)`, not IoU   | Small-inside-large is a common cadastre-vs-OBM pattern; symmetric metrics miss it        |
| Connected components for clustering             | N-to-M matching falls out of the graph structure — no special cases                      |
| Union-vs-union IoU for cluster scoring          | Terraced houses score correctly only when their union is compared to the roof's union    |
| Two IoU thresholds, not one                     | The middle band (10 %–50 %) is real signal: shape drift, not noise                       |
| Dissolved clusters degrade to one-side-only     | A 3 % IoU cluster isn't a partial match — it's two unrelated buildings, one on each side |
| In-script Union-Find instead of `networkx`      | ~30 lines vs a 7 MB dependency for one function call                                     |
| STRtree via `sindex` + `predicate="intersects"` | Pushes intersection pruning into C; the Python loop only sees real candidates            |
| Multi-layer GPKG as default output              | QGIS styles each layer independently — discrepancies pop visually                        |
| EPSG:4326 in the output file                    | Universal interop CRS, matches Stages 2 and 3                                            |

## Reading the output in QGIS

Drop the `.gpkg` into QGIS — all four layers appear in the Browser panel under the file. Recommended styling:

- `matched`        → green, ~40 % fill opacity
- `partial_match`  → yellow, ~50 % fill opacity
- `cadastre_only`  → red outline, no fill (these are "missing from reality")
- `satellite_only` → orange fill (these are "missing from the cadastre" — the snitch targets)

On top of an OSM/satellite basemap, the four colours give an immediate read on where the cadastre and reality disagree. Sort `satellite_only` by `area_m2` descending to triage the biggest undeclared structures first.

For deeper investigation: open the attribute table of `partial_match`, sort by `cluster_iou` ascending — the lowest IoU partial matches are typically buildings that have been rebuilt, expanded, or repositioned since the cadastre was last updated.

## Notes & limits

- **Thresholds are dataset-dependent.** The defaults (0.50 / 0.10 / 0.10) are tuned for urban areas where polygons are dense and cadastre-vs-OBM disagreements tend to be small. For rural areas with isolated buildings, you can usually push `--iou-match` up to 0.65 without losing recall. If you're seeing a lot of false `partial_match` rows that are really `matched`, drop `--iou-match` to 0.40.
- **The script is bound to a single metric CRS.** EPSG:25832 covers most of Italy well but distorts in the east. Pass `--metric-crs EPSG:25833` for areas in eastern Italy (Puglia, eastern Sicily) — same fix as the corresponding Stage 1 caveat.
- **No semantic filtering happens here.** Stage 5 is responsible for cross-referencing cadastral categories (F/2, F/3, etc.) and land-use layers (CORINE) to weed out legally non-declarable structures and false positives over greenhouses / photovoltaic plants. Stage 4 just produces the raw geometric verdict.
- **Performance is fine up to ~10⁵ polygons per side.** The bottleneck is the intersection computations in the match-graph stage, which scales roughly linearly with the number of polygon pairs returned by the spatial index — for typical Snitcher inputs (a town or neighbourhood) this is a few seconds. For city-wide runs, expect minutes.
- **The output is provisional, not authoritative.** A `satellite_only` polygon is a *candidate* undeclared building — confirmation requires a human eye on the satellite imagery, cross-reference with the cadastral category, and ideally a field check. Stage 4 is a triage tool, not an evidence tool.
- **Licensing.** OBM inputs are ODbL v1.0 (attribution + share-alike); cadastre inputs are CC BY 4.0 (attribution to Agenzia delle Entrate). The output inherits both. Preserve when redistributing.

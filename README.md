# Snitcher

Snitcher is a geospatial software pipeline that bridges the Italian Cadastral Cartography (Catasto) and open-source mapping platforms. It extracts high-precision building footprints from the Italian Cadastre via Web Map Service (WMS) layers, vectorises them into topologically valid polygons, and reconciles them against independent building footprints derived from satellite imagery, in order to surface discrepancies between declared and physically existing buildings.

## Goals

The project addresses two related problems.

The first is **discrepancy detection**, informally referred to in the project name as "snitching": identifying inconsistencies between the buildings recorded in the cadastre and the buildings actually present on the ground. Two classes of discrepancy are of interest. A building that appears in satellite imagery but is absent from the cadastre is a candidate undeclared structure (in Italian regulatory terminology, a potential *abusivo*). A building that appears in the cadastre but no longer exists in satellite imagery is a candidate demolition or mapping error.

The second is **integration with OpenStreetMap (OSM)**. Cadastral building geometries, once extracted and cleaned, are produced in a form that is topologically compatible with OSM (shared vertices between adjacent buildings, edge subdivision at T-junctions). This makes them a viable foundation for OSM contributions, although the contribution step itself is left outside the current pipeline.

## Pipeline overview

The pipeline is organised in four sequential stages. Each stage reads its input from disk and writes a self-contained artefact, which is the input to the next stage. This structure supports reproducibility, partial re-runs, and isolated benchmarking of each stage.

| Stage | Script              | Role                                  | Output            |
|-------|---------------------|---------------------------------------|-------------------|
| 1     | `1-downloader.py`   | Cadastral imagery acquisition         | GeoTIFF           |
| 2     | `2-vectorizer.py`   | Raster vectorisation                  | GeoJSON           |
| 3     | `3-truth-seeker.py` | Satellite-derived footprint acquisition | GeoPackage      |
| 4     | `4-snitcher.py`     | Geometric reconciliation              | GeoPackage        |

Stage 1 fetches cadastral imagery as a georeferenced raster. Stage 2 converts that raster into a clean vector layer of building polygons satisfying OSM topology rules. Stage 3, in parallel and independently of Stage 2, downloads building footprints from OpenBuildingMap as the satellite-derived "truth" layer. Stage 4 matches the two layers and classifies every polygon as matched, partial match, cadastre-only, or satellite-only.

## Installation

The pipeline targets Python 3.11+. All Python dependencies are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The transitive dependency graph includes `rasterio`, `geopandas`, `shapely`, `pyproj`, `fiona` (GDAL bindings), `opencv-python`, `scipy`, `numpy`, `pandas`, `Pillow`, and `requests`. On most platforms `pip install` is sufficient; on systems where GDAL is unavailable through pip wheels, install GDAL through the system package manager first.

## End-to-end run

The four stages compose with file paths. A typical run on a small target area (a few hundred metres on a side) looks as follows.

```bash
# Stage 1: download the cadastral raster for the target bbox
python 1-downloader.py \
    --bbox 45.468160,9.229277,45.471094,9.233311 \
    --layer fabbricati \
    --ppm 10 \
    --output cadastre.tif

# Stage 2: vectorise the raster into OSM-compliant polygons
python 2-vectorizer.py \
    --input cadastre.tif \
    --output buildings.geojson

# Stage 3: download the OpenBuildingMap "truth" layer for the same area
python 3-truth-seeker.py \
    --from-geotiff cadastre.tif \
    --output buildings_truth.gpkg

# Stage 4: reconcile the two layers and classify discrepancies
python 4-snitcher.py \
    --cadastre buildings.geojson \
    --truth buildings_truth.gpkg \
    --output discrepancies.gpkg
```

The final `discrepancies.gpkg` is a multi-layer GeoPackage that opens directly in QGIS for inspection.

## Stage 1: cadastral imagery acquisition

The role of Stage 1 is to download a layer from the Italian Cadastre WMS (`https://wms.cartografia.agenziaentrate.gov.it`) over a user-specified bounding box, at a resolution sufficient to resolve the thinnest cartographic line of the cadastre, and to persist the result as a single georeferenced raster.

The user supplies a bounding box in EPSG:4326 (latitude / longitude) and a target resolution expressed as pixels per metre. The script reprojects the bounding box to EPSG:25832 (ETRS89 / UTM zone 32N, metric), computes the total mosaic pixel size from the metric extent and the requested resolution, splits the mosaic into tiles no larger than the 2048 × 2048 WMS per-request limit, and issues the corresponding `GetMap` requests concurrently using a thread pool. Tiles are pasted into an in-memory canvas and the assembled image is written as either a GeoTIFF (default, georeferenced) or a plain PNG.

The working CRS is metric on purpose: pixels per metre is meaningful only against a metric reference, and adjacent-tile edge arithmetic is trivial in metres but error-prone in degrees. Tile bounding boxes are computed from pixel offsets using a single metres-per-pixel ratio derived from the mosaic dimensions, which guarantees that adjacent tiles share exact edge coordinates and never overlap or leave seams from floating-point drift.

A `Content-Type` check distinguishes a valid PNG response from the `ServiceExceptionReport` XML the WMS returns on error (with HTTP 200), and tile fetches are wrapped in a small retry loop with a one-second back-off. Concurrency is implemented with a `ThreadPoolExecutor` because the workload is I/O bound; worker threads do only the HTTP call, while PIL paste operations are kept on the main thread (PIL is not thread-safe for writes to the same image).

The default GeoTIFF output uses four bands (RGBA), LZW compression, and internal 256 × 256 tiling for efficient sub-window reads in Stage 2. The CRS, affine transform, and a small set of provenance tags (source WMS URL, version, layer name, UTC acquisition timestamp, bounding box in both CRSs, pixels per metre, licence, attribution) are embedded in the file.

## Stage 2: raster vectorisation

Stage 2 converts the cadastral raster from Stage 1 into a clean vector layer of building polygons. The output is a GeoJSON `FeatureCollection` in EPSG:4326 that satisfies two OSM-topology requirements: adjacent buildings sharing a corner have byte-identical vertex coordinates, and any vertex that lies on a foreign edge is inserted into that edge as an explicit vertex.

The script proceeds through seven substages.

**Raster load.** The GeoTIFF is read with `rasterio`, retaining the affine transform, the CRS, and the Stage 1 provenance tags. The transform is the link that allows subsequent steps to produce geometries directly in metric world coordinates.

**Orange mask.** The cadastral `fabbricati` layer renders buildings as a solid orange fill (RGB ≈ 236, 128, 19). A binary mask is built by converting RGB to HSV and applying a hue / saturation / value range filter, followed by a 3 × 3 `MORPH_OPEN` to remove isolated specks. No closing operation is applied: a closing would dilate orange across the one-pixel-wide black walls that separate attached buildings and fuse them into a single connected component, which is the precise outcome to avoid.

**Connected components.** `cv2.connectedComponents` with 4-connectivity assigns one label per building. Four-connectivity (rather than eight) prevents two orange pixels touching only diagonally, a typical artefact at sharp inner corners, from being merged across the black wall.

**Polygonisation.** `rasterio.features.shapes` walks each label and emits polygons in the working CRS, with courtyards captured automatically as interior rings of the surrounding polygon. Polygons below a minimum-area threshold (default 4 m²) are discarded as noise.

**Simplification.** Each polygon is simplified with the Douglas-Peucker algorithm at a 0.30 m tolerance. Cadastral buildings are blocky and almost always axis-aligned, so DP simplification at this tolerance retains the corner vertices and discards every pixel-level zig-zag along the perimeter. Empirically this yields a roughly seven-fold reduction in vertex count without losing real corners.

**Vertex snapping.** To enforce the first OSM-topology condition (shared corners), all vertices from all rings are flattened into a single array and clustered greedily using a KD-tree: any unassigned vertex within the snap radius (default 0.80 m) of an existing cluster's seed joins that cluster. Each cluster is then collapsed to its centroid and every ring is rebuilt from the centroid coordinates, with consecutive duplicates removed. After this pass, two corners that should coincide are at byte-identical coordinates.

**Edge subdivision.** To enforce the second OSM-topology condition (T-junctions), every distinct vertex coordinate is indexed in a KD-tree and, for every edge of every ring, foreign vertices that project onto the edge within tolerance (default 0.50 m) and inside the open interval `(0, 1)` of the segment parameter are inserted between the edge endpoints, sorted by their projection parameter. The ring's own vertices are excluded from this check to avoid re-inserting collinear vertices into a polygon's own edges.

**Reprojection and serialisation.** Each polygon is reprojected from EPSG:25832 to EPSG:4326 for the GeoJSON output. Areas are computed before reprojection so that the recorded `area_m2` is a true metric area. Exteriors are re-oriented counter-clockwise to comply with RFC 7946.

Intermediate PNG renderings are written to a debug directory at each substage, which makes failures localisable to a specific transformation.

## Stage 3: satellite-derived footprint acquisition

Stage 3 produces the "truth" layer against which the cadastral polygons are reconciled. Rather than training a roof-detection model from scratch, the pipeline ingests an existing public building dataset, **OpenBuildingMap** (OBM), which aggregates footprints from OpenStreetMap, Google Open Buildings, and Microsoft Global ML Building Footprints into a single harmonised layer.

OpenBuildingMap is published by GFZ Data Services as a set of GeoPackage files, each covering a Bing-style quadkey tile at zoom level 6. Italy is covered by eight such tiles. The script accepts a target bounding box either directly on the command line or by reading the `BBOX_EPSG_4326` provenance tag from a Stage 1 GeoTIFF, computes the set of zoom-6 quadkeys that intersect it, and for each quadkey ensures that the corresponding tile is present in an on-disk cache. Missing tiles are downloaded as compressed `.gpkg.bz2` archives (approximately 700 MB compressed, 1.9 GB decompressed each), with the bz2 stream decoded on the fly and written to a `.gpkg.part` staging path before atomic rename, so that an interrupted download cannot leave a corrupt cache entry.

Per-tile filtering uses `geopandas.read_file(path, bbox=...)`, which delegates to GDAL's spatial filter and exploits the R-tree index that is built into every GeoPackage. The 1.9 GB file is never fully materialised; only features whose envelope intersects the target bounding box are returned. Results from all relevant tiles are concatenated, every row is annotated with its source quadkey for provenance, and the merged layer is written as either a multi-feature GeoPackage (default) or a GeoJSON.

The output preserves OBM's attributes (`source`, `height`, `occupancy`) and adds the `obm_tile` column. The CRS is EPSG:4326, mirroring OBM's native CRS so that no precision is lost to an unnecessary reprojection at this stage.

## Stage 4: geometric reconciliation

Stage 4 compares the cadastral polygons from Stage 2 against the OBM footprints from Stage 3 and classifies each polygon into one of four categories: `matched`, `partial_match`, `cadastre_only`, or `satellite_only`. The output is a multi-layer GeoPackage (or a flattened GeoJSON) suitable for visual inspection in QGIS.

The non-trivial aspect of this stage is that the correspondence between the two layers is many-to-many. A row of four terraced houses appears as four polygons in the cadastre but typically as a single shared-roof polygon in OBM. Conversely, a single legal building can be drawn as several roof patches by an ML-based footprint detector. Comparing polygons one-to-one would assign artificially low scores to all such cases and lose the underlying agreement.

The script resolves this in four steps.

**Reprojection.** Both inputs are reprojected from EPSG:4326 to a metric CRS (default EPSG:25832), so that all thresholds the user passes (areas in m², distances in m) are honest. Invalid geometries are repaired with `buffer(0)`, and polygons below `--min-area` (default 1 m²) are discarded as noise.

**Match graph construction.** A bipartite graph is built between cadastral and OBM polygons. Two polygons are linked by an edge when the area of their intersection, divided by the area of the smaller of the two polygons, exceeds `--overlap-ratio` (default 0.10). The `min(area)` denominator is chosen so that a small cadastral unit lying mostly inside a much larger OBM roof still produces an edge — the small polygon's area dominates the ratio. Candidate pairs are pre-filtered with a `STRtree` spatial index (`geopandas.sindex` with `predicate="intersects"`), which prunes the pairwise check to a handful of candidates per cadastral polygon.

**Cluster discovery.** Connected components of the bipartite graph are computed with an in-script Union-Find structure (with path compression and union by rank, avoiding a dependency on `networkx` for a single function call). Each connected component containing polygons from both sides is a "mixed cluster" — a group of polygons that, collectively, describe the same physical building or buildings. Components consisting of only cadastral polygons or only OBM polygons are dispatched directly to `cadastre_only` and `satellite_only` respectively.

**Cluster scoring and classification.** For each mixed cluster, the union of the cadastre side and the union of the OBM side are computed, and aggregate metrics are derived from those unions: intersection-over-union (IoU), symmetric-difference area, centroid distance, and Hausdorff distance. Comparing unions, rather than scoring each polygon individually, is what makes the N-to-M case work correctly: four terraced houses are scored against the union of the OBM roof, not against the roof four times. Each cluster is classified by its aggregate IoU:

- `iou >= --iou-match` (default 0.50): `matched`.
- `--iou-partial <= iou < --iou-match` (default 0.10 to 0.50): `partial_match`. The cluster overlaps substantially but the geometry has drifted (shape, position, or both).
- `iou < --iou-partial`: the cluster is dissolved. Its polygons fall back individually to `cadastre_only` and `satellite_only`. This handles the case where two unrelated neighbouring buildings happened to graze each other above the edge threshold without honestly describing the same physical structure.

The output GeoPackage contains four layers, one per classification. Every row carries its source layer (`cadastre` or `obm`), its individual area in m², and, for the matched and partial-match categories, a complete set of cluster-level fields (`cluster_id`, `cluster_iou`, `cluster_n_cad`, `cluster_n_obm`, `cluster_cad_area_m2`, `cluster_obm_area_m2`, `cluster_centroid_dist_m`, `cluster_hausdorff_m`, `cluster_sym_diff_area_m2`). Useful OBM attributes are propagated onto OBM rows under an `obm_` prefix. The file is reprojected back to EPSG:4326 on write so that it opens cleanly in any downstream tool.

## Coordinate reference systems

The pipeline distinguishes between three CRSs throughout.

EPSG:4326 (WGS 84 geographic) is used at the boundaries: the user-supplied bounding box, the GeoJSON output of Stage 2, the GeoPackage outputs of Stages 3 and 4. It is the universal interoperability CRS and matches the native CRS of OpenBuildingMap.

EPSG:25832 (ETRS89 / UTM zone 32N) is used for all metric computation: WMS requests in Stage 1, polygon simplification, vertex snapping and edge subdivision in Stage 2, and all area, distance, and IoU computation in Stage 4. It is metric, supported natively by the cadastral WMS, and covers most of Italy with low distortion. For target areas in eastern Italy (Puglia, eastern Sicily), the equivalent constant should be switched to EPSG:25833 in Stage 1 and `--metric-crs EPSG:25833` should be passed to Stage 4.

The Web Mercator tile system (Bing quadkeys at zoom 6) is used only inside Stage 3 to identify which OBM tiles cover a target area; it never appears in any output.

## Output and inspection

The recommended workflow for reviewing Stage 4 results is to load `discrepancies.gpkg` in QGIS. Each of the four layers can be styled independently: green semi-transparent fill for `matched`, yellow for `partial_match`, red outline (no fill) for `cadastre_only` (candidate demolitions or mapping gaps), and orange fill for `satellite_only` (candidate undeclared buildings, the principal targets of the analysis). Sorting `satellite_only` by `area_m2` descending triages the largest candidate undeclared structures first. Opening the attribute table of `partial_match` and sorting by `cluster_iou` ascending surfaces buildings that have been rebuilt, expanded, or repositioned since the cadastre was last updated.

The output is provisional. A `satellite_only` polygon is a candidate, not evidence. Confirmation requires a human reviewer comparing the polygon against current satellite imagery, the relevant cadastral category, and, where possible, a field check.

## Licensing and attribution

The pipeline combines two upstream datasets, each with its own licence, which the output inherits.

The cadastral imagery acquired in Stage 1 is published by the Agenzia delle Entrate under **CC BY 4.0**, requiring attribution.

The OpenBuildingMap data acquired in Stage 3 is published under **ODbL v1.0**, requiring attribution and share-alike. OBM in turn aggregates data from OpenStreetMap, Google Open Buildings, and Microsoft Global ML Building Footprints, each of which has its own upstream attribution requirements that propagate through OBM.

Any redistribution of pipeline outputs must preserve both licences and the chain of attribution.

## Known limitations

The pipeline operates on raster imagery from the cadastral WMS rather than on the cadastre's underlying vector data, because the vector data is not publicly available. The vectorisation in Stage 2 is therefore a reconstruction, accurate to within the simplification, snapping, and subdivision tolerances. Where the cadastral imagery itself is inaccurate (older municipalities have not been re-surveyed in decades), Stage 2's output will inherit those inaccuracies.

The orange-fill colour thresholds in Stage 2 are tuned for the current `fabbricati` symbology. Should the WMS server change its rendering, the HSV bounds at the top of `2-vectorizer.py` would need re-tuning.

Stage 3 requires a one-time download of approximately 700 MB compressed per OBM tile (1.9 GB decompressed). For target areas inside a single tile, this is a one-off cost; for areas spanning multiple tiles, the cache grows accordingly. A complete Italy-wide cache totals roughly 6 GB.

Stage 4's classification is a purely geometric verdict. No semantic filtering is applied, so legally non-declarable cadastral categories (for example, F/2, F/3, F/4, F/5, F/7, or *beni comuni non censibili*) are not excluded, and false positives over greenhouses and photovoltaic plants are not filtered out. These steps are intended for a downstream semantic stage that cross-references cadastral category data and land-use layers such as CORINE Land Cover.

The pipeline does not currently produce OSM-ready changesets. The polygons emitted by Stage 2 are OSM-topology compliant by construction, so they are suitable as a foundation for an OSM contribution, but the conflation with existing OSM features, tag mapping, and changeset generation steps required by the OSM Italy import procedure are not implemented here.

## Repository layout

```
snitcher/
├── docs/
│   ├── 1-downloader.md      Stage 1: detailed documentation
│   ├── 2-vectorizer.md      Stage 2: detailed documentation
│   ├── 3-truth-seeker.md    Stage 3: detailed documentation
│   └── 4-snitcher.md        Stage 4: detailed documentation
├── 1-downloader.py          Stage 1: cadastral imagery acquisition
├── 2-vectorizer.py          Stage 2: raster vectorisation
├── 3-truth-seeker.py        Stage 3: satellite-derived footprint acquisition
├── 4-snitcher.py            Stage 4: geometric reconciliation
├── requirements.txt         Pinned Python dependencies
└── README.md                This file
```

Each file under `docs/` contains the complete design notes, parameter reference, and discussion of edge cases for the corresponding stage. The present document is the high-level entry point; the per-stage documents in `docs/` are the reference.

"""
Snitcher — Stage 4: Geometric reconciliation (a.k.a. "the snitcher").

Compares the cadastral building polygons from Stage 2 (GeoJSON) against
the OpenBuildingMap (OBM) footprints from Stage 3 (GeoPackage / GeoJSON)
and classifies each polygon as one of:

    matched         — the two layers agree on this building
    partial_match   — significant overlap but the shape/position drifted
    cadastre_only   — declared in the cadastre, no satellite footprint
                      (candidate demolition / mapping error)
    satellite_only  — visible on satellite, not declared in the cadastre
                      (candidate undeclared / "abusivo")

Reference systems
-----------------
Stage 2 emits GeoJSON in EPSG:4326 and Stage 3 emits its GeoPackage in
EPSG:4326. Areas and distances on a geographic CRS are not in metres,
so all geometry math here is done after reprojecting to a metric CRS
(default EPSG:25832, ETRS89 / UTM 32N — same as Stage 1). The script
reprojects back to EPSG:4326 just before writing the output, so it stays
portable for any GIS tool.

Many-to-many matching
---------------------
Terraced houses are the classic case: 4 cadastral units sharing one roof
will be 4 polygons in the cadastre and 1 polygon in OBM (the inverse
also happens — a single legal building drawn as multiple roof patches).

The trick is to never try to match polygon-by-polygon. Instead:
  1. Build a bipartite graph: an edge cadastre[i] <-> obm[j] when their
     intersection covers at least `overlap-ratio` of the smaller polygon.
  2. Find connected components — each component is a "cluster" of
     polygons that are talking about the same physical building(s).
  3. Per cluster, compute aggregate IoU between the *union* of cadastre
     polygons and the *union* of OBM polygons.

Output
------
GeoPackage with four layers (one per classification) — best for GIS
viewing since each layer can be styled independently. GeoJSON is also
supported via --format geojson; in that mode everything is merged into
one FeatureCollection and `classification` becomes a property.

Install
-------
    pip install geopandas shapely pyproj fiona pandas
"""

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union


# --- defaults --------------------------------------------------------------
DEFAULT_METRIC_CRS    = "EPSG:25832"    # ETRS89 / UTM 32N (covers most of Italy)
DEFAULT_OUTPUT_CRS    = "EPSG:4326"     # what the file gets saved in
DEFAULT_OUTPUT        = "discrepancies.gpkg"
DEFAULT_IOU_MATCH     = 0.50            # cluster IoU at/above this -> matched
DEFAULT_IOU_PARTIAL   = 0.10            # cluster IoU at/above this -> partial_match
DEFAULT_OVERLAP_RATIO = 0.10            # intersection / min(area) to draw an edge
DEFAULT_MIN_AREA      = 1.0             # m^2 — drop noise polygons under this

# Columns from the OBM input that we propagate into the output if present
OBM_PROPAGATE = ("source", "height", "occupancy", "obm_tile")


# --- Union-Find ------------------------------------------------------------
# Tiny in-script implementation so we don't have to depend on networkx
# just to find connected components.
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]   # path compression
            x = self.p[x]
        return x

    def union(self, x, y):
        a, b = self.find(x), self.find(y)
        if a == b:
            return
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1


# --- loading ---------------------------------------------------------------
def load_layer(path, label, metric_crs, min_area):
    """Read a vector file, reproject to the metric CRS, drop noise/invalids."""
    print(f"[DEBUG] loading {label}: {path}")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise SystemExit(f"{path}: input has no CRS — refusing to guess")
    print(f"[DEBUG]   {len(gdf)} features in {gdf.crs}")

    # Reproject to the metric CRS — every tolerance in this script is in metres
    if gdf.crs.to_string() != metric_crs:
        gdf = gdf.to_crs(metric_crs)
        print(f"[DEBUG]   reprojected to {metric_crs}")

    # Drop empties and try to fix any invalid geometries with buffer(0).
    # OBM occasionally ships polygons with self-intersections that would
    # otherwise blow up intersection() calls downstream.
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"[DEBUG]   fixing {int(invalid.sum())} invalid geometries via buffer(0)")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Drop tiny polygons — these are typically rasterisation noise
    gdf = gdf[gdf.geometry.area >= min_area].copy()
    if len(gdf) != before:
        print(f"[DEBUG]   dropped {before - len(gdf)} features (empty / invalid / below min-area)")

    gdf = gdf.reset_index(drop=True)
    return gdf


# --- match graph -----------------------------------------------------------
def build_match_graph(cad_gdf, obm_gdf, overlap_threshold):
    """Find cadastre<->OBM pairs that share enough area to be 'the same building'.

    Returns a list of (cad_idx, obm_idx) edges. The threshold is on
    intersection / min(area_a, area_b), so a small polygon mostly
    contained in a large one still produces an edge.
    """
    if len(cad_gdf) == 0 or len(obm_gdf) == 0:
        return []

    # geopandas' sindex hides the STRtree behind a clean API and lets us
    # push the 'intersects' predicate into the C side so we only see truly
    # intersecting candidates, not just envelope hits.
    sindex = obm_gdf.sindex

    edges = []
    for i, cad_geom in enumerate(cad_gdf.geometry):
        cad_area = cad_geom.area
        candidates = sindex.query(cad_geom, predicate="intersects")
        for j in candidates:
            j = int(j)
            obm_geom = obm_gdf.geometry.iloc[j]
            inter = cad_geom.intersection(obm_geom)
            if inter.is_empty:
                continue
            # min(area) -> we want to keep small-polygon-inside-large-polygon links
            denom = min(cad_area, obm_geom.area)
            if denom <= 0:
                continue
            if inter.area / denom >= overlap_threshold:
                edges.append((i, j))

    print(f"[DEBUG] match graph: {len(edges)} edges across "
          f"{len(cad_gdf)} cadastre x {len(obm_gdf)} OBM polygons")
    return edges


def find_clusters(n_cad, n_obm, edges):
    """Group polygons into connected components.

    Single index space: cadastre at [0, n_cad), OBM at [n_cad, n_cad+n_obm).
    Returns (mixed_clusters, lonely_cad_idxs, lonely_obm_idxs) where a
    'mixed' cluster has at least one polygon from each side; the lonely
    lists are everything else (polygons in components with no peer
    across the layer boundary).
    """
    uf = UnionFind(n_cad + n_obm)
    for ci, oi in edges:
        uf.union(ci, n_cad + oi)

    buckets = defaultdict(lambda: {"cad": [], "obm": []})
    for i in range(n_cad):
        buckets[uf.find(i)]["cad"].append(i)
    for j in range(n_obm):
        buckets[uf.find(n_cad + j)]["obm"].append(j)

    mixed, lonely_cad, lonely_obm = [], [], []
    for c in buckets.values():
        if c["cad"] and c["obm"]:
            mixed.append(c)
        elif c["cad"]:
            lonely_cad.extend(c["cad"])
        else:
            lonely_obm.extend(c["obm"])

    print(f"[DEBUG] clusters: {len(mixed)} mixed, "
          f"{len(lonely_cad)} lonely cadastre, {len(lonely_obm)} lonely OBM")
    return mixed, lonely_cad, lonely_obm


# --- per-cluster metrics ---------------------------------------------------
def cluster_metrics(cad_gdf, obm_gdf, cluster):
    """Compute aggregate IoU / centroid distance / Hausdorff for one cluster.

    The metrics are computed on the *unions* per side: this is what makes
    the N-to-M case work correctly — three terraced houses' union should
    match the one OBM roof's union, not match it three times.
    """
    cad_union = unary_union([cad_gdf.geometry.iloc[i] for i in cluster["cad"]])
    obm_union = unary_union([obm_gdf.geometry.iloc[j] for j in cluster["obm"]])

    inter_area = cad_union.intersection(obm_union).area
    union_area = cad_union.union(obm_union).area
    iou = inter_area / union_area if union_area > 0 else 0.0

    return {
        "iou":               iou,
        "cad_area_m2":       cad_union.area,
        "obm_area_m2":       obm_union.area,
        "centroid_dist_m":   cad_union.centroid.distance(obm_union.centroid),
        "hausdorff_m":       cad_union.hausdorff_distance(obm_union),
        "sym_diff_area_m2":  cad_union.symmetric_difference(obm_union).area,
        "n_cad":             len(cluster["cad"]),
        "n_obm":             len(cluster["obm"]),
    }


# --- output GeoDataFrames --------------------------------------------------
def _propagate_obm_props(obm_row):
    """Pull a small, fixed set of useful OBM columns into the output row."""
    out = {}
    for col in OBM_PROPAGATE:
        if col in obm_row.index:
            out[f"obm_{col}" if col != "obm_tile" else "obm_tile"] = obm_row[col]
    return out


def build_output_gdfs(cad_gdf, obm_gdf, clusters, lonely_cad, lonely_obm,
                     iou_match, iou_partial, metric_crs):
    """Assemble the four output GeoDataFrames, one per classification.

    Each cluster is classified by its aggregate IoU:
      iou >= iou_match    -> matched
      iou >= iou_partial  -> partial_match
      iou <  iou_partial  -> noise overlap; cluster is dissolved and its
                             polygons are pushed into cadastre_only /
                             satellite_only respectively
    """
    matched, partial = [], []
    extra_cad_only, extra_obm_only = [], []
    next_cid = 0

    for cluster in clusters:
        m = cluster_metrics(cad_gdf, obm_gdf, cluster)

        if m["iou"] >= iou_match:
            classification, sink = "matched", matched
        elif m["iou"] >= iou_partial:
            classification, sink = "partial_match", partial
        else:
            # Below partial threshold: not really the same building.
            # Demote everyone in the cluster to a one-side-only verdict.
            extra_cad_only.extend(cluster["cad"])
            extra_obm_only.extend(cluster["obm"])
            continue

        cid = next_cid
        next_cid += 1

        # Cluster-level fields shared by every row in this cluster
        cluster_fields = {
            "classification":            classification,
            "cluster_id":                cid,
            "cluster_iou":               round(m["iou"], 4),
            "cluster_n_cad":             m["n_cad"],
            "cluster_n_obm":             m["n_obm"],
            "cluster_cad_area_m2":       round(m["cad_area_m2"], 2),
            "cluster_obm_area_m2":       round(m["obm_area_m2"], 2),
            "cluster_centroid_dist_m":   round(m["centroid_dist_m"], 3),
            "cluster_hausdorff_m":       round(m["hausdorff_m"], 3),
            "cluster_sym_diff_area_m2":  round(m["sym_diff_area_m2"], 2),
        }

        # One row per polygon, both sides tagged with the cluster's metrics
        for ci in cluster["cad"]:
            geom = cad_gdf.geometry.iloc[ci]
            sink.append({
                "geometry":  geom,
                "layer":     "cadastre",
                "area_m2":   round(geom.area, 2),
                **cluster_fields,
            })
        for oi in cluster["obm"]:
            geom = obm_gdf.geometry.iloc[oi]
            row = {
                "geometry":  geom,
                "layer":     "obm",
                "area_m2":   round(geom.area, 2),
                **cluster_fields,
                **_propagate_obm_props(obm_gdf.iloc[oi]),
            }
            sink.append(row)

    # cadastre_only = lonely cadastre polygons + demoted-from-weak clusters
    cad_only_rows = []
    for ci in lonely_cad + extra_cad_only:
        geom = cad_gdf.geometry.iloc[ci]
        cad_only_rows.append({
            "geometry":       geom,
            "layer":          "cadastre",
            "classification": "cadastre_only",
            "area_m2":        round(geom.area, 2),
        })

    obm_only_rows = []
    for oi in lonely_obm + extra_obm_only:
        geom = obm_gdf.geometry.iloc[oi]
        obm_only_rows.append({
            "geometry":       geom,
            "layer":          "obm",
            "classification": "satellite_only",
            "area_m2":        round(geom.area, 2),
            **_propagate_obm_props(obm_gdf.iloc[oi]),
        })

    def _make_gdf(rows):
        if not rows:
            return gpd.GeoDataFrame(geometry=[], crs=metric_crs)
        return gpd.GeoDataFrame(rows, crs=metric_crs)

    return {
        "matched":         _make_gdf(matched),
        "partial_match":   _make_gdf(partial),
        "cadastre_only":   _make_gdf(cad_only_rows),
        "satellite_only":  _make_gdf(obm_only_rows),
    }


# --- writing ---------------------------------------------------------------
def write_output(gdfs, output_path, fmt, out_crs):
    """Write the per-classification layers to GPKG (multi-layer) or GeoJSON."""
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    if fmt == "gpkg":
        wrote_any = False
        for name, gdf in gdfs.items():
            if len(gdf) == 0:
                print(f"[DEBUG]   layer '{name}': empty (skipped)")
                continue
            gdf_out = gdf.to_crs(out_crs)
            # mode='a' on subsequent writes appends a new layer to the same .gpkg
            gdf_out.to_file(
                output_path, layer=name, driver="GPKG",
                mode="a" if wrote_any else "w",
            )
            wrote_any = True
            print(f"[DEBUG]   layer '{name}': {len(gdf_out)} features")
        if not wrote_any:
            print("[WARNING] all layers empty — output file not created")

    elif fmt == "geojson":
        # GeoJSON has no multi-layer concept, so merge everything into a
        # single FeatureCollection. The 'classification' property is how
        # the user splits it back apart in QGIS / mapshaper / whatever.
        parts = [g.to_crs(out_crs) for g in gdfs.values() if len(g) > 0]
        if not parts:
            print("[WARNING] all layers empty — output file not created")
            return
        merged = pd.concat(parts, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, crs=out_crs)
        merged.to_file(output_path, driver="GeoJSON")
        print(f"[DEBUG]   wrote {len(merged)} features to {output_path}")

    else:
        raise ValueError(f"unknown format: {fmt!r}")


def infer_format(output_path, explicit):
    if explicit:
        return explicit
    return {
        ".gpkg":    "gpkg",
        ".geojson": "geojson",
        ".json":    "geojson",
    }.get(Path(output_path).suffix.lower(), "gpkg")


# --- summary ---------------------------------------------------------------
def print_summary(gdfs, n_cad_total, n_obm_total):
    """Compact final summary so the operator gets the picture at a glance."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cadastre buildings in: {n_cad_total}")
    print(f"OBM buildings in:      {n_obm_total}")
    print()
    for name, gdf in gdfs.items():
        if len(gdf) == 0:
            print(f"  {name:14s}: 0")
            continue
        n_cad = int((gdf["layer"] == "cadastre").sum()) if "layer" in gdf else 0
        n_obm = int((gdf["layer"] == "obm").sum()) if "layer" in gdf else 0
        extra = ""
        if name in ("matched", "partial_match") and "cluster_id" in gdf:
            n_clusters = gdf["cluster_id"].nunique()
            extra = f", {n_clusters} clusters"
        print(f"  {name:14s}: {len(gdf):5d} features ({n_cad} cad + {n_obm} obm{extra})")
    print("=" * 60)


# --- CLI -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Snitcher Stage 4 — find discrepancies between cadastral "
                    "and OpenBuildingMap building polygons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cadastre", required=True,
        help="Stage-2 GeoJSON of cadastral building polygons (EPSG:4326)",
    )
    parser.add_argument(
        "--truth", required=True,
        help="Stage-3 GeoPackage/GeoJSON of OBM building footprints (EPSG:4326)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="output path",
    )
    parser.add_argument(
        "--format", choices=["gpkg", "geojson"], default=None,
        help="output format (default: inferred from --output extension)",
    )
    parser.add_argument(
        "--metric-crs", default=DEFAULT_METRIC_CRS,
        help="metric CRS for area/distance math; use EPSG:25833 for eastern Italy",
    )
    parser.add_argument(
        "--out-crs", default=DEFAULT_OUTPUT_CRS,
        help="CRS of the written file (input data is reprojected to this on write)",
    )
    parser.add_argument(
        "--iou-match", type=float, default=DEFAULT_IOU_MATCH,
        help="aggregate IoU at/above which a cluster is 'matched'",
    )
    parser.add_argument(
        "--iou-partial", type=float, default=DEFAULT_IOU_PARTIAL,
        help="aggregate IoU at/above which a cluster is 'partial_match'; "
             "below this its polygons are split into cadastre_only / satellite_only",
    )
    parser.add_argument(
        "--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO,
        help="min intersection/min(area) to link two polygons across layers — "
             "controls how aggressively the script groups N-to-M clusters",
    )
    parser.add_argument(
        "--min-area", type=float, default=DEFAULT_MIN_AREA,
        help="drop input polygons smaller than this many m^2 (noise filter)",
    )
    args = parser.parse_args()

    # Validate threshold ordering — gets the user a useful error early
    if args.iou_partial > args.iou_match:
        raise SystemExit("--iou-partial must be <= --iou-match")

    fmt = infer_format(args.output, args.format)
    print(f"[DEBUG] output format:    {fmt}")
    print(f"[DEBUG] metric CRS (math): {args.metric_crs}")
    print(f"[DEBUG] output CRS:       {args.out_crs}")
    print(f"[DEBUG] thresholds:       iou_match={args.iou_match}, "
          f"iou_partial={args.iou_partial}, overlap_ratio={args.overlap_ratio}, "
          f"min_area={args.min_area} m^2")

    cad_gdf = load_layer(args.cadastre, "cadastre", args.metric_crs, args.min_area)
    obm_gdf = load_layer(args.truth,    "OBM truth", args.metric_crs, args.min_area)

    edges = build_match_graph(cad_gdf, obm_gdf, args.overlap_ratio)
    clusters, lonely_cad, lonely_obm = find_clusters(len(cad_gdf), len(obm_gdf), edges)

    gdfs = build_output_gdfs(
        cad_gdf, obm_gdf, clusters, lonely_cad, lonely_obm,
        args.iou_match, args.iou_partial, args.metric_crs,
    )

    write_output(gdfs, args.output, fmt, args.out_crs)
    print_summary(gdfs, len(cad_gdf), len(obm_gdf))
    print(f"[DEBUG] done -> {args.output}")


if __name__ == "__main__":
    main()

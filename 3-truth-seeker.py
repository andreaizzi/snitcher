"""
Snitcher - Stage 3: Truth Seeker (acquisizione OpenBuildingMap).

Per una data bounding box, scarica i tile OpenBuildingMap che la coprono,
applica un filtro spaziale lato server tramite l'indice R-tree del
GeoPackage, e scrive un singolo GeoPackage (o GeoJSON) pronto per lo
Stage 4 (riconciliazione geometrica).
"""

import argparse
import bz2
import math
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests

# OpenBuildingMap dataset
OBM_BASE_URL = (
    "https://datapub.gfz.de/download/10.5880.GFZ.LKUT.2025.002-Caweb/"
    "2025-002_Oostwegel-et-al_data"
)
OBM_TILE_ZOOM = 6   # OBM pubblica un .gpkg per ogni tile quadkey a zoom 6
OBM_CRS = "EPSG:4326"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "obm_cache"
DEFAULT_OUTPUT = "buildings_truth.gpkg"
DOWNLOAD_CHUNK = 1024 * 1024   # chunk da 1 MiB per lo streaming del download e la decodifica bz2


# quadkey (sistema di tile Bing Maps)

def lonlat_to_tile_xy(lon, lat, zoom):
    """Project (lon, lat) to integer tile (x, y) at the given zoom (Web Mercator)."""
    lat_rad = math.radians(lat) # latitudine in radianti
    n = 2 ** zoom   # n è il numero di tile per lato a questo zoom (a zoom 6 ho 64x64 tile)
    x = int((lon + 180.0) / 360.0 * n) # x va da 0 a n-1 mappando linearmente la longitudine [-180, 180]

    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n) # asinh(tan(lat)) è la proiezione y standard di Web Mercator
    x = max(0, min(x, n - 1))
    y = max(0, min(y, n - 1))
    return x, y


def tile_xy_to_quadkey(x, y, zoom):
    """Convert tile (x, y, z) to a Bing-style quadkey string."""
    # il quadkey si costruisce bit per bit, partendo dal piu' significativo
    out = []
    for i in range(zoom, 0, -1):
        digit = 0
        # maschera per estrarre il bit i-esimo di x e y
        mask = 1 << (i - 1)
        # bit di x -> +1, bit di y -> +2: così ogni cifra finisce per essere 0, 1, 2 o 3
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        out.append(str(digit))
    return "".join(out)


def bbox_to_quadkeys(bbox_4326, zoom=OBM_TILE_ZOOM):
    """Return every quadkey tile at `zoom` that intersects `bbox_4326`."""

    lat_min, lng_min, lat_max, lng_max = bbox_4326
    # l'asse Y dei tile è invertito: latitudine max -> tile y minimo
    x_lo, y_hi = lonlat_to_tile_xy(lng_min, lat_min, zoom)
    x_hi, y_lo = lonlat_to_tile_xy(lng_max, lat_max, zoom)
    # genero tutte le combinazioni (x, y) nel rettangolo di tile coperto dalla bbox
    return [
        tile_xy_to_quadkey(x, y, zoom)
        for x in range(x_lo, x_hi + 1)
        for y in range(y_lo, y_hi + 1)
    ]


def bbox_from_geotiff(path):
    """Recover the EPSG:4326 bbox from a Stage-1 GeoTIFF.

    Prefers the `BBOX_EPSG_4326` tag written by 1-downloader.py - that is
    the exact input bbox, untouched by reprojection rounding. Falls back
    to reprojecting the dataset bounds if the tag is absent.
    """
    with rasterio.open(path) as ds:
        tag = ds.tags().get("BBOX_EPSG_4326") # tag scritto durante fase 1
        if tag:
            return tuple(float(v) for v in tag.split(","))

        # fallback: riproietto i bounds del dataset a EPSG:4326
        from rasterio.warp import transform_bounds
        # densify_pts=21 aggiunge punti lungo i bordi per non perdere precisione in riproiezione
        left, bottom, right, top = transform_bounds(
            ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21
        )

        return (bottom, left, top, right)


def ensure_tile_cached(quadkey, cache_dir):
    """Download + decompress the OBM tile if not already cached.

    Returns the path to the decompressed .gpkg.
    """

    cache_dir.mkdir(parents=True, exist_ok=True) # creo la cartella di cache
    gpkg_path = cache_dir / f"building.{quadkey}.gpkg"

    if gpkg_path.exists(): # se file in cache, lo riuso
        size_mb = gpkg_path.stat().st_size / 1e6
        print(f"[DEBUG] cache hit for tile {quadkey}: {gpkg_path} ({size_mb:.0f} MB)")
        return gpkg_path

    url = f"{OBM_BASE_URL}/building.{quadkey}.gpkg.bz2"
    print(f"[DEBUG] downloading tile {quadkey} from {url}")

    tmp_path = gpkg_path.with_suffix(".gpkg.part")
    decompressor = bz2.BZ2Decompressor() # decomprime mentre scarico
    bytes_in = 0
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            with open(tmp_path, "wb") as fout:
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK): # itero un chunk da 1 MiB alla volta
                    if not chunk:
                        continue
                    bytes_in += len(chunk)
                    fout.write(decompressor.decompress(chunk)) # decomprimo e scrivo sul disco senza occupare memoria
                    if total:
                        pct = 100 * bytes_in / total
                        print(f"[DEBUG]   downloaded {bytes_in/1e6:6.0f} / "
                              f"{total/1e6:.0f} MB ({pct:5.1f}%) - "
                              f"decompressed {fout.tell()/1e6:.0f} MB",
                              end="\r", flush=True)
            print()
        os.replace(tmp_path, gpkg_path) # rinomino il .part in .gpkg
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    size_mb = gpkg_path.stat().st_size / 1e6
    print(f"[DEBUG] tile {quadkey} cached at {gpkg_path} ({size_mb:.0f} MB)")
    return gpkg_path


def read_tile_clipped(gpkg_path, bbox_4326):
    """Read only the buildings intersecting `bbox_4326` from a tile GeoPackage.

    `geopandas.read_file(..., bbox=...)` delegates to GDAL's spatial
    filter, which uses the R-tree index built into every GeoPackage.
    """
    lat_min, lng_min, lat_max, lng_max = bbox_4326
    # geopandas vuole la bbox come (minx, miny, maxx, maxy)
    # read_file usa R-tree del GeoPackage, quindi non legge tutto il file
    gdf = gpd.read_file(
        gpkg_path,
        bbox=(lng_min, lat_min, lng_max, lat_max),
    )
    print(f"[DEBUG]   -> {len(gdf)} buildings intersect the bbox")
    return gdf


def write_output(gdf, output_path, fmt):
    """Write the merged GeoDataFrame as GPKG or GeoJSON."""
    output_path = Path(output_path)
    # se il file esiste lo cancello, così la scrittura riparte pulita
    if output_path.exists():
        output_path.unlink()

    if fmt == "gpkg":
        # singolo layer 'buildings_truth'
        gdf.to_file(output_path, layer="buildings_truth", driver="GPKG")
    elif fmt == "geojson":
        # GeoJSON RFC 7946 richiede EPSG:4326 (obm è già in 4326)
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        raise ValueError(f"unknown format {fmt!r}")
    size_mb = output_path.stat().st_size / 1e6
    print(f"[DEBUG] wrote {len(gdf)} features to {output_path} ({size_mb:.2f} MB)")


def main():
    # configuro il parser degli argomenti da linea di comando
    parser = argparse.ArgumentParser(
        description="Snitcher Stage 3: download OpenBuildingMap footprints "
                    "for a target area."
    )

    src = parser.add_mutually_exclusive_group(required=True)  # --bbox e --from-geotiff sono mutuamente esclusivi
    src.add_argument(
        "--bbox",
        help="EPSG:4326 bbox as 'lat_min,lng_min,lat_max,lng_max'"
    )
    src.add_argument(
        "--from-geotiff",
        help="path to a Stage-1 GeoTIFF; bbox read from its BBOX_EPSG_4326 tag"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"output path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--format", choices=["gpkg", "geojson"], default=None,
        help="output format; if omitted, inferred from --output extension"
    )
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE_DIR),
        help=f"directory for cached OBM tiles (default: {DEFAULT_CACHE_DIR})"
    )
    args = parser.parse_args()

    if args.bbox:
        # bbox esplicita tramite cli
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must be 'lat_min,lng_min,lat_max,lng_max'")
        bbox = tuple(parts)
        print(f"[DEBUG] bbox from CLI: {bbox}")
    else:
        # bbox da tag geotiff
        bbox = bbox_from_geotiff(args.from_geotiff)
        print(f"[DEBUG] bbox from {args.from_geotiff}: {bbox}")

    fmt = None
    if args.format:
        fmt = args.format
    else:
        ext = Path(args.output).suffix.lower()
        # default a gpkg se l'estensione non e' tra quelle riconosciute
        fmt = {".gpkg": "gpkg", ".geojson": "geojson", ".json": "geojson"}.get(ext, "gpkg")
    print(f"[DEBUG] output format: {fmt}")

    # 2. calcolo quali tile OBM coprono la bbox
    quadkeys = bbox_to_quadkeys(bbox)
    print(f"[DEBUG] bbox intersects {len(quadkeys)} OBM tile(s): {quadkeys}")

    # 3. per ogni tile: assicuro che sia in cache, poi filtro spazialmente sulla bbox
    cache_dir = Path(args.cache_dir)
    per_tile = []
    for qk in quadkeys:
        gpkg = ensure_tile_cached(qk, cache_dir) # scarica il tile se non è già in cache
        gdf = read_tile_clipped(gpkg, bbox) # legge solo gli edifici che intersecano la bbox (R-tree dietro le quinte)
        if not gdf.empty:
            gdf = gdf.copy()
            gdf["obm_tile"] = qk
            per_tile.append(gdf)

    if not per_tile:
        print("[WARNING] no buildings found inside the bbox")
        empty = gpd.GeoDataFrame(geometry=[], crs=OBM_CRS)
        write_output(empty, args.output, fmt)
        return

    # concateno i GeoDataFrame di ogni tile in uno solo
    merged = pd.concat(per_tile, ignore_index=True)
    # pd.concat restituisce un DataFrame normale, lo riconverto in GeoDataFrame mantenendo il CRS
    merged = gpd.GeoDataFrame(merged, crs=per_tile[0].crs)

    print(f"[DEBUG] {len(merged)} buildings total across {len(per_tile)} tile(s)")
    print(f"[DEBUG] attributes: {[c for c in merged.columns if c != 'geometry']}")
    if "source_id" in merged.columns:
        source_map = {
            0: "OpenStreetMaps", "0": "OpenStreetMaps",
            1: "Google Buildings", "1": "Google Buildings",
            2: "Microsoft Buildings", "2": "Microsoft Buildings"
        }
        mapped_sources = merged["source_id"].map(source_map).fillna(merged["source_id"])
        counts = mapped_sources.value_counts(dropna=False).to_dict()
        print(f"[DEBUG] source breakdown: {counts}")

    write_output(merged, args.output, fmt)


if __name__ == "__main__":
    main()
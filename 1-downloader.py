"""
Snitcher — Stage 1: Cadastral imagery acquisition.

Downloads a configurable layer from the Italian Cadastre WMS
(https://wms.cartografia.agenziaentrate.gov.it) for a given bbox,
issuing as many tile requests as needed to satisfy a target
"pixels per meter" resolution, then mosaics them into either a PNG
or a georeferenced GeoTIFF.

Input bbox is in EPSG:4326 (lat/lng) but WMS requests are made in
EPSG:25832 (ETRS89 / UTM zone 32N) so that "pixels per meter" maps
cleanly to a metric projection.
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import rasterio
import requests
import time
from PIL import Image
from pyproj import Transformer
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds

# WMS configuration
WMS_URL = "https://wms.cartografia.agenziaentrate.gov.it/inspire/wms/ows01.php"
WMS_VERSION = "1.3.0"
WORKING_CRS = "EPSG:25832"   # ETRS89 / UTM zone 32N copre buona parte d'Italia
INPUT_CRS = "EPSG:4326"      # lat/lng per l'input bbox
MAX_TILE_PX = 2048           # dimensione massima specificata dalla documentazione dell'Agenzia delle Entrate


def bbox_to_working_crs(lat_min, lng_min, lat_max, lng_max):
    """Reproject the input bbox from EPSG:4326 to EPSG:25832 (meters)."""
    # always_xy=True forza l'ordine (lon, lat) indipendetemente dal CRS
    transformer = Transformer.from_crs(INPUT_CRS, WORKING_CRS, always_xy=True)
    minx, miny = transformer.transform(lng_min, lat_min)
    maxx, maxy = transformer.transform(lng_max, lat_max)
    return minx, miny, maxx, maxy


def fetch_tile(session, layer, minx, miny, maxx, maxy, width, height):
    """Issue one WMS GetMap request and return the raw image bytes."""
    params = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "CRS": WORKING_CRS,
        # WMS 1.3.0 + projected CRS: axis order is easting,northing -> natural order
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
    }
    resp = session.get(WMS_URL, params=params, timeout=60)
    resp.raise_for_status()
    # On error, WMS returns a ServiceExceptionReport XML — catch it explicitly
    ctype = resp.headers.get("Content-Type", "")
    if not ctype.startswith("image/"):
        raise RuntimeError(
            f"WMS did not return an image (Content-Type={ctype}). "
            f"Response: {resp.text[:500]}"
        )
    return resp.content


def download_area(bbox_4326, layer, pixels_per_meter, workers):
    """Download the area as a tile mosaic, fetching tiles concurrently.

    Returns: (mosaic_image, bounds_in_working_crs, acquisition_timestamp_iso)
    """
    lat_min, lng_min, lat_max, lng_max = bbox_4326

    # 1. riproietta il bbox di input in coordinate metriche (EPSG:25832).
    minx, miny, maxx, maxy = bbox_to_working_crs(lat_min, lng_min, lat_max, lng_max)
    width_m = maxx - minx
    height_m = maxy - miny
    print(f"[DEBUG] Input bbox (EPSG:4326): "
          f"lat=[{lat_min}, {lat_max}], lng=[{lng_min}, {lng_max}]")
    print(f"[DEBUG] Reprojected bbox ({WORKING_CRS}): "
          f"x=[{minx:.2f}, {maxx:.2f}], y=[{miny:.2f}, {maxy:.2f}]")
    print(f"[DEBUG] Area size: {width_m:.2f} m x {height_m:.2f} m")

    # 2. calcola la risoluzione in pixel e la dimensione totale del mosaico in pixel.
    total_w_px = int(math.ceil(width_m * pixels_per_meter))
    total_h_px = int(math.ceil(height_m * pixels_per_meter))
    print(f"[DEBUG] Target resolution: {pixels_per_meter} px/m "
          f"-> mosaic {total_w_px} x {total_h_px} px")

    # 3. ogni tile deve essere al massimo MAX_TILE_PX x MAX_TILE_PX pixel per rispettare i limiti del WMS
    tile_w = min(MAX_TILE_PX, total_w_px)
    tile_h = min(MAX_TILE_PX, total_h_px)
    n_cols = math.ceil(total_w_px / tile_w)
    n_rows = math.ceil(total_h_px / tile_h)
    print(f"[DEBUG] Tile grid: {n_cols} cols x {n_rows} rows "
          f"({n_cols * n_rows} requests, base tile {tile_w}x{tile_h} px)")

    # Metri per pixel: usato per convertire gli offset in pixel delle tile nelle rispettive bbox geografiche.
    # Usare esattamente width_m / total_w_px garantisce che i bordi delle tile adiacenti coincidano senza scarti.
    mpp_x = width_m / total_w_px
    mpp_y = height_m / total_h_px

    # timestamp per metadata
    acquisition_ts = datetime.now(timezone.utc).isoformat()

    # 4. alloca il canvas del mosaico finale
    mosaic = Image.new("RGBA", (total_w_px, total_h_px))
    session = requests.Session() # usa la stessa connessione HTTP per tutte i tiles

    # calcolo prima tutte le bbox e i pixel rect delle tile, così i worker threads fanno solo la richiesta HTTP senza dover calcolare anche le bbox
    jobs = []
    for row in range(n_rows):
        for col in range(n_cols):
            # limiti in pixel di questa tile all'interno del mosaico (origine in alto a sinistra)
            x0 = col * tile_w
            y0 = row * tile_h
            x1 = min(x0 + tile_w, total_w_px)
            y1 = min(y0 + tile_h, total_h_px)
            w, h = x1 - x0, y1 - y0

            # ora che abbiamo i pixel rect, calcoliamo la bbox geografica corrispondente per questa tile
            # l'asse y delle immagini cresce verso il basso, ma maxy è in alto, quindi invertiamo y0 e y1 per calcolare miny e maxy correttamente.
            t_minx = minx + x0 * mpp_x
            t_maxx = minx + x1 * mpp_x
            t_maxy = maxy - y0 * mpp_y
            t_miny = maxy - y1 * mpp_y

            jobs.append({
                "id": f"r{row}c{col}",
                "x0": x0, "y0": y0, "w": w, "h": h,
                "bbox": (t_minx, t_miny, t_maxx, t_maxy),
            })
            print(f"[DEBUG] Tile {jobs[-1]['id']}: {w}x{h} px, "
                  f"bbox=({t_minx:.2f}, {t_miny:.2f}, {t_maxx:.2f}, {t_maxy:.2f})")

    # 5. scarica tutte le tile in parallelo e incollale nel mosaico finale. Gestiamo i retry in caso di errori temporanei
    total = len(jobs)
    workers = min(workers, total) # non ha senso avere più worker del numero di tile
    print(f"[DEBUG] Downloading {total} tiles with {workers} worker(s)...")

    def fetch_job(job):
        bbox = job["bbox"]
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            try:
                data = fetch_tile(session, layer, *bbox, job["w"], job["h"]) ## *bbox espande la tupla in argomenti separati
                return job, data
            except Exception as e:
                if attempt == max_retries:
                    print(f"[ERROR] Tile {job['id']} permanently failed after {max_retries} attempts: {e}")
                    raise
                
                print(f"[WARNING] Tile {job['id']} failed (attempt {attempt}/{max_retries}): {e}. Retrying in 1s...")
                time.sleep(1)  # aspetto un secondo prima di riprovare

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_job, j) for j in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            job, tile_bytes = future.result()
            tile_img = Image.open(BytesIO(tile_bytes)).convert("RGBA")
            mosaic.paste(tile_img, (job["x0"], job["y0"]))
            print(f"[DEBUG] [{i}/{total}] tile {job['id']} done")

    return mosaic, (minx, miny, maxx, maxy), acquisition_ts


def save_png(mosaic, output_path):
    """Plain PNG, no georeferencing."""
    mosaic.save(output_path, "PNG")
    print(f"[DEBUG] PNG saved to {output_path}")


def save_geotiff(mosaic, output_path, bounds_working, bbox_4326,
                 layer, ppm, acquisition_ts):
    """GeoTIFF with embedded CRS, geotransform and Stage-1 provenance tags."""
    minx, miny, maxx, maxy = bounds_working
    width, height = mosaic.size

    # 6. calcola il geotransform per il GeoTIFF, che mappa i pixel del mosaico alle coordinate reali
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # PIL RGBA -> numpy (H, W, 4) uint8
    arr = np.array(mosaic)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,                 # R, G, B, alpha
        "dtype": "uint8",
        "crs": WORKING_CRS,
        "transform": transform,
        "compress": "lzw",          # lossless, ~3x smaller than uncompressed
        "photometric": "RGB",
    }
    # se l'immagine è abbastanza grande, abilito il tiling interno del GeoTIFF per migliorare le prestazioni di lettura/scrittura e visualizzazione in GIS
    if width >= 256 and height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)

    with rasterio.open(output_path, "w", **profile) as dst:
        # rasterio writes one band at a time; bands are 1-indexed
        for i in range(4):
            dst.write(arr[:, :, i], i + 1)
        dst.colorinterp = (
            ColorInterp.red, ColorInterp.green,
            ColorInterp.blue, ColorInterp.alpha,
        )
        # Stage 1 GeoTIFF metadata è key/value
        dst.update_tags(
            SOURCE_WMS_URL=WMS_URL,
            WMS_VERSION=WMS_VERSION,
            LAYER=layer,
            ACQUISITION_TIMESTAMP=acquisition_ts,
            BBOX_EPSG_4326=",".join(str(v) for v in bbox_4326),
            BBOX_WORKING_CRS=f"{minx},{miny},{maxx},{maxy}",
            WORKING_CRS=WORKING_CRS,
            PIXELS_PER_METER=str(ppm),
            LICENSE="CC BY 4.0",
            ATTRIBUTION="Agenzia delle Entrate",
        )
    print(f"[DEBUG] GeoTIFF saved to {output_path}")
    print(f"[DEBUG]   CRS:       {WORKING_CRS}")
    print(f"[DEBUG]   Transform: {transform}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Italian Cadastre WMS imagery as a tiled mosaic."
    )
    parser.add_argument(
        "--bbox", required=True,
        help="bbox in EPSG:4326 as 'lat_min,lng_min,lat_max,lng_max'"
    )
    parser.add_argument(
        "--layer", default="fabbricati",
        help="WMS layer name (default: fabbricati). "
             "Other options: CP.CadastralParcel, CP.CadastralZoning, acque, strade, ..."
    )
    parser.add_argument(
        "--ppm", type=float, required=True,
        help="resolution in pixels per meter (e.g. 10 = 10 cm/pixel)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="number of concurrent tile downloads (default: 8)"
    )
    parser.add_argument(
        "--format", choices=["geotiff", "png"], default="geotiff",
        help="output format (default: geotiff)"
    )
    parser.add_argument(
        "--output", default=None,
        help="output path (default: cadastre.tif or cadastre.png depending on --format)"
    )
    args = parser.parse_args()

    parts = [float(x) for x in args.bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'lat_min,lng_min,lat_max,lng_max'")
    bbox = tuple(parts)

    output = args.output or ("cadastre.tif" if args.format == "geotiff" else "cadastre.png")
    download_area(
        bbox, args.layer, args.ppm, args.workers
    )

    mosaic, bounds_working, acquisition_ts = download_area(
        bbox, args.layer, args.ppm, args.workers
    )

    if args.format == "geotiff":
        save_geotiff(mosaic, output, bounds_working, bbox,
                     args.layer, args.ppm, acquisition_ts)
    else:
        save_png(mosaic, output)


if __name__ == "__main__":
    main()
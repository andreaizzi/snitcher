#!/usr/bin/env python3
"""
WMS Tile Downloader for Italian Cadastral Cartography
Downloads tiles at maximum resolution and stitches them together
"""

import requests
from PIL import Image
from io import BytesIO
import math
import os
from typing import Tuple, List, Optional
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
WMS_URL = "https://wms.cartografia.agenziaentrate.gov.it/inspire/wms/ows01.php"
LAYER_NAME = "fabbricati"
WMS_VERSION = "1.3.0"
IMAGE_FORMAT = "image/png"
MAX_TILE_SIZE = 2048  # Maximum tile size allowed by the WMS service
MAX_WORKERS = 8  # Number of parallel downloads

BBOX_LATLON = (45.459589,9.160709,45.469627,9.180429)
# duomo: 45.461801,9.179528,45.468664,9.193733
# isernia full: 41.582451,14.211416,41.612555,14.256477
# isernia centro storico: 41.583575,14.216609,41.595130,14.234376
# isernia concezione: 41.590496,14.226458,41.592446,14.229671
# roma tanti edifici adiacenti con cortili: 41.877150,12.473817,41.880873,12.479664
# fornelli: 41.592305,14.132452,41.611656,14.157815
# milano centro molto denso: 45.459589,9.160709,45.469627,9.180429

# Target resolution: pixels per meter (higher = more detail)
# Recommended: 5-20 for good detail. Higher values = larger files
PIXELS_PER_METER = 10

# Output CRS for Milan area: UTM Zone 32N (EPSG:25832)
TARGET_CRS = "EPSG:25832"
SOURCE_CRS = "EPSG:4326"  # WGS84 lat/lon


def latlon_to_utm(
    bbox_latlon: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Convert lat/lon bbox to UTM coordinates
    Returns: (min_x, min_y, max_x, max_y) in meters
    """
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)

    # Transform corner points
    min_x, min_y = transformer.transform(bbox_latlon[1], bbox_latlon[0])
    max_x, max_y = transformer.transform(bbox_latlon[3], bbox_latlon[2])

    print(
        f"Bbox: ({bbox_latlon[0]:.4f}, {bbox_latlon[1]:.4f}) → ({bbox_latlon[2]:.4f}, {bbox_latlon[3]:.4f})"
    )
    print(f"UTM:  ({min_x:.1f}, {min_y:.1f}) → ({max_x:.1f}, {max_y:.1f})")

    return (min_x, min_y, max_x, max_y)


def calculate_tile_grid(
    bbox_utm: Tuple[float, float, float, float], pixels_per_meter: float
) -> Tuple[int, int]:
    """
    Calculate tile grid dimensions
    Returns: (cols, rows)
    """
    width_m = bbox_utm[2] - bbox_utm[0]
    height_m = bbox_utm[3] - bbox_utm[1]

    # Calculate total pixel dimensions needed
    total_width_px = int(width_m * pixels_per_meter)
    total_height_px = int(height_m * pixels_per_meter)

    # Calculate number of tiles needed
    cols = math.ceil(total_width_px / MAX_TILE_SIZE)
    rows = math.ceil(total_height_px / MAX_TILE_SIZE)

    print(f"Area: {width_m:.1f}m × {height_m:.1f}m")
    print(f"Resolution: {total_width_px}×{total_height_px}px ({pixels_per_meter}px/m)")
    print(f"Grid: {cols}×{rows} tiles ({cols * rows} total)")

    return (cols, rows)


def download_tile(
    bbox: Tuple[float, float, float, float], width: int, height: int
) -> Image.Image:
    """
    Download a single WMS tile
    """
    params = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": LAYER_NAME,
        "CRS": TARGET_CRS,
        "BBOX": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": IMAGE_FORMAT,
        "TRANSPARENT": "TRUE",
        "STYLES": "",
    }

    response = requests.get(WMS_URL, params=params, timeout=30)
    response.raise_for_status()

    return Image.open(BytesIO(response.content))


def download_tile_task(
    row: int, col: int, bbox: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, Optional[Image.Image]]:
    """
    Download a single tile (for parallel execution)
    Returns: (row, col, image)
    """
    try:
        tile_image = download_tile(bbox, width, height)
        print(f"  ✓ Tile [{row},{col}]: {tile_image.width}x{tile_image.height}px")
        return (row, col, tile_image)
    except Exception as e:
        print(f"  ✗ Tile [{row},{col}]: {e}")
        return (row, col, None)


def download_tiles(
    bbox_utm: Tuple[float, float, float, float],
    cols: int,
    rows: int,
    pixels_per_meter: float,
) -> Image.Image:
    """
    Download all tiles in parallel and stitch them together
    """
    width_m = bbox_utm[2] - bbox_utm[0]
    height_m = bbox_utm[3] - bbox_utm[1]

    tile_width_m = width_m / cols
    tile_height_m = height_m / rows

    print(f"\nDownloading {cols * rows} tiles (parallel, max {MAX_WORKERS} workers)...")

    # Prepare all tile download tasks
    tasks = []
    for row in range(rows):
        for col in range(cols):
            # Calculate bbox for this tile
            tile_min_x = bbox_utm[0] + (col * tile_width_m)
            tile_max_x = tile_min_x + tile_width_m

            # Y-axis: start from north (max Y) for row 0
            tile_max_y = bbox_utm[3] - (row * tile_height_m)
            tile_min_y = tile_max_y - tile_height_m

            tile_bbox = (tile_min_x, tile_min_y, tile_max_x, tile_max_y)

            # Calculate pixel dimensions
            tile_px_width = int(tile_width_m * pixels_per_meter)
            tile_px_height = int(tile_height_m * pixels_per_meter)

            tasks.append((row, col, tile_bbox, tile_px_width, tile_px_height))

    # Download tiles in parallel
    tiles = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_tile_task, row, col, bbox, w, h): (row, col)
            for row, col, bbox, w, h in tasks
        }

        for future in as_completed(futures):
            row, col, tile = future.result()
            if tile:
                tiles[(row, col)] = tile

    # Calculate canvas dimensions
    col_widths = [0] * cols
    row_heights = [0] * rows

    for (row, col), tile in tiles.items():
        col_widths[col] = max(col_widths[col], tile.width)
        row_heights[row] = max(row_heights[row], tile.height)

    canvas_width = sum(col_widths)
    canvas_height = sum(row_heights)

    print(f"\nStitching {len(tiles)} tiles...")
    print(f"Final canvas: {canvas_width}x{canvas_height}px")

    # Create canvas and paste tiles
    final_image = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))

    for (row, col), tile in sorted(tiles.items()):
        paste_x = sum(col_widths[:col])
        paste_y = sum(row_heights[:row])
        final_image.paste(tile, (paste_x, paste_y))

    return final_image


def main():
    """
    Main execution function
    """
    print("=== WMS Cadastral Downloader ===\n")

    # Convert bbox to UTM
    bbox_utm = latlon_to_utm(BBOX_LATLON)

    # Calculate tile grid
    cols, rows = calculate_tile_grid(bbox_utm, PIXELS_PER_METER)

    # Download and stitch tiles
    final_image = download_tiles(bbox_utm, cols, rows, PIXELS_PER_METER)

    # Save result
    output_filename = "cadastral_map.png"
    final_image.save("test-maps/" + output_filename, "PNG")

    file_size_mb = os.path.getsize("test-maps/" + output_filename) / 1024 / 1024
    print(f"\n✓ Saved: {output_filename}")
    print(f"  Size: {final_image.width}x{final_image.height}px, {file_size_mb:.1f}MB")


if __name__ == "__main__":
    main()

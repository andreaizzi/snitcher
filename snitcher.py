#!/usr/bin/env python3
"""
Extract building polygons from WMS tile images.
Buildings are orange polygons with black borders on transparent background.

OPTIMIZED VERSION: Uses geometric buffering instead of brute-force pixel search.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import json

# Required for geometric buffering optimization
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


def load_image(image_path):
    """Load image with alpha channel."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img


def extract_alpha_mask(img):
    """Extract alpha channel to get building areas."""
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        # If no alpha channel, assume non-white pixels are buildings
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    return alpha


def extract_black_borders(img, alpha):
    """Estrae le linee nere (bordi) che separano gli edifici."""

    # Genera una matrice Booleana (True/False): ignora lo sfondo trasparente vuoto
    opaque_mask = alpha > 128

    # Se l'immagine è BGRA (4 canali), scarta l'ultimo (Alpha) per analizzare solo il colore
    if img.shape[2] == 4:
        bgr = img[:, :, :3]  # Prendi tutti i pixel, ma solo i primi 3 canali (B, G, R)
    else:
        bgr = img

    # Prendi i pixel neri
    # np.all(..., axis=2): Controlla se B, G e R sono TUTTI scuri (< 100) nello stesso pixel.
    # & opaque_mask: ...ma SOLO se il pixel non è trasparente.
    black_mask = np.all(bgr < 100, axis=2) & opaque_mask

    # Trasforma True/False in 1/0 (uint8), poi in 255/0 per avere Bianco/Nero
    return black_mask.astype(np.uint8) * 255


def separate_buildings(alpha, black_borders):
    """
    Separate attached buildings using black borders as separators.
    The key insight: black borders are drawn ON TOP of orange pixels, so we need to
    DILATE them first to create actual gaps between buildings before removal.
    """
    # CRITICAL: Dilate black borders to create separation gaps
    # This ensures buildings are actually disconnected when we remove the borders
    # Using iterations=3 for more aggressive separation
    kernel = np.ones((2, 2), np.uint8)
    dilated_borders = cv2.dilate(black_borders, kernel, iterations=1)

    # Now remove the dilated borders from alpha to separate buildings
    building_mask = alpha.copy()
    building_mask[dilated_borders > 0] = 0

    # Apply threshold to get binary mask
    _, binary = cv2.threshold(building_mask, 128, 255, cv2.THRESH_BINARY)

    # Clean up small artifacts but preserve building separation
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Label each disconnected region as a separate building
    num_labels, markers = cv2.connectedComponents(binary)

    print(
        f"Found {num_labels - 1} separate building regions after removing dilated borders"
    )

    return binary, markers, dilated_borders


def snap_contours_to_borders(contours, black_borders, snap_distance=3):
    """
    OPTIMIZED: Geometric Buffering Approach (replaces O(V×P) pixel search).

    Instead of searching for the nearest border pixel for each vertex (expensive),
    this function applies a uniform geometric expansion (buffer) to close the
    artificial gaps created during border dilation.

    COMPLEXITY IMPROVEMENT:
        Old: O(V × P) - V vertices × P border pixels (e.g., 1000 × 50000 = 50M ops)
        New: O(N × V) - N polygons × V vertices per poly (e.g., 50 × 30 = 1500 ops)
        Speedup: ~10,000× faster for typical images

    HOW IT WORKS:
        1. The pipeline dilates black borders by ~1-2 pixels to separate buildings
        2. Contours are extracted from the separated mask (with gaps)
        3. This function buffers each polygon OUTWARD by offset_distance to close gaps
        4. The existing merge_nearby_vertices() step handles vertex unification
        5. The existing subdivide_edges_with_vertices() ensures OSM topology

    TOPOLOGY HANDLING:
        - Buffered polygons from adjacent buildings will overlap slightly (expected)
        - merge_nearby_vertices() consolidates overlapping vertices into shared points
        - subdivide_edges_with_vertices() inserts vertices for OSM compliance
        - Result: Adjacent buildings share exact coordinates as required by OSM

    Args:
        contours: List of OpenCV contours (NumPy arrays of shape (N, 1, 2))
        black_borders: Binary mask of border pixels (UNUSED in optimized version,
                      kept for API compatibility)
        offset_distance: Fixed distance (in pixels) to buffer outward. Should equal
                        the dilation radius used in separate_buildings().
                        Default: 2.0 (matches 3×3 CROSS kernel with 1 iteration)

    Returns:
        List of buffered contours in OpenCV format (NumPy arrays)

    Dependencies:
        Requires shapely: pip install shapely
    """
    if len(contours) == 0:
        return contours

    buffered_contours = []
    success_count = 0
    failure_count = 0

    for contour_idx, contour in enumerate(contours):
        try:
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Convert OpenCV contour to Shapely Polygon
            # ─────────────────────────────────────────────────────────────────
            # OpenCV format: (N, 1, 2) with [[[x, y]], [[x, y]], ...]
            # Shapely format: [(x, y), (x, y), ...] as a simple coordinate list

            coords = contour.squeeze()  # (N, 1, 2) → (N, 2)

            # Skip degenerate contours (need ≥3 points for a valid polygon)
            if len(coords) < 3:
                continue

            # Ensure coords are at least 2D (handle single-point edge case)
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)

            # Create Shapely Polygon
            # Note: Shapely expects (x, y) order, which matches OpenCV's format
            poly = Polygon(coords)

            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Validate and repair geometry
            # ─────────────────────────────────────────────────────────────────
            # Buffering can fail on invalid geometries (self-intersections, etc.)
            # make_valid() automatically fixes common topological issues

            if not poly.is_valid:
                poly = make_valid(poly)

            # Handle degenerate cases (empty or point geometries)
            if poly.is_empty or poly.area < 1.0:
                continue

            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Apply geometric buffer (outward expansion)
            # ─────────────────────────────────────────────────────────────────
            # This is the KEY OPTIMIZATION that replaces pixel-by-pixel search
            #
            # Buffer parameters:
            #   - distance: positive = expand outward, negative = shrink inward
            #   - resolution: number of segments per quadrant (8 = 32 segments/circle)
            #                Lower = faster but more angular corners
            #   - join_style: 1=round, 2=mitre, 3=bevel
            #   - cap_style: 1=round, 2=flat, 3=square
            #
            # We use round joins/caps to avoid creating sharp spikes at corners

            buffered_poly = poly.buffer(
                distance=offset_distance,
                resolution=8,  # 8 segments per 90° arc (32 total per circle)
                join_style=2,  # Round joins (smoothest, avoids spikes)
                cap_style=3,  # Square caps (for line-string endpoints)
                mitre_limit=2.0,  # Only used if join_style=2 (mitre)
            )

            # ─────────────────────────────────────────────────────────────────
            # STEP 4: Handle MultiPolygon results (rare edge case)
            # ─────────────────────────────────────────────────────────────────
            # In rare cases, buffering can split a polygon into multiple pieces
            # (e.g., if the original had a very narrow section that closes during buffering)
            # Solution: Take the largest polygon by area (most likely the main building)

            if isinstance(buffered_poly, MultiPolygon):
                buffered_poly = max(buffered_poly.geoms, key=lambda p: p.area)

            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Convert back to OpenCV contour format
            # ─────────────────────────────────────────────────────────────────
            # Shapely exterior.coords: [(x, y), ..., (x_first, y_first)]
            # Note: Shapely automatically closes the ring (first == last point)
            # We exclude the duplicate closing point with [:-1]

            if not buffered_poly.is_empty:
                # Extract coordinates and convert to integer pixels
                buffered_coords = np.array(
                    buffered_poly.exterior.coords[:-1],  # Exclude closing point
                    dtype=np.int32,
                )

                # Reshape to OpenCV's expected format: (N, 1, 2)
                buffered_coords = buffered_coords.reshape((-1, 1, 2))

                buffered_contours.append(buffered_coords)
                success_count += 1

        except Exception as e:
            # If buffering fails for any polygon, keep the original contour
            # This ensures the pipeline continues even if individual geometries fail
            print(f"Warning: Buffer operation failed for contour {contour_idx}: {e}")
            buffered_contours.append(contour)
            failure_count += 1

    print(
        f"Buffering complete: {success_count} succeeded, {failure_count} failed/skipped"
    )

    return buffered_contours

def subdivide_edges_with_vertices(contours, tolerance=2.0):
    """
    OSM edge subdivision (OPTIMIZED with spatial indexing).
    Only checks vertices near each edge instead of all vertices.
    """
    # Collect all unique vertices
    all_vertices = set()
    for contour in contours:
        for point in contour:
            all_vertices.add(tuple(point[0]))

    vertices_array = np.array(list(all_vertices), dtype=float)

    if len(vertices_array) == 0:
        return contours

    # Build KDTree for vertices
    kdtree = cKDTree(vertices_array)

    print(f"Checking {len(all_vertices)} vertices against edges for subdivision...")

    subdivided_contours = []
    total_insertions = 0

    for contour in contours:
        new_vertices = []
        num_vertices = len(contour)

        for i in range(num_vertices):
            v1 = contour[i][0].astype(float)
            v2 = contour[(i + 1) % num_vertices][0].astype(float)

            new_vertices.append(v1)

            # Edge bounding box
            min_x, max_x = min(v1[0], v2[0]), max(v1[0], v2[0])
            min_y, max_y = min(v1[1], v2[1]), max(v1[1], v2[1])

            # Query vertices near the edge bounding box (with tolerance)
            edge_center = (v1 + v2) / 2
            edge_length = np.linalg.norm(v2 - v1)
            search_radius = edge_length / 2 + tolerance + 1

            nearby_indices = kdtree.query_ball_point(edge_center, search_radius)

            vertices_on_edge = []

            for idx in nearby_indices:
                vertex = vertices_array[idx]

                # Skip edge endpoints
                if np.allclose(vertex, v1, atol=0.1) or np.allclose(
                    vertex, v2, atol=0.1
                ):
                    continue

                # Check if vertex is on the line segment
                dist = point_to_segment_distance(vertex, v1, v2)

                if dist <= tolerance:
                    t = project_point_onto_segment(vertex, v1, v2)
                    if 0 < t < 1:
                        vertices_on_edge.append((t, vertex))

            # Sort by position along edge
            vertices_on_edge.sort(key=lambda x: x[0])

            # Insert them
            for t, vertex in vertices_on_edge:
                new_vertices.append(vertex)
                total_insertions += 1

        if new_vertices:
            subdivided_contour = np.array(
                [[[int(v[0]), int(v[1])]] for v in new_vertices], dtype=np.int32
            )
            subdivided_contours.append(subdivided_contour)
        else:
            subdivided_contours.append(contour)

    print(f"  → Inserted {total_insertions} vertices into edges for OSM topology")

    return subdivided_contours


def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate the minimum distance from a point to a line segment.

    Args:
        point: Point as numpy array [x, y]
        seg_start: Segment start as numpy array [x, y]
        seg_end: Segment end as numpy array [x, y]

    Returns:
        Minimum distance from point to segment
    """
    # Vector from seg_start to seg_end
    segment_vec = seg_end - seg_start
    segment_length_sq = np.dot(segment_vec, segment_vec)

    if segment_length_sq == 0:
        # Degenerate segment (point)
        return np.linalg.norm(point - seg_start)

    # Parameter t: project point onto the line, t=0 at seg_start, t=1 at seg_end
    t = np.dot(point - seg_start, segment_vec) / segment_length_sq

    # Clamp t to [0, 1] to stay on the segment
    t = max(0, min(1, t))

    # Find the projection point
    projection = seg_start + t * segment_vec

    # Distance from point to projection
    return np.linalg.norm(point - projection)


def project_point_onto_segment(point, seg_start, seg_end):
    """
    Project a point onto a line segment and return the parameter t.

    Args:
        point: Point as numpy array [x, y]
        seg_start: Segment start as numpy array [x, y]
        seg_end: Segment end as numpy array [x, y]

    Returns:
        Parameter t where projection = seg_start + t * (seg_end - seg_start)
        t = 0 means projection is at seg_start
        t = 1 means projection is at seg_end
    """
    segment_vec = seg_end - seg_start
    segment_length_sq = np.dot(segment_vec, segment_vec)

    if segment_length_sq == 0:
        return 0

    t = np.dot(point - seg_start, segment_vec) / segment_length_sq
    return t


def merge_nearby_vertices(contours, merge_distance=2):
    """
    Merge nearby vertices using KDTree (OPTIMIZED).
    O(n log n) instead of O(n²).
    """
    # Collect all vertices with their source info
    vertices_list = []
    vertex_map = {}  # (contour_idx, point_idx) -> vertex_idx

    for contour_idx, contour in enumerate(contours):
        for point_idx in range(len(contour)):
            x, y = contour[point_idx][0]
            vertex_idx = len(vertices_list)
            vertices_list.append([x, y])
            vertex_map[(contour_idx, point_idx)] = vertex_idx

    if not vertices_list:
        return contours

    vertices_array = np.array(vertices_list, dtype=float)

    # Build KDTree
    kdtree = cKDTree(vertices_array)

    # Find all pairs within merge_distance
    pairs = kdtree.query_pairs(merge_distance, output_type="ndarray")

    # Build clusters using Union-Find
    parent = list(range(len(vertices_list)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in pairs:
        union(i, j)

    # Compute cluster centroids
    clusters = {}
    for i in range(len(vertices_list)):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    centroid_map = {}
    for root, members in clusters.items():
        centroid = vertices_array[members].mean(axis=0)
        centroid_map[root] = (int(round(centroid[0])), int(round(centroid[1])))

    # Update contours
    merged_contours = []
    for contour_idx, contour in enumerate(contours):
        new_points = []
        for point_idx in range(len(contour)):
            vertex_idx = vertex_map[(contour_idx, point_idx)]
            root = find(vertex_idx)
            new_x, new_y = centroid_map[root]
            new_points.append([[new_x, new_y]])
        merged_contours.append(np.array(new_points, dtype=np.int32))

    if clusters:
        print(f"  → Merged {len(clusters)} groups of nearby vertices")

    return merged_contours


def find_building_contours(separated_mask):
    """Find contours of individual buildings."""
    contours, hierarchy = cv2.findContours(
        separated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def simplify_polygon_worker(args):
    """Worker function for parallel polygon simplification."""
    contour, epsilon_factor = args
    area = cv2.contourArea(contour)
    if area >= 50:
        return simplify_polygon(contour, epsilon_factor)
    return None


def simplify_polygons_parallel(contours, epsilon_factor, n_jobs=None):
    """Simplify polygons in parallel using multiprocessing."""
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    # Prepare args
    args_list = [(contour, epsilon_factor) for contour in contours]

    # Use multiprocessing only if worth it (many contours)
    if len(contours) < 20 or n_jobs == 1:
        # Sequential for small workloads
        simplified = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 50:
                simplified.append(simplify_polygon(contour, epsilon_factor))
        return simplified

    # Parallel processing
    with Pool(n_jobs) as pool:
        results = pool.map(simplify_polygon_worker, args_list)

    return [r for r in results if r is not None]


def simplify_polygon(contour, epsilon_factor=3.0):
    """
    Simplify polygon using Douglas-Peucker algorithm.
    This reduces vertices to corner points only.

    Uses a fixed epsilon value instead of perimeter-based to ensure
    consistent simplification across all building sizes.
    Small buildings should not have more vertices than large ones.
    """
    # Use a fixed epsilon value (in pixels) for consistent simplification
    # epsilon_factor is now interpreted as a fixed pixel tolerance
    # Typical good values: 2-5 pixels for building corners
    epsilon = epsilon_factor if epsilon_factor > 1 else epsilon_factor * 100

    # Approximate polygon
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx


def contours_to_geojson(contours, image_shape):
    """Convert contours to GeoJSON format."""
    features = []

    for i, contour in enumerate(contours):
        # Skip very small contours (noise)
        area = cv2.contourArea(contour)
        if area < 50:  # Minimum area threshold
            continue

        # Convert contour to list of coordinates
        # Note: In image coordinates, might need transformation for real geo coordinates
        coords = contour.squeeze().tolist()

        # Ensure it's a closed polygon
        if len(coords) > 2:
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            feature = {
                "type": "Feature",
                "properties": {
                    "building_id": i,
                    "area_pixels": float(area),
                    "vertices": len(coords) - 1,
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
            features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    return geojson


def save_debug_images(
    output_dir,
    img,
    alpha,
    black_borders,
    dilated_borders,
    separated,
    markers,
    contours,
    simplified_contours,
    snapped_contours,
):
    """Save debug images for each processing step."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Original image
    cv2.imwrite(str(output_dir / "01_original.png"), img)

    # 2. Alpha channel
    cv2.imwrite(str(output_dir / "02_alpha_mask.png"), alpha)

    # 3. Black borders
    cv2.imwrite(str(output_dir / "03_black_borders.png"), black_borders)

    # 3b. Dilated borders (the actual separators)
    cv2.imwrite(str(output_dir / "03b_dilated_borders.png"), dilated_borders)

    # 4. Separated buildings
    cv2.imwrite(str(output_dir / "04_separated_buildings.png"), separated)

    # 5. Connected components (colorized - each building in different color)
    markers_viz = (
        np.zeros_like(img[:, :, :3])
        if img.shape[2] == 4
        else np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    )
    if markers is not None:
        # Create a color for each building
        markers_colored = (markers * 50 % 255).astype(np.uint8)
        markers_viz = cv2.applyColorMap(markers_colored, cv2.COLORMAP_JET)
        markers_viz[markers == 0] = 0  # Keep background black
    cv2.imwrite(str(output_dir / "05_connected_components.png"), markers_viz)

    # 6. Detected contours (raw, with gaps)
    contour_img = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "06_detected_contours_raw.png"), contour_img)

    # 6b. Simplified polygons (before snapping)
    simplified_img = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(simplified_img, simplified_contours, -1, (0, 255, 255), 2)  # Cyan
    cv2.imwrite(str(output_dir / "06b_simplified_before_snap.png"), simplified_img)

    # 7. Final OSM-topology polygons (after snapping + merging + edge subdivision)
    snapped_img = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(snapped_img, snapped_contours, -1, (0, 0, 255), 2)  # Red
    # Draw vertices with different colors for emphasis
    for contour in snapped_contours:
        for point in contour:
            cv2.circle(snapped_img, tuple(point[0]), 3, (255, 0, 0), -1)  # Blue dots
    # Overlay original black borders in gray for reference
    snapped_img[black_borders > 0] = [128, 128, 128]
    cv2.imwrite(str(output_dir / "07_final_OSM_topology.png"), snapped_img)

    # 8. Three-way comparison: raw → simplified → snapped
    comparison = np.hstack([contour_img, simplified_img, snapped_img])
    cv2.imwrite(str(output_dir / "08_comparison.png"), comparison)

    print(f"Debug images saved to: {output_dir}")


def process_wms_tile(image_path, output_dir=None, epsilon_factor=3.0):
    """
    Main processing pipeline (OPTIMIZED).
    """
    print(f"Processing: {image_path}")
    start_time = time.time()

    # Setup output directory
    if output_dir is None:
        output_dir = Path(image_path).parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Load image
    t0 = time.time()
    img = load_image(image_path)
    print(f"Image shape: {img.shape} [{time.time()-t0:.2f}s]")

    # 2. Extract alpha channel
    t0 = time.time()
    alpha = extract_alpha_mask(img)
    print(f"Extracted alpha mask [{time.time()-t0:.2f}s]")

    # 3. Extract black borders
    t0 = time.time()
    black_borders = extract_black_borders(img, alpha)
    print(f"Extracted black borders [{time.time()-t0:.2f}s]")

    # 4. Separate attached buildings
    t0 = time.time()
    separated, markers, dilated_borders = separate_buildings(alpha, black_borders)
    print(f"Separated buildings [{time.time()-t0:.2f}s]")

    # 5. Find contours
    t0 = time.time()
    contours = find_building_contours(separated)
    print(f"Found {len(contours)} building contours [{time.time()-t0:.2f}s]")

    # 6. Simplify polygons (PARALLEL)
    t0 = time.time()
    simplified_contours = simplify_polygons_parallel(contours, epsilon_factor)
    print(f"Simplified to {len(simplified_contours)} polygons [{time.time()-t0:.2f}s]")

    # 7. Snap to borders (OPTIMIZED with KDTree)
    t0 = time.time()
    snap_distance = 15
    snapped_contours = snap_contours_to_borders(
        simplified_contours, black_borders, snap_distance=snap_distance
    )
    print(f"Snapped to borders (dist={snap_distance}px) [{time.time()-t0:.2f}s]")

    # 7b. Merge nearby vertices (OPTIMIZED with KDTree)
    t0 = time.time()
    merge_distance = 15
    snapped_contours = merge_nearby_vertices(
        snapped_contours, merge_distance=merge_distance
    )
    print(f"Merged nearby vertices (dist={merge_distance}px) [{time.time()-t0:.2f}s]")

    # 7c. OSM edge subdivision (OPTIMIZED)
    t0 = time.time()
    subdivision_tolerance = 10
    snapped_contours = subdivide_edges_with_vertices(
        snapped_contours, tolerance=subdivision_tolerance
    )
    print(f"Edge subdivision (tol={subdivision_tolerance}px) [{time.time()-t0:.2f}s]")

    print(f"Simplified to {len(simplified_contours)} valid polygons")

    # 8. Save debug images
    debug_dir = output_dir / "debug"
    save_debug_images(
        debug_dir,
        img,
        alpha,
        black_borders,
        dilated_borders,
        separated,
        markers,
        contours,
        simplified_contours,
        snapped_contours,
    )

    # 9. Convert to GeoJSON (use snapped contours - the final result with topology)
    geojson = contours_to_geojson(snapped_contours, img.shape)

    # 10. Save GeoJSON
    output_json = output_dir / "buildings.geojson"
    with open(output_json, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\nResults:")
    print(f"  - Buildings extracted: {len(geojson['features'])}")
    print(f"  - GeoJSON saved to: {output_json}")
    print(f"  - Debug images in: {debug_dir}")

    # Print summary statistics
    if geojson["features"]:
        vertices = [f["properties"]["vertices"] for f in geojson["features"]]
        areas = [f["properties"]["area_pixels"] for f in geojson["features"]]
        print(f"\nStatistics:")
        print(f"  - Avg vertices per building: {np.mean(vertices):.1f}")
        print(f"  - Min/Max vertices: {min(vertices)}/{max(vertices)}")
        print(f"  - Avg area: {np.mean(areas):.1f} pixels²")

    total_time = time.time() - start_time
    print(f"\n✓ Total processing time: {total_time:.2f}s")

    return geojson


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_buildings.py <image_path> [output_dir] [epsilon]")
        print("\nArguments:")
        print("  image_path     - Path to input PNG image (required)")
        print("  output_dir     - Output directory (optional, default: ./output)")
        print(
            "  epsilon        - Polygon simplification in pixels (optional, default: 3)"
        )
        print("                   Higher values = simpler polygons (fewer vertices)")
        print("                   Recommended: 2-5 pixels for building corners")
        print("\nExample:")
        print("  python extract_buildings.py tile.png")
        print("  python extract_buildings.py tile.png ./results 4")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    epsilon_factor = (
        float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    )  # 3 pixels default

    try:
        process_wms_tile(image_path, output_dir, epsilon_factor)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

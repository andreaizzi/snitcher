#!/usr/bin/env python3
"""
Extract building polygons from WMS tile images.
Buildings are orange polygons with black borders on transparent background.

OPTIMIZED VERSION: Uses geometric buffering instead of brute-force pixel search.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


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
    Separa gli edifici attaccati usando i bordi neri come separatori.
    Usa una dilatazione a CROCE per preservare meglio gli angoli retti.
    """

    # Usiamo MORPH_CROSS dimensione 3x3.
    # Forma:
    #   0 1 0
    #   1 1 1
    #   0 1 0
    # Questo espande il bordo nero nelle direzioni principali ma NON riempie gli angoli,
    # riducendo l'erosione della forma quadrata degli edifici.
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    dilated_borders = cv2.dilate(black_borders, kernel, iterations=0)

    # Copiamo la maschera degli edifici e impostiamo a 0 (nero) i pixel
    # dove passa il nostro bordo dilatato.
    building_mask = alpha.copy()
    building_mask[dilated_borders > 0] = 0

    # Puliamo l'immagine risultante: tutto ciò che non è nero diventa bianco puro.
    _, binary = cv2.threshold(building_mask, 128, 255, cv2.THRESH_BINARY)

    # Rimuove rumore puntiforme (piccoli pixel isolati rimasti dopo il taglio)
    # Qui usiamo ancora un kernel 3x3 classico perché è più efficace per pulire il rumore.
    kernel_clean = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    # Conta le isole separate. Connettività 4 (orizzontale/verticale) per evitare di collegare
    # diagonalmente edifici adiacenti.
    num_labels, markers = cv2.connectedComponents(binary, connectivity=4)

    print(
        f"Found {num_labels - 1} separate building regions after removing dilated borders"
    )

    return binary, markers, dilated_borders


def subdivide_edges_with_vertices(contours, tolerance=2.0):
    """
    OSM Topology: Insert vertices from adjacent polygons into edges.
    OPTIMIZED with spatial indexing for performance.
    """
    from scipy.spatial import cKDTree

    # Build vertex index: map each vertex to its source buildings
    vertex_to_buildings = {}
    building_vertices = []  # List of (building_idx, vertex_array)

    for contour_idx, contour in enumerate(contours):
        for point in contour:
            vertex = tuple(point[0])
            if vertex not in vertex_to_buildings:
                vertex_to_buildings[vertex] = []
                building_vertices.append(np.array(vertex, dtype=float))
            vertex_to_buildings[vertex].append(contour_idx)

    # Build KD-tree for fast spatial queries
    if len(building_vertices) == 0:
        return contours

    vertex_array = np.array(building_vertices)
    kdtree = cKDTree(vertex_array)

    subdivided_contours = []
    total_insertions = 0

    for contour_idx, contour in enumerate(contours):
        new_vertices = []
        num_vertices = len(contour)

        for i in range(num_vertices):
            v1 = np.array(contour[i][0], dtype=float)
            v2 = np.array(contour[(i + 1) % num_vertices][0], dtype=float)

            new_vertices.append(v1)

            # Find vertices near this edge using KD-tree
            # Query along the edge midpoint with expanded search radius
            edge_midpoint = (v1 + v2) / 2
            edge_length = np.linalg.norm(v2 - v1)
            search_radius = edge_length / 2 + tolerance

            # Find candidate vertices near this edge
            candidate_indices = kdtree.query_ball_point(edge_midpoint, search_radius)

            vertices_on_edge = []

            for idx in candidate_indices:
                vertex = vertex_array[idx]
                vertex_tuple = tuple(vertex.astype(int))

                # Skip if this vertex belongs to current building
                if contour_idx in vertex_to_buildings.get(vertex_tuple, []):
                    continue

                # Skip if vertex is an endpoint
                if np.allclose(vertex, v1, atol=0.1) or np.allclose(
                    vertex, v2, atol=0.1
                ):
                    continue

                # Check if vertex lies on edge
                dist = point_to_segment_distance(vertex, v1, v2)
                if dist <= tolerance:
                    t = project_point_onto_segment(vertex, v1, v2)
                    if 0 < t < 1:
                        vertices_on_edge.append((t, vertex))

            # Sort and insert
            vertices_on_edge.sort(key=lambda x: x[0])
            for t, vertex_to_insert in vertices_on_edge:
                new_vertices.append(vertex_to_insert)
                total_insertions += 1

        if len(new_vertices) > 0:
            subdivided_contour = np.array(
                [[[int(v[0]), int(v[1])]] for v in new_vertices], dtype=np.int32
            )
            subdivided_contours.append(subdivided_contour)
        else:
            subdivided_contours.append(contour)

    if total_insertions > 0:
        print(f"  → Inserted {total_insertions} vertices into edges (optimized)")

    return subdivided_contours


def point_to_segment_distance(point, segment_start, segment_end):
    """Calculate the shortest distance from a point to a line segment."""
    # Vector from segment_start to segment_end
    segment_vec = segment_end - segment_start
    segment_length_sq = np.dot(segment_vec, segment_vec)

    if segment_length_sq < 1e-10:  # Degenerate segment (point)
        return np.linalg.norm(point - segment_start)

    # Project point onto the line defined by the segment
    # t = 0 means point projects to segment_start
    # t = 1 means point projects to segment_end
    t = np.dot(point - segment_start, segment_vec) / segment_length_sq

    # Clamp t to [0, 1] to stay on the segment
    t = np.clip(t, 0, 1)

    # Find the closest point on the segment
    closest_point = segment_start + t * segment_vec

    # Return distance to closest point
    return np.linalg.norm(point - closest_point)


def project_point_onto_segment(point, segment_start, segment_end):
    """
    Project a point onto a line segment and return the parameter t.
    t = 0 means the projection is at segment_start
    t = 1 means the projection is at segment_end
    t in (0, 1) means the projection is between the endpoints
    """
    segment_vec = segment_end - segment_start
    segment_length_sq = np.dot(segment_vec, segment_vec)

    if segment_length_sq < 1e-10:  # Degenerate segment
        return 0

    t = np.dot(point - segment_start, segment_vec) / segment_length_sq
    return t


def merge_nearby_vertices(contours, merge_distance=5):
    """
    Merge nearby vertices so adjacent buildings share exact coordinates.
    OPTIMIZED with spatial indexing.
    """
    from scipy.spatial import cKDTree

    # Collect all vertices with their source information
    all_vertices = []
    vertex_info = []  # (x, y, contour_idx, point_idx)

    for contour_idx, contour in enumerate(contours):
        for point_idx, point in enumerate(contour):
            x, y = point[0]
            all_vertices.append([float(x), float(y)])
            vertex_info.append((x, y, contour_idx, point_idx))

    if len(all_vertices) == 0:
        return contours

    # Build KD-tree
    vertex_array = np.array(all_vertices)
    kdtree = cKDTree(vertex_array)

    # Find all pairs within merge_distance
    pairs = kdtree.query_pairs(merge_distance)

    # Build clusters using union-find
    parent = list(range(len(all_vertices)))

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

    # Group vertices by cluster
    clusters = {}
    for i in range(len(all_vertices)):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Calculate average position for each cluster
    vertex_mapping = {}
    for cluster_indices in clusters.values():
        if len(cluster_indices) > 1:
            avg_x = int(np.mean([all_vertices[i][0] for i in cluster_indices]))
            avg_y = int(np.mean([all_vertices[i][1] for i in cluster_indices]))
            for idx in cluster_indices:
                vertex_mapping[idx] = (avg_x, avg_y)

    # Update contours
    merged_contours = []
    for contour in contours:
        merged_contours.append(contour.copy())

    vertex_idx = 0
    for contour_idx, contour in enumerate(contours):
        for point_idx in range(len(contour)):
            if vertex_idx in vertex_mapping:
                new_x, new_y = vertex_mapping[vertex_idx]
                merged_contours[contour_idx][point_idx][0][0] = new_x
                merged_contours[contour_idx][point_idx][0][1] = new_y
            vertex_idx += 1

    num_merged = len(vertex_mapping)
    if num_merged > 0:
        print(f"  → Merged {num_merged} vertices into {len(clusters)} clusters")

    return merged_contours


def find_building_contours(separated_mask):
    """
    Find contours of individual buildings, including holes (courtyards).
    Uses RETR_CCOMP to get 2-level hierarchy: outer boundaries and holes.

    Returns:
        contours: List of all contours
        hierarchy: Array describing parent-child relationships
    """
    contours, hierarchy = cv2.findContours(
        separated_mask,
        cv2.RETR_CCOMP,  # 2-level: outer boundaries and holes
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours, hierarchy


def group_contours_with_holes(contours, hierarchy):
    """
    Group outer contours with their holes using OpenCV hierarchy.

    Hierarchy format: [Next, Previous, First_Child, Parent]
    - Parent = -1: outer boundary
    - Parent >= 0: hole inside parent contour

    Args:
        contours: List of all contours from cv2.findContours
        hierarchy: Hierarchy array from cv2.findContours

    Returns:
        List of dicts: [{'outer': contour, 'holes': [hole1, hole2, ...]}, ...]
    """
    if hierarchy is None or len(contours) == 0:
        return [{"outer": c, "holes": []} for c in contours]

    hierarchy = hierarchy[0]  # Remove extra dimension

    # Build parent-child relationships
    buildings = []
    processed = set()

    for i, contour in enumerate(contours):
        # Skip if already processed as a hole
        if i in processed:
            continue

        # Check if this is an outer boundary (parent = -1)
        parent_idx = hierarchy[i][3]
        if parent_idx == -1:
            # This is an outer boundary
            building = {"outer": contour, "holes": []}

            # Find all holes (children with this contour as parent)
            for j, h in enumerate(hierarchy):
                if h[3] == i:  # Parent is current contour
                    building["holes"].append(contours[j])
                    processed.add(j)

            buildings.append(building)

    return buildings


def simplify_polygon(contour, epsilon_factor=3.0):
    """
    Simplify polygon to corner points using Douglas-Peucker algorithm.

    For buildings, we want to preserve right angles and corners while
    removing unnecessary intermediate points along straight edges.

    Args:
        contour: OpenCV contour
        epsilon_factor: Simplification tolerance in pixels (higher = simpler)

    Returns:
        Simplified contour
    """
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)

    # Epsilon: allowed deviation in pixels
    # Typical values: 2-5 pixels for buildings
    epsilon = epsilon_factor

    # Apply Douglas-Peucker simplification
    simplified = cv2.approxPolyDP(contour, epsilon, True)

    return simplified


def transform_pixel_to_geographic(x, y, image_width, image_height, bbox):
    """
    Transform pixel coordinates to WGS84 lat/lon.

    Args:
        x, y: Pixel coordinates
        image_width, image_height: Image dimensions
        bbox: (lat_min, lon_min, lat_max, lon_max)

    Returns:
        (lon, lat) in WGS84
    """
    lat_min, lon_min, lat_max, lon_max = bbox

    # x maps to longitude
    lon = lon_min + (x / image_width) * (lon_max - lon_min)

    # y maps to latitude (inverted: y=0 is top, lat_max is north)
    lat = lat_max - (y / image_height) * (lat_max - lat_min)

    return (lon, lat)


def contours_to_geojson(buildings, image_shape, bbox=None):
    """
    Convert building contours (with holes) to GeoJSON format.

    Args:
        buildings: List of dicts with 'outer' contour and 'holes' list
        image_shape: Shape of the image (height, width, channels)
        bbox: Optional (lat_min, lon_min, lat_max, lon_max) for WGS84 transformation

    Returns:
        GeoJSON FeatureCollection with proper coordinates
    """
    features = []

    image_height, image_width = image_shape[:2]
    use_geographic = bbox is not None

    if use_geographic:
        print(f"Transforming to WGS84 using bbox: {bbox}")

    for i, building in enumerate(buildings):
        outer_contour = building["outer"]
        holes = building.get("holes", [])

        # Calculate area of outer boundary
        area = cv2.contourArea(outer_contour)
        if area < 50:  # Skip very small buildings
            continue

        # Convert outer contour to coordinates
        outer_coords = outer_contour.squeeze().tolist()

        # Ensure it's a valid list
        if not isinstance(outer_coords, list):
            continue

        # Make sure we have enough points
        if len(outer_coords) < 3:
            continue

        # Transform to geographic if bbox provided
        if use_geographic:
            outer_coords = [
                list(
                    transform_pixel_to_geographic(x, y, image_width, image_height, bbox)
                )
                for x, y in outer_coords
            ]

        # Ensure closed polygon
        if outer_coords[0] != outer_coords[-1]:
            outer_coords.append(outer_coords[0])

        # Start with outer ring
        coordinate_arrays = [outer_coords]

        # Add holes (inner rings)
        for hole in holes:
            hole_coords = hole.squeeze().tolist()

            # Validate hole
            if isinstance(hole_coords, list) and len(hole_coords) >= 3:
                # Transform to geographic if bbox provided
                if use_geographic:
                    hole_coords = [
                        list(
                            transform_pixel_to_geographic(
                                x, y, image_width, image_height, bbox
                            )
                        )
                        for x, y in hole_coords
                    ]

                # Ensure closed
                if hole_coords[0] != hole_coords[-1]:
                    hole_coords.append(hole_coords[0])

                coordinate_arrays.append(hole_coords)

        # Calculate total vertex count (outer + all holes)
        total_vertices = sum(len(coords) - 1 for coords in coordinate_arrays)

        # Build properties
        properties = {
            "building_id": i,
            "vertices": total_vertices,
            "has_holes": len(holes) > 0,
            "hole_count": len(holes),
        }

        # Add area - in pixels or approximate m² if geographic
        if use_geographic:
            # Rough approximation for area in m² (more accurate would need proper projection)
            # At equator: 1 degree ≈ 111km
            lat_center = (bbox[0] + bbox[2]) / 2
            meters_per_degree_lat = 111320
            meters_per_degree_lon = 111320 * np.cos(np.radians(lat_center))

            lat_span = bbox[2] - bbox[0]
            lon_span = bbox[3] - bbox[1]

            meters_height = lat_span * meters_per_degree_lat
            meters_width = lon_span * meters_per_degree_lon

            pixel_to_m2 = (meters_width / image_width) * (meters_height / image_height)
            area_m2 = area * pixel_to_m2

            properties["area_m2"] = float(area_m2)
        else:
            properties["area_pixels"] = float(area)

        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": {"type": "Polygon", "coordinates": coordinate_arrays},
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    return geojson


def process_wms_tile(image_path, output_dir=None, epsilon_factor=3.0, bbox=None):
    """
    Main processing pipeline for extracting building polygons.
    Args:
        image_path: Path to input PNG image
        output_dir: Directory for output files (default: same as input)
        epsilon_factor: Polygon simplification tolerance in pixels (default: 3.0)
                       Higher values = simpler polygons with fewer vertices
                       Typical range: 2-5 pixels

        bbox: Optional bounding box as (lat_min, lon_min, lat_max, lon_max) in WGS84
              If provided, outputs geographic coordinates; otherwise pixel coordinates
    """
    print(f"Processing: {image_path}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(image_path).parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Setup debug directory
    # unique folder name for each execution based on time
    debug_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load image
    img = load_image(image_path)
    print(f"Image shape: {img.shape}")
    cv2.imwrite(str(debug_dir / "01_original.png"), img)
    print(f"Saved: 01_original.png")

    # 2. Extract alpha channel
    alpha = extract_alpha_mask(img)
    print("Extracted alpha mask")
    cv2.imwrite(str(debug_dir / "02_alpha_mask.png"), alpha)
    print(f"Saved: 02_alpha_mask.png")

    # 3. Extract black borders
    black_borders = extract_black_borders(img, alpha)
    print("Extracted black borders")
    cv2.imwrite(str(debug_dir / "03_black_borders.png"), black_borders)
    print(f"Saved: 03_black_borders.png")

    # 4. Separate attached buildings
    separated, markers, dilated_borders = separate_buildings(alpha, black_borders)
    print("Separated buildings using black borders")
    cv2.imwrite(str(debug_dir / "03b_dilated_borders.png"), dilated_borders)
    cv2.imwrite(str(debug_dir / "04_separated_buildings.png"), separated)
    # Colorized connected components
    markers_viz = (
        np.zeros_like(img[:, :, :3])
        if img.shape[2] == 4
        else np.zeros((*img.shape[:2], 3), dtype=np.uint8)
    )
    if markers is not None:
        markers_colored = (markers * 50 % 255).astype(np.uint8)
        markers_viz = cv2.applyColorMap(markers_colored, cv2.COLORMAP_JET)
        markers_viz[markers == 0] = 0
    cv2.imwrite(str(debug_dir / "05_connected_components.png"), markers_viz)
    print(
        f"Saved: 03b_dilated_borders.png, 04_separated_buildings.png, 05_connected_components.png"
    )

    # 5. Find contours including holes (courtyards)
    all_contours, hierarchy = find_building_contours(markers)
    print(f"Found {len(all_contours)} total contours")
    contour_img = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, all_contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(debug_dir / "06_detected_contours_raw.png"), contour_img)
    print(f"Saved: 06_detected_contours_raw.png")

    # 5b. Group contours into buildings with holes
    buildings_raw = group_contours_with_holes(all_contours, hierarchy)
    print(f"Grouped into {len(buildings_raw)} buildings (some may have holes)")

    # Count buildings with holes
    buildings_with_holes = sum(1 for b in buildings_raw if len(b.get("holes", [])) > 0)
    if buildings_with_holes > 0:
        print(f"  → {buildings_with_holes} buildings have courtyards/holes")

    # 6. Simplify polygons to corner points (both outer and holes)
    buildings_simplified = []
    for building in buildings_raw:
        outer_area = cv2.contourArea(building["outer"])
        if outer_area < 50:
            continue

        simplified_building = {
            "outer": simplify_polygon(building["outer"], epsilon_factor),
            "holes": [],
        }

        # Simplify holes too
        for hole in building.get("holes", []):
            hole_area = cv2.contourArea(hole)
            if hole_area >= 20:  # Minimum hole size
                simplified_building["holes"].append(
                    simplify_polygon(hole, epsilon_factor)
                )

        buildings_simplified.append(simplified_building)

    print(f"Simplified to {len(buildings_simplified)} valid buildings")

    # 7. Flatten buildings to contour list for topology operations
    # We need to apply snapping/merging/subdivision to ALL contours (outer + holes)
    all_simplified_contours = []
    building_indices = []  # Track which building each contour belongs to

    for building_idx, building in enumerate(buildings_simplified):
        all_simplified_contours.append(building["outer"])
        building_indices.append((building_idx, "outer"))

        for hole_idx, hole in enumerate(building.get("holes", [])):
            all_simplified_contours.append(hole)
            building_indices.append((building_idx, ("hole", hole_idx)))

    # save images with vertices and borders for comparison
    simplified_viz = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(
        simplified_viz, all_simplified_contours, -1, (0, 255, 255), 2
    )  # Cyan
    cv2.imwrite(str(debug_dir / "06b_simplified_before_snap.png"), simplified_viz)
    print(f"Saved: 06b_simplified_before_snap.png")

    # 7b. Merge nearby vertices so adjacent buildings share exact same coordinates
    # After buffering, vertices from different buildings near the same border
    # may be close but not identical. This merges them into shared vertices.
    merge_distance = 6  # Should be slightly larger than typical black border thickness
    snapped_contours = merge_nearby_vertices(
        all_simplified_contours, merge_distance=merge_distance
    )
    print(
        f"Merged nearby vertices for shared boundaries (merge_distance={merge_distance}px)"
    )
    # Save after vertex merging
    merged_viz = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(merged_viz, snapped_contours, -1, (0, 0, 255), 2)
    for contour in snapped_contours:
        for point in contour:
            cv2.circle(merged_viz, tuple(point[0]), 3, (255, 0, 0), -1)
    merged_viz[black_borders > 0] = [128, 128, 128]
    cv2.imwrite(str(debug_dir / "07b_after_merging.png"), merged_viz)
    print(f"Saved: 07b_after_merging.png")

    # 7c. OSM TOPOLOGY: Subdivide edges with vertices from adjacent polygons
    # If a vertex from polygon P2 lies on an edge of polygon P1, insert it into that edge.
    # This is CRITICAL for OpenStreetMap compliance.
    subdivision_tolerance = 8.0  # Maximum distance to consider vertex "on" an edge
    snapped_contours = subdivide_edges_with_vertices(
        snapped_contours, tolerance=subdivision_tolerance
    )
    print(f"Subdivided edges for OSM topology (tolerance={subdivision_tolerance}px)")

    # 7d. Reconstruct buildings from flattened contours
    buildings_final = []
    for building_idx in range(len(buildings_simplified)):
        building_final = {"outer": None, "holes": []}

        # Find all contours belonging to this building
        for contour_idx, (bid, contour_type) in enumerate(building_indices):
            if bid == building_idx:
                if contour_type == "outer":
                    building_final["outer"] = snapped_contours[contour_idx]
                elif isinstance(contour_type, tuple) and contour_type[0] == "hole":
                    building_final["holes"].append(snapped_contours[contour_idx])

        if building_final["outer"] is not None:
            buildings_final.append(building_final)

    print(f"Final topology: {len(buildings_final)} buildings")

    # Save final OSM topology
    final_viz = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(final_viz, snapped_contours, -1, (0, 0, 255), 2)
    for contour in snapped_contours:
        for point in contour:
            cv2.circle(final_viz, tuple(point[0]), 3, (255, 0, 0), -1)
    final_viz[black_borders > 0] = [128, 128, 128]
    cv2.imwrite(str(debug_dir / "07_final_OSM_topology.png"), final_viz)
    print(f"Saved: 07_final_OSM_topology.png")
    print(f"Final polygon count: {len(snapped_contours)}")

    snapped_img = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    for building in buildings_final:
        # Draw outer boundary in red
        cv2.drawContours(snapped_img, [building["outer"]], -1, (0, 0, 255), 2)
        # Draw holes in magenta
        for hole in building.get("holes", []):
            cv2.drawContours(snapped_img, [hole], -1, (255, 0, 255), 2)

        # Draw vertices - blue for outer, green for holes
        for point in building["outer"]:
            cv2.circle(snapped_img, tuple(point[0]), 3, (255, 0, 0), -1)
        for hole in building.get("holes", []):
            for point in hole:
                cv2.circle(snapped_img, tuple(point[0]), 2, (0, 255, 0), -1)

    # Overlay original black borders in gray for reference
    snapped_img[black_borders > 0] = [128, 128, 128]
    cv2.imwrite(str(debug_dir / "08_final_OSM_topology.png"), snapped_img)

    # 9. Convert to GeoJSON (buildings with holes)
    geojson = contours_to_geojson(buildings_final, img.shape, bbox=bbox)

    # 10. Save GeoJSON
    output_json = debug_dir / "buildings.geojson"
    with open(output_json, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"\nResults:")
    print(f"  - Buildings extracted: {len(geojson['features'])}")
    print(f"  - GeoJSON saved to: {output_json}")
    print(f"  - Debug images in: {debug_dir}")

    # Print summary statistics
    if geojson["features"]:
        vertices = [f["properties"]["vertices"] for f in geojson["features"]]

        # Check if we have geographic or pixel coordinates
        if bbox is not None:
            areas = [f["properties"]["area_m2"] for f in geojson["features"]]
            area_unit = "m²"
        else:
            areas = [f["properties"]["area_pixels"] for f in geojson["features"]]
            area_unit = "pixels²"

        print(f"\nStatistics:")
        print(f"  - Avg vertices per building: {np.mean(vertices):.1f}")
        print(f"  - Min/Max vertices: {min(vertices)}/{max(vertices)}")
        print(f"  - Avg area: {np.mean(areas):.1f} {area_unit}")

        if bbox is not None:
            print(f"  - Coordinates: WGS84 (lat/lon)")
        else:
            print(f"  - Coordinates: Pixel (image space)")

    return geojson


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python extract_buildings.py <image_path> [output_dir] [epsilon] [bbox]"
        )
        print("\nArguments:")
        print("  image_path     - Path to input PNG image (required)")
        print("  output_dir     - Output directory (optional, default: ./output)")
        print(
            "  epsilon        - Polygon simplification in pixels (optional, default: 3)"
        )
        print("                   Higher values = simpler polygons (fewer vertices)")
        print("                   Recommended: 2-5 pixels for building corners")
        print(
            "  bbox           - Bounding box as 'lat_min,lon_min,lat_max,lon_max' (optional)"
        )
        print("                   Format: South,West,North,East in WGS84 (EPSG:4326)")
        print("                   If provided, outputs geographic coordinates")
        print("\nExamples:")
        print("  python extract_buildings.py tile.png")
        print("  python extract_buildings.py tile.png ./results 4")

        print(
            "  python extract_buildings.py tile.png ./results 5 '41.890,12.492,41.893,12.495'"
        )
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    epsilon_factor = 3.0

    # Parse bbox if provided
    bbox = None
    if len(sys.argv) > 4:
        try:
            bbox_str = sys.argv[4]
            bbox_parts = [float(x.strip()) for x in bbox_str.split(",")]
            if len(bbox_parts) != 4:
                raise ValueError("bbox must have exactly 4 values")
            bbox = tuple(bbox_parts)
            print(
                f"Using bounding box: lat_min={bbox[0]}, lon_min={bbox[1]}, lat_max={bbox[2]}, lon_max={bbox[3]}"
            )
        except Exception as e:
            print(f"Error parsing bbox: {e}")
            print("bbox format: 'lat_min,lon_min,lat_max,lon_max'")
            sys.exit(1)

    try:
        process_wms_tile(image_path, output_dir, epsilon_factor, bbox=bbox)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

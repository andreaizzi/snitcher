#!/usr/bin/env python3
"""
Extract building polygons from WMS tile images.
Buildings are orange polygons with black borders on transparent background.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import json


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
    """Extract black border lines that separate buildings."""
    # Create a mask of pixels that are opaque
    opaque_mask = alpha > 128

    # Convert to BGR if needed
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
    else:
        bgr = img

    # Detect black pixels (borders) in opaque areas
    # Black is low values in all channels
    black_mask = np.all(bgr < 100, axis=2) & opaque_mask

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
    Snap contour vertices to original black borders to ensure adjacent buildings
    share exact vertices/edges where they meet.

    This solves the topology problem where dilation creates gaps between buildings.
    Buildings that share a wall should share the same vertices.

    Args:
        contours: List of contours from cv2.findContours
        black_borders: Original (non-dilated) black border mask
        snap_distance: Maximum distance to snap (should match dilation radius)

    Returns:
        List of snapped contours
    """
    # Get coordinates of all black border pixels
    border_pixels = np.column_stack(np.where(black_borders > 0))  # Returns (y, x) pairs

    if len(border_pixels) == 0:
        return contours  # No borders to snap to

    snapped_contours = []

    for contour in contours:
        snapped_contour = []

        for point in contour:
            x, y = point[0]

            # Find nearest black border pixel within snap_distance
            # Calculate distances to all border pixels
            distances = np.sqrt(
                (border_pixels[:, 1] - x) ** 2 + (border_pixels[:, 0] - y) ** 2
            )
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            # If close enough to a border, snap to it
            if min_dist <= snap_distance:
                nearest_y, nearest_x = border_pixels[min_dist_idx]
                snapped_contour.append([[nearest_x, nearest_y]])
            else:
                # Keep original position if not near a border
                snapped_contour.append([[x, y]])

        # Convert back to numpy array format
        snapped_contours.append(np.array(snapped_contour, dtype=np.int32))

    return snapped_contours


def merge_nearby_vertices(contours, merge_distance=2):
    """
    After snapping, vertices from different buildings might snap to different
    but nearby pixels on the same border. This function merges vertices that
    are very close together so adjacent buildings share exact coordinates.

    Args:
        contours: List of snapped contours
        merge_distance: Maximum distance to consider vertices as "same" (pixels)

    Returns:
        List of contours with merged vertices
    """
    # Collect all vertices with their building index
    all_vertices = []
    for contour_idx, contour in enumerate(contours):
        for point_idx, point in enumerate(contour):
            x, y = point[0]
            all_vertices.append((x, y, contour_idx, point_idx))

    # Build a mapping of nearby vertices
    # For each unique location, find all vertices within merge_distance
    vertex_groups = {}
    processed = set()

    for i, (x1, y1, c1, p1) in enumerate(all_vertices):
        if i in processed:
            continue

        # Find all vertices close to this one
        group = [(x1, y1, c1, p1)]
        processed.add(i)

        for j, (x2, y2, c2, p2) in enumerate(all_vertices[i + 1 :], start=i + 1):
            if j in processed:
                continue
            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist <= merge_distance:
                group.append((x2, y2, c2, p2))
                processed.add(j)

        if len(group) > 1:
            # Calculate average position for this group
            avg_x = int(np.mean([v[0] for v in group]))
            avg_y = int(np.mean([v[1] for v in group]))
            vertex_groups[(avg_x, avg_y)] = group

    # Update contours with merged vertices
    merged_contours = [contour.copy() for contour in contours]

    for (avg_x, avg_y), group in vertex_groups.items():
        for x, y, contour_idx, point_idx in group:
            merged_contours[contour_idx][point_idx][0][0] = avg_x
            merged_contours[contour_idx][point_idx][0][1] = avg_y

    print(f"  → Merged {len(vertex_groups)} groups of nearby vertices")

    return merged_contours


def snap_shared_boundaries(contours, dilation_distance=2):
    """
    Snap nearby vertices together to create shared boundaries between adjacent buildings.

    After dilation creates gaps, adjacent buildings have separate vertices that should
    be the same. This function:
    1. Finds clusters of vertices that are very close (within dilation_distance)
    2. Replaces each cluster with a single shared vertex at the average position
    3. Ensures adjacent buildings share edges properly

    Args:
        contours: List of contours (polygons)
        dilation_distance: Maximum distance for vertices to be considered "same" (pixels)

    Returns:
        List of contours with snapped vertices
    """
    if not contours:
        return contours

    # Collect all vertices from all contours with their source information
    all_vertices = []
    for contour_idx, contour in enumerate(contours):
        for point_idx, point in enumerate(contour):
            x, y = point[0]
            all_vertices.append(
                {
                    "x": x,
                    "y": y,
                    "contour_idx": contour_idx,
                    "point_idx": point_idx,
                    "original": (x, y),
                }
            )

    if not all_vertices:
        return contours

    # Build clusters of nearby vertices using simple distance threshold
    # This is O(n²) but works fine for typical building counts
    tolerance = dilation_distance * 1.5  # Slightly larger than dilation gap
    vertex_clusters = []
    used = set()

    for i, v1 in enumerate(all_vertices):
        if i in used:
            continue

        cluster = [i]
        used.add(i)

        # Find all vertices within tolerance
        for j, v2 in enumerate(all_vertices):
            if j <= i or j in used:
                continue

            dist = np.sqrt((v1["x"] - v2["x"]) ** 2 + (v1["y"] - v2["y"]) ** 2)
            if dist <= tolerance:
                cluster.append(j)
                used.add(j)

        if len(cluster) > 0:
            vertex_clusters.append(cluster)

    # For each cluster, compute the average position (shared vertex)
    shared_positions = {}
    for cluster in vertex_clusters:
        avg_x = np.mean([all_vertices[i]["x"] for i in cluster])
        avg_y = np.mean([all_vertices[i]["y"] for i in cluster])
        shared_pos = (int(round(avg_x)), int(round(avg_y)))

        for vertex_idx in cluster:
            shared_positions[vertex_idx] = shared_pos

    # Rebuild contours with snapped vertices
    snapped_contours = []
    for contour_idx, contour in enumerate(contours):
        new_points = []
        for point_idx, point in enumerate(contour):
            # Find this vertex in all_vertices list
            vertex_idx = None
            for idx, v in enumerate(all_vertices):
                if v["contour_idx"] == contour_idx and v["point_idx"] == point_idx:
                    vertex_idx = idx
                    break

            if vertex_idx is not None and vertex_idx in shared_positions:
                # Use shared position
                new_x, new_y = shared_positions[vertex_idx]
            else:
                # Keep original position
                new_x, new_y = point[0]

            new_points.append([[new_x, new_y]])

        snapped_contours.append(np.array(new_points, dtype=np.int32))

    print(f"Snapped {len(vertex_clusters)} vertex clusters to create shared boundaries")

    return snapped_contours


def remove_duplicate_vertices(contour):
    """Remove consecutive duplicate vertices from a contour."""
    if len(contour) < 2:
        return contour

    unique_points = [contour[0]]
    for point in contour[1:]:
        if not np.array_equal(point, unique_points[-1]):
            unique_points.append(point)

    # Check if first and last are the same
    if len(unique_points) > 1 and np.array_equal(unique_points[0], unique_points[-1]):
        unique_points = unique_points[:-1]

    return np.array(unique_points, dtype=np.int32)


def find_building_contours(separated_mask):
    """Find contours of individual buildings."""
    contours, hierarchy = cv2.findContours(
        separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


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


def detect_corners_harris(img, contour):
    """
    Alternative: Detect corners using Harris corner detector.
    This is more sophisticated but may not be necessary.
    """
    # Create mask for this building
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Harris corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    # Threshold corners
    corner_mask = corners > 0.01 * corners.max()
    corner_mask = corner_mask & (mask > 0)

    return corner_mask


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

    # 7. Final snapped polygons (topology-corrected, sharing boundaries)
    snapped_img = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(snapped_img, snapped_contours, -1, (0, 0, 255), 2)  # Red
    # Draw vertices
    for contour in snapped_contours:
        for point in contour:
            cv2.circle(snapped_img, tuple(point[0]), 3, (255, 0, 0), -1)
    # Overlay original black borders in white for reference
    snapped_img[black_borders > 0] = [128, 128, 128]
    cv2.imwrite(str(output_dir / "07_final_snapped_polygons.png"), snapped_img)

    # 8. Three-way comparison: raw → simplified → snapped
    comparison = np.hstack([contour_img, simplified_img, snapped_img])
    cv2.imwrite(str(output_dir / "08_comparison.png"), comparison)

    print(f"Debug images saved to: {output_dir}")


def process_wms_tile(image_path, output_dir=None, epsilon_factor=3.0):
    """
    Main processing pipeline for extracting building polygons.

    Args:
        image_path: Path to input PNG image
        output_dir: Directory for output files (default: same as input)
        epsilon_factor: Polygon simplification tolerance in pixels (default: 3.0)
                       Higher values = simpler polygons with fewer vertices
                       Typical range: 2-5 pixels
    """
    print(f"Processing: {image_path}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(image_path).parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Load image
    img = load_image(image_path)
    print(f"Image shape: {img.shape}")

    # 2. Extract alpha channel
    alpha = extract_alpha_mask(img)
    print("Extracted alpha mask")

    # 3. Extract black borders
    black_borders = extract_black_borders(img, alpha)
    print("Extracted black borders")

    # 4. Separate attached buildings
    separated, markers, dilated_borders = separate_buildings(alpha, black_borders)
    print("Separated buildings using black borders")

    # 5. Find contours
    contours = find_building_contours(separated)
    print(f"Found {len(contours)} building contours")

    # 6. Simplify polygons to corner points FIRST
    simplified_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 50:  # Skip very small buildings
            simplified = simplify_polygon(contour, epsilon_factor)
            simplified_contours.append(simplified)

    print(f"Simplified to {len(simplified_contours)} valid polygons")

    # 7. Snap simplified contours to original black borders for proper topology
    # This ensures adjacent buildings share exact vertices/edges
    # IMPORTANT: Do this AFTER simplification so final vertices are snapped
    # Snap distance needs to be larger because simplification can move vertices away
    snap_distance = 15  # Larger to account for simplification displacement
    snapped_contours = snap_contours_to_borders(
        simplified_contours, black_borders, snap_distance=snap_distance
    )
    print(
        f"Snapped simplified contours to black borders (snap_distance={snap_distance}px)"
    )

    # 7b. Merge nearby vertices so adjacent buildings share exact same coordinates
    # After snapping to borders, vertices from different buildings on the same border
    # may snap to nearby but different pixels. This merges them into shared vertices.
    merge_distance = 15  # Should be slightly larger than typical black border thickness
    snapped_contours = merge_nearby_vertices(
        snapped_contours, merge_distance=merge_distance
    )
    print(
        f"Merged nearby vertices for shared boundaries (merge_distance={merge_distance}px)"
    )

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

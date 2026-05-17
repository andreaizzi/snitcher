"""
Snitcher - Stage 2: Vettorializzazione

Converte il GeoTIFF catastale prodotto dallo Stage 1 in un GeoJSON pulito
di poligoni di edifici che rispetta le regole di topologia OSM:
  (1) i vertici condivisi tra edifici adiacenti hanno coordinate identiche;
  (2) ogni vertice che giace su un lato altrui diventa un vertice di quel lato.

Pipeline:
  1. Lettura del GeoTIFF (RGBA + georeferenziazione + metadati dello Stage 1).
  2. Maschera sul riempimento arancione del layer fabbricati del catasto
  3. Componenti connesse sulla maschera -> un'etichetta per edificio.
     Gli edifici adiacenti restano separati grazie alla sottile linea nera
     che fa da perimetro.
  4. Vettorializzazione di ogni etichetta con rasterio.features.shapes
        -> usa GDALPolygonize
  5. Semplificazione di ogni anello (Douglas-Peucker)
  6. Snap dei vertici vicini tra tutti i poligoni -> angoli condivisi
     (Condizione 1).
  7. Suddivisione dei lati: ogni vertice estraneo che giace su un lato
     diventa un vertice esplicito di quel lato (Condizione 2).
  8. Riproiezione in EPSG:4326 e dump come FeatureCollection GeoJSON.

"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import shapes as rio_shapes
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, mapping, shape as shp_shape
from shapely.geometry.polygon import orient
from shapely.ops import transform as shp_transform

# riempimento arancione del layer fabbricati in HSV (in OpenCV la H va da 0 a 179).
# HEX color: #ec8013
ORANGE_HSV_LOWER = np.array([5,  120, 120], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([25, 255, 255], dtype=np.uint8)

# la 4-connettività tiene separati gli edifici anche quando il perimetro nero
# è sottile e diagonale: due pixel arancioni che si toccano solo in un angolo
# restano in componenti diverse.
CC_CONNECTIVITY = 4

# unità reali (metri / m²)
DEFAULT_MIN_AREA_M2 = 4.0
DEFAULT_SIMPLIFY_M  = 0.30
DEFAULT_SNAP_M      = 0.80
DEFAULT_SUBDIV_M    = 0.50


# Stage 2.1 - Caricamento raster

def load_raster(path):
    """Read RGBA + affine transform + CRS + Stage-1 metadata tags."""
    with rasterio.open(path) as src:
        bgra = src.read()                          # (bande, H, W) -> array tridimensionale
        rgba = np.transpose(bgra, (1, 2, 0))       # riordino in (H, W, bande), formato comodo per cv2
        transform = src.transform                  # matrice affine pixel <-> mondo
        crs = src.crs                              # sistema di riferimento (EPSG:25832 in questa pipeline)
        tags = src.tags()                          # metadati salvati dallo Stage 1 (url, timestamp, ppm, ...)
    print(f"[load] {rgba.shape[1]}x{rgba.shape[0]} px, "
          f"CRS={crs}, ppm={tags.get('PIXELS_PER_METER')}")
    return rgba, transform, crs, tags


# Stage 2.2 - Maschera arancione

def make_orange_mask(rgba):
    """Binary mask (uint8 0/1) of the orange building fill.

    No morphological closing here: internal walls between attached buildings
    are drawn as 1-pixel-wide black lines, and any close would dilate orange
    across them and fuse adjacent buildings into a single blob.
    A 3x3 OPEN removes isolated specks (single-pixel JPEG-ish noise) without
    touching real walls.
    """
    rgb = rgba[..., :3] # scarto canale alpha (trasparenza) che non ci serve per questa maschera
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) # convertiamo da RGB a HSV per facilitare la selezione del colore arancione
    mask = cv2.inRange(hsv, ORANGE_HSV_LOWER, ORANGE_HSV_UPPER) # ottengo una maschera binaria: 255 dove il pixel sta nel range arancione, 0 altrove
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) # kernel 3x3 per operazione morfologica
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1) # opening = erosione seguita da dilatazione, toglie i puntini di rumore senza chiudere i muri
    print(f"[mask] orange pixels: {int((mask > 0).sum()):,} / {mask.size:,}")
    return (mask > 0).astype(np.uint8)


# Stage 2.3 - Componenti connesse

def label_components(mask):
    """Unique label per building; 0 is background. Adjacent buildings end up
    in different components because the black perimeter line breaks orange
    connectivity between them."""
    # ogni edificio diventa un blob con un numero progressivo, 0 = sfondo
    n_labels, labels = cv2.connectedComponents(mask, connectivity=CC_CONNECTIVITY)
    print(f"[cc] {n_labels - 1} candidate buildings (excluding background)")
    return labels.astype(np.int32)


# Stage 2.4 - Poligonizzazione

def polygonize(labels, transform, min_area_m2):
    """Vectorize labels -> list of (label, shapely.Polygon) in working CRS.

    rasterio.features.shapes with `mask=(labels>0)` skips the background and,
    crucially, reports interior background regions (courtyards) as holes of
    the surrounding building's polygon.
    """
    polygons = []
    # rio_shapes percorre i blob di ogni etichetta e restituisce il contorno
    # come geometria GeoJSON già nelle coordinate del mondo (grazie a transform)
    for geom, val in rio_shapes(labels, mask=(labels > 0),
                                connectivity=4, transform=transform):
        poly = shp_shape(geom) # converto da dict GeoJSON a oggetto shapely
        if not poly.is_valid:
            poly = poly.buffer(0)  # buffer(0) ripara geometrie auto-intersecanti
        if poly.is_empty or poly.area < min_area_m2:
            continue # scarto i poligoni troppo piccoli
        polygons.append((int(val), poly))
    print(f"[poly] {len(polygons)} polygons after area>={min_area_m2} m² filter")
    return polygons


# Stage 2.5 - Semplificazione (Douglas-Peucker)

def simplify_polygons(polys, tol_m):
    """DP simplification per polygon. For axis-aligned buildings the corner
    vertices survive and everything between disappears - effectively the same
    output you'd get from a corner detector, but topologically robust."""
    out = []
    # conto i vertici prima della semplificazione (serve solo per log)
    n_before = sum(len(p.exterior.coords) - 1 # -1 per rimuovere il vertice di chiusura (è un duplicato del primo)
                   + sum(len(r.coords) - 1 for r in p.interiors)
                   for _, p in polys)
    for lbl, poly in polys:
        s = poly.simplify(tol_m, preserve_topology=True) # DP con preserve_topology=True per evitare di creare auto-intersezioni
        if s.is_empty or not s.is_valid:
            continue
        # in casi degeneri simplify può tornare un MultiPolygon;
        # prendo il pezzo più grande così tengo un poligono per edificio
        if s.geom_type == "MultiPolygon":
            s = max(s.geoms, key=lambda g: g.area)
        out.append((lbl, orient(s, sign=1.0)))  # orient con sign=1.0 forza l'esterno in senso antiorario (CCW), come vuole RFC 7946
    n_after = sum(len(p.exterior.coords) - 1
                  + sum(len(r.coords) - 1 for r in p.interiors)
                  for _, p in out)
    print(f"[simplify] tol={tol_m} m -> vertices {n_before:,} -> {n_after:,}")
    return out


# Stage 2.6 - Snap dei vertici (condizione 1: angoli condivisi)

def snap_vertices(polys, snap_m):
    """Cluster vertices within `snap_m` and replace each cluster with its
    centroid, so any two corners that *should* coincide end up at identical
    coordinates across all polygons."""
    # 1. appiattiamo tutti i vertici in un'unica lista, tenendo traccia di quale (poligono, anello, posizione) appartengono.
    #    ring_idx == -1 significa esterno, >=0 significa interiors[ring_idx].
    flat = []          # lista di (x, y)
    origin = []        # lista di (poly_idx, ring_idx, vert_idx)
    for pi, (_, poly) in enumerate(polys):
        ring_coords = [("ext", list(poly.exterior.coords)[:-1])]  # -1 per rimuovere il vertice di chiusura
        for ri, hole in enumerate(poly.interiors):
            ring_coords.append((ri, list(hole.coords)[:-1]))
        # ring_coords è una lista di tuple: (ring_key, coords).
        # tipo:
        # [("ext", [(x1,y1), (x2,y2), ...]),
        # (0,     [(xa,ya), (xb,yb), ...]),   primo foro
        # (1,     [(xp,yp), ...           )]  secondo foro

        # converto la chiave stringa "ext" in -1, mentre i fori mantengono il loro indice numerico
        for ring_key, coords in ring_coords:
            ring_idx = -1 if ring_key == "ext" else ring_key
            for vi, (x, y) in enumerate(coords):
                # flat e origin hanno la stessa lunghezza e sono allineate:
                #   - flat[i] è il vertice (x,y)
                #   - origin[i] dice a quale poligono/anello/posizione appartiene
                flat.append((x, y))
                origin.append((pi, ring_idx, vi))

    pts = np.asarray(flat) # converto in array numpy per poter creare KD-tree
    if len(pts) == 0:
        return polys # nessun vertice, non c'è nulla da snappare

    # 2. clustering greedy via KD-tree: assegno ogni punto non assegnato al
    #    cluster del primo vicino entro snap_m, altrimenti apro un nuovo cluster.
    tree = cKDTree(pts) # KD-tree per query spaziali veloci, O(N log N) invece di O(N^2)
    cluster_id = np.full(len(pts), -1, dtype=np.int64) # array di interi, uno per vertice, inizializzato a -1
    # (nessun vertice è stato ancora assegnato a nessun cluster)
    centroids = []
    for i in range(len(pts)):
        if cluster_id[i] != -1: # se il vertice i è già stato assegnato a un cluster, salto al prossimo vertice
            continue
        idxs = tree.query_ball_point(pts[i], r=snap_m) # trova tutti i vertici entro snap_m da pts[i]
        idxs = [j for j in idxs if cluster_id[j] == -1] # filtra solo quelli non ancora assegnati a un cluster (cluster_id[j] == -1)
        new_id = len(centroids) # nuovo id
        cluster_id[idxs] = new_id # assegna a tutti i punti (vicini) trovati lo stesso id di cluster
        centroids.append(pts[idxs].mean(axis=0)) # calcola il centroide del cluster come media dei punti assegnati
    centroids = np.asarray(centroids)
    print(f"[snap] {len(pts):,} vertices -> {len(centroids):,} unique "
          f"(snap={snap_m} m)")

    # 3. ricostruisco gli anelli usando i centroidi dei cluster e collasso eventuali duplicati consecutivi
    rings_per_poly = defaultdict(lambda: {"ext": None, "holes": {}})
    by_pi_ri = defaultdict(list)
    for k, (pi, ri, vi) in enumerate(origin):
        by_pi_ri[(pi, ri)].append((vi, cluster_id[k]))
    # esempio: by_pi_ri[(0, -1)] contiene la lista di vertici dell'anello esterno del poligono 0, con i rispettivi cluster_id a cui sono stati assegnati.
    # by_pi_ri è un dict di questo tipo (tuple come chiavi): (pi, ri) -> [(vi, cluster_id), ...]
    for (pi, ri), entries in by_pi_ri.items():
        entries.sort() # ordina per posizione originale del vertice (vi) nell'anello
        coords = [tuple(centroids[cid]) for _, cid in entries] # per ogni vertice nell'anello, sostituisco le sue coordinate con il centroide del cluster a cui è stato assegnato (centroids[cid])
        # due vertici nello stesso cluster diventano identici, ma possono essere distanti nella lista originale, quindi possono creare duplicati consecutivi che vanno rimossi
        dedup = [coords[0]]
        for c in coords[1:]:
            if c != dedup[-1]:
                dedup.append(c)
        # se non è già chiuso, chiudo l'anello (primo punto = ultimo punto)
        if dedup[0] != dedup[-1]:
            dedup.append(dedup[0])
        if ri == -1:
            rings_per_poly[pi]["ext"] = dedup
        else:
            rings_per_poly[pi]["holes"][ri] = dedup

    out = []
    for pi, (lbl, _) in enumerate(polys):
        # ricostruisco i poligoni con i nuovi vertici
        ext = rings_per_poly[pi]["ext"]
        if ext is None or len(ext) < 4: # un anello deve avere almeno 4 vertici (3 vertici è ripetizione del primo)
            continue
        holes = [h for _, h in sorted(rings_per_poly[pi]["holes"].items())
                 if len(h) >= 4]
        poly = Polygon(ext, holes)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        out.append((lbl, orient(poly, sign=1.0))) # orient forza orientamento CCW per l'anello esterno, come richiesto da RFC 7946
    return out


# Stage 2.7 - Suddivisione dei lati (condizione 2)

def _point_on_segment(p, a, b, tol):
    """Return projection parameter t in (0,1) iff p lies on segment (a,b)
    within distance `tol` and is not coincident with the endpoints."""
    ax, ay = a; bx, by = b; px, py = p # destrutturo le coordinate per leggibilità
    dx, dy = bx - ax, by - ay          # vettore direzione del segmento
    L2 = dx * dx + dy * dy             # lunghezza al quadrato del segmento (evito sqrt)
    if L2 == 0:
        return None # segmento degenere (a == b), non posso proiettare nulla
    # proietto p sul segmento: t = ((p-a) x (b-a)) / |b-a|^2.
    # t=0 corrisponde ad a, t=1 corrisponde a b, t in (0,1) sta in mezzo.
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    if t <= 1e-9 or t >= 1 - 1e-9:    # t agli estremi -> p coincide con un vertice già esistente, niente da inserire
        return None
    # distanza perpendicolare di p dal segmento (confronto al quadrato per evitare la sqrt)
    cx, cy = ax + t * dx, ay + t * dy
    if (px - cx) ** 2 + (py - cy) ** 2 > tol * tol:
        return None # troppo lontano dal lato, non lo considero "appoggiato"
    return t


def subdivide_edges(polys, tol_m):
    """For every edge of every ring, find any *foreign* vertex (from another
    polygon, or from a different ring of the same polygon) that lies on that
    edge within `tol_m`, and insert it. After this step the layer is OSM
    topology compliant."""
    # 1. raccolgo tutte le coordinate distinte come candidati "punti da inserire".
    candidate_pts = set() # set per deduplicare in automatico
    for _, poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            for c in list(ring.coords)[:-1]: # -1 per saltare il vertice di chiusura
                candidate_pts.add(tuple(c))
    candidate_arr = np.asarray(list(candidate_pts))
    tree = cKDTree(candidate_arr) # KD-tree sui candidati

    n_inserts = 0
    new_polys = []
    for lbl, poly in polys:
        new_rings = []
        for ring_idx, ring in enumerate([poly.exterior, *poly.interiors]):
            ring_coords = list(ring.coords)[:-1]   # apro l'anello rimuovendo il vertice di chiusura
            own = set(map(tuple, ring_coords)) # vertici dell'anello corrente, da escludere dai candidati
            new_ring = []
            for i in range(len(ring_coords)):
                a = ring_coords[i]                            # inizio del lato corrente
                b = ring_coords[(i + 1) % len(ring_coords)]   # fine del lato (modulo per chiudere l'anello)
                new_ring.append(a) # tengo sempre il vertice di partenza

                # candidati vicini al lato: bounding box del lato + tolleranza
                xmin, xmax = min(a[0], b[0]) - tol_m, max(a[0], b[0]) + tol_m
                ymin, ymax = min(a[1], b[1]) - tol_m, max(a[1], b[1]) + tol_m
                # query a raggio sul KD-tree, centrata sul punto medio del lato
                # raggio = mezza diagonale + tolleranza, così copro tutto il segmento.
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                r = 0.5 * ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 + tol_m
                idxs = tree.query_ball_point(mid, r=r)

                hits = [] # candidati che effettivamente cadono sul lato
                for j in idxs:
                    p = tuple(candidate_arr[j])
                    if p in own: # salto i vertici dell'anello corrente, altrimenti li reinserirei a vuoto
                        continue
                    if not (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax):
                        continue # taglio veloce: fuori dal bbox del lato, non può essere sul lato
                    t = _point_on_segment(p, a, b, tol_m) # test geometrico vero e proprio
                    if t is not None:
                        hits.append((t, p))
                hits.sort() # ordino per parametro t, così rispetto l'ordine lungo il lato
                for _, p in hits:
                    new_ring.append(p)
                    n_inserts += 1
            new_ring.append(new_ring[0])   # richiudo l'anello (primo punto = ultimo)
            new_rings.append(new_ring)

        ext = new_rings[0] # il primo anello è sempre l'esterno
        holes = new_rings[1:] # i successivi sono i fori (cortili interni)
        if len(ext) < 4:
            continue # anello con meno di 4 punti non è valido (3 vertici + chiusura)
        np_poly = Polygon(ext, [h for h in holes if len(h) >= 4])
        if not np_poly.is_valid:
            np_poly = np_poly.buffer(0) # di nuovo buffer(0) per ripulire eventuali invalidità
        if np_poly.is_empty:
            continue
        if np_poly.geom_type == "MultiPolygon":
            np_poly = max(np_poly.geoms, key=lambda g: g.area)
        new_polys.append((lbl, orient(np_poly, sign=1.0))) # CCW per coerenza con gli step precedenti

    print(f"[subdivide] inserted {n_inserts} vertices (tol={tol_m} m)")
    return new_polys


# Stage 2.8 - Riproiezione + GeoJSON

def to_geojson(polys, src_crs, dst_crs="EPSG:4326"):
    """Build a GeoJSON FeatureCollection in `dst_crs`."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True) # trasformatore di coordinate (always_xy=True per avere sempre lon,lat e non lat,lon)
    project = lambda x, y, z=None: transformer.transform(x, y) # funzione punto-per-punto che shp_transform può usare

    features = []
    for fid, (lbl, poly) in enumerate(polys):
        area = poly.area  # calcolo l'area PRIMA della riproiezione: in EPSG:25832 l'unità è il metro, quindi ottengo m² veri
        reproj = shp_transform(project, poly) # riproietto il poligono in lon/lat
        # la riproiezione può invertire l'orientamento, quindi rimetto l'esterno in CCW
        reproj = orient(reproj, sign=1.0)
        features.append({
            "type": "Feature",
            "id": fid,
            "properties": {
                "source_label": lbl,
                "area_m2": round(area, 3),
            },
            "geometry": mapping(reproj),
        })
    return {"type": "FeatureCollection", "features": features}


# Rendering di debug

def _world_to_pixel(coords, transform):
    """Inverse affine: world (x, y) -> pixel (col, row)."""
    inv = ~transform # affine inversa: da coordinate mondo a coordinate pixel
    return [(inv * (x, y)) for (x, y) in coords]


def _draw_polys(rgba, polys, transform, thickness=2, vertex_radius=3):
    """Render polygons on top of the input image for debug."""
    canvas = rgba[..., :3].copy() # disegno sopra una copia dell'immagine, senza canale alpha
    rng = np.random.default_rng(42) # seed fisso per avere colori riproducibili tra run diversi
    for _, poly in polys:
        colour = tuple(int(c) for c in rng.integers(60, 230, size=3))
        for ring in [poly.exterior, *poly.interiors]:
            pix = _world_to_pixel(ring.coords, transform) # porto i vertici da metri a pixel per disegnarli
            pts = np.array([[int(round(x)), int(round(y))] for x, y in pix],
                           dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=colour,
                          thickness=thickness)
            for x, y in pts:
                # disegno un cerchio bianco con bordo nero su ogni vertice, così si vedono bene
                cv2.circle(canvas, (x, y), vertex_radius, (255, 255, 255), -1)
                cv2.circle(canvas, (x, y), vertex_radius, (0, 0, 0), 1)
    return canvas


def save_debug(out_dir, rgba, mask, labels, polys_raw, polys_simp,
               polys_snap, polys_final, transform):
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "01_input.png"),
                cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(out_dir, "02_orange_mask.png"), mask * 255)

    # colore casuale per ogni label, lo sfondo (label 0) resta nero
    n = labels.max() + 1
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 255, size=(n, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    cv2.imwrite(os.path.join(out_dir, "03_labels.png"), palette[labels])

    cv2.imwrite(os.path.join(out_dir, "04_raw_polygons.png"),
                cv2.cvtColor(_draw_polys(rgba, polys_raw, transform, 1, 0),
                             cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "05_simplified.png"),
                cv2.cvtColor(_draw_polys(rgba, polys_simp, transform),
                             cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "06_snapped.png"),
                cv2.cvtColor(_draw_polys(rgba, polys_snap, transform),
                             cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "07_final.png"),
                cv2.cvtColor(_draw_polys(rgba, polys_final, transform),
                             cv2.COLOR_RGB2BGR))
    print(f"[debug] images written to {out_dir}/")


# CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Stage-1 GeoTIFF path")
    ap.add_argument("--output", default="buildings.geojson",
                    help="output GeoJSON path (default: buildings.geojson)")
    ap.add_argument("--debug-dir", default="debug_stage2",
                    help="directory for intermediate debug PNGs")
    ap.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA_M2,
                    help=f"discard polygons under this many m² (default {DEFAULT_MIN_AREA_M2})")
    ap.add_argument("--simplify", type=float, default=DEFAULT_SIMPLIFY_M,
                    help=f"Douglas-Peucker tolerance in m (default {DEFAULT_SIMPLIFY_M})")
    ap.add_argument("--snap", type=float, default=DEFAULT_SNAP_M,
                    help=f"vertex-snap radius in m (default {DEFAULT_SNAP_M})")
    ap.add_argument("--subdivide", type=float, default=DEFAULT_SUBDIV_M,
                    help=f"edge-subdivision tolerance in m (default {DEFAULT_SUBDIV_M})")
    args = ap.parse_args()

    rgba, transform, crs, tags = load_raster(args.input)
    mask = make_orange_mask(rgba)
    labels = label_components(mask)
    polys_raw = polygonize(labels, transform, args.min_area)
    polys_simp = simplify_polygons(polys_raw, args.simplify)
    polys_snap = snap_vertices(polys_simp, args.snap)
    polys_final = subdivide_edges(polys_snap, args.subdivide)

    gj = to_geojson(polys_final, crs, dst_crs="EPSG:4326")
    with open(args.output, "w") as f:
        json.dump(gj, f, separators=(",", ":"))
    print(f"[out] {len(gj['features'])} buildings -> {args.output}")

    save_debug(args.debug_dir, rgba, mask, labels,
               polys_raw, polys_simp, polys_snap, polys_final, transform)


if __name__ == "__main__":
    main()

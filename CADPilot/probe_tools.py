"""
Tier 4 — Agentic Probing Tools (read-only spatial measurements).

All functions run on the Fusion main thread in response to `probe_request`
WebSocket messages from the backend.  Results are plain dicts for JSON
serialisation.  All coordinates and distances are returned in millimetres.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)

_CM_TO_MM = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_MAP: Dict[str, Any] = {}  # populated at bottom of module


def execute_probe(
    app: adsk.core.Application,
    tool_name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch a named probe tool and return its result dict."""
    fn = _TOOL_MAP.get(tool_name)
    if fn is None:
        return {"error": f"Unknown probe tool: '{tool_name}'"}
    try:
        return fn(app, params)
    except Exception as exc:
        logger.exception("[probe] tool=%s raised: %s", tool_name, exc)
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

def probe_bounding_box(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return the axis-aligned bounding box of given entity tokens.
    If no tokens are supplied, measures every solid body in the design.
    """
    design = _require_design(app)
    tokens: List[str] = params.get("tokens") or []

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")

    def _expand(bbox: adsk.core.BoundingBox3D) -> None:
        nonlocal min_x, min_y, min_z, max_x, max_y, max_z
        try:
            min_x = min(min_x, bbox.minPoint.x)
            min_y = min(min_y, bbox.minPoint.y)
            min_z = min(min_z, bbox.minPoint.z)
            max_x = max(max_x, bbox.maxPoint.x)
            max_y = max(max_y, bbox.maxPoint.y)
            max_z = max(max_z, bbox.maxPoint.z)
        except Exception:
            pass

    if tokens:
        for token in tokens:
            try:
                found = design.findEntityByToken(token)
                if found:
                    bbox = getattr(found[0], "boundingBox", None)
                    if bbox:
                        _expand(bbox)
            except Exception as exc:
                logger.debug("bounding_box: token %s failed: %s", token, exc)
    else:
        for body in design.rootComponent.bRepBodies:
            try:
                bbox = getattr(body, "boundingBox", None)
                if bbox:
                    _expand(bbox)
            except Exception:
                pass

    if min_x == float("inf"):
        return {"error": "No bounding box data available"}

    return {
        "min": [_r(min_x * _CM_TO_MM), _r(min_y * _CM_TO_MM), _r(min_z * _CM_TO_MM)],
        "max": [_r(max_x * _CM_TO_MM), _r(max_y * _CM_TO_MM), _r(max_z * _CM_TO_MM)],
        "size": [
            _r((max_x - min_x) * _CM_TO_MM),
            _r((max_y - min_y) * _CM_TO_MM),
            _r((max_z - min_z) * _CM_TO_MM),
        ],
        "center": [
            _r((min_x + max_x) * 0.5 * _CM_TO_MM),
            _r((min_y + max_y) * 0.5 * _CM_TO_MM),
            _r((min_z + max_z) * 0.5 * _CM_TO_MM),
        ],
        "units": "mm",
    }


def probe_min_distance(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure minimum distance in mm between two B-Rep entities by token."""
    design = _require_design(app)
    token_a = params.get("token_a") or params.get("entity_a", "")
    token_b = params.get("token_b") or params.get("entity_b", "")

    if not token_a or not token_b:
        return {"error": "Requires 'token_a' and 'token_b'"}

    try:
        ea = design.findEntityByToken(token_a)
        eb = design.findEntityByToken(token_b)
        if not ea or not eb:
            return {"error": "One or both tokens could not be resolved"}

        result = app.measureManager.measureMinimumDistance(ea[0], eb[0])
        if not result:
            return {"error": "measureMinimumDistance returned None"}

        pa = result.positionOne
        pb = result.positionTwo
        return {
            "distance_mm": _r(result.value * _CM_TO_MM),
            "point_on_a": [_r(pa.x * _CM_TO_MM), _r(pa.y * _CM_TO_MM), _r(pa.z * _CM_TO_MM)],
            "point_on_b": [_r(pb.x * _CM_TO_MM), _r(pb.y * _CM_TO_MM), _r(pb.z * _CM_TO_MM)],
            "units": "mm",
        }
    except Exception as exc:
        logger.exception("probe_min_distance failed: %s", exc)
        return {"error": str(exc)}


def probe_connected_faces(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return faces adjacent to a given face — full per-face descriptors
    (Stage 2 / B4).

    Each entry in ``connected_faces`` carries the same dict shape as
    ``face_tools.list_faces`` produces for a single face (body_handle,
    body_index, face_index, surface_type, area, centroid, normal, etc.) plus
    the ``via_edge_token`` that links it to the source face. This means the
    agent receives entity handles it didn't have before and can refer back
    to them in code generation without needing a follow-up `lookup_entity`.
    """
    design = _require_design(app)
    face_token = params.get("face_token") or params.get("token", "")

    if not face_token:
        return {"error": "Requires 'face_token'"}

    try:
        found = design.findEntityByToken(face_token)
        if not found:
            return {"error": f"Token not found: {face_token}"}

        face = adsk.fusion.BRepFace.cast(found[0])
        if not face:
            return {"error": "Token does not resolve to a BRepFace"}

        # Pre-build a map { face_entity_token -> full_descriptor } once so we
        # can look up any adjacent face without re-walking list_faces.
        from . import face_tools  # local import — Fusion-only at runtime
        all_faces, _units = face_tools.list_faces(app)
        token_to_face: Dict[str, Dict[str, Any]] = {
            f.get("entity_token"): f
            for f in all_faces
            if isinstance(f, dict) and f.get("entity_token")
        }

        seen: set = set()
        adjacent: List[Dict[str, Any]] = []

        for edge in face.edges:
            for adj in edge.faces:
                tok = adj.entityToken
                if tok == face_token or tok in seen:
                    continue
                seen.add(tok)

                full = token_to_face.get(tok)
                if full is None:
                    # Fallback: minimal shape if the bulk extractor missed it
                    # (e.g. body in a sub-occurrence not iterated).
                    obj_type = "Unknown"
                    try:
                        g = adj.geometry
                        if g:
                            obj_type = g.objectType.split("::")[-1]
                    except Exception:
                        pass
                    full = {
                        "entity_token": tok,
                        "surface_type": obj_type,
                        "area": _r(adj.area * 100.0),
                    }

                # Annotate with the linking edge so adjacency walks are reversible.
                entry = dict(full)
                entry["via_edge_token"] = edge.entityToken
                # Keep the legacy field name alongside the full descriptor so
                # any older caller still finds {"token", "surface_type", ...}.
                entry.setdefault("token", tok)
                adjacent.append(entry)

        return {"connected_faces": adjacent, "count": len(adjacent)}

    except Exception as exc:
        logger.exception("probe_connected_faces failed: %s", exc)
        return {"error": str(exc)}


def probe_face_properties(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Return precise geometric properties of a face by entity token."""
    design = _require_design(app)
    face_token = params.get("face_token") or params.get("token", "")

    if not face_token:
        return {"error": "Requires 'face_token'"}

    try:
        found = design.findEntityByToken(face_token)
        if not found:
            return {"error": f"Token not found: {face_token}"}

        face = adsk.fusion.BRepFace.cast(found[0])
        if not face:
            return {"error": "Token does not resolve to a BRepFace"}

        result: Dict[str, Any] = {
            "token": face_token,
            "entity_token": face_token,
            "area_mm2": _r(face.area * 100.0),
            "area": _r(face.area * 100.0),
            "units": "mm",
        }

        # Stage 2 / B4: stamp parent body identifiers so the agent gets a
        # handle it can refer back to without a follow-up lookup_entity call.
        try:
            from .body_tools import compute_body_handle  # local import
            parent_body = getattr(face, "body", None)
            if parent_body is not None:
                parent_token = getattr(parent_body, "entityToken", None)
                root = design.rootComponent
                parent_idx = None
                face_idx = None
                for i in range(root.bRepBodies.count):
                    candidate = root.bRepBodies.item(i)
                    if candidate is parent_body:
                        parent_idx = i
                        for j in range(candidate.faces.count):
                            if candidate.faces.item(j) is face:
                                face_idx = j
                                break
                        break
                result["body_handle"] = compute_body_handle(parent_token, parent_idx)
                if parent_idx is not None:
                    result["body_index"] = parent_idx
                if face_idx is not None:
                    result["face_index"] = face_idx
                if getattr(parent_body, "name", None):
                    result["body_name"] = parent_body.name
        except Exception as exc:
            logger.debug("probe_face_properties: parent-body resolution failed: %s", exc)

        try:
            c = face.centroid
            result["centroid"] = [_r(c.x * _CM_TO_MM), _r(c.y * _CM_TO_MM), _r(c.z * _CM_TO_MM)]
        except Exception:
            pass

        try:
            geom = face.geometry
            if geom:
                obj_type = geom.objectType
                result["surface_type"] = obj_type.split("::")[-1]

                if "Plane" in obj_type:
                    try:
                        n = geom.normal
                        result["normal"] = [round(n.x, 6), round(n.y, 6), round(n.z, 6)]
                    except Exception:
                        pass
                    try:
                        o = geom.origin
                        result["origin"] = [_r(o.x * _CM_TO_MM), _r(o.y * _CM_TO_MM), _r(o.z * _CM_TO_MM)]
                    except Exception:
                        pass

                elif "Cylinder" in obj_type:
                    try:
                        result["radius_mm"] = _r(geom.radius * _CM_TO_MM)
                    except Exception:
                        pass
                    try:
                        a = geom.axis
                        result["axis"] = [round(a.x, 6), round(a.y, 6), round(a.z, 6)]
                    except Exception:
                        pass
                    try:
                        o = geom.origin
                        result["origin"] = [_r(o.x * _CM_TO_MM), _r(o.y * _CM_TO_MM), _r(o.z * _CM_TO_MM)]
                    except Exception:
                        pass

                elif "Cone" in obj_type:
                    try:
                        result["base_radius_mm"] = _r(geom.baseRadius * _CM_TO_MM)
                    except Exception:
                        pass
                    try:
                        result["half_angle_rad"] = round(geom.halfAngle, 6)
                    except Exception:
                        pass
                    try:
                        a = geom.axis
                        result["axis"] = [round(a.x, 6), round(a.y, 6), round(a.z, 6)]
                    except Exception:
                        pass
                    try:
                        o = geom.origin
                        result["origin"] = [_r(o.x * _CM_TO_MM), _r(o.y * _CM_TO_MM), _r(o.z * _CM_TO_MM)]
                    except Exception:
                        pass

                elif "Sphere" in obj_type:
                    try:
                        result["radius_mm"] = _r(geom.radius * _CM_TO_MM)
                    except Exception:
                        pass

                elif "Torus" in obj_type:
                    try:
                        result["major_radius_mm"] = _r(geom.majorRadius * _CM_TO_MM)
                        result["minor_radius_mm"] = _r(geom.minorRadius * _CM_TO_MM)
                    except Exception:
                        pass

        except Exception as exc:
            logger.debug("probe_face_properties: geometry extraction failed: %s", exc)

        return result

    except Exception as exc:
        logger.exception("probe_face_properties failed: %s", exc)
        return {"error": str(exc)}


def probe_edge_chain(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Walk tangent edge chain from a starting edge.
    Returns all connected tangent edges with their lengths and midpoints.
    """
    design = _require_design(app)
    edge_token = params.get("edge_token") or params.get("token", "")

    if not edge_token:
        return {"error": "Requires 'edge_token'"}

    try:
        found = design.findEntityByToken(edge_token)
        if not found:
            return {"error": f"Token not found: {edge_token}"}

        start_edge = adsk.fusion.BRepEdge.cast(found[0])
        if not start_edge:
            return {"error": "Token does not resolve to a BRepEdge"}

        visited: set = set()
        chain: List[Dict[str, Any]] = []

        def _walk(edge: adsk.fusion.BRepEdge) -> None:
            tok = edge.entityToken
            if tok in visited:
                return
            visited.add(tok)

            length_mm = 0.0
            try:
                lr = edge.evaluator.getLength()
                if isinstance(lr, (tuple, list)) and len(lr) >= 2 and lr[0]:
                    length_mm = float(lr[1]) * _CM_TO_MM
                else:
                    length_mm = float(getattr(edge, "length", 0.0)) * _CM_TO_MM
            except Exception:
                length_mm = float(getattr(edge, "length", 0.0)) * _CM_TO_MM

            midpoint = None
            try:
                ev = edge.evaluator
                ok, t0, t1 = ev.getParameterExtents()
                if ok:
                    ok2, pt = ev.getPointAtParameter((t0 + t1) / 2.0)
                    if ok2:
                        midpoint = [_r(pt.x * _CM_TO_MM), _r(pt.y * _CM_TO_MM), _r(pt.z * _CM_TO_MM)]
            except Exception:
                pass

            obj_type = "Unknown"
            try:
                g = edge.geometry
                if g:
                    obj_type = g.objectType.split("::")[-1]
            except Exception:
                pass

            chain.append({
                "token": tok,
                "length_mm": _r(length_mm),
                "midpoint": midpoint,
                "edge_type": obj_type,
            })

            for vertex in [edge.startVertex, edge.endVertex]:
                if not vertex:
                    continue
                for adj in vertex.edges:
                    if adj.entityToken in visited:
                        continue
                    try:
                        t1v = _tangent_at_vertex(edge, vertex)
                        t2v = _tangent_at_vertex(adj, vertex)
                        if t1v and t2v:
                            dot = sum(t1v[i] * t2v[i] for i in range(3))
                            if abs(dot) > 0.99:
                                _walk(adj)
                    except Exception:
                        pass

        _walk(start_edge)

        return {
            "chain": chain,
            "count": len(chain),
            "total_length_mm": _r(sum(e["length_mm"] for e in chain)),
        }

    except Exception as exc:
        logger.exception("probe_edge_chain failed: %s", exc)
        return {"error": str(exc)}


def probe_body_volume(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Return volume, mass, and bounding box of a body by token or index."""
    design = _require_design(app)
    token = params.get("body_token") or params.get("token", "")
    index = params.get("index")

    try:
        if token:
            found = design.findEntityByToken(token)
            if not found:
                return {"error": f"Token not found: {token}"}
            body = adsk.fusion.BRepBody.cast(found[0])
        elif index is not None:
            body = design.rootComponent.bRepBodies.item(int(index))
        else:
            body = design.rootComponent.bRepBodies.item(0)

        if not body:
            return {"error": "Could not resolve to a BRepBody"}

        result: Dict[str, Any] = {
            "name": body.name,
            "is_solid": body.isSolid,
            "units": "mm",
        }

        try:
            pp = body.physicalProperties
            result["volume_mm3"] = _r(pp.volume * 1000.0)
            result["mass_g"] = _r(pp.mass * 1000.0)
            c = pp.centerOfMass
            result["center_of_mass"] = [_r(c.x * _CM_TO_MM), _r(c.y * _CM_TO_MM), _r(c.z * _CM_TO_MM)]
        except Exception as exc:
            logger.debug("probe_body_volume: physicalProperties failed: %s", exc)

        try:
            bbox = body.boundingBox
            if bbox:
                result["bounding_box"] = {
                    "min": [_r(bbox.minPoint.x * _CM_TO_MM), _r(bbox.minPoint.y * _CM_TO_MM), _r(bbox.minPoint.z * _CM_TO_MM)],
                    "max": [_r(bbox.maxPoint.x * _CM_TO_MM), _r(bbox.maxPoint.y * _CM_TO_MM), _r(bbox.maxPoint.z * _CM_TO_MM)],
                }
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.exception("probe_body_volume failed: %s", exc)
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_design(app: adsk.core.Application) -> adsk.fusion.Design:
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("No active Fusion 360 design")
    return design


def _resolve_body_by_handle(
    design: adsk.fusion.Design,
    body_handle: str,
) -> Optional[adsk.fusion.BRepBody]:
    """Look a body up by its stable handle (sha1 prefix of entityToken).

    Falls back to the synthetic ``b_idxN`` handle for tokenless bodies.
    Returns None when nothing matches; callers should report a structured
    error instead of raising.
    """
    if not body_handle:
        return None
    # Local import keeps probe_tools importable even if body_tools is reloaded.
    from .body_tools import compute_body_handle  # noqa: WPS433

    root = design.rootComponent
    for i in range(root.bRepBodies.count):
        body = root.bRepBodies.item(i)
        token = getattr(body, "entityToken", None)
        if compute_body_handle(token, i) == body_handle:
            return body
    return None


def _r(v: float, n: int = 4) -> float:
    return round(v, n)


def _tangent_at_vertex(
    edge: adsk.fusion.BRepEdge,
    vertex: adsk.fusion.BRepVertex,
) -> Optional[List[float]]:
    """Return the normalised tangent of edge at the given vertex end."""
    try:
        ev = edge.evaluator
        ok, t0, t1 = ev.getParameterExtents()
        if not ok:
            return None
        sv = edge.startVertex
        use_start = sv and _pts_close(sv.geometry, vertex.geometry)
        t = t0 if use_start else t1
        ok2, tan = ev.getFirstDerivative(t)
        if not ok2:
            return None
        mag = math.sqrt(tan.x**2 + tan.y**2 + tan.z**2)
        if mag < 1e-10:
            return None
        return [tan.x / mag, tan.y / mag, tan.z / mag]
    except Exception:
        return None


def _pts_close(a: adsk.core.Point3D, b: adsk.core.Point3D, tol: float = 1e-4) -> bool:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) < tol


def probe_face_sketch_bounds(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return the FACE's axis-aligned bounding rectangle as it would appear in
    the sketch coordinate system of a sketch placed on that face.

    Agents use this to plan hole layouts without guessing: if the face's
    sketch-space bbox is 80mm wide × 50mm tall, the agent knows any circle
    placed outside that rectangle will raise EXTRUDE_BOOLEAN_FAIL.

    Params:
      face_token: entityToken of the target BRepFace.

    Returns:
      {
        "width_mm": 80.0, "height_mm": 50.0,
        "min": [x, y], "max": [x, y],   # sketch-space, mm
        "center": [x, y],
        "safe_interior_mm": {             # inset by 2 mm + hole_radius_mm
          "min": [x, y], "max": [x, y],
          "hole_radius_mm": float
        },
        "parent_body_name": "Body1",
        "parent_body_index": 0
      }
    """
    design = _require_design(app)
    face_token = params.get("face_token") or params.get("token", "")
    hole_radius_mm = float(params.get("hole_radius_mm", 0.0))
    margin_mm = float(params.get("margin_mm", 2.0))

    if not face_token:
        return {"error": "face_sketch_bounds requires 'face_token'"}

    try:
        found = design.findEntityByToken(face_token)
        if not found:
            return {"error": f"Token did not resolve: {face_token}"}
        face = adsk.fusion.BRepFace.cast(found[0])
        if face is None:
            return {"error": f"Token does not refer to a BRepFace: {face_token}"}
    except Exception as exc:
        return {"error": f"findEntityByToken failed: {exc}"}

    # Create a transient sketch on the face, read its bbox, then delete the
    # sketch to avoid polluting the timeline.
    root = design.rootComponent
    transient_sketch = None
    try:
        transient_sketch = root.sketches.add(face)

        face_bbox = face.boundingBox
        mn, mx = face_bbox.minPoint, face_bbox.maxPoint
        corners = [
            adsk.core.Point3D.create(mn.x, mn.y, mn.z),
            adsk.core.Point3D.create(mx.x, mn.y, mn.z),
            adsk.core.Point3D.create(mn.x, mx.y, mn.z),
            adsk.core.Point3D.create(mn.x, mn.y, mx.z),
            adsk.core.Point3D.create(mx.x, mx.y, mn.z),
            adsk.core.Point3D.create(mx.x, mn.y, mx.z),
            adsk.core.Point3D.create(mn.x, mx.y, mx.z),
            adsk.core.Point3D.create(mx.x, mx.y, mx.z),
        ]
        xs, ys = [], []
        for c in corners:
            p = transient_sketch.modelToSketchSpace(c)
            xs.append(p.x)
            ys.append(p.y)
        sk_min_x, sk_max_x = min(xs), max(xs)
        sk_min_y, sk_max_y = min(ys), max(ys)
    finally:
        # Clean up the transient sketch to keep the user's timeline tidy.
        try:
            if transient_sketch is not None:
                transient_sketch.deleteMe()
        except Exception:
            pass

    width_cm = sk_max_x - sk_min_x
    height_cm = sk_max_y - sk_min_y

    inset_cm = (hole_radius_mm + margin_mm) / _CM_TO_MM
    safe_min_x = sk_min_x + inset_cm
    safe_max_x = sk_max_x - inset_cm
    safe_min_y = sk_min_y + inset_cm
    safe_max_y = sk_max_y - inset_cm
    safe_fits = safe_max_x > safe_min_x and safe_max_y > safe_min_y

    # Find the parent body index / name for convenience.
    parent_body = getattr(face, "body", None)
    parent_name = getattr(parent_body, "name", None)
    parent_idx = None
    if parent_body is not None:
        for i in range(root.bRepBodies.count):
            if root.bRepBodies.item(i) is parent_body:
                parent_idx = i
                break

    return {
        "width_mm":  _r(width_cm * _CM_TO_MM),
        "height_mm": _r(height_cm * _CM_TO_MM),
        "min":       [_r(sk_min_x * _CM_TO_MM), _r(sk_min_y * _CM_TO_MM)],
        "max":       [_r(sk_max_x * _CM_TO_MM), _r(sk_max_y * _CM_TO_MM)],
        "center":    [_r((sk_min_x + sk_max_x) * 0.5 * _CM_TO_MM),
                      _r((sk_min_y + sk_max_y) * 0.5 * _CM_TO_MM)],
        "safe_interior_mm": {
            "fits":           safe_fits,
            "hole_radius_mm": hole_radius_mm,
            "margin_mm":      margin_mm,
            "min":            [_r(safe_min_x * _CM_TO_MM), _r(safe_min_y * _CM_TO_MM)] if safe_fits else None,
            "max":            [_r(safe_max_x * _CM_TO_MM), _r(safe_max_y * _CM_TO_MM)] if safe_fits else None,
        },
        "parent_body_name":  parent_name,
        "parent_body_index": parent_idx,
        "units":             "mm",
    }


def probe_lookup_entity(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return the full per-entity dict for a single face/edge/vertex inside the
    body identified by ``body_handle``. Mirrors the per-entity shape produced
    by ``face_tools.list_faces`` / ``edge_tools.list_edges`` / ``body_tools``.

    Params:
      body_handle: stable handle (sha1 prefix of entityToken) of the parent body.
      kind:        "face" | "edge" | "vertex".
      index:       0-based index inside that body's faces/edges/vertices.

    The brep dump now lists only one summary line per body and points the
    agent at this probe to read any specific face/edge/vertex on demand.
    """
    body_handle = (params.get("body_handle") or params.get("handle") or "").strip()
    kind = (params.get("kind") or "face").strip().lower()
    try:
        index = int(params.get("index", 0))
    except (TypeError, ValueError):
        return {"error": "lookup_entity requires integer 'index'"}

    if not body_handle:
        return {"error": "lookup_entity requires 'body_handle'"}
    if kind not in ("face", "edge", "vertex"):
        return {"error": f"lookup_entity 'kind' must be face|edge|vertex, got {kind!r}"}

    design = _require_design(app)
    body = _resolve_body_by_handle(design, body_handle)
    if body is None:
        return {"error": f"lookup_entity: body_handle {body_handle!r} not found"}

    body_index = None
    root = design.rootComponent
    for i in range(root.bRepBodies.count):
        if root.bRepBodies.item(i) is body:
            body_index = i
            break

    try:
        if kind == "face":
            from . import face_tools  # local import — Fusion-only at runtime
            faces, _units = face_tools.list_faces(app)
            for entry in faces:
                if (
                    entry.get("body_handle") == body_handle
                    and entry.get("face_index") == index
                ):
                    return entry
            return {
                "error": (
                    f"lookup_entity: face index {index} not found in body "
                    f"{body_handle!r} (have {body.faces.count} faces)"
                )
            }

        if kind == "edge":
            from . import edge_tools  # local import — Fusion-only at runtime
            edges, _units = edge_tools.list_edges(app)
            for entry in edges:
                if (
                    entry.get("body_handle") == body_handle
                    and entry.get("edge_index") == index
                ):
                    return entry
            return {
                "error": (
                    f"lookup_entity: edge index {index} not found in body "
                    f"{body_handle!r} (have {body.edges.count} edges)"
                )
            }

        # kind == "vertex"
        if not (0 <= index < body.vertices.count):
            return {
                "error": (
                    f"lookup_entity: vertex index {index} out of range "
                    f"(have {body.vertices.count} vertices)"
                )
            }
        vertex = body.vertices.item(index)
        position = None
        try:
            geom = vertex.geometry
            if geom is not None:
                position = [
                    _r(geom.x * _CM_TO_MM),
                    _r(geom.y * _CM_TO_MM),
                    _r(geom.z * _CM_TO_MM),
                ]
        except Exception:
            pass
        return {
            "id": f"v{index}",
            "token": getattr(vertex, "entityToken", None),
            "p": position,
            "body_handle": body_handle,
            "body_index": body_index,
            "vertex_index": index,
            "units": "mm",
        }

    except Exception as exc:  # pragma: no cover — defensive; Fusion runtime nuances.
        logger.exception("probe_lookup_entity failed: %s", exc)
        return {"error": str(exc)}


def probe_face_by_description(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Stage 2 / B2 — rank a body's faces against a natural-language phrase.

    Params:
      body_handle:  stable handle of the parent body (from the brep summary).
      phrase:       free-form description ("front face", "the round face on top",
                    "biggest flat face", "+Y face", …).
      camera_frame: optional camera basis (same shape as
                    `camera_tools.get_camera_frame_mm`); when supplied,
                    viewer-relative labels resolve against the user's view.
      top_n:        cap the returned candidate list (default 4).

    Returns ``{"candidates": [...], "count": N}``. ``candidates`` is the
    ranked list returned by ``face_finder.face_by_description`` (each entry
    carries body_handle / face_index / entity_token / score / rationale).
    Empty list means "no match" — caller should ask for clarification, NOT
    treat this as an error.
    """
    body_handle = (params.get("body_handle") or "").strip()
    phrase = (params.get("phrase") or "").strip()
    camera_frame = params.get("camera_frame")
    try:
        top_n = int(params.get("top_n", 4))
    except (TypeError, ValueError):
        top_n = 4

    if not body_handle:
        return {"error": "face_by_description requires 'body_handle'"}
    if not phrase:
        return {"error": "face_by_description requires 'phrase'"}

    design = _require_design(app)
    body = _resolve_body_by_handle(design, body_handle)
    if body is None:
        return {"error": f"face_by_description: body_handle {body_handle!r} not found"}

    try:
        from . import face_finder  # local import — Fusion-only at runtime
        candidates = face_finder.face_by_description(
            body, phrase, camera_frame=camera_frame, top_n=top_n,
        )
        return {"candidates": candidates, "count": len(candidates)}
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("probe_face_by_description failed: %s", exc)
        return {"error": str(exc)}


def probe_find_feature_by_phrase(
    app: adsk.core.Application,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Stage 2 / B3 — search the timeline DNA for features matching a phrase.

    Params:
      phrase:    free-form description ("the last hole", "the chamfer",
                 "the 12mm extrude", …).
      top_n:     cap the returned matches (default 4).

    Returns ``{"matches": [...], "count": N}``. Each match dict has
    ``{index, name, type, score, params}``. Empty list when nothing scores
    above zero.
    """
    phrase = (params.get("phrase") or "").strip()
    try:
        top_n = int(params.get("top_n", 4))
    except (TypeError, ValueError):
        top_n = 4

    if not phrase:
        return {"error": "find_feature_by_phrase requires 'phrase'"}

    try:
        from . import feature_tools  # local import — Fusion-only at runtime
        timeline = feature_tools.list_timeline(app)
        matches = feature_tools.find_feature_by_phrase(timeline, phrase, top_n=top_n)
        return {"matches": matches, "count": len(matches)}
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("probe_find_feature_by_phrase failed: %s", exc)
        return {"error": str(exc)}


# Populate dispatcher after all functions are defined
_TOOL_MAP = {
    "bounding_box":           probe_bounding_box,
    "min_distance":           probe_min_distance,
    "connected_faces":        probe_connected_faces,
    "face_properties":        probe_face_properties,
    "edge_chain":             probe_edge_chain,
    "body_volume":            probe_body_volume,
    "face_sketch_bounds":     probe_face_sketch_bounds,
    "lookup_entity":          probe_lookup_entity,
    "face_by_description":    probe_face_by_description,
    "find_feature_by_phrase": probe_find_feature_by_phrase,
}


__all__ = ["execute_probe"]

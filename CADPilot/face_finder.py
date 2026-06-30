"""
Runtime face-finding helpers for LLM-generated Fusion 360 code.

The agent frequently needs to locate a specific face of an existing body by
direction (e.g. "the -Y face of Body1"). Writing that logic from scratch in
every generated script is fragile — the LLM used to emit ad-hoc loops that
compared ``face.evaluator.getNormalAtPoint(...)`` against a target vector,
hit floating-point mismatches, and raised "Could not find front face (-Y)".

This module centralises the lookup so generated code can simply call:

    face = face_finder.face_by_direction(target_body, "-Y")

All tolerances, non-planar-face skipping, and outward-normal checks are
handled once, correctly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)


# Canonical direction vectors used throughout Fusion. All axes are expressed
# in the design's construction space, not the active camera's frame.
_AXIS_VECTORS = {
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0),
    "-Z": (0.0, 0.0, -1.0),
    # Friendly aliases the LLM tends to emit.
    "TOP": (0.0, 0.0, 1.0),
    "BOTTOM": (0.0, 0.0, -1.0),
    "FRONT": (0.0, -1.0, 0.0),
    "BACK": (0.0, 1.0, 0.0),
    "RIGHT": (1.0, 0.0, 0.0),
    "LEFT": (-1.0, 0.0, 0.0),
    "UP": (0.0, 0.0, 1.0),
    "DOWN": (0.0, 0.0, -1.0),
}

# Default angular tolerance (cosine) — 0.95 ≈ 18° cone. Generous on purpose so
# slightly-skewed manufactured geometry still matches the intended face.
_DEFAULT_COS_TOL = 0.95


# Direction labels that have a "what the user is looking at" meaning and should
# be resolved against the active camera frame when one is provided. World-axis
# labels ("+X", "-Y", arbitrary tuples, etc.) always resolve against world.
_CAMERA_RELATIVE_LABELS = {
    "FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM", "UP", "DOWN",
}


def _direction_from_camera_frame(
    label: str,
    camera_frame: Dict[str, Any],
) -> Optional[Tuple[float, float, float]]:
    """Resolve a viewer-relative label ("front", "left", …) against a
    camera frame produced by ``camera_tools.get_camera_frame_mm``.

    Camera basis (right-handed):
      forward = +eye_vector       (camera → target)
      up      = +up_vector
      right   = forward × up      (cross product, normalised)

    Mapping (the user is looking AT the model — "front" of the model is the
    side facing the camera, i.e. the OPPOSITE direction of `forward`):
      front  → -forward       back   → +forward
      top/up → +up            bottom/down → -up
      right  → +right         left   → -right

    Returns None if the basis is degenerate.
    """
    try:
        fwd = camera_frame.get("eye_vector")
        up = camera_frame.get("up_vector")
        if not fwd or not up:
            return None

        fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
        ux, uy, uz = float(up[0]), float(up[1]), float(up[2])

        # right = forward × up
        rx = fy * uz - fz * uy
        ry = fz * ux - fx * uz
        rz = fx * uy - fy * ux
        rmag = (rx * rx + ry * ry + rz * rz) ** 0.5
        if rmag < 1e-9:
            return None
        rx, ry, rz = rx / rmag, ry / rmag, rz / rmag

        if label in ("FRONT",):
            return (-fx, -fy, -fz)
        if label == "BACK":
            return (fx, fy, fz)
        if label in ("TOP", "UP"):
            return (ux, uy, uz)
        if label in ("BOTTOM", "DOWN"):
            return (-ux, -uy, -uz)
        if label == "RIGHT":
            return (rx, ry, rz)
        if label == "LEFT":
            return (-rx, -ry, -rz)
    except Exception as exc:
        logger.debug("_direction_from_camera_frame failed: %s", exc)
        return None
    return None


def _normalize_direction(
    direction: Union[str, Iterable[float], adsk.core.Vector3D],
) -> Tuple[float, float, float]:
    """Accept a direction as a label (``"-Y"``, ``"front"``), a 3-tuple, or a
    Fusion ``Vector3D`` and return a unit vector tuple."""
    if isinstance(direction, str):
        key = direction.strip().upper().replace(" ", "")
        if key in _AXIS_VECTORS:
            vec = _AXIS_VECTORS[key]
        else:
            raise ValueError(
                f"Unknown face direction '{direction}'. "
                f"Expected one of: {sorted(_AXIS_VECTORS.keys())}, or a 3-tuple / Vector3D."
            )
    elif isinstance(direction, adsk.core.Vector3D):
        vec = (direction.x, direction.y, direction.z)
    else:
        try:
            vec = tuple(float(c) for c in direction)
        except Exception as exc:
            raise ValueError(f"Invalid direction {direction!r}: {exc}") from exc
        if len(vec) != 3:
            raise ValueError(f"Direction vector must have 3 components, got {len(vec)}")

    mag = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5
    if mag < 1e-9:
        raise ValueError("Direction vector has zero magnitude")
    return (vec[0] / mag, vec[1] / mag, vec[2] / mag)


def _face_outward_normal(
    face: adsk.fusion.BRepFace,
) -> Optional[Tuple[float, float, float]]:
    """Return the face's outward normal as a unit vector, or None on failure.

    Respects ``face.isParamReversed`` so the sign matches what a human would
    call the "outside" of the body.
    """
    try:
        geom = face.geometry
        if geom is None:
            return None
        # Only planes have a well-defined single normal. For non-planar faces
        # we sample the normal at the centroid via the evaluator — callers
        # that need analytic precision should filter on surface type first.
        if "Plane" in geom.objectType:
            n = geom.normal
            nx, ny, nz = n.x, n.y, n.z
        else:
            evaluator = face.evaluator
            centroid = face.centroid
            if not evaluator or not centroid:
                return None
            ok, n = evaluator.getNormalAtPoint(centroid)
            if not ok or n is None:
                return None
            nx, ny, nz = n.x, n.y, n.z

        # Fusion's parametric surface can be "reversed" w.r.t. the solid body;
        # flip the vector so positive points OUT of the solid.
        if getattr(face, "isParamReversed", False):
            nx, ny, nz = -nx, -ny, -nz

        mag = (nx * nx + ny * ny + nz * nz) ** 0.5
        if mag < 1e-9:
            return None
        return (nx / mag, ny / mag, nz / mag)
    except Exception as exc:
        logger.debug("face_outward_normal failed: %s", exc)
        return None


def _resolve_direction_vector(
    direction: Union[str, Iterable[float], adsk.core.Vector3D],
    camera_frame: Optional[Dict[str, Any]],
) -> Tuple[float, float, float]:
    """Resolve a direction argument to a unit world-space vector.

    When ``direction`` is a viewer-relative label ("front", "left", …) AND a
    ``camera_frame`` is supplied, the label is resolved against the camera
    basis. Otherwise (axis labels like "+X", explicit tuples, Vector3D, or
    no camera frame), the legacy world-axis logic is used unchanged.
    """
    if camera_frame and isinstance(direction, str):
        label = direction.strip().upper().replace(" ", "")
        if label in _CAMERA_RELATIVE_LABELS:
            cam_vec = _direction_from_camera_frame(label, camera_frame)
            if cam_vec is not None:
                mag = (cam_vec[0] ** 2 + cam_vec[1] ** 2 + cam_vec[2] ** 2) ** 0.5
                if mag > 1e-9:
                    return (cam_vec[0] / mag, cam_vec[1] / mag, cam_vec[2] / mag)
    return _normalize_direction(direction)


def faces_by_direction(
    body: adsk.fusion.BRepBody,
    direction: Union[str, Iterable[float], adsk.core.Vector3D],
    *,
    tolerance_cos: float = _DEFAULT_COS_TOL,
    planar_only: bool = True,
    camera_frame: Optional[Dict[str, Any]] = None,
) -> List[adsk.fusion.BRepFace]:
    """Return every face of ``body`` whose outward normal aligns with
    ``direction`` within ``tolerance_cos`` (dot-product threshold).

    Args:
        body: The target BRepBody.
        direction: "+X" / "-Y" / "top" / a 3-tuple / a Vector3D.
        tolerance_cos: Minimum dot-product between the face normal and the
            target direction — 0.95 (~18°) by default.
        planar_only: When True (default), non-planar faces are skipped. Cut
            / Join operations almost always want a planar face.
        camera_frame: Optional camera basis dict (from
            ``camera_tools.get_camera_frame_mm``). When given AND ``direction``
            is one of {"front","back","left","right","top","bottom","up","down"},
            the direction is resolved RELATIVE TO THE CAMERA — so "front"
            picks the face the user is currently looking at, regardless of
            world orientation. Axis labels like "+X" / "-Y" still resolve
            against world axes.
    """
    if body is None:
        raise ValueError("face_finder: body is None")

    tx, ty, tz = _resolve_direction_vector(direction, camera_frame)
    matches: List[adsk.fusion.BRepFace] = []

    for i in range(body.faces.count):
        face = body.faces.item(i)
        geom = getattr(face, "geometry", None)
        if planar_only:
            if geom is None or "Plane" not in getattr(geom, "objectType", ""):
                continue
        normal = _face_outward_normal(face)
        if normal is None:
            continue
        dot = normal[0] * tx + normal[1] * ty + normal[2] * tz
        if dot >= tolerance_cos:
            matches.append(face)
    return matches


def face_by_direction(
    body: adsk.fusion.BRepBody,
    direction: Union[str, Iterable[float], adsk.core.Vector3D],
    *,
    tolerance_cos: float = _DEFAULT_COS_TOL,
    planar_only: bool = True,
    pick: str = "largest",
    camera_frame: Optional[Dict[str, Any]] = None,
) -> adsk.fusion.BRepFace:
    """Return exactly one face matching ``direction``.

    When multiple faces point the same way (a stepped block, a body with ribs,
    etc.) ``pick`` decides which one wins:
      - ``"largest"`` → the one with the greatest area (safest default for
        sketch placement and cut extrudes).
      - ``"farthest"`` → the one whose centroid is furthest along the target
        direction (useful for "the front face" on a multi-stepped part).

    When ``camera_frame`` is supplied AND ``direction`` is a viewer-relative
    label ("front", "left", "top", …), the label is resolved against the
    user's current viewpoint instead of world axes — so "fillet the front
    edge" picks the user-visible front, even when the model is rotated.

    Raises ``LookupError`` if no face matches, so generated code can simply
    propagate the failure via the standard try/except wrapper.
    """
    candidates = faces_by_direction(
        body,
        direction,
        tolerance_cos=tolerance_cos,
        planar_only=planar_only,
        camera_frame=camera_frame,
    )
    if not candidates:
        raise LookupError(
            f"face_finder: no face of body '{getattr(body, 'name', '?')}' "
            f"points in direction {direction!r} (tolerance_cos={tolerance_cos}, "
            f"camera_frame={'yes' if camera_frame else 'no'})."
        )
    if len(candidates) == 1:
        return candidates[0]

    if pick == "largest":
        return max(candidates, key=lambda f: getattr(f, "area", 0.0))

    if pick == "farthest":
        tx, ty, tz = _resolve_direction_vector(direction, camera_frame)

        def _score(f: adsk.fusion.BRepFace) -> float:
            c = f.centroid
            return c.x * tx + c.y * ty + c.z * tz if c else float("-inf")

        return max(candidates, key=_score)

    raise ValueError(f"face_finder: unknown pick mode {pick!r}")


def body_by_index_or_name(
    root_component: adsk.fusion.Component,
    identifier: Union[int, str],
) -> adsk.fusion.BRepBody:
    """Resolve a body from either its 0-based ``body_index`` or its name.

    Deprecated in favour of ``body_by_handle`` for new code — the index is not
    stable across feature replays, while the handle is derived from the body's
    entityToken and survives.
    """
    bodies = root_component.bRepBodies
    if isinstance(identifier, int):
        if 0 <= identifier < bodies.count:
            return bodies.item(identifier)
        raise LookupError(
            f"face_finder: body_index {identifier} out of range (have {bodies.count})"
        )
    if isinstance(identifier, str):
        for i in range(bodies.count):
            b = bodies.item(i)
            if b.name == identifier:
                return b
        raise LookupError(f"face_finder: no body named {identifier!r}")
    raise TypeError(
        f"face_finder: identifier must be int (body_index) or str (name), "
        f"got {type(identifier).__name__}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 / B2 — face_by_description
#   Resolve a *natural-language* face reference to a ranked list of candidates
#   without raising. The target_resolver_node combines the candidates with
#   selection_context and the screenshot to pick the one to act on, OR to ask
#   the user for clarification when nothing scores well.
# ─────────────────────────────────────────────────────────────────────────────

# Lightweight token sets used for cheap phrase parsing. Kept small on purpose
# — the LLM does the heavy semantic lifting; this is just enough to bias the
# ranking when the phrase contains a direction / surface-type / size hint.
_DIRECTION_KEYWORDS: Dict[str, str] = {
    "front": "front", "back": "back", "rear": "back",
    "left": "left", "right": "right",
    "top": "top", "upper": "top", "above": "top", "up": "up",
    "bottom": "bottom", "lower": "bottom", "below": "bottom", "down": "down",
    "+x": "+X", "-x": "-X", "+y": "+Y", "-y": "-Y", "+z": "+Z", "-z": "-Z",
}

_SURFACE_TYPE_KEYWORDS: Dict[str, str] = {
    "round":      "Cylinder",
    "circular":   "Cylinder",
    "cylindrical": "Cylinder",
    "cylinder":   "Cylinder",
    "flat":       "Plane",
    "planar":     "Plane",
    "plane":      "Plane",
    "curved":     "Cylinder",  # heuristic — also matches Cone/Sphere/Torus below
    "cone":       "Cone",
    "conical":    "Cone",
    "sphere":     "Sphere",
    "spherical":  "Sphere",
    "torus":      "Torus",
    "toroidal":   "Torus",
}

_SIZE_KEYWORDS_LARGE = {"big", "biggest", "large", "largest", "main"}
_SIZE_KEYWORDS_SMALL = {"small", "smallest", "tiny", "little"}


def _phrase_tokens(phrase: str) -> List[str]:
    return [t for t in (phrase or "").lower().replace("-", " ").split() if t]


def _surface_kind_of_face(face: adsk.fusion.BRepFace) -> str:
    try:
        g = face.geometry
        if g is None:
            return ""
        ot = getattr(g, "objectType", "") or ""
        if "Plane" in ot:
            return "Plane"
        if "Cylinder" in ot:
            return "Cylinder"
        if "Cone" in ot:
            return "Cone"
        if "Sphere" in ot:
            return "Sphere"
        if "Torus" in ot:
            return "Torus"
        if "Nurbs" in ot:
            return "Nurbs"
    except Exception:
        pass
    return ""


def face_by_description(
    body: adsk.fusion.BRepBody,
    phrase: str,
    *,
    camera_frame: Optional[Dict[str, Any]] = None,
    top_n: int = 4,
) -> List[Dict[str, Any]]:
    """Rank the body's faces against a natural-language phrase.

    Combines (and weights) several cheap strategies:
      - Direction match  — picks up "front" / "+Y" / "top" via face_by_direction
        (camera-relative when ``camera_frame`` is supplied).
      - Surface-type     — "round" / "flat" / "cylindrical" / etc.
      - Size hint        — "big" boosts large-area faces, "small" boosts tiny ones.

    Returns up to ``top_n`` ScoredCandidate dicts of the shape::

        {
          "body_handle":   "b_a39f...",
          "face_index":    7,
          "entity_token":  "...",
          "surface_type":  "Plane",
          "area_mm2":      125.4,
          "centroid":      [x, y, z],
          "score":         0.91,
          "rationale":     "matched direction='front' (camera-relative); ...",
        }

    Never raises. Returns ``[]`` when the body has no faces or every strategy
    rejects every candidate; callers (the resolver) treat that as "ask the
    user for clarification" rather than as an error.
    """
    if body is None:
        return []
    try:
        faces = list(body.faces)
    except Exception:
        return []
    if not faces:
        return []

    tokens = _phrase_tokens(phrase)
    token_set = set(tokens)

    # Detect intent components.
    direction_label: Optional[str] = None
    for tok in tokens:
        if tok in _DIRECTION_KEYWORDS:
            direction_label = _DIRECTION_KEYWORDS[tok]
            break

    surface_target: Optional[str] = None
    for tok in tokens:
        if tok in _SURFACE_TYPE_KEYWORDS:
            surface_target = _SURFACE_TYPE_KEYWORDS[tok]
            break

    wants_large = bool(token_set & _SIZE_KEYWORDS_LARGE)
    wants_small = bool(token_set & _SIZE_KEYWORDS_SMALL)

    # Resolve direction-match face set up front (cheap; one normal-vs-vector pass).
    direction_matches: set = set()
    if direction_label:
        try:
            for f in faces_by_direction(
                body, direction_label,
                planar_only=False,
                camera_frame=camera_frame,
            ):
                direction_matches.add(getattr(f, "entityToken", id(f)))
        except Exception as exc:
            logger.debug("face_by_description: direction match failed: %s", exc)

    # Pre-compute face areas so size scoring is normalised.
    areas = []
    for f in faces:
        try:
            areas.append(float(getattr(f, "area", 0.0) or 0.0))
        except Exception:
            areas.append(0.0)
    max_area = max(areas) if areas else 1.0
    min_area = min((a for a in areas if a > 0), default=max_area or 1.0)

    # Look up the parent body's handle once.
    body_handle = ""
    try:
        from .body_tools import compute_body_handle
        body_handle = compute_body_handle(getattr(body, "entityToken", None), None)
    except Exception:
        pass

    candidates: List[Dict[str, Any]] = []
    for idx, face in enumerate(faces):
        rationales: List[str] = []
        score = 0.0

        kind = _surface_kind_of_face(face)
        face_token = getattr(face, "entityToken", None)
        area = areas[idx] if idx < len(areas) else 0.0

        # Direction component — heaviest weight; the phrase usually anchors here.
        if direction_label:
            if face_token in direction_matches:
                score += 0.6
                cam_note = " (camera-relative)" if camera_frame else ""
                rationales.append(f"direction='{direction_label}' matches{cam_note}")

        # Surface-type component.
        if surface_target:
            if kind == surface_target or (
                surface_target == "Cylinder" and kind in ("Cylinder", "Cone", "Sphere", "Torus")
                and "curved" in token_set
            ):
                score += 0.3
                rationales.append(f"surface_type='{kind}' matches phrase")

        # Size component — only when the phrase asked for it.
        if wants_large and max_area > 0:
            ratio = area / max_area
            score += 0.2 * ratio
            if ratio > 0.7:
                rationales.append(f"large face (area {area:.1f} mm²)")
        if wants_small and area > 0 and min_area > 0:
            ratio = min_area / area
            score += 0.2 * ratio
            if ratio > 0.7:
                rationales.append(f"small face (area {area:.1f} mm²)")

        # Without any phrase signal, fall back to a mild area-based prior so
        # the resolver still gets an ordered list it can choose from.
        if not direction_label and not surface_target and not wants_large and not wants_small:
            if max_area > 0:
                score = 0.1 * (area / max_area)

        if score <= 0:
            continue

        centroid_mm: Optional[List[float]] = None
        try:
            c = face.centroid
            if c is not None:
                centroid_mm = [round(c.x * 10.0, 4), round(c.y * 10.0, 4), round(c.z * 10.0, 4)]
        except Exception:
            pass

        candidates.append({
            "body_handle":  body_handle,
            "face_index":   idx,
            "entity_token": face_token,
            "surface_type": kind or "Unknown",
            "area_mm2":     round(area * 100.0, 4),
            "centroid":     centroid_mm,
            "score":        round(min(score, 1.0), 4),
            "rationale":    "; ".join(rationales) if rationales else "weak area prior",
        })

    candidates.sort(key=lambda d: d["score"], reverse=True)
    return candidates[: max(1, int(top_n))]


def body_by_handle(
    root_component: adsk.fusion.Component,
    handle: str,
) -> adsk.fusion.BRepBody:
    """Resolve a body via the stable ``body_handle`` produced at extraction
    time (sha1 prefix of the body's entityToken).

    Falls back to the index-based handle (``b_idxN``) when present so bodies
    that lack a token still resolve. Raises ``LookupError`` when no match is
    found, which propagates through the standard try/except wrapper.
    """
    if not handle:
        raise LookupError("face_finder: body_handle is empty")

    # Local import avoids a circular at module load time.
    from .body_tools import compute_body_handle

    bodies = root_component.bRepBodies
    for i in range(bodies.count):
        b = bodies.item(i)
        token = getattr(b, "entityToken", None)
        if compute_body_handle(token, i) == handle:
            return b
    raise LookupError(
        f"face_finder: no body with handle {handle!r} (have {bodies.count} bodies)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BoundingBox helpers — Fusion's BoundingBox2D / BoundingBox3D are NOT
# iterable, so code like `mn, mx = face.boundingBox` raises
# `TypeError: cannot unpack non-iterable BoundingBox2D object`. These helpers
# give LLM-generated scripts a safe, tuple-based view of any bbox (in mm,
# never cm — Fusion's internal units are cm but the agent plans in mm).
# ─────────────────────────────────────────────────────────────────────────────

_CM_TO_MM = 10.0


def _point_xyz(point) -> Tuple[float, float, float]:
    """Extract (x, y, z) from a Point3D OR Point2D (Point2D has no z)."""
    x = getattr(point, "x", 0.0)
    y = getattr(point, "y", 0.0)
    z = getattr(point, "z", 0.0)
    return (float(x), float(y), float(z))


def bbox_tuple(entity) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Return ``((min_x, min_y, min_z), (max_x, max_y, max_z))`` in CENTIMETRES
    (Fusion's internal unit) for any entity that has a ``.boundingBox``
    attribute — BRepBody, BRepFace, BRepEdge, Occurrence, Sketch, Profile, …
    or for a BoundingBox2D / BoundingBox3D passed in directly.

    Safe to unpack: ``(mn_x, mn_y, mn_z), (mx_x, mx_y, mx_z) = bbox_tuple(body)``.
    """
    bbox = getattr(entity, "boundingBox", None)
    if bbox is None:
        # Treat ``entity`` itself as a bbox if it walks like one.
        if not (hasattr(entity, "minPoint") and hasattr(entity, "maxPoint")):
            raise TypeError(
                f"bbox_tuple: {type(entity).__name__} has neither a "
                "boundingBox attribute nor minPoint/maxPoint — cannot extract bbox."
            )
        bbox = entity
    mn = _point_xyz(bbox.minPoint)
    mx = _point_xyz(bbox.maxPoint)
    return (mn, mx)


def bbox_mm(entity) -> Dict[str, Any]:
    """
    Return a dict with the entity's bounding box converted to MILLIMETRES:
    ``{"min": [x,y,z], "max": [x,y,z], "size": [dx,dy,dz], "center": [cx,cy,cz]}``.

    Agents plan in mm; this is the form they can reason about directly.
    """
    (mn_x, mn_y, mn_z), (mx_x, mx_y, mx_z) = bbox_tuple(entity)
    return {
        "min":    [mn_x * _CM_TO_MM, mn_y * _CM_TO_MM, mn_z * _CM_TO_MM],
        "max":    [mx_x * _CM_TO_MM, mx_y * _CM_TO_MM, mx_z * _CM_TO_MM],
        "size":   [(mx_x - mn_x) * _CM_TO_MM, (mx_y - mn_y) * _CM_TO_MM, (mx_z - mn_z) * _CM_TO_MM],
        "center": [(mn_x + mx_x) * 0.5 * _CM_TO_MM,
                   (mn_y + mx_y) * 0.5 * _CM_TO_MM,
                   (mn_z + mx_z) * 0.5 * _CM_TO_MM],
    }


def body_thickness_cm(
    body: adsk.fusion.BRepBody,
    direction: Union[str, Iterable[float], adsk.core.Vector3D],
) -> float:
    """
    Return the body's extent (in Fusion's internal centimetres) along
    ``direction`` — i.e. how far a cut must travel to pass all the way through.

    Far safer than the LLM hand-rolling ``max(dx, dy, dz)`` which overshoots
    along the wrong axis and causes ``EXTRUDE_BOOLEAN_FAIL``.
    """
    (mn_x, mn_y, mn_z), (mx_x, mx_y, mx_z) = bbox_tuple(body)
    dx, dy, dz = mx_x - mn_x, mx_y - mn_y, mx_z - mn_z
    nx, ny, nz = _normalize_direction(direction)
    # Projection of the bbox diagonal onto the direction vector, using
    # absolute axis extents so the sign of ``direction`` doesn't flip it.
    return abs(nx) * dx + abs(ny) * dy + abs(nz) * dz


# ─────────────────────────────────────────────────────────────────────────────
# Safe cut-through helper
#   The most common LLM failure on modify-requests is a cut-extrude that:
#     (a) targets the wrong face,
#     (b) is missing ``participantBodies`` (causes "body not found to extrude
#         through" when using ThroughAll), or
#     (c) uses an unbounded / wrong-signed distance (causes
#         EXTRUDE_BOOLEAN_FAIL). This helper does all three correctly in one
#     call so the LLM only has to supply the profile.
# ─────────────────────────────────────────────────────────────────────────────

def _profile_centroid_world(profile: adsk.fusion.Profile) -> adsk.core.Point3D:
    """Return the profile centroid converted from sketch space to world (cm)."""
    sketch = profile.parentSketch
    bbox = profile.boundingBox  # BoundingBox2D in sketch coords
    cx = (bbox.minPoint.x + bbox.maxPoint.x) * 0.5
    cy = (bbox.minPoint.y + bbox.maxPoint.y) * 0.5
    return sketch.sketchToModelSpace(adsk.core.Point3D.create(cx, cy, 0.0))


def _point_in_body(body: adsk.fusion.BRepBody, world_pt: adsk.core.Point3D) -> bool:
    """Return True if ``world_pt`` (cm) is inside the solid volume of ``body``."""
    try:
        containment = body.pointContainment(world_pt)
    except Exception:
        return False
    # PointInsidePointContainment / PointOnPointContainment both count as hit.
    return containment in (
        adsk.fusion.PointContainment.PointInsidePointContainment,
        adsk.fusion.PointContainment.PointOnPointContainment,
    )


def _sketch_normal_world(sketch: adsk.fusion.Sketch) -> adsk.core.Vector3D:
    """World-space normal of a sketch (the +Z of the sketch frame in world)."""
    return sketch.xDirection.crossProduct(sketch.yDirection)


def _choose_cut_direction(
    body: adsk.fusion.BRepBody,
    profile: adsk.fusion.Profile,
) -> adsk.fusion.ExtentDirections:
    """
    Decide whether to cut Positive or Negative relative to the sketch plane.

    Picks the direction that actually has body material behind it, by
    sampling a point a millimetre off each side of the profile centroid and
    checking which side is inside the body. This is the ONLY way to avoid
    "body not found to extrude through" when the sketch plane sits on or near
    the body boundary — probing the face normal alone is not enough because
    reversed parametric faces, inclined sketches, and non-planar profiles
    all confound direction choice from the face alone.
    """
    sketch = profile.parentSketch
    centroid = _profile_centroid_world(profile)
    normal = _sketch_normal_world(sketch)
    # Sample 0.01 cm (=0.1 mm) off the sketch plane in both directions.
    eps = 0.01
    pos_pt = adsk.core.Point3D.create(
        centroid.x + normal.x * eps,
        centroid.y + normal.y * eps,
        centroid.z + normal.z * eps,
    )
    neg_pt = adsk.core.Point3D.create(
        centroid.x - normal.x * eps,
        centroid.y - normal.y * eps,
        centroid.z - normal.z * eps,
    )
    pos_inside = _point_in_body(body, pos_pt)
    neg_inside = _point_in_body(body, neg_pt)

    if neg_inside and not pos_inside:
        return adsk.fusion.ExtentDirections.NegativeExtentDirection
    if pos_inside and not neg_inside:
        return adsk.fusion.ExtentDirections.PositiveExtentDirection
    if pos_inside and neg_inside:
        # Sketch plane is INSIDE the body — either side cuts through, but
        # Negative matches the common "sketch on the face of interest" idiom.
        return adsk.fusion.ExtentDirections.NegativeExtentDirection
    # Neither side is inside — the sketch plane is fully outside the body.
    # We still try Negative first (Fusion will report the correct error) so
    # the caller sees a deterministic failure instead of silent nonsense.
    logger.warning(
        "cut_through_body: sketch plane lies outside body %r — cut will likely "
        "fail because no material is adjacent to the profile.",
        getattr(body, "name", "?"),
    )
    return adsk.fusion.ExtentDirections.NegativeExtentDirection


def _profile_is_inside_face(
    profile: adsk.fusion.Profile,
    face: Optional[adsk.fusion.BRepFace],
) -> bool:
    """
    Return True if the profile's bounding box lies entirely inside the
    face's sketch-space bounding box.

    Uses the public ``face_sketch_bbox`` so the validation rectangle is the
    same as the one ``plan_hole_layout`` lays holes inside — keeps the two
    paths in agreement.
    """
    if face is None:
        return True  # Nothing to check against — assume OK.
    sketch = profile.parentSketch
    try:
        (face_min_x, face_min_y), (face_max_x, face_max_y) = face_sketch_bbox(sketch, face)
    except Exception:
        return True  # Bail out gracefully — don't block the cut on a projection failure.

    pb = profile.boundingBox
    return (
        face_min_x - 1e-6 <= pb.minPoint.x
        and pb.maxPoint.x <= face_max_x + 1e-6
        and face_min_y - 1e-6 <= pb.minPoint.y
        and pb.maxPoint.y <= face_max_y + 1e-6
    )


def cut_through_body(
    body: adsk.fusion.BRepBody,
    profile: Union[adsk.fusion.Profile, adsk.core.ObjectCollection, Iterable[adsk.fusion.Profile]],
    direction: Union[str, Iterable[float], adsk.core.Vector3D, None] = None,
    *,
    symmetric: bool = False,
    target_face: Optional[adsk.fusion.BRepFace] = None,
) -> adsk.fusion.ExtrudeFeature:
    """
    Cut a closed profile all the way through ``body``.

    ``direction`` is ACCEPTED BUT IGNORED — retained for backwards
    compatibility. Through-all cuts always travel along the sketch plane's
    own normal; the helper decides which way (Positive/Negative) based on
    which side of the sketch plane contains body material, using
    ``BRepBody.pointContainment``. This is what eliminates the old
    "body not found to extrude through" error: we do NOT trial-and-error
    retry — we compute the right side up front.

    Failure modes this prevents vs. raises:
      - "body not found to extrude through"  → prevented (correct direction).
      - "participantBodies missing"          → prevented (always set here).
      - "profile outside body boundary"      → raised EARLY with a clear
            message if ``target_face`` is supplied and any profile lies
            outside its bounds, instead of letting Fusion emit
            ``EXTRUDE_BOOLEAN_FAIL`` deep in the boolean engine.
      - "setSymmetricExtent signature"       → prevented — for symmetric
            through-cuts we use the OVERSIZED-distance overload Fusion
            actually exposes, not a phantom extent-object overload.

    Args:
        body:      The BRepBody to cut.
        profile:   A single Profile, an ObjectCollection of profiles, or any
                   iterable of profiles. Non-Fusion iterables are wrapped.
        direction: Ignored (kept for backwards compatibility).
        symmetric: If True, cut symmetrically through the body. Uses
                   ``setSymmetricExtent(distance, True)`` with a distance
                   equal to 2.1× the body's bbox diagonal, which guarantees
                   the cut pierces the entire body from either side.
        target_face: Optional face the sketch was placed on. When supplied,
                   each profile is validated to lie inside the face's
                   sketch-space bbox BEFORE the extrude is sent to Fusion —
                   ensures EXTRUDE_BOOLEAN_FAIL never surfaces from here.
    """
    if body is None:
        raise ValueError("cut_through_body: body is None")
    if profile is None:
        raise ValueError("cut_through_body: profile is None")

    # Materialise a Python list of profiles so we can both validate each one
    # and still hand Fusion the right argument type.
    if isinstance(profile, adsk.core.ObjectCollection):
        profile_list = [profile.item(i) for i in range(profile.count)]
        prof_arg: Union[adsk.fusion.Profile, adsk.core.ObjectCollection] = profile
    elif isinstance(profile, adsk.fusion.Profile):
        profile_list = [profile]
        prof_arg = profile
    else:
        profile_list = list(profile)
        col = adsk.core.ObjectCollection.create()
        for p in profile_list:
            col.add(p)
        prof_arg = col

    if not profile_list:
        raise ValueError("cut_through_body: profile is empty")

    # Early profile-inside-face validation.
    if target_face is not None:
        for i, p in enumerate(profile_list):
            if not _profile_is_inside_face(p, target_face):
                raise ValueError(
                    f"cut_through_body: profile {i} is OUTSIDE the target "
                    f"face's bounds — Fusion would raise EXTRUDE_BOOLEAN_FAIL. "
                    f"Use face_finder.plan_hole_layout(sketch, face, ...) to "
                    f"get centre points that are guaranteed to lie inside the "
                    f"face, or shrink the hole radius."
                )

    root_comp = body.parentComponent
    extrudes = root_comp.features.extrudeFeatures
    ext_input = extrudes.createInput(
        prof_arg, adsk.fusion.FeatureOperations.CutFeatureOperation
    )
    # REQUIRED for ThroughAll cuts. Without this Fusion raises
    # "body not found to extrude through" regardless of direction.
    ext_input.participantBodies = [body]

    if symmetric:
        # Fusion's setSymmetricExtent takes a DISTANCE (ValueInput), not an
        # extent-definition object. For a symmetric through-cut we pass an
        # oversized distance (>= body diagonal) so the cut pierces end-to-end
        # from either side of the sketch plane.
        (mn_x, mn_y, mn_z), (mx_x, mx_y, mx_z) = bbox_tuple(body)
        diag_cm = (
            (mx_x - mn_x) ** 2 + (mx_y - mn_y) ** 2 + (mx_z - mn_z) ** 2
        ) ** 0.5
        over_distance_cm = max(diag_cm, 1.0) * 2.1  # 2.1× diag = pierces both sides
        ext_input.setSymmetricExtent(
            adsk.core.ValueInput.createByReal(over_distance_cm),
            True,  # isFullLength — means the value IS the full symmetric length
        )
    else:
        # Resolve which side of the sketch plane the body actually sits on.
        chosen_dir = _choose_cut_direction(body, profile_list[0])
        ext_input.setOneSideExtent(
            adsk.fusion.ThroughAllExtentDefinition.create(),
            chosen_dir,
        )

    return extrudes.add(ext_input)


# ─────────────────────────────────────────────────────────────────────────────
# Hole-layout planner — returns centre points (sketch-space Point3D) that are
# guaranteed to lie INSIDE the face's sketch-space bbox, respecting a margin.
# Agents use this instead of computing offsets from the body bbox, which mixes
# axes that don't lie on the face plane and is the direct cause of the
# "profile falls outside the boundary of the selected body" failure.
# ─────────────────────────────────────────────────────────────────────────────


def face_sketch_bbox(
    sketch: adsk.fusion.Sketch,
    face: adsk.fusion.BRepFace,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Return the face's bounding rectangle in the SKETCH's 2-D coordinate system
    (centimetres). ``((min_x, min_y), (max_x, max_y))``. Projects the face's
    3-D bbox corners through ``Sketch.modelToSketchSpace`` and takes the
    axis-aligned envelope.
    """
    face_bbox = face.boundingBox
    mn, mx = face_bbox.minPoint, face_bbox.maxPoint
    # 8 corners of the 3D bbox — any four of them reduce to a tight 2-D hull
    # when projected, but we use all 8 for robustness on rotated faces.
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
        p = sketch.modelToSketchSpace(c)
        xs.append(p.x)
        ys.append(p.y)
    return ((min(xs), min(ys)), (max(xs), max(ys)))


def plan_hole_layout(
    sketch: adsk.fusion.Sketch,
    face: adsk.fusion.BRepFace,
    rows: int,
    cols: int,
    *,
    hole_radius_mm: float,
    margin_mm: float = 2.0,
) -> List[adsk.core.Point3D]:
    """
    Return a list of sketch-space ``Point3D`` centres for a ``rows × cols``
    grid of holes, evenly spaced inside the face with a ``margin_mm`` safety
    border — so drawing circles of radius ``hole_radius_mm`` at these points
    is GUARANTEED to stay within the face.

    Raises ``ValueError`` if the face is too small to fit the requested grid
    at the given radius + margin, instead of letting the caller push a
    profile off the face and trigger ``EXTRUDE_BOOLEAN_FAIL``.
    """
    if rows < 1 or cols < 1:
        raise ValueError("plan_hole_layout: rows and cols must be >= 1")

    (mn_x, mn_y), (mx_x, mx_y) = face_sketch_bbox(sketch, face)
    # Convert mm → cm for Fusion's internal units.
    r_cm = hole_radius_mm / 10.0
    m_cm = margin_mm / 10.0
    inset = r_cm + m_cm

    usable_min_x = mn_x + inset
    usable_max_x = mx_x - inset
    usable_min_y = mn_y + inset
    usable_max_y = mx_y - inset
    if usable_max_x <= usable_min_x or usable_max_y <= usable_min_y:
        raise ValueError(
            f"plan_hole_layout: face is too small to fit {rows}×{cols} holes "
            f"of radius {hole_radius_mm}mm with margin {margin_mm}mm. "
            f"Face usable region: "
            f"{(usable_max_x - usable_min_x) * 10:.1f}×"
            f"{(usable_max_y - usable_min_y) * 10:.1f} mm."
        )

    def _axis_positions(n: int, lo: float, hi: float) -> List[float]:
        if n == 1:
            return [(lo + hi) * 0.5]
        step = (hi - lo) / (n - 1)
        return [lo + i * step for i in range(n)]

    xs = _axis_positions(cols, usable_min_x, usable_max_x)
    ys = _axis_positions(rows, usable_min_y, usable_max_y)

    centres: List[adsk.core.Point3D] = []
    for y in ys:
        for x in xs:
            centres.append(adsk.core.Point3D.create(x, y, 0.0))
    return centres

"""
Vetted Fusion 360 modify-feature helpers.

Every helper in this module wraps one of the failure-prone Fusion 360
``Features`` APIs in a way that:

  • Sets the required ``participantBodies`` / target on the input object —
    omitting these is the #1 cause of "body not found" / "EXTRUDE_BOOLEAN_FAIL"
    style errors.
  • Accepts both a Fusion entity AND a friendly string (``"+Z"``,
    ``"front"``, …) where appropriate, so generated code does not have to
    construct ``Vector3D`` objects manually.
  • Converts mm → cm at the boundary, since the agent reasons in mm but
    Fusion's API only accepts cm.
  • Raises ``ValueError`` / ``LookupError`` with specific, actionable
    messages BEFORE handing the input to Fusion's solver, so failures
    surface as clear errors instead of opaque runtime tracebacks.

All helpers run synchronously on the Fusion main thread.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import adsk.core
import adsk.fusion

from . import face_finder

logger = logging.getLogger(__name__)

_MM_TO_CM = 0.1
_CM_TO_MM = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _root_component(body_or_face: Union[adsk.fusion.BRepBody, adsk.fusion.BRepFace]) -> adsk.fusion.Component:
    """Return the parent component of a body or face."""
    if isinstance(body_or_face, adsk.fusion.BRepFace):
        return body_or_face.body.parentComponent
    return body_or_face.parentComponent


def _to_object_collection(items: Iterable) -> adsk.core.ObjectCollection:
    """Wrap any iterable of Fusion entities in an ``ObjectCollection``."""
    if isinstance(items, adsk.core.ObjectCollection):
        return items
    col = adsk.core.ObjectCollection.create()
    for it in items:
        if it is None:
            continue
        col.add(it)
    return col


def _value(distance_cm: float) -> adsk.core.ValueInput:
    """Shorthand for ``ValueInput.createByReal(distance_cm)``."""
    return adsk.core.ValueInput.createByReal(distance_cm)


def _resolve_body(
    body_or_id: Union[adsk.fusion.BRepBody, int, str],
    root_comp: Optional[adsk.fusion.Component] = None,
) -> adsk.fusion.BRepBody:
    """Accept a ``BRepBody``, a body_index (int), or a body name (str)."""
    if isinstance(body_or_id, adsk.fusion.BRepBody):
        return body_or_id
    if root_comp is None:
        raise ValueError("modify_tools: root_comp required when resolving body by index/name")
    return face_finder.body_by_index_or_name(root_comp, body_or_id)


# ─────────────────────────────────────────────────────────────────────────────
# Edge collection helpers — let agents grab the right edges by intent rather
# than iterating ``body.edges`` and writing fragile geometry tests by hand.
# ─────────────────────────────────────────────────────────────────────────────

def edges_of_face(face: adsk.fusion.BRepFace) -> List[adsk.fusion.BRepEdge]:
    """Every edge of ``face`` — useful for filleting / chamfering one face."""
    if face is None:
        raise ValueError("edges_of_face: face is None")
    return [face.edges.item(i) for i in range(face.edges.count)]


def all_edges_of_body(body: adsk.fusion.BRepBody) -> List[adsk.fusion.BRepEdge]:
    """Every edge of ``body`` — useful for "fillet all edges 2 mm" requests."""
    if body is None:
        raise ValueError("all_edges_of_body: body is None")
    return [body.edges.item(i) for i in range(body.edges.count)]


def _edge_direction_world(edge: adsk.fusion.BRepEdge) -> Optional[Tuple[float, float, float]]:
    """
    Return the unit direction vector of an edge in world space, or None if it
    isn't a straight line. Used to filter ``edges_by_direction``.
    """
    geom = getattr(edge, "geometry", None)
    if geom is None or "Line" not in getattr(geom, "objectType", ""):
        return None
    try:
        sp = edge.startVertex.geometry
        ep = edge.endVertex.geometry
    except Exception:
        return None
    dx, dy, dz = ep.x - sp.x, ep.y - sp.y, ep.z - sp.z
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    if mag < 1e-9:
        return None
    return (dx / mag, dy / mag, dz / mag)


def edges_by_direction(
    body: adsk.fusion.BRepBody,
    direction: Union[str, Iterable[float], adsk.core.Vector3D],
    *,
    tolerance_cos: float = 0.95,
    linear_only: bool = True,
) -> List[adsk.fusion.BRepEdge]:
    """
    Return every linear edge of ``body`` whose direction (start→end) is
    parallel to ``direction`` (or anti-parallel — sign-agnostic by design,
    since "vertical edges" includes both +Z and -Z runs).

    For "fillet all vertical edges of the cube", call
    ``modify_tools.edges_by_direction(body, "+Z")``.
    """
    if body is None:
        raise ValueError("edges_by_direction: body is None")
    tx, ty, tz = face_finder._normalize_direction(direction)
    matches: List[adsk.fusion.BRepEdge] = []
    for i in range(body.edges.count):
        edge = body.edges.item(i)
        d = _edge_direction_world(edge)
        if d is None:
            if linear_only:
                continue
            matches.append(edge)
            continue
        dot = abs(d[0] * tx + d[1] * ty + d[2] * tz)
        if dot >= tolerance_cos:
            matches.append(edge)
    return matches


def _coerce_edges(
    body: adsk.fusion.BRepBody,
    edges_or_spec: Union[
        adsk.fusion.BRepEdge,
        Iterable[adsk.fusion.BRepEdge],
        adsk.core.ObjectCollection,
        str,
    ],
) -> List[adsk.fusion.BRepEdge]:
    """
    Resolve any of:
      - a single BRepEdge
      - a list / ObjectCollection of edges
      - a direction string ("+Z", "vertical", "all")
    to a concrete list of BRepEdge objects of ``body``.
    """
    if isinstance(edges_or_spec, adsk.fusion.BRepEdge):
        return [edges_or_spec]
    if isinstance(edges_or_spec, adsk.core.ObjectCollection):
        return [edges_or_spec.item(i) for i in range(edges_or_spec.count)]
    if isinstance(edges_or_spec, str):
        key = edges_or_spec.strip().lower()
        if key in ("all", "every", "*"):
            return all_edges_of_body(body)
        if key in ("vertical", "z", "+z", "-z", "up"):
            return edges_by_direction(body, "+Z")
        if key in ("horizontal-x", "x", "+x", "-x"):
            return edges_by_direction(body, "+X")
        if key in ("horizontal-y", "y", "+y", "-y"):
            return edges_by_direction(body, "+Y")
        # Otherwise treat as a direction string and try a 1-axis filter.
        return edges_by_direction(body, edges_or_spec)
    return [e for e in edges_or_spec]


# ─────────────────────────────────────────────────────────────────────────────
# Fillet / Chamfer
#
# Pitfall the LLM hits: passing a Python list to ``addConstantRadiusEdgeSet``
# raises "Wrong number or type of arguments" because Fusion expects an
# ``ObjectCollection``. These helpers convert for you.
# ─────────────────────────────────────────────────────────────────────────────

def fillet_edges(
    body: adsk.fusion.BRepBody,
    edges_or_spec: Union[
        adsk.fusion.BRepEdge,
        Iterable[adsk.fusion.BRepEdge],
        adsk.core.ObjectCollection,
        str,
    ],
    radius_mm: float,
    *,
    tangent_chain: bool = True,
) -> adsk.fusion.FilletFeature:
    """
    Apply a constant-radius fillet to one or more edges of ``body``.

    Args:
        body:          The target body (the fillet feature is created on this
                       body's parent component).
        edges_or_spec: A single ``BRepEdge``, an iterable of edges, an
                       ``ObjectCollection``, or a direction-string like
                       ``"+Z"`` / ``"all"`` / ``"vertical"``.
        radius_mm:     Fillet radius in millimetres.
        tangent_chain: If True (default), Fusion automatically extends the
                       fillet along tangent-continuous edges.

    Raises ``ValueError`` if no edges resolve.
    """
    if body is None:
        raise ValueError("fillet_edges: body is None")
    if radius_mm <= 0:
        raise ValueError(f"fillet_edges: radius_mm must be > 0, got {radius_mm}")
    edges = _coerce_edges(body, edges_or_spec)
    if not edges:
        raise ValueError(
            f"fillet_edges: no edges resolved from {edges_or_spec!r} on body "
            f"{getattr(body, 'name', '?')}"
        )
    fillets = body.parentComponent.features.filletFeatures
    fillet_input = fillets.createInput()
    fillet_input.addConstantRadiusEdgeSet(
        _to_object_collection(edges),
        _value(radius_mm * _MM_TO_CM),
        tangent_chain,
    )
    return fillets.add(fillet_input)


def chamfer_edges(
    body: adsk.fusion.BRepBody,
    edges_or_spec: Union[
        adsk.fusion.BRepEdge,
        Iterable[adsk.fusion.BRepEdge],
        adsk.core.ObjectCollection,
        str,
    ],
    distance_mm: float,
    *,
    tangent_chain: bool = True,
) -> adsk.fusion.ChamferFeature:
    """
    Apply an equal-distance chamfer to one or more edges of ``body``.

    The Fusion API for chamfers is unusual:
        ChamferFeatures.createInput(edge_collection, isTangentChain)
    The CHAMFER DISTANCE goes on the ``ChamferFeatureInput`` afterwards via
    ``setToEqualDistance(value)``. Getting this order wrong yields
    "Wrong number or type of arguments" from SWIG; this helper fixes it.
    """
    if body is None:
        raise ValueError("chamfer_edges: body is None")
    if distance_mm <= 0:
        raise ValueError(f"chamfer_edges: distance_mm must be > 0, got {distance_mm}")
    edges = _coerce_edges(body, edges_or_spec)
    if not edges:
        raise ValueError(
            f"chamfer_edges: no edges resolved from {edges_or_spec!r} on body "
            f"{getattr(body, 'name', '?')}"
        )
    chamfers = body.parentComponent.features.chamferFeatures
    chamfer_input = chamfers.createInput(_to_object_collection(edges), tangent_chain)
    chamfer_input.setToEqualDistance(_value(distance_mm * _MM_TO_CM))
    return chamfers.add(chamfer_input)


# ─────────────────────────────────────────────────────────────────────────────
# Hole feature — Fusion's HoleFeatures gives a proper hole entity (with
# semantic depth/diameter) instead of a generic cut extrude. Use this when
# the user says "hole" specifically and you want it to appear in the timeline
# as a hole feature for downstream tooling (drilling specs, etc.).
# ─────────────────────────────────────────────────────────────────────────────

def add_simple_hole(
    target_face: adsk.fusion.BRepFace,
    center_world: adsk.core.Point3D,
    diameter_mm: float,
    *,
    depth_mm: Optional[float] = None,
) -> adsk.fusion.HoleFeature:
    """
    Add a single round hole at ``center_world`` on ``target_face``.

    Fusion's ``HoleFeatures.createSimpleInput`` takes a diameter; we then
    position it via ``setPositionByPoint(face, point)`` and set the depth
    (defaults to "through all" when ``depth_mm`` is None).

    Easier to read than a sketch+extrude, AND it shows up as a real Hole in
    the timeline, which downstream patterning / CAM tools recognise.
    """
    if target_face is None:
        raise ValueError("add_simple_hole: target_face is None")
    if diameter_mm <= 0:
        raise ValueError(f"add_simple_hole: diameter_mm must be > 0, got {diameter_mm}")

    holes = target_face.body.parentComponent.features.holeFeatures
    hole_input = holes.createSimpleInput(_value(diameter_mm * _MM_TO_CM))
    hole_input.setPositionByPoint(target_face, center_world)
    if depth_mm is None:
        hole_input.setAllExtent(adsk.fusion.ExtentDirections.NegativeExtentDirection)
    else:
        hole_input.setDistanceExtent(_value(depth_mm * _MM_TO_CM))
    return holes.add(hole_input)


# ─────────────────────────────────────────────────────────────────────────────
# Shell — hollow out a body, optionally removing one or more faces to make
# it open on those sides (typical "make a box hollow with the top open").
# ─────────────────────────────────────────────────────────────────────────────

def shell_body(
    body: adsk.fusion.BRepBody,
    thickness_mm: float,
    *,
    faces_to_remove: Optional[Iterable[adsk.fusion.BRepFace]] = None,
    inside: bool = True,
) -> adsk.fusion.ShellFeature:
    """
    Shell a body to ``thickness_mm`` wall, optionally opening it at the
    specified faces.

    Args:
        body:            The body to shell.
        thickness_mm:    Wall thickness in millimetres.
        faces_to_remove: Faces to delete (open sides). Empty / None ⇒ a fully
                         enclosed shell.
        inside:          True (default) shells inward; False shells outward.
    """
    if body is None:
        raise ValueError("shell_body: body is None")
    if thickness_mm <= 0:
        raise ValueError(f"shell_body: thickness_mm must be > 0, got {thickness_mm}")

    shells = body.parentComponent.features.shellFeatures
    input_collection = adsk.core.ObjectCollection.create()
    if faces_to_remove:
        for f in faces_to_remove:
            input_collection.add(f)
    else:
        # No faces to remove ⇒ shell the body itself (closed shell).
        input_collection.add(body)

    shell_input = shells.createInput(input_collection, False)
    shell_input.insideThickness = _value(thickness_mm * _MM_TO_CM) if inside else _value(0.0)
    if not inside:
        shell_input.outsideThickness = _value(thickness_mm * _MM_TO_CM)
    return shells.add(shell_input)


# ─────────────────────────────────────────────────────────────────────────────
# Combine bodies — boolean ops on already-existing bodies.
# ─────────────────────────────────────────────────────────────────────────────

_COMBINE_OPS = {
    "join":      adsk.fusion.FeatureOperations.JoinFeatureOperation,
    "add":       adsk.fusion.FeatureOperations.JoinFeatureOperation,
    "union":     adsk.fusion.FeatureOperations.JoinFeatureOperation,
    "cut":       adsk.fusion.FeatureOperations.CutFeatureOperation,
    "subtract":  adsk.fusion.FeatureOperations.CutFeatureOperation,
    "intersect": adsk.fusion.FeatureOperations.IntersectFeatureOperation,
}


def combine_bodies(
    target_body: adsk.fusion.BRepBody,
    tool_bodies: Union[adsk.fusion.BRepBody, Iterable[adsk.fusion.BRepBody]],
    *,
    operation: str = "join",
    keep_tools: bool = False,
) -> adsk.fusion.CombineFeature:
    """
    Combine bodies via boolean op. ``operation`` accepts ``"join"`` /
    ``"cut"`` / ``"intersect"`` (and the synonyms ``add``, ``union``,
    ``subtract``).
    """
    if target_body is None:
        raise ValueError("combine_bodies: target_body is None")
    op = _COMBINE_OPS.get(operation.strip().lower())
    if op is None:
        raise ValueError(
            f"combine_bodies: unknown operation {operation!r}. Expected one of: "
            f"{sorted(_COMBINE_OPS.keys())}"
        )
    tools = (
        [tool_bodies]
        if isinstance(tool_bodies, adsk.fusion.BRepBody)
        else list(tool_bodies)
    )
    if not tools:
        raise ValueError("combine_bodies: tool_bodies is empty")

    combines = target_body.parentComponent.features.combineFeatures
    combine_input = combines.createInput(target_body, _to_object_collection(tools))
    combine_input.operation = op
    combine_input.isKeepToolBodies = keep_tools
    return combines.add(combine_input)


# ─────────────────────────────────────────────────────────────────────────────
# Move / Rotate body — uses MoveFeatures, the only modify path that works for
# parametric bodies. Setting body.transform directly is a NON-PARAMETRIC move
# (won't replay correctly when the design is rebuilt) and is a frequent
# source of "the move disappeared after editing an upstream feature" bugs.
# ─────────────────────────────────────────────────────────────────────────────

def move_body(
    body: adsk.fusion.BRepBody,
    *,
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    dz_mm: float = 0.0,
) -> adsk.fusion.MoveFeature:
    """
    Translate ``body`` by (dx, dy, dz) millimetres as a parametric MoveFeature.
    """
    if body is None:
        raise ValueError("move_body: body is None")
    if dx_mm == 0 and dy_mm == 0 and dz_mm == 0:
        raise ValueError("move_body: zero translation has no effect")

    moves = body.parentComponent.features.moveFeatures
    bodies_col = adsk.core.ObjectCollection.create()
    bodies_col.add(body)
    transform = adsk.core.Matrix3D.create()
    transform.translation = adsk.core.Vector3D.create(
        dx_mm * _MM_TO_CM, dy_mm * _MM_TO_CM, dz_mm * _MM_TO_CM
    )
    move_input = moves.createInput2(bodies_col)
    move_input.defineAsFreeMove(transform)
    return moves.add(move_input)


# ─────────────────────────────────────────────────────────────────────────────
# Patterns — circular and rectangular. A common request "4 holes around the
# centre of the face" is a circular pattern of one hole feature, NOT four
# hand-laid sketches.
# ─────────────────────────────────────────────────────────────────────────────

def _features_collection(
    features: Union[adsk.fusion.Feature, Iterable[adsk.fusion.Feature]],
) -> adsk.core.ObjectCollection:
    if isinstance(features, adsk.fusion.Feature):
        col = adsk.core.ObjectCollection.create()
        col.add(features)
        return col
    return _to_object_collection(features)


def circular_pattern(
    features: Union[adsk.fusion.Feature, Iterable[adsk.fusion.Feature]],
    axis: Union[adsk.fusion.ConstructionAxis, adsk.fusion.BRepEdge, adsk.fusion.BRepFace],
    count: int,
    *,
    total_angle_deg: float = 360.0,
    component: Optional[adsk.fusion.Component] = None,
) -> adsk.fusion.CircularPatternFeature:
    """
    Circular-pattern one or more features around ``axis``.

    Args:
        features:        Feature(s) to copy. The most common case is a single
                         hole or extrude feature.
        axis:            A ConstructionAxis, a linear BRepEdge, or a planar
                         BRepFace (its normal is used as the axis).
        count:           Number of copies including the original (e.g. 4 for
                         a 4-hole bolt circle).
        total_angle_deg: Sweep angle in degrees (default 360°).
    """
    if count < 2:
        raise ValueError(f"circular_pattern: count must be >= 2, got {count}")
    feat_col = _features_collection(features)
    if feat_col.count == 0:
        raise ValueError("circular_pattern: features collection is empty")

    if component is None:
        sample = feat_col.item(0)
        component = sample.parentComponent or feat_col.item(0).parentComponent

    patterns = component.features.circularPatternFeatures
    pattern_input = patterns.createInput(feat_col, axis)
    pattern_input.quantity = _value(float(count))
    pattern_input.totalAngle = adsk.core.ValueInput.createByString(f"{total_angle_deg} deg")
    return patterns.add(pattern_input)


def rectangular_pattern(
    features: Union[adsk.fusion.Feature, Iterable[adsk.fusion.Feature]],
    direction_one: Union[adsk.fusion.BRepEdge, adsk.fusion.ConstructionAxis],
    count_one: int,
    distance_one_mm: float,
    *,
    direction_two: Optional[Union[adsk.fusion.BRepEdge, adsk.fusion.ConstructionAxis]] = None,
    count_two: int = 1,
    distance_two_mm: float = 0.0,
    component: Optional[adsk.fusion.Component] = None,
) -> adsk.fusion.RectangularPatternFeature:
    """
    Rectangular-pattern one or more features along ``direction_one`` (and
    optionally ``direction_two``).

    Distances are TOTAL pattern lengths (not pitch) — Fusion's API expects
    the distance from first to last instance.
    """
    if count_one < 2:
        raise ValueError(f"rectangular_pattern: count_one must be >= 2, got {count_one}")
    feat_col = _features_collection(features)
    if feat_col.count == 0:
        raise ValueError("rectangular_pattern: features collection is empty")

    if component is None:
        sample = feat_col.item(0)
        component = sample.parentComponent

    patterns = component.features.rectangularPatternFeatures
    distance_type = adsk.fusion.PatternDistanceType.ExtentPatternDistanceType
    pattern_input = patterns.createInput(
        feat_col,
        direction_one,
        _value(float(count_one)),
        _value(distance_one_mm * _MM_TO_CM),
        distance_type,
    )
    if direction_two is not None and count_two > 1:
        pattern_input.setDirectionTwo(
            direction_two,
            _value(float(count_two)),
            _value(distance_two_mm * _MM_TO_CM),
        )
    return patterns.add(pattern_input)


# ─────────────────────────────────────────────────────────────────────────────
# Mirror — about a face or construction plane.
# ─────────────────────────────────────────────────────────────────────────────

def mirror_features(
    features: Union[adsk.fusion.Feature, Iterable[adsk.fusion.Feature]],
    mirror_plane: Union[adsk.fusion.BRepFace, adsk.fusion.ConstructionPlane],
    *,
    component: Optional[adsk.fusion.Component] = None,
) -> adsk.fusion.MirrorFeature:
    """Mirror feature(s) across ``mirror_plane`` (a face or construction plane)."""
    feat_col = _features_collection(features)
    if feat_col.count == 0:
        raise ValueError("mirror_features: features collection is empty")
    if component is None:
        sample = feat_col.item(0)
        component = sample.parentComponent
    mirrors = component.features.mirrorFeatures
    mirror_input = mirrors.createInput(feat_col, mirror_plane)
    return mirrors.add(mirror_input)


# ─────────────────────────────────────────────────────────────────────────────
# Boss / Join extrude — adding material to an existing body. Mirror image of
# face_finder.cut_through_body. Centralises the participantBodies setting.
# ─────────────────────────────────────────────────────────────────────────────

def extrude_join(
    target_body: adsk.fusion.BRepBody,
    profile: Union[adsk.fusion.Profile, adsk.core.ObjectCollection, Iterable[adsk.fusion.Profile]],
    distance_mm: float,
    *,
    direction: str = "outward",
    target_face: Optional[adsk.fusion.BRepFace] = None,
) -> adsk.fusion.ExtrudeFeature:
    """
    Extrude a profile and JOIN the result onto ``target_body``.

    Args:
        target_body:  The body to add material to. participantBodies is set
                      to [target_body] so Fusion fuses the new volume in.
        profile:      Single Profile, ObjectCollection, or iterable.
        distance_mm:  Extrusion distance in millimetres.
        direction:    ``"outward"`` (default) — extrude AWAY from the body
                      along the sketch normal. ``"inward"`` — extrude INTO
                      the body (rarely useful for a Join, but exposed).
                      Also accepts ``"positive"`` / ``"negative"`` for
                      explicit sketch-frame control.
        target_face:  Optional. When supplied, the profile is validated to
                      lie inside the face's sketch-space bounds before the
                      extrude is sent to Fusion (mirrors ``cut_through_body``).
    """
    if target_body is None:
        raise ValueError("extrude_join: target_body is None")
    if distance_mm <= 0:
        raise ValueError(f"extrude_join: distance_mm must be > 0, got {distance_mm}")

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
        raise ValueError("extrude_join: profile is empty")

    if target_face is not None:
        for i, p in enumerate(profile_list):
            if not face_finder._profile_is_inside_face(p, target_face):
                raise ValueError(
                    f"extrude_join: profile {i} is OUTSIDE the target face — "
                    "Fusion would silently fail or produce disconnected geometry. "
                    "Use face_finder.plan_hole_layout(...) or shrink the profile."
                )

    extrudes = target_body.parentComponent.features.extrudeFeatures
    ext_input = extrudes.createInput(
        prof_arg, adsk.fusion.FeatureOperations.JoinFeatureOperation
    )
    ext_input.participantBodies = [target_body]

    # Direction resolution: "outward" / "inward" use pointContainment to pick
    # the side that AVOIDS the body; explicit positive/negative pass through.
    direction = direction.strip().lower()
    if direction in ("positive", "+"):
        ext_dir = adsk.fusion.ExtentDirections.PositiveExtentDirection
    elif direction in ("negative", "-"):
        ext_dir = adsk.fusion.ExtentDirections.NegativeExtentDirection
    else:
        # Reuse the cut-direction logic but invert it — for a Join we want to
        # extrude AWAY from the body when "outward", and INTO the body when
        # "inward".
        cut_dir = face_finder._choose_cut_direction(target_body, profile_list[0])
        if direction == "inward":
            ext_dir = cut_dir
        else:  # outward (default)
            ext_dir = (
                adsk.fusion.ExtentDirections.PositiveExtentDirection
                if cut_dir == adsk.fusion.ExtentDirections.NegativeExtentDirection
                else adsk.fusion.ExtentDirections.NegativeExtentDirection
            )

    # Apply ext_dir via the sign of the distance value. setDistanceExtent does
    # NOT take a direction argument — Fusion always extrudes along the
    # positive sketch normal unless you pass a negative distance. The earlier
    # implementation only flipped the sign for explicit negative/inward, so
    # an "outward" Join whose ext_dir resolved to Negative (sketch normal
    # points into the body) was silently extruded INTO the body instead of
    # away from it. Always honour ext_dir now.
    extent_value_cm = distance_mm * _MM_TO_CM
    if ext_dir == adsk.fusion.ExtentDirections.NegativeExtentDirection:
        extent_value_cm = -extent_value_cm
    ext_input.setDistanceExtent(False, _value(extent_value_cm))

    return extrudes.add(ext_input)


# ─────────────────────────────────────────────────────────────────────────────
# Circular hole layout (radial pattern alternative when the user gives just
# a centre + N + radius). Returns sketch-space Point3Ds; pair with
# face_finder.cut_through_body for the actual cut.
# ─────────────────────────────────────────────────────────────────────────────

def plan_circular_hole_layout(
    sketch: adsk.fusion.Sketch,
    face: adsk.fusion.BRepFace,
    n_holes: int,
    *,
    radius_from_center_mm: float,
    hole_radius_mm: float,
    margin_mm: float = 2.0,
    start_angle_deg: float = 0.0,
) -> List[adsk.core.Point3D]:
    """
    Return ``n_holes`` sketch-space ``Point3D`` centres equally spaced on a
    circle of radius ``radius_from_center_mm`` around the face centroid in
    sketch coords.

    Validates the resulting circles fit inside the face's sketch-space bbox
    with ``margin_mm`` clearance — raises ``ValueError`` otherwise so the
    caller can shrink the radius or count rather than letting Fusion's
    boolean engine fail with EXTRUDE_BOOLEAN_FAIL.
    """
    if n_holes < 1:
        raise ValueError("plan_circular_hole_layout: n_holes must be >= 1")
    (mn_x, mn_y), (mx_x, mx_y) = face_finder.face_sketch_bbox(sketch, face)
    cx = (mn_x + mx_x) * 0.5
    cy = (mn_y + mx_y) * 0.5

    r_layout_cm = radius_from_center_mm * _MM_TO_CM
    r_hole_cm = hole_radius_mm * _MM_TO_CM
    margin_cm = margin_mm * _MM_TO_CM

    # Each hole occupies (centre ± r_hole). The outermost x/y reached is
    # cx + r_layout + r_hole. Must be inside (mx_x - margin).
    outer = r_layout_cm + r_hole_cm + margin_cm
    if (cx + outer) > mx_x or (cx - outer) < mn_x or (cy + outer) > mx_y or (cy - outer) < mn_y:
        raise ValueError(
            f"plan_circular_hole_layout: a {n_holes}-hole circle of layout "
            f"radius {radius_from_center_mm}mm with hole radius "
            f"{hole_radius_mm}mm and margin {margin_mm}mm does not fit "
            f"inside the face. Face usable region: "
            f"{(mx_x - mn_x) * _CM_TO_MM:.1f}×{(mx_y - mn_y) * _CM_TO_MM:.1f} mm."
        )

    centres: List[adsk.core.Point3D] = []
    for i in range(n_holes):
        theta = math.radians(start_angle_deg + (360.0 / n_holes) * i)
        x = cx + r_layout_cm * math.cos(theta)
        y = cy + r_layout_cm * math.sin(theta)
        centres.append(adsk.core.Point3D.create(x, y, 0.0))
    return centres


# ─────────────────────────────────────────────────────────────────────────────
# Construction-axis helpers — patterns and mirrors need an axis or plane,
# and the LLM otherwise hand-constructs them and gets the API wrong.
# ─────────────────────────────────────────────────────────────────────────────

def construction_axis_through_face(
    face: adsk.fusion.BRepFace,
) -> adsk.fusion.ConstructionAxis:
    """
    Build a construction axis through the centre of ``face`` along its
    normal. Useful as the rotational axis for ``circular_pattern``.
    """
    if face is None:
        raise ValueError("construction_axis_through_face: face is None")
    component = face.body.parentComponent
    axes = component.constructionAxes
    axis_input = axes.createInput()
    axis_input.setByNormalToFaceAtPoint(face, face.pointOnFace)
    return axes.add(axis_input)


__all__ = [
    "edges_of_face",
    "all_edges_of_body",
    "edges_by_direction",
    "fillet_edges",
    "chamfer_edges",
    "add_simple_hole",
    "shell_body",
    "combine_bodies",
    "move_body",
    "circular_pattern",
    "rectangular_pattern",
    "mirror_features",
    "extrude_join",
    "plan_circular_hole_layout",
    "construction_axis_through_face",
]

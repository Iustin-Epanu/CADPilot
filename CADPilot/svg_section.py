"""
Tier 2 — SVG Cross-Section Extraction.

Slices the active design at a named construction plane (XY / XZ / YZ)
with an optional offset, projects the intersection edges into a temporary
Fusion sketch, converts every sketch curve to SVG path primitives, then
deletes the scratch sketch before returning.

The LLM can read the resulting SVG text as exact 2-D math: circle radii,
arc centres, line endpoints — no ambiguity from perspective projection.

All output coordinates are in millimetres (mm).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import adsk.core
import adsk.fusion

logger = logging.getLogger(__name__)

_CM_TO_MM = 10.0
_SKETCH_NAME = "_cadpilot_svg_tmp"


class SvgSectionError(RuntimeError):
    """Raised when SVG cross-section extraction cannot proceed."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_svg_section(
    app: adsk.core.Application,
    plane_axis: str = "XY",
    offset_mm: float = 0.0,
) -> Dict[str, Any]:
    """
    Extract an SVG cross-section of the active design at a given plane.

    Args:
        app:        Fusion 360 Application instance.
        plane_axis: One of "XY", "XZ", "YZ".
        offset_mm:  Offset from the named plane along its normal (mm).

    Returns a dict:
        svg         – Full SVG markup string (empty if no geometry found).
        viewBox     – [min_x, min_y, width, height] in mm.
        curve_count – Number of primitive curves extracted.
        plane_axis  – Echo of input.
        offset_mm   – Echo of input.
        warning     – Present only when no curves were found.
    """
    design = _require_design(app)
    root = design.rootComponent

    # ── 1. Resolve cutting plane ──────────────────────────────────────────
    base_plane = _get_named_plane(root, plane_axis)
    cutting_plane: Any = base_plane
    offset_feat = None

    if abs(offset_mm) > 1e-6:
        try:
            inp = root.constructionPlanes.createInput()
            inp.setByOffset(
                base_plane,
                adsk.core.ValueInput.createByReal(offset_mm / _CM_TO_MM),
            )
            offset_feat = root.constructionPlanes.add(inp)
            cutting_plane = offset_feat
        except Exception as exc:
            logger.warning("Could not create offset plane: %s — using base plane", exc)

    # ── 2. Create temp sketch ─────────────────────────────────────────────
    sketch: adsk.fusion.Sketch = root.sketches.add(cutting_plane)
    sketch.name = _SKETCH_NAME

    curves: List[Dict[str, Any]] = []

    try:
        # ── 3. Project cut edges for every body ───────────────────────────
        bodies = [b for b in root.bRepBodies if b.isSolid]
        if not bodies:
            bodies = list(root.bRepBodies)

        for body in bodies:
            try:
                sketch.projectCutEdges(body)
            except Exception as exc:
                logger.debug("projectCutEdges skipped body '%s': %s", body.name, exc)

        # ── 4. Extract sketch curves ──────────────────────────────────────
        curves = _extract_curves(sketch)

    finally:
        # ── 5. Clean up temp sketch and optional offset plane ─────────────
        try:
            sketch.deleteMe()
        except Exception as exc:
            logger.debug("Failed to delete temp sketch: %s", exc)
        if offset_feat:
            try:
                offset_feat.deleteMe()
            except Exception as exc:
                logger.debug("Failed to delete offset plane: %s", exc)

    if not curves:
        return {
            "svg": "",
            "viewBox": [],
            "curve_count": 0,
            "plane_axis": plane_axis,
            "offset_mm": offset_mm,
            "warning": "No cross-section geometry found at this plane.",
        }

    # ── 6. Build SVG ──────────────────────────────────────────────────────
    svg_text, view_box = _build_svg(curves)

    return {
        "svg": svg_text,
        "viewBox": view_box,
        "curve_count": len(curves),
        "plane_axis": plane_axis,
        "offset_mm": offset_mm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Curve extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_curves(sketch: adsk.fusion.Sketch) -> List[Dict[str, Any]]:
    """Walk all sketch curve collections and normalise to plain dicts (mm)."""
    out: List[Dict[str, Any]] = []

    # Lines
    try:
        for line in sketch.sketchCurves.sketchLines:
            try:
                s = line.startSketchPoint.geometry
                e = line.endSketchPoint.geometry
                out.append({
                    "type": "line",
                    "x1": s.x * _CM_TO_MM, "y1": s.y * _CM_TO_MM,
                    "x2": e.x * _CM_TO_MM, "y2": e.y * _CM_TO_MM,
                })
            except Exception as exc:
                logger.debug("Skipping line: %s", exc)
    except Exception:
        pass

    # Arcs
    try:
        for arc in sketch.sketchCurves.sketchArcs:
            try:
                cp = arc.centerSketchPoint.geometry
                r = arc.radius * _CM_TO_MM
                sp = arc.startSketchPoint.geometry
                ep = arc.endSketchPoint.geometry
                sa = math.atan2(sp.y - cp.y, sp.x - cp.x)
                ea = math.atan2(ep.y - cp.y, ep.x - cp.x)
                out.append({
                    "type": "arc",
                    "cx": cp.x * _CM_TO_MM, "cy": cp.y * _CM_TO_MM,
                    "r": r,
                    "x1": sp.x * _CM_TO_MM, "y1": sp.y * _CM_TO_MM,
                    "x2": ep.x * _CM_TO_MM, "y2": ep.y * _CM_TO_MM,
                    "start_deg": math.degrees(sa),
                    "end_deg": math.degrees(ea),
                })
            except Exception as exc:
                logger.debug("Skipping arc: %s", exc)
    except Exception:
        pass

    # Circles
    try:
        for circle in sketch.sketchCurves.sketchCircles:
            try:
                cp = circle.centerSketchPoint.geometry
                r = circle.radius * _CM_TO_MM
                out.append({
                    "type": "circle",
                    "cx": cp.x * _CM_TO_MM, "cy": cp.y * _CM_TO_MM,
                    "r": r,
                })
            except Exception as exc:
                logger.debug("Skipping circle: %s", exc)
    except Exception:
        pass

    # Fitted splines (approximated by fit-point polyline)
    try:
        for spline in sketch.sketchCurves.sketchFittedSplines:
            try:
                pts: List[List[float]] = []
                for fp in spline.fitPoints:
                    g = fp.geometry
                    pts.append([g.x * _CM_TO_MM, g.y * _CM_TO_MM])
                if len(pts) >= 2:
                    out.append({"type": "spline", "fit_points": pts})
            except Exception as exc:
                logger.debug("Skipping spline: %s", exc)
    except Exception:
        pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
# SVG builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_svg(curves: List[Dict[str, Any]]) -> Tuple[str, List[float]]:
    """
    Build an SVG string from the extracted 2-D curves.

    SVG Y axis points down; Fusion sketch Y points up.  We negate Y
    so the geometry appears correct in any SVG viewer.

    Returns (svg_string, [vb_min_x, vb_min_y, vb_width, vb_height]).
    """
    xs: List[float] = []
    ys: List[float] = []

    for c in curves:
        t = c["type"]
        if t == "line":
            xs += [c["x1"], c["x2"]]
            ys += [c["y1"], c["y2"]]
        elif t == "arc":
            xs += [c["x1"], c["x2"], c["cx"] - c["r"], c["cx"] + c["r"]]
            ys += [c["y1"], c["y2"], c["cy"] - c["r"], c["cy"] + c["r"]]
        elif t == "circle":
            xs += [c["cx"] - c["r"], c["cx"] + c["r"]]
            ys += [c["cy"] - c["r"], c["cy"] + c["r"]]
        elif t == "spline":
            for p in c.get("fit_points", []):
                xs.append(p[0])
                ys.append(p[1])

    if not xs:
        return "", []

    margin = 5.0
    min_x = min(xs) - margin
    min_y = min(ys) - margin
    max_x = max(xs) + margin
    max_y = max(ys) + margin
    w = max_x - min_x
    h = max_y - min_y

    # After negating Y, y → -y.  So original y_max becomes -y_max (the new SVG min_y).
    svg_vy_min = -max_y

    elements: List[str] = []

    for c in curves:
        t = c["type"]

        if t == "line":
            elements.append(
                f'<line x1="{c["x1"]:.3f}" y1="{-c["y1"]:.3f}" '
                f'x2="{c["x2"]:.3f}" y2="{-c["y2"]:.3f}"/>'
            )

        elif t == "circle":
            elements.append(
                f'<circle cx="{c["cx"]:.3f}" cy="{-c["cy"]:.3f}" r="{c["r"]:.3f}"/>'
            )

        elif t == "arc":
            large, sweep = _arc_flags(c["start_deg"], c["end_deg"])
            elements.append(
                f'<path d="M {c["x1"]:.3f} {-c["y1"]:.3f} '
                f'A {c["r"]:.3f} {c["r"]:.3f} 0 {large} {sweep} '
                f'{c["x2"]:.3f} {-c["y2"]:.3f}"/>'
            )

        elif t == "spline":
            pts = c.get("fit_points", [])
            if len(pts) >= 2:
                d = f"M {pts[0][0]:.3f} {-pts[0][1]:.3f}" + "".join(
                    f" L {p[0]:.3f} {-p[1]:.3f}" for p in pts[1:]
                )
                elements.append(f'<path d="{d}"/>')

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{w:.1f}mm" height="{h:.1f}mm"',
        f'     viewBox="{min_x:.3f} {svg_vy_min:.3f} {w:.3f} {h:.3f}">',
        '  <g stroke="black" stroke-width="0.5" fill="none">',
    ] + [f"    {el}" for el in elements] + ["  </g>", "</svg>"]

    view_box = [round(min_x, 3), round(min_y, 3), round(w, 3), round(h, 3)]
    return "\n".join(lines), view_box


def _arc_flags(start_deg: float, end_deg: float) -> Tuple[int, int]:
    """Return (large-arc-flag, sweep-flag) for an SVG A command."""
    delta = (end_deg - start_deg) % 360.0
    large = 1 if delta > 180.0 else 0
    return large, 1  # sweep=1: counter-clockwise in screen space (Y-flipped)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_design(app: adsk.core.Application) -> adsk.fusion.Design:
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise SvgSectionError("No active Fusion 360 design")
    return design


def _get_named_plane(
    component: adsk.fusion.Component,
    axis: str,
) -> adsk.fusion.ConstructionPlane:
    a = axis.strip().upper()
    if a == "XY":
        return component.xYConstructionPlane
    if a == "XZ":
        return component.xZConstructionPlane
    if a == "YZ":
        return component.yZConstructionPlane
    logger.warning("Unknown plane axis '%s'; defaulting to XY", axis)
    return component.xYConstructionPlane


__all__ = ["extract_svg_section", "SvgSectionError"]

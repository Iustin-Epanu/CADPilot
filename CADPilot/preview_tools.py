"""
Stage 3 / C1+C2 — Preview helpers for the clarification dialogue and
optional pre-coder preview.

The clarification widget needs a way to show the user *which* entity an
option corresponds to. ``highlight_entity`` resolves a stable handle (or
entityToken) to the live BRepEntity, pushes it into the active selection
set so it lights up in the viewport, optionally captures a screenshot for
the widget to embed, then restores the prior selection.

All operations run on the Fusion main thread (callers marshal via the
custom-event mechanism CADAgent already uses for probes).

Coordinates / dimensions are reported in millimetres (mm) — Fusion's
internal cm × 10 — to match the rest of the agent's vocabulary.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import adsk.core
import adsk.fusion

from . import camera_tools
from .body_tools import compute_body_handle

logger = logging.getLogger(__name__)


def _require_design(app: adsk.core.Application) -> adsk.fusion.Design:
    design = adsk.fusion.Design.cast(app.activeProduct)
    if not design:
        raise RuntimeError("preview_tools: no active Fusion 360 design")
    return design


def _resolve_handle_to_entity(
    app: adsk.core.Application,
    handle: str,
) -> Optional[Any]:
    """Resolve a stable handle to the corresponding BRep entity.

    Accepts:
      - body_handle ("b_a39f1234"): looks up by sha1-prefix of entityToken.
      - raw entityToken: passed through `Design.findEntityByToken`.
      - "<body_handle>:<kind>:<index>": resolves to a face/edge/vertex of
        the named body (e.g. ``"b_a39f1234:face:7"``). Useful for the
        clarification widget's per-face options.
    """
    if not handle:
        return None
    design = _require_design(app)
    root = design.rootComponent

    # Compound handle?  body:kind:index
    if handle.startswith("b_") and ":" in handle:
        try:
            body_part, kind, index_str = handle.split(":", 2)
            index = int(index_str)
        except ValueError:
            body_part, kind, index = handle, "", -1
    else:
        body_part, kind, index = handle, "", -1

    # Body lookup by handle
    if body_part.startswith("b_"):
        for i in range(root.bRepBodies.count):
            body = root.bRepBodies.item(i)
            token = getattr(body, "entityToken", None)
            if compute_body_handle(token, i) == body_part:
                if kind == "face" and 0 <= index < body.faces.count:
                    return body.faces.item(index)
                if kind == "edge" and 0 <= index < body.edges.count:
                    return body.edges.item(index)
                if kind == "vertex" and 0 <= index < body.vertices.count:
                    return body.vertices.item(index)
                return body
        # Fall through — body_part might also be a stale entityToken.

    # Try as raw entityToken.
    try:
        found = design.findEntityByToken(handle)
        if found:
            return found[0]
    except Exception as exc:
        logger.debug("preview_tools: findEntityByToken(%r) failed: %s", handle, exc)
    return None


def _save_active_selection(app: adsk.core.Application) -> List[Any]:
    """Snapshot the current active selection so we can restore it later."""
    saved: List[Any] = []
    try:
        ui = app.userInterface
        if not ui:
            return saved
        sel = ui.activeSelections
        for i in range(sel.count):
            try:
                saved.append(sel.item(i).entity)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("preview_tools: failed to snapshot selection: %s", exc)
    return saved


def _restore_active_selection(app: adsk.core.Application, saved: List[Any]) -> None:
    try:
        ui = app.userInterface
        if not ui:
            return
        ui.activeSelections.clear()
        for ent in saved:
            try:
                ui.activeSelections.add(ent)
            except Exception:
                continue
    except Exception as exc:
        logger.debug("preview_tools: failed to restore selection: %s", exc)


def highlight_entity(
    app: adsk.core.Application,
    handle: str,
    *,
    capture_screenshot: bool = True,
    fit: bool = True,
    settle_seconds: float = 0.15,
    restore_selection: bool = True,
    max_screenshot_width: int = 1024,
) -> Dict[str, Any]:
    """Highlight an entity in the viewport (and optionally screenshot it).

    Args:
        handle: body_handle, raw entityToken, or compound "<body>:kind:idx".
        capture_screenshot: if True, return a base64 PNG of the viewport
            with the entity selected (for embedding into the clarification
            widget).
        fit: if True, ``viewport.fit()`` to frame the highlighted entity.
            Generally desirable for the widget — the user sees the option
            centred.
        settle_seconds: short pause after selection so the viewport repaints
            before screenshot. 0.15 s matches the existing camera_tools idiom.
        restore_selection: if True, restore the user's prior selection set
            before returning. Set False for the C2 pre-coder preview where
            we want the highlight to persist into code execution.

    Returns:
        ``{success, handle, kind, image_base64?, error?}`` — ``image_base64``
        is present only when ``capture_screenshot=True`` succeeded.
    """
    result: Dict[str, Any] = {"success": False, "handle": handle}

    try:
        entity = _resolve_handle_to_entity(app, handle)
        if entity is None:
            result["error"] = f"highlight_entity: handle {handle!r} did not resolve"
            return result
        result["kind"] = type(entity).__name__

        ui = app.userInterface
        if not ui:
            result["error"] = "highlight_entity: no userInterface available"
            return result

        saved = _save_active_selection(app) if restore_selection else []

        try:
            ui.activeSelections.clear()
            ui.activeSelections.add(entity)
        except Exception as exc:
            result["error"] = f"highlight_entity: could not select entity: {exc}"
            if restore_selection:
                _restore_active_selection(app, saved)
            return result

        viewport = app.activeViewport
        if fit and viewport is not None:
            try:
                viewport.fit()
                viewport.refresh()
            except Exception:
                pass

        if settle_seconds > 0:
            try:
                time.sleep(settle_seconds)
            except Exception:
                pass

        if capture_screenshot:
            shot = camera_tools.capture_viewport_screenshot_base64(
                app, max_width=max_screenshot_width,
            )
            if shot:
                result["image_base64"] = shot

        if restore_selection:
            _restore_active_selection(app, saved)

        result["success"] = True
        return result

    except Exception as exc:
        logger.exception("highlight_entity failed: %s", exc)
        result["error"] = str(exc)
        if restore_selection:
            _restore_active_selection(app, _save_active_selection(app))
        return result


def flash_face(
    app: adsk.core.Application,
    handle: str,
    *,
    flashes: int = 2,
    on_seconds: float = 0.25,
    off_seconds: float = 0.15,
) -> Dict[str, Any]:
    """Briefly toggle the highlight of an entity to draw the eye.

    Used by the clarification widget after the user clicks an option to
    confirm the locked-in pick visually before the planner regenerates.
    Caller's prior selection set is preserved.
    """
    result: Dict[str, Any] = {"success": False, "handle": handle}

    try:
        entity = _resolve_handle_to_entity(app, handle)
        if entity is None:
            result["error"] = f"flash_face: handle {handle!r} did not resolve"
            return result

        ui = app.userInterface
        if not ui:
            result["error"] = "flash_face: no userInterface available"
            return result

        saved = _save_active_selection(app)
        try:
            for _ in range(max(1, int(flashes))):
                ui.activeSelections.clear()
                ui.activeSelections.add(entity)
                try:
                    app.activeViewport.refresh()
                except Exception:
                    pass
                time.sleep(max(0.0, on_seconds))
                ui.activeSelections.clear()
                try:
                    app.activeViewport.refresh()
                except Exception:
                    pass
                time.sleep(max(0.0, off_seconds))
        finally:
            _restore_active_selection(app, saved)

        result["success"] = True
        return result

    except Exception as exc:
        logger.exception("flash_face failed: %s", exc)
        result["error"] = str(exc)
        return result


__all__ = ["highlight_entity", "flash_face"]

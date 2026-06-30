"""
Execute Python snippets received from the backend inside the Fusion 360 context.

The executor provides a scoped namespace with key Fusion objects and keeps a registry
of created sketches so subsequent operations can reference them directly by ID.
"""

from __future__ import annotations

import logging
import re
import traceback
from typing import Any, Dict, Optional

import adsk.core
import adsk.fusion

from . import camera_tools
from . import code_validator
from . import face_finder
from . import modify_tools
from . import plane_manager as plane_manager_module

logger = logging.getLogger(__name__)

# Stage 4 / D2 — recover the resolver's target_body_handle from the
# generated script so the validator can pre-resolve face_by_direction
# against the right body. Coders are taught to emit either
#   target_body_handle = "b_..."
# OR call body_by_handle(rootComp, "b_...") directly.
_TARGET_HANDLE_ASSIGN_RE = re.compile(
    r'\btarget_body_handle\s*=\s*[\'"]([^\'"]+)[\'"]'
)
_BODY_BY_HANDLE_LITERAL_RE = re.compile(
    r'\bface_finder\.body_by_handle\(\s*\w+\s*,\s*[\'"]([^\'"]+)[\'"]\s*\)'
)


def _infer_target_body_handle(code: str) -> Optional[str]:
    """Best-effort: pull a literal body handle out of the generated code so
    the validator can pre-resolve face_by_direction calls. Returns None
    when the script uses a runtime-supplied handle (unlikely after Stage 2)."""
    m = _TARGET_HANDLE_ASSIGN_RE.search(code)
    if m:
        return m.group(1)
    m = _BODY_BY_HANDLE_LITERAL_RE.search(code)
    if m:
        return m.group(1)
    return None


class CodeExecutor:
    """Executes backend-supplied Python code scoped per Fusion document."""

    def __init__(self, app: adsk.core.Application):
        self._app = app
        # doc_id -> dict(name->Sketch) registry
        self._sketch_registries: Dict[str, Dict[str, adsk.fusion.Sketch]] = {}
        # doc_id -> PlaneManager registry
        self._plane_managers: Dict[str, plane_manager_module.PlaneManager] = {}

    def reset_context(self, doc_id: str) -> None:
        """Clear cached sketch registry and plane manager for a document (e.g., when it closes)."""
        self._sketch_registries.pop(doc_id, None)
        self._plane_managers.pop(doc_id, None)

    def execute_code(
        self,
        doc_id: str,
        design: Optional[adsk.fusion.Design],
        code: str,
        operation: str
    ) -> Dict[str, Any]:
        """Execute the provided code snippet for the specified design."""
        if not design:
            logger.error("Cannot execute code for doc %s: design reference missing", doc_id)
            return {
                "success": False,
                "operation": operation,
                "error": "Design context unavailable for this document."
            }

        root_comp = design.rootComponent
        extrudes = root_comp.features.extrudeFeatures

        sketch_registry = self._sketch_registries.setdefault(doc_id, {})

        # Get or create plane manager for this document (persists across execution batches)
        if doc_id not in self._plane_managers:
            self._plane_managers[doc_id] = plane_manager_module.PlaneManager(root_comp)
        _plane_manager = self._plane_managers[doc_id]

        # Stage 1 / D4 — capture the live camera frame so generated code can
        # do viewer-relative direction lookups. ``None`` is acceptable; the
        # face_finder helpers fall back to world-axis logic when the frame is
        # missing or the direction is not viewer-relative.
        try:
            camera_frame = camera_tools.get_camera_frame_mm(self._app)
        except Exception as exc:
            logger.debug("Failed to capture camera_frame for exec: %s", exc)
            camera_frame = None

        # Stage 4 / D2 — dry-run validator. Catches geometric impossibilities
        # and known API-misuse patterns BEFORE Fusion's solver sees the code.
        # On hard error we short-circuit and return a structured payload the
        # recovery loop can read; warnings are surfaced in logs but don't
        # block execution.
        try:
            target_handle = _infer_target_body_handle(code)
            validation = code_validator.validate(
                code,
                root_comp=root_comp,
                target_body_handle=target_handle,
                camera_frame=camera_frame,
            )
        except Exception as exc:
            logger.warning("[code_validator] precheck raised, skipping: %s", exc)
            validation = None

        if validation is not None:
            for w in validation.warnings:
                logger.warning(
                    "[code_validator] WARN/%s line=%s: %s",
                    w.rule, w.line, w.message,
                )
            if not validation.ok:
                summary = validation.summary()
                logger.error(
                    "[code_validator] BLOCKED execution — %d error(s): %s",
                    len(validation.errors), summary,
                )
                return {
                    "success":         False,
                    "operation":       operation,
                    "error":           summary,
                    "error_type":      "ValidatorError",
                    "validator_error": validation.to_dict(),
                }

        exec_globals: Dict[str, Any] = {
            "adsk": adsk,
            "app": self._app,
            "design": design,
            "rootComp": root_comp,
            "extrudes": extrudes,
            "camera_tools": camera_tools,
            "plane_manager": _plane_manager,
            # Runtime helpers for LLM-generated code. ``face_finder`` replaces
            # the brittle face-by-normal loops the agent used to write by hand
            # that raised "Could not find front face (-Y)" etc. ``modify_tools``
            # exposes vetted wrappers for fillet / chamfer / hole / shell /
            # combine / move / circular & rectangular patterns / mirror /
            # extrude-join, so the agent never hand-rolls those error-prone
            # Features-API calls.
            "face_finder":  face_finder,
            "modify_tools": modify_tools,
            # Live camera basis (mm, world-space) — generated code can pass
            # this to face_finder.face_by_direction(..., camera_frame=...) so
            # words like "front" resolve relative to the user's view.
            "camera_frame": camera_frame,
        }
        exec_globals.update(sketch_registry)

        try:
            exec(code, exec_globals)  # noqa: S102 - intentional dynamic execution
            # Capture new sketches created during execution.
            new_sketches = []
            for name, value in exec_globals.items():
                if isinstance(value, adsk.fusion.Sketch):
                    if name not in sketch_registry:
                        new_sketches.append(name)
                    sketch_registry[name] = value

            # ==========================================
            # VISION CRITIC: Auto-Screenshot Capture
            # ==========================================
            img_base64 = None
            try:
                # Force Fusion 360 to frame the object and take a picture
                self._app.activeViewport.fit()
                self._app.activeViewport.refresh()
                img_base64 = camera_tools.capture_screenshot()
            except Exception as e:
                logger.warning("Failed to capture screenshot for Vision Critic: %s", e)
            # ==========================================

            # Check if code set _result (e.g., from camera_tools.capture_screenshot)
            result_data = exec_globals.get("_result")
            
            # DEBUG: Log _result capture
            logger.info("=== DEBUG code_executor ===")
            logger.info("Operation: %s", operation)
            logger.info("_result in exec_globals: %s", "_result" in exec_globals)
            logger.info("result_data value: %s", result_data)
            
            if result_data and isinstance(result_data, dict):
                logger.info("_result is a dict with keys: %s", list(result_data.keys()))
                
                # INJECT SCREENSHOT HERE for the early return path
                if img_base64:
                    result_data["screenshot_base64"] = img_base64
                
                # Return the result data directly if it contains success info
                if "success" in result_data:
                    result_data["operation"] = operation
                    if new_sketches:
                        result_data["created_sketches"] = new_sketches
                    logger.info("Returning result_data directly: %s", result_data)
                    return result_data
                
                # Otherwise merge with standard response
                result = {
                    "success": True,
                    "operation": operation,
                    "message": f"{operation} executed successfully.",
                    **result_data
                }
                if new_sketches:
                    result["created_sketches"] = new_sketches
                logger.info("Returning merged result: %s", result)
                return result
            else:
                logger.info("No _result dict found, using standard response")

            result = {
                "success": True,
                "operation": operation,
                "message": f"{operation} executed successfully.",
            }
            # INJECT SCREENSHOT HERE for the standard fallback path
            if img_base64:
                result["screenshot_base64"] = img_base64
                
            if new_sketches:
                result["created_sketches"] = new_sketches
            return result
            
        except Exception as exc:  # pragma: no cover - Fusion runtime errors are handled at runtime
            # Capture full traceback for debugging context
            tb = traceback.format_exc()
            logger.exception("Failed to execute Fusion code for operation %s", operation)
            return {
                "success": False,
                "operation": operation,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": tb,
            }

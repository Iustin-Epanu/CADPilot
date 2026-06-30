"""
Stage 4 / D2 — Dry-run validator for LLM-generated Fusion 360 Python.

Catches a curated set of geometric / API-misuse mistakes BEFORE the script
hits Fusion's solver — the kind of failures whose stack traces have so far
been the loudest signal in the recovery loop. By raising a structured
``validator_error`` early, we (a) save a Fusion solver round-trip, (b) feed
the recovery LLM a precise message instead of an opaque traceback, and (c)
let the brief's introspective checks (face-by-direction pre-resolve,
profile-inside-face, sane pattern counts) run as Python rather than as
Fusion features that mutate the timeline.

Usage::

    result = code_validator.validate(code, root_comp=root_comp,
                                     target_body_handle=...)
    if not result.ok:
        return {"success": False, "validator_error": result.errors}

The validator is intentionally lenient: when in doubt, it warns rather
than fails, matching the brief's "hard-fail on high-precision rules,
warn-only on heuristics, promote rules to hard-fail as confidence grows".
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ValidatorIssue:
    """One problem found by a validator pass."""
    severity: str          # "error" | "warning"
    rule: str              # short rule id, e.g. "BREP_BODIES_ITEM_0_AMBIGUOUS"
    message: str           # human-readable explanation
    line: Optional[int] = None
    snippet: Optional[str] = None


@dataclass
class ValidatorResult:
    """Aggregate result. ``ok`` is True iff no error-severity issues fired.
    ``errors`` and ``warnings`` are flat lists; the caller assembles the
    payload it returns from ``execute_code``.
    """
    ok: bool = True
    errors: List[ValidatorIssue] = field(default_factory=list)
    warnings: List[ValidatorIssue] = field(default_factory=list)

    def add(self, issue: ValidatorIssue) -> None:
        if issue.severity == "error":
            self.errors.append(issue)
            self.ok = False
        else:
            self.warnings.append(issue)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok":       self.ok,
            "errors":   [_issue_to_dict(i) for i in self.errors],
            "warnings": [_issue_to_dict(i) for i in self.warnings],
        }

    def summary(self) -> str:
        """Human-readable one-paragraph summary suitable for the recovery
        LLM's `error_log`."""
        if self.ok and not self.warnings:
            return "Validator: OK"
        parts: List[str] = []
        for issue in self.errors:
            parts.append(f"[ERROR/{issue.rule}] {issue.message}")
        for issue in self.warnings:
            parts.append(f"[WARN/{issue.rule}] {issue.message}")
        return "Code validator caught issues BEFORE Fusion ran — fix these and retry:\n" + "\n".join(parts)


def _issue_to_dict(i: ValidatorIssue) -> Dict[str, Any]:
    return {
        "severity": i.severity,
        "rule":     i.rule,
        "message":  i.message,
        "line":     i.line,
        "snippet":  i.snippet,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Static (text + AST) checks. These run with NO Fusion calls — safe to
# unit-test in any Python and safe to run before any solver work.
# ─────────────────────────────────────────────────────────────────────────────

# `bRepBodies.item(0)` is the canonical "fall through to first body" silent
# failure — fine on a fresh design, but ambiguous on a design that already
# has multiple bodies. We deliberately match `.item(0)` only and ignore
# `.item(1)` etc.: secondary bodies are legitimately addressed by index in
# Combine, Mirror, and similar multi-body operations.
_BREP_ITEM_ZERO_RE = re.compile(r"\bbRepBodies\.item\(\s*0\s*\)")
_BREP_BY_HANDLE_RE = re.compile(r"\bface_finder\.body_by_handle\(")
_BODY_INDEX_NAME_RE = re.compile(r"\bface_finder\.body_by_index_or_name\(")
_TARGET_HANDLE_REF_RE = re.compile(r"\btarget_body_handle\b")

# setSymmetricExtent must take (ValueInput, bool[, ValueInput]). Passing a
# ThroughAllExtentDefinition is the §2 / 9 signature mistake.
_SETSYM_THROUGHALL_RE = re.compile(
    r"setSymmetricExtent\s*\(\s*[^)]*ThroughAllExtentDefinition", re.DOTALL,
)

# `bbox = body.boundingBox; mn, mx = bbox` — BoundingBox isn't iterable.
_BBOX_UNPACK_RE = re.compile(
    r"^\s*\w+\s*,\s*\w+\s*=\s*[^=#\n]+\.boundingBox\s*$", re.MULTILINE,
)

# adsk imports forbidden — code runs via exec() with adsk already in scope.
_ADSK_IMPORT_RE = re.compile(r"^\s*import\s+adsk(\.|\s|$)", re.MULTILINE)


def _strip_comments_and_strings(src: str) -> str:
    """Return a copy of `src` with line comments and triple-quoted strings
    blanked out, so regex checks don't fire on docstrings or commented-out
    examples in the script."""
    out: List[str] = []
    in_triple_single = False
    in_triple_double = False
    i = 0
    while i < len(src):
        if not in_triple_single and src.startswith('"""', i):
            in_triple_double = not in_triple_double
            out.append('"""')
            i += 3
            continue
        if not in_triple_double and src.startswith("'''", i):
            in_triple_single = not in_triple_single
            out.append("'''")
            i += 3
            continue
        if in_triple_single or in_triple_double:
            out.append(" ")
            i += 1
            continue
        ch = src[i]
        if ch == "#":
            # Skip to end of line.
            while i < len(src) and src[i] != "\n":
                out.append(" ")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _find_face_by_direction_calls(tree: ast.AST) -> List[Tuple[ast.Call, str, Optional[str]]]:
    """Return every (call_node, direction_literal, body_arg_name) for
    static-resolvable ``face_finder.face_by_direction(...)`` invocations.

    ``direction_literal`` is set only when arg[1] is a plain string;
    callers skip pre-resolution otherwise. ``body_arg_name`` is the
    identifier name passed as arg[0] when it's a simple ``Name`` node — used
    to gate introspective pre-resolution to calls that target the resolver's
    body, so multi-body scripts (combine/mirror tool body) don't trigger
    spurious "no -X face" errors against the wrong body.
    """
    hits: List[Tuple[ast.Call, str, Optional[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "face_by_direction"
            and isinstance(func.value, ast.Name)
            and func.value.id == "face_finder"
        ):
            direction = ""
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                if isinstance(node.args[1].value, str):
                    direction = node.args[1].value
            body_arg_name: Optional[str] = None
            if node.args and isinstance(node.args[0], ast.Name):
                body_arg_name = node.args[0].id
            hits.append((node, direction, body_arg_name))
    return hits


def _try_literal(node: ast.AST) -> Any:
    """Resolve a literal constant from an AST node — including
    ``-10.0`` which Python parses as ``UnaryOp(USub, Constant(10.0))``.
    Returns ``None`` for anything that isn't a static literal."""
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


def _find_pattern_calls(tree: ast.AST) -> List[Dict[str, Any]]:
    """Return circular_pattern / rectangular_pattern call descriptors with
    their statically known counts and distances."""
    hits: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "modify_tools"
            and func.attr in ("circular_pattern", "rectangular_pattern")
        ):
            continue
        kind = func.attr
        # Position-arg + kwarg accumulation.
        info: Dict[str, Any] = {"kind": kind, "lineno": node.lineno}
        for kw in node.keywords:
            if kw.arg in (
                "count", "count_one", "count_two",
                "total_angle_deg", "distance_one_mm", "distance_two_mm",
            ):
                v = _try_literal(kw.value)
                if v is not None:
                    info[kw.arg] = v
        # Best-effort positional reading.
        if kind == "circular_pattern":
            # signature: features, axis, count, total_angle_deg=360
            if len(node.args) >= 3:
                v = _try_literal(node.args[2])
                if v is not None:
                    info.setdefault("count", v)
            if len(node.args) >= 4:
                v = _try_literal(node.args[3])
                if v is not None:
                    info.setdefault("total_angle_deg", v)
        else:
            # rectangular_pattern: features, dir1, count_one, distance_one_mm,
            #                      dir2=None, count_two=1, distance_two_mm=0.0
            if len(node.args) >= 3:
                v = _try_literal(node.args[2])
                if v is not None:
                    info.setdefault("count_one", v)
            if len(node.args) >= 4:
                v = _try_literal(node.args[3])
                if v is not None:
                    info.setdefault("distance_one_mm", v)
        hits.append(info)
    return hits


# Set of direction labels `face_finder.face_by_direction` accepts. Mirrors
# `_AXIS_VECTORS` + camera-relative aliases in face_finder.py.
_KNOWN_DIRECTIONS = {
    "+X", "-X", "+Y", "-Y", "+Z", "-Z",
    "TOP", "BOTTOM", "FRONT", "BACK", "RIGHT", "LEFT", "UP", "DOWN",
}


def _direction_known(label: str) -> bool:
    return label.strip().upper().replace(" ", "") in _KNOWN_DIRECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Static-only validator entry point. Use this when no Fusion runtime is
# available (unit tests, CI). The introspective entry point below piggybacks
# on Fusion globals to pre-resolve face_by_direction calls.
# ─────────────────────────────────────────────────────────────────────────────


def validate_static(
    code: str,
    *,
    body_count: Optional[int] = None,
    target_body_handle: Optional[str] = None,
) -> ValidatorResult:
    """Static (text + AST) validation pass.

    Args:
        code:               the LLM-generated script (already de-fenced).
        body_count:         number of bodies in the live design. When >1,
                            ``bRepBodies.item(0)`` triggers an error if no
                            ``target_body_handle`` is in scope.
        target_body_handle: the resolver's pick. When supplied, scripts
                            that ignore it (no occurrence of the literal
                            ``target_body_handle`` AND no
                            ``face_finder.body_by_handle`` call) earn an
                            error — Stage 2's contract.
    """
    result = ValidatorResult()
    cleaned = _strip_comments_and_strings(code)

    # Forbidden adsk import.
    # `import adsk.*` is redundant (the modules are already in exec_globals)
    # but it's a NO-OP, not a runtime error — re-importing rebinds the same
    # module object. Hard-failing the script for a no-op cost the user a full
    # retry round-trip in production. Downgraded to warning so the script
    # still runs and the LLM is nudged via logs to drop the redundant lines.
    for m in _ADSK_IMPORT_RE.finditer(cleaned):
        line = code.count("\n", 0, m.start()) + 1
        result.add(ValidatorIssue(
            severity="warning",
            rule="ADSK_IMPORT_REDUNDANT",
            message=(
                "Redundant `import adsk.*` — `adsk`, `app`, `design`, "
                "`rootComp`, etc. are already in the exec() scope. The "
                "import is harmless (no-op) but unnecessary."
            ),
            line=line,
            snippet=m.group(0).strip(),
        ))

    # setSymmetricExtent(ThroughAllExtentDefinition.create(), ...) — phantom overload.
    for m in _SETSYM_THROUGHALL_RE.finditer(cleaned):
        line = code.count("\n", 0, m.start()) + 1
        result.add(ValidatorIssue(
            severity="error",
            rule="SETSYMMETRIC_PHANTOM_OVERLOAD",
            message=(
                "setSymmetricExtent does NOT accept a ThroughAllExtentDefinition. "
                "Use face_finder.cut_through_body(body, profile, symmetric=True) "
                "which uses the (ValueInput, bool) overload Fusion actually exposes."
            ),
            line=line,
            snippet=m.group(0)[:120].strip(),
        ))

    # bbox unpack.
    for m in _BBOX_UNPACK_RE.finditer(cleaned):
        line = code.count("\n", 0, m.start()) + 1
        result.add(ValidatorIssue(
            severity="error",
            rule="BBOX_UNPACK",
            message=(
                "BoundingBox{2D,3D} is NOT iterable; `mn, mx = body.boundingBox` "
                "raises TypeError. Use bbox.minPoint / bbox.maxPoint, or "
                "face_finder.bbox_tuple(entity)."
            ),
            line=line,
            snippet=m.group(0).strip(),
        ))

    # bRepBodies.item(0) — only an error when there are multiple bodies AND
    # the script does NOT also resolve by handle. If the script calls
    # body_by_handle anywhere, we treat it as having addressed the resolver
    # — secondary `.item(0)` calls in such scripts may be intentional
    # (combine/mirror tool body, iteration setup) and we must not block
    # them. Keep this rule scoped to .item(0) only; .item(1)+ is left alone
    # — addressing a secondary body by index is legitimate in multi-body
    # workflows like combine/mirror.
    item_hits = list(_BREP_ITEM_ZERO_RE.finditer(cleaned))
    if item_hits and (body_count is None or body_count > 1):
        uses_handle = bool(_BREP_BY_HANDLE_RE.search(cleaned))
        if target_body_handle and not uses_handle:
            for m in item_hits:
                line = code.count("\n", 0, m.start()) + 1
                result.add(ValidatorIssue(
                    severity="error",
                    rule="BREP_BODIES_ITEM_0_AMBIGUOUS",
                    message=(
                        f"target_body_handle={target_body_handle!r} was supplied "
                        "by the resolver but the script reaches for "
                        "bRepBodies.item(0) without calling body_by_handle. "
                        "Use `face_finder.body_by_handle(rootComp, target_body_handle)`."
                    ),
                    line=line,
                    snippet=m.group(0),
                ))
        elif not target_body_handle and (body_count or 0) > 1:
            for m in item_hits:
                line = code.count("\n", 0, m.start()) + 1
                result.add(ValidatorIssue(
                    severity="warning",
                    rule="BREP_BODIES_ITEM_0_MULTIBODY",
                    message=(
                        f"Design has {body_count} bodies but the script picks "
                        "bRepBodies.item(0) without resolving by handle. "
                        "Prefer face_finder.body_by_handle(rootComp, ...)."
                    ),
                    line=line,
                    snippet=m.group(0),
                ))

    # face_by_direction on unknown labels.
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.add(ValidatorIssue(
            severity="error",
            rule="SYNTAX_ERROR",
            message=f"Generated code does not parse: {exc.msg}",
            line=exc.lineno,
        ))
        return result

    for call_node, direction, _body_arg in _find_face_by_direction_calls(tree):
        if direction and not _direction_known(direction):
            result.add(ValidatorIssue(
                severity="error",
                rule="FACE_BY_DIRECTION_UNKNOWN_LABEL",
                message=(
                    f"face_by_direction({direction!r}) — direction not in "
                    "{+X,-X,+Y,-Y,+Z,-Z, top,bottom,front,back,left,right,up,down}. "
                    "Use one of the canonical labels or pass a 3-tuple."
                ),
                line=getattr(call_node, "lineno", None),
            ))

    # Pattern sanity.
    for info in _find_pattern_calls(tree):
        line = info["lineno"]
        if info["kind"] == "circular_pattern":
            count = info.get("count")
            angle = info.get("total_angle_deg", 360)
            if isinstance(count, (int, float)) and count < 1:
                result.add(ValidatorIssue(
                    severity="error",
                    rule="PATTERN_COUNT_LT_1",
                    message=f"circular_pattern count={count} — must be >= 1.",
                    line=line,
                ))
            if isinstance(angle, (int, float)) and (angle <= 0 or angle > 720):
                result.add(ValidatorIssue(
                    severity="warning",
                    rule="PATTERN_ANGLE_INSANE",
                    message=(
                        f"circular_pattern total_angle_deg={angle} — "
                        "expected (0, 720]; double-check this isn't a bug."
                    ),
                    line=line,
                ))
        else:
            for k in ("count_one", "count_two"):
                v = info.get(k)
                if isinstance(v, (int, float)) and v < 1:
                    result.add(ValidatorIssue(
                        severity="error",
                        rule="PATTERN_COUNT_LT_1",
                        message=f"rectangular_pattern {k}={v} — must be >= 1.",
                        line=line,
                    ))
            d1 = info.get("distance_one_mm")
            if isinstance(d1, (int, float)) and d1 < 0:
                result.add(ValidatorIssue(
                    severity="warning",
                    rule="PATTERN_NEGATIVE_DISTANCE",
                    message=(
                        f"rectangular_pattern distance_one_mm={d1} — "
                        "negative distance is uncommon; ensure direction vector handles it."
                    ),
                    line=line,
                ))

    # Heuristic: legacy body_by_index_or_name + supplied handle -> warn.
    if target_body_handle and _BODY_INDEX_NAME_RE.search(cleaned):
        result.add(ValidatorIssue(
            severity="warning",
            rule="BODY_BY_INDEX_OR_NAME_DEPRECATED",
            message=(
                "target_body_handle was supplied; prefer "
                "face_finder.body_by_handle(rootComp, target_body_handle) over "
                "the deprecated body_by_index_or_name."
            ),
        ))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Introspective pass — pre-resolves face_by_direction against the live body
# so we catch "no -Y face" etc. without invoking the solver. Only runs the
# checks the brief calls out as "Fusion-introspective" — does NOT mutate
# the timeline.
# ─────────────────────────────────────────────────────────────────────────────


def validate_introspective(
    code: str,
    *,
    root_comp: Any,
    target_body_handle: Optional[str] = None,
    camera_frame: Optional[Dict[str, Any]] = None,
) -> ValidatorResult:
    """Introspective dry-run — calls ``face_finder.face_by_direction`` on the
    live target body for every static-resolvable invocation, and reports any
    LookupError as an error issue. Does NOT execute the script.

    Heavy / unsafe checks (sketch creation, profile-inside-face) are kept
    out of this pass for now — they require constructing transient sketches
    on the Fusion main thread, and the brief explicitly recommends
    "warn-only on the heuristic ones". They will move into validator
    passes promoted to hard-fail in a later iteration.

    Args:
        code:               script source.
        root_comp:          ``rootComp`` from the executor (live Fusion
                            component).
        target_body_handle: if supplied, face_by_direction calls without
                            an explicit body argument can still be checked
                            against this body.
        camera_frame:       optional viewport basis, so camera-relative
                            labels resolve the way generated code will.
    """
    result = ValidatorResult()

    # Static pass first — it's cheap and may already have caught everything.
    body_count = None
    try:
        body_count = root_comp.bRepBodies.count
    except Exception:
        pass
    static = validate_static(
        code,
        body_count=body_count,
        target_body_handle=target_body_handle,
    )
    for issue in static.errors:
        result.add(issue)
    for issue in static.warnings:
        result.add(issue)

    # If the static pass already saw a SyntaxError, AST walks below would
    # raise — bail out early.
    if any(i.rule == "SYNTAX_ERROR" for i in result.errors):
        return result

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result

    if target_body_handle is None:
        return result

    # Resolve the target body once.
    try:
        from . import face_finder  # local import — Fusion-only at runtime.
        target_body = face_finder.body_by_handle(root_comp, target_body_handle)
    except Exception as exc:
        result.add(ValidatorIssue(
            severity="warning",
            rule="TARGET_BODY_UNRESOLVED",
            message=(
                f"Could not pre-resolve target_body_handle={target_body_handle!r} "
                f"({exc}); skipping introspective face_by_direction checks."
            ),
        ))
        return result

    # Identify which Python variable names refer to the resolver's target
    # body, by walking assignments of the form
    #     <name> = face_finder.body_by_handle(<...>, <handle_arg>)
    # If the second arg is a literal that matches `target_body_handle`, OR
    # the second arg is a Name (`target_body_handle`), we treat the LHS
    # name as bound to the target body. Pre-resolution is skipped for any
    # face_by_direction(<other_body>, ...) call so multi-body scripts that
    # legitimately address a secondary body (combine/mirror tool body) are
    # never falsely hard-failed.
    target_var_names: set = set()
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "body_by_handle"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "face_finder"
        ):
            continue
        if len(call.args) < 2:
            continue
        handle_arg = call.args[1]
        bound = False
        if isinstance(handle_arg, ast.Constant) and handle_arg.value == target_body_handle:
            bound = True
        elif isinstance(handle_arg, ast.Name) and handle_arg.id == "target_body_handle":
            bound = True
        if not bound:
            continue
        for tgt in stmt.targets:
            if isinstance(tgt, ast.Name):
                target_var_names.add(tgt.id)

    for call_node, direction, body_arg_name in _find_face_by_direction_calls(tree):
        if not direction:
            continue  # dynamic — can't pre-resolve.
        # Only pre-resolve when the call's first arg is a variable bound to
        # the target body. Otherwise this call is operating on a different
        # body whose -X/-Y face availability we can't infer here.
        if body_arg_name is None or body_arg_name not in target_var_names:
            continue
        try:
            face_finder.face_by_direction(
                target_body, direction, camera_frame=camera_frame,
            )
        except LookupError as exc:
            result.add(ValidatorIssue(
                severity="error",
                rule="FACE_BY_DIRECTION_NO_MATCH",
                message=(
                    f"face_by_direction({direction!r}) on target body has no "
                    f"match: {exc}. Pick a different face / direction, or "
                    "place the sketch on a construction plane."
                ),
                line=getattr(call_node, "lineno", None),
            ))
        except Exception as exc:
            result.add(ValidatorIssue(
                severity="warning",
                rule="FACE_BY_DIRECTION_PRECHECK_FAILED",
                message=(
                    f"face_by_direction({direction!r}) precheck raised "
                    f"{type(exc).__name__}: {exc}; skipping."
                ),
                line=getattr(call_node, "lineno", None),
            ))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public façade — picks the right pass based on what's available.
# ─────────────────────────────────────────────────────────────────────────────


def validate(
    code: str,
    *,
    root_comp: Any = None,
    target_body_handle: Optional[str] = None,
    camera_frame: Optional[Dict[str, Any]] = None,
) -> ValidatorResult:
    """Run the most thorough validator pass available.

    When ``root_comp`` is supplied, includes the introspective face-by-
    direction pre-resolution. Otherwise runs the static pass alone. Both
    paths return a ``ValidatorResult`` with ``ok``, ``errors``, ``warnings``.
    """
    if root_comp is not None:
        return validate_introspective(
            code,
            root_comp=root_comp,
            target_body_handle=target_body_handle,
            camera_frame=camera_frame,
        )
    body_count = None
    return validate_static(
        code,
        body_count=body_count,
        target_body_handle=target_body_handle,
    )


__all__ = [
    "ValidatorIssue",
    "ValidatorResult",
    "validate",
    "validate_static",
    "validate_introspective",
]

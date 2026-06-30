"""
CADPilot Smart Core — LangGraph Agent

Implements a multi-node AI agent that:
1. Analyzes user requests and B-Rep context
2. Plans a sequence of Fusion 360 operations
3. Generates Python code targeting the Fusion 360 API
4. Validates the code before execution
5. Recovers from execution errors with self-correction

Architecture:
    planner → coder → critic → vision_critic → END
                         ↓ (execution error)    ↓ (visual mismatch)
                      error_recovery ←──────────┘
                           └→ coder

All LLM calls support dynamic API keys passed from the Fusion client via
the authenticate message, enabling BYOK (Bring Your Own Key).
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# State Definition
# ═══════════════════════════════════════════════════════════════════════════════



class AgentState(TypedDict, total=False):
    """
    Mutable state dict that flows through every node in the graph.
    """
    screenshot_base64: Optional[str]
    user_prompt: str
    brep_context: Dict[str, Any]
    selection_context: Optional[Any]
    spatial_context: Optional[Any]
    timeline_context: Optional[List[Dict[str, Any]]]
    # Tier 4 — Agentic Probing
    probe_callback: Optional[Callable]          # async (tool_name, params) → dict
    probe_results: Optional[List[Dict[str, Any]]]
    probe_summary: Optional[str]
    # Stage 1 / A4 — Eyes-on planning. Captured by the add-in on every
    # execute_request / planning_request so the planner can resolve "front",
    # "the visible hole", etc. against what the user is actually looking at.
    viewport_screenshot: Optional[str]          # base64 PNG of the live viewport
    camera_frame: Optional[Dict[str, Any]]      # {origin, target, up_vector, eye_vector} (mm)
    # Stage 2 / B1 — Target resolver output. Bound to a structured handle
    # BEFORE planning so the coder physically cannot fall through to "create
    # a new body" — the target is an input, not a guess. See §3.B1 schema.
    resolved_target: Optional[Dict[str, Any]]
    target_source:   Optional[str]              # "selection" | "prompt" | "inference"
    # Stage 3 / C1 — Clarification round-trip. The agent suspends inside
    # `clarify_node`, the user picks an option in the palette widget, and
    # the callback resolves with their pick. Signature:
    #   result = await clarify_callback(question, options, rationale=...)
    clarify_callback: Optional[Callable]
    # Stage 4 / D3 — Persistent failure memory. Every error_recovery round
    # appends one entry; the recovery system prompt reads back the log so
    # the LLM stops repeating the same fix. Schema per entry:
    #   {approach: str, error_type: str, error_message: str, fix_tried: str}
    attempts_log: Optional[List[Dict[str, Any]]]
    plan: str
    generated_code: str
    error_log: str
    retry_count: int
    model_name: str
    api_keys: Dict[str, str]
    stream_callback: Optional[Callable]


# Default retry limit before giving up.
MAX_RETRIES = 2

# ═══════════════════════════════════════════════════════════════════════════════
# LLM Factory — dynamic model & API key selection
# ═══════════════════════════════════════════════════════════════════════════════

# Model name normalisation: accept user-facing names and map to Anthropic IDs.
_MODEL_ALIASES: Dict[str, str] = {
    "claude-sonnet-4-5": "claude-opus-4-6",
    "claude-sonnet-4.5": "claude-opus-4-6",
    "claude-3.5-sonnet": "claude-opus-4-6",
    "claude-3-5-sonnet": "claude-opus-4-6",
    "claude-opus-4": "claude-opus-4-6",
    "claude-haiku-4": "claude-opus-4-6",
    "claude-haiku-4-5": "claude-opus-4-6",
    # UI uses dotted 4.6 / 4.7 naming — normalise to Anthropic's hyphenated IDs.
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-sonnet-4.7": "claude-sonnet-4-6",
    "claude-opus-4.7": "claude-opus-4-7",
}


def _resolve_model(model_name: str) -> str:
    """Normalise a user-provided model name to a valid Anthropic model ID."""
    cleaned = model_name.strip().lower()
    resolved = _MODEL_ALIASES.get(cleaned, cleaned)
    
    # STRICT ANTHROPIC ENFORCEMENT: 
    # If the UI asks for GPT, Gemini, or an invalid string, force it to Claude 4.6 Opus.
    if not resolved.startswith("claude"):
        logger.warning(f"[model_resolver] Trapped invalid/non-Anthropic model '{model_name}'. Forcing Claude 4.6 Opus.")
        resolved = "claude-opus-4-6"
        
    logger.info("[model_resolver] %r → %r", model_name, resolved)
    return resolved


def _make_llm(
    model_name: str = "claude-opus-4-6",
    api_keys: Optional[Dict[str, str]] = None,
    temperature: float = 0,
    max_tokens: int = 4096,
) -> ChatAnthropic:
    """
    Build a ChatAnthropic instance, preferring client-supplied API keys
    (BYOK) and falling back to the ANTHROPIC_API_KEY env var.
    """
    api_key = None
    if api_keys:
        api_key = api_keys.get("anthropic_api_key") or api_keys.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    resolved = _resolve_model(model_name)

    return ChatAnthropic(
        model=resolved,
        temperature=temperature,
        max_tokens=max_tokens,
        anthropic_api_key=api_key,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Cost-reduction — Anthropic prompt caching helpers
# ───────────────────────────────────────────────────────────────────────────────
# Strategy (https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
#   • Cache reads cost 0.10× the base input rate; cache writes cost 1.25× base
#     for the 5-min TTL or 2× for the 1-hour TTL. Within a single user turn the
#     same massive system prompts and the same brep-context block are sent to
#     planner, coder, critic and (on failure) error_recovery + vision_critic.
#     Each repeated send becomes a 90% discount instead of a full reprice.
#   • Up to 4 explicit breakpoints per request. Hierarchy: tools → system →
#     messages. We use one breakpoint on the system prompt (covers tools too)
#     and optionally one on a stable prefix at the start of the human message
#     (the brep / timeline / probe block).
#   • Below-threshold prompts are silently un-cached by Anthropic (Opus 4.7 =
#     4096-token min, Sonnet 4.6 = 2048, Haiku 4.5 = 4096), so it's safe to
#     mark every system prompt — short ones just no-op.
#   • Batch API (50% discount) is intentionally NOT used: the interactive
#     WebSocket loop demands sub-second turnaround, and Batch can take up to
#     24 h. Caching gives most of the savings without that latency cost.
# ═══════════════════════════════════════════════════════════════════════════════

# Default TTL for system-prompt and stable-context cache breakpoints.
# 1h covers a typical multi-turn design session without re-paying the write
# cost on every turn. The write surcharge (2× base) is amortised after the
# second hit; users typically issue 5+ requests per session.
_CACHE_TTL = "1h"


def _cacheable_system(text: str, *, ttl: str = _CACHE_TTL) -> SystemMessage:
    """Build a ``SystemMessage`` whose single text block is marked cacheable.

    Anthropic ignores ``cache_control`` on blocks below the per-model token
    threshold, so this is safe for short prompts too — the field is just
    dropped server-side. ``ChatAnthropic`` forwards block-level
    ``cache_control`` when ``content`` is a list of dicts.
    """
    return SystemMessage(
        content=[
            {
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral", "ttl": ttl},
            }
        ]
    )


def _cacheable_human(
    cached_prefix: str,
    dynamic_tail: Optional[str] = None,
    *,
    image_b64: Optional[str] = None,
    ttl: str = _CACHE_TTL,
) -> HumanMessage:
    """Build a ``HumanMessage`` whose first content block is cached and the
    rest is per-turn dynamic content.

    Layout of the message body (in order):
      1. Cached text block (marked with ``cache_control``) — typically the
         brep / timeline / probe / camera context that's identical across
         every node within a user turn.
      2. Dynamic text block (un-cached) — the per-node tail (the user
         request, the plan, the failed code, the error log).
      3. Optional image (un-cached — never cache screenshots: cache_control
         on an image block invalidates on the next snapshot anyway).
    """
    blocks: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": cached_prefix,
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }
    ]
    if dynamic_tail:
        blocks.append({"type": "text", "text": dynamic_tail})
    if image_b64:
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64,
            },
        })
    return HumanMessage(content=blocks)


def _log_cache_usage(node_name: str, response: Any) -> None:
    """Log the cache hit/write counts from an LLM response so we can monitor
    cost savings without parsing API invoices.

    The Anthropic API reports three input-token buckets in
    ``response_metadata['usage']``:
        - cache_read_input_tokens     (paid at 0.10×)
        - cache_creation_input_tokens (paid at 1.25× or 2×)
        - input_tokens                (paid at 1×)
    """
    try:
        usage = (getattr(response, "response_metadata", None) or {}).get("usage") or {}
        if not usage:
            usage = (getattr(response, "usage_metadata", None) or {}) or {}
        read = usage.get("cache_read_input_tokens") or usage.get("input_token_details", {}).get("cache_read") or 0
        write = usage.get("cache_creation_input_tokens") or usage.get("input_token_details", {}).get("cache_creation") or 0
        plain = usage.get("input_tokens") or 0
        if read or write:
            logger.info(
                "[cache] %s: read=%s, write=%s, plain=%s (hit_rate=%.0f%%)",
                node_name, read, write, plain,
                100.0 * read / max(1, read + write + plain),
            )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Context Formatting — turn raw B-Rep / selection dicts into LLM-readable text
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Context Formatting — turn raw B-Rep / selection dicts into LLM-readable text
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_pt(pt: Any) -> tuple:
    """Safely extract x,y,z from a dict or list point representation."""
    if isinstance(pt, dict):
        try:
            return (float(pt.get('x', 0)), float(pt.get('y', 0)), float(pt.get('z', 0)))
        except (ValueError, TypeError):
            pass
    elif isinstance(pt, list) and len(pt) >= 3:
        try:
            return (float(pt[0]), float(pt[1]), float(pt[2]))
        except (ValueError, TypeError):
            pass
    return (0.0, 0.0, 0.0)

def _fmt_brep(brep: Optional[Dict[str, Any]]) -> str:
    """Format B-Rep context into a per-body summary plus a retrieval pointer.

    The dump used to truncate at 10 bodies / 8 faces / 8 edges, which made
    every entity past those indices invisible to the planner. We now emit one
    summary line per body for ALL bodies, then tell the agent to use the
    ``lookup_entity`` probe to read any individual face/edge/vertex on demand.
    Tokens / indices stay in the dump so existing helpers keep working.
    """
    if not brep or not isinstance(brep, dict):
        return "No existing geometry (empty design)."

    parts: List[str] = []
    bodies = brep.get("bodies", [])
    faces = brep.get("faces", [])
    edges = brep.get("edges", [])
    vertices = brep.get("vertices", [])

    # Build per-body face/edge counts from the snapshot if the body dict is
    # missing them (older clients).
    face_counts_by_body: Dict[Any, int] = {}
    edge_counts_by_body: Dict[Any, int] = {}
    if isinstance(faces, list):
        for f in faces:
            if isinstance(f, dict):
                key = f.get("body_handle") or f.get("body_index")
                if key is not None:
                    face_counts_by_body[key] = face_counts_by_body.get(key, 0) + 1
    if isinstance(edges, list):
        for e in edges:
            if isinstance(e, dict):
                key = e.get("body_handle") or e.get("body_index")
                if key is not None:
                    edge_counts_by_body[key] = edge_counts_by_body.get(key, 0) + 1

    if isinstance(bodies, list) and bodies:
        parts.append(
            f"Existing bodies ({len(bodies)}) — these already exist in the design; "
            "modify them rather than creating new geometry unless the user asked "
            "for a new separate body. Resolve a body in generated code via "
            "`face_finder.body_by_handle(rootComp, \"<body_handle>\")`:"
        )
        for i, b in enumerate(bodies):
            if not isinstance(b, dict):
                continue
            name = b.get("name", f"Body{i}")
            body_handle = b.get("body_handle") or b.get("handle") or "?"
            vol = b.get("volume", "?")
            solid = "solid" if b.get("is_solid", True) else "surface"
            face_count = b.get("face_count")
            if face_count is None:
                face_count = face_counts_by_body.get(body_handle) or face_counts_by_body.get(b.get("body_index"))
            edge_count = b.get("edge_count")
            if edge_count is None:
                edge_count = edge_counts_by_body.get(body_handle) or edge_counts_by_body.get(b.get("body_index"))

            bbox = b.get("bbox") or b.get("bounding_box") or b.get("boundingBox")
            dims_str = ""
            if isinstance(bbox, dict):
                mn = _safe_pt(bbox.get("min"))
                mx = _safe_pt(bbox.get("max"))
                dims_str = f", bbox {mx[0]-mn[0]:.1f}×{mx[1]-mn[1]:.1f}×{mx[2]-mn[2]:.1f} mm"
            elif isinstance(bbox, list) and len(bbox) == 2:
                mn, mx = _safe_pt(bbox[0]), _safe_pt(bbox[1])
                dims_str = f", bbox {mx[0]-mn[0]:.1f}×{mx[1]-mn[1]:.1f}×{mx[2]-mn[2]:.1f} mm"

            extras = []
            if face_count is not None:
                extras.append(f"{face_count} faces")
            if edge_count is not None:
                extras.append(f"{edge_count} edges")
            extra_str = f", {', '.join(extras)}" if extras else ""

            parts.append(
                f"  - body_handle={body_handle} name='{name}': {vol} mm³ ({solid}){dims_str}{extra_str}"
            )

        parts.append(
            "  (Use the `lookup_entity` probe with "
            "{body_handle, kind: 'face'|'edge'|'vertex', index} to read the "
            "full descriptor of any face/edge/vertex on demand — the dump "
            "intentionally omits per-face/edge details to stay compact.)"
        )

    if isinstance(faces, list) and faces:
        parts.append(
            f"Total faces across all bodies: {len(faces)}. "
            "Use `lookup_entity` with kind='face' to fetch any specific face."
        )
    if isinstance(edges, list) and edges:
        parts.append(
            f"Total edges across all bodies: {len(edges)}. "
            "Use `lookup_entity` with kind='edge' to fetch any specific edge."
        )
    if isinstance(vertices, list) and vertices:
        parts.append(
            f"Total vertices: {len(vertices)}. "
            "Use `lookup_entity` with kind='vertex' to fetch any specific vertex."
        )

    return "\n".join(parts) if parts else "No existing geometry (empty design)."

def _fmt_selection(sel: Optional[Dict[str, Any]]) -> str:
    """Format selection context for the LLM."""
    if not sel:
        return ""
    
    entities = []
    count = 0
    if isinstance(sel, dict):
        entities = sel.get("entities", [])
        count = sel.get("count", len(entities) if isinstance(entities, list) else 0)
    elif isinstance(sel, list):
        entities = sel
        count = len(sel)
        
    if not isinstance(entities, list) or not entities:
        return ""

    lines = [
        f"User has selected {count} entity/entities — these are the explicit "
        "target(s) of the request. Modify the parent body of each selection "
        "rather than creating new geometry."
    ]
    for i, ent in enumerate(entities[:6], 1):
        if not isinstance(ent, dict): continue
        etype = ent.get("type", "unknown")
        body_name = ent.get("body_name")
        body_handle = ent.get("body_handle")
        face_idx = ent.get("face_index")
        edge_idx = ent.get("edge_index")
        parent = ""
        if body_handle or body_name:
            parent = f", parent body: body_handle={body_handle} name='{body_name}'"
        if etype == "face":
            gt = ent.get("geometry_type", "?")
            area = ent.get("area", "?")
            nx, ny, nz = _safe_pt(ent.get("normal"))
            fi = f", face_index={face_idx}" if face_idx is not None else ""
            lines.append(
                f"  {i}. Face ({gt}): {area} cm², normal=({nx:.2f}, {ny:.2f}, {nz:.2f}){fi}{parent}"
            )
        elif etype == "edge":
            gt = ent.get("geometry_type", "?")
            length = ent.get("length", "?")
            ei = f", edge_index={edge_idx}" if edge_idx is not None else ""
            lines.append(f"  {i}. Edge ({gt}): {length} cm{ei}{parent}")
        elif etype == "body":
            name = ent.get("name", "?")
            vol = ent.get("volume", "?")
            bh = f" (body_handle={body_handle})" if body_handle else ""
            lines.append(f"  {i}. Body '{name}'{bh}: {vol} cm³")
        else:
            lines.append(f"  {i}. {etype}: {ent}")
    return "\n".join(lines)

def _fmt_spatial(spatial: Optional[Dict[str, Any]]) -> str:
    """Format spatial context (parallel faces, body relationships) for the LLM."""
    if not spatial or not isinstance(spatial, dict):
        return ""
    parts = []
    overview = spatial.get("overview")
    if isinstance(overview, dict):
        bc = overview.get("body_count", 0)
        parts.append(f"Workspace: {bc} bodies")
        wb = overview.get("workspace_bounds")
        if wb:
            parts.append(f"  Bounds: {wb}")

    pf = spatial.get("parallel_faces", [])
    if isinstance(pf, list) and pf:
        parts.append(f"Parallel face groups: {len(pf)}")
        for i, group in enumerate(pf[:3]):
            parts.append(f"  Group {i}: {group}")

    br = spatial.get("body_relationships", [])
    if isinstance(br, list) and br:
        parts.append(f"Body relationships: {len(br)}")
        for rel in br[:4]:
            parts.append(f"  - {rel}")

    return "\n".join(parts) if parts else ""


def _fmt_probe_results(probe_results: list, probe_summary: str) -> str:
    """Format probe measurement results for injection into LLM context.

    Stage 2 / B4: when a probe returned full per-entity descriptors (e.g.
    `connected_faces` now carries `body_handle` / `face_index` /
    `entity_token` / `surface_type` for every adjacent face; `face_by_description`
    returns ranked candidates with handles), render those handles inline so
    the agent can refer back to them in code generation without a follow-up
    probe call. Other tool outputs keep the compact one-line JSON form.
    """
    if not probe_results and not probe_summary:
        return ""
    parts = ["Spatial measurements (from probing tools):"]
    for r in probe_results:
        tool = r.get("tool", "?")
        result = r.get("result", {})

        # Probe results often arrive as a JSON string (the LangChain wrapper
        # serialises before returning). Try to re-hydrate so we can pick out
        # the structured fields B4 promised.
        parsed: Any = result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except Exception:
                parts.append(f"  [{tool}] {result[:300]}")
                continue

        if not isinstance(parsed, dict):
            parts.append(f"  [{tool}] {str(parsed)[:300]}")
            continue
        if "error" in parsed:
            continue

        # connected_faces (B4) — full face descriptors per neighbour.
        if tool == "connected_faces" and isinstance(parsed.get("connected_faces"), list):
            entries = parsed["connected_faces"]
            parts.append(f"  [connected_faces] {len(entries)} adjacent face(s):")
            for c in entries[:6]:
                if not isinstance(c, dict):
                    continue
                bh = c.get("body_handle") or "?"
                fi = c.get("face_index")
                st = c.get("surface_type") or "?"
                area = c.get("area") or c.get("area_mm2")
                parts.append(
                    f"    - body_handle={bh}, face_index={fi}, "
                    f"surface_type={st}, area={area}"
                )
            continue

        # face_by_description (B2) — ranked candidates.
        if tool == "face_by_description" and isinstance(parsed.get("candidates"), list):
            cands = parsed["candidates"]
            parts.append(f"  [face_by_description] {len(cands)} candidate(s):")
            for c in cands[:6]:
                if not isinstance(c, dict):
                    continue
                parts.append(
                    f"    - body_handle={c.get('body_handle')}, "
                    f"face_index={c.get('face_index')}, "
                    f"surface_type={c.get('surface_type')}, "
                    f"score={c.get('score')}, rationale={c.get('rationale')!r}"
                )
            continue

        # find_feature_by_phrase (B3) — ranked timeline matches.
        if tool == "find_feature_by_phrase" and isinstance(parsed.get("matches"), list):
            matches = parsed["matches"]
            parts.append(f"  [find_feature_by_phrase] {len(matches)} match(es):")
            for m in matches[:6]:
                if not isinstance(m, dict):
                    continue
                parts.append(
                    f"    - index={m.get('index')}, name={m.get('name')!r}, "
                    f"type={m.get('type')}, score={m.get('score')}, "
                    f"rationale={m.get('rationale')!r}"
                )
            continue

        # face_properties (B4) — single full descriptor.
        if tool == "face_properties":
            parts.append(
                f"  [face_properties] body_handle={parsed.get('body_handle')}, "
                f"face_index={parsed.get('face_index')}, "
                f"surface_type={parsed.get('surface_type')}, "
                f"area={parsed.get('area_mm2') or parsed.get('area')}, "
                f"centroid={parsed.get('centroid')}, normal={parsed.get('normal')}"
            )
            continue

        # Default: compact one-line JSON.
        parts.append(f"  [{tool}] {json.dumps(parsed, separators=(',', ':'))[:300]}")

    if probe_summary:
        parts.append(f"\nProbe summary:\n{probe_summary}")
    return "\n".join(parts)


def _fmt_timeline(timeline: list) -> str:
    """Format timeline history chronologically for the LLM.

    Always renders a header so the planner/coder can see whether the design
    has any prior features. When the timeline is empty we emit an explicit
    "(no features yet)" line — this is what tells the agent that references
    like "the last hole" or "the cut I just added" are not addressable yet.
    """
    if not timeline:
        return "Timeline history: (no features yet)"
    parts = ["Timeline history:"]
    for item in timeline:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", "?")
        name = item.get("name", "unnamed")
        ftype = item.get("type", "?")
        details = []
        dist = item.get("distance_mm")
        if dist is not None:
            details.append(f"dist={dist}mm")
        radius = item.get("radius_mm")
        if radius is not None:
            details.append(f"r={radius}mm")
        qty = item.get("quantity_one") or item.get("quantity")
        if qty is not None:
            details.append(f"count={qty}")
        qty2 = item.get("quantity_two")
        if qty2 is not None:
            details.append(f"count2={qty2}")
        angle = item.get("total_angle_deg")
        if angle is not None:
            details.append(f"angle={angle}°")
        detail_str = f" ({', '.join(details)})" if details else ""
        parts.append(f"  {idx}. {name} [{ftype}]{detail_str}")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 4 — Probe Tool Schemas + Factory
# ═══════════════════════════════════════════════════════════════════════════════

class _BBoxInput(BaseModel):
    tokens: List[str] = Field(default=[], description="Entity tokens to measure. Empty = all bodies.")

class _MinDistInput(BaseModel):
    token_a: str = Field(description="First entity token")
    token_b: str = Field(description="Second entity token")

class _ConnFacesInput(BaseModel):
    face_token: str = Field(description="Face entity token")

class _FacePropInput(BaseModel):
    face_token: str = Field(description="Face entity token")

class _EdgeChainInput(BaseModel):
    edge_token: str = Field(description="Starting edge entity token")

class _BodyVolInput(BaseModel):
    token: str = Field(default="", description="Body entity token (uses index 0 if empty)")
    index: int = Field(default=0, description="Body index if no token given")

class _SvgSecInput(BaseModel):
    plane_axis: str = Field(default="XY", description="Cutting plane: 'XY', 'XZ', or 'YZ'")
    offset_mm: float = Field(default=0.0, description="Offset from plane in mm")

class _BlueprintInput(BaseModel):
    views: List[str] = Field(default=["top", "front", "right"], description="Views to capture")
    width: int = Field(default=1920, description="Image width in pixels")
    height: int = Field(default=1080, description="Image height in pixels")


class _FaceSketchBoundsInput(BaseModel):
    face_token: str = Field(description="Entity token of the target BRepFace")
    hole_radius_mm: float = Field(
        default=0.0,
        description="Planned hole radius in mm — used to compute the safe interior (face bbox minus radius+margin).",
    )
    margin_mm: float = Field(
        default=2.0,
        description="Extra safety margin in mm inside the face edges.",
    )


class _LookupEntityInput(BaseModel):
    body_handle: str = Field(
        description="Stable handle of the parent body, e.g. 'b_a39f...' (from the brep summary)."
    )
    kind: str = Field(
        default="face",
        description="One of 'face', 'edge', or 'vertex'.",
    )
    index: int = Field(
        default=0,
        description="0-based index inside that body's faces/edges/vertices.",
    )


def _make_probe_tools(probe_callback: Callable) -> List[StructuredTool]:
    """Build LangChain StructuredTools backed by the async probe_callback."""

    async def bounding_box(tokens: List[str] = []) -> str:
        return json.dumps(await probe_callback("bounding_box", {"tokens": tokens}))

    async def min_distance(token_a: str, token_b: str) -> str:
        return json.dumps(await probe_callback("min_distance", {"token_a": token_a, "token_b": token_b}))

    async def connected_faces(face_token: str) -> str:
        return json.dumps(await probe_callback("connected_faces", {"face_token": face_token}))

    async def face_properties(face_token: str) -> str:
        return json.dumps(await probe_callback("face_properties", {"face_token": face_token}))

    async def edge_chain(edge_token: str) -> str:
        return json.dumps(await probe_callback("edge_chain", {"edge_token": edge_token}))

    async def body_volume(token: str = "", index: int = 0) -> str:
        return json.dumps(await probe_callback("body_volume", {"token": token, "index": index}))

    async def svg_section(plane_axis: str = "XY", offset_mm: float = 0.0) -> str:
        r = await probe_callback("svg_section", {"plane_axis": plane_axis, "offset_mm": offset_mm})
        return r.get("svg", "") if isinstance(r, dict) else json.dumps(r)

    async def blueprint_views(views: List[str] = ["top", "front", "right"],
                              width: int = 1920, height: int = 1080) -> str:
        r = await probe_callback("blueprint_views", {"views": views, "width": width, "height": height})
        # Strip large base64 from the summary returned to LLM; just confirm success
        if isinstance(r, dict):
            summary = {k: ("(image captured)" if isinstance(v, dict) and "image_base64" in v else v)
                       for k, v in r.items() if k != "success"}
            return json.dumps({"success": r.get("success"), "captured_views": list(summary.keys())})
        return json.dumps(r)

    async def face_sketch_bounds(
        face_token: str,
        hole_radius_mm: float = 0.0,
        margin_mm: float = 2.0,
    ) -> str:
        return json.dumps(await probe_callback("face_sketch_bounds", {
            "face_token":     face_token,
            "hole_radius_mm": hole_radius_mm,
            "margin_mm":      margin_mm,
        }))

    async def lookup_entity(
        body_handle: str,
        kind: str = "face",
        index: int = 0,
    ) -> str:
        return json.dumps(await probe_callback("lookup_entity", {
            "body_handle": body_handle,
            "kind":        kind,
            "index":       index,
        }))

    async def face_by_description(
        body_handle: str,
        phrase: str,
        top_n: int = 4,
    ) -> str:
        return json.dumps(await probe_callback("face_by_description", {
            "body_handle": body_handle,
            "phrase":      phrase,
            "top_n":       top_n,
        }))

    async def find_feature_by_phrase(
        phrase: str,
        top_n: int = 4,
    ) -> str:
        return json.dumps(await probe_callback("find_feature_by_phrase", {
            "phrase": phrase,
            "top_n":  top_n,
        }))

    return [
        StructuredTool.from_function(
            coroutine=bounding_box, name="bounding_box",
            description="Get bounding box (mm) — size, center, min/max — of entities or all bodies.",
            args_schema=_BBoxInput,
        ),
        StructuredTool.from_function(
            coroutine=min_distance, name="min_distance",
            description="Measure minimum clearance in mm between two B-Rep entities by entity token.",
            args_schema=_MinDistInput,
        ),
        StructuredTool.from_function(
            coroutine=connected_faces, name="connected_faces",
            description="Return faces adjacent to a face (by token), with surface types and areas.",
            args_schema=_ConnFacesInput,
        ),
        StructuredTool.from_function(
            coroutine=face_properties, name="face_properties",
            description="Get exact geometry of a face: type, area, normal, centroid, radius, axis, origin.",
            args_schema=_FacePropInput,
        ),
        StructuredTool.from_function(
            coroutine=edge_chain, name="edge_chain",
            description="Walk tangent edge chain from a starting edge, returning lengths and midpoints.",
            args_schema=_EdgeChainInput,
        ),
        StructuredTool.from_function(
            coroutine=body_volume, name="body_volume",
            description="Get volume (mm³), mass (g), and center of mass of a body.",
            args_schema=_BodyVolInput,
        ),
        StructuredTool.from_function(
            coroutine=svg_section, name="svg_section",
            description=(
                "Slice the 3D model at a plane (XY/XZ/YZ) and return an SVG wireframe "
                "of the cross-section as text. Use to read complex 2D profiles as pure math."
            ),
            args_schema=_SvgSecInput,
        ),
        StructuredTool.from_function(
            coroutine=blueprint_views, name="blueprint_views",
            description=(
                "Capture orthographic Top/Front/Right engineering blueprint views "
                "and confirm which were captured."
            ),
            args_schema=_BlueprintInput,
        ),
        StructuredTool.from_function(
            coroutine=face_sketch_bounds, name="face_sketch_bounds",
            description=(
                "Return a face's bounding rectangle in SKETCH-SPACE (mm) — width, "
                "height, min/max/center — plus a 'safe_interior' rectangle pre-inset "
                "by (hole_radius_mm + margin_mm). Use this BEFORE placing circles on "
                "a face so profiles never fall outside the face, which would cause "
                "EXTRUDE_BOOLEAN_FAIL. You MUST pass face_token from the current "
                "geometry context."
            ),
            args_schema=_FaceSketchBoundsInput,
        ),
        StructuredTool.from_function(
            coroutine=lookup_entity, name="lookup_entity",
            description=(
                "Fetch the full per-entity descriptor (surface type, area, normal, "
                "axis, radius, token, etc.) of a single face/edge/vertex inside a "
                "body. The brep summary intentionally omits per-face/edge details "
                "to stay compact; call this whenever you need to read one. "
                "Pass the body's `body_handle` (from the brep summary), the kind "
                "('face'|'edge'|'vertex'), and the 0-based index inside that body."
            ),
            args_schema=_LookupEntityInput,
        ),
        StructuredTool.from_function(
            coroutine=face_by_description, name="face_by_description",
            description=(
                "Rank the faces of a body against a natural-language phrase such "
                "as 'front face', 'the round face on top', or 'biggest flat face'. "
                "Returns a list of scored candidates (each with body_handle / "
                "face_index / entity_token / score / rationale). Returns an empty "
                "list when nothing matches — that means ASK FOR CLARIFICATION, "
                "not retry. When a camera_frame is in the agent's context, "
                "viewer-relative labels resolve against the user's view."
            ),
            args_schema=_FaceByDescriptionInput,
        ),
        StructuredTool.from_function(
            coroutine=find_feature_by_phrase, name="find_feature_by_phrase",
            description=(
                "Search the timeline DNA for prior features matching a phrase "
                "such as 'the last hole', 'the chamfer', 'the 12mm extrude'. "
                "Returns ranked matches (each with index / name / type / params / "
                "score / rationale). Use for prompts that reference earlier "
                "features ('make the last hole 2mm deeper', 'remove the chamfer')."
            ),
            args_schema=_FindFeatureByPhraseInput,
        ),
    ]


class _FaceByDescriptionInput(BaseModel):
    body_handle: str = Field(
        description="Stable handle of the parent body (from the brep summary), e.g. 'b_a39f...'."
    )
    phrase: str = Field(
        description="Free-form description of the target face, e.g. 'front face', 'the round face on top'."
    )
    top_n: int = Field(default=4, description="Maximum number of ranked candidates to return.")


class _FindFeatureByPhraseInput(BaseModel):
    phrase: str = Field(
        description="Free-form description of the target timeline feature, e.g. 'the last hole'."
    )
    top_n: int = Field(default=4, description="Maximum number of ranked matches to return.")


# Stage 3 / C1 — confidence threshold below which the resolver flags
# `needs_clarification = True` instead of guessing. Stage 3's clarify_node
# consumes this; for telemetry-driven tuning, change this single constant.
# Default 0.7 matches the brief's recommendation (§3 / C1).
CLARIFICATION_CONFIDENCE_THRESHOLD = 0.7
# Backwards-compat alias so any caller that already imported the old name
# keeps working. (Both names point at the same value — change in one place.)
TARGET_RESOLVER_CONFIDENCE_THRESHOLD = CLARIFICATION_CONFIDENCE_THRESHOLD


TARGET_RESOLVER_SYSTEM_PROMPT = """\
You are the TARGET RESOLVER for a Fusion 360 agent.

Your job: convert the user's natural-language target reference into a
concrete, validated set of entity handles. You DO NOT write code. You DO NOT
plan operations. You PICK THE TARGET.

════════════════════════════════════════════════════════════════════════════
HARD RULES
════════════════════════════════════════════════════════════════════════════
1. **Selection is authoritative.** If the Selection Context is non-empty,
   THAT selection IS the target. Use it unless the user prompt explicitly
   contradicts it ("not this one — the other body"). Set
   `target_source = "selection"` and confidence = 1.0.

2. **When unsure, ask — do not guess.** If multiple candidates score
   similarly OR the prompt is genuinely ambiguous OR no candidate scores
   above the confidence threshold, set `needs_clarification = true` and
   list 2–4 candidates in `alternatives`. A wrong silent pick is far worse
   than a clarification round.

3. **Use handles, not indices.** Every body reference MUST be a
   `body_handle` (the sha1-prefix stamp from the brep summary). Face
   references should carry both `entity_token` and `face_index`.

4. **Output STRICT JSON inside a ```json ... ``` block.** The schema:

   ```json
   {
     "primary_body":  {"handle": "b_a39f1234", "name": "Bracket", "confidence": 0.92},
     "primary_face":  {"entity_token": "...", "face_index": 7,
                       "direction": "-Y", "confidence": 0.81},
     "primary_feature": null,
     "alternatives":  [{"kind": "body|face|feature", "handle": "...",
                        "label": "...", "confidence": 0.55}],
     "rationale":     "User said 'the front', selection empty. -Y face matches the camera-front of the bracket.",
     "needs_clarification": false,
     "target_source": "selection|prompt|inference"
   }
   ```
   Omit `primary_face` / `primary_feature` if not applicable. `alternatives`
   may be empty when confidence is high.

5. **Tool budget: at most 5 calls.** You have:
     lookup_entity            — read full descriptor of a face/edge/vertex
     face_by_description      — rank faces by phrase (camera-aware)
     find_feature_by_phrase   — search timeline features by phrase
     bounding_box             — measure entities or whole design
     face_properties          — read precise face geometry by token

   Use them only when the brep summary + selection don't already make the
   pick obvious.

When a viewport screenshot or camera_frame is supplied, use them to ground
viewer-relative phrases ("front", "the visible hole", "the side facing me").
"""


PROBING_SYSTEM_PROMPT = """\
You are a precision CAD measurement assistant for Autodesk Fusion 360.

You have access to spatial probing tools that query the live 3D geometry.
Use them to collect only the measurements actually needed for the user's request.

Tool catalogue:
  bounding_box       → dimensions, center, min/max of entities or whole design
  min_distance       → clearance gap between two surfaces
  connected_faces    → adjacent face types and areas (for context on neighbours)
  face_properties    → exact radius, normal, axis, origin of a specific face
  face_sketch_bounds → width/height, sketch-space bbox, and a pre-inset "safe
                       interior" rectangle of a face. REQUIRED before planning
                       any hole layout on a face — tells you how many holes
                       of a given radius will actually fit without exceeding
                       the face bounds (which would cause EXTRUDE_BOOLEAN_FAIL).
  edge_chain         → total length of a tangent edge loop (fillets, chamfers)
  body_volume        → volume, mass, center of mass
  svg_section        → SVG wireframe of a 2D cross-section (for complex profiles)
  blueprint_views    → confirm orthographic blueprint capture

Rules:
- Make 2–5 targeted measurements maximum; avoid redundant calls.
- After your tool calls, write a concise MEASUREMENTS SUMMARY paragraph.
- Reference entity tokens from the geometry context when calling tools.
- Do NOT write any Fusion API code here — that is the coder node's job.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """\
You are a master Mechanical Engineer and expert Fusion 360 CAD architect.

Analyze the user's natural-language request together with the current B-Rep
geometry context.  Produce a **concise, numbered, step-by-step plan** of the
Fusion 360 operations needed to fulfil the request.

════════════════════════════════════════════════════════════════════════════════
MODIFY vs. CREATE — READ THIS FIRST
════════════════════════════════════════════════════════════════════════════════
If the "Current Geometry" section lists one or more existing bodies, the user
is almost certainly asking you to MODIFY those bodies, not to create brand new
geometry next to them.  Verbs like "add holes", "make a pocket", "cut", "round",
"fillet", "chamfer", "shell", "drill", "extend", "thicken", "combine", "pattern",
and any reference words like "this", "it", "the cube", "the part" all mean
"operate ON the existing body".

When modifying existing bodies you MUST:
  1. Name the target body explicitly in the plan, using its name AND body_index
     from the Current Geometry context (e.g. "target: Body1 (body_index=0)").
  2. Place sketches on an EXISTING face of that body (e.g. "the +Z top face of
     Body1") rather than on a raw construction plane whenever possible.  If no
     face is suitable, use an offset construction plane referenced to the body.
  3. Use a subtractive / additive / intersect operation against the existing
     body — NEVER `NewBodyFeatureOperation`.  Pick one of:
       - CutFeatureOperation        (holes, pockets, slots, drilling, removing material)
       - JoinFeatureOperation       (adding bosses, ribs, combining)
       - IntersectFeatureOperation  (trimming to a volume)
  4. Declare `participantBodies = [<the target body>]` so the operation knows
     which body to affect.

Only use `NewBodyFeatureOperation` when the design is empty OR the user
explicitly asks for a separate new body ("add a second cube next to it",
"create a cylinder on the side as a new body").  When in doubt, MODIFY.

════════════════════════════════════════════════════════════════════════════════
GENERAL GUIDELINES
════════════════════════════════════════════════════════════════════════════════
- Think in primitive Fusion 360 operations: sketch → profile → extrude/revolve/cut → fillet/chamfer → pattern/mirror.
- Always specify which construction plane (XY, XZ, YZ, offset) OR which face of
  which existing body each sketch goes on.
- If the user has a selection (see "Selection Context"), that selection is the
  explicit target — use it and reference its parent body.
- All measurements in the plan should be in **millimetres (mm)** — the code
  generator will convert to Fusion's internal centimetres.
- Keep the plan to at most 8 steps.  Prefer fewer, larger steps over many tiny ones.
- Do NOT write code.  Only the plan.

You may also be given a screenshot of the current Fusion 360 viewport and a
``camera_frame`` describing the user's viewpoint. Use them to ground references
like "the front face" or "the visible hole" against what the user is actually
looking at — not the world axes. When the user says "front", they almost always
mean the face that's currently facing them in the viewport.

If a ``Resolved Target`` block is present at the top of the context, treat
its handles as authoritative — they were already validated by the
target_resolver step. Reference the supplied ``body_handle`` and (when
present) ``face_index`` / ``entity_token`` directly in your plan; do not
second-guess them.
"""

CODER_SYSTEM_PROMPT = """\
You are an expert Autodesk Fusion 360 Python API developer.
Write a Python script that implements the provided CAD plan.
You must wrap all of your Python code inside standard markdown fences (```python ... ```)

════════════════════════════════════════════════════════════════════════════════
CRITICAL RULES
════════════════════════════════════════════════════════════════════════════════
1. The script runs via `exec()` inside Fusion 360. The following variables are
   already in scope — do NOT re-declare or import them:
      adsk          (the adsk module)
      app           (adsk.core.Application.get())
      design        (adsk.fusion.Design, the active design)
      rootComp      (design.rootComponent)
      extrudes      (rootComp.features.extrudeFeatures)
      camera_tools  (module with capture_screenshot())
      plane_manager (PlaneManager instance for creating offset/construction planes)
      face_finder   (face / body / bbox / cut-through lookup helpers — Rule 11+)
      modify_tools  (vetted wrappers for fillet, chamfer, hole, shell, combine,
                     move, circular_pattern, rectangular_pattern, mirror,
                     extrude_join, edges_by_direction — Rule 14)

2. ALL distance/coordinate values in the Fusion 360 Python API use
   **centimetres (cm)** internally.  When the plan specifies millimetres,
   convert:  mm = cm / 10.0  or use the helper `_mm_to_cm(value_mm)`.

3. Only output raw Python code.  Do NOT wrap it in markdown fences (```python).
   Do NOT add `import adsk.core` / `import adsk.fusion` at the top.

4. Wrap the entire script in a single try/except that sets a variable
   `_result = {"success": True, "message": "..."}` on success and
   `_result = {"success": False, "error": str(e), "traceback": ...}` on failure.

5. Prefer `extrudes.addSimple(profile, distance, operation)` for simple
   extrusions.  Use `extrudes.createInput(profile, operation)` only when you
   need taper angle, symmetric extent, or two-side extent.

6. For creating holes, prefer sketch circles with cut extrude over the hole
   feature API (more reliable across Fusion versions).

7. Use `adsk.core.ValueInput.createByReal(value_cm)` for numeric inputs and
   `adsk.core.ValueInput.createByString("10 mm")` when you want Fusion to
   parse the unit string.

8. When referencing an existing body for join/cut/intersect operations, obtain
   it via `rootComp.bRepBodies.item(index)` or by name lookup — do not assume
   variable names from prior steps are still in scope (each exec() starts fresh).

9. Name features clearly: `feature.timelineObject.name = "Main Box Extrude"`.

10. **MODIFY, DO NOT CREATE** when bodies already exist.
    If the "Current Geometry" context lists one or more bodies, the user's
    request is almost always a MODIFICATION of those bodies (holes, pockets,
    fillets, joins, …).  Do NOT use `NewBodyFeatureOperation` in that case.
    Instead:
      a. **The target body is supplied as `target_body_handle` in the
         context — use it directly via `face_finder.body_by_handle`:**
             target_body = face_finder.body_by_handle(rootComp, target_body_handle)
         The Stage-2 target_resolver picked this handle BEFORE planning ran;
         do not re-resolve, do not guess, and do not fall back to
         `body_by_index_or_name(0)` when a handle is supplied.
         (When `target_body_handle` is genuinely missing — empty design or
         the resolver explicitly returned `null` — only then fall back to a
         fresh resolution from the brep summary.)
         The legacy `face_finder.body_by_index_or_name(rootComp, 0)` still
         works but is DEPRECATED — body_index drifts after boolean ops, splits,
         and deletions, while body_handle does not.
      b. Place the sketch on an existing FACE of that body whenever the plan
         calls for sketching on "the top/front/etc. face", e.g.:
             target_face = face_finder.face_by_direction(target_body, "+Z")
             sketch = rootComp.sketches.add(target_face)
         If you need a specific face by index, fetch its descriptor first via
         the `lookup_entity` probe (kind='face') so you can confirm surface
         type, normal, and area before sketching on it.
         If the plan only specifies a construction plane, still keep the cut /
         join operation bound to the existing body via `participantBodies`.
      c. Use the correct FeatureOperations enum:
             - CutFeatureOperation       → removing material (holes, pockets, slots)
             - JoinFeatureOperation      → adding material (bosses, ribs)
             - IntersectFeatureOperation → trimming the body to a volume
      d. Always set `ext_input.participantBodies = [target_body]` for Cut / Join /
         Intersect extrudes, otherwise Fusion may quietly create a new body.

    The ONLY times `NewBodyFeatureOperation` is appropriate are:
      - The design is completely empty, OR
      - The user explicitly asked for a separate new body ("add a second
        cube next to it as a new body").

11. **BoundingBox objects are NOT iterable — never unpack them.**
    Fusion's ``BoundingBox2D`` (returned by ``Profile.boundingBox``) and
    ``BoundingBox3D`` (returned by ``BRepBody.boundingBox``,
    ``BRepFace.boundingBox``, ``Sketch.boundingBox``, …) have NO
    ``__iter__``. Writing::

        mn, mx = body.boundingBox           # TypeError — DON'T
        (x, y, z) = face.boundingBox.minPoint  # TypeError — DON'T

    raises ``TypeError: cannot unpack non-iterable BoundingBox{2D,3D} object``.
    Access the points explicitly instead::

        bbox = body.boundingBox
        mn = bbox.minPoint            # a Point3D
        mx = bbox.maxPoint            # a Point3D
        dx = mx.x - mn.x              # centimetres (Fusion internal units)

    Or — preferred — use the helpers already in scope::

        mn_tuple, mx_tuple = face_finder.bbox_tuple(body)   # both in cm
        bbox_mm = face_finder.bbox_mm(body)
        # bbox_mm == {"min":[...],"max":[...],"size":[...],"center":[...]}  (mm)
        thickness_cm = face_finder.body_thickness_cm(body, "-Y")

12. **Cut-through-body: ALWAYS use `face_finder.cut_through_body`. Do NOT
    hand-roll `ThroughAllExtentDefinition`, `setOneSideExtent`, or
    `setSymmetricExtent` calls.**
    Direct use of the extrude API has three common silent failures that
    collectively account for ~all "cut failed" errors:

    (a) Missing ``ext_input.participantBodies = [target_body]`` →
        "Could not complete Through All Extrude, body not found to extrude
        through". Fusion does NOT infer the target body.

    (b) Wrong ``ExtentDirections`` value (Positive vs Negative). Through-all
        travels along the sketch plane's own normal; whichever side of the
        plane has body material is the direction you need. Trial-and-error
        retries produce nondeterministic results on multi-body or partially-
        coincident sketch planes.

    (c) ``setSymmetricExtent(ThroughAllExtentDefinition.create(), True)``
        DOES NOT EXIST. The Fusion API signature is
            setSymmetricExtent(distance: ValueInput, isFullLength: bool)
            setSymmetricExtent(distance: ValueInput, isFullLength: bool,
                               taperAngle: ValueInput)
        — there is NO overload that accepts an extent-definition object.
        Passing one yields
            "Wrong number or type of arguments for overloaded function
             'ExtrudeFeatureInput_setSymmetricExtent'".

    The helper handles all three correctly. Call it like this::

        face_finder.cut_through_body(target_body, profile)
        # symmetric (sketch plane sits inside the body):
        face_finder.cut_through_body(target_body, profile, symmetric=True)
        # with profile-inside-face validation for safety:
        face_finder.cut_through_body(
            target_body, profile_or_profiles, target_face=target_face
        )

    ``profile`` may be a single ``Profile``, an ``ObjectCollection``, or any
    iterable of profiles. The ``direction`` argument is accepted for
    backwards compatibility but IGNORED — the helper computes the correct
    side from ``BRepBody.pointContainment`` rather than guessing.

14. **Use `modify_tools` for every modify-feature operation.** The Fusion
    Features API has many subtle shape pitfalls (e.g. ``ChamferFeatures
    .createInput(edges, isTangentChain)`` — distance is set AFTER, not in the
    constructor; ``setSymmetricExtent`` does not take an extent definition;
    pattern ``quantity`` and ``totalAngle`` are ``ValueInput`` not int/float).
    The helpers below set every required field and accept friendly inputs:

      modify_tools.fillet_edges(body, edges_or_spec, radius_mm,
                                tangent_chain=True)
        edges_or_spec accepts a BRepEdge, a list, an ObjectCollection, OR a
        string: "all", "vertical", "+Z", "+X", "+Y" — for "fillet all
        vertical edges 2mm" pass ``"+Z"``.

      modify_tools.chamfer_edges(body, edges_or_spec, distance_mm,
                                 tangent_chain=True)

      modify_tools.add_simple_hole(target_face, center_world, diameter_mm,
                                   depth_mm=None)
        center_world: a world-space Point3D. depth_mm=None ⇒ through-all.

      modify_tools.shell_body(body, thickness_mm, faces_to_remove=None,
                              inside=True)

      modify_tools.combine_bodies(target_body, tool_bodies,
                                  operation="join"|"cut"|"intersect",
                                  keep_tools=False)

      modify_tools.move_body(body, dx_mm=0, dy_mm=0, dz_mm=0)
        Parametric translate — survives feature replay. Do NOT set
        body.transform directly; that is non-parametric.

      modify_tools.circular_pattern(features, axis, count,
                                    total_angle_deg=360)
        ``axis`` may be a ConstructionAxis, a linear edge, or a planar face
        (its normal is used). For "4 holes around the centre of a face":
            axis = modify_tools.construction_axis_through_face(face)
            modify_tools.circular_pattern(hole_feature, axis, 4)

      modify_tools.rectangular_pattern(features, direction_one, count_one,
                                       distance_one_mm,
                                       direction_two=None, count_two=1,
                                       distance_two_mm=0.0)
        Distances are TOTAL pattern lengths, not pitch.

      modify_tools.mirror_features(features, mirror_plane)

      modify_tools.extrude_join(target_body, profile, distance_mm,
                                direction="outward"|"inward",
                                target_face=None)
        Adds material onto an existing body. Mirror image of
        face_finder.cut_through_body. Sets participantBodies.

      modify_tools.edges_by_direction(body, direction, tolerance_cos=0.95)
        Returns linear edges parallel (or anti-parallel) to ``direction``.

      modify_tools.plan_circular_hole_layout(sketch, face, n_holes,
                                             radius_from_center_mm,
                                             hole_radius_mm, margin_mm=2,
                                             start_angle_deg=0)
        Sketch-space Point3Ds for a bolt circle, validated to fit the face.

    DO NOT hand-roll ``filletFeatures.createInput()``,
    ``chamferFeatures.createInput()``, ``shellFeatures.createInput()``,
    ``combineFeatures.createInput()``, ``moveFeatures.createInput*()``,
    ``circularPatternFeatures.createInput()``,
    ``rectangularPatternFeatures.createInput()``, or
    ``mirrorFeatures.createInput()``. The helpers above do all of that
    correctly.

15. **Use `face_finder` to locate faces by direction.**
    Do NOT write hand-rolled loops like
        `for f in body.faces: if f.normal == (0, -1, 0): ...`
    They produce errors such as `"Could not find front face (-Y)"` because
    of float mismatches, reversed parametric surfaces, and unit issues.
    Instead call the exposed helper:

        target_body = face_finder.body_by_handle(rootComp, "<body_handle>")
        # legacy: face_finder.body_by_index_or_name(rootComp, 0)
        front_face = face_finder.face_by_direction(target_body, "-Y")
        top_face   = face_finder.face_by_direction(target_body, "top")

    Accepted directions: ``"+X"``, ``"-X"``, ``"+Y"``, ``"-Y"``, ``"+Z"``,
    ``"-Z"``, plus the aliases ``"top"``, ``"bottom"``, ``"front"`` (=-Y),
    ``"back"``, ``"left"``, ``"right"``, ``"up"``, ``"down"``. You may also
    pass a 3-tuple (x, y, z) or an ``adsk.core.Vector3D``.

    **Viewer-relative directions** — when the user's prompt references the
    viewport ("the front face", "the visible side", "round the edge facing
    me") pass the in-scope ``camera_frame`` so the lookup is resolved
    relative to the user's viewpoint instead of world axes:

        # camera_frame is already in scope; pass it explicitly so labels like
        # "front" track the user's view even on a rotated body.
        front_face = face_finder.face_by_direction(
            target_body, "front", camera_frame=camera_frame,
        )

    Camera-relative resolution only kicks in for the viewer labels
    {"front","back","left","right","top","bottom","up","down"}; world-axis
    labels ("+X" / "-Y" / tuples / Vector3D) always resolve against world.
    When ``camera_frame`` is None or omitted, the helper falls back to the
    legacy world-axis logic — safe to always pass.

    `face_by_direction` raises ``LookupError`` when no matching face exists,
    which propagates naturally through the required try/except wrapper, so
    you never need to raise a custom "could not find face" error.

    For all matching faces, use ``face_finder.faces_by_direction(body, dir)``
    which returns a list; ``face_by_direction`` picks the largest by default.

════════════════════════════════════════════════════════════════════════════════
COMMON PATTERNS
════════════════════════════════════════════════════════════════════════════════

Cut a hole through an existing body (most common modify-request pattern):
    # 1. Resolve target body — use the body_handle from Current Geometry.
    target_body = face_finder.body_by_handle(rootComp, "<body_handle from brep summary>")

    # 2. Pick the face that the hole should be CUT INTO. The face you pick
    #    must be the face you sketch on — the sketch lives on that face's
    #    plane, and the extrude direction is the face's inward normal.
    #    For a hole on a "side panel" of a box: pick -Y or +Y, NOT "top".
    target_face = face_finder.face_by_direction(target_body, "-Y")
    sketch = rootComp.sketches.add(target_face)

    # 3. Draw the hole profile INSIDE the face's bounding region.
    #    Placing a circle at (0,0,0) blindly is a common cause of
    #    "EXTRUDE_BOOLEAN_FAIL / extrusion profile falls outside the
    #    boundary of the selected body". The sketch coordinate system is
    #    relative to the chosen face, so convert a world-space point (the
    #    face centroid is a safe default) to sketch coords.
    hole_centre_world  = target_face.centroid                       # Point3D (cm)
    hole_centre_sketch = sketch.modelToSketchSpace(hole_centre_world)
    circles = sketch.sketchCurves.sketchCircles
    circles.addByCenterRadius(hole_centre_sketch, 0.6)              # 12mm dia -> 6mm radius
    profile = sketch.profiles.item(0)

    # 4. Cut THROUGH the existing body using the helper. It sets
    #    participantBodies, picks the correct extent direction via
    #    BRepBody.pointContainment (no trial-and-error), and validates the
    #    profile lies inside the target face when one is supplied.
    cut_feature = face_finder.cut_through_body(
        target_body, profile, target_face=target_face
    )

Cut multiple holes on the same face (e.g. "4 holes on the side panel"):
    # ALWAYS use face_finder.plan_hole_layout — never hand-roll offsets from
    # the body bbox. The face lies in a 2-D plane, so two of the three body
    # axes are irrelevant; mixing them puts circles outside the face and
    # raises "EXTRUDE_BOOLEAN_FAIL — profile falls outside the boundary".
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    target_face = face_finder.face_by_direction(target_body, "-Y")
    sketch = rootComp.sketches.add(target_face)

    # plan_hole_layout returns sketch-space Point3Ds guaranteed to lie inside
    # the face, leaving a 2mm margin around the holes. It raises ValueError
    # if the face is too small — let that propagate through the try/except.
    centres = face_finder.plan_hole_layout(
        sketch, target_face,
        rows=2, cols=2,
        hole_radius_mm=6.0,   # 12mm diameter
        margin_mm=2.0,
    )

    circles = sketch.sketchCurves.sketchCircles
    for c in centres:
        circles.addByCenterRadius(c, 0.6)  # 6mm radius in cm

    all_profiles = [sketch.profiles.item(i) for i in range(sketch.profiles.count)]

    # Pass target_face so cut_through_body validates profile-inside-face
    # BEFORE Fusion's boolean engine sees it; validation raises a clear
    # ValueError instead of a cryptic EXTRUDE_BOOLEAN_FAIL.
    face_finder.cut_through_body(
        target_body, all_profiles, target_face=target_face
    )

Create a sketch and draw a rectangle (ONLY when no existing body should be modified):
    sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
    lines = sketch.sketchCurves.sketchLines
    # Coordinates in cm (mm * 10)
    lines.addTwoPointRectangle(
        adsk.core.Point3D.create(0, 0, 0),
        adsk.core.Point3D.create(5.0, 3.0, 0)  # 50mm x 30mm
    )
    profile = sketch.profiles.item(0)

Simple extrude (New Body — ONLY when design is empty or user explicitly asked for a new body):
    dist = adsk.core.ValueInput.createByReal(2.0)  # 20mm
    extrude_feature = extrudes.addSimple(
        profile, dist,
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )

Join extrude (add a boss onto an existing body) — preferred via helper:
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    target_face = face_finder.face_by_direction(target_body, "+Z")
    sketch = rootComp.sketches.add(target_face)
    centre = sketch.modelToSketchSpace(target_face.centroid)
    sketch.sketchCurves.sketchCircles.addByCenterRadius(centre, 0.4)  # 8mm radius
    profile = sketch.profiles.item(0)
    modify_tools.extrude_join(
        target_body, profile, distance_mm=10.0, target_face=target_face,
    )

Fillet edges by intent (e.g. "fillet all vertical edges 2 mm"):
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    modify_tools.fillet_edges(target_body, "+Z", radius_mm=2.0)
    # Or: every edge of the body
    # modify_tools.fillet_edges(target_body, "all", radius_mm=2.0)
    # Or: edges of one specific face
    # face = face_finder.face_by_direction(target_body, "top")
    # modify_tools.fillet_edges(target_body,
    #                           modify_tools.edges_of_face(face),
    #                           radius_mm=2.0)

Chamfer edges:
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    modify_tools.chamfer_edges(target_body, "all", distance_mm=1.5)

Hole feature on a face (proper Fusion Hole, not a cut extrude):
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    target_face = face_finder.face_by_direction(target_body, "top")
    centre_world = target_face.centroid  # cm Point3D
    modify_tools.add_simple_hole(target_face, centre_world, diameter_mm=10.0)
    # depth_mm=None ⇒ through-all; pass a number for a blind hole.

Bolt-circle (4 holes around the centre of a face — circular pattern of one hole):
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    target_face = face_finder.face_by_direction(target_body, "top")

    # 1. One reference hole. The pattern produces the other three.
    centre_sk = face_finder.face_sketch_bbox(
        rootComp.sketches.add(target_face), target_face
    )  # only used to compute layout — discard the throwaway sketch is fine.
    sample_hole = modify_tools.add_simple_hole(
        target_face,
        target_face.pointOnFace,   # arbitrary world-space point on the face
        diameter_mm=8.0,
    )

    # 2. Pattern around the face normal axis.
    axis = modify_tools.construction_axis_through_face(target_face)
    modify_tools.circular_pattern(sample_hole, axis, count=4)

Rectangular grid of cuts (e.g. "4×3 holes on the side panel"):
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    target_face = face_finder.face_by_direction(target_body, "-Y")
    sketch = rootComp.sketches.add(target_face)
    centres = face_finder.plan_hole_layout(
        sketch, target_face, rows=3, cols=4,
        hole_radius_mm=4.0, margin_mm=3.0,
    )
    for c in centres:
        sketch.sketchCurves.sketchCircles.addByCenterRadius(c, 0.4)
    profiles = [sketch.profiles.item(i) for i in range(sketch.profiles.count)]
    face_finder.cut_through_body(
        target_body, profiles, target_face=target_face,
    )

Shell a closed body open on the top:
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    top_face = face_finder.face_by_direction(target_body, "top")
    modify_tools.shell_body(target_body, thickness_mm=2.5,
                            faces_to_remove=[top_face])

Combine two bodies:
    # body_a is the resolver's target; body_b is the secondary tool body.
    # Address the secondary by handle from the brep summary too — addressing
    # by index 1 still works but drifts after boolean ops / replays.
    body_a = face_finder.body_by_handle(rootComp, target_body_handle)
    body_b = face_finder.body_by_handle(rootComp, "<other body_handle from brep summary>")
    modify_tools.combine_bodies(body_a, body_b, operation="cut")

Mirror a feature across a face / construction plane:
    target_body = face_finder.body_by_handle(rootComp, target_body_handle)
    mirror_plane = face_finder.face_by_direction(target_body, "+X")
    modify_tools.mirror_features(some_feature, mirror_plane)

Move a body parametrically:
    body = face_finder.body_by_handle(rootComp, target_body_handle)
    modify_tools.move_body(body, dx_mm=20.0, dz_mm=-5.0)

Fillet edges:
    fillets = rootComp.features.filletFeatures
    edge_col = adsk.core.ObjectCollection.create()
    edge = rootComp.bRepBodies.item(0).edges.item(0)
    edge_col.add(edge)
    fillet_input = fillets.createInput()
    fillet_input.addConstantRadiusEdgeSet(
        edge_col,
        adsk.core.ValueInput.createByReal(0.2),  # 2mm radius
        True  # tangent chain
    )
    fillet_feature = fillets.add(fillet_input)

Revolve:
    revolve_feats = rootComp.features.revolveFeatures
    revolve_input = revolve_feats.createInput(
        profile,
        sketch_line,  # axis line
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )
    angle = adsk.core.ValueInput.createByReal(2 * math.pi)
    revolve_input.setAngleExtent(False, angle)
    revolve_feats.add(revolve_input)

Create a circle and extrude:
    sketch = rootComp.sketches.add(rootComp.xYConstructionPlane)
    circles = sketch.sketchCurves.sketchCircles
    center = adsk.core.Point3D.create(0, 0, 0)
    circles.addByCenterRadius(center, 1.5)  # radius in cm (15mm)
    profile = sketch.profiles.item(0)
    dist = adsk.core.ValueInput.createByReal(3.0)  # 30mm height
    extrudes.addSimple(profile, dist,
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)

Offset construction plane:
    plane = plane_manager.get_or_create_offset_plane("XY", 5.0)  # offset 50mm
    sketch = rootComp.sketches.add(plane)

════════════════════════════════════════════════════════════════════════════════
"""

CRITIC_SYSTEM_PROMPT = """\
You are a senior Fusion 360 Python API code reviewer.  Your job is to catch
bugs, API misuse, and unit-conversion mistakes before the code is sent to
Fusion 360 for live execution.

Review the generated Python code against the plan and B-Rep context.
Check for:
1. Unit consistency — all numeric values passed to the Fusion API must be in cm.
2. Missing `try/except` — the script must always set _result on success or failure.
3. Stale references — don't reference sketch/profile variables that may not
   exist after a failed prior step.
4. API misuse — wrong method names, missing required arguments, incorrect enum values.
5. Off-by-one — Fusion collections are 0-indexed (`item(0)`, not `item(1)`).

If the code looks correct, respond with exactly:
    APPROVED

If there are problems, respond with the corrected full script. You must wrap the corrected Python code inside standard markdown fences (```python ... ```).
"""

VISION_CRITIC_SYSTEM_PROMPT = """\
You are a visual QA inspector for Autodesk Fusion 360 CAD designs.

You will be shown a screenshot of the Fusion 360 viewport after a Python script
was executed to fulfil a user's design request.

Evaluate whether the resulting geometry visually matches what the user asked for.

If the design looks correct and matches the intent, respond with exactly:
    APPROVED

If there is a visible geometric mistake (wrong shape, missing feature, wrong
dimensions relative to the description, extra geometry, etc.), respond with a
concise description of what is wrong and what needs to change in the code to
fix it.  Be specific — reference geometry (bodies, faces, edges), approximate
dimensions, and what operation went wrong.
"""

ERROR_RECOVERY_SYSTEM_PROMPT = """\
You are an expert Fusion 360 Python API debugger.

The previous code that was sent to Fusion 360 failed with the error shown below.
Analyze the error, determine the root cause, and rewrite the ENTIRE script
to fix it.  Follow all the same rules as the code generator:
- No import statements for adsk
- All distances in cm (mm * 10.0)
- Wrap in try/except setting _result
- Wrap the ENTIRE fixed script in a single ```python ... ``` markdown block.

════════════════════════════════════════════════════════════════════════════════
CRITICAL — DO NOT GIVE UP ON MODIFYING THE EXISTING BODY
════════════════════════════════════════════════════════════════════════════════
If the original request was a MODIFY-style request (add holes, cut a pocket,
fillet, etc.) and bodies already exist in "Current Geometry", then:

  • You MUST keep using `CutFeatureOperation` / `JoinFeatureOperation` /
    `IntersectFeatureOperation` with `participantBodies = [target_body]`.
  • You MUST NOT "fix" the error by switching to `NewBodyFeatureOperation`.
    Creating a new separate body when the user asked to modify the existing
    one is a SILENT FAILURE, not a recovery — it looks like the script ran
    but the cube wasn't modified.  This is strictly forbidden.

Even if the error is ambiguous, prefer: keep the operation, fix the geometry
(sketch position, face choice, extent direction or distance, participant
body) — do NOT change the FeatureOperation type.

════════════════════════════════════════════════════════════════════════════════
COMMON FAILURE MODES
════════════════════════════════════════════════════════════════════════════════
1. "No profile found" → the sketch didn't form a closed region.  Redraw more carefully.
2. "Index out of range" on .item(N) → fewer items exist than expected; use a safer index.
3. "Invalid argument" on extrude/fillet → wrong units (cm vs mm) or null profile.
4. "No target body" → the body to cut/join was not found; resolve it via the
   resolver's stable handle: `face_finder.body_by_handle(rootComp, target_body_handle)`.
   (Legacy `body_by_index_or_name(rootComp, <index or name>)` still works as a
   fallback, but body_index drifts after boolean ops / replays.)
5. "EXTRUDE_BOOLEAN_FAIL" / "extrusion profile falls outside the boundary of
   the selected body" / "Cannot extend extrusion to object":
     • The sketch was placed on the wrong face, OR the circle/rectangle is
       outside the body's footprint on that face, OR the cut extent pushes
       the cut past the body with no material to remove.
     • FIX by (a) sketching on the correct face — use
       `face_finder.face_by_direction(target_body, "<dir>")` to pick a face
       that the hole actually passes THROUGH (e.g. a side face for holes
       through the side, not the bottom);
       (b) placing the circle/rectangle centre inside that face's bbox —
       use `target_body.boundingBox` to stay inside;
       (c) using a bounded extent — either a positive `setDistanceExtent`
       equal to the body thickness in that direction, or
       `setOneSideToExtent(ToEntityExtentDefinition(opposite_face, False))`
       / `setSymmetricExtent(...)` so the cut does not overshoot.
     • DO NOT switch to NewBodyFeatureOperation to "avoid" this error —
       the user wants the existing body modified.
6. "A feature could not be created because of geometric conditions" →
   almost always the same class of issue as #5 — reposition the sketch,
   do not drop the participantBodies list.

7. "Could not complete Through All Extrude, body not found to extrude through" →
   The ``setOneSideExtent(ThroughAllExtentDefinition.create(), ...)`` (or the
   equivalent ``setSymmetricExtent``) call was made but
   ``ext_input.participantBodies`` was NOT set to the target body. Fusion then
   has no body to bore through and raises this error. FIX by either
     (a) calling ``face_finder.cut_through_body(body, profile, "-Y")`` which
         sets participantBodies and the ThroughAll extent correctly, OR
     (b) explicitly setting
             ext_input.participantBodies = [target_body]
         BEFORE calling ``extrudes.add(ext_input)``.
   Also verify that the sketch face and direction actually have the body
   ahead of them — a "through all" cut starting from a face that points
   AWAY from the body cannot find material to bore through.

8. "cannot unpack non-iterable BoundingBox2D object" (or BoundingBox3D) →
   The script tried to do ``mn, mx = some.boundingBox`` or similar.
   BoundingBox objects are NOT iterable. FIX by accessing points directly::
       bbox = body.boundingBox
       mn, mx = bbox.minPoint, bbox.maxPoint  # both are Point3D
       dx = mx.x - mn.x                       # cm, Fusion internal units
   Or use ``face_finder.bbox_tuple(entity)`` which returns
   ``((mn_x, mn_y, mn_z), (mx_x, mx_y, mx_z))`` — safely unpackable.

9. "Wrong number or type of arguments for overloaded function
    'ExtrudeFeatureInput_setSymmetricExtent'" →
   You called ``setSymmetricExtent(ThroughAllExtentDefinition.create(), ...)``.
   That overload DOES NOT EXIST. The Fusion API only defines
       setSymmetricExtent(distance: ValueInput, isFullLength: bool)
       setSymmetricExtent(distance: ValueInput, isFullLength: bool,
                          taperAngle: ValueInput)
   FIX by calling ``face_finder.cut_through_body(body, profile,
   symmetric=True)`` — it uses an oversized distance (2.1× body diagonal)
   with the correct two-argument signature. Do NOT re-emit the faulty
   signature.

10. If the SAME through-all or cut error re-appears on retry →
    Stop hand-rolling the extrude input. Replace the entire cut block with
    a SINGLE call to ``face_finder.cut_through_body(body, profile,
    target_face=target_face)``. It handles participantBodies, direction
    choice via pointContainment, and profile-inside-face validation. If the
    helper raises ``ValueError("profile ... is OUTSIDE the target face")``,
    the fix is NOT to try again with the same geometry — shrink the hole
    or reposition it using ``face_finder.plan_hole_layout(sketch, face,
    rows, cols, hole_radius_mm=...)`` which always returns points inside
    the face.

If the ORIGINAL request was explicitly for a new body (empty design or the
user said "add a new separate body"), then NewBodyFeatureOperation is of
course fine — but verify that against the "Current Geometry" + user request
before switching.

════════════════════════════════════════════════════════════════════════════════
PREVIOUS ATTEMPTS LOG — DO NOT REPEAT THE SAME FIX
════════════════════════════════════════════════════════════════════════════════
A list of prior recovery attempts is appended to the human prompt under
"Previous attempts". Read it carefully. If a fix you would have tried is
already listed there with the same error, change strategy — do not re-emit
the same script with the same justification. The retry budget is small
(MAX_RETRIES); each round must be PROVABLY DIFFERENT from the previous one.
"""


def _fmt_attempts_log(attempts_log: Optional[List[Dict[str, Any]]]) -> str:
    """Render the persistent attempts log for the recovery prompt."""
    if not attempts_log:
        return ""
    lines = ["Previous attempts (do not repeat the same fix):"]
    for i, entry in enumerate(attempts_log, 1):
        if not isinstance(entry, dict):
            continue
        approach   = entry.get("approach")   or "?"
        error_type = entry.get("error_type") or "?"
        error_msg  = (entry.get("error_message") or "").strip().splitlines()
        first_line = error_msg[0] if error_msg else "?"
        fix_tried  = entry.get("fix_tried") or "?"
        lines.append(
            f"  {i}. approach={approach!r}, error_type={error_type}, "
            f"error={first_line[:200]!r}, fix_tried={fix_tried!r}"
        )
    return "\n".join(lines)


def _summarise_approach(code: str, max_len: int = 200) -> str:
    """Best-effort one-line summary of what a generated script *did*."""
    if not isinstance(code, str):
        return "(no code)"
    keep: List[str] = []
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("import ") or line.startswith("from "):
            continue
        keep.append(line)
        if len(keep) >= 6:
            break
    snippet = "; ".join(keep)
    return snippet[:max_len] + ("…" if len(snippet) > max_len else "")


def _summarise_fix_tried(code: str, prev_code: str, max_len: int = 200) -> str:
    """Best-effort description of what changed between the previous and the
    new script. Pure structural — keeps recovery prompts honest about what
    the LLM ACTUALLY changed, not what it said it changed."""
    new_lines = [l.strip() for l in (code or "").splitlines() if l.strip() and not l.strip().startswith("#")]
    prev_lines = set(l.strip() for l in (prev_code or "").splitlines() if l.strip() and not l.strip().startswith("#"))
    added = [l for l in new_lines if l not in prev_lines][:4]
    if not added:
        return "(no structural change — likely a no-op rewrite)"
    return ("added: " + " | ".join(added))[:max_len]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper — strip markdown code fences from LLM output
# ═══════════════════════════════════════════════════════════════════════════════

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Extract code from inside markdown fences, discarding any preamble/postamble text."""
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Agent Nodes
# ═══════════════════════════════════════════════════════════════════════════════


async def probing_node(state: AgentState) -> dict:
    """
    Tier 4 — Agentic spatial probing (ReAct style).

    The LLM calls live measurement tools (bounding_box, min_distance, etc.)
    before planning.  Skipped when no probe_callback is available (e.g. during
    testing or when the design is empty).
    """
    probe_callback = state.get("probe_callback")
    if not probe_callback:
        logger.info("[probing_node] No probe_callback — skipping")
        return {}

    brep = state.get("brep_context") or {}
    if not brep.get("bodies") and not brep.get("faces"):
        logger.info("[probing_node] Empty geometry — skipping probing")
        return {}

    logger.info("[probing_node] Starting spatial probing…")

    tools = _make_probe_tools(probe_callback)
    tool_map = {t.name: t for t in tools}

    llm = _make_llm(
        model_name=state.get("model_name", "claude-opus-4-6"),
        api_keys=state.get("api_keys"),
        max_tokens=2048,
    ).bind_tools(tools)

    brep_text = _fmt_brep(brep)
    timeline_text = _fmt_timeline(state.get("timeline_context") or [])
    ctx = brep_text + (f"\n\n{timeline_text}" if timeline_text else "")

    # Cache the system prompt + the stable geometry context. The probing
    # ReAct loop sends this same prefix on every round; only the assistant /
    # tool-result messages grow per iteration.
    messages = [
        _cacheable_system(PROBING_SYSTEM_PROMPT),
        _cacheable_human(
            cached_prefix=f"Available geometry:\n{ctx}",
            dynamic_tail=(
                f"User request: {state['user_prompt']}\n\n"
                "Use probing tools to gather exact measurements you need, then write your MEASUREMENTS SUMMARY."
            ),
        ),
    ]

    probe_results: List[Dict[str, Any]] = []
    stream_cb = state.get("stream_callback")

    for _ in range(5):  # max 5 tool-call rounds
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            logger.warning("[probing_node] LLM invoke failed: %s", exc)
            break
        _log_cache_usage("probing_node", response)

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break  # LLM finished probing

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            call_id = tc["id"]

            logger.info("[probing_node] → %s(%s)", tool_name, tool_args)
            if stream_cb:
                try:
                    await stream_cb("reasoning_chunk", f"\n[Probe: {tool_name}]")
                except Exception:
                    pass

            try:
                fn = tool_map.get(tool_name)
                if fn:
                    tool_result = await fn.ainvoke(tool_args)
                else:
                    tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                probe_results.append({"tool": tool_name, "args": tool_args, "result": tool_result})
            except Exception as exc:
                tool_result = json.dumps({"error": str(exc)})
                logger.warning("[probing_node] Tool %s failed: %s", tool_name, exc)

            messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id))

    # Extract text summary from the last non-tool-call LLM response
    probe_summary = ""
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                probe_summary = content.strip()
                break

    logger.info("[probing_node] Done: %d measurements, summary=%d chars",
                len(probe_results), len(probe_summary))

    return {"probe_results": probe_results, "probe_summary": probe_summary}


def _resolver_tool_subset(probe_callback: Callable) -> List[StructuredTool]:
    """Return only the tools the target_resolver is allowed to call.

    Restricting the toolkit keeps the resolver's prompt short and signals
    intent: don't measure the world generally — pick the target.
    """
    allowed = {
        "lookup_entity",
        "face_by_description",
        "find_feature_by_phrase",
        "bounding_box",
        "face_properties",
    }
    return [t for t in _make_probe_tools(probe_callback) if t.name in allowed]


# Strict-JSON block extractor: supports ```json fences, plain ``` fences,
# and bare JSON. The resolver prompt asks for fenced JSON; this is the
# tolerant fallback so a stray missing fence doesn't blow up the node.
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_resolved_target_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the resolver's JSON payload out of an LLM response, tolerantly."""
    if not isinstance(text, str) or not text.strip():
        return None

    # Try a fenced block first.
    m = _JSON_BLOCK_RE.search(text)
    candidates = []
    if m:
        candidates.append(m.group(1).strip())
    # Also try the entire payload as a JSON object (no fences).
    candidates.append(text.strip())
    # And the substring between the first '{' and the last '}'.
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])

    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _selection_to_resolved_target(selection_context: Any) -> Optional[Dict[str, Any]]:
    """Build a 100%-confidence resolved_target dict directly from
    selection_context. Used as a fast path BEFORE invoking the LLM when the
    user has explicitly clicked entities — selection is authoritative.
    """
    if not selection_context:
        return None
    entities = []
    if isinstance(selection_context, dict):
        entities = selection_context.get("entities") or []
    elif isinstance(selection_context, list):
        entities = selection_context
    if not entities:
        return None

    primary_body: Optional[Dict[str, Any]] = None
    primary_face: Optional[Dict[str, Any]] = None
    rationale_bits: List[str] = []

    for ent in entities:
        if not isinstance(ent, dict):
            continue
        etype = ent.get("type")
        bh = ent.get("body_handle")
        bn = ent.get("body_name") or ent.get("name")
        if etype == "body" and bh and not primary_body:
            primary_body = {"handle": bh, "name": bn, "confidence": 1.0}
            rationale_bits.append(f"selection contained body '{bn}' (handle={bh})")
        elif etype == "face":
            if bh and not primary_body:
                primary_body = {"handle": bh, "name": bn, "confidence": 1.0}
            if not primary_face:
                primary_face = {
                    "entity_token": ent.get("entity_token") or ent.get("token"),
                    "face_index":   ent.get("face_index"),
                    "direction":    None,
                    "confidence":   1.0,
                }
                rationale_bits.append(
                    f"selection contained face_index={ent.get('face_index')} of '{bn}'"
                )
        elif etype == "edge" and bh and not primary_body:
            primary_body = {"handle": bh, "name": bn, "confidence": 1.0}
            rationale_bits.append(
                f"selection contained edge of '{bn}' (handle={bh})"
            )

    if not primary_body and not primary_face:
        return None

    return {
        "primary_body":         primary_body,
        "primary_face":         primary_face,
        "primary_feature":      None,
        "alternatives":         [],
        "rationale":            "; ".join(rationale_bits) or "selection used as target",
        "needs_clarification":  False,
        "target_source":        "selection",
    }


async def target_resolver_node(state: AgentState) -> dict:
    """
    Stage 2 / B1 — pick the operation target BEFORE planning runs.

    Pre-selection fast path: if `selection_context` is non-empty, we lock
    the selection in as the target without burning an LLM call (selection
    is authoritative). Otherwise we run a bounded ReAct loop with a small
    toolkit (lookup_entity, face_by_description, find_feature_by_phrase,
    bounding_box, face_properties) and parse the LLM's strict-JSON output
    into `state["resolved_target"]`.

    Output keys:
      resolved_target  — see TARGET_RESOLVER_SYSTEM_PROMPT schema.
      target_source    — "selection" | "prompt" | "inference".
    """
    logger.info("[target_resolver_node] Resolving target…")

    # ── Fast path: selection-as-target ──────────────────────────────────────
    selection_resolved = _selection_to_resolved_target(state.get("selection_context"))
    if selection_resolved is not None:
        logger.info(
            "[target_resolver_node] Selection-authoritative path: handle=%s",
            (selection_resolved.get("primary_body") or {}).get("handle"),
        )
        return {
            "resolved_target": selection_resolved,
            "target_source":   "selection",
        }

    brep = state.get("brep_context") or {}
    if not brep.get("bodies"):
        logger.info("[target_resolver_node] Empty design — no target to resolve, skipping")
        return {"resolved_target": None, "target_source": None}

    probe_callback = state.get("probe_callback")
    tools = _resolver_tool_subset(probe_callback) if probe_callback else []
    tool_map = {t.name: t for t in tools}

    llm = _make_llm(
        model_name=state.get("model_name", "claude-opus-4-6"),
        api_keys=state.get("api_keys"),
        max_tokens=2048,
    )
    if tools:
        llm = llm.bind_tools(tools)

    # Build the human prompt: same context the planner gets, minus probe
    # results (resolver runs its own measurements when needed).
    brep_text = _fmt_brep(brep)
    selection_text = _fmt_selection(state.get("selection_context"))
    timeline_text = _fmt_timeline(state.get("timeline_context") or [])
    camera_frame = state.get("camera_frame")

    context_parts = [f"Current Geometry:\n{brep_text}", f"\n{timeline_text}"]
    if selection_text:
        context_parts.append(f"\nSelection Context:\n{selection_text}")
    else:
        context_parts.append("\nSelection Context: (empty — user did not click anything)")
    if camera_frame:
        context_parts.append(
            "\nCamera Frame (mm, world-space):\n"
            f"  origin    = {camera_frame.get('origin')}\n"
            f"  target    = {camera_frame.get('target')}\n"
            f"  up_vector = {camera_frame.get('up_vector')}\n"
            f"  eye_vector= {camera_frame.get('eye_vector')}"
        )

    # Cache the static brep / timeline / selection / camera context so the
    # ReAct loop's repeated sends are billed at 0.10x.
    cached_prefix = "\n".join(context_parts)
    dynamic_tail = (
        f"User Request: {state['user_prompt']}\n\n"
        "Resolve the target now and emit the JSON block."
    )
    viewport_screenshot = state.get("viewport_screenshot")
    first_message: Any = _cacheable_human(
        cached_prefix=cached_prefix,
        dynamic_tail=dynamic_tail,
        image_b64=viewport_screenshot,
    )

    messages: List[Any] = [
        _cacheable_system(TARGET_RESOLVER_SYSTEM_PROMPT),
        first_message,
    ]

    stream_cb = state.get("stream_callback")
    final_text = ""

    for _ in range(5):  # max 5 ReAct rounds
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            logger.warning("[target_resolver_node] LLM invoke failed: %s", exc)
            break
        _log_cache_usage("target_resolver_node", response)

        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(response, "content", "")
            final_text = content if isinstance(content, str) else str(content)
            break

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            call_id = tc["id"]
            logger.info("[target_resolver_node] → %s(%s)", tool_name, tool_args)
            if stream_cb:
                try:
                    await stream_cb("reasoning_chunk", f"\n[Resolver: {tool_name}]")
                except Exception:
                    pass
            try:
                fn = tool_map.get(tool_name)
                if fn:
                    tool_result = await fn.ainvoke(tool_args)
                else:
                    tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            except Exception as exc:
                tool_result = json.dumps({"error": str(exc)})
                logger.warning("[target_resolver_node] tool %s failed: %s", tool_name, exc)

            messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id))

    parsed = _extract_resolved_target_json(final_text) if final_text else None

    if not parsed:
        logger.warning(
            "[target_resolver_node] Could not parse JSON resolution; "
            "passing through with needs_clarification=true (text=%r)",
            (final_text or "")[:200],
        )
        return {
            "resolved_target": {
                "primary_body":        None,
                "primary_face":        None,
                "primary_feature":     None,
                "alternatives":        [],
                "rationale":           "Resolver produced no parseable JSON.",
                "needs_clarification": True,
                "target_source":       "inference",
            },
            "target_source": "inference",
        }

    parsed.setdefault("needs_clarification", False)
    parsed.setdefault("alternatives", [])
    parsed.setdefault("target_source", "prompt")
    parsed.setdefault("rationale", "")

    # Apply the confidence threshold: low primary confidence → clarify.
    primary_body = parsed.get("primary_body") or {}
    primary_face = parsed.get("primary_face") or {}
    confidences = [c for c in (primary_body.get("confidence"), primary_face.get("confidence")) if isinstance(c, (int, float))]
    max_confidence = max(confidences) if confidences else 0.0

    if (
        not parsed.get("needs_clarification")
        and max_confidence < CLARIFICATION_CONFIDENCE_THRESHOLD
    ):
        logger.info(
            "[target_resolver_node] max_confidence=%.2f < threshold %.2f → forcing needs_clarification",
            max_confidence, CLARIFICATION_CONFIDENCE_THRESHOLD,
        )
        parsed["needs_clarification"] = True

    target_source = parsed.get("target_source") or "inference"
    logger.info(
        "[target_resolver_node] Resolved: source=%s, needs_clarification=%s, max_conf=%.2f, "
        "rationale=%r",
        target_source, parsed.get("needs_clarification"), max_confidence,
        (parsed.get("rationale") or "")[:200],
    )
    return {"resolved_target": parsed, "target_source": target_source}


def _fmt_resolved_target(resolved: Optional[Dict[str, Any]]) -> str:
    """Render `resolved_target` as a one-paragraph banner for planner/coder."""
    if not resolved:
        return ""
    parts: List[str] = []
    pb = resolved.get("primary_body") or {}
    pf = resolved.get("primary_face") or {}
    pfeat = resolved.get("primary_feature") or {}
    target_source = resolved.get("target_source")
    rationale = resolved.get("rationale") or ""
    needs_clar = resolved.get("needs_clarification")

    bits: List[str] = []
    if pb.get("handle"):
        name = pb.get("name") or ""
        conf = pb.get("confidence")
        bits.append(f"body_handle={pb['handle']}" + (f" name={name!r}" if name else "")
                    + (f" conf={conf:.2f}" if isinstance(conf, (int, float)) else ""))
    if pf.get("entity_token") or pf.get("face_index") is not None:
        face_bits = []
        if pf.get("entity_token"):
            face_bits.append(f"entity_token={pf['entity_token']}")
        if pf.get("face_index") is not None:
            face_bits.append(f"face_index={pf['face_index']}")
        if pf.get("direction"):
            face_bits.append(f"direction={pf['direction']}")
        bits.append("face: " + ", ".join(face_bits))
    if pfeat.get("token") or pfeat.get("index") is not None:
        feat_bits = []
        if pfeat.get("type"):
            feat_bits.append(f"type={pfeat['type']}")
        if pfeat.get("index") is not None:
            feat_bits.append(f"index={pfeat['index']}")
        bits.append("feature: " + ", ".join(feat_bits))

    parts.append("Resolved Target: " + ("; ".join(bits) if bits else "(none)"))
    if target_source:
        parts.append(f"  source: {target_source}")
    if needs_clar:
        parts.append("  needs_clarification: true (Stage 3 will surface a UI; "
                     "for now, proceed using primary_* if present, otherwise the most "
                     "plausible candidate from alternatives, and note your chosen pick "
                     "in the plan).")
    if rationale:
        parts.append(f"  rationale: {rationale}")
    return "\n".join(parts)


async def clarify_node(state: AgentState) -> dict:
    """Stage 3 / C1 — surface a clarification dialogue to the user.

    Triggered by `_post_resolver_route` when the resolver flagged
    `needs_clarification=True`. Builds an option list from
    `resolved_target.alternatives` (and falls back to the primary candidates
    when the resolver didn't enumerate any), emits a `clarification_request`
    over the WebSocket, and suspends until the user clicks an option.

    On response, the chosen handle is written back into
    `resolved_target.primary_*` and `target_source` is set to "user".
    On cancel / timeout / missing callback, leaves `resolved_target` as-is
    and lets the planner proceed with the resolver's best guess so the
    pipeline doesn't deadlock.
    """
    resolved = state.get("resolved_target") or {}
    callback = state.get("clarify_callback")
    user_prompt = state.get("user_prompt", "")

    if not callback:
        logger.warning("[clarify_node] No clarify_callback in state — skipping dialogue")
        return {"resolved_target": resolved}

    # Build the option list. Prefer `alternatives`; otherwise fall back to
    # whichever primary candidates the resolver did surface so the user can
    # at least confirm vs reject.
    options: List[Dict[str, Any]] = []
    alternatives = resolved.get("alternatives") or []
    for i, alt in enumerate(alternatives):
        if not isinstance(alt, dict):
            continue
        kind = alt.get("kind") or "body"
        handle = alt.get("handle") or alt.get("entity_token")
        label = alt.get("label") or _default_alternative_label(alt)
        confidence = alt.get("confidence")
        if not handle:
            continue
        options.append({
            "id":         f"alt_{i}",
            "handle":     handle,
            "kind":       kind,
            "label":      label,
            "confidence": confidence,
        })

    if not options:
        # No enumerated alternatives — fall back to primary_body / primary_face
        # as a confirm-or-cancel choice.
        pb = resolved.get("primary_body") or {}
        pf = resolved.get("primary_face") or {}
        if pb.get("handle"):
            options.append({
                "id":         "alt_primary_body",
                "handle":     pb["handle"],
                "kind":       "body",
                "label":      f"{pb.get('name') or 'Body'} (resolver's pick)",
                "confidence": pb.get("confidence"),
            })
        if pf.get("entity_token") or pf.get("face_index") is not None:
            options.append({
                "id":         "alt_primary_face",
                "handle":     pf.get("entity_token") or "",
                "kind":       "face",
                "label":      f"face_index={pf.get('face_index')} ({pf.get('direction')})",
                "confidence": pf.get("confidence"),
            })

    if not options:
        logger.warning(
            "[clarify_node] No usable options to present — passing through "
            "with the resolver's best guess (rationale=%r)",
            (resolved.get("rationale") or "")[:200],
        )
        return {"resolved_target": resolved}

    question = (
        f"I'm not sure which target to use for: '{user_prompt}'. Please pick one."
    )
    rationale = resolved.get("rationale") or ""

    logger.info(
        "[clarify_node] Asking user — %d option(s); rationale=%r",
        len(options), rationale[:120],
    )
    stream_cb = state.get("stream_callback")
    if stream_cb:
        try:
            await stream_cb("reasoning_chunk", "\n[Asking user to clarify the target…]")
        except Exception:
            pass

    try:
        response = await callback(question, options, rationale=rationale)
    except Exception as exc:
        logger.warning("[clarify_node] callback raised: %s — proceeding with best guess", exc)
        return {"resolved_target": resolved}

    if not isinstance(response, dict) or response.get("cancelled"):
        logger.info(
            "[clarify_node] User did not pick (response=%s) — proceeding with best guess",
            response,
        )
        return {"resolved_target": resolved}

    selected_id = response.get("selected_option_id")
    chosen = next((o for o in options if o["id"] == selected_id), None)
    if chosen is None:
        # Allow the widget to also pass back the raw handle if it didn't
        # echo the option id.
        raw_handle = (response.get("raw") or {}).get("handle")
        chosen = next((o for o in options if o["handle"] == raw_handle), None)
    if chosen is None:
        logger.warning(
            "[clarify_node] Response %s did not match any option — proceeding with best guess",
            response,
        )
        return {"resolved_target": resolved}

    # Write the locked-in handle back into the resolved target.
    updated = dict(resolved)
    updated["needs_clarification"] = False
    updated["target_source"] = "user"
    if chosen["kind"] == "body":
        updated["primary_body"] = {
            "handle":     chosen["handle"],
            "name":       chosen.get("label"),
            "confidence": 1.0,
        }
    elif chosen["kind"] == "face":
        updated["primary_face"] = {
            "entity_token": chosen["handle"],
            "face_index":   None,  # widget may not know it; coder still has the token
            "direction":    None,
            "confidence":   1.0,
        }
    elif chosen["kind"] == "feature":
        updated["primary_feature"] = {
            "token":      chosen["handle"],
            "type":       chosen.get("label"),
            "confidence": 1.0,
        }

    rationale_bits = [updated.get("rationale") or ""]
    rationale_bits.append(
        f"User clarification: picked option '{chosen['id']}' "
        f"(kind={chosen['kind']}, handle={chosen['handle']})."
    )
    updated["rationale"] = "; ".join(b for b in rationale_bits if b).strip("; ")

    logger.info(
        "[clarify_node] Locked in handle=%s kind=%s",
        chosen["handle"], chosen["kind"],
    )

    return {
        "resolved_target": updated,
        "target_source":   "user",
    }


def _default_alternative_label(alt: Dict[str, Any]) -> str:
    """Derive a concise human label from an alternative dict when the
    resolver didn't supply one explicitly."""
    handle = alt.get("handle") or alt.get("entity_token") or "?"
    kind = alt.get("kind") or "entity"
    name = alt.get("name")
    if name:
        return f"{name} ({kind} {handle})"
    return f"{kind} {handle}"


async def planner_node(state: AgentState) -> dict:
    """
    Analyse the user's request and geometry context, then produce a
    numbered step-by-step plan of Fusion 360 operations.

    When a `viewport_screenshot` is present in state (Stage 1 / A4), it is
    attached to the planner's HumanMessage as a multimodal image alongside
    the text context, so the LLM can ground references like "the front face"
    against what the user is actually looking at. The optional `camera_frame`
    is rendered as a small text block giving eye/target/up.
    """
    logger.info("[planner_node] Generating plan…")

    llm = _make_llm(
        model_name=state.get("model_name", "claude-sonnet-4-5-20250514"),
        api_keys=state.get("api_keys"),
    )

    brep_text = _fmt_brep(state.get("brep_context"))
    selection_text = _fmt_selection(state.get("selection_context"))
    spatial_text = _fmt_spatial(state.get("spatial_context"))
    timeline_text = _fmt_timeline(state.get("timeline_context") or [])
    probe_text = _fmt_probe_results(
        state.get("probe_results") or [], state.get("probe_summary") or ""
    )
    camera_frame = state.get("camera_frame")
    resolved_target_text = _fmt_resolved_target(state.get("resolved_target"))

    # Stage 2: resolved target leads the context so the planner anchors on a
    # concrete handle instead of guessing from the brep dump.
    context_parts: List[str] = []
    if resolved_target_text:
        context_parts.append(resolved_target_text)
    context_parts.append(f"\nCurrent Geometry:\n{brep_text}")
    if selection_text:
        context_parts.append(f"\nSelection Context:\n{selection_text}")
    if spatial_text:
        context_parts.append(f"\nSpatial Context:\n{spatial_text}")
    # Timeline is always rendered (with a "(no features yet)" placeholder
    # when empty) so the agent always knows whether prior features exist.
    context_parts.append(f"\n{timeline_text}")
    if probe_text:
        context_parts.append(f"\n{probe_text}")
    if camera_frame:
        context_parts.append(
            "\nCamera Frame (mm, world-space):\n"
            f"  origin    = {camera_frame.get('origin')}\n"
            f"  target    = {camera_frame.get('target')}\n"
            f"  up_vector = {camera_frame.get('up_vector')}\n"
            f"  eye_vector= {camera_frame.get('eye_vector')}"
        )

    # Cache the heavy stable prefix (resolved target + brep + selection +
    # spatial + timeline + probes + camera) so coder/critic/recovery on the
    # SAME turn re-use this exact text and pay 0.10× for it.
    cached_prefix = "\n".join(context_parts)
    dynamic_tail = f"User Request: {state['user_prompt']}"

    viewport_screenshot = state.get("viewport_screenshot")
    human_message = _cacheable_human(
        cached_prefix=cached_prefix,
        dynamic_tail=dynamic_tail,
        image_b64=viewport_screenshot,
    )
    if viewport_screenshot:
        logger.info(
            "[planner_node] Multimodal payload: text(cached)=%d chars + dynamic=%d chars + image=%d b64 chars",
            len(cached_prefix), len(dynamic_tail), len(viewport_screenshot),
        )
    else:
        logger.info(
            "[planner_node] Text-only payload: cached=%d chars + dynamic=%d chars",
            len(cached_prefix), len(dynamic_tail),
        )

    # Stream plan chunks to client if callback is available
    callback = state.get("stream_callback")

    collected: List[str] = []
    final_response: Any = None
    async for chunk in llm.astream(
        [
            _cacheable_system(PLANNER_SYSTEM_PROMPT),
            human_message,
        ],
    ):
        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        collected.append(token)
        final_response = chunk  # last chunk carries the aggregate usage_metadata
        if callback:
            try:
                await callback("plan_chunk", token)
            except Exception:
                pass  # don't let streaming errors break the pipeline

    plan = "".join(collected)
    if final_response is not None:
        _log_cache_usage("planner_node", final_response)
    logger.info("[planner_node] Plan generated (%d chars)", len(plan))
    return {"plan": plan}


async def coder_node(state: AgentState) -> dict:
    """
    Translate the plan into executable Fusion 360 Python API code.
    """
    logger.info("[coder_node] Generating code…")

    llm = _make_llm(
        model_name=state.get("model_name", "claude-sonnet-4-5-20250514"),
        api_keys=state.get("api_keys"),
    )

    brep_text = _fmt_brep(state.get("brep_context"))
    selection_text = _fmt_selection(state.get("selection_context"))
    spatial_text = _fmt_spatial(state.get("spatial_context"))
    timeline_text = _fmt_timeline(state.get("timeline_context") or [])
    probe_text = _fmt_probe_results(
        state.get("probe_results") or [], state.get("probe_summary") or ""
    )
    resolved_target_text = _fmt_resolved_target(state.get("resolved_target"))

    # Stage 2: resolved target leads the context — generated code MUST refer
    # to the supplied body_handle via face_finder.body_by_handle().
    context_parts: List[str] = []
    if resolved_target_text:
        context_parts.append(resolved_target_text)
    context_parts.append(f"\nCurrent Geometry:\n{brep_text}")
    if selection_text:
        context_parts.append(f"\nSelection Context:\n{selection_text}")
    if spatial_text:
        context_parts.append(f"\nSpatial Context:\n{spatial_text}")
    # Timeline is always rendered (with a "(no features yet)" placeholder
    # when empty) so the agent always knows whether prior features exist.
    context_parts.append(f"\n{timeline_text}")
    if probe_text:
        context_parts.append(f"\n{probe_text}")

    error_log = state.get("error_log", "")
    if error_log:
        context_parts.append(f"\nPrevious Errors (fix these):\n{error_log}")

    # Surface the canonical target_body_handle for the coder so it can be
    # referenced verbatim from the prompt when generating code.
    resolved = state.get("resolved_target") or {}
    primary_body = resolved.get("primary_body") or {}
    target_body_handle = primary_body.get("handle")
    if target_body_handle:
        context_parts.append(
            f"\nCanonical handle for generated code:\n"
            f"  target_body_handle = \"{target_body_handle}\"\n"
            f"  → use: face_finder.body_by_handle(rootComp, target_body_handle)"
        )

    # Cache the heavy stable prefix (resolved target + brep + selection +
    # spatial + timeline + probes + canonical handle). The plan changes per
    # turn but is short; the brep dump is identical to what planner just
    # sent, so it gets a 0.10× cache read.
    cached_prefix = "\n".join(context_parts)
    dynamic_tail = f"Plan:\n{state['plan']}"

    # Stream reasoning chunks to client if callback is available
    callback = state.get("stream_callback")

    collected: List[str] = []
    final_response: Any = None
    async for chunk in llm.astream(
        [
            _cacheable_system(CODER_SYSTEM_PROMPT),
            _cacheable_human(cached_prefix=cached_prefix, dynamic_tail=dynamic_tail),
        ],
    ):
        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        collected.append(token)
        final_response = chunk
        if callback:
            try:
                await callback("reasoning_chunk", token)
            except Exception:
                pass

    raw_code = "".join(collected)
    clean_code = _strip_code_fences(raw_code)
    if final_response is not None:
        _log_cache_usage("coder_node", final_response)
    logger.info("[coder_node] Code generated (%d chars)", len(clean_code))
    # Clear error_log now that the coder has consumed it. The new code has not
    # executed yet, so any prior error is no longer "the current" error — leaving
    # it in state would cause _should_retry to route straight back into
    # error_recovery, producing a redundant regeneration loop.
    return {"generated_code": clean_code, "error_log": ""}


async def critic_node(state: AgentState) -> dict:
    """
    Review generated code for correctness before it is sent to Fusion 360.

    If the LLM approves, the code passes through unchanged.  If the LLM
    returns corrected code (e.g. fixing unit bugs), that corrected code
    replaces the generated code.
    """
    logger.info("[critic_node] Reviewing code…")

    llm = _make_llm(
        model_name=state.get("model_name", "claude-sonnet-4-5-20250514"),
        api_keys=state.get("api_keys"),
        temperature=0,
        max_tokens=4096,
    )

    brep_text = _fmt_brep(state.get("brep_context"))
    # Brep + plan are already in cache from coder_node — let the critic re-use
    # the same prefix shape so it bills as a cache read.
    cached_prefix = (
        f"Current Geometry:\n{brep_text}\n\n"
        f"Plan:\n{state['plan']}"
    )
    dynamic_tail = f"Generated Code:\n{state['generated_code']}"

    response = await llm.ainvoke(
        [
            _cacheable_system(CRITIC_SYSTEM_PROMPT),
            _cacheable_human(cached_prefix=cached_prefix, dynamic_tail=dynamic_tail),
        ],
    )
    _log_cache_usage("critic_node", response)

    content = response.content.strip()
    # If the critic only says "APPROVED" (possibly with minor surrounding text),
    # pass the code through unchanged.
    if content.upper().startswith("APPROVED"):
        logger.info("[critic_node] Code approved — passing through")
        return {"generated_code": state["generated_code"]}

    # Otherwise, the critic returned corrected code.
    corrected = _strip_code_fences(content)
    logger.info("[critic_node] Code corrected (%d chars)", len(corrected))
    return {"generated_code": corrected}


async def error_recovery_node(state: AgentState) -> dict:
    """
    Analyse an execution error from Fusion 360 and regenerate code.

    This node is reached when the client reports an `execution_result` with
    success=False. Before rewriting, it runs a short ReAct probing loop (max
    3 rounds) so the LLM can MEASURE the live geometry to figure out WHY the
    previous attempt failed (e.g. "did the profile fall outside the face?",
    "is the body still where I thought it was?"). Without this, recovery is
    forced to guess again from the same evidence that produced the wrong
    answer the first time.
    """
    retry = state.get("retry_count", 0)
    logger.info("[error_recovery_node] Retry %d — regenerating code", retry)

    brep_text = _fmt_brep(state.get("brep_context"))
    timeline_text = _fmt_timeline(state.get("timeline_context") or [])
    error_log = state.get("error_log", "Unknown error")
    probe_callback = state.get("probe_callback")

    # ── Phase 1 — bounded probing ReAct loop ────────────────────────────────
    new_probe_results: List[Dict[str, Any]] = []
    recovery_probe_summary = ""

    if probe_callback is not None:
        tools = _make_probe_tools(probe_callback)
        tool_map = {t.name: t for t in tools}
        probe_llm = _make_llm(
            model_name=state.get("model_name", "claude-opus-4-6"),
            api_keys=state.get("api_keys"),
            max_tokens=2048,
        ).bind_tools(tools)

        existing_probe_text = _fmt_probe_results(
            state.get("probe_results") or [], state.get("probe_summary") or ""
        )

        recovery_probe_prompt = (
            ERROR_RECOVERY_SYSTEM_PROMPT
            + "\n\n"
            + "════════════════════════════════════════════════════════════\n"
            + "RECOVERY PROBING PHASE — MEASURE BEFORE REWRITING\n"
            + "════════════════════════════════════════════════════════════\n"
            + "Before rewriting the script, use probing tools to check WHY "
              "the previous attempt failed. Examples:\n"
            + "  • Cut overshot the body → call `bounding_box` on the target "
              "body and `face_sketch_bounds` on the cut face to confirm the "
              "profile lies inside the face.\n"
            + "  • Wrong face picked → call `lookup_entity` to read the face's "
              "actual surface_type / normal before sketching on it.\n"
            + "  • Body not found → call `lookup_entity` with kind='face' on "
              "each candidate body_handle to confirm the body is still there.\n"
            + "Use AT MOST 3 probe rounds, then return a brief MEASUREMENTS "
              "SUMMARY paragraph. Do NOT write code in this phase."
        )

        # Cache the heavy prefix (system + brep + timeline + prior probe
        # context) so each round of the recovery ReAct loop bills as a 0.10×
        # cache read after the first.
        cached_prefix = (
            f"Current Geometry:\n{brep_text}\n\n"
            f"{timeline_text}"
            + (f"\n\n{existing_probe_text}" if existing_probe_text else "")
            + f"\n\nFailed Code:\n{state['generated_code']}"
        )
        dynamic_tail = (
            f"User Request: {state['user_prompt']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Error:\n{error_log}\n\n"
            "Identify the most likely root cause and use 1-3 probe calls to "
            "verify it, then write your MEASUREMENTS SUMMARY."
        )
        messages: List[Any] = [
            _cacheable_system(recovery_probe_prompt),
            _cacheable_human(cached_prefix=cached_prefix, dynamic_tail=dynamic_tail),
        ]

        stream_cb = state.get("stream_callback")
        for _ in range(3):  # max 3 ReAct rounds in recovery
            try:
                response = await probe_llm.ainvoke(messages)
            except Exception as exc:
                logger.warning("[error_recovery_node] probe LLM invoke failed: %s", exc)
                break
            _log_cache_usage("error_recovery_node.probing", response)

            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break

            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                call_id = tc["id"]
                logger.info("[error_recovery_node] probe → %s(%s)", tool_name, tool_args)
                if stream_cb:
                    try:
                        await stream_cb("reasoning_chunk", f"\n[Recovery probe: {tool_name}]")
                    except Exception:
                        pass

                try:
                    fn = tool_map.get(tool_name)
                    if fn:
                        tool_result = await fn.ainvoke(tool_args)
                    else:
                        tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})
                    new_probe_results.append({"tool": tool_name, "args": tool_args, "result": tool_result})
                except Exception as exc:
                    tool_result = json.dumps({"error": str(exc)})
                    logger.warning("[error_recovery_node] probe tool %s failed: %s", tool_name, exc)

                messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id))

        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and not getattr(msg, "tool_calls", None):
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    recovery_probe_summary = content.strip()
                    break

        logger.info(
            "[error_recovery_node] probing done: %d new measurements, summary=%d chars",
            len(new_probe_results),
            len(recovery_probe_summary),
        )
    else:
        logger.info("[error_recovery_node] No probe_callback — skipping probing phase")

    # Merge fresh probe results back into state so the next coder pass also sees them.
    merged_probe_results = list(state.get("probe_results") or []) + new_probe_results
    if recovery_probe_summary:
        prior_summary = state.get("probe_summary") or ""
        if prior_summary:
            merged_probe_summary = (
                f"{prior_summary}\n\n[Recovery probe summary]\n{recovery_probe_summary}"
            )
        else:
            merged_probe_summary = recovery_probe_summary
    else:
        merged_probe_summary = state.get("probe_summary") or ""

    # ── Phase 2 — regenerate the code with measurements in hand ─────────────
    llm = _make_llm(
        model_name=state.get("model_name", "claude-sonnet-4-5-20250514"),
        api_keys=state.get("api_keys"),
    )

    probe_text = _fmt_probe_results(merged_probe_results, merged_probe_summary)
    probe_section = f"\n\n{probe_text}\n" if probe_text else ""

    # Stage 4 / D3 — render the persistent attempts log so the recovery LLM
    # can see what was already tried. We render BEFORE appending the current
    # round's entry; the current round will be appended on return so the
    # NEXT recovery round sees this attempt too.
    prior_attempts = list(state.get("attempts_log") or [])
    attempts_text = _fmt_attempts_log(prior_attempts)
    attempts_section = f"\n\n{attempts_text}\n" if attempts_text else ""

    # Heavy stable prefix: brep + timeline + probes + failed code + attempts.
    # The error_log and the user request are the per-round dynamic tail.
    cached_prefix = (
        f"Current Geometry:\n{brep_text}\n\n"
        f"{timeline_text}"
        f"{probe_section}"
        f"{attempts_section}"
        f"Failed Code:\n{state['generated_code']}"
    )
    dynamic_tail = (
        f"User Request: {state['user_prompt']}\n\n"
        f"Plan:\n{state['plan']}\n\n"
        f"Error:\n{error_log}"
    )

    response = await llm.ainvoke(
        [
            _cacheable_system(ERROR_RECOVERY_SYSTEM_PROMPT),
            _cacheable_human(cached_prefix=cached_prefix, dynamic_tail=dynamic_tail),
        ],
    )
    _log_cache_usage("error_recovery_node.regen", response)

    corrected_code = _strip_code_fences(response.content)
    logger.info("[error_recovery_node] Regenerated code (%d chars)", len(corrected_code))

    # Stage 4 / D3 — append this round to the persistent attempts log.
    error_first_line = (error_log or "").strip().splitlines()[:1]
    error_type = "ExecutionError"
    if error_first_line:
        first = error_first_line[0]
        # The recovery flow's error_log starts with "Error: <type>: <message>".
        # Pull a coarse type out for the log, leave full message in error_message.
        if first.startswith("Error: "):
            tail = first[len("Error: "):]
            error_type = tail.split(":", 1)[0][:80] or "ExecutionError"
    new_attempt = {
        "approach":      _summarise_approach(state.get("generated_code", "")),
        "error_type":    error_type,
        "error_message": (error_log or "")[:500],
        "fix_tried":     _summarise_fix_tried(corrected_code, state.get("generated_code", "")),
        "retry_index":   retry,
    }
    updated_attempts = prior_attempts + [new_attempt]

    # Clear error_log: the recovery node has just consumed it to rewrite the
    # script. The new script has not run yet, so there is no current error.
    # Without this, the critic -> _should_retry path would loop back into
    # error_recovery on the same stale error until MAX_RETRIES is exhausted.
    return {
        "generated_code": corrected_code,
        "retry_count": retry + 1,
        "error_log": "",
        "probe_results": merged_probe_results,
        "probe_summary": merged_probe_summary,
        "attempts_log": updated_attempts,
    }

async def vision_critic_node(state: AgentState) -> dict:
    """
    Visually evaluate the Fusion 360 viewport screenshot after code execution.

    Sends the screenshot to Claude as a multimodal message.  If the geometry
    looks wrong, writes a descriptive error into error_log so the graph loops
    back through error_recovery → coder.  If it looks correct, clears error_log.

    Skipped automatically when no screenshot is available (e.g. first pass
    before any execution has happened).
    """
    logger.info("[vision_critic_node] Evaluating result visually…")

    screenshot_b64 = state.get("screenshot_base64", "")
    if not screenshot_b64:
        logger.info("[vision_critic_node] No screenshot in state — skipping")
        return {"error_log": ""}

    llm = _make_llm(
        model_name=state.get("model_name", "claude-sonnet-4-5-20250514"),
        api_keys=state.get("api_keys"),
        temperature=0,
    )

    prompt = (
        f"The user asked: '{state['user_prompt']}'\n\n"
        "Look at the Fusion 360 viewport screenshot. Does the resulting geometry "
        "match the request? If yes, respond APPROVED. Otherwise describe the "
        "problem precisely so the code can be fixed."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            },
        ]
    )

    response = await llm.ainvoke([
        _cacheable_system(VISION_CRITIC_SYSTEM_PROMPT),
        message,
    ])
    _log_cache_usage("vision_critic_node", response)

    content = response.content.strip()
    if content.upper().startswith("APPROVED"):
        logger.info("[vision_critic_node] Visual check passed")
        return {"error_log": ""}

    logger.info("[vision_critic_node] Visual issue detected: %s", content[:200])
    return {"error_log": f"Visual Evaluation Failed: {content}"}

# ═══════════════════════════════════════════════════════════════════════════════
# Routing Logic
# ═══════════════════════════════════════════════════════════════════════════════


def _should_retry(state: AgentState) -> str:
    """
    After the critic validates code, decide next step:
      - execution error + retries remaining  → error_recovery
      - execution error + retries exhausted  → end
      - no error + screenshot present        → vision_critic
      - no error + no screenshot             → end
    """
    error_log = state.get("error_log", "")
    retry_count = state.get("retry_count", 0)

    if error_log and retry_count < MAX_RETRIES:
        logger.info("[routing] Error detected (retry %d/%d) → error_recovery", retry_count, MAX_RETRIES)
        return "error_recovery"

    if error_log:
        logger.warning("[routing] Max retries exceeded — returning code as-is")
        return "end"

    if state.get("screenshot_base64"):
        logger.info("[routing] Code approved, screenshot present → vision_critic")
        return "vision_critic"

    return "end"


def _vision_critic_route(state: AgentState) -> str:
    """
    After vision_critic runs, decide next step:
      - visual issue + retries remaining  → error_recovery
      - visual issue + retries exhausted  → end
      - approved                          → end
    """
    error_log = state.get("error_log", "")
    retry_count = state.get("retry_count", 0)

    if error_log and retry_count < MAX_RETRIES:
        logger.info("[routing] Visual issue (retry %d/%d) → error_recovery", retry_count, MAX_RETRIES)
        return "error_recovery"

    if error_log:
        logger.warning("[routing] Visual issue but max retries exceeded — ending")

    return "end"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Compilation
# ═══════════════════════════════════════════════════════════════════════════════

def _post_resolver_route(state: AgentState) -> str:
    """Stage 3 conditional edge after `target_resolver`.

    Route to `clarify` when the resolver flagged `needs_clarification=True`
    AND a clarify_callback is bound (i.e. there's a user to ask). When the
    callback is missing — e.g. headless evaluation or a transport error —
    we still fall through to `planner` so the pipeline completes with the
    resolver's best guess instead of deadlocking.
    """
    resolved = state.get("resolved_target") or {}
    if resolved.get("needs_clarification"):
        if state.get("clarify_callback") is not None:
            logger.info(
                "[routing] target_resolver needs_clarification=True → clarify"
            )
            return "clarify"
        logger.warning(
            "[routing] needs_clarification=True but no clarify_callback; "
            "falling through to planner with best guess. rationale=%r",
            (resolved.get("rationale") or "")[:200],
        )
    return "planner"


workflow = StateGraph(AgentState)

workflow.add_node("probing", probing_node)                    # Tier 4 — spatial measurements
workflow.add_node("target_resolver", target_resolver_node)    # Stage 2 — pick the target
workflow.add_node("clarify", clarify_node)                    # Stage 3 — ask the user
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("critic", critic_node)
workflow.add_node("error_recovery", error_recovery_node)
workflow.add_node("vision_critic", vision_critic_node)

# Graph topology:
#   probing → target_resolver → (clarify →)? planner → coder → critic
#                              → (error_recovery → coder) | (vision_critic) | END
workflow.set_entry_point("probing")
workflow.add_edge("probing", "target_resolver")
workflow.add_conditional_edges("target_resolver", _post_resolver_route, {
    "planner":  "planner",
    "clarify":  "clarify",
})
workflow.add_edge("clarify", "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "critic")

workflow.add_conditional_edges("critic", _should_retry, {
    "error_recovery": "error_recovery",
    "vision_critic": "vision_critic",
    "end": END,
})
workflow.add_edge("error_recovery", "coder")

workflow.add_conditional_edges("vision_critic", _vision_critic_route, {
    "error_recovery": "error_recovery",
    "end": END,
})

agent_graph = workflow.compile()
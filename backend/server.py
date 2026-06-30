"""
CADPilot Smart Core Backend

FastAPI server that handles WebSocket connections from Fusion 360 dumb clients,
routes messages to the LangGraph AI agent, and sends execution commands back.

Enhanced to support:
- Streaming plan/reasoning chunks to the client
- Dynamic API keys (BYOK) from the authenticate message
- Planning requests that send plan_complete for UI approval
- Error recovery loop: execution failures feed back into the agent
- Model selection per request
"""

import asyncio
import json
import logging
import uuid as _uuid_mod
from typing import Any, Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .agent import AgentState, agent_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="CADPilot Smart Core Backend")


# ═══════════════════════════════════════════════════════════════════════════════
# Session State — per-connection API keys and preferences
# ═══════════════════════════════════════════════════════════════════════════════

class SessionState:
    """Stores per-session data such as API keys and model preferences."""

    def __init__(self):
        self.api_keys: Dict[str, str] = {}
        self.model_name: str = "claude-opus-4-6"
        self.last_user_request: str = ""
        # Server-side retry counter. The Fusion client does NOT echo
        # retry_count back on execution_result payloads, so we must track it
        # here; otherwise every failure would restart the count and produce
        # unbounded retry loops.
        self.retry_count: int = 0
        # Cache of the last user request's full context. The Fusion client
        # only sends `entity_context`, `selection_context`, `spatial_context`,
        # and `timeline_context` on the initial `execute_request` / `planning_request`;
        # `execution_result` payloads carry NOTHING of the sort. Without this
        # cache, the error-recovery retry path used to run with an empty B-Rep
        # ("[LLM Agent] B-Rep context: 0 keys"), which made the agent forget
        # the existing geometry and generate a brand-new body instead of
        # modifying the original.
        self.last_brep_context: Dict[str, Any] = {}
        self.last_selection_context: Optional[Dict[str, Any]] = None
        self.last_spatial_context: Optional[Dict[str, Any]] = None
        self.last_timeline_context: Optional[List[Dict[str, Any]]] = None
        # Stage 1 / A4 — eyes-on planning: viewport snapshot + camera basis
        # captured at request time, cached for retries so the recovery loop
        # also sees the same view the planner used.
        self.last_viewport_screenshot: Optional[str] = None
        self.last_camera_frame: Optional[Dict[str, Any]] = None

    def update_api_keys(self, api_keys: Dict[str, str]):
        """Merge incoming API keys, preserving any not overridden."""
        if api_keys:
            self.api_keys.update({k: v for k, v in api_keys.items() if v})

    def update_model(self, model_name: str):
        if model_name:
            self.model_name = model_name


# session_id -> SessionState
_sessions: Dict[str, SessionState] = {}


def _get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Connection Manager
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages active WebSocket connections per session."""

    def __init__(self):
        # session_id -> websocket
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: session_id={session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        _sessions.pop(session_id, None)
        # Stage 3 / C1: free any clarification futures the agent was awaiting
        # so it stops blocking and exits cleanly.
        for fut in list(_pending_clarifications.pop(session_id, {}).values()):
            if not fut.done():
                fut.set_result({"cancelled": True, "error": "websocket disconnected"})
        # Stage 4 / D1: free any pending state-snapshot futures.
        for fut in list(_pending_state_snapshots.pop(session_id, {}).values()):
            if not fut.done():
                fut.set_result(None)
        logger.info(f"WebSocket disconnected: session_id={session_id}")

    async def send_json(self, session_id: str, payload: Dict[str, Any]) -> bool:
        """Send JSON message to a specific session. Returns True if successful."""
        if session_id not in self.active_connections:
            logger.warning(f"Cannot send to session {session_id}: not connected")
            return False

        websocket = self.active_connections[session_id]
        try:
            await websocket.send_json(payload)
            logger.info(f"Sent message to session {session_id}: {payload.get('type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {e}")
            return False


connection_manager = ConnectionManager()

# ═══════════════════════════════════════════════════════════════════════════════
# Pending Probe Futures — session_id → {probe_id → asyncio.Future}
# ═══════════════════════════════════════════════════════════════════════════════

_pending_probes: Dict[str, Dict[str, asyncio.Future]] = {}

# Stage 3 / C1 — Pending clarification round-trips. Mirror the probe
# pattern: agent suspends on a Future, the user clicks an option, the
# Future resolves with the locked-in handle.
_pending_clarifications: Dict[str, Dict[str, asyncio.Future]] = {}

# Per-session timeout for the user to answer the clarification widget.
# Long enough for genuine reading time, short enough that an abandoned UI
# doesn't tie up the agent forever. The brief targets <100 ms transport
# overhead — actual wait is dominated by user reading time.
_CLARIFICATION_TIMEOUT_S: float = 300.0

# Stage 4 / D1 — Pending fresh-state-snapshot requests. Same pattern as
# probes / clarifications: send a request_state_snapshot, await the
# matching state_snapshot reply.
_pending_state_snapshots: Dict[str, Dict[str, asyncio.Future]] = {}

# Brief specifies a 5 s wait before falling back to the cached snapshot.
_STATE_SNAPSHOT_TIMEOUT_S: float = 5.0

# Per-tool timeout budgets (seconds). Fusion main-thread operations get queued
# behind each other; the viewport-fit + multi-view capture done by
# `blueprint_views` alone can take tens of seconds on a cold document, and
# `svg_section` iterates all body edges + projects onto a plane. The old flat
# 20s budget expired before the client could even begin some of these.
_PROBE_TIMEOUTS_S: Dict[str, float] = {
    "bounding_box":       30.0,
    "min_distance":       30.0,
    "connected_faces":    30.0,
    "face_properties":    30.0,
    "edge_chain":         30.0,
    "body_volume":        30.0,
    "lookup_entity":      30.0,
    "face_by_description":    30.0,
    "find_feature_by_phrase": 30.0,
    # Creates + deletes a transient sketch on the face on the Fusion main
    # thread; on a busy document this can take several seconds.
    "face_sketch_bounds": 45.0,
    "svg_section":        60.0,
    "blueprint_views":    90.0,
}
_DEFAULT_PROBE_TIMEOUT_S = 30.0


def _make_probe_callback(session_id: str) -> Callable:
    """
    Return an async callable that sends a probe_request to the Fusion client
    and blocks until the matching probe_result arrives (or times out).

    The callback signature:  result = await callback(tool_name, params_dict)
    """
    async def callback(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        probe_id = str(_uuid_mod.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()

        _pending_probes.setdefault(session_id, {})[probe_id] = future

        # Route svg_section to its own message type so the client can handle it
        if tool_name == "svg_section":
            ws_msg = {
                "type": "svg_section_request",
                "probe_id": probe_id,
                "plane_axis": params.get("plane_axis", "XY"),
                "offset_mm": params.get("offset_mm", 0.0),
            }
        elif tool_name == "blueprint_views":
            ws_msg = {
                "type": "blueprint_request",
                "probe_id": probe_id,
                "views": params.get("views", ["top", "front", "right"]),
                "width": params.get("width", 1920),
                "height": params.get("height", 1080),
            }
        else:
            ws_msg = {
                "type": "probe_request",
                "probe_id": probe_id,
                "tool": tool_name,
                "params": params,
            }

        timeout_s = _PROBE_TIMEOUTS_S.get(tool_name, _DEFAULT_PROBE_TIMEOUT_S)

        try:
            sent = await connection_manager.send_json(session_id, ws_msg)
            if not sent:
                return {"error": "Failed to send probe request to Fusion client"}

            return await asyncio.wait_for(future, timeout=timeout_s)

        except asyncio.TimeoutError:
            logger.warning(
                "[probe] tool=%s timed out for session %s after %.0fs",
                tool_name, session_id, timeout_s,
            )
            return {"error": f"Probe '{tool_name}' timed out after {int(timeout_s)} s"}
        except Exception as exc:
            logger.error("[probe] callback error: %s", exc)
            return {"error": str(exc)}
        finally:
            _pending_probes.get(session_id, {}).pop(probe_id, None)

    return callback


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Callback — sends chunks to the client in real-time
# ═══════════════════════════════════════════════════════════════════════════════

def _make_stream_callback(session_id: str, chunk_type: str):
    """
    Create an async callback that sends text chunks to the Fusion client.

    The agent nodes call this for each LLM token so the palette UI can show
    real-time progress.
    """
    async def callback(msg_type: str, token: str):
        await connection_manager.send_json(session_id, {
            "type": msg_type if msg_type else chunk_type,
            "content": token,
        })
    return callback


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Agent (LangGraph)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_clarify_callback(session_id: str) -> Callable:
    """Stage 3 / C1 — async callback the agent's `clarify_node` awaits.

    Signature::
        result = await callback(question, options)

    where ``options`` is a list of ``{id, label, kind, preview_image_base64?}``
    dicts (the Stage-2 `alternatives`). Returns the user's response dict
    once `handle_clarification_response` resolves the future, or
    ``{"cancelled": True}`` on timeout / failure.

    Mirrors `_make_probe_callback`: outbound WebSocket message + asyncio
    Future keyed by `clarification_id`.
    """
    async def callback(
        question: str,
        options: List[Dict[str, Any]],
        *,
        rationale: str = "",
        timeout_s: float = _CLARIFICATION_TIMEOUT_S,
    ) -> Dict[str, Any]:
        clarification_id = str(_uuid_mod.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        _pending_clarifications.setdefault(session_id, {})[clarification_id] = future

        ws_msg = {
            "type": "clarification_request",
            "clarification_id": clarification_id,
            "question": question,
            "options": options,
            "rationale": rationale,
        }

        try:
            sent = await connection_manager.send_json(session_id, ws_msg)
            if not sent:
                return {"cancelled": True, "error": "Failed to send clarification_request"}

            return await asyncio.wait_for(future, timeout=timeout_s)

        except asyncio.TimeoutError:
            logger.warning(
                "[clarify] session=%s clarification_id=%s timed out after %.0fs",
                session_id, clarification_id, timeout_s,
            )
            return {"cancelled": True, "error": f"Clarification timed out after {int(timeout_s)} s"}
        except Exception as exc:
            logger.error("[clarify] callback error: %s", exc)
            return {"cancelled": True, "error": str(exc)}
        finally:
            _pending_clarifications.get(session_id, {}).pop(clarification_id, None)

    return callback


async def request_fresh_state_snapshot(
    session_id: str,
    *,
    timeout_s: float = _STATE_SNAPSHOT_TIMEOUT_S,
) -> Optional[Dict[str, Any]]:
    """Stage 4 / D1 — ask the add-in for a fresh entity_context + screenshot.

    Sends a `request_state_snapshot` over the WebSocket and awaits the
    matching `state_snapshot` reply. Returns ``None`` if the WS isn't open,
    the reply times out, or the client signalled failure — callers should
    fall back to the cached snapshot in that case (see brief: "If the
    snapshot times out, fall back to the cached one and log a warning.").
    """
    snapshot_id = str(_uuid_mod.uuid4())
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    _pending_state_snapshots.setdefault(session_id, {})[snapshot_id] = future

    ws_msg = {
        "type": "request_state_snapshot",
        "snapshot_id": snapshot_id,
    }

    try:
        sent = await connection_manager.send_json(session_id, ws_msg)
        if not sent:
            logger.warning(
                "[state_snapshot] failed to send request to session %s", session_id
            )
            return None
        return await asyncio.wait_for(future, timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning(
            "[state_snapshot] session=%s snapshot_id=%s timed out after %.1fs — "
            "falling back to cached state",
            session_id, snapshot_id, timeout_s,
        )
        return None
    except Exception as exc:
        logger.error("[state_snapshot] error: %s", exc)
        return None
    finally:
        _pending_state_snapshots.get(session_id, {}).pop(snapshot_id, None)


async def handle_state_snapshot(data: Dict[str, Any], session_id: str) -> None:
    """Resolve a pending state-snapshot Future with the add-in's reply."""
    snapshot_id = data.get("snapshot_id", "")
    success = data.get("success", True)

    pending = _pending_state_snapshots.get(session_id, {})
    future = pending.get(snapshot_id)

    if future is None or future.done():
        logger.debug(
            "[state_snapshot] late arrival for snapshot_id=%s session=%s",
            snapshot_id, session_id,
        )
        return

    if not success:
        future.set_result(None)
        return

    # The add-in echoes its current entity_context / timeline_context /
    # selection_context / spatial_context / viewport_screenshot_base64 /
    # camera_frame in the reply. Keep the dict shape the same as the
    # initial execute_request so the recovery path can treat them
    # uniformly.
    future.set_result({
        "entity_context":     data.get("entity_context") or {},
        "timeline_context":   data.get("timeline_context"),
        "selection_context":  data.get("selection_context"),
        "spatial_context":    data.get("spatial_context"),
        "viewport_screenshot_base64": data.get("viewport_screenshot_base64"),
        "camera_frame":       data.get("camera_frame"),
    })


async def handle_clarification_response(data: Dict[str, Any], session_id: str) -> None:
    """Resolve a pending clarification Future with the user's pick."""
    clarification_id = data.get("clarification_id", "")
    selected_option_id = data.get("selected_option_id") or data.get("option_id")
    cancelled = bool(data.get("cancelled", False))

    pending = _pending_clarifications.get(session_id, {})
    future = pending.get(clarification_id)

    if future is None or future.done():
        logger.debug(
            "[clarification_response] Late arrival for clarification_id=%s session=%s",
            clarification_id, session_id,
        )
        return

    payload = {
        "cancelled": cancelled,
        "selected_option_id": selected_option_id,
        # Pass through the entire user payload so the agent can read any
        # extra fields the widget chose to send (e.g. free-text override).
        "raw": dict(data),
    }
    future.set_result(payload)


async def handle_probe_result(data: Dict[str, Any], session_id: str) -> None:
    """Resolve a pending probe Future with the result returned by the Fusion client."""
    probe_id = data.get("probe_id", "")
    result = data.get("result") or {}
    success = data.get("success", True)
    error = data.get("error")

    pending = _pending_probes.get(session_id, {})
    future = pending.get(probe_id)

    if future is None or future.done():
        # A probe_result arriving after its future has been popped is normal:
        # it means the client legitimately did the work but the timeout fired
        # first (or the future was already resolved). Log at debug so cold
        # Fusion start-up doesn't spam WARN lines.
        logger.debug(
            "[probe_result] Late arrival for probe_id=%s session=%s (future gone/done)",
            probe_id, session_id,
        )
        return

    if success:
        future.set_result(result)
    else:
        future.set_result({"error": error or "Probe failed with no message"})


async def run_llm_agent(
    user_request: str,
    brep_context: Dict[str, Any],
    session_id: str,
    model_name: str = "claude-opus-4-6",
    api_keys: Optional[Dict[str, str]] = None,
    selection_context: Optional[Dict[str, Any]] = None,
    spatial_context: Optional[Dict[str, Any]] = None,
    timeline_context: Optional[List[Dict[str, Any]]] = None,
    probe_callback: Optional[Callable] = None,
    error_log: str = "",
    retry_count: int = 0,
    viewport_screenshot: Optional[str] = None,
    camera_frame: Optional[Dict[str, Any]] = None,
    clarify_callback: Optional[Callable] = None,
) -> str:
    """
    Run the LangGraph LLM agent to generate Fusion 360 Python code.

    Feeds the user request and B-Rep context into the state graph,
    then extracts the generated code from the final state.

    Args:
        user_request: Natural language CAD request from user
        brep_context: B-Rep data (bodies, faces, edges, vertices) in mm
        session_id: Session ID for streaming callbacks
        model_name: LLM model to use
        api_keys: Optional BYOK API keys
        selection_context: Currently selected entities
        spatial_context: Spatial relationships
        error_log: Previous execution error for retry
        retry_count: Number of previous retry attempts

    Returns:
        Python code string to execute in Fusion 360
    """
    logger.info(f"[LLM Agent] Processing request: {user_request[:100]}...")
    logger.info(f"[LLM Agent] B-Rep context: {len(brep_context)} keys")

    stream_cb = _make_stream_callback(session_id, "reasoning_chunk")

    # Build the initial state
    initial_state: AgentState = {
        "user_prompt": user_request,
        "brep_context": brep_context,
        "selection_context": selection_context,
        "spatial_context": spatial_context,
        "timeline_context": timeline_context,
        "probe_callback": probe_callback,
        "plan": "",
        "generated_code": "",
        "error_log": error_log,
        "retry_count": retry_count,
        "model_name": model_name,
        "api_keys": api_keys or {},
        "stream_callback": stream_cb,
        "viewport_screenshot": viewport_screenshot,
        "camera_frame": camera_frame,
        "clarify_callback": clarify_callback,
    }

    # Run the agent graph
    try:
        logger.info("[LLM Agent] Starting agent graph execution...")
        final_state = await agent_graph.ainvoke(initial_state)
        logger.info("[LLM Agent] Agent graph completed")

        # Extract generated code from final state
        generated_code = final_state.get("generated_code", "")
        error_from_state = final_state.get("error_log", "")

        if error_from_state:
            logger.warning(f"[LLM Agent] Errors encountered: {error_from_state}")

        logger.info(f"[LLM Agent] Generated code length: {len(generated_code)} characters")
        return generated_code
    except Exception as e:
        logger.error(f"[LLM Agent] Graph execution failed: {e}")
        raise


async def run_planning_agent(
    user_request: str,
    brep_context: Dict[str, Any],
    session_id: str,
    model_name: str = "claude-opus-4-6",
    api_keys: Optional[Dict[str, str]] = None,
    selection_context: Optional[Dict[str, Any]] = None,
    spatial_context: Optional[Dict[str, Any]] = None,
    viewport_screenshot: Optional[str] = None,
    camera_frame: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Run only the planner node to generate a step-by-step plan for user approval.

    Returns the plan text which is then sent as plan_complete to the client.
    """
    logger.info(f"[Planning Agent] Generating plan for: {user_request[:100]}...")

    stream_cb = _make_stream_callback(session_id, "plan_chunk")

    from .agent import planner_node, _make_llm, _fmt_brep, _fmt_selection, _fmt_spatial

    # We invoke just the planner node directly (not the whole graph)
    # to get the plan without generating code.
    state: AgentState = {
        "user_prompt": user_request,
        "brep_context": brep_context,
        "selection_context": selection_context,
        "spatial_context": spatial_context,
        "plan": "",
        "generated_code": "",
        "error_log": "",
        "retry_count": 0,
        "model_name": model_name,
        "api_keys": api_keys or {},
        "stream_callback": stream_cb,
        "viewport_screenshot": viewport_screenshot,
        "camera_frame": camera_frame,
    }

    result = await planner_node(state)
    plan = result.get("plan", "")
    logger.info(f"[Planning Agent] Plan generated ({len(plan)} chars)")
    return plan


# ═══════════════════════════════════════════════════════════════════════════════
# B-Rep Context Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_brep_context(entity_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract B-Rep context from entity_context.

    Args:
        entity_context: The entity_context dict from the incoming message

    Returns:
        Dict with bodies, faces, edges, vertices lists (all in mm coordinates)
    """
    brep_context = {}

    if "bodies" in entity_context:
        brep_context["bodies"] = entity_context["bodies"]

    if "faces" in entity_context:
        brep_context["faces"] = entity_context["faces"]

    if "edges" in entity_context:
        brep_context["edges"] = entity_context["edges"]

    if "vertices" in entity_context:
        brep_context["vertices"] = entity_context["vertices"]

    return brep_context


# ═══════════════════════════════════════════════════════════════════════════════
# Message Handlers
# ═══════════════════════════════════════════════════════════════════════════════

async def handle_planning_request(data: Dict[str, Any], session_id: str) -> None:
    """
    Handle a planning request — generate a step-by-step plan for user approval
    before executing any code.

    Sends plan_chunk messages during streaming, then a plan_complete message.
    """
    user_request = data.get("user_request", "")
    model_name = data.get("model_name", "claude-opus-4-6")

    logger.info(f"[planning_request] session_id={session_id}, model={model_name}")
    logger.info(f"  Request: {user_request}")

    session = _get_session(session_id)

    # Extract context
    spatial_context = data.get("spatial_context")
    entity_context = data.get("entity_context", {})
    selection_context = data.get("selection_context")
    timeline_context = data.get("timeline_context")
    brep_context = extract_brep_context(entity_context)
    viewport_screenshot = data.get("viewport_screenshot_base64")
    camera_frame = data.get("camera_frame")

    # Cache for a subsequent execute_request or any retry path that loses
    # geometry information.
    session.last_brep_context = brep_context
    session.last_selection_context = selection_context
    session.last_spatial_context = spatial_context
    session.last_timeline_context = timeline_context
    session.last_viewport_screenshot = viewport_screenshot
    session.last_camera_frame = camera_frame

    if spatial_context:
        logger.info(f"  Spatial context: {len(spatial_context.get('parallel_faces', []))} relationships")
    if entity_context:
        logger.info(f"  Entity context: {len(entity_context.get('bodies', []))} bodies")
    if viewport_screenshot:
        logger.info(f"  Viewport screenshot: {len(viewport_screenshot)} b64 chars")
    if camera_frame:
        logger.info(f"  Camera frame: eye={camera_frame.get('origin')} target={camera_frame.get('target')}")

    try:
        plan_text = await run_planning_agent(
            user_request=user_request,
            brep_context=brep_context,
            session_id=session_id,
            model_name=model_name,
            api_keys=session.api_keys,
            selection_context=selection_context,
            spatial_context=spatial_context,
            viewport_screenshot=viewport_screenshot,
            camera_frame=camera_frame,
        )

        # Send the complete plan for UI approval
        await connection_manager.send_json(session_id, {
            "type": "plan_complete",
            "full_plan": plan_text,
            "display_plan": plan_text,
            "display_plan_plain": plan_text,
            "requires_approval": True,
        })

    except Exception as e:
        logger.error(f"[planning_request] Planning failed: {e}")
        await connection_manager.send_json(session_id, {
            "type": "error",
            "message": f"Planning failed: {str(e)}",
        })


async def handle_execute_request(data: Dict[str, Any], session_id: str) -> None:
    """
    Handle an execute request — run the full agent pipeline and send
    generated Python code back to Fusion 360 for execution.
    """
    user_request = data.get("user_request", "")
    model_name = data.get("model_name", "claude-opus-4-6")

    logger.info(f"[execute_request] session_id={session_id}, model={model_name}")
    logger.info(f"  Request: {user_request}")

    session = _get_session(session_id)
    session.last_user_request = user_request
    # Each new user-initiated execute_request starts a fresh retry budget.
    session.retry_count = 0

    # Extract B-Rep context from entity_context
    entity_context = data.get("entity_context", {})
    brep_context = extract_brep_context(entity_context)

    # Extract other contexts
    selection_context = data.get("selection_context")
    spatial_context = data.get("spatial_context")
    timeline_context = data.get("timeline_context")
    viewport_screenshot = data.get("viewport_screenshot_base64")
    camera_frame = data.get("camera_frame")

    # Cache for the error-recovery path — `execution_result` payloads do NOT
    # re-send geometry, so without this the retry would run with empty B-Rep
    # and the agent would "forget" the existing body and create a new one.
    session.last_brep_context = brep_context
    session.last_selection_context = selection_context
    session.last_spatial_context = spatial_context
    session.last_timeline_context = timeline_context
    session.last_viewport_screenshot = viewport_screenshot
    session.last_camera_frame = camera_frame

    logger.info(f"[execute_request] B-Rep context: {len(brep_context)} keys")
    if viewport_screenshot:
        logger.info(f"[execute_request] Viewport screenshot: {len(viewport_screenshot)} b64 chars")
    if camera_frame:
        logger.info(f"[execute_request] Camera frame: eye={camera_frame.get('origin')} target={camera_frame.get('target')}")

    # Run the LLM agent to generate code
    probe_cb = _make_probe_callback(session_id)
    clarify_cb = _make_clarify_callback(session_id)
    try:
        python_code = await run_llm_agent(
            user_request=user_request,
            brep_context=brep_context,
            session_id=session_id,
            model_name=model_name,
            api_keys=session.api_keys,
            selection_context=selection_context,
            spatial_context=spatial_context,
            timeline_context=timeline_context,
            probe_callback=probe_cb,
            viewport_screenshot=viewport_screenshot,
            camera_frame=camera_frame,
            clarify_callback=clarify_cb,
        )
    except Exception as e:
        logger.error(f"[execute_request] Agent failed: {e}")
        await connection_manager.send_json(session_id, {
            "type": "error",
            "message": f"Agent execution failed: {str(e)}",
        })
        return

    # Send the code back to Fusion 360 for execution
    await connection_manager.send_json(session_id, {
        "type": "execute_code",
        "operation": "llm_generated",
        "code": python_code,
    })


async def handle_authenticate(data: Dict[str, Any], session_id: str) -> None:
    """
    Handle authentication message from client.
    Stores API keys (BYOK) and acknowledges auth success.
    """
    token = data.get("token")
    api_keys = data.get("api_keys", {})
    llm_api_keys = data.get("llm_api_keys", {})

    # Merge api_keys and llm_api_keys (backward compatibility)
    if not api_keys:
        api_keys = llm_api_keys

    logger.info(f"[authenticate] session_id={session_id}")
    if token:
        logger.info(f"  Token present (length: {len(token)})")
    if api_keys:
        configured = []
        if api_keys.get("anthropic_api_key"):
            configured.append("anthropic")
        if api_keys.get("openai_api_key"):
            configured.append("openai")
        if api_keys.get("google_api_key"):
            configured.append("google")
        logger.info(f"  Configured providers: {configured}")

    # Store API keys in session state
    session = _get_session(session_id)
    session.update_api_keys(api_keys)

    # Acknowledge authentication with API key status
    has_anthropic = bool(session.api_keys.get("anthropic_api_key"))
    has_openai = bool(session.api_keys.get("openai_api_key"))
    has_google = bool(session.api_keys.get("google_api_key"))

    await connection_manager.send_json(session_id, {
        "type": "authentication_ack",
        "authenticated": bool(token),
        "needs_api_keys": not (has_anthropic or has_openai or has_google),
        "has_anthropic": has_anthropic,
        "has_openai": has_openai,
        "has_google": has_google,
    })


async def handle_execution_result(data: Dict[str, Any], session_id: str) -> None:
    """
    Handle execution result from Fusion 360.

    If execution failed, feed the error back into the agent for self-correction.
    The agent's error recovery loop will regenerate code and send it back.
    """
    success = data.get("success", False)

    session = _get_session(session_id)

    if success:
        logger.info(f"[execution_result] SUCCESS in session {session_id}")
        # Reset the retry budget; the next failure should start fresh.
        session.retry_count = 0
        # Notify the client that execution completed
        await connection_manager.send_json(session_id, {
            "type": "completed",
            "message": "Execution completed successfully.",
        })
    else:
        error = data.get("error", "Unknown error")
        traceback_str = data.get("traceback", "")
        logger.error(f"[execution_result] FAILED: {error}")
        logger.error(f"[execution_result] Traceback: {traceback_str}")

        # Build the error log for the recovery loop
        error_log = f"Error: {error}\nTraceback:\n{traceback_str}" if traceback_str else f"Error: {error}"

        # Track retries server-side. The Fusion client does not echo
        # retry_count back in execution_result payloads, so reading it from
        # `data` would restart the counter on every failure and cause
        # unbounded retry loops.
        session.retry_count += 1
        retry_count = session.retry_count

        if retry_count <= 2:
            logger.info(f"[execution_result] Initiating retry {retry_count}/2")

            # Recover the original request + geometry context from the session.
            # The Fusion client's execution_result payload does NOT include
            # entity_context / selection_context / spatial_context / timeline_context,
            # so we MUST use the cached values from the initial execute_request;
            # otherwise the retry runs blind and the agent creates a new body
            # instead of modifying the existing one.
            user_request = data.get("user_request", session.last_user_request)
            if data.get("entity_context"):
                brep_context = extract_brep_context(data["entity_context"])
            else:
                brep_context = session.last_brep_context or {}
            selection_context = data.get("selection_context") or session.last_selection_context
            spatial_context = data.get("spatial_context") or session.last_spatial_context
            timeline_context = data.get("timeline_context") or session.last_timeline_context
            viewport_screenshot = data.get("viewport_screenshot_base64") or session.last_viewport_screenshot
            camera_frame = data.get("camera_frame") or session.last_camera_frame
            model_name = data.get("model_name", session.model_name)

            # Stage 4 / D1 — fresh state on retry. Step 1 of a script may
            # have CREATED/DELETED bodies; reusing the request-time snapshot
            # is what causes "create new body instead of modifying" silent
            # failures. Ask the add-in for a fresh snapshot, capped at 1
            # round-trip per retry. Falls back to the cached state on
            # timeout / failure (logged as a warning).
            try:
                snapshot = await request_fresh_state_snapshot(session_id)
            except Exception as exc:
                logger.warning("[execution_result] state_snapshot fetch raised: %s", exc)
                snapshot = None
            if snapshot:
                fresh_entity = snapshot.get("entity_context") or {}
                if fresh_entity:
                    brep_context = extract_brep_context(fresh_entity)
                    session.last_brep_context = brep_context
                if snapshot.get("timeline_context") is not None:
                    timeline_context = snapshot["timeline_context"]
                    session.last_timeline_context = timeline_context
                if snapshot.get("selection_context") is not None:
                    selection_context = snapshot["selection_context"]
                    session.last_selection_context = selection_context
                if snapshot.get("spatial_context") is not None:
                    spatial_context = snapshot["spatial_context"]
                    session.last_spatial_context = spatial_context
                if snapshot.get("viewport_screenshot_base64"):
                    viewport_screenshot = snapshot["viewport_screenshot_base64"]
                    session.last_viewport_screenshot = viewport_screenshot
                if snapshot.get("camera_frame"):
                    camera_frame = snapshot["camera_frame"]
                    session.last_camera_frame = camera_frame
                logger.info(
                    "[execution_result] Fresh state_snapshot applied — "
                    "%d bodies, viewport_screenshot=%s",
                    len((brep_context or {}).get("bodies") or []),
                    "yes" if viewport_screenshot else "no",
                )
            else:
                logger.warning(
                    "[execution_result] No fresh state_snapshot — "
                    "falling back to cached request-time state"
                )

            logger.info(
                f"[execution_result] Retry context — brep_keys={len(brep_context)}, "
                f"selection={'yes' if selection_context else 'no'}, "
                f"spatial={'yes' if spatial_context else 'no'}, "
                f"timeline={'yes' if timeline_context else 'no'}"
            )

            # Send a status message to the client
            await connection_manager.send_json(session_id, {
                "type": "llm_message",
                "message": f"Execution failed, retrying ({retry_count}/2)...\nError: {error[:200]}",
            })

            # Stage 4 / D1: bind probe + clarify callbacks on retry too,
            # so error_recovery_node can MEASURE before regenerating and
            # the resolver can still ask for clarification mid-recovery.
            retry_probe_cb = _make_probe_callback(session_id)
            retry_clarify_cb = _make_clarify_callback(session_id)

            try:
                python_code = await run_llm_agent(
                    user_request=user_request,
                    brep_context=brep_context,
                    session_id=session_id,
                    model_name=model_name,
                    api_keys=session.api_keys,
                    selection_context=selection_context,
                    spatial_context=spatial_context,
                    timeline_context=timeline_context,
                    probe_callback=retry_probe_cb,
                    error_log=error_log,
                    retry_count=retry_count,
                    viewport_screenshot=viewport_screenshot,
                    camera_frame=camera_frame,
                    clarify_callback=retry_clarify_cb,
                )

                # Send the corrected code with retry info
                await connection_manager.send_json(session_id, {
                    "type": "execute_code",
                    "operation": "llm_generated_retry",
                    "code": python_code,
                    "retry_count": retry_count,
                })
            except Exception as e:
                logger.error(f"[execution_result] Retry failed: {e}")
                await connection_manager.send_json(session_id, {
                    "type": "error",
                    "message": f"Execution failed and retry was unsuccessful: {str(e)}",
                })
        else:
            logger.warning(f"[execution_result] Max retries exceeded for session {session_id}")
            await connection_manager.send_json(session_id, {
                "type": "error",
                "message": f"Execution failed after {retry_count} attempts: {error}",
                "details": traceback_str,
            })


async def handle_message(data: Dict[str, Any], session_id: str) -> None:
    """Route incoming messages to appropriate handlers."""
    msg_type = data.get("type", "<unknown>")

    logger.info(f"[incoming] session_id={session_id}, type={msg_type}")

    # Log full payload for debugging (strip sensitive data)
    debug_data = dict(data)
    if "token" in debug_data:
        debug_data["token"] = "***REDACTED***"
    if "api_keys" in debug_data:
        debug_data["api_keys"] = "***REDACTED***"
    if "llm_api_keys" in debug_data:
        debug_data["llm_api_keys"] = "***REDACTED***"
    logger.debug(f"  Payload: {json.dumps(debug_data, indent=2)}")

    if msg_type == "planning_request":
        await handle_planning_request(data, session_id)
    elif msg_type == "execute_request":
        await handle_execute_request(data, session_id)
    elif msg_type == "authenticate":
        await handle_authenticate(data, session_id)
    elif msg_type == "execution_result":
        await handle_execution_result(data, session_id)
    elif msg_type == "probe_result":
        await handle_probe_result(data, session_id)
    elif msg_type == "clarification_response":
        await handle_clarification_response(data, session_id)
    elif msg_type == "state_snapshot":
        await handle_state_snapshot(data, session_id)
    elif msg_type == "cancel_request":
        logger.info(f"[cancel_request] User cancelled the operation in session {session_id}")
        # Resolve any pending clarification futures so the agent doesn't hang.
        for fut in list(_pending_clarifications.get(session_id, {}).values()):
            if not fut.done():
                fut.set_result({"cancelled": True, "error": "user cancelled"})
        # Same for any pending state-snapshot futures.
        for fut in list(_pending_state_snapshots.get(session_id, {}).values()):
            if not fut.done():
                fut.set_result(None)
    else:
        logger.warning(f"Unknown message type: {msg_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/")
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str = "default_session"):
    """
    WebSocket endpoint for CADPilot Fusion 360 clients.
    """
    await connection_manager.connect(session_id, websocket)

    try:
        while True:
            # Receive JSON message
            data = await websocket.receive_json()

            # Route to handler
            await handle_message(data, session_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session_id={session_id}")
        connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        connection_manager.disconnect(session_id)


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Endpoints (for health checks, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "service": "CADPilot Smart Core Backend",
        "active_sessions": len(connection_manager.active_connections),
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "name": "CADPilot Smart Core Backend",
        "version": "0.2.0",
        "endpoints": {
            "websocket": "/ws/{session_id}",
            "health": "/health",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("CADPilot Smart Core Backend Starting")
    logger.info("=" * 60)
    logger.info(f"Server: http://localhost:8000")
    logger.info(f"WebSocket: ws://localhost:8000/ws/{{session_id}}")
    logger.info(f"Health: http://localhost:8000/health")
    logger.info("=" * 60)

    uvicorn.run(
        "backend.server:app",
        host="localhost",
        port=8000,
        reload=False,  # Set to True during development
    )
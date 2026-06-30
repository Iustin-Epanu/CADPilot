# CADPilot

**An agentic AI CAD assistant for Autodesk Fusion 360.**

CADPilot replaces the click-driven Fusion 360 UI with a natural-language collaborator that can plan, sketch, extrude, fillet, pattern, and modify parametric geometry as fluently as a human draughtsperson — with the user staying in the loop via their own CAD viewport.

You type *"add a 5 mm counterbored hole through the top face and fillet the four vertical edges"*; CADPilot reads the live B-Rep model, plans the geometric operations, writes Fusion 360 Python API code, validates it, and executes it locally inside your design.

---

## How it works

CADPilot is two halves talking over a WebSocket:

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Fusion 360 Add-In (client) │  ◄────► │  LangGraph Backend (server)  │
│  CADPilot/                  │   WS    │  backend/                    │
│                             │         │                              │
│  • Extracts B-Rep geometry  │         │  • Multi-node agent graph    │
│  • Captures viewport / blue-│         │  • Plans operations          │
│    print screenshots        │         │  • Generates Fusion API code │
│  • Executes injected Python │         │  • Critiques & self-corrects │
│  • Read-only spatial probes │         │  • Streams reasoning to UI   │
└─────────────────────────────┘         └──────────────────────────────┘
```

1. **Fusion 360 Add-In (`CADPilot/`)** — extracts strictly-typed B-Rep data, intercepts user selections, captures orthographic blueprints, and executes Python injected dynamically by the backend. Talks to the server over a background-thread WebSocket client.
2. **LangGraph Agent (`backend/`)** — a FastAPI WebSocket router fronting a multi-node [LangGraph](https://langchain-ai.github.io/langgraph/) agent. It analyses the request, evaluates the 3D environment, plans, writes code, validates it, and recovers from execution errors.

The agent topology:

```
probing → planner → coder → critic → (error_recovery → coder) | vision_critic | END
```

---

## The "God Mode" 5-tier spatial awareness system

Plain B-Rep dumps and screenshots aren't enough for reliable parametric CAD. CADPilot grants the LLM spatial and geometric awareness through five complementary tiers:

| Tier | Name | What it provides |
|------|------|------------------|
| **1** | **Timeline DNA** | Parametric feature history (Extrudes, Patterns, Fillets) and their exact math, injected into the system prompt. |
| **2** | **SVG Cross-Sections** | Slices a body and returns the 2D profile as exact SVG `<path>` text — no perspective ambiguity. |
| **3** | **Strict Geometric Typing** | Upgraded B-Rep extraction that resolves analytical surfaces (`Cylinder(radius, origin, axis)`, …), all units in mm. |
| **4** | **Agentic Probing Tools** | Read-only "white cane" probes — `bounding_box`, `min_distance`, `connected_faces`, `edge_chain`, `body_volume`, and more — the agent calls to *look before it acts*. |
| **5** | **Orthographic Blueprinting** | Forces the viewport into distortion-free Top / Front / Right wireframe views for vision-grounded reasoning. |

---

## Repository layout

```
backend/
├── server.py        FastAPI WebSocket router, session state, probe plumbing
├── agent.py         LangGraph nodes, system prompts, graph compilation
└── requirements.txt

CADPilot/            The Fusion 360 add-in
├── CADAgent.py      Add-in entry point, message dispatch
├── code_executor.py Dynamic exec() wrapper for generated code
├── websocket_client.py
├── palette_manager.py     In-Fusion palette UI
├── body_tools.py / face_tools.py / edge_tools.py   B-Rep extraction
├── feature_tools.py       Tier 1 — Timeline DNA
├── svg_section.py         Tier 2 — SVG cross-sections
├── probe_tools.py         Tier 4 — read-only spatial probes
├── camera_tools.py        Tier 5 — orthographic blueprints
├── face_finder.py / modify_tools.py   Vetted geometry helpers
└── lib/             Vendored runtime deps (httpx, websockets, certifi)
```

---

## Getting started

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Fusion 360 Add-In

1. Copy the `CADPilot/` folder into your Fusion 360 add-ins directory:
   - **macOS**: `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/AddIns`
   - **Windows**: `%APPDATA%\Autodesk\Autodesk Fusion 360\API\AddIns`
2. In Fusion 360: **Tools → Add-Ins → Scripts and Add-Ins**, select **CADAgent**, click **Run**.
3. To point the add-in at your local backend before launching Fusion:
   ```bash
   export BACKEND_HOST=localhost
   export BACKEND_PORT=8000
   export BACKEND_USE_SSL=false
   ```
4. Type a request into the palette (e.g. *"Create a cylinder 5 cm tall with 2 cm diameter"*), optionally enable **Planning Mode** to review the plan, and **Execute**.

---

## Status

CADPilot works today and is under active development. Current work focuses on making the agent's perception of the model **lossless, stably-handled, and live on retry** — so it reliably modifies the body the user means instead of silently creating new geometry.

# Turingz — Hybrid PDE Solver demo (React + FastAPI)

Interactive demo for the viva. Backend (FastAPI) wraps the real trust module and ML models;
frontend (React + Vite + Tailwind) provides a clean UI with a live, interactive Trust Score page.

```
demo_app/
  backend/    FastAPI app (Python)  — wraps hybrid_pde/trust, the ML models, Cole-Hopf
  frontend/   React + Vite + Tailwind — sidebar shell + interactive Trust page
```

## Run the backend

Requires the project's Python environment (torch, deepxde, neuraloperator) so the ML models load.

```bash
cd demo_app/backend
pip install -r requirements.txt          # fastapi/uvicorn (torch etc. come from your project env)
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs to see the API. PINN runs with no torch (uses saved predictions);
FNO / DeepONet on a custom initial condition run the real models, so torch must be installed.

## Run the frontend

```bash
cd demo_app/frontend
npm install
npm run dev
```

Open http://localhost:5173. The frontend talks to the backend at `localhost:8000`
(change in `src/api.js` if needed).

## What the Trust page does

- Choose a model (FNO / DeepONet / PINN).
- FNO / DeepONet: build an initial condition with the sliders (more sine modes = sharper,
  out-of-distribution input). PINN: pick one of its trained waves.
- FNO also offers a "cheap-reference" mode.
- Press run: the wave animates, the trust gauge falls, and the switch fires — with a live
  breakdown of the physics signals and the cutoff/patience rule.

## Inputs and outputs (contract)

- Input to the backend: model + an initial condition (512-point wave, on x in [-1, 1)) or a PINN index.
- Live output, one message per time step:

```json
{ "t": 1.20, "u": [512 numbers], "true": [512 numbers], "trust": 0.48, "ok": false,
  "true_error": 0.11, "signals": {"residual": 0.14, "energy": 0.03, "roughness": 0.01},
  "switch_t": 1.20 }
```

## How teammates add their sections

Each module is one sidebar tab + one backend route, kept independent:

1. Backend: add a new router file (e.g. `member2.py`) and include it in `main.py`.
2. Frontend: add a page in `src/pages/`, then register it in the `SECTIONS` list in `src/App.jsx`.
3. Reuse the shared components in `src/components/` (Card, LineChart, Gauge) so every section looks the same.

The placeholder tabs (Numerical Solver, ML Reliability, Hybrid Engine, Module 2, Module 3)
show exactly where each part slots in.

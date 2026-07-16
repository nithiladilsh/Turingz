# Turingz - Hybrid PDE Prediction Engine ⚙️

## Project Overview
Turingz is a unified software pipeline designed to solve complex Partial Differential Equations (PDEs) by bridging the gap between high-speed Scientific Machine Learning (SciML) and highly accurate Classical Numerical methods. 

Pure machine learning models (like FNOs and PINNs) fail catastrophically during long-time extrapolation, while classical numerical solvers require intractable amounts of computational time and memory. Turingz solves this by actively orchestrating a hybrid approach: utilizing ML for speed, and dynamically triggering numerical solvers only when physical violations are detected and hardware budgets permit.

## System Architecture
This repository houses a microservice-oriented architecture divided into three core operational modules:

* **Module 1: Reliability & Failure Detection (The Sensors)**
    * Continuously monitors the ML predictions in the background. Tracks Mean Squared Error (MSE) and physics-based residual errors to generate a live "Failure Probability Score."
* **Module 2: Verified ML-to-Numerical Coupling (The Handoff)**
    * Provides the restart-verified handoff from learned PDE states to the numerical continuation solver (bit-for-bit equal to the production pseudo-spectral scheme), characterizes when the handoff helps (switch-time viability boundary) and when restart fidelity becomes safety-critical (low-viscosity stress boundary), and exposes the `M2Coupling` adapter that Module 3's runtime calls. Also includes the OOD/robustness evaluation used to locate ML failure.
* **Module 3: Deployment & Orchestration (The Engine Control Unit)**
    * A latency-aware middleware API. Ingests failure scores and calculates the **Hybrid Efficiency Index (HEI)** to make split-second routing decisions. It dynamically throttles the classical numerical engine to prevent Out-Of-Memory (OOM) crashes and guarantee strict IT latency budgets.

## Tech Stack
* **Core ML/Math:** PyTorch, NumPy, SciPy
* **Orchestration API:** Python, FastAPI, Uvicorn
* **Hardware Profiling:** `psutil`, `pynvml` (OS-level memory/thread tracking)
* **Deployment:** Docker (Containerization)

## Developer Setup
To ensure strict dependency alignment across the team, do not use global Python environments.

1. Clone the repository:
   ```bash
   git clone [https://github.com/nithiladilsh/Turingz.git](https://github.com/nithiladilsh/Turingz.git)
   cd Turingz

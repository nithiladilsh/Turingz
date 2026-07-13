from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from hybrid_pde.control_214133E.demo import default_standins, demo_frame

st.set_page_config(page_title="Turingz M3 Demo", layout="wide")
st.title("Turingz - Module 3: Cost-Aware Hybrid PDE Solver")
st.caption("One knob: pick an accuracy target; the hybrid spends the least numerical effort to hit it.")

parts = default_standins()
target = st.slider("Accuracy target (relative L2 error)", 0.01, 0.30, 0.10, 0.01)
f = demo_frame(target, **parts)
comp = f["comparison"]
methods = ["pure-ML", "pure-numerical", "hybrid"]
colors = ["#C0392B", "#1F4E79", "#1E8449"]
errs = [comp[m]["error"] for m in methods]
costs = [comp[m]["cost"] for m in methods]

saving = (1 - comp["hybrid"]["cost"] / comp["pure-numerical"]["cost"]) * 100
st.success(f"Hybrid reaches numerical-grade accuracy at ~{saving:.0f}% lower cost than the pure-numerical solver.")

st.subheader("Comparison: pure-ML vs pure-numerical vs hybrid")
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(methods, errs, color=colors)
    ax.set_ylabel("relative L2 error  (lower is better)"); ax.set_title("Accuracy")
    st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(methods, costs, color=colors)
    ax.set_ylabel("cost, units  (lower is better)"); ax.set_title("Cost")
    st.pyplot(fig)

st.subheader("What the solver did")
k = -1
c3, c4 = st.columns(2)
with c3:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(f["x"], f["truth"][k], label="truth (numerical)", lw=2.5, color="#1F4E79")
    ax.plot(f["x"], f["ml"][k], label="pure-ML", ls="--", lw=2, color="#C0392B")
    ax.plot(f["x"], f["hybrid"][k], label="hybrid", ls=":", lw=2.5, color="#1E8449")
    ax.set_title("Solution at final time"); ax.set_xlabel("x"); ax.legend(frameon=False)
    st.pyplot(fig)
with c4:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(f["t"], f["trust"], color="#7A5C00", lw=2)
    for s in f["switch"]:
        ax.axvline(s, color="#C9A227", lw=1.5)
    ax.axhline(0.5, color="#888", ls="--", lw=1)
    ax.set_title("Trust signal (gold = switch to numerical)"); ax.set_xlabel("t"); ax.set_ylabel("trust")
    st.pyplot(fig)

c = f["cost"]
st.subheader("Hybrid cost report")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Achieved error", f"{c.achieved_error:.4f}")
m2.metric("ML steps", c.ml_steps)
m3.metric("Correction steps", c.correction_steps)
m4.metric("Met target", "yes" if c.met_target else "no")

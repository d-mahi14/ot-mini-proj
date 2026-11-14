# --------------------------------------------------------------
# ui_demo.py – Streamlit UI for the VJTI Student‑Alumni matcher
# --------------------------------------------------------------
#   • Select instance (small / medium / large)
#   • Calibrate θ / δ  (reads/writes results/calibration.json)
#   • Run the multi‑seed benchmark (calls the existing scripts)
#   • Show tables, plots and download links
# --------------------------------------------------------------

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

CONFIG_MAP = {
    "Small (50 × 30)": "experiments/configs/small_50x30.yaml",
    "Medium (300 × 150)": "experiments/configs/medium_300x150.yaml",
    "Large (1000 × 500)": "experiments/configs/large_1000x500.yaml",
}

def run_cmd(command: str, description: str) -> None:
    """Run a shell command, showing a progress spinner."""
    with st.spinner(description):
        subprocess.check_call(command, shell=True, cwd=PROJECT_ROOT)

def figure_path(name: str) -> Path:
    return FIGURES_DIR / name

# ----------------------------------------------------------------------
# Streamlit layout
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Student‑Alumni Matching Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧑‍🎓 Student ↔ Alumni Matching – Live Demo UI")
st.markdown(
    """
    This UI wraps the full research pipeline:

    * **Calibration** – automatic derivation of the quality‑floor (θ) and branch‑parity (δ) thresholds.
    * **Benchmark** – 30‑seed (small) or 10‑seed (medium/large) runs of all baselines, meta‑heuristics and the Hybrid memetic NSGA‑II.
    * **Results** – tables, hyper‑volume box‑plots, Pareto surfaces, workload distributions, and downloadable artefacts.

    All heavy computations are already cached; the UI only re‑runs them when you explicitly press a button.
    """
)

# ----------------------------------------------------------------------
# Sidebar – instance selection
# ----------------------------------------------------------------------
st.sidebar.header("⚙️  Settings")
instance_label = st.sidebar.selectbox(
    "Choose instance size",
    options=list(CONFIG_MAP.keys()),
    index=0,
)
config_path = CONFIG_MAP[instance_label]
instance_name = Path(config_path).stem  # e.g. "small_50x30"

st.sidebar.markdown("---")
run_calibration = st.sidebar.button("🔧  Calibrate θ & δ")
run_benchmark   = st.sidebar.button("🚀  Run Benchmark (seeds)")

# ----------------------------------------------------------------------
# Main panel – actions & output
# ----------------------------------------------------------------------
# ---------- Calibration ----------
if run_calibration:
    try:
        run_cmd(
            f"python experiments/run_calibration.py --config {config_path}",
            "Calibrating thresholds …",
        )
        st.success("✅ Calibration finished – see the values below.")
    except subprocess.CalledProcessError as e:
        st.error(f"❗ Calibration failed: {e}")

cal_json_path = RESULTS_DIR / "calibration.json"
if cal_json_path.exists():
    with open(cal_json_path) as f:
        cal_data = json.load(f)
    st.subheader("🔎  Calibration results")
    st.json(cal_data.get(instance_name, {}))
else:
    st.info("⚙️  Calibration has not been run yet – press the button above.")

# ---------- Benchmark ----------
if run_benchmark:
    try:
        run_cmd(
            f"python experiments/run_experiments.py --config {config_path}",
            "Running full benchmark … (may take a minute)",
        )
        st.success("✅ Benchmark completed – tables & figures are now available.")
    except subprocess.CalledProcessError as e:
        st.error(f"❗ Benchmark failed: {e}")

# ---------- Summary table ----------
summary_file = RESULTS_DIR / f"{instance_name}_summary.csv"
if summary_file.exists():
    df_summary = pd.read_csv(summary_file)
    st.subheader(f"📊  Summary for **{instance_label}**")
    st.dataframe(
        df_summary.style.format({
            "hv_mean": "{:.3f}",
            "z1_mean": "{:.2f}",
            "z2_mean": "{:.3f}",
            "z3_mean": "{:.2f}",
        })
    )
    csv_bytes = summary_file.read_bytes()
    st.download_button(
        label="Download summary CSV",
        data=csv_bytes,
        file_name=summary_file.name,
        mime="text/csv",
    )
else:
    st.info("⚙️  Benchmark not run yet – press **Run Benchmark** above.")

# ----------------------------------------------------------------------
# Figures – show the most important plots
# ----------------------------------------------------------------------
st.subheader("🖼️  Visualisations")
fig_cols = st.columns(3)

# HV box‑plot
hv_file = figure_path(f"{instance_name}_hv_boxplot.png")
if hv_file.exists():
    with fig_cols[0]:
        st.image(str(hv_file), caption="Hyper‑volume box‑plot", use_column_width=True)
        with open(hv_file, "rb") as f:
            st.download_button(
                "⬇️ Download HV box‑plot",
                f,
                file_name=hv_file.name,
                mime="image/png",
            )
else:
    st.info("HV box‑plot not yet generated.")

# Pareto 3‑D surface
pareto_file = figure_path(f"{instance_name}_pareto_3d.png")
if pareto_file.exists():
    with fig_cols[1]:
        st.image(str(pareto_file), caption="Pareto front (3‑D)", use_column_width=True)
        with open(pareto_file, "rb") as f:
            st.download_button(
                "⬇️ Download Pareto 3‑D",
                f,
                file_name=pareto_file.name,
                mime="image/png",
            )
else:
    st.info("Pareto 3‑D plot not yet generated.")

# Workload distribution
workload_file = figure_path(f"{instance_name}_workload_distribution.png")
if workload_file.exists():
    with fig_cols[2]:
        st.image(str(workload_file), caption="Alumni workload distribution", use_column_width=True)
        with open(workload_file, "rb") as f:
            st.download_button(
                "⬇️ Download workload plot",
                f,
                file_name=workload_file.name,
                mime="image/png",
            )
else:
    st.info("Workload distribution plot not yet generated.")

# ----------------------------------------------------------------------
# Extra helpful plots (optional)
# ----------------------------------------------------------------------
extra = st.expander("🧩  Additional figures (optional)")
with extra:
    extra_figs = [
        "delta_sensitivity_tradeoff.png",
        "ablation_study.png",
        "coverage_gap.png",
        "scalability_curve.png",
        "sensitivity_analysis.png",
    ]
    for fname in extra_figs:
        p = figure_path(fname)
        if p.exists():
            st.image(str(p), caption=fname, use_column_width=True)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown("---")
st.caption(
    """
    📌  The UI is a thin wrapper around the existing scripts:
    * `experiments/run_calibration.py`
    * `experiments/run_experiments.py`
    * All results live under `results/` and `figures/`.

    The code lives in the repository, so you can edit the UI any time,
    commit the changes, and share the updated demo with collaborators.
    """
)


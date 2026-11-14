# --------------------------------------------------------------
# run_demo.py – a single‑command demo of the whole pipeline
# --------------------------------------------------------------
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>>> {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=PROJECT_ROOT)

def main():
    # 1️⃣ Generate (or load) the three benchmark instances
    for cfg in [
        "experiments/configs/small_50x30.yaml",
        "experiments/configs/medium_300x150.yaml",
        "experiments/configs/large_1000x500.yaml",
    ]:
        run(f"python experiments/run_calibration.py --config {cfg}")

    # 2️⃣ Run the full multi‑seed benchmark (30‑seed small, 10‑seed others)
    run("python experiments/run_experiments.py --config experiments/configs/small_50x30.yaml")
    run("python experiments/run_experiments.py --config experiments/configs/medium_300x150.yaml")
    run("python experiments/run_experiments.py --config experiments/configs/large_1000x500.yaml")

    # 3️⃣ Show a quick summary table on the console
    for csv in Path("results").glob("*_summary.csv"):
        print("\n===", csv.name, "===\n")
        print(Path(csv).read_text())

    # 4️⃣ Open the most important figures (HV boxplot & Pareto 3‑D)
    # (On Windows `start` opens the default image viewer)
    hv = Path("figures/small_50x30_hv_boxplot.png")
    pareto = Path("figures/small_50x30_pareto_3d.png")
    if hv.exists():
        subprocess.run(f'start "" "{hv}"', shell=True)
    if pareto.exists():
        subprocess.run(f'start "" "{pareto}"', shell=True)

    print("\n✅ Demo finished – all results are in the `results/` and `figures/` folders.")
    print("Open `report/final_report.html` for the complete PDF‑style report.\n")

if __name__ == "__main__":
    main()
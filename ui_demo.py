"""Interactive Decision-Support UI & Demonstration Dashboard for VJTI Matching.

Launches the interactive Web UI Portal or Streamlit dashboard to explore
the Pareto frontiers, multi-seed metrics, and student-alumni allocations.

Usage:
    python ui_demo.py                 # Launches built-in web portal (default)
    streamlit run ui_demo.py          # Launches Streamlit app (if installed)
"""

from __future__ import annotations

import sys
import os
import webbrowser
from pathlib import Path


def run_streamlit_app():
    """Run Streamlit dashboard if executed via 'streamlit run ui_demo.py'."""
    try:
        import streamlit as st
        import pandas as pd
        import json
    except ImportError:
        return False

    st.set_page_config(
        page_title="VJTI Student-Alumni Matching | UI Demo",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 VJTI Student-Alumni Multi-Objective Matching")
    st.markdown(
        "**Multi-Objective Capacitated Bipartite Assignment with Rawlsian Fairness & Departmental Parity**  \n"
        "*Veermata Jijabai Technological Institute (VJTI), Mumbai*"
    )

    # Top KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Quality (Z1)", "71.63", "98.6% of Peak")
    col2.metric("Rawlsian Fairness (Z2)", "1.09", "100% of Peak")
    col3.metric("Peak Mentor Load (Z3)", "6", "≤ Capacity Cap")
    col4.metric("Active Mentors", "28 / 30", "93.3% Mobilized")
    col5.metric("Constraint Violations", "0.00", "100% Feasible")

    st.markdown("---")

    # Tabs for Visuals, Roster, and Reports
    tab_figs, tab_roster, tab_docs = st.tabs(["📊 Visual Optimization Fronts", "👥 Knee Roster Allocation", "📑 Reports & Presentations"])

    with tab_figs:
        st.subheader("Pareto Trade-offs & Empirical Visualizations")
        fig_option = st.selectbox(
            "Select Figure to Inspect:",
            [
                "3D Pareto Frontier (small_50x30)",
                "2D Trade-off Projections (small_50x30)",
                "Departmental Quality Parity (Branch Equity)",
                "Mentor Workload Distribution",
                "Exact HiGHS MILP vs Metaheuristics",
                "Scalability Curve Across Cohorts",
                "Ablation Study Distributions",
                "Hypervolume Boxplots (30 Seeds)",
            ],
        )

        fig_map = {
            "3D Pareto Frontier (small_50x30)": "figures/small_50x30_pareto_3d.png",
            "2D Trade-off Projections (small_50x30)": "figures/small_50x30_tradeoffs_2d.png",
            "Departmental Quality Parity (Branch Equity)": "figures/small_50x30_branch_parity.png",
            "Mentor Workload Distribution": "figures/small_50x30_workload_distribution.png",
            "Exact HiGHS MILP vs Metaheuristics": "figures/exact_vs_metaheuristics.png",
            "Scalability Curve Across Cohorts": "figures/scalability_curve.png",
            "Ablation Study Distributions": "figures/ablation_study.png",
            "Hypervolume Boxplots (30 Seeds)": "figures/small_50x30_hv_boxplot.png",
        }

        fig_path = Path(fig_map[fig_option])
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        else:
            st.warning(f"Figure not found at {fig_path}")

    with tab_roster:
        st.subheader("Pseudonymous Knee Point Matching Schedule")
        csv_path = Path("results/matching_knee_small_50x30.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True)
            st.download_button(
                label="📥 Download Matching Roster (CSV)",
                data=csv_path.read_bytes(),
                file_name="matching_knee_small_50x30.csv",
                mime="text/csv",
            )
        else:
            st.info("Knee roster CSV not yet generated in results/.")

    with tab_docs:
        st.subheader("Project Deliverables & Compilations")
        st.markdown(
            "- **Interactive Presentation Slides (HTML)**: `presentation/presentation_slides.html`\n"
            "- **Full Academic Technical Report (HTML)**: `report/final_report.html`\n"
            "- **Web Portal & Hub**: `index.html`\n"
        )

    return True


def run_standalone_server():
    """Launch built-in HTTP server and open web portal."""
    import http.server
    import socketserver
    import threading

    port = 8000
    host = "127.0.0.1"

    print("================================================================================")
    print("      VJTI STUDENT-ALUMNI MULTI-OBJECTIVE MATCHING OPTIMIZATION")
    print("      Interactive Decision-Support UI Portal")
    print("================================================================================")
    print(f"\n[+] Serving interactive web UI portal at: http://{host}:{port}/")
    print("    - Portal Hub:             http://localhost:8000/")
    print("    - Presentation Slides:    http://localhost:8000/presentation/presentation_slides.html")
    print("    - Final Report:           http://localhost:8000/report/final_report.html")
    print("    - Matching Knee Roster:   http://localhost:8000/results/matching_knee_small_50x30.csv")
    print("\n[+] Press Ctrl+C to terminate the local server.\n")

    os.chdir(Path(__file__).resolve().parent)
    webbrowser.open(f"http://{host}:{port}/")

    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer((host, port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] Server stopped.")


def main():
    # Check if executed inside streamlit runtime
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_streamlit_app()
    else:
        # Check if first arg is streamlit or if streamlit run was requested
        try:
            import streamlit
            # If run directly and streamlit is installed, offer streamlit option
            if "--streamlit" in sys.argv:
                import subprocess
                subprocess.run(["streamlit", "run", __file__])
                return
        except ImportError:
            pass

        run_standalone_server()


if __name__ == "__main__":
    main()

"""PDF Generator for Final Report and Presentation Slides.

Converts Markdown documents into professional academic HTML and renders them
to publication-grade PDFs via headless Microsoft Edge Chromium engine.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import markdown


EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def markdown_to_html_report(md_content: str, base_dir: Path) -> str:
    """Convert final report markdown to styled printable A4 HTML."""
    # Convert image relative paths to absolute file URIs for Chromium rendering
    def replace_img(match):
        alt = match.group(1)
        rel_path = match.group(2)
        abs_path = (base_dir / rel_path).resolve().as_uri()
        return f'<div class="figure-container"><img src="{abs_path}" alt="{alt}" /><div class="figure-caption">{alt}</div></div>'

    processed_md = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_img, md_content)

    html_body = markdown.markdown(
        processed_md,
        extensions=["tables", "fenced_code", "toc", "nl2br", "sane_lists"],
    )

    # Inject visual figures into report sections where referenced
    pareto_img = (base_dir / "figures" / "small_50x30_pareto_3d.png").resolve().as_uri()
    tradeoff_img = (base_dir / "figures" / "small_50x30_tradeoffs_2d.png").resolve().as_uri()
    parity_img = (base_dir / "figures" / "small_50x30_branch_parity.png").resolve().as_uri()
    exact_img = (base_dir / "figures" / "exact_vs_metaheuristics.png").resolve().as_uri()
    ablation_img = (base_dir / "figures" / "ablation_study.png").resolve().as_uri()
    scale_img = (base_dir / "figures" / "scalability_curve.png").resolve().as_uri()

    # Enhance HTML body with inline figures
    fig_markup = f"""
    <div class="figure-grid">
        <div class="figure-card">
            <img src="{pareto_img}" alt="3D Pareto Front" />
            <p><strong>Figure 1:</strong> 3D Pareto Frontier (Total Quality $Z_1$, Worst Quality $Z_2$, Max Load $Z_3$)</p>
        </div>
        <div class="figure-card">
            <img src="{tradeoff_img}" alt="2D Trade-offs" />
            <p><strong>Figure 2:</strong> 2D Pairwise Projections of Objective Trade-offs</p>
        </div>
    </div>
    """
    html_body = html_body.replace("### 3.1 Small Instance ($50 \\times 30$) Benchmark", fig_markup + "### 3.1 Small Instance ($50 \\times 30$) Benchmark")

    fig_markup_analysis = f"""
    <div class="figure-grid">
        <div class="figure-card">
            <img src="{parity_img}" alt="Branch Parity" />
            <p><strong>Figure 3:</strong> Branch-wise Quality Parity across VJTI Departments</p>
        </div>
        <div class="figure-card">
            <img src="{exact_img}" alt="Exact vs Metaheuristic" />
            <p><strong>Figure 4:</strong> True MILP Pareto Frontier & Secondary Alumni Minimization</p>
        </div>
    </div>
    <div class="figure-grid">
        <div class="figure-card">
            <img src="{ablation_img}" alt="Price of Parity" />
            <p><strong>Figure 5:</strong> Component Ablation & Price of Parity Sensitivity ($\\delta$)</p>
        </div>
        <div class="figure-card">
            <img src="{scale_img}" alt="Scalability Curve" />
            <p><strong>Figure 6:</strong> Empirical Runtime Scalability Across Dimensions</p>
        </div>
    </div>
    """
    html_body = html_body.replace("## 4. Trade-Off Analysis & Policy Insights", fig_markup_analysis + "## 4. Trade-Off Analysis & Policy Insights")

    css = """
    @page {
        size: A4;
        margin: 18mm 16mm 20mm 16mm;
        @bottom-right {
            content: "Page " counter(page);
            font-size: 9pt;
            color: #666;
        }
        @top-right {
            content: "VJTI Optimization Techniques | Student-Alumni Matching";
            font-size: 8pt;
            color: #888;
        }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #222;
        line-height: 1.55;
        font-size: 10pt;
    }
    h1 {
        color: #0b3c6d;
        font-size: 20pt;
        border-bottom: 2px solid #0b3c6d;
        padding-bottom: 6px;
        margin-top: 0;
        margin-bottom: 8px;
    }
    h2 {
        color: #1a5276;
        font-size: 14pt;
        border-bottom: 1px solid #d0d7de;
        padding-bottom: 4px;
        margin-top: 18px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }
    h3 {
        color: #2874a6;
        font-size: 11pt;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }
    p {
        margin: 6px 0;
        text-align: justify;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 12px 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }
    th {
        background-color: #0b3c6d;
        color: #ffffff;
        font-weight: 600;
        padding: 6px 8px;
        border: 1px solid #0b3c6d;
        text-align: center;
    }
    td {
        padding: 5px 8px;
        border: 1px solid #d0d7de;
        text-align: center;
    }
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    code {
        background-color: #f3f4f6;
        padding: 2px 4px;
        border-radius: 4px;
        font-family: Consolas, monospace;
        font-size: 8.5pt;
    }
    pre {
        background-color: #f6f8fa;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #e1e4e8;
        font-size: 8pt;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    .figure-grid {
        display: flex;
        gap: 14px;
        margin: 14px 0;
        page-break-inside: avoid;
    }
    .figure-card {
        flex: 1;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 8px;
        background: #fafbfc;
        text-align: center;
    }
    .figure-card img {
        max-width: 100%;
        height: auto;
        max-height: 180px;
        object-fit: contain;
    }
    .figure-card p {
        font-size: 8pt;
        color: #444;
        margin: 4px 0 0 0;
        text-align: center;
    }
    .header-box {
        background: #f0f4f8;
        border-left: 4px solid #0b3c6d;
        padding: 10px 14px;
        margin-bottom: 16px;
        font-size: 9pt;
    }
    """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VJTI Student-Alumni Matching - Final Report</title>
    <style>{css}</style>
</head>
<body>
    <div class="header-box">
        <strong>VEERMATA JIJABAI TECHNOLOGICAL INSTITUTE (VJTI), MUMBAI</strong><br>
        Department of Production & Industrial Engineering / Computer Engineering<br>
        Course Project: Optimization Techniques | Academic Cohort Assignment
    </div>
    {html_body}
</body>
</html>"""


def markdown_to_html_slides(md_content: str, base_dir: Path) -> str:
    """Convert presentation slides markdown to styled 16:9 landscape presentation HTML."""
    # Split slides by ---
    raw_slides = [s.strip() for s in md_content.split("\n---") if s.strip()]

    slides_html = []
    for idx, slide_md in enumerate(raw_slides):
        slide_body = markdown.markdown(
            slide_md,
            extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
        )
        slides_html.append(f"""
        <div class="slide">
            <div class="slide-header">
                <span class="conf-title">VJTI Optimization Techniques &bull; Student-Alumni Matching</span>
                <span class="slide-num">Slide {idx+1} of {len(raw_slides)}</span>
            </div>
            <div class="slide-content">
                {slide_body}
            </div>
            <div class="slide-footer">
                Veermata Jijabai Technological Institute (VJTI), Mumbai &bull; Multi-Objective Capacitated Assignment
            </div>
        </div>
        """)

    css = """
    @page {
        size: A4 landscape;
        margin: 0;
    }
    body {
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: #eaedf1;
        -webkit-print-color-adjust: exact;
    }
    .slide {
        width: 297mm;
        height: 210mm;
        box-sizing: border-box;
        padding: 14mm 18mm 12mm 18mm;
        background: #ffffff;
        page-break-after: always;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
    }
    .slide-header {
        display: flex;
        justify-content: space-between;
        font-size: 10pt;
        color: #0b3c6d;
        font-weight: 600;
        border-bottom: 2px solid #0b3c6d;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .slide-footer {
        border-top: 1px solid #d0d7de;
        padding-top: 6px;
        font-size: 8.5pt;
        color: #666;
        text-align: center;
        margin-top: 8px;
    }
    .slide-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    h1 {
        color: #0b3c6d;
        font-size: 24pt;
        margin: 0 0 8px 0;
    }
    h2 {
        color: #1a5276;
        font-size: 18pt;
        margin: 0 0 10px 0;
    }
    h3 {
        color: #2874a6;
        font-size: 13pt;
        margin: 8px 0 4px 0;
    }
    p, li {
        font-size: 11pt;
        line-height: 1.5;
        color: #333;
    }
    ul {
        margin: 6px 0;
        padding-left: 24px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 9.5pt;
    }
    th {
        background-color: #0b3c6d;
        color: #ffffff;
        padding: 6px 10px;
        border: 1px solid #0b3c6d;
        text-align: center;
    }
    td {
        padding: 6px 10px;
        border: 1px solid #d0d7de;
        text-align: center;
    }
    tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    pre {
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 10px;
        font-family: Consolas, monospace;
        font-size: 8.5pt;
        line-height: 1.35;
        margin: 8px 0;
    }
    code {
        font-family: Consolas, monospace;
        background: #f0f3f6;
        padding: 2px 4px;
        border-radius: 3px;
    }
    """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>VJTI Presentation Slides</title>
    <style>{css}</style>
</head>
<body>
    {''.join(slides_html)}
</body>
</html>"""


def render_html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Use Microsoft Edge in headless mode to render HTML to PDF."""
    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path.resolve()}",
        str(html_path.resolve()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode == 0 and pdf_path.exists() and pdf_path.stat().st_size > 0


def main() -> None:
    """Generate both final report and presentation slides PDFs."""
    base_dir = Path(__file__).parent.resolve()

    # 1. Final Report PDF
    report_md_path = base_dir / "report" / "final_report.md"
    report_html_path = base_dir / "report" / "final_report.html"
    report_pdf_path = base_dir / "report" / "final_report.pdf"

    print("Reading report/final_report.md...")
    with open(report_md_path, "r", encoding="utf-8") as f:
        report_md = f.read()

    report_html = markdown_to_html_report(report_md, base_dir)
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Generated HTML report: {report_html_path}")

    print("Rendering report/final_report.pdf with Microsoft Edge...")
    success_report = render_html_to_pdf(report_html_path, report_pdf_path)
    if success_report:
        print(f"Successfully generated: {report_pdf_path} ({report_pdf_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"Failed to generate {report_pdf_path}")

    # 2. Presentation Slides PDF
    slides_md_path = base_dir / "presentation" / "presentation_slides.md"
    slides_html_path = base_dir / "presentation" / "presentation_slides.html"
    slides_pdf_path = base_dir / "presentation" / "presentation_slides.pdf"

    print("\nReading presentation/presentation_slides.md...")
    with open(slides_md_path, "r", encoding="utf-8") as f:
        slides_md = f.read()

    slides_html = markdown_to_html_slides(slides_md, base_dir)
    with open(slides_html_path, "w", encoding="utf-8") as f:
        f.write(slides_html)
    print(f"Generated HTML slides: {slides_html_path}")

    print("Rendering presentation/presentation_slides.pdf with Microsoft Edge...")
    success_slides = render_html_to_pdf(slides_html_path, slides_pdf_path)
    if success_slides:
        print(f"Successfully generated: {slides_pdf_path} ({slides_pdf_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"Failed to generate {slides_pdf_path}")


if __name__ == "__main__":
    main()

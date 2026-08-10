"""PDF report (TZ §11.5): summary + waveform + causal graph + conclusion + disclaimer.

Uses a Cyrillic-capable TrueType font (DejaVuSans, shipped with matplotlib) so
Russian and Uzbek text render correctly in the PDF.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from cardiolens.outputs.conclusion import format_conclusion
from cardiolens.outputs.graph_viz import render_causal_graph
from cardiolens.outputs.waveform import render_waveform
from cardiolens.pipeline import PipelineResult

_FONT = "DejaVuSans"


def _register_font() -> str:
    try:
        import matplotlib

        base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont(_FONT, str(base / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont(_FONT + "-Bold", str(base / "DejaVuSans-Bold.ttf")))
        return _FONT
    except Exception:
        return "Helvetica"  # fallback (Cyrillic may not render)


def write_pdf(result: PipelineResult, path: str) -> str:
    font = _register_font()
    tmp = Path(tempfile.mkdtemp(prefix="cardiolens_"))
    wave_png = str(tmp / "wave.png")
    graph_png = str(tmp / "graph.png")
    render_waveform(result, wave_png)
    render_causal_graph(result, graph_png)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName=font,
                          fontSize=8, leading=11)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName=font + "-Bold"
                        if font != "Helvetica" else "Helvetica-Bold", fontSize=14)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName=font + "-Bold"
                        if font != "Helvetica" else "Helvetica-Bold", fontSize=10)

    doc = SimpleDocTemplate(path, pagesize=A4, title="CardioLens ECG Report")
    flow = [Paragraph("CardioLens — заключение по ЭКГ", h1), Spacer(1, 0.3 * cm)]

    m = result.measurements
    meas = (f"ЧСС: {m.heart_rate_bpm} /мин · PR: {m.intervals.pr_ms} мс · "
            f"QRS: {m.intervals.qrs_ms} мс · QTc(B): {m.intervals.qtc_bazett_ms} мс · "
            f"ось QRS: {m.axes.qrs_deg}°")
    flow.append(Paragraph(meas, body))
    flow.append(Spacer(1, 0.3 * cm))

    flow.append(Paragraph("Кривая ЭКГ", h2))
    flow.append(Image(wave_png, width=17 * cm, height=17 * cm * 0.5))
    flow.append(Spacer(1, 0.2 * cm))

    flow.append(Paragraph("Причинно-следственный граф", h2))
    flow.append(Image(graph_png, width=15 * cm, height=15 * cm * 0.55))
    flow.append(Spacer(1, 0.2 * cm))

    flow.append(Paragraph("Заключение (RU)", h2))
    for line in format_conclusion(result, "ru").split("\n"):
        flow.append(Paragraph(_esc(line) or "&nbsp;", body))
    flow.append(Spacer(1, 0.3 * cm))

    flow.append(Paragraph("Xulosa (UZ)", h2))
    for line in format_conclusion(result, "uz").split("\n"):
        flow.append(Paragraph(_esc(line) or "&nbsp;", body))

    doc.build(flow)
    for f in (wave_png, graph_png):
        try:
            os.remove(f)
        except OSError:
            pass
    return path


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

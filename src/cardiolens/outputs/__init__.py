"""Output renderers — the five CardioLens output formats (TZ §11)."""

from cardiolens.outputs.conclusion import format_conclusion
from cardiolens.outputs.graph_viz import render_causal_graph
from cardiolens.outputs.json_out import result_to_dict, write_json
from cardiolens.outputs.pdf_report import write_pdf
from cardiolens.outputs.waveform import render_waveform


def write_all_outputs(result, out_dir: str, stem: str = "ecg") -> dict[str, str]:
    """Emit all five output formats (TZ §11) into ``out_dir``."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "json": write_json(result, os.path.join(out_dir, f"{stem}.json")),
        "waveform_svg": render_waveform(result, os.path.join(out_dir, f"{stem}_wave.svg")),
        "causal_graph_png": render_causal_graph(result, os.path.join(out_dir, f"{stem}_graph.png")),
        "pdf": write_pdf(result, os.path.join(out_dir, f"{stem}_report.pdf")),
    }
    # Bilingual conclusion text (TZ §11.2).
    conc = os.path.join(out_dir, f"{stem}_conclusion.txt")
    with open(conc, "w", encoding="utf-8") as fh:
        fh.write(format_conclusion(result, "ru"))
        fh.write("\n\n" + "=" * 60 + "\n\n")
        fh.write(format_conclusion(result, "uz"))
    paths["conclusion_txt"] = conc
    return paths


__all__ = [
    "format_conclusion",
    "render_causal_graph",
    "render_waveform",
    "result_to_dict",
    "write_all_outputs",
    "write_json",
    "write_pdf",
]

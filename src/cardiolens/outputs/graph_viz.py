"""Causal-graph visualisation (TZ §11.4).

Renders the Layer-5 causal graph: nodes coloured by physiologic/pathologic,
observed findings emphasised, edges labelled with the causal relation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from cardiolens.pipeline import PipelineResult

_COLORS = {"physiologic": "#2e7d32", "pathologic": "#b00020"}


def render_causal_graph(result: PipelineResult, path: str) -> str:
    g = result.causal_graph
    G = nx.DiGraph()
    for n in g.nodes:
        G.add_node(n.id, physiologic=n.physiologic, observed=n.observed,
                   label=n.label_ru)
    for e in g.edges:
        G.add_edge(e.from_, e.to, relation=e.relation)

    fig, ax = plt.subplots(figsize=(9, 5))
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Каузальных цепочек не выявлено",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    pos = nx.spring_layout(G, seed=7, k=1.2)
    node_colors = [_COLORS.get(G.nodes[n]["physiologic"], "#607d8b") for n in G.nodes]
    node_sizes = [1800 if G.nodes[n]["observed"] else 1000 for n in G.nodes]
    edgecolors = ["#000000" if G.nodes[n]["observed"] else "#ffffff" for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors=edgecolors, linewidths=1.5, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18,
                           edge_color="#455a64", width=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes},
                            font_size=7, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={(u, v): G.edges[u, v]["relation"] for u, v in G.edges},
        font_size=6, ax=ax,
    )
    ax.set_title("Причинно-следственный граф (зелёный=физиологично, красный=патологично)",
                 fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

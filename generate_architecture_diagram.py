import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUTPUT_PATH = Path("outputs") / "paper_results_section" / "architecture_diagram.png"

fig, ax = plt.subplots(figsize=(15, 10.4))
ax.set_xlim(0, 15)
ax.set_ylim(0.1, 11)
ax.axis("off")

COLORS = {
    "data": "#dbe9f6",
    "graph": "#fde9d9",
    "model": "#d9f2d9",
    "genai": "#f3d9f6",
    "validate": "#fde2e2",
    "output": "#fff2cc",
}
EDGE = "#444444"


def box(x, y, w, h, text, color, fontsize=9.5, bold=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.3, edgecolor=EDGE, facecolor=color, zorder=2,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, zorder=3, wrap=True)
    return (x, y, w, h)


def arrow(b1, b2, side1="right", side2="left", style="-|>", color=EDGE, lw=1.4, connectionstyle=None):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    points = {
        "right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
        "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1),
    }
    points2 = {
        "right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
        "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2),
    }
    p1 = points[side1]
    p2 = points2[side2]
    conn = connectionstyle or "arc3,rad=0.0"
    patch = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                             linewidth=lw, color=color, zorder=1, connectionstyle=conn)
    ax.add_patch(patch)


# ---- Title ----
ax.text(7.5, 10.6, "ProtLigGNN + GenAI + Drug-Response Pipeline — System Architecture",
        ha="center", va="center", fontsize=15, weight="bold")

# ---- Row 1: Data sources ----
d_pdbbind = box(0.3, 9.1, 2.6, 0.8, "PDBbind v2020\n(protein-ligand complexes)", COLORS["data"])
d_gdsc = box(3.2, 9.1, 2.6, 0.8, "GDSC2 + DepMap\n(drug + real gene expression\n→ IC50)", COLORS["data"])
d_drugcomb = box(6.1, 9.1, 2.6, 0.8, "DrugComb + DepMap\n(drug pair + real gene\nexpression → synergy)", COLORS["data"])
d_casf = box(9.0, 9.1, 2.6, 0.8, "CASF-2016 core set\n(285 held-out complexes)", COLORS["data"])
d_2l3r = box(11.9, 9.1, 2.6, 0.8, "2L3R target\n(GenAI / R.L protein)", COLORS["data"])

# ---- Row 2: Graph / feature construction ----
g_ligand = box(0.3, 7.6, 2.6, 0.8, "Ligand graph\n(atoms, bonds, RDKit)", COLORS["graph"])
g_pocket = box(3.2, 7.6, 2.6, 0.8, "Protein pocket graph\n(residues ≤6Å of ligand)", COLORS["graph"])
g_drug = box(6.1, 7.6, 2.6, 0.8, "Drug graph + real gene-\nexpression tokens (stacked\nautoencoder, 2000→128 dims)", COLORS["graph"])
g_pair = box(9.0, 7.6, 2.6, 0.8, "Drug-pair graphs + real gene-\nexpression tokens (same frozen\nautoencoder, reused)", COLORS["graph"])

arrow(d_pdbbind, g_ligand, "bottom", "top")
arrow(d_pdbbind, g_pocket, "bottom", "top")
arrow(d_gdsc, g_drug, "bottom", "top")
arrow(d_drugcomb, g_pair, "bottom", "top")

g_autoencoder = box(11.9, 7.6, 2.6, 0.8, "Stacked autoencoder\n(706 DepMap cell lines,\n2000 genes → 128-dim latent)", COLORS["graph"], fontsize=8.5)
arrow(g_autoencoder, g_drug, "left", "right", connectionstyle="arc3,rad=0.25")
arrow(g_autoencoder, g_pair, "left", "right", connectionstyle="arc3,rad=0.15")

# ---- Row 3: Core models ----
m_protlig = box(1.3, 6.0, 3.0, 0.9, "ProtLigGNN\n(GATConv ligand + GCNConv pocket,\nconcat vs. cross-attention ablation)\n→ predicted pKd", COLORS["model"], bold=True)
m_ic50 = box(5.0, 6.0, 2.9, 0.9, "GNAE Drugnet (IC50)\n(GAT/GCN drug encoder +\ngene tokens, concat vs.\nbidirectional cross-attention)\n→ predicted LN(IC50)", COLORS["model"], bold=True)
m_cdps = box(8.3, 6.0, 3.1, 0.9, "CDPS GNN (real transcriptomics)\n(shared LigandEncoder ×2 +\ngene tokens, concat vs.\nbidirectional cross-attention)\n→ predicted synergy (ZIP)", COLORS["model"], bold=True)

arrow(g_ligand, m_protlig, "bottom", "top")
arrow(g_pocket, m_protlig, "bottom", "top")
arrow(g_drug, m_ic50, "bottom", "top")
arrow(g_pair, m_cdps, "bottom", "top")

# ---- Row 4: GenAI + R.L loop (left) ----
genai_lstm = box(0.3, 4.4, 2.3, 0.8, "SMILES LSTM\n(trained on 12,223\nPDBbind ligand SMILES)", COLORS["genai"])
genai_gen = box(2.9, 4.4, 2.3, 0.8, "Generate + RDKit filter\n(10,000 sampled →\n4,097 valid unique)", COLORS["genai"])
genai_gaps = box(5.5, 4.4, 2.3, 0.8, "ProtLigGNN scoring +\nGAPS ranking", COLORS["genai"])
genai_rl = box(8.1, 4.4, 2.6, 0.8, "R.L fine-tuning\n(REINFORCE, reward =\nProtLigGNN pKd on 2L3R)", COLORS["genai"], bold=True)

arrow(genai_lstm, genai_gen, "right", "left")
arrow(genai_gen, genai_gaps, "right", "left")
arrow(m_protlig, genai_gaps, "bottom", "top", connectionstyle="arc3,rad=-0.15")
arrow(genai_lstm, genai_rl, "right", "left", connectionstyle="arc3,rad=0.35")
arrow(m_protlig, genai_rl, "bottom", "top", connectionstyle="arc3,rad=-0.3")
arrow(d_2l3r, genai_gaps, "bottom", "top", connectionstyle="arc3,rad=0.25")
arrow(d_2l3r, genai_rl, "bottom", "top", connectionstyle="arc3,rad=0.1")

# ---- Row 5: Validation ----
v_vina = box(0.3, 2.8, 3.4, 0.8, "AutoDock Vina docking\n(known top-10 + GenAI top-10\ninto 2L3R; independent physics check)", COLORS["validate"])
v_casf = box(4.0, 2.8, 3.6, 0.8, "CASF-2016 benchmark\nscoring / ranking / docking power\n(285 complexes, leakage-free)", COLORS["validate"])
v_split = box(7.9, 2.8, 2.9, 0.8, "Held-out 80/10/10\ntrain/val/test splits\n(all four model families)", COLORS["validate"])

arrow(genai_gaps, v_vina, "bottom", "top", connectionstyle="arc3,rad=-0.2")
arrow(m_protlig, v_casf, "bottom", "top", connectionstyle="arc3,rad=0.0")
arrow(m_ic50, v_split, "bottom", "top", connectionstyle="arc3,rad=0.15")
arrow(m_cdps, v_split, "bottom", "top", connectionstyle="arc3,rad=-0.15")

# ---- Row 6: Final output ----
out_box = box(3.0, 1.2, 6.0, 0.9, "Final Results & Paper\n(ablation tables, GAPS top candidates, R.L comparison,\nCASF-2016 metrics, graph visualizations)", COLORS["output"], bold=True)
arrow(v_vina, out_box, "bottom", "top", connectionstyle="arc3,rad=0.1")
arrow(v_casf, out_box, "bottom", "top")
arrow(v_split, out_box, "bottom", "top", connectionstyle="arc3,rad=-0.1")

# ---- Legend (manual row, explicit coordinates) ----
legend_items = [
    ("Data source", COLORS["data"]),
    ("Graph / feature construction", COLORS["graph"]),
    ("Core GNN model (ablated)", COLORS["model"]),
    ("GenAI / R.L generation loop", COLORS["genai"]),
    ("Validation", COLORS["validate"]),
    ("Final output", COLORS["output"]),
]
lx, ly = 0.3, 0.35
for label, color in legend_items:
    ax.add_patch(FancyBboxPatch((lx, ly - 0.08), 0.22, 0.22, boxstyle="round,pad=0.02",
                                 linewidth=1.0, edgecolor=EDGE, facecolor=color, zorder=2))
    ax.text(lx + 0.32, ly + 0.03, label, ha="left", va="center", fontsize=8.8, zorder=3)
    lx += 0.32 + len(label) * 0.072 + 0.35

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()
print(f"saved {OUTPUT_PATH}")

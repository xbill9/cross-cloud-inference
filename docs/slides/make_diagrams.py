"""Diagrams for the deck and the article, drawn in code so every label is exact.

    python3 docs/slides/make_diagrams.py     writes docs/slides/*.png, named by a hash of their bytes

overview   one Strands agent on the laptop, three cloud columns, one box per backend
tokps      client tok/s per backend, from results/comparison.json (computed by compare_clouds.py)
"""

import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(OUT))
INK, MUTED, BG = "#1E1E1E", "#6B6B6B", "#FFFFFF"
CLOUD = {"GCP": "#1E8E3E", "AWS": "#FF9900", "Azure": "#0078D4"}
TINT = {"GCP": "#E6F4EA", "AWS": "#FFF1DB", "Azure": "#E5F1FB"}
plt.rcParams["font.family"] = ["Roboto", "DejaVu Sans"]

COLUMNS = {
    "GCP": [("Gemini 3.5 Flash-Lite", "per token", "ok"), ("Gemma 4 26B, MaaS", "per token", "ok"),
            ("Vertex AI endpoint", "L4", "ok"), ("Cloud Run", "L4, scales to zero", "ok"),
            ("GKE Autopilot", "L4", "ok"), ("Compute Engine g2", "L4, Montreal", "ok"),
            ("Cloud TPU v5e-1", "v6e: no capacity", "ok")],
    "AWS": [("Bedrock, Nova Micro", "per token", "ok"), ("SageMaker endpoint", "L4", "ok"),
            ("EC2 g6", "L4", "ok"), ("EC2 g5g", "T4G + Graviton2", "ok"),
            ("EC2 inf2", "Inferentia2", "ok"), ("EC2 trn1", "Trainium", "ok")],
    "Azure": [("Foundry, gpt-4.1-nano", "per token", "ok"), ("Foundry, Phi-4-mini", "per token", "ok"),
              ("NV36ads_A10_v5", "A10", "ok"), ("Container Apps GPU", "quota 0", "blocked"),
              ("Azure ML endpoint", "quota 0", "blocked")],
}


def save(fig, stem):
    tmp = os.path.join(OUT, f"{stem}.png")
    fig.savefig(tmp, dpi=200, facecolor=BG)
    plt.close(fig)
    h = hashlib.sha256(open(tmp, "rb").read()).hexdigest()[:8]
    final = os.path.join(OUT, f"{stem}.{h}.png")
    os.replace(tmp, final)
    print("wrote", os.path.relpath(final, REPO))


def overview():
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 32); ax.set_ylim(0, 18); ax.axis("off")
    ax.add_patch(FancyBboxPatch((11.5, 15.6), 9, 1.9, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="#F4F4F4", ec=INK, lw=2.5))
    ax.text(16, 16.95, "One Strands agent", ha="center", va="center", fontsize=19, weight="bold", color=INK)
    ax.text(16, 16.1, "crosscloud.py: one grader, the same prompts", ha="center", va="center", fontsize=12.5, color=INK)
    xs = {"GCP": 1.6, "AWS": 12.6, "Azure": 23.6}
    for cloud, x in xs.items():
        cx = x + 3.4
        ax.add_patch(FancyArrowPatch((16, 15.5), (cx, 14.35), arrowstyle="-|>", mutation_scale=24,
                                     lw=2.5, color=CLOUD[cloud], connectionstyle="arc3,rad=0"))
        ax.text(cx, 13.75, cloud if cloud != "GCP" else "Google Cloud", ha="center", va="center",
                fontsize=21, weight="bold", color=CLOUD[cloud])
        for i, (name, sub, state) in enumerate(COLUMNS[cloud]):
            y = 11.6 - i * 1.85
            blocked = state == "blocked"
            ax.add_patch(FancyBboxPatch((x, y), 6.8, 1.5, boxstyle="round,pad=0.02,rounding_size=0.25",
                                        fc="#FFFFFF" if blocked else TINT[cloud],
                                        ec=MUTED if blocked else CLOUD[cloud], lw=2.2,
                                        ls="--" if blocked else "-"))
            ax.text(cx, y + 1.0, name, ha="center", va="center", fontsize=14, weight="bold",
                    color=MUTED if blocked else INK)
            ax.text(cx, y + 0.42, sub, ha="center", va="center", fontsize=12, color=MUTED if blocked else INK)
    save(fig, "overview")


def tokps():
    rows = json.load(open(os.path.join(REPO, "results", "comparison.json")))
    rows = sorted((r for r in rows if r.get("tok_s") and r["id"] != "gcp.maas"), key=lambda r: r["tok_s"])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.92, bottom=0.08)
    labels = [f"{r['cloud']}  {r['hardware'] if r['hardware'] != 'managed' else r['model']}" for r in rows]
    ax.barh(labels, [r["tok_s"] for r in rows], color=[CLOUD[r["cloud"]] for r in rows], height=0.65)
    for i, r in enumerate(rows):
        ax.text(r["tok_s"] + 2, i, f"{r['tok_s']:g}", va="center", fontsize=13, color=INK)
    ax.set_xlabel("tokens per second, one request, measured on the client (median)", fontsize=14, color=INK)
    ax.tick_params(axis="y", labelsize=13); ax.tick_params(axis="x", labelsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Same 100-token prompt, every backend", fontsize=20, weight="bold", color=INK, loc="left")
    save(fig, "tokps")


if __name__ == "__main__":
    overview()
    tokps()

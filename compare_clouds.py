"""Cross-cloud tables, computed from the three single-cloud repos' result files. All arithmetic is here.

    python3 compare_clouds.py > results/comparison.txt
    python3 compare_clouds.py --json results/comparison.json

Sources (each repo's own runs; nothing is re-measured here):
    AWS    ~/aws-inference-strands/results/2026-10-05-stage/{demo1-engine,demo1-rows,bench-chat}.json
           ~/aws-inference-strands/results/2026-10-12-cost/prices.json
    GCP    ~/gcp-inference/results/2026-10-12/{demo1-engine,demo1-rows,bench}-<backend>.json, prices.json
           (SKU rates, turned into USD/hr per backend by gcp-inference's cost.hourly())
    Azure  ~/azure-inference/results/2026-10-12/{demo1-engine,demo1-rows,bench-chat}.json, prices.json

Demo 1 answers are re-graded with one grader (azure-inference/demo1_counting.py, the same file as
gcp-inference's), so a run is correct under the same rule on every cloud.

USD per million output tokens for an hourly backend = USD per hour / (median tok/s * 3600) * 1e6:
one request at a time, busy for the whole hour. Per-token backends use their list output price.
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import statistics
import sys

HOME = os.path.expanduser("~")
AWS = f"{HOME}/aws-inference-strands/results"
GCP = f"{HOME}/gcp-inference/results/2026-10-12"
AZ = f"{HOME}/azure-inference/results/2026-10-12"
sys.path.insert(0, f"{HOME}/azure-inference")
import demo1_counting as d1  # noqa: E402

# id -> (cloud, path kind, hardware, model) : descriptive labels, from each repo's backends.py and stage/up.sh
WHAT = {
    "aws.bedrock": ("AWS", "per-token API", "managed", "Amazon Nova Micro"),
    "aws.sagemaker": ("AWS", "managed endpoint", "L4 (ml.g6.xlarge)", "Gemma 4 E2B repack"),
    "aws.g6": ("AWS", "your VM", "L4 (g6.xlarge)", "Gemma 4 E2B repack"),
    "aws.g5g": ("AWS", "your VM", "T4G + Graviton2 (g5g.2xlarge)", "Gemma 4 E2B fp16"),
    "aws.inf2": ("AWS", "your VM, AWS chip", "Inferentia2 (inf2.xlarge)", "Gemma 4 E2B, Neuron port"),
    "aws.trn1": ("AWS", "your VM, AWS chip", "Trainium (trn1.2xlarge)", "Gemma 4 E2B, Neuron port"),
    "gcp.gemini": ("GCP", "per-token API", "managed", "Gemini 3.5 Flash-Lite"),
    "gcp.maas": ("GCP", "per-token API, open model", "managed", "Gemma 4 26B"),
    "gcp.vertex": ("GCP", "managed endpoint", "L4 (Vertex AI endpoint)", "Gemma 4 E2B repack"),
    "gcp.cloudrun": ("GCP", "serverless GPU", "L4 (Cloud Run)", "Gemma 4 E2B repack"),
    "gcp.gke": ("GCP", "managed Kubernetes", "L4 (GKE Autopilot)", "Gemma 4 E2B repack"),
    "gcp.gce": ("GCP", "your VM", "L4 (g2-standard-4)", "Gemma 4 E2B repack"),
    "gcp.tpu": ("GCP", "your VM, Google chip", "TPU v5e-1 (v5litepod-1)", "Gemma 4 E2B bf16"),
    "azure.foundry": ("Azure", "per-token API", "managed", "gpt-4.1-nano"),
    "azure.phi": ("Azure", "per-token API, open model", "managed", "Phi-4-mini-instruct"),
    "azure.vm": ("Azure", "your VM", "A10 (NV36ads_A10_v5)", "Gemma 4 E2B repack"),
}


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def demo1_runs():
    """{(id, tools): [runs]} from every cloud, re-graded here."""
    out = {}
    for tools in ("engine", "rows"):
        for path, cloud in ((f"{AWS}/2026-10-05-stage/demo1-{tools}.json", "aws"),
                            (f"{AZ}/demo1-{tools}.json", "azure"), (f"{AZ}/demo1-{tools}-phi.json", "azure")):
            d = jload(path)
            for r in (d or {}).get("runs", []):
                out.setdefault((f"{cloud}.{r['backend']}", tools), []).append(r)
        for path in glob.glob(f"{GCP}/demo1-{tools}-*.json"):
            for r in jload(path)["runs"]:
                out.setdefault((f"gcp.{r['backend']}", tools), []).append(r)
    for runs in out.values():
        for r in runs:
            nums = [int(n) for n in re.findall(r"\b\d+\b", d1.THRESHOLD.sub(" ", r.get("answer") or ""))]
            r["regraded"] = bool(nums) and nums[-1] == len(d1.TRUTH) and not r.get("error")
    return out


def bench():
    """{id: summary row} : median tok/s and seconds of the same 100-token prompt."""
    out = {}
    d = jload(f"{AWS}/2026-10-05-stage/bench-chat.json")
    for s in (d or {}).get("summary", []):
        out[f"aws.{s['backend']}"] = s
    for path in (f"{AZ}/bench-chat.json", f"{AZ}/bench-phi.json"):
        for s in (jload(path) or {}).get("summary", []):
            out[f"azure.{s['backend']}"] = s
    for path in glob.glob(f"{GCP}/bench-*.json"):
        for s in jload(path)["summary"]:
            out[f"gcp.{s['backend']}"] = s
    return out


def prices():
    """{id: ('hour', usd) | ('token', usd per 1M output)}"""
    out = {}
    a = jload(f"{AWS}/2026-10-12-cost/prices.json") or {}
    for b, usd in a.get("usd_per_hour", {}).items():
        out[f"aws.{b}"] = ("hour", usd)
    if "bedrock_usd_per_1k_output" in a:
        out["aws.bedrock"] = ("token", a["bedrock_usd_per_1k_output"] * 1000)
    z = jload(f"{AZ}/prices.json") or {}
    for b, p in z.get("hourly", {}).items():
        out[f"azure.{b}"] = ("hour", p["usd"])
    for b, p in z.get("per_token", {}).items():
        out[f"azure.{b}"] = ("token", p["output"]["usd"] * 1000)
    g = jload(f"{GCP}/prices.json")
    if g:  # SKU rates; the per-backend hour is gcp-inference's own cost.hourly(), GKE node g2-standard-8
        spec = importlib.util.spec_from_file_location("gcp_cost", f"{HOME}/gcp-inference/cost.py")
        gcost = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gcost)
        for b, usd in gcost.hourly(g, "g2-standard-8").items():
            out[f"gcp.{b}"] = ("hour", round(usd, 4))
        out["gcp.gemini"] = ("token", g["rates"]["gemini_out"]["usd"] * 1e6)
        out["gcp.maas"] = ("token", g["rates"]["maas_out"]["usd"] * 1e6)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    runs, b, p = demo1_runs(), bench(), prices()
    ids = [i for i in WHAT if any(k[0] == i for k in runs) or i in b]
    rows = []
    for i in ids:
        cloud, kind, hw, model = WHAT[i]
        eng, raw = runs.get((i, "engine"), []), runs.get((i, "rows"), [])
        row = {"id": i, "cloud": cloud, "path": kind, "hardware": hw, "model": model,
               "engine_runs": len(eng), "engine_correct": sum(r["regraded"] for r in eng),
               "engine_right_filter": sum(bool(r.get("filter_ok")) for r in eng),
               "engine_median_s": round(statistics.median([r["seconds"] for r in eng]), 2) if eng else None,
               "rows_runs": len(raw), "rows_correct": sum(r["regraded"] for r in raw),
               "filters_sent": sorted({f for r in eng for f in r.get("filters", [])})}
        if i in b:
            row["tok_s"] = b[i]["median_tokens_per_second"]
            row["bench_median_s"] = b[i]["median_seconds"]
        if i in p:
            unit, usd = p[i]
            row["price"] = f"{usd} USD/hr" if unit == "hour" else f"{usd:.3f} USD/1M out"
            if unit == "hour" and row.get("tok_s"):
                row["usd_per_1m_out"] = round(usd / (row["tok_s"] * 3600) * 1e6, 2)
                row["usd_per_month_24x7"] = round(usd * 730)
            elif unit == "token":
                row["usd_per_1m_out"] = round(usd, 3)
        rows.append(row)

    print("Demo 1 (11 ids, how many are 10 or more? answer 8), one grader for every cloud\n")
    print(f"{'backend':<15} {'engine':>7} {'filter':>7} {'med s':>6} {'rows':>7}  filters sent")
    for r in rows:
        if r["engine_runs"] or r["rows_runs"]:
            e = f"{r['engine_correct']}/{r['engine_runs']}" if r["engine_runs"] else "-"
            f = f"{r['engine_right_filter']}/{r['engine_runs']}" if r["engine_runs"] else "-"
            w = f"{r['rows_correct']}/{r['rows_runs']}" if r["rows_runs"] else "-"
            m = r["engine_median_s"] if r["engine_median_s"] is not None else "-"
            print(f"{r['id']:<15} {e:>7} {f:>7} {m:>6} {w:>7}  {', '.join(r['filters_sent']) or '-'}")

    print("\nSame prompt, one request at a time (100-token cap), and list price\n")
    print(f"{'backend':<15} {'hardware':<30} {'tok/s':>7} {'price':>20} {'USD/1M out':>11} {'USD/mo 24x7':>12}")
    for r in sorted((r for r in rows if r.get("tok_s")), key=lambda r: -r["tok_s"]):
        print(f"{r['id']:<15} {r['hardware']:<30} {r['tok_s']:>7} {r.get('price', '-'):>20} "
              f"{r.get('usd_per_1m_out', '-'):>11} {r.get('usd_per_month_24x7', '-'):>12}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""One Strands agent, three clouds: every backend from the three single-cloud repos, side by side.

Each repo keeps its own backends.py; this file loads all three under their own names and
prefixes every backend with its cloud, so `aws.bedrock`, `gcp.gemini` and `azure.foundry`
sit in one list. Whatever each repo's environment configures is included (source the
three stage/demo.env files first); the three per-token APIs need no launch.

    python3 crosscloud.py list
    python3 crosscloud.py agent  --runs 5            # demo 1's question on every tool-capable backend
    python3 crosscloud.py ask    --orchestrator gcp.gemini   # one agent asks every backend, scoreboard in code
    python3 crosscloud.py bench  --repeats 5         # the same 100-token prompt everywhere, timed here

Grading, timing and ranking are done in code; the model calls the tools and quotes the result.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
from dataclasses import dataclass

HOME = os.path.expanduser("~")
REPOS = {"aws": f"{HOME}/aws-inference-strands", "gcp": f"{HOME}/gcp-inference", "azure": f"{HOME}/azure-inference"}
MANAGED = {"aws.bedrock", "gcp.gemini", "azure.foundry"}


def _load(cloud: str, path: str):
    """Import <path>/backends.py as <cloud>_backends, and its demo1_counting.py beside it."""
    sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location(f"{cloud}_backends", os.path.join(path, "backends.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backends"] = mod          # each repo's demo1_counting does `import backends`
    spec.loader.exec_module(mod)
    sys.path.remove(path)
    return mod


@dataclass
class Entry:
    cloud: str
    backend: object      # that repo's backends.Backend
    module: object       # that repo's backends module

    @property
    def id(self) -> str:
        return f"{self.cloud}.{self.backend.name}"

    @property
    def label(self) -> str:
        return f"{self.cloud.upper()} {self.backend.label}"

    def chat(self, prompt: str, max_tokens: int = 64) -> dict:
        r = self.module.chat_once(self.backend, prompt, max_tokens=max_tokens)
        r.update(backend=self.id, label=self.label)
        return r

    def model(self):
        return self.module.strands_model(self.backend)

    @property
    def tool_capable(self) -> bool:
        return bool(getattr(self.backend, "tool_capable", True))


def entries(only: str | None = None) -> list[Entry]:
    out = []
    for cloud, path in REPOS.items():
        if not os.path.exists(os.path.join(path, "backends.py")):
            continue
        mod = _load(cloud, path)
        try:
            found = mod.configured()
        except Exception as exc:  # a cloud that cannot even list its backends is reported, not fatal
            print(f"[{cloud}] not loaded: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        out += [Entry(cloud, b, mod) for b in found]
    if only:
        keep = {s.strip() for s in only.split(",")}
        out = [e for e in out if e.id in keep or e.cloud in keep or (("managed" in keep) and e.id in MANAGED)]
    return out


# ---------------------------------------------------------------- agent: demo 1 on every cloud

def cmd_agent(a) -> int:
    sys.path.insert(0, REPOS["azure"])     # the shared grader: same as gcp-inference/demo1_counting.py
    import demo1_counting as d1
    rows = []
    usable = [e for e in entries(a.only) if e.tool_capable]
    print(f"Question: {d1.QUESTION}\nIds: {d1.IDS}\nTrue answer: {len(d1.TRUTH)}  tools: {a.tools}\n")
    for e in usable:
        for n in range(a.runs):
            from strands import Agent
            calls: list = []
            agent = Agent(model=e.model(), system_prompt=d1.SYSTEM, tools=d1.make_tools(a.tools, calls),
                          callback_handler=None)
            import time
            t0 = time.perf_counter()
            try:
                answer, error = str(agent(d1.QUESTION)), None
            except Exception as exc:
                answer, error = "", f"{type(exc).__name__}: {exc}"[:300]
            r = d1.grade(answer, calls, a.tools)
            r.update(backend=e.id, seconds=round(time.perf_counter() - t0, 2), error=error, answer=answer.strip()[-200:])
            rows.append(r)
            mark = "ERR" if error else ("ok " if r["correct"] else "BAD")
            print(f"  {e.id:<16} run {n + 1}/{a.runs}  {mark}  quoted={r['quoted']}  filters={r['filters']}  {r['seconds']}s"
                  + (f"  {error}" if error else ""), flush=True)
    summary = d1.summarize(rows, a.tools)
    print("\nbackend           runs  correct  right-filter  errors  median s  filters sent")
    for s in summary:
        rf = "-" if s["filter_ok"] is None else f"{s['filter_ok']}/{s['runs']}"
        print(f"{s['backend']:<16} {s['runs']:>5}  {s['correct']:>3}/{s['runs']:<3}  {rf:>12}  {s['errors']:>6}  "
              f"{s['median_seconds']:>8}  {', '.join(s['filters_sent']) or '-'}")
    if a.json:
        json.dump({"question": d1.QUESTION, "ids": d1.IDS, "truth": len(d1.TRUTH), "tools": a.tools,
                   "summary": summary, "runs": rows}, open(a.json, "w"), indent=2)
    return 0


# ---------------------------------------------------------------- ask: one agent, every cloud as tools

SYSTEM = ("You coordinate model backends on three clouds. When asked, put the user's question to every "
          "backend using its ask_ tool, then call scoreboard and quote its table exactly. "
          "Do not compute, round or re-rank any number yourself.")


def cmd_ask(a) -> int:
    from strands import Agent, tool
    found = entries(a.only)
    orch = next((e for e in entries() if e.id == a.orchestrator), None)
    if orch is None:
        print(f"orchestrator {a.orchestrator} is not configured")
        return 1
    results: list = []

    def make_ask(e: Entry):
        def ask(prompt: str) -> dict:
            try:
                r = e.chat(prompt)
            except Exception as exc:
                r = {"backend": e.id, "label": e.label, "error": f"{type(exc).__name__}: {exc}"[:200]}
            results.append(r)
            return r
        ask.__name__ = "ask_" + e.id.replace(".", "_").replace("-", "_")
        ask.__doc__ = (f"Ask {e.label} ({e.backend.model}) one question and return its answer with "
                       f"measured seconds and tokens per second.\n\nArgs:\n    prompt: the question")
        return tool(ask)

    @tool
    def scoreboard() -> str:
        """Return the measured results of every ask_ call so far as a table, fastest first."""
        ok = sorted((r for r in results if "error" not in r), key=lambda r: r["seconds"])
        lines = ["| backend | answer | seconds | tokens/s |", "|---|---|---|---|"]
        lines += [f"| {r['label']} | {r['text'][:40]} | {r['seconds']} | {r['tokens_per_second']} |" for r in ok]
        lines += [f"| {r['label']} | ERROR {r['error'][:60]} | - | - |" for r in results if "error" in r]
        return "\n".join(lines)

    agent = Agent(model=orch.model(), system_prompt=SYSTEM, tools=[make_ask(e) for e in found] + [scoreboard],
                  callback_handler=None)
    print(f"Orchestrator: {orch.label} ({orch.backend.model})\nBackends as tools: {', '.join(e.id for e in found)}\n")
    print(str(agent(f"Ask every backend this question and show the scoreboard: {a.question}")).strip())
    asked = {r["backend"] for r in results}
    missed = [e.id for e in found if e.id not in asked]
    print(f"\n[harness] {len(results)} backend calls made; not asked by the agent: {', '.join(missed) or 'none'}")
    if a.json:
        json.dump({"question": a.question, "orchestrator": orch.id, "results": results}, open(a.json, "w"), indent=2)
    return 0


# ---------------------------------------------------------------- bench: same prompt everywhere

PROMPT = "Write a short paragraph about the ocean."


def cmd_bench(a) -> int:
    rows = []
    for e in entries(a.only):
        try:
            e.chat("Hi", max_tokens=4)          # warm the path; not recorded
            for _ in range(a.repeats):
                rows.append(e.chat(PROMPT, max_tokens=a.max_tokens))
        except Exception as exc:
            print(f"  {e.id}: {type(exc).__name__}: {exc}"[:200])
    print(f"Prompt: {PROMPT!r}, max_tokens={a.max_tokens}, {a.repeats} repeats per backend\n")
    print("backend           median s  median tok/s  tok/s range     tokens")
    summary = []
    for name in dict.fromkeys(r["backend"] for r in rows):
        mine = [r for r in rows if r["backend"] == name]
        tps, secs, toks = ([r[k] for r in mine] for k in ("tokens_per_second", "seconds", "completion_tokens"))
        s = {"backend": name, "label": mine[0]["label"], "median_seconds": round(statistics.median(secs), 3),
             "median_tokens_per_second": round(statistics.median(tps), 1), "min_tps": min(tps), "max_tps": max(tps),
             "tokens": f"{min(toks)}-{max(toks)}"}
        summary.append(s)
        print(f"{name:<16} {s['median_seconds']:>9}  {s['median_tokens_per_second']:>12}  "
              f"{s['min_tps']:>6}-{s['max_tps']:<7}  {s['tokens']}")
    if a.json:
        json.dump({"prompt": PROMPT, "max_tokens": a.max_tokens, "summary": summary, "runs": rows},
                  open(a.json, "w"), indent=2)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("agent"); p.add_argument("--tools", choices=["engine", "rows"], default="engine")
    p.add_argument("--runs", type=int, default=5)
    p = sub.add_parser("ask"); p.add_argument("--orchestrator", default="aws.bedrock")
    p.add_argument("--question", default="What is the capital of France? Answer in one word.")
    p = sub.add_parser("bench"); p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=100)
    for p in sub.choices.values():
        p.add_argument("--only", help="comma list of backend ids, clouds, or 'managed'")
        p.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "list":
        for e in entries(a.only):
            print(f"{e.id:<16} {e.label:<45} {e.backend.model}  tools={'yes' if e.tool_capable else 'no'}")
        return 0
    return {"agent": cmd_agent, "ask": cmd_ask, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

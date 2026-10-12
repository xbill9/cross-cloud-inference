# One Strands agent, three clouds

LLM inference on Google Cloud, AWS and Azure, compared with the same agent, grader, prompts and Gemma 4 E2B weights. Each cloud has its own repository with its backends, bring-up scripts and results; this one loads all three and holds the comparison.

| Repository | Backends |
|---|---|
| [gcp-inference](https://github.com/xbill9/gcp-inference) | Gemini, Gemma 4 26B on Vertex AI MaaS, Vertex AI endpoint, Cloud Run, GKE Autopilot, Compute Engine g2 (L4), Cloud TPU v6e |
| [aws-inference-strands](https://github.com/xbill9/aws-inference-strands) | Bedrock, SageMaker, EC2 g6 (L4), g5g (T4G), inf2, trn1 |
| [azure-inference](https://github.com/xbill9/azure-inference) | Foundry (gpt-4.1-nano, Phi-4-mini), NV36ads_A10_v5 (A10); Container Apps GPU blocked by quota |

Clone the three beside this one in `~`.

| File | What it does |
|---|---|
| `crosscloud.py` | Loads each repo's `backends.py` and runs one agent across every configured backend: `list`, `agent` (demo 1), `ask` (one agent asks every cloud, scoreboard in code), `bench` |
| `compare_clouds.py` | The cross-cloud tables, computed from the three repos' result files; demo 1 re-graded with one grader |
| `results/` | `comparison.txt` and the 2026-10-12 cross-cloud runs |
| `docs/article/` | The write-up |

```bash
source ~/azure-inference/stage/demo.env        # and the AWS and GCP demo.env for their self-hosted backends
python3 crosscloud.py list
python3 crosscloud.py agent --runs 5 --only managed
python3 crosscloud.py ask --orchestrator gcp.gemini
python3 compare_clouds.py
```

The three per-token APIs (Bedrock, Gemini, Foundry) need no launch, so `--only managed` runs with nothing billing by the hour.

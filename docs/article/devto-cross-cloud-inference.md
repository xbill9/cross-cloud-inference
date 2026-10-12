---
title: "LLM Inference on GCP, AWS and Azure: One Strands Agent, Three Clouds, the Same Prompts"
published: false
description: "A step by step comparison of serving a model on Google Cloud, AWS and Azure, from per-token APIs to your own GPU, driven by one Strands agent with the same grader and prompts on every cloud."
tags: ai, machinelearning, cloud, gemma
cover_image: https://raw.githubusercontent.com/xbill9/cross-cloud-inference/main/docs/article/devto-cover.248ace7f.jpg
---

This article provides a step by step comparison of LLM inference on Google Cloud, AWS and Azure. One Strands agent drives every backend on all three clouds: the per-token APIs, managed endpoints, serverless and Kubernetes GPUs, VMs you run yourself, and the clouds' own accelerator chips. The same grader, the same prompts and the same Gemma 4 E2B weights run everywhere a GPU is involved.

https://github.com/xbill9/cross-cloud-inference

---

#### What Is the Problem?

Every cloud sells a model four ways: as an API call billed per token, as a managed endpoint, as a GPU you rent by the hour, and as its own accelerator chip. Comparisons usually stay inside one cloud, or compare list prices without running anything.

This comparison runs all of them. Three repositories, one per cloud, share the same demo scripts, grader and prompts; only each cloud's `backends.py` changes. A fourth repository loads all three and puts every backend behind one agent.

---

#### Which Backends?

| Cloud | Per-token API | Managed endpoint / serverless | Your own VM |
|---|---|---|---|
| Google Cloud | Gemini 3.5 Flash-Lite; Gemma 4 26B (Vertex AI MaaS) | Vertex AI endpoint, Cloud Run, GKE Autopilot (L4) | Compute Engine g2 (L4); Cloud TPU v5e-1 |
| AWS | Amazon Nova Micro (Bedrock) | SageMaker endpoint (L4) | EC2 g6 (L4), g5g (T4G), inf2, trn1 |
| Azure | gpt-4.1-nano; Phi-4-mini (Foundry) | Container Apps serverless GPU: quota 0 | NV36ads_A10_v5 (A10) |

Every GPU row serves `xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4` with vLLM 0.30.0 and `--enable-auto-tool-choice --tool-call-parser gemma4`, except the AWS g5g (stock E2B in fp16 on a patched vLLM), the AWS Neuron chips (a hand-ported E2B build) and the Cloud TPU (stock E2B in bf16 on `vllm/vllm-tpu:v0.30.0`). Each per-token API serves its own cloud's model.

---

#### At This Point You Should Have…

- Accounts on all three clouds, with `gcloud`, `aws` and `az` signed in.
- Python 3 with `pip install 'strands-agents[openai,gemini,sagemaker]' boto3 google-genai`.
- The three cloud repositories cloned beside each other: [gcp-inference](https://github.com/xbill9/gcp-inference), [aws-inference-strands](https://github.com/xbill9/aws-inference-strands) and [azure-inference](https://github.com/xbill9/azure-inference).

---

#### Step 1 — Check Quota, Then Check Capacity, on Every Cloud

Each cloud refused something on the day, and each refused it differently.

On Google Cloud the project holds quota for 16 L4 GPUs in every US region. Compute Engine still reported `STOCKOUT` in 15 zones across us-east4, us-central1, us-east1, us-west1 and us-west4, and the L4 VM launched in `northamerica-northeast1-b`:

```
18:19:52Z gce: no capacity in any of: us-east4-a us-east4-c us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d
18:22:51Z gce: no capacity in us-west4-c, trying the next zone
18:23:03Z gce: launched in northamerica-northeast1-b
```

The TPU v6e was refused in every zone the project can use, on-demand and Spot, and a queued flex-start request never got capacity. A v5e-1 came up through flex-start in us-west4-a.

On AWS, inf2 was refused in five zones across us-east-2 and us-east-1 and launched in us-west-2a.

On Azure the limit is quota, before capacity is ever asked. The subscription holds 36 vCPUs of the NVadsA10 v5 family in eastus and nothing in the T4 or A100 families. Container Apps refused the serverless GPU outright:

```
ContainerAppInvalidScaleExceedsQuota: minReplicas exceeds the maxium quota value of '0' for: 'ManagedEnvironmentConsumptionT4Gpus'.
```

```
ManagedEnvironmentConsumptionNCA100Gpus	0	0
ManagedEnvironmentConsumptionT4Gpus	0	0
```

Azure Machine Learning's managed endpoints showed GPU quota only for the retired NC and NV families, so Azure ran its per-token models and one A10 VM.

---

#### Step 2 — The Per-Token APIs Need Nothing Started

Bedrock, Gemini on Vertex AI and Azure AI Foundry answer from the first request. Each cloud's flagship small model drives the agent through its own Strands provider: `BedrockModel`, `GeminiModel`, and `OpenAIModel` pointed at Foundry's OpenAI-compatible endpoint.

```python
OpenAIModel(client_args={"base_url": "https://<account>.services.ai.azure.com/openai/v1",
                         "api_key": key}, model_id="cci-gpt41-nano")
```

Both clouds' open models on per-token billing behaved differently from their flagships. Gemma 4 26B on Vertex AI MaaS took a median 13.23 s per agent run. Phi-4-mini on Foundry answered 12 of 20 plain requests and timed out at 60 s on the other 8, and its streamed responses carry no tool calls:

```
{"choices":[{"index":0,"delta":{"role":"assistant","content":""},...}]}
{"choices":[],"usage":{"prompt_tokens":340,"total_tokens":367,"completion_tokens":27}}
data: [DONE]
```

The same request without streaming returns the tool call. Strands streams, so Phi-4-mini cannot drive the agent through Foundry.

---

#### Step 3 — Your Own GPU on Each Cloud

The same container runs on all three: `vllm/vllm-openai:v0.30.0`, the same digest on the Azure VM and the GCP VM (`sha256:8a69ffad…`), with the repack and the tool-call flags.

| Cloud | Instance | GPU | Launch to serving |
|---|---|---|---|
| AWS | g6.xlarge | L4 | 13.9 min to the first correct answer |
| GCP | g2-standard-4 | L4 | 9.65 min to a passing health check |
| Azure | Standard_NV36ads_A10_v5 | A10 (vGPU, `A10-24Q`) | 9.65 min to a passing health check |

The Azure A10 sizes are vGPU-backed and need NVIDIA's GRID driver, which Azure publishes; `stage/vm-setup.sh` installs it at first boot (driver 595.91.07), then the container toolkit and vLLM. The driver install took under two minutes:

```
2026-10-12T18:37:17Z GRID driver installed
2026-10-12T18:41:24Z vllm container started
```

The A10 holds 24 GB against the L4's 23 GB, and decodes faster. On the same repack, vLLM's decode rate on the A10 was 194.2 tok/s against 141.7 on an L4, measured with the same `compare.py` on the instance; the GCP L4s report 139.9 to 143.1 server-side.

---

#### Step 4 — Managed Endpoints, Serverless and Kubernetes

AWS and Google Cloud both run the same vLLM container behind a managed endpoint: SageMaker on an `ml.g6.xlarge`, and a Vertex AI endpoint reached through `:rawPredict`. Google Cloud adds two more ways to run it: Cloud Run with an L4 and GKE Autopilot with an L4 node.

Cloud Run scales to zero, so the first request after an idle period waits for an instance to start, load the weights and compile:

```
2026-10-12T19:19:40.532090Z  Starting new instance.
2026-10-12T19:22:14.236742Z  Model loading took 2.86 GiB memory and 77.896787 seconds
2026-10-12T19:24:12.926565Z  torch.compile took 116.09 s in total
2026-10-12T19:24:52.957321Z  Application startup complete.
```

That request took 361.8 s and returned 200; two requests during start-up were refused with 429. Set `min-instances 1` before a demo.

Azure's equivalent, Container Apps serverless GPU, is the path the subscription's quota of 0 blocked. The image for it, vLLM 0.30.0 with a shared-memory patch for the T4, built in Azure Container Registry in 17 minutes and is in `azure-inference/stage/image/`.

---

#### Step 5 — Demo 1: One Agent, Every Cloud

Eleven ids, "how many are 10 or more?", true answer 8, temperature 0, 20 runs per backend. With `count_ids(op, value)` the engine returns the exact count, and a run has the right filter when the predicate it sent selects the same ids as `id >= 10`. One grader scores every cloud.

```
python3 crosscloud.py agent --runs 20 --only aws.bedrock,gcp.gemini,azure.foundry,azure.vm
```

```
backend           runs  correct  right-filter  errors  median s  filters sent
aws.bedrock         20   20/20          20/20       0      1.09  id >= 10
gcp.gemini          20   20/20          20/20       0      2.88  id >= 10
azure.foundry       20   20/20          20/20       0      3.17  id >= 10
azure.vm            20   20/20          20/20       0      0.38  id >= 10
```

Across the per-cloud runs, every GPU backend on every cloud scored 20 of 20 with the right filter. Gemma 4 26B on Vertex AI MaaS scored 19 of 20; the twentieth run was a 503 from the service.

With `list_ids()` the model counts the rows itself. Gemma 4 26B on MaaS answered 9 in every run that returned, 0 of 20 correct, and gpt-4.1-nano miscounted once in 20. With the count computed by the engine, the same two models scored 19 and 20.

---

#### Step 6 — Demo 2: One Agent Asks All Three Clouds

The agent gets one `ask_<cloud>_<backend>` tool per backend and a `scoreboard()` that sorts in code. Any cloud's model can orchestrate.

```
python3 crosscloud.py ask --orchestrator gcp.gemini --only aws.bedrock,gcp.gemini,azure.foundry,azure.vm
```

```
| backend | answer | seconds | tokens/s |
|---|---|---|---|
| AZURE vLLM on NV36ads_A10_v5 (A10) | Paris | 0.12 | 16.7 |
| AWS Amazon Bedrock | Paris | 0.64 | 3.1 |
| GCP Gemini on Vertex AI | Paris | 1.48 | 0.7 |
| AZURE Azure AI Foundry (gpt-4.1-nano) | Paris | 1.587 | 0.6 |

[harness] 4 backend calls made; not asked by the agent: none
```

Gemini and Nova Micro each quoted the table exactly as measured. gpt-4.1-nano called every tool and then summarised in one sentence instead of quoting the table; its one figure, 0.139 s for the A10, matches the recorded call.

---

#### Step 7 — Same Prompt, Every Backend

The same 100-token prompt, one request at a time, timed on the client. These figures include the network between the laptop and each region, so they rank serving paths as a user reaches them.

| Backend | Cloud | tok/s (median) |
|---|---|---|
| 🥇 vLLM on NV36ads_A10_v5 (A10) | Azure | 166.9 |
| 🥈 vLLM on EC2 g6 (L4) | AWS | 132.45 |
| 🥉 vLLM on Cloud Run (L4) | GCP | 119.3 |
| Amazon Bedrock, Nova Micro | AWS | 115.85 |
| SageMaker endpoint (L4) | AWS | 105.2 |
| vLLM on GKE Autopilot (L4) | GCP | 104.7 |
| vLLM on Compute Engine g2 (L4) | GCP | 94.3 |
| Gemini 3.5 Flash-Lite | GCP | 93.3 |
| Vertex AI endpoint (L4) | GCP | 90.2 |
| Azure AI Foundry, gpt-4.1-nano | Azure | 49.7 |

The Compute Engine L4 ran in Montreal and is the slowest L4 from the client while its server-side decode, 143.1 tok/s, is the fastest of the GCP L4s: the network path accounts for the difference.

---

#### Step 8 — What It Costs

List prices on the day, from each cloud's pricing API: the AWS Pricing API, the Azure Retail Prices API and the Cloud Billing Catalog API. Cost per million tokens is arithmetic: the hourly price divided by the measured tokens per hour, one request at a time.

| Backend | $/hr | $ per 1M output tokens, one request |
|---|---|---|
| Amazon Bedrock, Nova Micro | per token | 0.14 |
| Azure AI Foundry, gpt-4.1-nano | per token | 0.40 |
| EC2 g6.xlarge, L4 | 0.8048 | 1.69 |
| SageMaker ml.g6.xlarge, L4 | 1.1267 | 2.98 |
| Azure NV36ads_A10_v5, A10 | 3.20 | 5.33 |

The A10 is the fastest GPU here and the most expensive per hour: Azure sells a whole A10 only with 36 vCPUs, and the smaller NVadsA10 v5 sizes are fractions of the GPU. Batching closes most of the gap. With `google/gemma-4-E2B-it-qat-w4a16-ct`, the model AWS measured in parallel on a g6, the A10 reached 2118.05 tok/s with 16 requests at once, which is 0.420 USD per million output tokens against 0.150 on the g6.

---

#### Step 9 — Clean Up Three Clouds

Each repository's `stage/down.sh` deletes what its `up.sh` created and lists anything still billing.

```
vm: deleting cci-demo-vm with its disk, NIC and public IP
  deleted cci-demo-vmVMNic
  deleted cci-demo-vmPublicIP
  deleted cci-demo-vm_OsDisk_1_240204d13eb14717bfd8b163c8ad598a
```

Azure leaves a VM's NIC, public IP and disk behind when the VM alone is deleted; the script removes them by name.

---

#### 🔎 Tip: Quota Is Asked For, Capacity Is Found

On AWS and Google Cloud the quota was there and the zones were full; on Azure the quota for two of the three GPU paths was zero. Check both before the day: the quota page for each GPU family and serverless GPU type, then a launch loop that walks zones and regions. Every repository's `up.sh` does that walk.

---

#### 🔎 Tip: Leave Headroom on a vGPU

`google/gemma-4-E2B-it-qat-w4a16-ct` ran out of memory on the Azure A10 at `--gpu-memory-utilization 0.90` and restarted 21 times; at 0.85 it started first time, in 270 s.

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 23.72 GiB of which 204.75 MiB is free.
```

---

#### 🔎 Tip: Turn Streaming Off to Test Tool Calls

A backend can return tool calls without streaming and drop them in a stream, which is what Phi-4-mini does on Foundry. Agent frameworks stream, so test tool calls with `"stream": true` before choosing a model.

---

#### 🔎 Tip: Let the Engine Count, Then Check the Filter

The model that counted rows wrong in all 20 runs got the right answer in 19 of 20 when the engine counted. Push the arithmetic into a tool, have the model quote it, and grade the predicate it sent as well as the number.

---

#### Compare and Contrast

| | Google Cloud | AWS | Azure |
|---|---|---|---|
| Per-token API | Gemini, 20/20, 93.3 tok/s | Nova Micro, 20/20, 115.85 tok/s | gpt-4.1-nano, 20/20, 49.7 tok/s |
| Open model per token | Gemma 4 26B, 19/20, 13.23 s per run | none used | Phi-4-mini, no streamed tool calls |
| Managed GPU endpoint | Vertex AI, 20/20 | SageMaker, 20/20 | blocked: quota 0 |
| Serverless GPU | Cloud Run L4, 20/20, scales to zero | none | blocked: quota 0 |
| Your own GPU VM | L4 in Montreal after 15 refused zones | L4 and T4G | A10, the fastest GPU here |
| Own chip | TPU v5e-1; v6e had no capacity | Inferentia2, Trainium | none |

---

#### So, Which One?

Start on the per-token API of the cloud you already use: nothing to run, tool calls work, and Nova Micro and gpt-4.1-nano cost cents per million output tokens. Move to vLLM on your own GPU when you need your own weights; the same container and flags ran unchanged on all three clouds. Google Cloud offers the most ways to run that container, from Cloud Run to GKE to a Vertex AI endpoint. AWS was the cheapest per token on its own GPU and is the only cloud here with two chips of its own that run Gemma 4. Azure's A10 decoded fastest per request, at the highest hourly price, and its serverless and managed GPU paths need a quota request first.

---

#### Summary

The goal of this article was to compare LLM inference on Google Cloud, AWS and Azure behind one agent. The key to the solution was one Strands agent with a model object per backend, one grader, and the same prompts and weights on every cloud. The results were:

- 🟢 **The same vLLM container and flags served the same Gemma 4 repack on all three clouds**, and scored 20 of 20 with the right filter on every GPU backend.
- 🟢 **One agent drove all three clouds at once**, with any cloud's model as orchestrator and every number computed in code.
- 🟢 **The Azure A10 decoded fastest**: 166.9 tok/s from the client and 194.2 on the instance, against 141.7 for an L4 on the same repack.
- 🟢 **Every per-token flagship scored 20 of 20**: Nova Micro, Gemini 3.5 Flash-Lite and gpt-4.1-nano.
- ⚠️ **Quota did not buy capacity on AWS or Google Cloud**: 15 zones refused the L4 on Google Cloud and five refused inf2 on AWS.
- ⚠️ **The A10 costs four times the L4 per hour**: 3.20 against 0.8048 USD, which batching brings to 0.420 against 0.150 USD per million output tokens.
- ❌ **Azure's serverless and managed GPU paths had a quota of 0**, so Azure ran one VM and its per-token models.
- ❌ **The open models on per-token billing lagged their clouds' flagships**: Gemma 4 26B on MaaS counted rows wrong in every run, and Phi-4-mini timed out on 8 of 20 requests and drops tool calls when streaming.

Scope: one day, 2026-10-12, for Google Cloud and Azure, and two runs on 2026-10-05 for AWS, on-demand instances and list prices, client on a laptop on the US east coast. Google Cloud's L4 VM ran in Montreal, AWS in us-east-2, Azure's VM in eastus, and the Azure and AWS VMs were reached through SSH and SSM tunnels, so client tokens per second include different network paths; the on-instance and server-side figures compare the GPUs. Twenty agent runs per backend in demo 1 and five to ten timed requests per backend in the benchmark. Each per-token API serves a different model, the AWS g5g and Neuron rows serve different builds of Gemma 4 E2B, and the A10 parallel figures use the QAT model AWS used in parallel on the g6, so the speed table compares serving paths with the model each path supports.

The strategy for serving one model on Google Cloud, AWS and Azure behind one Strands agent was validated with an incremental step by step approach.

---

#### References

* [cross-cloud-inference | GitHub](https://github.com/xbill9/cross-cloud-inference)
* [gcp-inference | GitHub](https://github.com/xbill9/gcp-inference)
* [aws-inference-strands | GitHub](https://github.com/xbill9/aws-inference-strands)
* [azure-inference | GitHub](https://github.com/xbill9/azure-inference)
* [Strands Agents | GitHub](https://github.com/strands-agents/sdk-python)
* [vLLM | GitHub](https://github.com/vllm-project/vllm)
* [Install NVIDIA GPU drivers on N-series VMs running Linux | Microsoft Learn](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/n-series-driver-setup)
* [Serverless GPUs in Azure Container Apps | Microsoft Learn](https://learn.microsoft.com/en-us/azure/container-apps/gpu-serverless-overview)
* [GPUs on Cloud Run | Google Cloud](https://cloud.google.com/run/docs/configuring/services/gpu)
* [Amazon Bedrock Converse API | AWS](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)

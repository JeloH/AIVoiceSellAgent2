
# AI Sales Agent LLM Pipeline

This project develops a production-grade LLM pipeline for ABC Energy’s Voice AI Sales Agent, targeting real-time telemarketing and energy sales conversations. It covers the full LLM lifecycle including data preprocessing, synthetic dialogue generation, SFT fine-tuning, DPO alignment, model merging, quantization, and evaluation. Performance is validated using LLM-as-a-Judge evaluation and Locust-based load testing.

---

# Architectural vision


<img width="1401" height="761" alt="pica" src="https://github.com/user-attachments/assets/b230383c-0197-4387-acc7-28d6878970cf" />


# Project Structure

```bash
scripts/
│
├── bench/
│   ├── locustfile.py
│   ├── output.txt
│
├── data/
│   ├── datasets/
│   ├── clean_prs.py
│   ├── norm.py
│
├── eval/
│   ├── llm-as-judgment.py
│   ├── llm-as-judgment_output.txt
│   ├── pre_prx_output.txt
│   ├── Prepexitly_domain.py
│
├── quantization/
│   ├── guantize_model.sh
│   ├── test_gguf.sh
│
├── training/
│   ├── 1_sft_train/
│   │   ├── sft_train.py
│   │
│   ├── 2_merge_adapter/
│   │   ├── merge.py
│   │   ├── test.py
│   │
│   ├── 3_dpo_train/
│   │   ├── dbo_train.py
│   │   ├── test.py
```

---

# Overview

This project implements a complete engineering pipeline for a specialized Voice AI Sales Agent using:

* Supervised Fine-Tuning (SFT)
* Direct Preference Optimization (DPO)
* LLM-as-a-Judge evaluation
* GGUF quantization
* Benchmark/load testing
* Dockerized deployment

The model is optimized for:

* sales persuasion
* conversational engagement
* customer retention
* energy-plan recommendation
* enterprise sales interaction

---

# Models

## Hugging Face Models

### GGUF Quantized Model

* `JeloH/aisales-agent-7b-gguf`

### SFT + DPO Fine-Tuned Model

* `JeloH/aisales-agent-7b-merged3`

---

# Datasets

### Synthetic SFT + DPO Datasets

* `JeloH/SFT_tk-callagent-sales-synthetic-dpo`
* `JeloH/tk-callagent-sales-synthetic-dpo`

---

# Training Pipeline

## 1. Data Processing

```bash
python3 scripts/data/clean_prs.py
python3 scripts/data/norm.py
```

Performs:

* cleaning
* normalization
* formatting
* dialogue preparation

---

## 2. Supervised Fine-Tuning (SFT)

```bash
python3 scripts/training/1_sft_train/sft_train.py
```

Used for:

* instruction tuning
* sales conversation adaptation
* domain specialization

---

## 3. Merge LoRA Adapter

```bash
python3 scripts/training/2_merge_adapter/merge.py
```

Merges LoRA adapter into the base model.

---

## 4. DPO Training

```bash
python3 scripts/training/3_dpo_train/dbo_train.py
```

Optimizes:

* response preference
* persuasion quality
* conversational flow
* engagement strength

---

# Evaluation



### LLM-as-a-Judge

A stronger instruction-tuned model is used as an automated referee to compare the base model against the fine-tuned sales model.

The judge evaluates:

* persuasion quality
* engagement
* conversational flow
* professionalism
* ability to move conversations forward

### Example Evaluation

```text id="o3m4kv"
USER:
"I don’t know my energy usage, can you still recommend a plan?"

Judge Result:
A: Informative and clear explanation, but less interactive.
B: Strong engagement and better conversational progression through personalized follow-up questions.
```

This evaluation framework enables scalable offline validation of sales effectiveness and alignment quality without requiring continuous manual review.


## Judge Model

* `Qwen/Qwen2.5-7B-Instruct`

## Evaluation Focus

The judge evaluates:

* persuasion
* engagement
* conversion strength
* ability to move conversation forward

## Run Evaluation

```bash
python3 scripts/eval/llm-as-judgment.py
```

Example evaluation compares:

* base model
* fine-tuned model

The judge generates strengths and weaknesses for each response.

---

# Quantization

## GGUF Quantization

```bash
bash scripts/quantization/guantize_model.sh
```

## Test GGUF Model

```bash
bash scripts/quantization/test_gguf.sh
```

Supports lightweight inference and local deployment.

---

# Benchmarking

The project includes Locust-based API load testing.

## Run Benchmark

```bash
locust -f scripts/bench/locustfile.py
```

The benchmark simulates concurrent users sending chat-completion requests to the LLM API.

### Metrics

* latency
* throughput
* concurrency
* response stability

---

# Running the Model

## Method 1 — Load Directly from Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "JeloH/aisales-agent-7b-merged3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)
```

---

## Method 2 — Docker Pipeline

The project also supports a fully Dockerized pipeline for:

* data processing
* SFT training
* adapter merging
* evaluation
* automated pipeline execution

The Docker workflow automatically runs all stages sequentially inside the containerized environment.

---

# Key Features

* End-to-end LLM pipeline
* Synthetic dialogue generation
* SFT + DPO optimization
* LLM-as-a-Judge evaluation
* GGUF quantization
* Dockerized workflow
* API benchmarking with Locust
* Hugging Face deployment support


---

# Use Cases

* AI sales agents
* Voice assistants
* Customer support automation
* Energy-service recommendation systems
* Conversational AI benchmarking
* Domain-specific LLM research

---

# Author

Developed for production-grade Voice AI Sales Agent research and deployment.

---


This repository represents a full-stack LLM engineering workflow for specialized conversational AI systems.

# Hallucination Detector — Activation-Probed Routing & Audited Wikipedia Evidence QA

> **Research & Engineering Notice:** All combined and model scores produced by this pipeline are **experimental heuristic decision indices**, **not** calibrated truth or correctness probabilities. `CuratorKIT` and `AuditKIT` enforce deterministic deduplication and policy/contract auditing.

---

## 1. Overview

**Hallucination Detector** is a fail-closed factual Question Answering (QA) pipeline built around **`CohereLabs/tiny-aya-global`**. It combines:

1. **Closed-Book Activation Probing & Token Likelihood Routing:** Generates an initial answer and inspects hidden-state activations via a Platt-calibrated logistic probe (`P`) and teacher-forced token likelihood (`L`) to decide whether to answer directly or trigger live Wikipedia retrieval.
2. **Live Wikipedia Retrieval & Neural Reranking:** Searches English Wikipedia across query variants, extracts clean passages with provenance metadata, and reranks them using TF-IDF + `cross-encoder/ms-marco-MiniLM-L6-v2`.
3. **Mandatory Two-KIT Evidence Curation & Release Auditing:**
   - **CuratorKIT** performs exact-only deduplication on retrieved passages (failing closed if any distinct passage is dropped or altered).
   - **AuditKIT** executes deterministic custom policy scorers over raw QA, embedding, and NLI outputs for per-source ranking and final release contract verification (`release_audit`) across both closed-book and RAG branches.
4. **Best-Source Extractive QA Verification:** Selects the single highest-scoring eligible source span (using `deepset/roberta-base-squad2`, `all-MiniLM-L6-v2`, `DeBERTa-v3-base-mnli-fever-anli`, and `nli-deberta-v3-small`) and enforces strict provenance, span fidelity, and threshold gates before releasing an answer—otherwise abstaining (`"abstain"`).

---

## 2. File & Module Map

The codebase consists of 13 Python modules (listed below with their attached filenames and logical module roles):

| Attached File | Module / Role | Responsibility |
| :--- | :--- | :--- |
| `train_probe_v3.py` | Probe Training CLI (`prepare`, `extract`, `train`) | CuratorKIT train-split dedup, LLM activation hook extraction, probe/Platt training, risk-coverage policy selection, AuditKIT test audit, and `probe.joblib` export. |
| `kit_preflight.py` | `kit_preflight.py` | Offline SDK contract test for CuratorKIT and AuditKIT; generates `kits.lock.json`. |
| `kit_adapters.py` | `kit_adapters.py` | Mandatory third-party SDK wrappers (`require_kits`, `curate_evidence`, `audit_values`, `audited_source`, `release_contract`, `release_audit`). |
| `runtime.py` | `runtime.py` | Thread-safe `Runtime` wrapper; validates `pipeline_manifest.json` & artifact SHA-256 hashes, pins model revisions from `models.lock.json`, instruments stage timings, and clears model caches per query. |
| `tag_rag_v3.py` | `tag_rag_v3.py` | Core `TAG` model wrapper, `Wiki` MediaWiki API client, `PairModels` (ranker + NLI1), `rerank()`, `run()` / `_run_before_required_kits()`, and Markdown/JSON formatters. |
| `kit_workflow.py` | `kit_workflow.py` | Live `finalize()` and `guarded_run()` orchestrating CuratorKIT, `simple_source_workflow`, AuditKIT per-source & release audits, and `compact_result()`. |
| `simple_source_workflow.py` | `simple_source_workflow.py` | `score_source()`, `choose()`, `answer_match()`, and `finalize()` / `compact_result()` for best-source extractive QA scoring and probe-matching composite logic. |
| `generic_verifier.py` | `generic_verifier.py` | Loads auxiliary CPU models (`AdditionalModels`), `settings()`, `provenance()`, and `agreement()`. Also provides standalone `qa_dual_nli_v1` verification & offline CLI `reevaluate_file`. |
| `fetch_triviaqa.py` | Dataset Freezer (`fetch_triviaqa`) | Downloads official HuggingFace Parquet shards (`mandarjoshi/trivia_qa`) into an immutable JSONL + `.manifest.json`. |
| `benchmark.py` | Paired Benchmark Runner | Evaluates closed-book baseline (`MA`) vs pipeline (`PA`) on TriviaQA with leakage exclusion, resource monitoring (`Monitor`), and paired bootstrap metrics. |
| `metrics.py` | `metrics.py` | Alias-aware normalized Exact Match (`em`), `token_f1`, `alias_span`, and paired bootstrap confidence interval summarization (`summarize`). |
| `audit_benchmark.py` | Post-Hoc AuditKIT Runner | Runs AuditKIT `alias_em` scorer over recorded `records.jsonl` predictions for `MA` and `PA`. |
| `app.py` | Gradio Web UI | Persistent serialized Gradio interface (`0.0.0.0:8888`) exposing predictions, heuristic scores, sources, and toggleable `result.json`. |

---

## 3. Block Diagrams

### A. Offline Preparation, Training & Preflight Pipeline

```text
+-----------------------+     +----------------------------+     +----------------------------+
|  Labeled CSV Dataset  | --> | train_probe_v3.py (prepare)| --> | prepared.csv +             |
| (train / val / test)  |     | CuratorKIT exact dedup     |     | unknown_labels_for_review  |
+-----------------------+     +----------------------------+     +----------------------------+
                                                                               |
                                                                               v
+-----------------------+     +----------------------------+     +----------------------------+
|  probe.joblib +       | <-- | train_probe_v3.py (train)  | <-- | train_probe_v3.py (extract) |
|  TRAINING_COMPLETE    |     | Grid search -> Platt cal   |     | TAG forward hooks at       |
|  + AuditKIT headline  |     | -> Risk-coverage threshold |     | layers {n/3, 2n/3, n-1}    |
+-----------------------+     +----------------------------+     +----------------------------+
                                                                               
+-----------------------+     +----------------------------+     +----------------------------+
|  kit_preflight.py     | --> | Exact SDK fixture tests    | --> | kits.lock.json             |
|  (--write-lock)       |     | CuratorKIT + AuditKIT      |     | (Required before runtime)  |
+-----------------------+     +----------------------------+     +----------------------------+
```

### B. Online Inference, Curation, Verification & Release Pipeline

```text
User Question
      |
      v
+------------------------------------------------------------------------------+
| Runtime (runtime.py)                                                         |
|  - Verify SHA-256 of sources/artifacts (pipeline_manifest.json)              |
|  - Pin model revisions (models.lock.json) & load generic_config.json         |
+------------------------------------------------------------------------------+
      |
      v
+------------------------------------------------------------------------------+
| tag_rag_v3.py :: guarded_run() -> _run_before_required_kits()                |
|  1. Verify TRAINING_COMPLETE.json & dependency versions                      |
|  2. TAG.generate(BASE_PROMPT, max_new_tokens=100, do_sample=False)           |
|  3. TAG.measure(): hook activations -> Platt P_correct + token likelihood L  |
|  4. Routing index: O = 0.8 * P + 0.2 * L                                     |
+------------------------------------------------------------------------------+
      |
      +---[ answer not unknown AND O>=original_threshold AND not always_retrieve ]---+
      |                                                                                 |
      | (Retrieval Triggered: answer unknown / O<threshold / always_retrieve)                | (High Confidence)
      v                                                                                 |
+------------------------------------------------------------------+                    |
| Wikipedia Retrieval & Rerank (tag_rag_v3.py)                     |                    |
|  - Wiki.retrieve(): 3 query variants -> MediaWiki API (cached)   |                    |
|  - Sliding-window passages (<=3 sentences, <=135 words)          |                    |
|  - TF-IDF prefilter (top 60 + leads) -> ms-marco MiniLM ranker   |                    |
|  - Jaccard overlap filter (<=0.8) & max 2/page -> top_k (S1..S6) |                    |
+------------------------------------------------------------------+                    |
      |                                                                                 |
      v                                                                                 |
+------------------------------------------------------------------+                    |
| Evidence Curation & Scoring (kit_workflow.py)                    |                    |
|  1. require_kits() -> verify CuratorKIT & AuditKIT vs kits.lock  |                    |
|  2. CuratorKIT exact dedup (curate_evidence)                     |                    |
|  3. simple_source_workflow.score_source() per unique passage:    |                    |
|     - RoBERTa-SQuAD2 QA span + null_margin + answerability (B)   |                    |
|     - MiniLM cosine(question, passage)                           |                    |
|     - DeBERTa NLI2 entailment(passage, containing_sentence)      |                    |
|     - S = 0.35*B + 0.25*clip(margin/15) + 0.20*cos + 0.10*R      |                    |
|           + 0.10*NLI2_entailment                                 |                    |
|  4. AuditKIT custom scorers (audited_source) -> score & eligible |                    |
|  5. choose(): pick single highest-scoring eligible >= final_thr  |                    |
+------------------------------------------------------------------+                    |
      |                                                                                 |
      +---[ No eligible source >= final_threshold ]---> ABSTAIN                         |
      |                                                                                 |
      | (Selected source passed)                                                        |
      v                                                                                 |
+------------------------------------------------------------------+                    |
| Final Scoring Branch (kit_workflow.py)                           |                    |
|  - Run OOD diagnostic TAG.measure(rag_prompt, answer, rag=True)  |                    |
|  - answer_match(original_output, selected_answer)                |                    |
|    * Match & probe ok: F = 0.8*S + 0.2*(w*P_RAG + (1-w)*L_RAG)   |                    |
|      -> Release if F >= final_threshold, else ABSTAIN            |                    |
|    * Different answer or probe unavailable: F = S -> Release     |                    |
+------------------------------------------------------------------+                    |
      |                                                                                 |
      +---------------------------------------------------------------------------------+
      |
      v
+------------------------------------------------------------------------------+
| AuditKIT Final Release Gate (kit_adapters.py :: release_audit)               |
|  - Evaluates release_contract() inside AuditKIT over final result dict       |
|  - Any contract violation or pipeline error -> Fail Closed ("abstain")       |
+------------------------------------------------------------------------------+
      |
      v
Final JSON / Markdown / Gradio Output
```

---

## 4. Detailed Code Workflow

### Step 1: Probe Training & Calibration (`train_probe_v3.py`)
The probe is trained in three sequential subcommands (`prepare`, `extract`, `train`):
1. **`prepare`**: Reads a labeled CSV (`question_id`, `split` in `{train, validation, test}`, `question`, `model_answer`, `correctness_label` in `{-1, 0, 1}`). Runs **CuratorKIT** (`dedup='exact'`, `clean=False`) on the `train` split questions, exports `prepared.csv`, isolates `-1` (unknown) rows to `unknown_labels_for_review.csv`, and writes `data_report.json`.
2. **`extract`**: Loads `CohereLabs/tiny-aya-global` (`TAG`) and registers PyTorch forward hooks on three decoder blocks (`n//3`, `2*n//3`, `n-1`). For every row in `prepared.csv`, it replays `BASE_PROMPT + model_answer` (teacher forcing) and captures 3 pooled vectors per hooked layer:
   - Last prompt token (`h[0, n-1]`)
   - Mean over answer tokens (`h[0, n:].mean(0)`)
   - Last answer token (`h[0, -1]`)
   Outputs `feature_meta.json`, per-row `.npy` files, and compressed `features.npz`.
3. **`train`**:
   - Splits `train` (75/25 stratified into `fit`/`tune`) and grid-searches `(layer_index, mode in {'prompt', 'answer', 'joint'}, C in {0.001, 0.01, 0.1, 1.0})` using `StandardScaler + LogisticRegression`, picking the minimum `tune` log-loss.
   - Refits the winning probe on full `train`.
   - Splits `validation` (50/50 stratified into `cal`/`policy`): fits Platt scaling (`LogisticRegression`) on `cal` decision scores, then sweeps probability thresholds on `policy` (excluding native abstentions flagged by `unknown()`) to maximize coverage subject to `--min-coverage` and `--max-risk` (`validation_risk_coverage.csv`).
   - Evaluates on `test` (reporting Brier score, AUROC, baseline vs gated coverage & selective risk in `test_report.json`), runs **AuditKIT** `exact_match` verification (`auditkit_headline.json`), serializes `probe.joblib`, and writes `TRAINING_COMPLETE.json`.

### Step 2: Mandatory KIT Preflight (`kit_preflight.py` )
Before live inference can execute, `python kit_preflight.py --write-lock` runs offline fixture tests against the installed `curatorkit` and `auditkit` packages (testing exact deduplication retention and custom scorer evaluation) and writes `kits.lock.json` (pinning package versions and `direct_url.json` metadata). `kit_adapters.require_kits()` verifies this lock at runtime and fails closed if packages change or the lock is absent.

### Step 3: Runtime Initialization & Integrity Verification (`runtime.py`)
`Runtime` provides a thread-safe (`threading.RLock`) wrapper for both the Gradio UI and the benchmark script:
- Validates SHA-256 checksums of all source files and model artifacts (`probe.joblib`, etc.) against `pipeline_manifest.json`.
- Exports `TAG_GENERIC_CONFIG` if `generic_config.json` exists and sets `TAG_QA_REVISION`, `TAG_NLI2_REVISION`, `TAG_EMBED_REVISION` from `models.lock.json`.
- Instruments `TAG.generate` and `TAG.measure` with CUDA-synchronized stage timers.
- Supports `mode='rag'` (full pipeline) and `mode='probe'` (probe-only ablation without Wikipedia retrieval).
- Clears `qa_cache`, `emb_cache`, and `nli_cache` on `AdditionalModels` after every `answer()` call.

### Step 4: Closed-Book Generation & Routing (`tag_rag_v3.py`)
When `tag_rag_v3.run(args)` starts (wrapped inside `kit_workflow.guarded_run`):
1. Checks `TRAINING_COMPLETE.json` and verifies `torch`, `transformers`, and `scikit-learn` match the versions stored in `probe.joblib`.
2. Formats `BASE_PROMPT` and runs greedy decoding (`do_sample=False`, `max_new_tokens=100`) to produce `original_tag_output`.
3. Calls `TAG.measure()` to compute:
   - `P = p_correct` from the hooked activations + Platt calibrator.
   - `L = exp(mean_t log_softmax(logits_t)[answer_token_t])` (chunked over LM head to save memory).
   - Combined routing index: `O = 0.8 * P + 0.2 * L`.
4. Retrieval is triggered if **any** of the following hold:
   - `unknown(original_tag_output)` (empty or explicit abstention phrase)
   - `O` is unavailable or `O < original_threshold` (default `0.9` in `Runtime`, `0.85` in CLI)
   - `--always-retrieve` is set
   - `--respect-learned-policy` is set and the learned policy is infeasible or `P < learned_threshold`
5. If no trigger fires, the closed-book answer is passed directly to the final `release_audit` gate.

### Step 5: Wikipedia Retrieval & Neural Reranking (`tag_rag_v3.py`)
If retrieval is triggered (requires `WIKIMEDIA_USER_AGENT`):
1. `Wiki.retrieve()` builds up to 3 query variants (`raw question`, `stopword-stripped keywords`, plus `'... film cast actor'` or `'... facts'`) and queries the MediaWiki API (`srlimit=4`, cached on disk for 24 hours with retry/backoff).
2. Fetches plaintext extracts for up to `max_pages` (default `10`), discards disambiguation pages and pages lacking `fullurl` or `lastrevid`, and chunks extracts into overlapping passages (up to 3 sentences, `<= 135` words, `>= 8` words), attaching `pageid`, `revision_id`, `url`, and `retrieved_at`.
3. `rerank()` applies a TF-IDF lexical prefilter (top 60 passages + up to 2 lead passages per page), scores them with **`cross-encoder/ms-marco-MiniLM-L6-v2`** (`relevance = sigmoid(logit)`), enforces max 2 passages per page and Jaccard word similarity `<= 0.8`, and keeps `top_k` (default `6`, IDs `S1..S6`).

### Step 6: Two-KIT Evidence Curation, Scoring & Release (`kit_workflow.py`, `simple_source_workflow.py`, `kit_adapters.py`)
`tag_rag_v3._run_before_required_kits()` delegates post-retrieval processing to `kit_workflow.finalize()`:
1. **CuratorKIT Exact Deduplication (`curate_evidence`)**: Serializes passages into Alpaca-format JSONL and runs `Curator(CuratorConfig(dedup='exact', clean=False))`. Retains the highest-relevance representative for exact duplicates, logs SDK export SHA-256 digests and `stage_counts`, and fails closed if any distinct passage text is modified or lost.
2. **Per-Source Auxiliary Scoring (`score_source`)**: Uses CPU-hosted `AdditionalModels` (`generic_verifier.py`) to compute for each curated passage:
   - Extractive QA span (`deepset/roberta-base-squad2`): `answer`, `null_margin`, `B = sigmoid(null_margin)`, and `exact_answer_span`.
   - Question-passage semantic linkage (`sentence-transformers/all-MiniLM-L6-v2` cosine).
   - Verbatim containing sentence extraction and NLI fidelity check (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, premise = passage, hypothesis = verbatim sentence).
   - Composite source score:
     $$S = 0.35 B + 0.25 \cdot \mathrm{clip}\!\left(\frac{\text{margin}}{15}\right) + 0.20 \cdot \mathrm{clip}(\text{cosine}) + 0.10 \cdot \mathrm{clip}(R) + 0.10 \cdot \mathrm{clip}(\text{NLI2\_entailment})$$
   - Hard eligibility gates: `exact_answer_span`, `B >= 0.70`, `null_margin >= 1.0`, `R >= min_relevance (0.4)`, `cosine >= 0.10`, and valid Wikipedia HTTPS `provenance`.
3. **AuditKIT Per-Source Execution (`audited_source`)**: Evaluates all 8 metrics/gates inside `auditkit.evaluate()` custom scorers over serialized raw model outputs; the SDK-returned headline values populate `record['score']` and `record['eligible']`.
4. **Best-Source Selection (`choose`)**: Picks the single highest-scoring eligible source. If none are eligible or the best score `< final_threshold` (default `0.75`), the pipeline abstains. *(Note: this policy does not require cross-source consensus and does not veto based on unselected passages.)*
5. **Original-Answer Matching & Final Score**:
   - Runs `TAG.measure(..., rag=True)` as an out-of-distribution diagnostic (`rag_probe_diagnostic`).
   - Extracts a QA span from `original_tag_output` and checks equivalence (`generic_verifier.agreement`: lexical F1 `>= 0.65` with identical numbers, OR cosine `>= 0.88` with identical numbers and bidirectional NLI2 entailment `>= 0.85`).
   - **If answers match and RAG probe is available:**
     $$F = 0.80 \cdot S + 0.20 \cdot \bigl(w \cdot P_{\text{RAG}} + (1 - w) \cdot L_{\text{RAG}}\bigr)$$
     Released only if `F >= final_threshold`.
   - **If answers differ or RAG probe is unavailable:** `F = S` (the original answer and OOD RAG probe cannot veto a correction).
6. **AuditKIT Final Release Gate (`guarded_run` -> `release_audit`)**: Runs `release_contract()` inside `auditkit.evaluate()` to verify every non-abstaining answer satisfies exact span/provenance/audit/threshold invariants (or closed-book threshold invariants). Fails closed (`abstain`) on any contract violation or pipeline error.

### Step 7: Offline Alternative Verifier (`generic_verifier.py` / `input_file_4.py`)
In addition to supplying `AdditionalModels`, `settings()`, `provenance()`, and `agreement()` to the active workflow, `generic_verifier.py` contains a standalone `qa_dual_nli_v1` verifier (`verify()`, `comparison()`, `finalize()`) and an offline CLI (`reevaluate_file`) that re-runs fresh CPU QA + dual-NLI verification (requiring multi-passage support, zero competing-answer consensus contradiction, and an extra NLI2 floor when contradicting `original_tag_output`) on previously saved JSON reports without needing a GPU or TAG weights.


### Step 8: Dataset Freezing, Benchmarking, Auditing & UI
- **Dataset Freezing (`fetch_triviaqa.py`)**: Exports official `mandarjoshi/trivia_qa` (`rc.nocontext` / `validation`) Parquet shards from `refs/convert/parquet` into `data/triviaqa_val.jsonl` + `.manifest.json` (fails if any row lacks labeled answers).
- **Paired Benchmark (`benchmark.py` + `metrics.py`)**: Evaluates `MA` (`original_tag_output`) vs `PA` (`final_answer`) on non-overlapping questions (filtered via `training_exclusions.json`). Records `records.jsonl`, `metrics.json` (alias-aware EM, token F1, alias span, coverage, selective EM error, fixed/regressed counts, and 2000-sample paired bootstrap 95% CI), `footprint.json` (wall/CPU latency, sampled tree RSS, CUDA peak memory, HF cache disk delta), `results.csv`, and `environment.freeze.txt`.
- **Post-Hoc Audit (`audit_benchmark.py`)**: Re-scores `records.jsonl` using AuditKIT's `@ak.scorer` (`alias_em`) for both `MA` and `PA`.
- **Gradio UI (`app.py`)**: Serves a single-concurrency interactive UI on port `8888` showing Original TAG output, Selected retrieval answer, Final output, Final score, Wikipedia sources, and a checkbox for full compact `result.json`.

---

## 5. Models & Default Thresholds

### Models Used
| Role | Model ID | Device | Notes |
| :--- | :--- | :--- | :--- |
| Primary Generator & Probe Backbone | `CohereLabs/tiny-aya-global` | CUDA / Auto | Revision pinned in `probe.joblib`; greedy decoding (`do_sample=False`). |
| Passage Cross-Encoder Ranker | `cross-encoder/ms-marco-MiniLM-L6-v2` | CPU | Scores query-passage relevance (`sigmoid(logit)`). |
| NLI1 (PairModels) | `cross-encoder/nli-deberta-v3-small` | CPU | Used in `generic_verifier` / `score_source` diagnostics (strictly `<= 512` tokens, no silent truncation). |
| Extractive QA | `deepset/roberta-base-squad2` | CPU | Extracts spans (`max_length=384`, `stride=96`) and CLS null margin. |
| NLI2 | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | CPU | Verifies sentence fidelity and semantic equivalence (strictly `<= 512` tokens). |
| Sentence Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | CPU | L2-normalized mean pooling (`max_length=256`) for cosine linkage & equivalence. |

### Key Defaults (`generic_verifier.DEFAULTS` & Runtime/CLI)
- `original_threshold`: `0.90` (`Runtime` / Gradio / Benchmark) or `0.85` (`tag_rag_v3.py` CLI)
- `final_threshold`: `0.75`
- `probe_weight`: `0.80`
- `min_relevance`: `0.40`
- `qa_answerability`: `0.70`, `qa_margin`: `1.0`, `qa_agreement`: `0.65`, `semantic_equivalence`: `0.88`, `embedding_relevance`: `0.10`
- `max_evidence`: `6`, `max_answer_words`: `65`

---

## 6. Environment Variables & CLI Commands

### Environment Variables
| Variable | Purpose |
| :--- | :--- |
| `WIKIMEDIA_USER_AGENT` | **Required for retrieval.** Compliant Wikimedia User-Agent with app name/version and contact info. |
| `WIKI_CACHE_DIR` | Directory for 24-hour MediaWiki response cache (default: `wiki_cache_v3`). |
| `ARTIFACT_DIR` | Artifact directory for Gradio Space (default: `artifacts`). |
| `DEVICE` | Inference device for TAG (default: `cuda`). |
| `ORIGINAL_THRESHOLD` / `FINAL_THRESHOLD` | Routing and release thresholds for Gradio Space (defaults: `0.9` / `0.75`). |
| `TAG_GENERIC_CONFIG` | Optional JSON file path overriding `generic_verifier.DEFAULTS` (auto-set by `Runtime` if `generic_config.json` exists). |
| `TAG_QA_REVISION`, `TAG_NLI2_REVISION`, `TAG_EMBED_REVISION` | Optional HuggingFace model revisions (auto-populated by `Runtime` from `models.lock.json`). |
| `HF_HOME` | HuggingFace cache path tracked by benchmark disk footprint monitoring. |

### Quick Command Reference

```bash
# 1. Train probe pipeline (CuratorKIT -> Activation Extraction -> Probe + AuditKIT)
python train_probe.py prepare --csv dataset.csv --out artifacts
python train_probe.py extract --out artifacts --device cuda
python train_probe.py train   --out artifacts --min-coverage 0.5 --max-risk 0.1

# 2. Run mandatory SDK preflight & write kits.lock.json
python kit_preflight.py --write-lock

# 3. Single-question CLI query
export WIKIMEDIA_USER_AGENT="TAG-RAG-Research/3.0 (contact@example.org)"
python tag_rag_v3.py --out artifacts --question "Who directed Inception?" --format markdown

# 4. Launch Gradio Space UI (port 8888)
python app.py

# 5. Freeze TriviaQA validation split & run paired benchmark
python fetch_triviaqa.py --output data/triviaqa_val.jsonl
python benchmark.py --data data/triviaqa_val.jsonl --out runs/rag_eval --artifacts artifacts
python audit_records.py --records runs/rag_eval/records.jsonl --output runs/rag_eval/auditkit.json

# 6. Offline re-verification of a saved report (no GPU / TAG required)
python generic_verifier.py --input saved_report.json --output reevaluated_report.json
```

---

## 7. Operational Caveats & Design Guarantees

- **Strict Fail-Closed Policy:** Modified SDK packages, NLI sequences exceeding `512` tokens, or AuditKIT `release_contract` failures all abort or force `"abstain"` rather than silently falling back.
- **Best-Source Policy Tradeoff:** `kit_workflow` / `simple_source_workflow` selects the single highest-scoring eligible source and ignores disagreement in unselected passages; it can still select a confidently wrong Wikipedia passage.
- **Benchmark Interpretation:** Normalized alias-aware EM (`M1`/`M2`) penalizes full-sentence closed-book answers relative to short extractive RAG spans. Always inspect `token_f1`, `alias_span`, `coverage`, and `selective_em_error` alongside `M1`/`M2`; benchmark non-matches and abstentions are not proof of factual hallucination.

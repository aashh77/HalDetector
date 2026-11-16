# HalDetector: Activation-Based Hallucination Detection and Evidence-Grounded Answer Verification

HalDetector combines internal activation probing, retrieval-based evidence verification and controlled abstention to improve Tiny Aya Global's correct-answer rate from 36.8% to 76.5% on a 2,000-question evaluation, while reducing incorrect answers from 61.7% to 13.1%.

---

## 1. Tracks and Problem Statements Covered

Track 2b — AI Safety and Correction

Problem Statement 13: Hallucination detection and abstention: Train probes on the model's internal activations to flag likely hallucinations, and wire them in so the model abstains or triggers retrieval when a probe fires. Show fewer wrong answers while it still answers most questions.

---

## 2. Abstract

Large language models can generate fluent and convincing answers even when their underlying information is incorrect. HalDetector addresses this reliability problem through a combination of internal activation probing, retrieval-augmented verification and controlled answer release.

Built around `CohereLabs/tiny-aya-global`, the system uses a lightweight logistic-regression probe trained on the model's hidden-state representations to estimate answer reliability. This signal is combined with token likelihood to determine whether an answer can be released directly or requires further verification.

Uncertain answers enter a retrieval pipeline using English Wikipedia. Retrieved passages undergo TF-IDF filtering, neural cross-encoder reranking, extractive question answering, semantic similarity and natural language inference (NLI) checks. CuratorKIT and AuditKIT provide evidence curation, auditing and release validation. Answers that fail verification are withheld through a fail-closed abstention policy.

On a 2,000-question evaluation, the original model correctly answered 36.8% of questions. With HalDetector, this increased to 76.5%. The incorrect-answer rate fell from 61.7% to 13.1%, a reduction of 48.6 percentage points.

An initial 200-question evaluation showed a similar improvement: the correct-answer rate increased from 26.5% to 67.5%, while the incorrect-answer rate decreased from 73% to 23%.

These results demonstrate that useful reliability signals can be extracted from internal model representations using a relatively lightweight probe, even with limited training data and computational resources. HalDetector establishes a foundation for improving factual reliability without full-model fine-tuning.

### Headline: Before vs After

#### Primary evaluation — 2,000 questions

| Metric | Original model | HalDetector |
|---|---|---|
| Correct answers | 36.8% | 76.5% |
| Incorrect answers | 61.7% | 13.1% |
| Abstentions | 1.4% | 10.4% |

#### Initial evaluation — 200 questions

| Metric | Original model | HalDetector |
|---|---|---|
| Correct answers | 26.5% | 67.5% |
| Incorrect answers | 73% | 23% |
| Abstentions | 0.5% | 9.5% |

Base model + method: Tiny Aya Global + calibrated activation probe + token-likelihood routing + Wikipedia retrieval + neural evidence verification.

### For judges: One minute

HalDetector detects potentially unreliable answers by examining a language model's internal activations. A lightweight probe estimates answer reliability and determines whether the response should be accepted or sent for retrieval-based verification.

Uncertain answers undergo Wikipedia retrieval, neural reranking, extractive QA and NLI-based evidence checks. CuratorKIT and AuditKIT support evidence curation and deterministic auditing, while a fail-closed policy prevents insufficiently verified answers from reaching users.

On the 2,000-question evaluation, HalDetector increased correct answers from 36.8% to 76.5% and reduced incorrect answers from 61.7% to 13.1%. The initial 200-question evaluation also showed an increase in correct answers from 26.5% to 67.5% and a reduction in incorrect answers from 73% to 23%.

The result is a safety-first factual QA pipeline that combines internal reliability signals with external evidence.

---

## 3. Problem Statement and Approach

### 3.1 Motivation and real-world relevance

A language model's fluency does not guarantee factual correctness. In applications such as education, research and information retrieval, confidently presented incorrect information can mislead users and undermine trust.

HalDetector introduces a reliability layer between answer generation and delivery. Rather than trusting every generated response or relying exclusively on external retrieval, it combines internal model signals with evidence-based verification.

The system follows three principles:

- **Detect:** Estimate answer reliability using internal activations.
- **Verify:** Retrieve and evaluate external evidence when uncertainty is detected.
- **Abstain:** Withhold answers that fail the required verification conditions.

### 3.2 Technical approach

#### Stage 1 — Activation-based detection

Tiny Aya Global generates an initial response. PyTorch forward hooks extract hidden-state representations from selected transformer layers.

A StandardScaler and logistic-regression classifier estimate the likelihood that the response belongs to the correct-answer class. The training pipeline evaluates prompt-level, answer-level and joint representations, performs hyperparameter selection and applies Platt scaling for calibration.

The probe score is combined with teacher-forced token likelihood to produce an initial routing score.

#### Stage 2 — Retrieval and evidence verification

Uncertain responses trigger retrieval through English Wikipedia. The pipeline generates multiple query variants, extracts relevant passages and applies:

- TF-IDF filtering.
- Neural cross-encoder reranking.
- CuratorKIT exact deduplication.
- Extractive QA using RoBERTa-SQuAD2.
- Semantic similarity using sentence embeddings.
- NLI-based entailment and consistency checks.

Candidate sources are scored using answerability, relevance, semantic similarity and entailment.

#### Stage 3 — Auditing and answer release

AuditKIT evaluates source-level eligibility and final release conditions. Candidate answers undergo answer-span, provenance and verification checks.

Only answers satisfying the configured requirements are released. If verification fails, the system abstains through a fail-closed policy.

### 3.3 Research hypothesis

Our hypothesis is that internal hidden-state representations contain useful information about answer correctness that can be extracted using a lightweight classifier.

Combining this signal with retrieval and evidence verification should reduce incorrect answer delivery without requiring extensive modification of the underlying model.

---

## 4. Hackathon Repositories (Kits) Used

### CuratorKIT

Used during probe-data preparation and retrieval evidence curation.

- Exact duplicate detection in training data.
- Preservation of distinct examples.
- Deduplication of retrieved evidence.
- Evidence-integrity checks and curation reporting.

### AuditKIT

Integrated into evaluation and final answer release.

- Deterministic scoring of QA, embedding and NLI outputs.
- Source-level eligibility checks.
- Evidence and provenance auditing.
- Final release-contract verification.

### How the repositories connect

CuratorKIT maintains training-data and retrieval-evidence quality, while AuditKIT evaluates the verification results and enforces final release conditions. Together, they provide a curation and auditing layer around the activation probe and retrieval pipeline.

---

## 5. Models and Licenses

### Base Aya model

| Configuration | Details |
|---|---|
| Model | `CohereLabs/tiny-aya-global` |
| Framework | PyTorch, Hugging Face Transformers |
| Inference | Greedy decoding |
| Device | CUDA / automatic device configuration |
| Model modification | Frozen base model; no full-model fine-tuning |
| Activation extraction | PyTorch forward hooks |
| Revision | Recorded in `models.lock.json` |

### Auxiliary models

| Model | Purpose |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Neural passage reranking |
| `deepset/roberta-base-squad2` | Extractive QA and answerability |
| `cross-encoder/nli-deberta-v3-small` | NLI verification |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Entailment and semantic verification |
| `sentence-transformers/all-MiniLM-L6-v2` | Semantic similarity |

The base model remains frozen; only the separate activation probe is trained. Model identifiers and revisions are maintained in the repository configuration.

---

## 6. Datasets, Benchmarks and Licenses

### 6.1 Data sources

The project uses labeled factual QA examples to train the activation probe.

- Benchmark: TriviaQA.
- Source: `mandarjoshi/trivia_qa` on Hugging Face.
- Evaluation: 2,000-question evaluation, alongside an initial 200-question evaluation.
- Languages: English.

The `fetch_triviaqa.py` utility freezes benchmark data into JSONL format with an accompanying manifest.

### 6.2 Training labels

| Label | Meaning |
|---|---|
| 1 | Correct |
| 0 | Incorrect |
| -1 | Abstained |

Unknown examples are separated rather than automatically treated as correct or incorrect.

### 6.3 Splits and leakage prevention

The probe-training pipeline uses separate training, validation and test splits.

- Training data is used for classifier fitting.
- Validation data supports hyperparameter and policy selection.
- A separate calibration subset is used for Platt scaling.
- Test data is reserved for evaluation.

CuratorKIT performs exact deduplication on training data, and the benchmark supports exclusion of training-overlapping questions.

---

## 7. Metrics and Why These Matter

| Metric | Purpose | Preferred direction |
|---|---|---|
| Correct-answer rate | Overall answer correctness | Higher |
| Incorrect-answer rate | Incorrect answers delivered | Lower |
| Coverage | Questions receiving answers | Higher |
| Abstention rate | Questions withheld | Context-dependent |
| Selective risk | Errors among released answers | Lower |
| Exact Match | Agreement with reference answers | Higher |
| Token F1 | Token-level answer overlap | Higher |
| AUROC | Correct/incorrect discrimination | Higher |
| Brier score | Confidence calibration | Lower |

The evaluation emphasizes both correctness and coverage. This is important because a system should not appear reliable simply by abstaining on nearly every question.

The repository also supports paired bootstrap confidence intervals and resource monitoring.

---

## 8. Method and Experimental Setup

### 8.1 Activation extraction

PyTorch forward hooks capture hidden states at three selected transformer depths: approximately one-third, two-thirds and the final layer.

Three representation modes are evaluated:

- **Prompt representation:** last prompt-token state.
- **Answer representation:** mean answer-token state.
- **Joint representation:** combined prompt and answer representations.

The extracted features are stored in NumPy format for probe training.

### 8.2 Probe training and calibration

The classifier uses StandardScaler followed by Logistic Regression.

The training pipeline evaluates three layers, three representation modes and regularization values: `{0.001,0.01,0.1,1.0}`

The configuration with the lowest validation tuning log-loss is selected. Platt scaling is subsequently applied using a separate calibration subset.

### 8.3 Routing and retrieval

The activation-probe score (P) and teacher-forced token likelihood (L) are combined:

```text
O=0.8P+0.2L
```

This routing index determines whether an answer proceeds directly or requires retrieval-based verification.

For uncertain answers, the pipeline generates up to three query variants and searches Wikipedia. Retrieved passages undergo TF-IDF filtering, cross-encoder reranking and evidence deduplication.

### 8.4 Evidence verification and release

RoBERTa-SQuAD2 extracts candidate answer spans, MiniLM evaluates semantic similarity, and DeBERTa-based models perform entailment checks.

The resulting evidence scores are evaluated alongside provenance and answer-span requirements.

AuditKIT validates the final release conditions. Answers that satisfy the requirements are returned; otherwise, the system abstains.

---

## 9. Results

### 9.1 Original model vs HalDetector — 2,000-question evaluation

The following results compare the original model (`MA_check`) against the HalDetector pipeline (`PA_check`) on the 2,000-question evaluation.

| Outcome | Original model (`MA_check`) | HalDetector (`PA_check`) |
|---|---|---|
| Correct-answer rate | 36.8% | 76.5% |
| Incorrect-answer rate | 61.7% | 13.1% |
| Abstention rate | 1.4% | 10.4% |

#### Observation 1: 39.7-percentage-point improvement in accuracy

The correct-answer rate improved from 36.8% to 76.5%, a 39.7-percentage-point gain.

This was achieved without full-model fine-tuning, using a lightweight activation probe alongside retrieval and verification.

#### Observation 2: 48.6-percentage-point reduction in incorrect answers

The incorrect-answer rate fell from 61.7% to 13.1%, a 48.6-percentage-point reduction.

This directly addresses the project's central objective: preventing confidently presented but incorrect answers from reaching users.

#### Observation 3: Retrieval as a correction mechanism

The retrieval pipeline provides an additional opportunity to recover uncertain responses rather than simply rejecting them.

Neural reranking, extractive QA, semantic similarity and NLI checks evaluate whether external evidence supports a candidate answer.

This makes HalDetector both a detection and correction pipeline.

#### Observation 4: Safety-first abstention

Abstentions increased from 1.4% to 10.4%. The system deliberately prioritizes reliable answers over maximum coverage. Responses that cannot satisfy verification requirements are withheld rather than presented as established facts.

#### Observation 5: Sample efficiency of activation probing

Despite the limited training dataset, the probe learned useful patterns in Tiny Aya Global's internal representations.

Achieving a 39.7-percentage-point accuracy gain and 48.6-percentage-point reduction in incorrect answers under constrained compute and dataset availability provides encouraging evidence of the sample efficiency of this approach.

### 9.2 Initial 200-question evaluation

The initial evaluation also demonstrated substantial improvements in answer reliability.

| Outcome | Original model | HalDetector |
|---|---|---|
| Correct answers | 53 | 135 |
| Incorrect answers | 146 | 46 |
| Abstentions / unknown | 1 | 19 |
| Total | 200 | 200 |
| Correct-answer rate | 26.5% | 67.5% |
| Incorrect-answer rate | 73% | 23% |
| Abstention rate | 0.5% | 9.5% |

#### Observation 1: 41-percentage-point improvement in accuracy

The original model correctly answered 53 questions. With HalDetector, this increased to 135.

The correct-answer rate improved from 26.5% to 67.5%, a 41-percentage-point gain.

This was achieved without full-model fine-tuning, using a lightweight activation probe alongside retrieval and verification.

#### Observation 2: 50-percentage-point reduction in incorrect answers

Incorrect answers decreased from 146 to 46, representing a reduction of 100 incorrect responses.

The incorrect-answer rate fell from 73% to 23%, a 50-percentage-point reduction, or approximately 68.5% relative reduction.

This directly addresses the project's central objective: preventing confidently presented but incorrect answers from reaching users.

#### Observation 3: Retrieval as a correction mechanism

The retrieval pipeline provides an additional opportunity to recover uncertain responses rather than simply rejecting them.

Neural reranking, extractive QA, semantic similarity and NLI checks evaluate whether external evidence supports a candidate answer.

This makes HalDetector both a detection and correction pipeline.

#### Observation 4: Safety-first abstention

Abstentions increased from 1 to 19, giving the system 90.5% answer coverage.

Of the 181 released answers, 135 were correct, corresponding to approximately 74.6% correctness among released responses.

The system deliberately prioritizes reliable answers over maximum coverage. Responses that cannot satisfy verification requirements are withheld rather than presented as established facts.

#### Observation 5: Sample efficiency of activation probing

Despite the limited training dataset, the probe learned useful patterns in Tiny Aya Global's internal representations.

Achieving a 41-percentage-point accuracy gain and 50-percentage-point reduction in incorrect answers under constrained compute and dataset availability provides encouraging evidence of the sample efficiency of this approach.

### 9.3 Remaining gap

The system still abstains on some potentially correct answers.

The primary reason is limited training data: the probe may not have encountered enough representative activation patterns to assign sufficiently high confidence to every correct response.

Conservative confidence requirements further contribute to these rejections.

This is a deliberate safety-first trade-off. Expanding the training data and improving calibration are the next steps toward increasing answer coverage without sacrificing reliability.

---

## 10. Limitations

#### Hardware and computational requirements

Hidden-state extraction requires running the base model and storing intermediate representations. Retrieval adds inference for reranking, extractive QA and evidence verification.

Limited GPU availability, memory and inference time constrained the scale of experimentation.

#### Dependence on the base model

The probe is trained on Tiny Aya Global's hidden-state representations. Its learned patterns may not transfer directly to other models or different model versions. Changes in weights, tokenization or generation settings may require retraining or recalibration.

#### Dataset size and coverage

The probe uses a relatively small labeled dataset. Long-form reasoning, conversational queries and specialized domains require further evaluation.

#### Retrieval limitations

The current pipeline relies primarily on English Wikipedia. Information may be incomplete or outdated, and retrieval quality depends on query formulation, page coverage, passage extraction and reranking.

#### Conservative answer release

The system prioritizes reliability over maximum coverage. Limited training data and conservative confidence requirements can cause some correct answers to be withheld.

---

## 11. With More Credits

Additional compute, data and development time would enable the following improvements.

- **Expand training data:** Include more factual domains, question structures and answer types to improve probe generalization.
- **Improve calibration:** Optimize risk-coverage trade-offs to reduce unnecessary abstentions while maintaining low incorrect-answer rates.
- **Expand retrieval:** Integrate additional trusted sources and stronger cross-source evidence verification.
- **Evaluate other models:** Retrain probes on additional base models to study transferability and generalization.
- **Scale benchmarking:** Evaluate on larger datasets and report confidence intervals, selective risk and retrieval-specific metrics.
- **Optimize inference:** Use more GPU resources, batching and memory-efficient execution to scale activation extraction and verification.

These extensions aim to increase answer coverage and retrieval quality while preserving the substantial reduction in incorrect responses already observed.

---

## 12. Artifacts

GitHub repository: https://github.com/aashh77/HalDetector

The repository contains the training pipeline, activation extraction, retrieval and verification components, benchmark utilities, model configurations and interactive application.

- **Dataset:** TriviaQA data preparation utility and generated evaluation artifacts.
- **Probe:** Exported as `probe.joblib`.
- **Model configurations:** `models.lock.json`.
- **Kit configurations:** `kits.lock.json`.
- **Demo:** Gradio application in `app.py`.

The application exposes original answers, retrieved candidates, final responses, confidence and verification information, and supporting sources.

---

## 13. Documentation and Reproduction

### 13.1 Environment requirements

Python, PyTorch, Hugging Face Transformers, Scikit-learn, NumPy, Pandas, Sentence Transformers, CuratorKIT, AuditKIT, Gradio and CUDA-enabled inference support.

Dependencies are recorded in `requirements.txt` and `requirements.lock.txt`.

### 13.2 Repository layout

| File | Responsibility |
|---|---|
| `train_probe_v3.py` | Dataset preparation, activation extraction and probe training |
| `tag_rag_v3.py` | Model inference, activation routing and retrieval |
| `kit_preflight.py` | Kit contract validation |
| `kit_adapters.py` | Kit integration wrappers |
| `kit_workflow.py` | Evidence curation and release workflow |
| `simple_source_workflow.py` | Evidence scoring and source selection |
| `generic_verifier.py` | Auxiliary verification models |
| `runtime.py` | Runtime initialization and integrity checks |
| `fetch_triviaqa.py` | Dataset preparation |
| `benchmark.py` | Baseline and pipeline evaluation |
| `metrics.py` | Evaluation metrics |
| `audit_benchmark.py` | AuditKIT benchmark auditing |
| `app.py` | Gradio interface |

### 13.3 Reproduction workflow

```bash
# Prepare training data
python train_probe_v3.py prepare --csv dataset.csv --out artifacts

# Extract hidden states
python train_probe_v3.py extract --out artifacts --device cuda

# Train and calibrate the probe
python train_probe_v3.py train --out artifacts --min-coverage 0.5 --max-risk 0.1

# Validate kit integration
python kit_preflight.py --write-lock

# Run an individual question
python tag_rag_v3.py --out artifacts --question "Who directed Inception?" --format markdown

# Launch the interface
python app.py

# Prepare and run benchmark
python fetch_triviaqa.py --output data/triviaqa_val.jsonl
python benchmark.py --data data/triviaqa_val.jsonl --out runs/rag_eval --artifacts artifacts

# Audit benchmark results
python audit_benchmark.py --records runs/rag_eval/records.jsonl --output runs/rag_eval/auditkit.json
```

---

## 14. Discussion and What We Learned

HalDetector demonstrates that useful reliability signals can be extracted from a language model's internal representations without extensive modification of its weights.

An important lesson is that hallucination detection and correction are complementary. A probe can identify potentially unreliable answers, but retrieval and evidence verification provide a mechanism to recover better-supported responses.

The integration of CuratorKIT and AuditKIT further emphasizes the importance of maintaining evidence quality and enforcing deterministic release conditions.

The results are particularly encouraging given the limited training data and computational resources. The remaining challenge is improving the balance between reliability and coverage. With more representative training data, improved calibration and additional compute, the architecture can be developed into a more scalable factual reliability layer.

Our central takeaway: A language model should not be trusted solely because it can generate an answer. Its internal reliability signals and the external evidence supporting its response should both influence whether that answer reaches the user.

---

## 15. Teammates Tagged

Team members: Aayushi Agarwal [https://github.com/aashh77] , Priyangi Gupta [https://github.com/PriyangiLI]

---

## 16. References

- HalDetector — https://github.com/aashh77/HalDetector
- Tiny Aya Global — https://huggingface.co/CohereLabs/tiny-aya-global
- TriviaQA — https://huggingface.co/datasets/mandarjoshi/trivia_qa
- CuratorKIT and AuditKIT
- MS MARCO MiniLM cross-encoder.
- RoBERTa SQuAD2.
- DeBERTa NLI models.
- Sentence Transformers MiniLM.

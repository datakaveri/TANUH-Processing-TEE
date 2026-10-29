# TANUH benchmarking: current phase and next phase

This document is split into two independent parts:

- **Part 1: Current phase.** Run the **ONNX model + adaptor** flow end to end, on the two datasets OCS and BCD. It covers only what this needs: the flow, the live deployment, the status, and each component's remaining tasks.
- **Part 2: Next phase.** Everything after that:
  - easier preprocessing (wrapping tools, Option C, provider scripts)
  - a simpler, generalised adaptor
  - TorchScript and Hugging Face in the UI
  - more problem types
  - proper model and dataset contracts
  - security hardening

  Each topic has its own task list per component.

**What changed since 1 Oct** (the 15 Oct outage, the streaming release, new datasets) is summarised in **1.0**; the sections below are updated in place.

**To find your work:** go to your component's task table in **1.8** (current phase) and in the Part 2 section for your topic. Each task says **where** (repo and file) and **done when**. Line numbers are approximate; search for the named function.

**Task status:** ✅ = done, with the date. Everything else is open.

---

## Shared: key terms

| Term | Meaning |
|---|---|
| **Dataset** | Files such as images or DICOM scans, plus a ground-truth table with the correct answer for each file. Uploaded encrypted by a data provider. |
| **Model** | A trained file that takes an image and returns numbers. The platform runs ONNX, TorchScript and Hugging Face; **the current phase uses ONNX only**. |
| **Preprocessing** | Converting a raw image into the exact format the model was trained on: size, colour order, number range. If it's wrong, **nothing fails but the score is lower**. Think of it as sending Fahrenheit to an API trained on Celsius. |
| **Wrapping** | Adding a model's missing preprocessing steps to the front of its ONNX file, without changing the trained network (`tools/reference/wrap_onnx.py`). |
| **Adaptor** (`adaptor.py`) | A small script from the model owner that turns the model's raw output into the standard `predictions.csv`. |
| **Bucket** | A problem type, such as `binary_classification` or `multiclass_classification`. Each bucket has one evaluator, which we own. |
| **Evaluator** (`evaluate.py`) | Compares `predictions.csv` with the ground truth and computes metrics such as accuracy and F2. |
| **Catalogue** | Stores each dataset's settings (bucket, class names, required metrics). The Processing TEE reads them at job time. |

## Shared: decisions (30 Sep 2026)

| Topic | Decision |
|---|---|
| **Current phase scope** | ONNX models + `adaptor.py`, on OCS and BCD, CPU only, in `proj-tanuh-benchmark-ptfm` (asia-south1). |
| **Model preprocessing** | **Inside the model file.** The platform only decodes, resizes, scales to 0–1 and arranges channels. Our two trained models are wrapped with `wrap_onnx.py`. |
| **Option C** (preprocessing from settings) | **Next phase** (2.2.2). |
| **Provider preprocessing script** | **Next phase** (2.2.3), after the sandbox. |
| **Adaptor contract** | The current phase keeps the **script** contract (1.3). A simpler one-function contract with a built-in default is **next phase** (2.3). |
| **TorchScript / Hugging Face** | Supported by the TEE code; **UI support is next phase** (2.4). |
| **Debug Confidential Space image** | Fine for now. Moving to the production image is **next phase** (2.8). |
| **Sandbox for owner code** | **Next phase** (2.8). It's a known risk. |
| **New buckets** (1 Oct) | `segmentation` and `object_detection` built and live in the Processing TEE (PT12, PT13), ahead of the UI (UI9). Both need an adaptor. |
| **Catalogue fixes** (1 Oct) | Live entries may be fixed directly: the controlplane API (merges keys) or Kibana scripted updates (can delete keys). Back up the entry first. |

---

# Part 1: Current phase: ONNX + adaptor, end to end

## 1.0 What changed since 1 Oct (updated 22 Oct 2026)

### The outage, 15–19 Oct

On 15 Oct the Buffer TEE (`test-cpu-c3`) was killed by the kernel's out-of-memory killer while dispatching a job with **1.95 GB of weights**. Confidential Space's default restart policy is `Never`, so the container stayed down for **about 4 days** (`/tee` returned 502) until the VM was restarted on 19 Oct, 04:49 UTC. The queue lived in the container's filesystem and was lost; users had to resubmit.

**The cause was whole-file buffering.** Measured peak memory, for artifacts of total size S:

| Step | Peak memory | Why |
|---|---|---|
| Buffer upload | ≈ 2.4×S | every decrypted chunk appended into one array |
| Buffer dispatch | ≈ 6.4×S (7.8×S in production) | read the files, base64, `json.Marshal`, copies, plus a plaintext copy in `outgoing/` |
| Processing receive | ≈ 8×S | `ReadAll`, `Unmarshal`, base64 decode, receipts; **silently cut at 2 GiB** |

On the 16 GB `c3-standard-4`, the safe limit was about 1 GiB of artifacts per job.

### The fix: streaming store-and-forward (payload v3)

Branch **`streaming-changes`** in both repos, pushed 22 Oct, not merged yet:
- Processing: `7e01d3a` (code), `fb81152` (README)
- Buffer: `c190ac5` (code), `d44f211` (README)

| | Before | Now |
|---|---|---|
| **Browser → Buffer** | one `PUT`; the Buffer decrypts the whole file in memory, hashes it, stores **plaintext** | one `PUT`, streamed to `artifacts/<slot>.enc` **still encrypted**. The Buffer only checks that every announced chunk arrived and nothing else, and HPKE-unwraps the file's AES key, which it keeps **in memory only** (never in `job.json`) |
| **Buffer → Processing** | payload v2: one JSON with base64 files, plus a plaintext copy in `outgoing/` | payload v3: a multipart stream. First a `metadata` part (job fields and, per file, its AES key, size and committed SHA-256), then each `.enc` file exactly as the browser uploaded it |
| **Processing intake** | read whole (2 GiB cap), decode, full-payload receipts holding the token and the model | decrypts chunk by chunk straight to disk and checks each file's size and SHA-256 against the commitment. Busy → 409 before any file bytes are read; not enough disk → 507. The receipt is redacted (no token, keys or file bytes). |
| **Peak memory (measured)** | ≈ 6–8×S | **Buffer 14 MB, Processing 44 MB** for a 1.5 GiB model, end to end with real encryption; independent of size |
| **Long uploads** | cut off after 10 minutes (server `ReadTimeout`) | no total limit; dropped only after **5 minutes with no bytes**, also over HTTP/2 |

**Decisions made with it:**
- **v3 only.** The Processing TEE no longer accepts v2 (415), so both TEEs ship together.
- **No file-size caps in the TEEs.** The model/weights 4 GiB, adaptor 1 MiB and 16,384-chunk limits are gone; free disk is checked instead (507). **Memory caps stay:** metadata part 64 KiB, chunk header 8 KiB, chunk 64 MiB, `input_spec.json` 64 KiB. **The UI will cap uploads** (UI13).
- **The Buffer checks structure only** (no decryption, no hashing). A corrupt or mismatched upload is caught by the Processing TEE and the job fails as **`PayloadRejected`** with the TEE's reason. Busy, network errors and stalls are retried on the next tick.
- **No durable queue yet.** A Buffer restart still loses the queue and the in-memory keys; affected jobs must be resubmitted (2.9, BT11).
- **`tee-restart-policy=OnFailure`** set on `test-cpu-c3` on 22 Oct. Confidential Space reads it at boot, so it's active from the rollout restart on. A crashed container then comes back in about a minute instead of staying down.

### Rollout status (22 Oct)

| Piece | State |
|---|---|
| Buffer image | ✅ pushed: `buffer-tee:stream-20261022` = **`sha256:8da7b6ba37f6f064b5f995d8e18f456586ac23a48f1821467f8be722355694b1`** |
| Processing CPU image | ✅ pushed: `cpu-cs:stream-20261022` (commit `fb81152`) = **`sha256:7023f89a272fd32e092aa7df76cddda82ed216769c9ec04dfc2af5b0b82e9819`** |
| Processing GPU image | **not rebuilt** (unused while `MAX_GPU_PROVISION_ATTEMPTS=0`). It only speaks v2: rebuild it before turning the GPU path on. |
| UI `lib/` | changed, **not committed** (UI12): `EXPECTED_IMAGE_DIGEST` → `8da7b6ba…`, the IV-carry fix, and the encrypted body built as a `Blob` |
| VMs | ✅ **switched 22 Oct, 11:52 UTC.** `cpu-tdx` → `cpu-cs@sha256:7023f89a…` (stopped; starts with it on the next job). `test-cpu-c3` → `buffer-tee@sha256:8da7b6ba…`, `CPU_CS_IMAGE_DIGEST=sha256:7023f89a…`, restarted. Boot log: launcher `RestartPolicy:OnFailure`, `buffer-tee: listening on :8443` at 11:53:54; `https://bodh.tanuh.ai/tee/healthz` 200. The restart dropped `job-f56f2685eaf6` (CT Fracture, still uploading). Previous settings: `scratchpad/rollout/*.before-stream.json`. |
| Processing `det-20261023b` (23 Oct) | ✅ **live.** Commits `5cc7f40` (detection bucket renamed to `bounding_box`, all UI detection metrics, `dcm_path` ground-truth column) + `90eceaa` (`infer.py` loads ONNX models from their path: models with external weights over 2 GB now run; peak memory about 2x the model instead of 3x). Built on `ba-build-vm-us` (`p3dx-depa-sandbox`). Image **`cpu-cs@sha256:a57a70b023e152f48a7b84b9fd9eef526141055a39e4d149194185606a6cca42`**; `cpu-tdx` and the Buffer's `CPU_CS_IMAGE_DIGEST` point at it; Buffer restarted 11:10 UTC. Evaluator `gs://tanuh-evaluators/bounding_box/evaluate.py` uploaded. Buffer disk 30 → 100 GB (partition 95 GB). Tested on the image: 4 GiB transfer end to end (Buffer 15 MB, Processing 45 MB peak) and inference on a 4 GiB model (5 s, 8.2 GB peak for 16 inputs). Previous VM settings: `scratchpad/rollout/*.before-det.json`. |

**Steps:**
1. Drain the queue.
2. `cpu-tdx`: `tee-image-reference` → `…/tanuh-processing-tee/cpu-cs@sha256:7023f89a…`.
3. `test-cpu-c3`: `CPU_CS_IMAGE_DIGEST` → `sha256:7023f89a…`; `tee-image-reference` → `…/tanuh-buffer-tee/buffer-tee@sha256:8da7b6ba…`.
4. Restart `test-cpu-c3`.
5. Deploy the UI **in the same window**. The new UI accepts only the new Buffer digest.
6. Run OCS end to end, then one large model.

### UI bug found: the IV carry

`lib/index.ts` advanced each chunk's IV with `for (j…) { if (++iv[j] !== 0) break }`. On a `Uint8Array`, `++iv[j]` evaluates to 256 while storing 0, so **the carry never happens.** When a chunk's IV low byte passes `ff`, that chunk's IV is wrong and the TEE can't decrypt it.
- **Who is affected:** model uploads larger than one 16 MB chunk. A file of K chunks hits a wrap with probability about K/256, so **a 2 GB model (128 chunks) failed about half the time**.
- **Fix:** in UI `lib/` (`index.ts` and `hpke.ts`, UI12). The dataset uploader was already correct.
- **Proof:** the fixed loop reproduces the shared golden test vector byte for byte; the old loop doesn't.

### New datasets

Two new catalogue entries, **CT Fracture** and **Renal Health**, plus three abandoned onboardings. Details are in the 1.5 table; neither new dataset can run yet (tasks CA5–CA7, PT15, PT17, PT18, D6, D7).

## 1.1 Scope

**In scope:**
- A model owner submits an **ONNX** model (`.onnx`, plus an optional `.onnx.data`) and an **`adaptor.py`** through the UI.
- The Buffer TEE accepts it and hands it to the Processing TEE.
- The Processing TEE runs the model on the encrypted dataset, runs the adaptor and the bucket evaluator, and posts the metrics to the leaderboard.
- The UI shows the result.
- Datasets: **OCS** (binary, `c7c117e8…`) and **BCD** (4-class, `a4f792ad…`).

**Out of scope (Part 2):**
- a preprocessing script or settings
- the one-function or default adaptor
- TorchScript and Hugging Face in the UI
- new buckets, and a data-card overhaul
- production hardening

## 1.2 Components

| Component | Repo / location | What it does in this phase |
|---|---|---|
| **UI** | `ui-dx/apps/TANUH` | Data providers onboard datasets (folder upload, encrypted). Model owners submit an ONNX model + adaptor and see the result. |
| **Buffer TEE** | `Buffer_TEE/Tanuh-buffer-tee-ratls/enclave` | Receives the submission, checks each file against its committed hash, starts the Processing VM, and dispatches the job. |
| **Processing TEE** | `Processing_TEE` | Decrypts the dataset, runs the model, adaptor and evaluator, and posts the result. |
| **Catalogue** | `bodh.tanuh.ai/controlplane/iudx/v2/cat/item` | Holds each dataset's bucket, class names, required metrics and Croissant metadata (1.11). |
| **Leaderboard** | `tanuh-leaderboards-apis` at `bodh.tanuh.ai/leaderboard` | Stores and ranks results; checks the metrics against the catalogue. |
| **GCS** | `gs://file-server-data/<dataset-uuid>/` (datasets), `gs://tanuh-evaluators/<bucket>/evaluate.py` (evaluators) | storage |
| **Cloud KMS** | `keyRings/tanuh/cryptoKeys/dataset-dek-wrap` | unlocks each dataset file's key |

## 1.3 How a job runs

### What each party provides

**The data provider** (once, through the UI's onboarding):
- **Data files:** encrypted one per file, each with a `<name>.manifest.json`.
  - **Where:** in the dataset folder or in subfolders at any depth (a folder upload keeps each file's path, e.g. `Suspicious/a.jpg`).
  - **Types:** detected from content: DICOM, JPEG, PNG, TIFF, BMP, WebP, GIF.
- **Ground truth:** one table **in the dataset folder itself** (not a subfolder), uploaded through its own picker and encrypted the same way.
  - **Format:** CSV/TSV/JSON/JSON Lines, with flexible column names and labels as class indexes or class names.
  - **Ids:** a file's path, file name, name without extension, or a folder holding one file.
  - **Unmatched rows:** skipped. The job fails only if fewer than half the rows match.
- **Catalogue fields** (read by the Processing TEE at job time):

  | Field | Example | Written by |
  |---|---|---|
  | `problemStatement` | `binary_classification` | UI metadata form (the TEE also accepts the older `task_type`) |
  | `class_names` | `["Non-Suspicious","Suspicious"]` (position = label number; for binary, index 1 = positive class) | **Today:** the "Class names" field in the metadata step (comma-separated, optional; create flow only; PR #843, not deployed yet). The two live entries were set through the API (CA2). **Planned:** a Classes section in the Croissant builder that also writes the Croissant document (UI7, 1.11). |
  | `datasetMetrics`, `primaryMetric` | required metrics; the one used to rank | UI metadata form; metric names are picked from the list the bucket's evaluator produces (UI8) |

**The model owner** (per submission, in the UI):

| Upload | What | Required |
|---|---|---|
| model | `.onnx`, with the model's preprocessing inside (1.7) | yes |
| weights | `.onnx.data` (external weights, for big models) | only if the model has one |
| adaptor | `adaptor.py` (contract below) | yes |

### The flow

1. **Submit (UI → Buffer).** `POST /v1/submit`, HPKE-encrypted: `{dataset_id: "<catalogue uuid>", model_format: "onnx", model_sha256, adaptor_sha256, weights_sha256?}`. This creates a job in `pending_upload`.
2. **Upload (UI → Buffer).** `PUT /v1/upload/{job_id}/{model|weights|adaptor|input_spec}`, chunk-encrypted (AES-256-GCM, 16 MB chunks).
   - **Stored encrypted**, exactly as uploaded (`artifacts/<slot>.enc`). The Buffer checks only that every announced chunk arrived and nothing else; the file's AES key is unwrapped and kept in memory. A `plaintext_sha256` in the stream header must equal the committed hash.
   - **No size limits in the TEEs** (the UI caps uploads, UI13); not enough disk → 507.
   - **Queue:** the job becomes `queued` when every committed file has arrived.
3. **Dispatch (Buffer → Processing).** The Buffer:
   - checks every file and its key are still there (else the job fails before a VM starts)
   - starts `cpu-tdx` if it's stopped
   - checks its attestation, including that the image digest equals `CPU_CS_IMAGE_DIGEST`
   - streams **payload v3** (`POST /api/load-model`): metadata with each file's key, size and hash, then each encrypted file

   The job becomes `dispatched`. A refusal the Processing TEE would repeat (bad upload, version mismatch, no disk) fails the job as `PayloadRejected`.
4. **Prepare (Processing, Go).**
   0. **Receive:** decrypt each file chunk by chunk to disk and check its size and SHA-256 against the commitment.
   1. **Read the catalogue item:** bucket, `class_names`, required metrics.
   2. **Fetch the evaluator:** `gs://tanuh-evaluators/<bucket>/evaluate.py`.
   3. **Check and place the model:** it must really be ONNX.
   4. **Install the adaptor's imports:** anything the image lacks (`dep_scanner.py`).
   5. **Decrypt the ground truth first, then match:** its rows are matched to the dataset's files, and only matched files are decrypted, 16 at a time, into a temporary folder.
   6. **Write the standard inputs:** a canonical `ground_truth.csv` (`file` = path in the dataset, `label` = class index) and `inputs.txt`.
5. **Run three stages (Processing, Python).**

   | Stage | Command | Output |
   |---|---|---|
   | 1. Inference (ours) | `infer.py --format onnx --model-dir … --inputs inputs.txt --output-dir raw/` | `raw/raw_outputs.npz` (`ids` + one array per ONNX output) and `raw/meta.json` |
   | 2. Adaptor (owner's) | `python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/` | `predictions/predictions.csv` |
   | 3. Evaluator (ours) | `evaluate.py --predictions … --ground-truth … --spec … --results results.json` | `results.json` |

   Before the model sees an image, inference does:
   1. decode to RGB (DICOM: middle frame, MONOCHROME1 inverted, min-max to 0–255, grey → 3 channels)
   2. resize to the model's size (bilinear, or bicubic if the model's `tanuh.resize` metadata says so)
   3. scale to 0–1
   4. arrange the model's layout

   **Nothing else,** so any other preprocessing must be inside the model.
6. **Report.** The TEE:
   - renames metrics to the catalogue's keys (e.g. `f2` → `f2_score`)
   - checks their ranges
   - posts **exactly** the required keys to the leaderboard (`POST /submit-solution`)
   - tells the Buffer the job finished (`complete` / `error`)
   - deletes the decrypted data

### Adaptor contract (current phase)

`adaptor.py` is a script run as `python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/`.

- **Reads:**
  - `raw/raw_outputs.npz`: `ids` (each image's path in the dataset) and one array per ONNX output, named as in the model
  - `dataset_spec.json`: `task_type`, `class_names`, `num_classes`
- **Writes:** `predictions/predictions.csv` with one row per image:

  | Bucket | Columns |
  |---|---|
  | `binary_classification` | `file,label,score` (`score` = probability of `class_names[1]`) |
  | `multiclass_classification` | `file,label,prob_0,…,prob_{C-1}` (probabilities in `class_names` order) |

- **The evaluator is lenient:**
  - any delimiter, and common column names
  - ids as path, file name or name without extension
  - label optional
  - logits accepted (sigmoid/softmax applied)
  - extra rows ignored

  It's still **strict on coverage:** every ground-truth image needs exactly one prediction.
- **Templates:** `tools/reference/adaptor_sigmoid_binary.py` (one logit → sigmoid) and `tools/reference/adaptor_softmax_multiclass.py` (C logits → softmax). Most models can use one unchanged.
- **Environment:** Python 3 with numpy, pandas, scipy, scikit-learn, torch and OpenCV; other imports are installed from PyPI. Time limit 10 minutes.

### Error codes (sent to the leaderboard)

| Code | Meaning | Examples |
|---|---|---|
| 1 | Data or pre/post-processing problem | image can't be decoded; adaptor crashed or wrote bad predictions |
| 2 | Model problem | not ONNX; fails to load or run |
| 3 | Platform or runtime problem | timeout, evaluator error, required metric missing or out of range |

## 1.4 What's built and tested

- **Buffer:** format-aware upload slots. Payload v2 is live; **payload v3** (streaming, uploads stored encrypted) is built on branch `streaming-changes` (BT9). Tests pass.
- **Processing (Go):**
  - catalogue reader (bucket from `problemStatement` or `task_type`)
  - evaluator fetch; model checks
  - dataset planning at any folder depth; ground truth first, then only the matched files
  - flexible ground-truth reading and matching; the three stages; metric mapping; error codes

  All tests pass.
- **Python:** `infer.py`, both evaluators (lenient predictions), reference adaptors, wrap tools, `check_model.py`. All 47 tests pass.
- **UI `lib/`:** the new submit body, adaptor upload, status polling and leaderboard read. It type-checks together with the current component's calls.
- **Real-data results:**

  | Case | Result |
  |---|---|
  | OCS, wrapped ONNX | identical to the old flow: accuracy 0.992126, F2 0.996377 |
  | BCD, wrapped ONNX | identical to the old flow: QWK 0.978758, AUC 0.999699 |
  | Both, provider's original ground truth and folders | same numbers (OCS `Unique_case_ID,Label` 135 rows, 8 skipped; BCD `folder_id,label` with `A–D`) |
  | Both, messy predictions | same numbers (other column names, bare file names, semicolons, logits) |
  | Live GCS listing, OCS `c7c117e8` | 127 of 127 images matched as uploaded |

## 1.5 The deployment today (`proj-tanuh-benchmark-ptfm`, checked 1 Oct 2026, 17:30 UTC; datasets re-checked 22 Oct)

**The machines below still run the 1 Oct images.** The streaming release replaces both TEE images; see the rollout in 1.0.

### Machines

| | Buffer TEE | Processing TEE |
|---|---|---|
| VM | `test-cpu-c3`: TDX, `c3-standard-4`, asia-south1-b, 10.50.2.34 | `cpu-tdx`: TDX, `c3-standard-4`, asia-south1-b, 10.50.2.35. **Stops itself after 5 minutes idle;** the Buffer starts it for each job. |
| Boot image | `confidential-space-debug-260600` (debug, fine for now) | `confidential-space-debug-260600` |
| Service account | `cs-buffer-tee@proj-tanuh-benchmark-ptfm` | `tanuh-processing-tee@proj-tanuh-benchmark-ptfm` |
| Container image | `asia-south1-docker.pkg.dev/proj-tanuh-benchmark-ptfm/tanuh-buffer-tee/buffer-tee@sha256:acd31e73a763a0ddd56ce9be89f8d2cb562fd6ebf96a2e75b3e604e5d7177ad0` (tag `flex-20260930`) | `asia-south1-docker.pkg.dev/proj-tanuh-benchmark-ptfm/tanuh-processing-tee/cpu-cs@sha256:aa65ad1837d7368cfd6326dfa2bed6b862ced907a6c1fcbd48728e6abd25077d` (tag `det-20261001`, commit `882d8a9`; CPU onnxruntime, 15.5 GB). Classification, segmentation and object detection; attested policy hash `71d0682d…` |
| Env that matters | `PROJECT=proj-tanuh-benchmark-ptfm`, `CPU_CS_INSTANCE=cpu-tdx`, `CPU_CS_ADDR=10.50.2.35:443`, `CPU_CS_IMAGE_DIGEST=sha256:aa65ad18…`, `MAX_GPU_PROVISION_ATTEMPTS=0` (CPU only), `BUFFER_CALLBACK_BASE=https://bodh.tanuh.ai/tee`, `KEYCLOAK_ISSUER=https://bodh.tanuh.ai/auth/realms/tanuh`, `KEYCLOAK_JWKS_URL=…/protocol/openid-connect/certs` (moved to `bodh.tanuh.ai` on 1 Oct, BT8) | `DATASETS_BUCKET=file-server-data`, `CATALOGUE_BASE_URL=https://bodh.tanuh.ai`, `LEADERBOARD_SUBMIT_URL=https://bodh.tanuh.ai/leaderboard/submit-solution` (1 Oct, PT7), `GCP_PROJECT_ID`, `PROJECT`, `ZONE`, `INSTANCE`. **Every env variable must be in the image's `tee.launch_policy.allow_env_override` label, or Confidential Space refuses to start the container.** |
| Logs | `gcloud compute instances get-serial-port-output test-cpu-c3 --zone=asia-south1-b --project=proj-tanuh-benchmark-ptfm` | the same, for `cpu-tdx` |

### Access in place

| Needed by | Granted by |
|---|---|
| Processing: read `file-server-data` and `tanuh-evaluators` | project-level `roles/storage.objectViewer` |
| Processing: pull its image | `roles/artifactregistry.reader` on `tanuh-processing-tee` |
| Processing: unwrap dataset keys | `roles/cloudkms.cryptoKeyDecrypter` on `dataset-dek-wrap` |
| Processing: secrets | project-level `roles/secretmanager.secretAccessor` |
| Processing: reach GCS / KMS / Secret Manager | Private Google Access on the subnet |
| Processing: reach `bodh.tanuh.ai` (catalogue, leaderboard, Buffer callback at `/tee`) | Cloud NAT `nat-tanuh` (confirmed by the 1 Oct job) |
| Buffer: start/stop `cpu-tdx` | `compute.instanceAdmin.v1`, `iam.serviceAccountUser` |
| Buffer: pull its image | `roles/artifactregistry.reader` on `tanuh-buffer-tee` |

### Datasets in `gs://file-server-data`

| Dataset | Folder | Contents | Catalogue entry (checked 1 Oct, 17:30 UTC; new rows 22 Oct) |
|---|---|---|---|
| **CT Fracture** (new) | `887a1be8-cc32-4ed9-b477-eeaa13602a2c` (uploaded 8–19 Oct) | **8,490 DICOM slices** under `v2/` (1.88 GB): about **12%** of the card's 69,486 slices in 195 cases. 5 objects without a manifest (3 `.init`, 2 with no extension): most likely interrupted uploads. `croissant_public.json`. **No ground truth file.** Nothing uploaded since 19 Oct. **Not ready.** | `binary_classification`, modality "DICOM CT (axial, bone kernel)". **No `class_names`.** Requires `auc` (primary), `sensitivity` (secondary), `specificity`, `ppv`, `npv`, `accuracy`, `f1`: all producible. **But a sample is a case: a 3-D series of 141–651 slices, with one score per case.** The platform scores one file per sample today (PT18). |
| **Renal Health** (new) | `e423d178-d295-42c2-b34f-276d4c0e57aa` (uploaded 6 Oct) | 200 ultrasound DICOMs: 120 in `automated_labels_benchmarking/`, 80 in `manual_annotation_finalized/` (plus 1 CSV there), and **`benchmark_ground_truth.csv`** at the root. | `problemStatement: "bounding_box"` (the TEE's bucket is `object_detection`: PT15). **No `class_names`** (one class). Requires 10 metrics: `map_50_95` (primary), **`efficiency_fps_gmacs`** (secondary), `giou_matched_map_50_95`, `ap_50`, `ap_75`, `mean_iou`, `detection_rate`, `precision_50`, `recall_50`, `f1_50`. **Mapped today:** `map_50_95`, `ap_50`, `ap_75`, `mean_iou`, `recall_50`. **Not:** `giou_matched_map_50_95` (alias missing), `detection_rate`, `precision_50`, `f1_50` (not computed), `efficiency_fps_gmacs` (not computed; 1.9 q1). **Every job would fail today** (PT15, PT17, CA6). |
| none | `6e6dbd6b-…`, `a54c7d50-…`, `a66be990-…` (8 Oct) | only `croissant.json` (and `DATA_CARD_2.docx` in two) | **no catalogue entry** (404): abandoned onboardings (CA7) |
| OCS | `c7c117e8-7ed4-4dab-bcca-7d21bb2e2a7d` | 127 JPEGs in `Suspicious/` and `Non-Suspicious/`, `desktop.ini` (skipped), and the provider's original `ocs_ground_truth.csv`. **Ready.** | ✅ `binary_classification`; `class_names` `["Non-Suspicious","Suspicious"]`; requires `f2_score` (primary), `sensitivity_recall` (secondary), `specificity_true_negative_rate`, `ppv_positive_predictive_value`, `npv_negative_predictive_value`; no efficiency metric |
| BCD | **`12ff1e0d-b32a-40e1-9868-5fd1f89ed30f`** (re-onboarded 1 Oct, 13:53 UTC) | **All 240** DICOMs under `mammogram_benchmark/subject_*/`, and the **corrected** `ground_truth.csv`: its manifest's `plaintext_sha256` `108a32b7…` (6,252 bytes) equals `Data/ground_truth/BCD/ground_truth.csv`. **Ready** (D1, D2 done). | ✅ `multiclass_classification`; `class_names` `["A","B","C","D"]`; requires `macro_f1` (primary), `qwk_quadratic_weight_kappa` (secondary, `"-1-1"`), `accuracy`, `macro_f2`, `weighted_f2`, `sensitivity_recall`, `specificity_true_negative_rate`, `ppv_positive_predictive_value`, `npv_negative_predictive_value`, `auc_area_under_curve`; no efficiency metric |

- **The first BCD entry, `a4f792ad-…`, was deleted** on 1 Oct, 11:55 UTC (controlplane `DELETE`, from the UI), together with its folder in `gs://file-server-data`.
- **The catalogue fixes on 1 Oct were made two ways:**
  - **OCS:** through the controlplane API: `class_names`, the Croissant `classes` record set, and efficiency set to `"not evaluated"`. Later, in Kibana, efficiency was removed and the secondary metric changed.
  - **New BCD entry:** in **Kibana Dev Tools**, with scripted `_update` requests on index `tanuh__cat`. They removed `efficiency_fps_gmacs` (metric and label), set QWK to `"-1-1"` and changed the secondary metric. A scripted update can delete keys; the controlplane's `PUT` merges and cannot.
- **Card vs data mismatch, still open:** the new BCD card says 200 cases (A 56 / B 59 / C 60 / D 25). The files and the ground truth have 240 (60 per class), matching the provider's spreadsheet.

### First real job (1 Oct 2026, `job-cbe9b2e5646d`)

Submitted from the deployed UI: BCD (`a4f792ad…`), ONNX model 166 KB + weights 85 MB + adaptor 2.3 KB.

| Time (UTC) | Step | Result |
|---|---|---|
| 10:13:54–10:14:17 | submit + 3 uploads to the Buffer, then `queued` | ✅ Keycloak auth with the `bodh.tanuh.ai` issuer works (BT8) |
| 10:14:18–10:17:22 | Buffer starts `cpu-tdx` (CPU fallback); the VM boots, pulls the image, verifies its policy | ✅ about 3 minutes from start to listening |
| 10:17:26–10:17:28 | dispatch: RA-TLS check of `cpu-tdx` (TDX, digest `82a9e4b9…`), payload v2 delivered | ✅ C4 confirmed: the Buffer accepts the Processing TEE's attestation |
| 10:17:33 | Processing reads the catalogue at `bodh.tanuh.ai` (PT7) | ✅ bucket `multiclass_classification` read from `problemStatement` (PT6) |
| 10:17:33 | | ❌ **`catalogue: the dataset needs at least 2 class_names, has 0`** (CA2). It stopped before downloading any data. |
| 10:17:35 | failure report to the leaderboard | ❌ **500 `failed to store submission`.** The leaderboard's catalogue check passed, so its deployed catalogue URL works (LB5). The Keycloak name lookup failed (Keycloak was restarting), but that is only logged. **The 500 came from the database insert:** a failed submission carries no `onnx_runtime_providers`, and that column is `NOT NULL` (likely cause; the DB error isn't logged). The failed result was not stored (LB6). |
| 10:17:38–10:17:40 | callback to the Buffer (`bodh.tanuh.ai/tee`), then `cpu-tdx` stops itself | ✅ the job shows `error` in the Buffer |

**Take-aways:**
- Everything up to the catalogue read works end to end on the new domain.
- With CA2 fixed, this job would next have failed on `efficiency_fps_gmacs` (CA1), after the full inference run. BCD itself isn't ready either: only 197 of 240 files are uploaded, and its `ground_truth.csv` doesn't match the provider's label files (D1, D2). **Run OCS first.**

### All runs on 1 Oct 2026

Each run taught us something; the fix is in the last column. Times are UTC. "Unwrapped" means the provider's original model, without its preprocessing (1.7).

| Time | Job | Dataset, model | Result | What it showed, and the fix |
|---|---|---|---|---|
| 10:13 | `job-cbe9b2e5646d` | BCD (old entry), unwrapped | ❌ catalogue | No `class_names` → CA2 |
| 10:40 | `job-7f5ab3b2ee0c` | BCD (old entry), unwrapped | ❌ catalogue 401 | The submitter's Keycloak token (~5 min) expired while the VM booted; the TEE sent it to the catalogue → **PT14** (retry without it) |
| 13:00 | `job-d3771bafb26c` | OCS, unwrapped | ✅ scored, ❌ leaderboard 500 | The live DB's `dataset_id` was still `BIGINT` → **LB7** |
| 13:14 | `job-f99aa5bc2c88` | OCS, unwrapped | ✅ scored, ❌ leaderboard 500 | Leftover `accuracy NOT NULL` column → **LB8** |
| 13:25 | `job-37f0c5120999` | OCS, unwrapped | ✅ **leaderboard 201** | **The first full end-to-end run.** Scores are the unwrapped model's (F2 0.8586) |
| 14:40 | `job-180cf8926189` | BCD (old entry), unwrapped | ❌ catalogue 404 | The old BCD entry had been deleted; the UI still sent its id → the UI now sends the catalogue UUID (PR #845) |
| 14:57 | `job-ea2a7a05d14b` | BCD (new entry), unwrapped | ❌ `MetricMappingError` after inference | The new entry required `efficiency_fps_gmacs` → removed in Kibana. The failed result was stored (201), which confirms LB6 |
| 15:05 | `job-537856a013a1` | BCD (new entry), unwrapped | ✅ **leaderboard 201** | **The first full BCD run.** Macro F1 0.1073, QWK −0.0012: the negative QWK is why its range must be `"-1-1"` |
| 15:13 | `job-8318d26c4a53` | OCS (model not checked) | ✅ succeeded | — |
| 16:20 | `job-9ab8bd85a4b0` | OCS, **wrapped** | ✅ **leaderboard 201** | **The first valid OCS benchmark** (expected F2 0.9964). PT14 saved it: the token had expired after a 3-minute weights upload |
| 16:44–16:56 | 4 jobs | BCD, wrapped | ❌ never queued | The 85 MB weights upload failed (408 from the ingress after 130–195 s, at ~100–160 KB/s on a slow connection) → **UI11** |
| 17:06 | `job-6d227b40e673` | BCD (new entry), unwrapped | ✅ leaderboard 201 | The same scores as 15:05 (unwrapped again); PT14 needed again |

**Still to do:** a BCD run with **`wrapped-models/bcd/`** (`model.onnx` 169,387 bytes). Expected: macro F1 0.9709, QWK 0.9788, accuracy 0.9708. The entries from unwrapped models should not be selected for the public leaderboard.

**Timing of a run:**
- **VM start:** about 3–3.5 minutes on a cold start.
- **OCS:** about 26 s (127 JPEGs).
- **BCD:** about 2 minutes (240 DICOMs, 3 GB). Decryption takes 12 s; most of the rest is CPU inference.

### Rollout log (30 Sep – 1 Oct 2026)

| Change | Details |
|---|---|
| New registry | `tanuh-buffer-tee` (asia-south1). `tanuh-processing-tee` already existed. |
| Images | built locally and pushed: `buffer-tee:flex-20260930`, `cpu-cs:flex-20260930`, then `cpu-cs:flex-20260930b` (bucket from `problemStatement`) |
| Registry read access | added for both service accounts (neither could pull before) |
| `cpu-tdx` | removed `tee-env-EVAL_SCRIPTS_BUCKET`; set `tee-env-DATASETS_BUCKET=file-server-data`; image → `cpu-cs@sha256:82a9e4b9…`; restarted |
| `test-cpu-c3` | image → `buffer-tee@sha256:acd31e73…`; `CPU_CS_IMAGE_DIGEST` → `sha256:82a9e4b9…`; removed two expired `ssh-keys`; restarted. The first restart dropped 3 old-format queued jobs, as agreed. |
| Evaluators | `gs://tanuh-evaluators` (versioned) with both evaluators; hashes checked |
| Verified after restart | **Buffer:** new digest booted, TDX attestation, listening on :8443. **Processing:** new image pulled, policy hash `40e9b4dc…` verified in its attestation token, listening on :443. |
| Domain move (1 Oct, BT8 / PT7) | `test-cpu-c3`: `KEYCLOAK_ISSUER`, `KEYCLOAK_JWKS_URL` and `BUFFER_CALLBACK_BASE` → `https://bodh.tanuh.ai…`; restarted. `cpu-tdx`: `CATALOGUE_BASE_URL` and `LEADERBOARD_SUBMIT_URL` → `https://bodh.tanuh.ai…`; started to check it boots. Both images allow these variables, so no rebuild was needed. |
| Segmentation image (1 Oct, 13:58) | `cpu-cs:seg-20261001` = `sha256:a0dc1d13…`: segmentation (PT12), catalogue token retry (PT14), PT8. Rolled out to `cpu-tdx`, plus `CPU_CS_IMAGE_DIGEST` on `test-cpu-c3` (restarted). Boot check: policy hash **`71d0682d…`** verified. Evaluator `gs://tanuh-evaluators/segmentation/evaluate.py` (generation `1790862969513516`) |
| Detection image (1 Oct, 15:28) | `cpu-cs:det-20261001` = **`sha256:aa65ad18…`** (commit `882d8a9`): adds object detection (PT13). Same rollout and boot check; all 31 `infer.py` tests also pass inside the image. Evaluator `gs://tanuh-evaluators/object_detection/evaluate.py` (generation `1790868419257043`) |
| Leaderboard (1 Oct) | `2ebf51a` (LB6, LB5), `c751b68` (LB7), `02a1a0e` (LB8); each deployed, last at 13:24:55 UTC (pod `leaderboard-api-757c9498b-hnn5j`) |

**Previous values, for rollback:**

| VM | Setting | Previous value |
|---|---|---|
| `test-cpu-c3` | `tee-image-reference` | `us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/buffer-tee@sha256:3506b9b291e8fbab882afd3c5d21e4fca0823be40e3086e848bbae242834ccad` |
| `test-cpu-c3` | `CPU_CS_IMAGE_DIGEST` | `sha256:10eb1e5a448713f19664cceaf3e51c6a537a04eeece0d8fc7bb3471ce4fd56dc`, then `sha256:c3501be25fbb2df707d87d98a4ece99ddaf1ae837baf19a0ec8b83c2d3ce321b` |
| `cpu-tdx` | `tee-image-reference` | `…/tanuh-processing-tee/us-central1-docker.pkg.dev/p3dx-depa-sandbox/ratls/cpu-cs:44f54d9_data_encryption_dev`, then `cpu-cs@sha256:c3501be2…` (tag `flex-20260930`) |
| `cpu-tdx` | `DATASETS_BUCKET` / `EVAL_SCRIPTS_BUCKET` | `tanuh-datasets-prod` / `tanuh-datasets-prod` |
| `test-cpu-c3` | `KEYCLOAK_ISSUER` / `KEYCLOAK_JWKS_URL` / `BUFFER_CALLBACK_BASE`, before 1 Oct | `https://benchmark.tanuh.ai/auth/realms/tanuh` / `…/protocol/openid-connect/certs` / `https://benchmark.tanuh.ai/tee` |
| `cpu-tdx` | `CATALOGUE_BASE_URL` / `LEADERBOARD_SUBMIT_URL`, before 1 Oct | not set (the image defaults, `https://benchmark.tanuh.ai…`) |
| `cpu-tdx` / `test-cpu-c3` | Processing image / `CPU_CS_IMAGE_DIGEST`, before 1 Oct 13:58 | `cpu-cs@sha256:82a9e4b9…` (`flex-20260930b`); then `a0dc1d13…` (`seg-20261001`) until 15:28. Full metadata saved in `scratchpad/rollout/*.before-{seg,det}.json` |

## 1.6 Status and the first-run plan

**Where things stand (1 Oct 2026, 17:30 UTC): Part 1 works end to end for both datasets.**
- **Done:**
  - **Every part of the flow works on live data:** submit, upload, VM start, attested dispatch, catalogue read, decryption, inference, adaptor, evaluator, leaderboard (201), and the callback. This holds for both OCS and BCD (1.5, "All runs on 1 Oct").
  - **The first valid OCS benchmark:** `job-9ab8bd85a4b0` (16:20 UTC), with `wrapped-models/ocs/`.
  - **Catalogue:** both entries are clean (CA1–CA3). BCD was re-onboarded as `12ff1e0d-…` with all 240 DICOMs and the corrected ground truth (D1, D2).
  - **Leaderboard:** LB5–LB8 fixed and deployed.
  - **Processing image `det-20261001`:**
    - segmentation (PT12) and object detection (PT13)
    - the catalogue token retry (PT14), which has already saved three runs
    - PT8: defaults and policy on `bodh.tanuh.ai`
  - **UI:**
    - every URL uses `bodh.tanuh.ai` (UI6)
    - submits send the dataset's catalogue UUID (PR #845)
    - deployed at 15:56 UTC
  - **Models wrapped and checked (D4):** `wrapped-models/{ocs,bcd}/`.
- **Left for Part 1:**
  1. **BCD with the wrapped model.** Every BCD run so far used the unwrapped one. Upload `wrapped-models/bcd/` from a connection fast enough for the 85 MB weights (UI11).
  2. **Select the wrapped OCS entry** (`job-9ab8bd85a4b0`) for the public leaderboard, not the unwrapped ones.
  3. **Not blocking:** UI2, UI3, UI10, UI11, PT9, and the BCD card's 200-vs-240 count.

**Order:**
1. **OCS** ✅ done (`job-9ab8bd85a4b0`).
   - **Upload used:** `wrapped-models/ocs/model.onnx` + `model.onnx.data` + `adaptor.py` (= `adaptor_sigmoid_binary.py`).
   - **Expected leaderboard metrics:** F2 0.9964, sensitivity 1.0, specificity 0.9861, PPV 0.9821, NPV 1.0; 127 images scored, 8 ground-truth rows skipped.
2. **BCD prerequisites** ✅ done: the dataset is re-onboarded with all files and the corrected ground truth.
3. **BCD:** Upload `wrapped-models/bcd/` (the wrapped ResNet34 + `adaptor_softmax_multiclass.py`). **Not the original model:** every BCD job on 1 Oct uploaded the provider's unwrapped `resnet34_density_trained.onnx` (166,230 bytes; the wrapped one is 169,387), which scores macro F1 0.1073. **Expected** (240 images): accuracy 0.9708, macro F1 0.9709, QWK 0.9788, AUC 0.9997.

**Following a job:**
- **Buffer:** `GET /v1/status/{job_id}` gives `job.status`: `pending_upload` → `queued` → `dispatched` → `complete` / `error`. On error, `job.completion_error` holds the details.
- **Processing log:** the `cpu-tdx` serial console shows the stages, the ground-truth match summary and the metrics.
  - **The serial console is empty once `cpu-tdx` stops** (it stops itself after each job). Read the same lines from Cloud Logging instead:

    ```bash
    gcloud logging read 'resource.type="gce_instance" AND timestamp>="2026-10-01T10:17:00Z"' \
      --project=proj-tanuh-benchmark-ptfm --order=asc --limit=300 \
      --format='value(timestamp,jsonPayload.MESSAGE,textPayload)'
    ```
- **Uploads:** the ingress logs each upload's status, bytes received and duration (`namespace_name="ingress-nginx"`, `jsonPayload.request=~"/tee/v1/upload"`). A **408** with a partial byte count means the upload was too slow (UI11).
- **Leaderboard log:** `gcloud logging read 'resource.type="k8s_container" AND resource.labels.namespace_name="leaderboards"' …` (container `leaderboard-api`).
- **Leaderboard:** `GET my-submissions?dataset_id=<uuid>` returns the entry.

## 1.7 Instructions for model owners (current phase: ONNX + adaptor)

This is the text for the UI's submit page. The full reference is in [model-submission.md](model-submission.md).

1. **Your model receives:**
   - **Colours and values:** RGB images, values **0–1**.
   - **Size:** resized to your model's input size (bilinear, or bicubic if your ONNX metadata sets `tanuh.resize=bicubic`).
   - **Layout:** `[N, 3, H, W]`, or the layout your model declares.
2. **Build every other preprocessing step from your training code into the model**, such as ImageNet mean/std, BGR order or colour balancing. Otherwise nothing fails but your score drops (the OCS model fell from accuracy 0.99 to 0.76).
   - Exporting from PyTorch: put the steps in a small wrapper module in front of your network, then `torch.onnx.export` it with a fixed input size and a dynamic batch.
   - Already have an ONNX file that only needs ImageNet normalisation: `python3 tools/reference/wrap_onnx.py --mode imagenet --src model.onnx --dst model_tanuh.onnx`.
3. **Write `adaptor.py`** (contract in 1.3): read your model's output, write `predictions.csv`. Start from `adaptor_sigmoid_binary.py` (binary) or `adaptor_softmax_multiclass.py` (multi-class). Keep your output columns in the dataset's **class order**.
4. **Check locally** (with numpy<2, onnxruntime 1.19.2):

   ```bash
   python3 tools/check_model.py --format onnx --model model.onnx [--weights model.onnx.data] --image sample.jpg
   ```

   Compare the printed output with what your training code gives for the same image. **If they differ, your preprocessing is wrong or missing.**
5. **Upload:** `model.onnx`, `model.onnx.data` (if you have one), `adaptor.py`.

| Common mistake | Result | Fix |
|---|---|---|
| Normalisation left out of the model | score silently lower | build it into the model |
| Dividing by 255 inside the model | inputs ≈ 0, score silently lower | don't: the platform already gives 0–1 |
| Trained on BGR (`cv2.imread`), model expects RGB | score silently lower | flip the channels inside the model |
| Dynamic height/width in the ONNX input | job fails (code 2) | export with a fixed size |
| Adaptor's class order differs from `class_names` | score silently lower | reorder in the adaptor |
| Adaptor misses an image | job fails (code 1) | one prediction per image |

## 1.8 Current-phase tasks by component

Priority: **B** = blocks the first run; **F** = needed for a usable first run.

### Catalogue (owner of `cat/item`)

The Processing TEE reads these fields at job time; they aren't sent with the job.

| # | P | Task | Done when |
|---|---|---|---|
| CA1 | B | ✅ **Done 1 Oct** (controlplane API, `catalogue-update/apply.py`): `datasetMetrics.efficiency_fps_gmacs` set to `"not evaluated"` on both entries. **Why a string, not a deletion:** the controlplane **merges** nested objects on `PUT cat/item`, so a key left out of the body survives. `null` doesn't work either, because the leaderboard counts JSON `null` as a required number. A non-range string is ignored as a metric by both the TEE and the leaderboard. **Where it came from:** not the code; a provider typed it as a free-text custom metric, which UI8 no longer allows. | ✅ the required metrics are now OCS 5, BCD 10, all producible |
| CA2 | B | ✅ **Done 1 Oct** (same script): top-level `class_names`, OCS `["Non-Suspicious","Suspicious"]` and BCD `["A","B","C","D"]`, plus a `classes` record set in each entry's Croissant document (1.11). | ✅ `GET cat/item` returns them in order |
| CA3 | B (BCD) | ✅ **Done 1 Oct:** BCD `qwk_quadratic_weight_kappa` = `"-1-1"` (the leaderboard and the TEE read it as −1…1). **The UI still can't write this itself:** the form writes ranges to `customMetricRanges`, which nothing reads (2.7). | ✅ |
| CA5 | B (CT Fracture) | **CT Fracture `class_names`.** The entry has none, so every job fails at the catalogue check. Set them in label order, positive class second, e.g. `["No fracture","Fracture"]` (the ground truth's labels decide the exact values). | a CT Fracture job passes the catalogue check |
| CA6 | B (Renal) | **Renal Health:**<br>• `class_names`: one class, e.g. `["kidney"]`<br>• **remove `efficiency_fps_gmacs`** (decided 23 Oct: not a platform metric, the TEE doesn't compute it) and pick another secondary metric, e.g. `ap_50`. It got there through the UI bug in UI8: the entry was saved on 8 Oct, 10:05 IST, minutes before the UI fix (`339591fb`).<br>• `problemStatement`: stays `bounding_box` (PT15) | the entry's required metrics are all producible |
| CA7 | — | **Three abandoned onboardings** in `gs://file-server-data` (`6e6dbd6b`, `a54c7d50`, `a66be990`, 8 Oct): only `croissant.json` / `DATA_CARD_2.docx`, no catalogue entry. Delete after their owner confirms. | the folders are gone |
| CA4 | F | **A Modify-asset save keeps the fixes (checked in code, `ca9aaaaa`).**<br>• **Why:** the controlplane merges, so `class_names`, `"not evaluated"` and `"-1-1"` survive a save that leaves them out, and the edit form never sends them. It lists only numeric metrics, so the QWK range and the placeholders don't appear in it.<br>• **The trap:** if an editor re-adds "QWK" as a metric, the form writes `0` (= range 0–1) over `"-1-1"`.<br>• **Not done yet:** the edit form doesn't read back or write `class_names`; only the create flow writes it (UI7). | a re-save of either entry leaves `class_names`, the QWK range and the placeholders unchanged |

**Metric keys the evaluators can produce.** The UI's metric check (UI8, `ALLOWED_CUSTOM_METRIC_NAMES`) accepts exactly these. The aliases are from `internal/leaderboard/metrics.go`.

| Evaluator name | Also accepted as a catalogue key | binary | multiclass |
|---|---|:-:|:-:|
| `accuracy` | | ✓ | ✓ |
| `sensitivity` | `sensitivity_recall`, `recall`, `tpr`, `true_positive_rate` | ✓ | ✓ (macro) |
| `specificity` | `specificity_true_negative_rate`, `tnr`, `true_negative_rate` | ✓ | ✓ |
| `ppv` | `ppv_positive_predictive_value`, `precision`, `positive_predictive_value` | ✓ | ✓ (macro) |
| `npv` | `npv_negative_predictive_value`, `negative_predictive_value` | ✓ | ✓ |
| `auc` | `auc_area_under_curve`, `roc_auc`, `auroc` | ✓ | ✓ (one-vs-rest macro) |
| `f1` / `f2` | `f1_score`, `f1score` / `f2_score`, `f2score` | ✓ | — |
| `fnr` / `fpr` | `fnr_false_negative_rate`, `false_negative_rate` / `fpr_false_positive_rate`, `false_positive_rate` | ✓ | — |
| `macro_f1`, `weighted_f1`, `macro_f2`, `weighted_f2` | | — | ✓ |
| `qwk` | `qwk_quadratic_weight_kappa`, `qwk_quadratic_weighted_kappa`, `quadratic_weighted_kappa` | — | ✓ |

`confusion_matrix` also passes through, as a matrix.

**Anything else declared as a bare number or `"lo-hi"` string fails every job.** That includes a metric the other bucket produces, e.g. `qwk` on a binary dataset.

### UI (`ui-dx/apps/TANUH/src`)

**Latest checked: `origin/development` at `8d36d2f5`** (1 Oct, PR #845). The live site was redeployed at 15:56 UTC; it uses the new worker bundle, with the Buffer digest `acd31e73…`.

- **PR #844** (Amarthya03):
  - leaderboard metric legend
  - **`segmentation` problem type**
  - segmentation metrics allowed (`dice`, `iou`, `global_dice`, `global_iou`, `pixel_accuracy`)
  - **but `efficiency`, `efficiency_fps_gmacs` added back to the allowed metrics** (UI8)
- **PR #845** (Amarthya03):
  - **`bounding_box` problem type** (UI9)
  - **submit sends the selected dataset's catalogue UUID:** `LEGACY_ONBOARD_DATASET_UUIDS` is removed, so re-onboarded datasets work without a code change

Merged earlier (after PR #840):
- **PR #842** (tanishdhankhar):
  - metric display names (`customMetricLabels`, `resolveMetricKeyLabel`)
  - a Croissant JSON uploaded in Modify asset replaces the stored one
  - metadata-step layout
- **PR #843** (Amarthya03):
  - the submit page's dataset tab is now a select dropdown, with better error handling
  - the Modify-asset refactor
  - a class-names field and the metric check (UI7/UI8, partly)
  - `environment.prod.ts` `bufferTeeUrl` → `https://bodh.tanuh.ai/tee`

The 15:56 UTC deploy includes these. The 17:06 BCD run reached the new entry `12ff1e0d-…`, so the UUID change is live.

**Already in the repo before that** (`0f3f2917`):
- folder uploads that keep each file's path (`getFilePath()`, unchanged by the 1 Oct file-tree refactor)
- the ground truth uploaded separately to the dataset root
- the bucket written as `problemStatement`
- leaderboard pages keyed by catalogue UUID

**Submit changes, 1 Oct** (commit `38014482` "submit model library fix", plus later merges; **deployed** to `bodh.tanuh.ai` at 09:07 UTC; the deployed worker bundle contains the new Buffer digest):
- **Error messages:** attest, submit and upload errors now carry the server's own message. The form shows it, e.g. the Buffer's validation text or the TEE's `error_message`, instead of fixed per-step text.
- **Session refresh:** new `renewSession` option in `submitOnnxModel`. The form re-attests the Buffer after hashing, because hashing a large model can outlive the session. The Keycloak token is refreshed to keep at least 2 minutes of validity per TEE request.
- **Dataset matching:** the form accepts catalogue ids and labels, mapping anything containing "breast" / "oral" to the legacy ids `1` / `2`. It still depends on `LEGACY_ONBOARD_DATASET_UUIDS`.
- **Adaptor field:** still labelled **"Preprocessing script"**, now **required** when shown, but still shown **only for Oral Cancer** (or the embedded single-dataset page). See UI2.
- **Weights file:** still **required** by the form (`modelWeightsFile`), although `lib/` and the TEE allow an ONNX model without `.onnx.data`. Both first-run models have one, so it doesn't block this phase.
- **Field note:** the submit form now says that the ground-truth template should use the file and label format.

**`lib/` changes (made 30 Sep; committed and deployed 1 Oct):**
- **`constants.ts`:**
  - `EXPECTED_IMAGE_DIGEST = sha256:acd31e73…` (the live Buffer)
  - `LEGACY_ONBOARD_DATASET_UUIDS`: a temporary map from the form's `1` (Breast Cancer) → `a4f792ad…` and `2` (Oral Cancer) → `c7c117e8…`
  - `LEADERBOARD_API_BASE`
  - `REQUIRE_NON_DEBUG_TEE = false`
- **`transfer-onnx.ts`:**
  - `submitOnnxModel` sends the new body and uploads the adaptor to `/adaptor`. It accepts the existing `pythonFile` as the adaptor and refuses a missing adaptor.
  - `fetchJobStatus` polls the Buffer's status.
  - `fetchJobResults` returns the leaderboard entry, or throws the TEE's error.
- **`index.ts` / `types.ts`:** the session exposes its token for the leaderboard read.
- **`oidc.ts`, `worker*.ts`:** the strict-verifier switch, kept off.

| # | P | Task | Where | Done when |
|---|---|---|---|---|
| UI1 | B | ✅ **Done 1 Oct:** `lib/` committed (`38014482`) and deployed to `bodh.tanuh.ai` (09:07 UTC). A local build needs `npm ci --legacy-peer-deps` (the lockfile needs it: `zone.js` 0.16 vs Angular 20.0's optional peer `~0.15`). The 1 Oct BCD submit from the live UI reached `queued`. | `apps/TANUH/src/lib/*` | ✅ |
| UI2 | F | **Show the adaptor field for every dataset.** **Still open in `8d36d2f5`:** `showPythonUploadField` is still `mapTextToOnboardDatasetId(…) === 2` (Oral Cancer only), and the label is still "Preprocessing script". Show it for every dataset, relabel it **adaptor.py**, and pass it as `adaptorFile`. **No longer blocking:** every 1 Oct BCD submit carried an adaptor, through the embedded single-dataset page, where `embedOnboardOnly` makes the field required. | `showPythonUploadField` / `requiresPythonFile`, validators and label in `reference-datasets.component.ts` (around line 1201) / `.html` (line 536) | a BCD submit from the reference-datasets page reaches `queued` |
| UI3 | F | **Show the real result.** **Still open in `8d36d2f5`:** the success toast shows `Accuracy: …` from `result.accuracy`, which is `NaN` for OCS (its metrics don't include accuracy). Show `result.metrics`, highlighting the dataset's primary metric. | result toast in `reference-datasets.component.ts` (around line 1908) | the user sees the dataset's metrics, or the TEE's error |
| UI6 | B | ✅ **Done and deployed 1 Oct** (vijay's PRs #839 and #840 on `development`, `9d8ce105`). The live bundle calls `bodh.tanuh.ai` for `/tee`, `/leaderboard`, `/auth` and the challenge evaluator, and contains no `benchmark.tanuh.ai`. The 1 Oct job reached the Buffer through it.<br>**Leftovers, not blocking:**<br>• the challenge-evaluator default in `leadership-api.service.ts` is `http://` (it redirects to https)<br>• the Croissant `tanuh` vocabulary name is now `https://bodh.tanuh.ai/schema/1.0#`; the validator checks only that the key exists, so older documents still pass | — | ✅ |
| UI7 | F | **Partly done in PR #843 (deployed): class names.** PR #844 also saves them for `segmentation`.<br>• **Landed:** a free-text **"Class names"** field (comma-separated, in label order; optional) in the metadata details. On **create**, a classification dataset with 2 or more names gets top-level `class_names` (`parseClassNames`, `CLASSIFICATION_PROBLEM_STATEMENTS`).<br>• **Missing:**<br>  – the field isn't required, so a new dataset can still be created without `class_names` and fail every job, as on 1 Oct<br>  – binary isn't held to exactly 2 names, and the positive class isn't explained<br>  – the edit flows don't read or write `class_names`<br>  – no `classes` record set in the Croissant document<br>  – no per-class "value in the labels file"<br>• The full spec is in `instructions.md` §1. | metadata step; `tanuh-dataset-item-payload.util.ts`; `manage-contribution(s).component.ts`; `croissant.util.ts` | a new classification dataset can't be created without ordered `class_names` |
| UI8 | **B** | **Only producible metrics (PR #843, deployed).** ⚠️ **PR #844 put `'efficiency', 'efficiency_fps_gmacs'` back on `ALLOWED_CUSTOM_METRIC_NAMES`** (`tanuh-dataset-item-payload.util.ts:107`), and its leaderboard legend calls efficiency the secondary metric of every dataset. **Any dataset that declares it fails every job after inference**, which is exactly what happened to the new BCD entry at 14:57 UTC on 1 Oct. Remove both until the TEE computes efficiency (2.5).<br>• **Landed:** `evaluatorMetricNameValidator` rejects a custom metric whose slug isn't in `ALLOWED_CUSTOM_METRIC_NAMES` (the table above) in both the create and edit forms ("Not produced by the evaluator"). The primary/secondary dropdowns offer only allowed names.<br>• **Missing:**<br>  – the list isn't per bucket, so `qwk` is still accepted on a binary dataset<br>  – metric ranges are still bare `0` (2.7)<br>  – removing a stored metric needs the `"not evaluated"` placeholder, because the controlplane merges (`instructions.md` §2.2). It doesn't matter for the two live entries, which were fixed directly (CA1). | metadata / preview-metadata steps; `tanuh-dataset-item-payload.util.ts` | no new entry can require a metric its bucket doesn't produce |
| UI9 | F | **Bounding boxes (PR #845) don't match the TEE yet.**<br>• **Problem type name:** the UI stores `problemStatement: "bounding_box"`; the TEE's bucket is `object_detection` (PT13). **Fix (either):** UI value `object_detection` (the label can stay "Bounding box"), or **PT15**, the TEE accepting `bounding_box` as another name.<br>• **Class names:** saved only when there are at least 2 (`parseClassNames(…).length >= 2`, line 1086). A one-class detection dataset (e.g. `kidney`) gets none, and every job fails. **Allow 1 for detection**, and add the detection value to `CLASSIFICATION_PROBLEM_STATEMENTS`.<br>• **Metrics:** add `map`, `ap50`, `ap75`, `map_giou`, `recall_50`, `mean_iou` and the legend's slugs `map_0_5_0_95`, `ap_0_5`, `ap_0_75`, `giou_matched_map_0_5_0_95`, `recall_0_5` to `ALLOWED_CUSTOM_METRIC_NAMES`. | `hackathon-datasets.data.ts`, `contribution-setup.component.ts`, `tanuh-dataset-item-payload.util.ts` | a one-class detection dataset can be onboarded and its job is scored |
| UI10 | F | **Stale `main.js` after a deploy.** `main.js` has no content hash in its name but is served with `cache-control: public, max-age=31536000, immutable`. After the 15:56 UTC deploy, a browser still holding the old `main.js` requested `worker-4ZCPCQLC.js` (404, replaced by `worker-CSS7BFN2.js`), and "Verifying TEE" failed with "Worker error: unknown". A hard refresh fixes it for one user. **Fix:** serve `index.html` and `main.js` with `Cache-Control: no-cache`, or make sure `main.js` is hashed too (`outputHashing: "all"` is already set in `apps/TANUH/project.json`, so check the deploy step). | web server / ingress config for the UI; build output | a returning browser loads the new build without a hard refresh |
| UI11 | F | **Large uploads time out on slow connections.** The upload sends each encrypted file in **one** `PUT`. On 1 Oct, between 16:44 and 16:56 UTC, four 85 MB weights uploads failed with **408** from the ingress (nginx) after 130–195 s, at 100–160 KB/s. The same file had taken about 10 s at 14:57. **Fix (either):** raise the ingress body timeout for `/tee` (`nginx.ingress.kubernetes.io/proxy-body-size`, `client-body-timeout`, `proxy-read-timeout`); or upload in separate chunk requests with retries. The latter needs a Buffer endpoint that accepts chunks. | ingress annotations for the Buffer route; `lib/index.ts` `uploadEncrypted` | an 85 MB upload completes at 100 KB/s |
| UI11b | F | **(22 Oct) The Buffer side of UI11 is fixed** in the streaming release: no 10-minute server limit, uploads are dropped only after 5 minutes with no bytes. **The ingress is still open:** the 1 Oct 408s came from nginx. For `/tee`, set `nginx.ingress.kubernetes.io/proxy-request-buffering: "off"` (stream the body through instead of buffering it to disk first), `proxy-body-size: "0"`, and `client-body-timeout` / `proxy-read-timeout` / `proxy-send-timeout` of at least 300 s, to match the Buffer's idle limit. | ingress annotations for the Buffer route | a 2 GB upload at 1 MB/s completes |
| UI12 | **B** (rollout) | **(22 Oct) `lib/` changes for the streaming release: made, not committed.**<br>• `constants.ts`: `EXPECTED_IMAGE_DIGEST` → `sha256:8da7b6ba…` (`buffer-tee:stream-20261022`)<br>• `index.ts` `uploadEncrypted`: the **IV-carry fix** (1.0), and the encrypted body built as a `Blob` one chunk at a time, instead of one array the size of the file (a 2 GB model needed 2 GB of page memory). Still one `PUT`.<br>• `hpke.ts` `uploadChunked`: the same IV fix<br>• `transfer-onnx.ts`: header comment for the new flow<br>Type-check: 0 errors. The loop reproduces the golden vector. **Deploy in the same window as the Buffer restart**: this UI accepts only the new Buffer. | `apps/TANUH/src/lib/{constants,index,hpke,transfer-onnx}.ts` | committed, and deployed with the Buffer |
| UI13 | F | **File-size caps** per upload slot, now that the TEEs have none (1.0). Values to be decided: model, weights, adaptor, `input_spec`, and dataset files. | submit and onboarding forms | an oversized file is refused before hashing |
| UI4 | F | **Encryption flag:** `encryptedDatasetUpload` is `true` in dev/local and `false` in stg/prod. Uploads without it have no manifest and are ignored by the TEE. **The live site is a development build with `encryptedDatasetUpload: true`** (checked in the bundle on 1 Oct), so uploads there (D1, D2) are encrypted. Needed only if a stg or prod build is deployed. | `environments/environment.{stg,prod}.ts` | uploads have manifests |
| UI5 | — | ✅ `EXPECTED_IMAGE_DIGEST` set (30 Sep). **Update it after every Buffer rebuild.** | `lib/constants.ts` | attestation passes |

**Buffer API contract, for reference:**
- `POST /v1/submit` `{dataset_id, model_format: "onnx", model_sha256, adaptor_sha256, weights_sha256?}`.
- `PUT /v1/upload/{job_id}/{model|weights|adaptor}`. From the streaming release on: 400 = the stream is incomplete or malformed (or the header hash isn't the committed one), 409 = another upload to the same slot is running, 507 = not enough disk on the Buffer.
- `GET /v1/status/{job_id}` → `{"status": "success", "job": {"status": …, "completion_error"?: {error_code, error_type, error_message}}}`.

### Buffer TEE (`Buffer_TEE/Tanuh-buffer-tee-ratls/enclave`)

| # | P | Task | Done when |
|---|---|---|---|
| BT1 | B | ✅ **Done 30 Sep:** queue emptied (3 old-format jobs dropped). | no old jobs |
| BT2 | B | ✅ **Done 30 Sep:** image built and rolled out to `test-cpu-c3` with `CPU_CS_IMAGE_DIGEST=sha256:82a9e4b9…`. | boot log: new digest, TDX attestation, listening on :8443 |
| BT8 | B | ✅ **Applied 1 Oct** (restarted). **Domain move (1.10):** on `test-cpu-c3`, set `tee-env-KEYCLOAK_ISSUER=https://bodh.tanuh.ai/auth/realms/tanuh`, `tee-env-KEYCLOAK_JWKS_URL=https://bodh.tanuh.ai/auth/realms/tanuh/protocol/openid-connect/certs` and `tee-env-BUFFER_CALLBACK_BASE=https://bodh.tanuh.ai/tee`, then restart. All three are env settings the image allows, so no rebuild is needed. **Why:** the Buffer compares each token's `iss` with `KEYCLOAK_ISSUER` as an exact string (`internal/server/auth.go`), and fetches signing keys from `KEYCLOAK_JWKS_URL`. Both still point at the old host. | a login from the UI passes the Buffer's auth, and the callback URL in payloads uses `bodh.tanuh.ai` |

| BT9 | **B** | **Streaming release (22 Oct, branch `streaming-changes`, `c190ac5`):** uploads stored encrypted with a structure-only check, keys in memory only, payload v3 streamed through `io.Pipe`, a 5-minute no-progress watchdog, permanent refusals fail the job, no file-size caps, deadlines that move over HTTP/2, `GOMEMLIMIT=1GiB`. 9 test packages pass; end to end, 1.5 GiB at 14 MB peak. **Image pushed** (`8da7b6ba…`); **rollout pending** (1.0). | boot log on `test-cpu-c3` shows the new digest; an OCS job completes |
| BT10 | **B** | ✅ **Set 22 Oct:** `tee-restart-policy=OnFailure` on `test-cpu-c3`. **Active from the next boot** (Confidential Space reads it at launch). | after the rollout restart, a killed container comes back by itself |
| BT11 | F | **Durable queue** (deferred): job records and encrypted uploads in GCS, AES keys wrapped by Cloud KMS, so a restart resumes the queue instead of losing it. Ideally the KMS key is bound to the Buffer's attestation (Workload Identity Federation). Design notes in 2.9. | a Buffer restart keeps queued jobs |

**Otherwise nothing is open for this phase.** After the rollout, every Processing image change needs `CPU_CS_IMAGE_DIGEST` updated and a restart, as before.

### Processing TEE (`Processing_TEE`)

| # | P | Task | Done when |
|---|---|---|---|
| PT1 | B | ✅ **Done 30 Sep:** CPU image built and rolled out to `cpu-tdx`. | boot log: policy verified, listening on :443 |
| PT6 | B | ✅ **Done 30 Sep:** bucket read from `problemStatement` (fallback `task_type`); `cpu-cs:flex-20260930b` = `sha256:82a9e4b9…`. | the live entries read as `binary_classification` / `multiclass_classification` |
| PT7 | B | ✅ **Applied 1 Oct** (VM started once to check). **Domain move (1.10):** on `cpu-tdx`, set `tee-env-CATALOGUE_BASE_URL=https://bodh.tanuh.ai` and `tee-env-LEADERBOARD_SUBMIT_URL=https://bodh.tanuh.ai/leaderboard/submit-solution`. Both are allowed env settings, so no rebuild is needed; the Buffer starts the VM with the new settings on the next job. **Why:** the image's built-in defaults are `https://benchmark.tanuh.ai…` (`internal/config/config.go`), and `cpu-tdx` sets no URL itself. The Buffer callback URL comes from the Buffer (BT8). | a job's catalogue read and leaderboard submit succeed |
| PT8 | F | ✅ **Done 1 Oct** (`seg-20261001` and later). Defaults in `internal/config/config.go` and the attested `policy/network_policy.json` name `bodh.tanuh.ai` (rule 5) and `bodh.tanuh.ai/tee` (rule 6, the Buffer callback; this was BT6). New policy hash **`71d0682d…`** (`policy_test.go`); nothing else pins the old one. | `internal/config/config.go`, `policy/network_policy.json`, `internal/policy/policy_test.go` | ✅ boot log: `PASS: Policy hash verified` with `71d0682d…` |
| PT12 | — | ✅ **Done 1 Oct: segmentation bucket** (`segmentation`; image `seg-20261001`, commit `882d8a9`).<br>• `groundtruth.MatchMasks`: the label column names each image's mask file; masks are resolved first, and images only against the other files.<br>• Masks are decrypted but kept out of inference; the evaluator gets `file,mask`.<br>• Evaluator `evaluators/segmentation/evaluate.py`: `dice`, `iou`, `global_dice`, `global_iou`, `sensitivity`, `ppv`, `specificity`, `pixel_accuracy`, per class.<br>• Reference adaptor `tools/reference/adaptor_argmax_segmentation.py`.<br>• Data-provider rules: 2.5. | `internal/groundtruth`, `internal/pipeline`, `evaluators/segmentation/` | ✅ unit tests; synthetic end-to-end run, Dice 0.9685 |
| PT13 | — | ✅ **Done 1 Oct: object detection bucket** (`object_detection`; image `det-20261001`, commit `882d8a9`).<br>• `groundtruth.ReadBoxes` / `MatchBoxes`: box CSV/TSV (`xyxy` or `xywh`), JSON records or COCO JSON; images without objects; one-class datasets allowed.<br>• `infer.py --per-sample`: one image at a time; variable-length outputs stored as `<name>`, `<name>__shape`, `<name>__offset`, plus `orig_hw` and `input_hw` for box scaling. The model must take batch size 1 or a dynamic batch.<br>• Evaluator `evaluators/object_detection/evaluate.py`: COCO-style `map`, `ap50`, `ap75`, `map_giou`, `recall_50`, `mean_iou`, per-class AP. **mAP matches `pycocotools` exactly.**<br>• Reference adaptor `tools/reference/adaptor_boxes_detection.py` (NMS'd `[K,6]`, or boxes/scores/labels). | `internal/groundtruth/boxes.go`, `tools/infer/infer.py`, `evaluators/object_detection/` | ✅ unit tests; synthetic end-to-end run, mAP 1.0 across 4 image sizes |
| PT14 | — | ✅ **Done 1 Oct: catalogue read survives an expired token.** On a 401 with the submitter's token, `catalogue.FetchSpec` retries once without it: an open entry loads, and a restricted one still fails. **Why:** the token lives about 5 minutes, while a cold VM start takes about 3, plus the upload time. It saved the 16:20 and 17:06 runs. | `internal/catalogue/catalogue.go` | ✅ in production |
| PT15 | ✅ (code, 23 Oct) | **Detection bucket renamed to the UI's `bounding_box`** (was `object_detection`): `catalogue.Validate`, `pipeline.bucketKinds`, the evaluator folder `evaluators/bounding_box/` and its `task_type`. No alias: the TEE uses the UI's value. **To ship:** upload `evaluators/bounding_box/evaluate.py` to `gs://tanuh-evaluators/bounding_box/evaluate.py`, rebuild and roll out the Processing image. | `internal/catalogue`, `internal/pipeline`, `evaluators/bounding_box/` | a `bounding_box` dataset is scored as detection |
| PT16 | **B** | **Streaming release (22 Oct, branch `streaming-changes`, `7e01d3a`):** payload v3 intake (`internal/pipeline/intake.go`), streaming chunk decrypt (`internal/crypto/chunked.go`, golden vector shared with the Buffer), `modelpkg.Place` on file paths, typed `Request`, redacted receipt, no file-size caps, a 5-minute idle limit, `GOMEMLIMIT=1GiB`. 9 test packages pass; a 2 GiB payload received at 44 MB peak. **Image built** (`cpu-cs:stream-20261022`); **rollout pending** (1.0). | boot log on `cpu-tdx`: policy verified; an OCS job completes |
| PT17 | ✅ (code, 23 Oct) | **All 17 UI detection metrics** (`BOUNDING_BOX_METRICS` in the UI) are produced or mapped: new `precision_50/75`, `recall_75`, `f1_50/75` (all detections, no score threshold), `median_iou`, `p25_iou`, `mean_giou` (-1 for a missed box), `detection_rate` (share of images with objects that get at least one box), `avg_boxes_per_image`; aliases for every UI key (e.g. `map_50_95_giou`, `mean_iou_all_images`). Ground truth: `dcm_path` / `dicom_path` / `case` are recognised file columns (Renal's CSV names its image by `dcm_path`). **Not computed:** `efficiency_fps_gmacs` (CA6). | `evaluators/bounding_box/evaluate.py`, `internal/leaderboard/metrics.go`, `internal/groundtruth` | Renal's 9 producible metrics are all mapped |
| PT18 | **B (CT)** | **3-D volume samples (CT Fracture):** a sample is a **folder of slices** (one CT series), with one score per case. Design in 2.5. | `internal/groundtruth`, `internal/pipeline`, `tools/infer/infer.py` | a CT case is read as one `[D,H,W]` volume and scored once |
| PT19 | F | **Large datasets:**<br>• stream the dataset decrypt (`DecryptDatasetV1` holds each file in memory; fine for images, not for multi-GB files)<br>• size `cpu-tdx`'s disk for the largest dataset; today the job decrypts the whole dataset to disk<br>Design in 2.11. | `internal/crypto/dataset_v1.go`, `internal/pipeline/dataset.go`; VM disk | a 50 GB dataset is scored without running out of memory or disk |

### Cloud setup (`proj-tanuh-benchmark-ptfm`)

| # | P | Task | Done when |
|---|---|---|---|
| C1 | B | ✅ **Done 30 Sep:** removed `tee-env-EVAL_SCRIPTS_BUCKET` from `cpu-tdx` (the new image's launch policy doesn't allow it). | — |
| C2 | B | ✅ **Done 30 Sep:** `tee-env-DATASETS_BUCKET=file-server-data` on `cpu-tdx`. | — |
| C3 | B | ✅ **Done 30 Sep:** `gs://tanuh-evaluators` with both evaluators. | — |
| C4 | F | ✅ **Rolled out 30 Sep**, with registry read access for both accounts. **Confirmed 1 Oct** by `job-cbe9b2e5646d`: the Buffer started `cpu-tdx`, and its RA-TLS check passed (TDX, digest `82a9e4b9…`). | ✅ |

### Leaderboard (`tanuh-leaderboards-apis`)

| # | P | Task | Done when |
|---|---|---|---|
| LB1 | F | ✅ **Confirmed 1 Oct:** the deployed version validated the 1 Oct submission's UUID `dataset_id` against the catalogue (`ValidateDataset`). The README's integer `dataset_id` and top-level `accuracy` are out of date (LB4). | ✅ |
| LB5 | B | ✅ **Done 1 Oct.** **Deployed setting:** worked already; the 1 Oct submission passed `ValidateDataset`. **Code default:** `https://bodh.tanuh.ai/controlplane/iudx/v2` in `internal/config/config.go`, `.env.example` and `docker-compose.yml` (commit `2ebf51a`). | ✅ |
| LB6 | B | ✅ **Fixed 1 Oct** (commit `2ebf51a`, Amarthya Ravi).<br>• **The bug:** a failed submission got 500 "failed to store submission". A failed body has no `onnx_runtime_providers`; Go's nil slice is sent as NULL, and the column is `TEXT[] NOT NULL`. So every failed result was lost. The Keycloak name lookup failure on 1 Oct is only logged and wasn't the cause.<br>• **The fix:** `normalizeSubmission` defaults the field to `[]`; DB insert errors are logged; tests in `internal/api/handlers_test.go`.<br>• **Deployed:** a new `leaderboard-api` pod started at 11:40 UTC, 4 minutes after the commit. | ✅ the 14:57 UTC failed BCD run was stored (201) |
| LB7 | B | ✅ **Fixed 1 Oct** (commit `c751b68`). The live `submissions` table was created by the first commit, when `dataset_id` was `BIGINT`. `CREATE TABLE IF NOT EXISTS` never changes an existing table, so every UUID insert failed (`invalid input syntax for type bigint`). Now `ALTER COLUMN dataset_id TYPE TEXT` on `submissions` and `leaderboard_selections`. | ✅ inserts reach the next check |
| LB8 | B | ✅ **Fixed 1 Oct** (commit `02a1a0e`). Leftover `NOT NULL` columns from the first schema: `accuracy` (never written now), and `num_samples`, `elapsed_seconds`, `model_sha256` (empty for failed runs). Now dropped to nullable (guarded for a fresh database), and `job_id` changed to `TEXT`. **Lesson:** compare the live table with the code's schema (`\d submissions`) after any schema change. | ✅ the 13:29 UTC run stored (201) |

### Data providers and the first-run models

| # | P | Task | Done when |
|---|---|---|---|
| D1 | B (BCD) | ✅ **Done 1 Oct:** BCD was re-onboarded as `12ff1e0d-…` with **all 240** DICOMs. | ✅ the 15:05 and 17:06 runs decrypted and scored 240 of 240 |
| D2 | B (BCD) | ✅ **Done 1 Oct:** the re-onboarded `ground_truth.csv` is `Data/ground_truth/BCD/ground_truth.csv`, confirmed by the manifest's `plaintext_sha256`. | ✅ "240 dicom samples … matched by file_name: 240" |
| D5 | F | **The BCD card doesn't match the data:** `volume`, `description` and the class distribution say 200 cases (A 56 / B 59 / C 60 / D 25); the files and the ground truth have 240 (60 per class). Correct the card, or confirm which 200 belong in the benchmark. | catalogue entry `12ff1e0d-…` | card and data agree |
| D6 | **B (CT)** | **CT Fracture upload is about 12% done** (8,490 of 69,486 slices; nothing since 19 Oct) and has **no ground truth**. Finish it with the command-line uploader (2.11) rather than the browser. Then upload the ground truth, one row per **case**: the case id (= the case's folder) and its label. Remove the 5 objects without manifests. | every case's slices and the ground truth are in the folder, each with a manifest |
| D7 | F (Renal) | **Renal Health ground truth:** check `benchmark_ground_truth.csv` against what `groundtruth.ReadBoxes` reads: one row per box (`case,xmin,ymin,xmax,ymax`) with the case matching a file's path, name or folder. The card says coordinates are half-open (`xmax` exclusive), which matches how the evaluator computes areas. Also, which CSV is the benchmark: the root file, or the one in `manual_annotation_finalized/`? | a dry run matches 200 of 200 images |
| D3 | — | OCS: nothing to do; the original ground truth works as uploaded. | — |
| D4 | F | ✅ **Done 1 Oct.** Both models wrapped into `/home/prathmesh/Desktop/tanuh-backup/wrapped-models/` with `wrap_onnx.py` (`--mode ocs` / `--mode imagenet`). Each folder has `model.onnx` + `model.onnx.data` + `adaptor.py` (the matching reference adaptor), ready to upload.<br>• `check_model.py` on a real image: **PASS** (OCS 3×256×256 bicubic, output `(N,1)`; BCD 3×224×224 bilinear, output `(N,4)`).<br>• **Full local run** (`infer.py` → adaptor → evaluator, onnxruntime 1.19.2, numpy 1.26.4): **OCS** 127 images, F2 0.996377, accuracy 0.992126; **BCD** 240 images, macro F1 0.970923, QWK 0.978758, AUC 0.999699. All **exactly** the expected values.<br>• sha256: OCS `model.onnx` `1ebd0429…`, BCD `model.onnx` `dfd29447…`. | ✅ |

## 1.9 Current-phase open questions

1. ✅ **`efficiency_fps_gmacs`** (23 Oct): **dropped**, not a platform metric; removed from Renal Health (CA6). The UI's leaderboard legend still calls it every dataset's secondary metric (`provider.component.ts`), which should go too (UI8). Earlier: on 1 Oct it was decided that it isn't a leaderboard metric, and it was removed from both entries (CA1). **But** the UI's PR #844 allows it again and its legend calls it every dataset's secondary metric (UI8). **Renal Health (new) declares it as its secondary metric**, so every Renal job would fail on it. Is efficiency wanted? If so, it needs a definition and TEE support (2.5) before any dataset may declare it.
2. ✅ **BCD ground truth:** the deleted entry's file no longer matters. The re-onboarded BCD uses the corrected ground truth (D2).
3. ✅ **Catalogue keeps top-level `class_names`:** yes, both on create (the new BCD entry) and through `PUT` and Kibana updates (CA4).
4. ✅ **Detection's problem type name:** `bounding_box`, the UI's value, everywhere (decided 23 Oct; PT15).
5. **BCD size:** 200 cases (card) or 240 (data)? See D5.
6. **Domain:** is `bodh.tanuh.ai` the permanent host? Everything now uses it, including the attested policy (PT8).
7. **CT Fracture:** is a 3-D case model (one score per series) the intended task, or would per-slice models also be accepted? This decides the input contract of PT18.
8. **Upload caps (UI13):** what limits per slot, now that the TEEs have none?
9. **Large datasets (2.11):** is a command-line uploader acceptable for data providers, and who hosts the signed-URL endpoint?

## 1.10 Domain move: `benchmark.tanuh.ai` → `bodh.tanuh.ai` (found 1 Oct 2026)

**What changed:**

| Host | DNS | TLS certificate | Paths (`/`, `/controlplane/…/cat/item`, `/tee/healthz`, `/auth/realms/tanuh/.well-known/openid-configuration`) |
|---|---|---|---|
| `benchmark.tanuh.ai` | `8.231.81.37` (unchanged) | the ingress controller's placeholder ("Kubernetes Ingress Controller Fake Certificate", `ingress.local`) | **every path: 404** (the ingress's default page) — the ingress has no rule for this host |
| `bodh.tanuh.ai` | `8.231.81.37` (same ingress) | Let's Encrypt, `CN=bodh.tanuh.ai` | catalogue 200 (the same OCS entry), Buffer `/tee/healthz` 200, Keycloak discovery 200, UI 302 |

The UI repo's "tanuh → bodh" commits on 1 Oct changed branding and contact emails only; **every URL in the code still says `benchmark.tanuh.ai`.**

**Why Keycloak's issuer matters:**
- **What the issuer is:** Keycloak writes its own address into each token's `iss` field. The discovery document at the new host says `issuer: https://bodh.tanuh.ai/auth/realms/tanuh`, so users logging in there get tokens with that issuer.
- **Why the Buffer rejects them:** it compares `iss` with its `KEYCLOAK_ISSUER` setting as an exact string, and that setting is still `…benchmark.tanuh.ai…`. So every authenticated request fails with "JWT issuer mismatch".
- **Signature checks:** the Buffer fetches Keycloak's signing keys from `KEYCLOAK_JWKS_URL`, also on the old host. Once its hourly key cache expires, it can't verify any token at all.
- **Still to confirm:** decode one real token from a browser session (the middle part of the JWT, base64) to see its `iss` value.

**What still points at the old host, and the task for each:**

| Where | Setting | Today | Task |
|---|---|---|---|
| UI | `ONBOARD_MODEL_TEE_URL`, `LEADERBOARD_API_BASE`, leaderboard service, `environments/*.ts` | ✅ **`bodh.tanuh.ai`, deployed 1 Oct** (PRs #839, #840) | UI6 ✅ |
| Buffer VM `test-cpu-c3` | `KEYCLOAK_ISSUER`, `KEYCLOAK_JWKS_URL`, `BUFFER_CALLBACK_BASE` | ✅ **`https://bodh.tanuh.ai…` since 1 Oct** (was `benchmark.tanuh.ai`) | BT8 ✅ |
| Processing VM `cpu-tdx` | `CATALOGUE_BASE_URL`, `LEADERBOARD_SUBMIT_URL` | ✅ **`https://bodh.tanuh.ai…` since 1 Oct** (were unset: image defaults) | PT7 ✅; PT8 (defaults and policy at the next rebuild) |
| Leaderboard | `CATALOGUE_BASE_URL` | **the deployed value works** (shown by the 1 Oct job); the code default is still `https://benchmark.tanuh.ai/controlplane/iudx/v2` | LB5 |

**Commands used for BT8 and PT7** (run 1 Oct):

```bash
P=proj-tanuh-benchmark-ptfm; Z=asia-south1-b; H=https://bodh.tanuh.ai
gcloud compute instances add-metadata test-cpu-c3 --zone=$Z --project=$P --metadata=\
tee-env-KEYCLOAK_ISSUER=$H/auth/realms/tanuh,\
tee-env-KEYCLOAK_JWKS_URL=$H/auth/realms/tanuh/protocol/openid-connect/certs,\
tee-env-BUFFER_CALLBACK_BASE=$H/tee
gcloud compute instances stop test-cpu-c3 --zone=$Z --project=$P && gcloud compute instances start test-cpu-c3 --zone=$Z --project=$P
gcloud compute instances add-metadata cpu-tdx --zone=$Z --project=$P --metadata=\
tee-env-CATALOGUE_BASE_URL=$H,tee-env-LEADERBOARD_SUBMIT_URL=$H/leaderboard/submit-solution
```

**Alternative:** if the platform team restores `benchmark.tanuh.ai` (ingress rule, certificate, and the old Keycloak hostname), none of this is needed (question 4 in 1.9).

## 1.11 Croissant metadata and class names

**What Croissant is:** [MLCommons Croissant 1.0](https://mlcommons.org/croissant) is a standard way to describe an ML dataset, written as JSON-LD. A document has three parts:
- **Dataset facts:** name, description, licence, creator, citation, keywords, version.
- **`distribution`:** the files. A `FileObject` is one file (e.g. the labels CSV); a `FileSet` is a glob for many (e.g. `*/*.dcm`).
- **`recordSet` → `field`:** the logical columns. Each has a `dataType` and a `source` (which file, which column).

**How the platform uses it** (`ui-dx/apps/TANUH/src/app/shared/utils/croissant.util.ts`):
- **Onboarding step 1:** the data provider fills in the **Croissant builder** form, or uploads a Croissant JSON that is validated and then fills in the builder. A template can be downloaded.
- **Validation:** `validateCroissant`, a TypeScript port of `Platform_Template/validate_croissant.py`.
- **TANUH `platform` block (optional):** `task`, `card`, `metrics`, `statistics`, `input`, `output`. It needs a `"tanuh"` entry in `@context` (PR #840 changed its value to `https://bodh.tanuh.ai/schema/1.0#`; the validator checks only that the key exists). The UI copies the block into the form, which saves it as `datasetMetrics` and the profile fields, then drops it from the stored document so the two can't disagree.
- **Storage:** the document is saved as a JSON **string** in the catalogue entry's `croissant` field. Some required catalogue fields are filled from it (description, tags, year range, file format).
- **The TEEs don't read Croissant.** The Processing TEE reads only the flat fields in 1.3 (`problemStatement`, `class_names`, `datasetMetrics`, `primaryMetric`).

**Class names (UI7):** binary and multiclass both have a fixed, ordered class list. Label `i` = `class_names[i]`, and for binary, index 1 is the positive class (sensitivity, PPV and F2 are computed for it).

**The design below is the target** (`instructions.md` §1). What's built so far (PR #843) is a comma-separated "Class names" field that writes only the top-level `class_names`, on create. In the target design, the builder's Classes section is the single source, written to two places from the same form values:

1. **Top-level `class_names`** on the catalogue entry, which the Processing TEE reads.
2. **A `classes` record set in the Croissant document**, the standard Croissant pattern for a class list. The label field in `test_points` references it with `"references": {"field": {"@id": "classes/index"}}`.

```json
{
  "@type": "cr:RecordSet", "@id": "classes", "name": "classes",
  "key": { "@id": "classes/index" },
  "field": [
    { "@type": "cr:Field", "@id": "classes/index", "name": "index", "dataType": "sc:Integer" },
    { "@type": "cr:Field", "@id": "classes/name",  "name": "name",  "dataType": "sc:Text" },
    { "@type": "cr:Field", "@id": "classes/value", "name": "value", "dataType": "sc:Text" }
  ],
  "data": [
    { "classes/index": 0, "classes/name": "Non-Suspicious", "classes/value": "0" },
    { "classes/index": 1, "classes/name": "Suspicious",     "classes/value": "1" }
  ]
}
```

- **`classes/value`** (optional) records how a class is written in the provider's labels file (e.g. `1–4`, `Yes/No`). Today it's stored in Croissant only; the TEE doesn't read it yet (2.7, PT10). **Until then, ground-truth labels must be the class names or 0-based indices.** Both datasets' ground truths already meet this: OCS's original file, and BCD's `A–D` labels, were read correctly in the 30 Sep parity runs (1.4).
- **`validateCroissant`** checks an uploaded `classes` record set: indices are exactly 0…N-1 and names are unique. Plain Croissant without `classes` is still accepted.
- **The live entries** were onboarded before UI7. Since 1 Oct (CA2) both documents have a `classes` record set. Their label fields are still plain `sc:Text`, with no `references`.

---

# Part 2: Next phase

Each topic below is independent: a design, then tasks per component. Suggested order: **2.3** (easier adaptor) and **2.2.1** (easier wrapping) first, since they cut owner effort the most for little platform work; **2.8** (hardening) before any real benchmark.

## 2.1 Summary

| # | Topic | Why |
|---|---|---|
| 2.2 | Preprocessing: easier wrapping, Option C, provider scripts | owners shouldn't have to edit their models |
| 2.3 | Generalising the adaptor: built-in default, one-function contract | most owners write no code; the rest write one function |
| 2.4 | TorchScript and Hugging Face in the UI | the TEE already runs them |
| 2.5 | More buckets and problem types | segmentation and detection ✅ live (1 Oct); multi-label, regression, efficiency next |
| 2.6 | Model contract | one clear, versioned statement of what a submission is |
| 2.7 | Dataset contract | data cards and catalogue fields that describe the files exactly |
| 2.8 | Hardening and security | before real, confidential benchmarks |
| 2.9 | Platform improvements | speed, robustness |
| 2.11 | Large datasets: a command-line uploader | browser uploads don't scale to 50 GB+ (CT Fracture stalled at 12%) |

## 2.2 Preprocessing

**The problem:** a model is fed only the platform's standard input (RGB, resized, 0–1). If its training used extra steps (ImageNet normalisation, BGR order, colour balancing), those must be inside the model. Otherwise the score drops without any error. Today that means owners wrap or re-export their model. There are three ways to make it easier, from least to most platform work.

### 2.2.1 Easier wrapping (local tools only)

- **A flag-based wrap tool.** `wrap_onnx.py` today knows only two fixed recipes (`--mode imagenet`, `--mode ocs`). A general version lets owners describe their training preprocessing:

  ```bash
  python3 wrap_onnx.py --src my_model.onnx --dst my_model_tanuh.onnx \
      --normalize imagenet            # or --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5
      --channel-order bgr             # if trained with cv2.imread
      --resize bicubic
  ```

  Same for `wrap_torchscript.py`. Custom steps like colour balancing stay named recipes (`--mode ocs`) or go in the owner's own export.
- **A check that catches the silent drop.** The owner runs their own training pipeline on one image, notes the output, and gives it to the tool:

  ```
  $ python3 check_model.py --model my_model_tanuh.onnx --image sample.jpg --expect 0.91
  OK   platform output: 0.912 — matches your training pipeline (±0.01)
  ```

| Component | Task | Where |
|---|---|---|
| Processing (tools) | flags `--normalize/--mean/--std/--channel-order/--resize` in `wrap_onnx.py` and `wrap_torchscript.py`; keep `--mode ocs` | `tools/reference/` |
| Processing (tools) | `check_model.py --expect <value(s)> [--tolerance]` | `tools/check_model.py` |
| Docs / UI | the owner instructions (1.7) point to both tools | submit page, `model-submission.md` |

### 2.2.2 Option C: the platform applies standard preprocessing from settings

**The idea:** the owner uploads the model **as it is** and picks their preprocessing on the submit form; **our code** applies it. The same pattern appears in TFLite metadata, Core ML `ImageType`, OpenVINO mean/scale flags and Hugging Face `preprocessor_config.json`. Unusual preprocessing still goes inside the model.

**The input spec** (`input_spec.json`, built by the UI):

```json
{"resize": "bilinear", "channel_order": "RGB",
 "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
 "input_shape": [3, 224, 224], "layout": "NCHW"}
```

| Field | Values | Default | Status |
|---|---|---|---|
| `resize` | `bilinear`, `bicubic` | `bilinear` | **built** |
| `input_shape` / `input_size` | `[C,H,W]`, `[H,W]`, an int, … | from the model | **built** (TorchScript; ONNX with dynamic size) |
| `layout`, `dtype` | NCHW/NHWC; float32/float16/uint8 | from the model | **built** |
| `channel_order` | `RGB`, `BGR` | `RGB` | **to build** (refused today) |
| `normalize` | `null` or `{mean, std}` per channel | `null` | **to build** (refused today) |

**Already built:**
- the Buffer's `input_spec` upload slot (ONNX and TorchScript, optional, ≤ 64 KiB, hash-committed)
- the payload artifact
- `modelpkg` placing it next to the model
- `infer.py` parsing size, layout, dtype and resize with flexible key names
- `check_model.py --input-spec`

**Per-image order with Option C:**
1. decode
2. resize
3. **reorder channels**
4. scale to 0–1
5. **`(x − mean) / std`**
6. layout and dtype
7. model

**Example:** the original BCD ResNet34 with the preset "ImageNet standard" would need **no wrapping**. OCS (colour balancing) would still be wrapped.

**Risks:**
- **A wrong setting lowers the score silently.** Mitigate with `check_model.py --input-spec` and by showing the settings with the result.
- **Resize libraries differ slightly.** Keep PIL for bilinear and OpenCV for bicubic, and document it.

| Component | Task | Where |
|---|---|---|
| Processing | accept `channel_order` and `normalize` (mean/std length = channels) | `tools/infer/infer.py` (`parse_input_spec`) |
| Processing | apply them per image | `tools/infer/infer.py` (`InputSpec`, `to_tensor`) |
| Processing | tests; parity: the **original** BCD model + "ImageNet standard" must match the old evaluator | `tools/infer/test_infer.py` |
| Buffer | nothing (slot exists) | — |
| UI | "Input preprocessing" on the submit form: presets ("model includes its own preprocessing" = no spec; "ImageNet standard"; custom). Shape/layout fields for TorchScript; hidden for Hugging Face. Pass as `inputSpec` (supported in `lib/`). | `reference-datasets.component.ts` / `.html` |
| UI | show the chosen settings with the result | results view |
| Docs | model-owner instructions with the settings | `model-submission.md`, submit page |

### 2.2.3 Provider preprocessing script (after the sandbox)

**The idea:** the owner uploads Python that turns each input into the model's tensor.

| | Provider script | Option C |
|---|---|---|
| Flexibility | anything (CLAHE, crops, windowing, colour balancing) | standard steps only |
| Reusing training code | copy-paste | translate into settings |
| Security | owner code next to the decrypted data and the ground truth: it can read every file, copy answers into the input, or send data out | no owner code |
| Raw files | the owner handles EXIF, DICOM transfer syntaxes and bit depths **blind**, so it needs a complete, correct data card | the platform decodes, the same way for everyone |
| Failures | silent, and the owner can't debug them | settings validated |
| Dependencies | installed per job | none |
| Fairness | decoding differs per model | same for all |

**Middle path:** give the script the **platform-decoded image array**, not the raw file. Decoding stays consistent, the data card only needs to describe the decoded result, and a sandboxed script needs no file, ground-truth or network access.

| Component | Task |
|---|---|
| Processing | sandbox first (2.8, H1); then a preprocessing stage that feeds decoded arrays to the script and collects tensors |
| Buffer | a `preprocessing` upload slot |
| UI | an upload field and instructions |
| Dataset contract | the technical data-card fields (2.7) |

## 2.3 Generalising the adaptor

**The problem:** today's adaptor is a full script: it parses arguments, reads the `.npz` file and the dataset spec, applies sigmoid/softmax and writes a CSV. For most models that's pure boilerplate. And models for the same problem return **different output formats**: raw scores or probabilities, one output or two for yes/no, a different class order, extra outputs, `[N]` vs `[N,1]`. So something must bridge each model's format to the evaluator's.

### 2.3.1 Design

1. **A built-in default adaptor, making the adaptor optional.** It handles the common conventions:

   | Model output | Default does |
   |---|---|
   | binary, `[N,1]` raw score | sigmoid → probability of `class_names[1]` |
   | binary, `[N,2]` | softmax → take column 1 |
   | multi-class, `[N,C]` | softmax → probabilities |
   | already probabilities (0–1, rows sum to 1) | used as they are |
   | several outputs, or an unexpected shape | a clear error asking for an adaptor |

   It **can't** detect a model whose class order differs from `class_names`; the instructions must state the class order plainly.
2. **One function instead of a script.** The owner writes only the conversion; the platform does the rest:

   ```python
   # adaptor.py — the whole file
   import numpy as np

   def adapt(outputs, ctx):
       """outputs: {output_name: array, one row per image}.
       ctx: class_names, task_type (later also original image sizes, model input size).
       Return the bucket's result type, e.g. binary: [N] probabilities of class_names[1]."""
       x = outputs["logits"][:, 0]
       return 1 / (1 + np.exp(-x))
   ```

   **How it runs:**
   1. `infer.py` saves the outputs.
   2. **Our runner** imports `adaptor.py` and calls `adapt(outputs, ctx)`, in batches so large outputs fit in memory.
   3. It checks the returned shape and range, with a clear error such as *"adapt() returned shape (127, 2), expected (127,)"*.
   4. It writes `predictions.csv` itself, pairing each row with its image.
   5. The evaluator scores it.

   A file **without** `adapt` is still run as a current-phase script, so existing adaptors keep working.
3. **Local check:** `check_model.py --adaptor adaptor.py` runs the function on the owner's sample image output and validates the result.

**Owner instructions after this change:** "Upload your model. If its output is standard logits or probabilities, you're done. Otherwise upload an `adaptor.py` with one function, `adapt(outputs, ctx)`, that returns the probability per class for each image."

### 2.3.2 It generalises per problem type

Each bucket defines three things: `adapt`'s **return type**, its **default** (if any), and the **predictions file** the platform writes (which the bucket's evaluator reads).

| Problem type | Typical model output | `adapt` returns | Default possible? |
|---|---|---|---|
| Binary classification | `[N,1]` or `[N,2]` | `[N]` probability of class 1 | ✅ sigmoid / softmax |
| Multi-class | `[N,C]` | `[N,C]` probabilities | ✅ softmax |
| Multi-label | `[N,C]` | `[N,C]` independent probabilities | ✅ sigmoid per class |
| Regression | `[N]` / `[N,1]` | `[N]` numbers | ✅ identity |
| Segmentation | `[N,C,H,W]` at the model's input size | a mask per image **at the original size** | ⚠️ argmax, if the platform rescales masks (needs `ctx.original_sizes`) |
| Detection | varies wildly (anchors, NMS or not, coordinate conventions) | a list of `[x1,y1,x2,y2,score,class]` per image, in original coordinates | ❌ adaptor required |

### 2.3.3 Tasks

| Component | Task | Where |
|---|---|---|
| Processing | adaptor runner: load outputs, import `adaptor.py`, call `adapt(outputs, ctx)` per batch, validate, write `predictions.csv`; fall back to the script contract | new `tools/adaptor/run_adaptor.py`; `internal/pipeline/pipeline.go` (stage 2) |
| Processing | built-in default adaptor per bucket (table 2.3.1), used when none is uploaded | `tools/adaptor/`; `internal/pipeline/pipeline.go` (`materialize`: adaptor optional) |
| Processing | `check_model.py --adaptor` | `tools/check_model.py` |
| Processing | tests; parity re-run on OCS/BCD with the default adaptor | `tools/.../test_*.py` |
| Buffer | make the `adaptor` slot optional (required only if a hash was submitted); `adaptor_sha256` optional in `/v1/submit` | `internal/jobs/artifacts.go`, `internal/server/submit.go` |
| UI `lib/` | `adaptorFile` optional again | `lib/transfer-onnx.ts` (`submitOnnxModel` check) |
| UI | "Adaptor (optional)" field with the one-function instructions and example | `reference-datasets.component.ts` / `.html` |
| Docs | owner instructions and templates rewritten for `adapt()` | `model-submission.md`, `tools/reference/` |

**Security:** a custom adaptor is still owner code (2.8, H1). With a default, most submissions run **no owner code at all**.

## 2.4 TorchScript and Hugging Face in the UI

**Already in the TEE and Buffer:** both formats load and run:
- **TorchScript:** the input size comes from an uploaded `input_spec.json` or an embedded `extra/tanuh.json` / `input_spec.json`.
- **Hugging Face:** the zip's model folder is found at any depth; only `config.json`, `preprocessor_config.json` and safetensors are used; `auto_map` is removed; pickle-only weights are refused.

`lib/` already supports `modelFormat`, optional weights and `inputSpec`. **What's missing is the UI.**

| Component | Task | Where |
|---|---|---|
| UI | model-format selector. ONNX: `.onnx` + optional `.onnx.data`. TorchScript: `.pt` + input size fields (H, W; optional channels, layout, resize) passed as `inputSpec`. Hugging Face: `.zip`, no spec. Hide the weights step when there's no weights file. | accept getters (around lines 1033–1047), validators, `handleOnboardUploadStep` in `reference-datasets.component.ts` |
| Processing | end-to-end runs: OCS × TorchScript (`wrap_torchscript.py --mode ocs`), a Hugging Face test model (`tools/reference/make_hf_fixture.py`) | — |
| Docs | TorchScript and Hugging Face owner instructions (already in `model-submission.md`) on the submit page | submit page |

## 2.5 More buckets and problem types

**Adding a bucket means:**
1. an `evaluate.py` in `gs://tanuh-evaluators/<bucket>/`
2. its predictions format
3. its `adapt` return type and default (2.3.2)
4. its metric keys in the TEE's alias table (`internal/leaderboard/metrics.go`)
5. an option in the UI's `problemStatement` dropdown

A bucket whose ground truth isn't one class per image also needs a **ground-truth kind** in the Processing TEE (`internal/pipeline` `bucketKinds`: `class`, `mask` or `box`).

### Buckets live on 1 Oct 2026 (image `det-20261001`)

| Bucket (`problemStatement`) | Ground truth | Predictions (`predictions.csv`) | Metrics | Adaptor |
|---|---|---|---|---|
| `binary_classification` | `file,label` | `file,label,score` | accuracy, sensitivity, specificity, ppv, npv, fnr, fpr, f1, f2, auc | needed today (2.3) |
| `multiclass_classification` | `file,label` | `file,label,prob_0..` | accuracy, macro/weighted F1/F2, sensitivity, specificity, ppv, npv, auc, qwk | needed today |
| **`segmentation`** (PT12) | `image,mask` (one mask file per image) | `file,mask` (mask files under `predictions/`) | dice, iou, global_dice, global_iou, sensitivity, ppv, specificity, pixel_accuracy | `adaptor_argmax_segmentation.py` |
| **`object_detection`** (PT13) | one row per box: `file,x_min,y_min,x_max,y_max,class` (or `x,y,w,h`), or COCO JSON | one row per box: `file,x_min,y_min,x_max,y_max,score,class` | map (mAP@[.5:.95]), ap50, ap75, map_giou, recall_50, mean_iou | **always required**: `adaptor_boxes_detection.py` as a template |

**What data providers need to know:**
- **Segmentation:**
  - **Class names:** background first, in pixel-value order.
  - **Masks:** one per image, the same size as the image, lossless (PNG/TIFF/BMP, or `.npy`), single-channel class indices. A binary mask may use 0/255. No JPEG masks, no colour-coded RGB masks, and no pixel values outside the classes.
  - **Ground truth:** a two-column table in the dataset's root folder. Give mask names with their extension.
  - **Not supported:** 3D volumes, overlapping classes, instance masks. Ignore regions aren't wired up yet: `ignore_index` and `empty_score` exist in the evaluator but aren't passed from the catalogue.
- **Detection:**
  - **Class names:** no background; one class is fine.
  - **Boxes:** in original-image pixels. An image with no objects gets one row with empty box fields.
- **Model owners (detection):**
  - The model must take batch size 1 or a dynamic batch.
  - The platform resizes without letterboxing.
  - The adaptor maps boxes back with `orig_hw` / `input_hw`.

**Open for these buckets:**
- **UI9:** detection name, one-class `class_names`, and metric list.
- **PT15:** `bounding_box` alias, if the UI keeps that name.
- **Wiring `ignore_index` / `empty_score`** from the catalogue to segmentation.
- **More detection formats:** YOLO txt and VOC XML (one file per image) aren't read yet.
- **The legend's "Efficiency (FPS / GMACs)":** the TEE doesn't compute it. It needs a definition first: FPS on a CPU TDX VM, MACs from the graph, and a range that isn't 0–1.

**A new file type** (CT/MRI volumes, NIfTI, pathology slides) needs a new decoder in `infer.py` (content detection is in `file_kind`). Some need **dataset-level decode settings** (e.g. CT windowing) in the catalogue, the same for every model.

### 3-D volumes: what CT Fracture needs (PT18)

**The dataset:** 195 cases. Each case is one axial bone-kernel CT series of 141–651 slices (median 369), 512×512 (one case 1024×1024), stored as one DICOM per slice. One label and one score **per case**. Pixel data is untouched (no resampling or windowing), and decoding needs JPEG Lossless and JPEG 2000 (`pylibjpeg[all]`, already in the image).

**What changes:**
1. **Sample = a folder.** The ground truth lists case ids; each id matches **a folder of slices**, not a file. `groundtruth.Match` already matches "a per-sample folder" but expects one file in it, so it needs a volume mode: all DICOMs under the folder form the sample. `inputs.txt` lists folders.
2. **Decode to a volume.** In a new `infer.py` input kind, `volume`:
   - read every slice, sort by `ImagePositionPatient` along the slice normal (fallback `InstanceNumber`), and check one series and consistent spacing
   - apply `RescaleSlope` / `RescaleIntercept` to get **Hounsfield units**, as float32 `[D, H, W]`
   - no windowing or resampling by default, as the card says. A dataset-level setting can add a standard window later, the same for every model.
3. **Model input.** Volumes differ in depth and spacing, so either:
   - **(a)** the model takes a dynamic depth: `[1, 1, D, H, W]` or `[1, D, H, W]`, one case at a time (like detection's `--per-sample`), or
   - **(b)** the platform resamples to a fixed size the owner declares in `input_spec.json` (`depth`, `height`, `width`, `resample`).

   **(a)** is simpler and fairer. Memory: up to 651×512×512 float32 ≈ 680 MB per case, fine on 16 GB.
4. **Adaptor and evaluator:** unchanged. The adaptor writes `file,score` with `file` = the case id, and the binary evaluator scores per case.
5. **Data volume:** about 15 GB of slices, so it needs the large-dataset path (2.11, PT19), and `cpu-tdx` needs the disk for it.

**Open question (1.9 q7):** would per-slice models (2-D slices plus aggregation in the adaptor) also be accepted? If yes, the volume kind also needs a "slices" mode that feeds each slice and gives the adaptor all of a case's outputs.

| Component | Task |
|---|---|
| Processing | evaluator + metrics aliases + reference adaptor per new bucket; decoders per new file type |
| Catalogue / UI | new `problemStatement` values that match the TEE's bucket names; metric names on `ALLOWED_CUSTOM_METRIC_NAMES`; dataset-level decode settings where needed |
| Leaderboard | nothing, as long as metrics follow the catalogue rules |

## 2.6 Model contract

One versioned statement of what a submission is, so owners, the UI and the TEE agree:

- **Formats:** file rules, loading, versions (onnxruntime 1.19.2, torch 2.5.1, transformers 4.46.3). Already written in `model-submission.md`.
- **Input:** what the platform feeds (1.3), plus the input spec (2.2.2).
- **Output:** the bucket's result type through the adaptor or default (2.3).
- **Limits:** since the streaming release (22 Oct) the TEEs cap no file sizes (only `input_spec.json` ≤ 64 KiB); the **UI's caps** (UI13) are the published limits. A Hugging Face zip may expand to ≤ 8 GiB and ≤ 10,000 files (zip-bomb guard). Time limits: inference 1 h, adaptor 10 min. Memory: whatever inference needs on the VM (16 GB on `cpu-tdx`).
- **Versioning:** a `contract_version` recorded with each result, so scores stay comparable when the contract changes.

| Component | Task |
|---|---|
| Docs | one "Model contract v1" page merging `model-submission.md` and 1.7; linked from the submit page |
| Processing | record the contract version in `results.json` |
| UI | the submit page shows the contract for the chosen format and bucket |

## 2.7 Dataset contract

**Ground truth (done):** flexible formats, columns, labels, ids and folder layouts (1.3).

**Catalogue fields:**
- the UI writes `problemStatement` ✅
- the UI writes **`class_names`** on create, from a comma-separated field (partly done, UI7; the full Classes section is 1.11)
- only metrics some evaluator produces can be chosen (mostly done, UI8; not yet per bucket)
- **still open:** metric ranges are written as bare numbers (so QWK is wrongly 0–1)
- **still open:** each class's "value in the labels file" is stored in Croissant but not passed to the TEE

**Data cards** today cover study metadata but little of what decoding depends on. **Checking the real files against their cards found:**
- **OCS:** 125 of 127 JPEGs have EXIF orientation 6 (rotate 90°); OpenCV applies it, Pillow doesn't; the card doesn't say. The CSV has 135 rows for 127 images.
- **BCD:** 86 of 240 DICOMs have no transfer syntax in the file meta; 154 are JPEG Lossless; 12-bit pixels in 16. None of that is in the card.
- **BCD card examples:** `L_CC.dcm` / `R_MLO.dcm`, but every file is a left CC view named `subject_XXXXX_CC_L.dcm`.
- **BCD labels:** the card says 0–3; the xlsx uses `A`–`D`.

**The technical data-card fields to add:**
1. **Files and layout:** format/container, files per sample, naming, what a ground-truth id refers to, file vs row counts.
2. **Image pixels:** sizes, channels/colour space, bit depth, compression, **EXIF orientation**, overlays/borders.
3. **DICOM:**
   - modality; presentation/processing; photometric interpretation
   - bits allocated/stored; signedness; rescale slope/intercept
   - **transfer syntaxes**, and files missing their file meta
   - window center/width, and **which intensity mapping the reference model used**
   - views and laterality; pixel spacing; frames; manufacturers
4. **Processing already applied** (crops, resizing, conversions, de-identification), as facts.
5. **Aggregate intensity statistics** (per-channel mean, std, range).
6. **Labels:** class names in index order, and how they're written.
7. **Sample files:** 2–5 files in the exact format, with no patient data.

| Component | Task | Where |
|---|---|---|
| UI | **Finish UI7** (1.8): make class names required for classification, hold binary to exactly 2, add read and write in the edit flows, add the Croissant `classes` record set. The other `updateItem` flows (additional-info, publish-assets, manage-files-modal) keep `class_names` anyway, because the controlplane merges. | metadata step, `manage-contribution(s).component.ts`, `croissant.util.ts` |
| UI | ✅ only producible metrics (UI8). **Remaining: metric ranges** as `"min-max"` strings (QWK `"-1-1"`), not bare numbers. Today every metric is written as `0` and the form's min/max go to `customMetricRanges`, which neither the leaderboard nor the TEE reads. Default the range per metric (QWK −1–1, the rest 0–1). | `buildTanuhDatasetMetricsFromForm`, `buildCustomMetricsMap` in `tanuh-dataset-item-payload.util.ts` |
| UI | send each class's **value in the labels file** (UI7's optional `value` column) as top-level `class_values` (same length and order as `class_names`) | `buildTanuhDatasetItemPayload` |
| Processing | **PT10:** read `class_values` from the catalogue and map ground-truth labels through it (e.g. `1–4` → 0–3, `Yes/No` → 1/0), before the existing name/index parsing | `internal/catalogue/catalogue.go`, `internal/groundtruth/groundtruth.go` (`labelParser`) |
| UI | ground-truth accept list: add `.tsv`, `.jsonl` (the TEE reads them) | `upload.component.html` (around line 275) |
| UI | ground-truth preview before encrypting: which columns were read, label counts, rows with no matching file (the TEE skips them; below 50% matched it fails the job) | upload step |
| UI | send the selected dataset's catalogue UUID from the submit form, then delete `LEGACY_ONBOARD_DATASET_UUIDS` | `resolveOnboardDatasetId` (around line 1867); `lib/constants.ts` |
| Data providers | data-card template with the technical fields above; correct the BCD card | data-card template |
| Processing | a check at job start that every required catalogue metric can be produced (fail in seconds, not after inference) | `internal/pipeline/pipeline.go` |

## 2.8 Hardening and security

**Found on the live deployment and code (30 Sep 2026):**
- **Debug image:** both TEEs boot `confidential-space-debug`. Operators can reach the workload, and no verifier checks `dbgstat`.
- **SSH:** project-wide `ssh-keys` still apply to both TEE VMs (neither sets `block-project-ssh-keys`).
- **No egress limits on Processing:** every restrictive egress rule targets `gpu-cs-restricted`, but `cpu-tdx` is tagged `processing-tee`, so its outbound traffic is unrestricted.
- **The restricted tag isn't restrictive:** `restricted-allow-artifact-registry` includes `0.0.0.0/0:443`.
- **Stale firewall rule:** `restricted-allow-buffer-tee-egress` points at 10.50.2.28 (a terminated VM).
- **Broad storage access:** the Processing account can read every bucket in the project.
- **Owner code unsandboxed:** the adaptor runs with the TEE's permissions. It can read the decrypted data and the ground truth, and could copy the answers.
- **Token leak:** the Buffer's `GET /v1/status/{job_id}` (no login) returns `keycloak_token`.
- **Forgeable submissions:** the leaderboard reads the Bearer token without verifying it (`ParseUnverified`) and doesn't require TEE attestation.
- **Hardcoded credentials:** the UI's downloadable upload script has a Keycloak username and password (`upload.component.ts`, around line 648). **Rotate that password now.**

**Confidential Space image lifecycle** (Google notice, effective 1 Apr 2027):
- **Support window:** images are supported for 6 months, then lose `STABLE` and vulnerability scanning.
- **Cut-off:** images released before December 2025 stop getting attestation tokens.
- **Our images:** from 2026, so there's no immediate impact.
- **Why it matters later:** our VMs are long-lived (the Buffer only stops and starts `cpu-tdx`), so they drift. Once the UI's strict verifier is on, an image without `STABLE`/`LATEST` is refused.
- **`p3dx-depa-sandbox`:** its `tanuh-tee-pool` grants access by instance ID, so recreating VMs would break it. Retire it if unused.

| # | Component | Task | Where / done when |
|---|---|---|---|
| H1 | Processing | **Sandbox for owner code:** separate user, no network, no access to the ground truth or other files, only the raw outputs | adaptor and preprocessing stages |
| C5 | Cloud | egress rules for `cpu-tdx`; remove `0.0.0.0/0` from `restricted-allow-artifact-registry`; fix `restricted-allow-buffer-tee-egress` | only policy hosts reachable |
| C6 | Cloud | bucket-level read grants instead of project-level | Processing reads only its buckets |
| C7 | Cloud | `block-project-ssh-keys=true` on both TEE VMs | project keys no longer apply |
| C8 | Cloud | recreate both TEEs from `--image-family=confidential-space` (**non-debug**) | token shows `dbgstat: disabled-since-boot` |
| C9 | Cloud | reserve static internal IPs (10.50.2.34 / .35); recreate VMs with each TEE roll-out and at least every 6 months | images under 6 months old |
| BT7 | Buffer | require `dbgstat == "disabled-since-boot"` for the Processing token; optionally `STABLE` (after C8/C9) | `internal/dispatch/client.go`, `verify_oidc.go` |
| U18 | UI | set `REQUIRE_NON_DEBUG_TEE = true` after C8. The workers then use the strict `verifyOIDC` (`dbgstat`, support attributes). **Ready in `lib/`.** | `lib/constants.ts` |
| BT4 | Buffer | stop returning `keycloak_token` from `/v1/status` | the response has no token |
| BT5 | Buffer | remove the dead `GET /v1/results` (nothing writes it) | no dead endpoint |
| BT6 | Processing | ✅ **Done 1 Oct with PT8:** the callback rule is `bodh.tanuh.ai/tee`. | ✅ |
| LB2 | Leaderboard | verify the Bearer token (JWKS, issuer, expiry) | forged tokens refused |
| LB3 | Leaderboard | accept results only from an attested Processing TEE | only TEE results reach the board |
| U8 | UI | remove the hardcoded credentials from the upload script | the script asks for credentials |

## 2.9 Platform improvements

| # | Component | Task | Why |
|---|---|---|---|
| PT5 | Processing | slimmer CPU image: CPU torch wheels when `ONNXRUNTIME_PKG` is the CPU package (today 15.5 GB; 13-minute `pip install`; ~2-minute pull) | faster builds, pulls and job start |
| PT2 | Processing | EXIF rotation in the Pillow fallback of `decode_image` (GIF, some TIFFs), as OpenCV does for JPEG | same image, same decode on both paths |
| PT4 | Processing / Cloud | GPU image and VM (none in this project; `MAX_GPU_PROVISION_ATTEMPTS=0`) | larger models |
| U17 | UI | "Select for leaderboard" action (`PUT /select-submission`, already in `LeadershipApiService` around line 670); `fetchJobResults` returns `is_selected` | a submission appears publicly only after it's selected |
| LB4 | Leaderboard | update the README and the reference UI's metric templates (short keys like `sensitivity` vs the live `sensitivity_recall`) | docs match the API |
| PT9 | Processing | **retry the leaderboard submit** (a few attempts with backoff on 5xx or a network error) and report a still-failing submit to the Buffer's job status. On 1 Oct a single 500 (LB6) lost the result. Also send `onnx_runtime_providers: []` on failed submissions, so the TEE doesn't depend on the leaderboard's default. | a short leaderboard outage doesn't lose a result |
| PT11 | Processing | **check the catalogue's required metrics before inference** (same as 2.7's row): today a metric no evaluator produces fails the job only after the full run. | such a job fails in seconds |
| BT11 | Buffer / Cloud | **Durable queue** (deferred on 22 Oct; 1.8). Today the queue, the encrypted uploads and their keys live only in the container, so a restart loses them. **Design:** job records and `.enc` files in a GCS bucket; each AES key wrapped by a Cloud KMS key that only the Buffer can use. **Best:** the KMS grant is bound to the Buffer's attestation (a Workload Identity pool that trusts Confidential Space tokens with the Buffer's image digest; the IAM condition is updated on every rebuild). Then on start-up the Buffer re-reads the queue and unwraps the keys. | a Buffer restart resumes queued jobs |
| C10 | Cloud | ✅ **22 Oct:** `tee-restart-policy=OnFailure` on `test-cpu-c3` (BT10). **Also consider it for `cpu-tdx`:** a crashed Processing container today leaves the VM running idle until someone stops it. | — |

## 2.10 Next-phase open questions

1. **Order:** which comes first after the current phase: the easier adaptor (2.3), easier wrapping (2.2.1), or Option C (2.2.2)?
2. **Next datasets:** which file types and problem types come next (2.5)?
3. **Sandbox:** when? It gates the provider-script option and safe owner code.
4. **Production:** when do the TEEs move to the non-debug image (C8)?

## 2.11 Large datasets: a command-line uploader

**The problem.** The browser encrypts and uploads every file itself. For tens of GB that breaks down: WebCrypto runs on one thread, the tab must stay open, a dropped connection loses work, and every byte goes through the web stack. CT Fracture (about 15 GB, 69,486 files) stalled at 12% after 11 days.

**What not to change: the encryption model.**

| Option | Verdict |
|---|---|
| **Plaintext in GCS** | ❌ Anyone with read access to the bucket sees the data: project owners, admins, any service account with `objectViewer`, a leaked key. Google-managed or CMEK encryption doesn't help, because GCS decrypts transparently for every principal with read access. |
| **Encrypted, keys in Secret Manager** | ⚠️ A secret is a raw key: whoever has `secretAccessor` can decrypt everything. Thousands of per-file keys don't fit, and providers would need write access to our secret store. |
| **Encrypted, keys wrapped by Cloud KMS** (today's `tanuh-enc-dataset-v1`) | ✅ Each file has a random AES key (DEK), wrapped with the **public** half of `dataset-dek-wrap`. Providers can encrypt but never decrypt; only the Processing TEE's identity can unwrap, and that can later be bound to attestation. **Keep this; only change where it runs.** |

### The tool: `tanuh-upload`

A small, single-binary command-line tool (Go, like the TEEs; Linux/macOS/Windows). The data provider runs it on the machine that holds the data:

```bash
tanuh-upload login                          # Keycloak device login; no password stored
tanuh-upload check  --dataset <catalogue-uuid> /data/ct_fracture/      # dry run: what will be uploaded
tanuh-upload upload --dataset <catalogue-uuid> /data/ct_fracture/ \
                    --ground-truth /data/ct_fracture_labels.csv --parallel 16
tanuh-upload status --dataset <catalogue-uuid>                          # what is in the bucket
```

**What `upload` does, per file:**
1. **Plan.** Walk the folder, keeping each file's relative path (like the UI's folder upload), and skip junk (`.DS_Store`, `desktop.ini`, `Thumbs.db`). `check` prints the counts per type and folder, the total size, and how the ground truth's ids match the files (the TEE's matching rules), **before anything is uploaded**.
2. **Encrypt locally**, exactly like the browser: a random 32-byte DEK; AES-256-GCM in chunks with IV = `base_iv + i` and AAD `"i:total:file_id"`; the DEK wrapped with the KMS **public** key (RSA-OAEP-3072, fetched once and pinned by version). Output: the encrypted object plus `<name>.manifest.json` with `plaintext_sha256`, `plaintext_size`, `wrapped_dek` and `kms_key_version`. **Byte-for-byte the format the TEE already reads.** The plaintext never leaves the machine.
3. **Upload** straight to `gs://file-server-data/<uuid>/<path>` as **resumable uploads**, 8–32 files in parallel, retrying with backoff. Data first, manifest last: a file counts as uploaded only once its manifest exists.
4. **Resume.** On a re-run, a file whose manifest exists and whose `plaintext_sha256` and size match the local file is skipped. An interrupted resumable upload continues from its last byte. A dropped connection after 40 GB costs nothing.
5. **Verify.** At the end, list the bucket and compare it with the local plan; report missing or extra objects and objects without a manifest (like the 5 in CT Fracture). Exit non-zero if anything is off.

**Credentials: least privilege.**
- The tool never gets project credentials. It logs in to Keycloak, then asks **our backend** for **signed resumable-upload URLs** (V4, about 1 hour), one per object, limited to `gs://file-server-data/<uuid>/…` and to a dataset the caller owns.
- The backend that signs needs `iam.serviceAccounts.signBlob` on a dedicated uploader service account with `storage.objectCreator` on that bucket only. It could be a small new endpoint next to the controlplane, or part of the file server that the UI already uses for dataset uploads.
- The provider can create objects but **can't read or list** anyone's data, including their own ciphertext.

**Speed.** Encryption with AES-NI runs at about 1 GB/s per core, so the provider's uplink is the only limit: about 70 minutes for 50 GB at 100 Mbit/s, about 7 minutes at 1 Gbit/s. If the data already sits in a cloud bucket, the provider runs the tool on a VM next to it.

**The UI stays** for small datasets and for everything that isn't bulk data: the catalogue entry, metadata, class names and metrics. The onboarding page links to the tool and shows the exact command with the dataset's UUID filled in.

### Measured: GCS reads from a VM like `cpu-tdx` (23 Oct)

A temporary `c3-standard-4` in `asia-south1-b` (same private subnet, 100 GB `pd-balanced` disk, no external IP), reading its own bucket in `asia-south1` with the TEE's REST calls; deleted afterwards. Production numbers from the TEE's logs (16 workers, KMS unwrap and decryption included): OCS 129 files / 477 MB in 1.9–2.6 s, BCD 241 files / 3.2 GB in 11.9–12.8 s (about 215–260 MB/s).

| Pattern | Result |
|---|---|
| One 2 GiB object, 1 stream | 134–168 MB/s |
| Same, sliced into 4 / 8 / 16 / 32 ranges | 255 / **378** / 375 / 324 MB/s (best at 8: about 2.3x one stream) |
| 2 GiB to the `pd-balanced` disk (fsync), 1 / 16 streams | 100 / 129 MB/s; raw disk write 163 MB/s |
| 1,000 x 220 KB (CT-like), manifest + object GET each, 16 / 32 / 64 / 128 workers | 198 / 487 / 487 / **1,046** files/s (p50 71 / 60 / 124 / 80 ms per file) |
| 120 x 13 MB (BCD-like), 16 / 32 workers | 341 / 375 MB/s |

**What it means:**
- **Many small files (CT):** more workers is what helps; slicing does nothing. CT's 69,486 files: about 6 min of GETs at 16 workers, about 2.4 min at 32, about 1.1 min at 128, plus one KMS unwrap per file (not measured; the per-file key design means 69,486 KMS calls per job: check the quota).
- **Large files:** slicing doubles network speed, but the **100 GB `pd-balanced` disk caps writes at about 130–165 MB/s**, below even one stream. A sliced download only pays off with a faster disk (`pd-balanced` throughput grows with size; or Hyperdisk / local SSD), or by not writing plaintext to disk at all.

### What the Processing TEE needs for 50 GB+ (PT19)

| Need | Today | Change |
|---|---|---|
| Memory per file | `DecryptDatasetV1` decrypts a whole file in memory | stream it chunk by chunk to disk, reusing `crypto.DecryptChunkedStream`'s loop (the dataset format's chunks have a fixed size and no length prefixes). Fine for images today; required for multi-GB files (pathology slides, large volumes). |
| Disk | the whole matched dataset is decrypted into the job folder | size `cpu-tdx`'s disk to the largest dataset plus the model, or attach local SSD; check free space before decrypting and fail clearly |
| Download per job | every job re-downloads and re-decrypts | acceptable at first (same region; a few minutes per 50 GB with 16 parallel downloads). Later: a per-dataset cache that survives between jobs on an encrypted disk. |

### Tasks

| Component | Task |
|---|---|
| New: tool | `tanuh-upload` (plan, check, encrypt, resumable parallel upload, resume, verify), sharing the format code with the TEE's tests (golden vector) |
| Backend | signed-URL endpoint: Keycloak auth, dataset ownership check, per-object V4 resumable URLs |
| Cloud | uploader service account (`objectCreator` on `file-server-data`, `signBlob` for the backend) |
| UI | onboarding: "large dataset? use the uploader" with the command for this dataset; show the upload status from the bucket |
| Processing | PT19 (streaming dataset decrypt, disk sizing, free-space check) |
| Docs | a data-provider page: install, login, check, upload, resume |

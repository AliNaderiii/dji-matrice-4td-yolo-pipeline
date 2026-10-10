# Experiment card — `<run_id>`

> Copy this file into `docs/experiments/` only after an actual run. Do not fill uncertain fields with estimates.

## Scope

- **Purpose:**
- **Date / operator:**
- **Git commit:**
- **Config digest:**
- **Environment:** OS, Python, package versions, accelerator (if any)

## Data provenance

- **Dataset source and usage rights:**
- **Local data root:** *(do not publish sensitive paths)*
- **Split manifest SHA-256:**
- **Split counts / negative images:**
- **Class contract:** `person`, `hard_hat`, `no_hard_hat`
- **Leakage checks:** sample-ID overlap status; any duplicate-image analysis performed

## Training protocol

- **Base weights:**
- **Hyperparameters:**
- **Selection rule:** validation-only; never select using test results
- **Checkpoint digest:**

## Held-out test evaluation

- **Command:**
- **Metric JSON digest:**
- **Metrics:** copy exactly from `reports/test_metrics.json`
- **Limitations:** label quality, operating domain, confidence/IoU settings, and any known failure modes

## Reporting statement

This card describes one recorded run only. It does not establish real-world safety performance, DJI deployment compatibility, latency, AWS/MQTT integration, or performance outside the stated dataset unless separately evidenced.

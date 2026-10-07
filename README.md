# Preoperative vowel acoustics and difficult laryngoscopic exposure: public methods

Data-free reference code for the historical four-model study. The principal
acoustic block contains **eight cross-vowel measurements plus age**; age is not
an acoustic measurement. This repository is research software, **not a
clinical decision tool**.

中文说明：本仓库仅公开方法代码，不公开患者数据、录音、逐人特征、预测或模型文件。
原稿四方案与后续补充实验的 N 组是不同分析，不能互换结果。

## What is included

| File | Purpose |
| --- | --- |
| `airway_methods/audio.py` | Extract the measurements needed for the fixed eight features from already segmented `/a/`, `/e/`, `/i/` WAVs. |
| `airway_methods/features.py` | Construct the eight signed or absolute cross-vowel differences. |
| `airway_methods/models.py` | Define the historical four model families, input columns, patient-level cross-validation and training-only thresholds. |
| `airway_methods/__main__.py` | Print a column schema, extract features locally, or print aggregate CV metrics. |
| `tests/test_methods.py` | Unit tests using in-memory artificial numbers, never research records. |

The audio input is **one manually selected, already segmented WAV per vowel
per person**. Segmentation, recording quality review, cohort selection,
medical endpoint adjudication and clinical-field interpretation must be done
by the authorized data holder; they are not automated here. Raw sound and
patient-level tables never need to enter this repository.

## Installation and local use

Requires Python 3.10 or newer. Run in this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m airway_methods schema --scheme 3
```

Only for extraction from privately held, presegmented WAVs:

```bash
python -m pip install -r requirements-audio.txt
python -m airway_methods extract --manifest /private/vowels.csv --output /private/features.csv
```

The private manifest must contain `patient_id,a_path,e_path,i_path` and may
contain `target`, `age` and the normalized `clinical_*` columns listed by
`schema`; extra columns are rejected to prevent accidental copying of names.
WAV paths can be absolute or relative to the manifest directory.
Each person must occur once. `patient_id` must be a local pseudonym, never a
name. The extracted CSV contains identifiable or linkable research records:
keep it **outside this repository**. If precomputed features are already
available, skip `extract`; provide a private CSV with the columns printed by
`schema`. `target` is 1 for recorded difficult exposure and 0 otherwise.
`age` and `clinical_age` are deliberately separate columns in historical
scheme 3; do not silently combine them.

```bash
python -m airway_methods evaluate --input /private/features.csv --scheme 1
python -m airway_methods evaluate --input /private/features.csv --scheme 2
python -m airway_methods evaluate --input /private/features.csv --scheme 3
python -m airway_methods evaluate --input /private/features.csv --scheme 4
```

`evaluate` prints only aggregate internal-validation metrics. It does not
write patient-level predictions or fitted models. The four schemes are:

| Scheme | Inputs | Model family |
| --- | --- | --- |
| 1 | Ten clinical inputs | L2-regularized logistic regression |
| 2 | Eight acoustic inputs plus age | Sigmoid-calibrated random forest |
| 3 | Scheme 2 block plus ten clinical columns (age appears twice) | Sigmoid-calibrated random forest |
| 4 | Eight acoustic inputs, age, OSAHS, TMD, Mallampati and neck extension | Calibrated RF and ExtraTrees plus RBF-SVM, logistic meta-model |

Missing-value imputation and applicable scaling are fitted inside training
folds. For schemes 1-3, the classification threshold is selected using
training-fold out-of-fold predictions and balanced accuracy. Scheme 4 uses
accuracy; its historical meta-model threshold was selected from the same
inner predictions used to fit the meta-model, **not** a separate threshold
validation set. All splitting is by person, not by individual vowel.

## Feature definitions

Every expression below uses measurements extracted from the same person's
labelled vowels. A signed difference is right minus left; `abs_delta` is its
absolute value. Missing measurements remain missing until model imputation.

| Feature | Definition |
| --- | --- |
| `rel_i_lpc_F3_vs_e_abs_delta` | `abs(LPC-F3(i) - LPC-F3(e))`, Hz |
| `rel_i_mid_high_energy_ratio_vs_e_delta` | `R(i) - R(e)`; R is mean squared STFT amplitude at 1-3 kHz divided by that at 3-8 kHz |
| `rel_i_b1_stable_vs_a_abs_delta` | `abs(stable B1(i) - stable B1(a))`, Hz |
| `rel_i_stable_score_vs_a_abs_delta` | `abs(stability score(i) - stability score(a))` |
| `rel_i_rolloff95_mean_vs_a_delta` | `95% spectral rolloff(i) - rolloff(a)`, Hz |
| `rel_i_h1_h3_db_vs_e_delta` | `(H1-H3)(i) - (H1-H3)(e)`, dB |
| `rel_i_cpps_full_vs_a_delta` | `whole-vowel CPPS(i) - CPPS(a)`, dB |
| `rel_i_lpc_F1_vs_a_abs_delta` | `abs(LPC-F1(i) - LPC-F1(a))`, Hz |

The WAV code uses 16 kHz mono for spectral/LPC measurements; the stable
window and CPPS use Praat via Parselmouth. It records no patient audio.
The original versions of some historical audio dependencies were not fully
preserved. `requirements-audio.txt` specifies a new-install range, **not**
the proven historical environment.

## Reproducibility and limitations

This is a **standalone reconstruction of the documented methods**, not an
archived byte-for-byte copy of the original scripts. No patient dataset,
historical audio cache, original cohort selection, or fitted historical model
is supplied; running this repository alone cannot reproduce the paper's
numerical results. In particular, the historical feature set and the four
schemes were chosen after exploratory comparisons on the development cohort.
This public implementation does not undo that selection or constitute
independent or external validation. A later nested supplementary analysis
used different N-model definitions and is not represented as the historical
four models here.

Before using private records, verify their inclusion criteria, outcome labels,
clinical parsing and missingness against the approved research protocol.
The resulting predictions are for research only and must not guide patient
care. No training output should be published without a separate disclosure
review.

## Publication boundary

Do not add recordings, source spreadsheets, IDs or their mappings, local
manifests, extracted features, individual predictions, checkpoints, trained
models or restricted review documents. Even pseudonymized patient rows may
be linkable. `.gitignore` blocks common extensions but is **not** a privacy
guarantee: review `git status` and the entire Git history before making a
repository public. Licensing and rights to publish source code should be
confirmed by the project owners before selecting a license or uploading.

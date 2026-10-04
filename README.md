# Ontology-Guided Opportunistic CT Agent for Early Osteoporosis Prognosis Using Vertebral Attenuation, Trabecular Texture, and Fracture-Risk Semantics

This repository holds the measurement stages, the released ontology artefact, the symbolic
constraint operator, the learned scorer, the ten-tool agent, and the training and evaluation
code for the opportunistic bone-health decision layer described in the manuscript. The system
localises vertebral bodies with level labels, measures trabecular attenuation inside a region
that excludes cortical bone and endplates, harmonises each measurement to a standard value with
an uncertainty half-width, retains only those trabecular texture descriptors that survive a
test-retest stability filter, grounds every measured element in a published identifier, and only
then lets a learned scorer produce a candidate decision. That candidate is admitted, repaired
under a fixed violation order and a budget, or rejected as a typed abstention by a symbolic
constraint operator whose axioms come from the ISCD/WHO diagnostic bands, FRAX risk semantics,
the Genant semiquantitative grading, and an internal evidence-trail requirement.

The importable package is `osteoct`. Two statuses are reported for every run: the classification
endpoint with its patient-level bootstrap interval, and standard conformance, the share of the
admitted record's axiom weight that the released artefact satisfies.

## Continuous integration

`ruff`, `black`, `isort`, `mypy --strict` and the test suite are invoked directly:

```bash
ruff check .
black --check .
isort --check-only .
mypy src/osteoct
pytest -q
```

## Installation

The pinned runtime is Python 3.11 or newer with torch 2.11; `requirements.txt` carries the exact
versions the verification pass ran against.

pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

conda:

```bash
conda env create -f environment.yml
conda activate osteoct
pip install -e .
```

Container:

```bash
docker build -t osteoct .
docker run --rm osteoct
```

The image is pinned to a CUDA 12.4 runtime whose torch build matches `requirements.txt`. On a
host without an accelerator the CPU wheel index substitutes for the CUDA index and everything
except throughput runs unchanged. DICOM Part-10 encoding is optional and needs `pydicom`; install
the `sr` extra for it. Without it the content tree is still built and checked, and only the byte
encoding is skipped.

## Data

The multicentre cohort is not distributed. It is held under the data-usage terms of the
contributing institutions, the manuscript's data-availability statement says the same, and the
records cannot be released at record level. Nothing in this repository reconstructs,
approximates or redistributes it.

Every mechanism therefore executes against a schema-compatible cohort fixture in
`src/osteoct/cohort/fixture.py`. The fixture reproduces the record layout, the per-site
acquisition bias structure, the latent attenuation, the constructed height reduction and the
property the dissociation analysis depends on: a subgroup carries a deformity grade at or above
the moderate floor while remaining densitometrically normal or osteopenic, and that subgroup
carries a higher fracture hazard. Two checks hold the fixture to a closed form: the harmonisation
stage must recover the latent attenuation from the acquisition-biased measurement, and the
measured height loss must reproduce the constructed reduction.

Every value computed from the fixture is a fixture value and is reported as such. No manuscript
table value is claimed. To run on real records, write a reader that produces `CohortRecord`
objects with the same fields and point `configs/cohort/fixture.yaml`'s record path at it; the
field contract is `RECORD_FIELDS` in `src/osteoct/cohort/cohort.py`.

Externally hosted resources, their licences and their access conditions are listed in
`dataset_urls.txt` and `NOTICE`. CTSpine1K, TotalSegmentator and VerSe are named because the
manuscript uses them for anatomical pre-training and protocol stress; none carries a bone-density
reference or a fragility-fracture label, and none contributes clinical evidence to any endpoint
reported here.

## Repository layout

```
configs/
  ontology/osteoct.yaml        bands, Genant thresholds, protocol envelope, active families
  acquisition/harmonisation.yaml  harmonisation model and the density indicator
  cohort/fixture.yaml          fixture scale, acquisition ranges and the record contract
  pipeline/agent.yaml          the ten tools, the repair order, the budget, the comparison arms
  model/scorer.yaml            scorer width, arm selection and encoder depth
  train/default.yaml           optimisation profile
  experiment/*.yaml            one file per reported arm, ablation and supplementary analysis
src/osteoct/
  schema/                      concept bindings, record datatypes, clinical vocabulary
  ontology/                    released concept index, OWL 2 EL profile, bands, tiers, severity, axioms
  acquisition/                 attenuation, scanner families, harmonisation, density indicator
  texture/                     grey-level co-occurrence, run-length, wavelet, local binary patterns, stability
  localisation/                vertebral body localisation, morphometry
  grounding/                   binding of every measured element to a published identifier
  decision/                    the constraint operator, repair loop, conformance, evidence trail, abstention
  agent/                       ten-tool toolbox, planner, feature encoding, the end-to-end pipeline
  models/                      three-dimensional encoders, the scorer, baseline scorers, architecture arms
  losses/                      head losses, calibration terms, the additive axiom penalty
  metrics/                     discrimination, concordance, calibration, decision utility, reclassification
  statistics/                  patient-level bootstrap, DeLong, permutation, multiplicity
  evaluation/                  study assembly, main comparison, family ablation, outcome validation,
                               reader study, silent-mode audit, dissociation, findings, reporting
  training/                    seeding, optimiser, schedule, precision, checkpoints, EMA, trainer
  serialisation/               DICOM Structured Reporting content tree
  cli/                         prepare, train, evaluate, agent, verify
tests/                         unit, metric, model, training and verification tests
scripts/                       data preparation and launcher examples
```

## Paper-to-code map

The full mapping, with one entry per claim and its manuscript location, is written to
`claim_to_code.json` by the verification pass. The load-bearing entries:

| Paper location | What it defines | Where it lives |
| --- | --- | --- |
| Sec. III-B ¶1 | level labels and per-vertebra regions | `localisation/vertebra.py` |
| Sec. III-B ¶2 | trabecular region that excludes cortical bone and endplates; mean and spread | `acquisition/attenuation.py` |
| Sec. III-B ¶2 | harmonisation to a standard value with an uncertainty half-width | `acquisition/harmonise.py` |
| Sec. III-B ¶2 | band consistency over the whole interval, not the point estimate | `ontology/bands.py` |
| Sec. III-B ¶3 | co-occurrence, run-length, wavelet and local binary pattern descriptors | `texture/` |
| Sec. III-B ¶3 | test-retest stability filter | `texture/stability.py` |
| Sec. III-B ¶3 | volumetric density indicator from `(m*, u, phi, k)` | `acquisition/density.py` |
| Sec. III-C, Table I | concept index over SNOMED CT, RadLex, LOINC, DICOM SR | `ontology/index.py`, `grounding/standard.py` |
| Table I caption | OWL 2 EL profile; negation, universal restrictions and cardinality excluded | `ontology/profile.py` |
| Table I | the five axiom families, each hard or soft | `ontology/axioms.py` |
| Algorithm 1 | the constraint operator, its resolve step, its repair loop and its budget | `decision/operator.py`, `decision/repair.py` |
| Algorithm 1 L10 | the fixed violation order id, band, tier, severity, evidence | `ontology/axioms.py` |
| Sec. III-D | one reason string per termination route | `decision/abstention.py` |
| Algorithm 2 | the ten tools and the end-to-end agent | `agent/tools.py`, `agent/pipeline.py` |
| Sec. III-E | the conformance endpoint a third party can recompute | `ontology/axioms.py`, `decision/conformance.py` |
| Sec. III-E | the pre-specified conformance floor of 0.90 | `cli/support.py`, `configs/experiment/main.yaml` |
| Sec. III-E | the four perception architectures and the planner substitution | `models/arms.py`, `models/blocks.py` |
| Sec. III-E | patient-level partitions with Site C held out | `cohort/fixture.py`, `training/trainer.py` |
| Sec. IV-A, Table II | comparison metrics: interval, operating point, conformance, unsupported, abstention | `evaluation/main_comparison.py`, `metrics/discrimination.py` |
| Sec. IV-B, Table III | family ablation, single-family removal, component substitution | `evaluation/family_ablation.py` |
| Sec. IV-C, Table IV | concordance with treatment-aware censoring, calibration, decision utility | `metrics/survival.py`, `metrics/calibration.py`, `metrics/utility.py`, `evaluation/outcome_validation.py` |
| Sec. IV-C | reclassification against the clinical comparator | `metrics/reclassification.py` |
| Sec. IV-D | the three-arm reading study and the silent-mode deployment audit | `evaluation/reader_study.py`, `evaluation/silent_mode.py` |
| Sec. IV-E, Finding 2 | the deformity-density dissociation and its prognostic weight | `evaluation/dissociation.py`, `evaluation/findings.py` |
| Sec. IV-F, Finding 1 | the two axes admit no monotone order | `evaluation/findings.py` |
| Algorithm 2 L17 | DICOM Structured Reporting serialisation | `serialisation/dcm_sr.py` |

## Training

One command per arm; the experiment name selects the configuration file and the arm selects the
model variant.

```bash
python -m osteoct.cli.train experiment=main
python -m osteoct.cli.train experiment=main arm=text_only_planner
python -m osteoct.cli.evaluate experiment=main
python -m osteoct.cli.agent experiment=main --partition internal_test --limit 32
```

Under `torchrun`:

```bash
scripts/launch_train.sh main
```

Reported profile, from `configs/train/default.yaml`:

| Setting | Value |
| --- | --- |
| seeds | 5, the count the manuscript reports |
| epochs | at most 60, early stopping at patience 12 |
| batch size | 8 examinations per device |
| gradient accumulation | 2 |
| world size | 1 node |
| effective batch | 16 examinations per optimiser step |
| optimiser | AdamW, weight decay 1e-4, gradient clipping 1.0 |
| learning rate | 3e-4 for the scorer, 1e-4 for the encoder group |
| schedule | cosine decay with a linear warmup over the first 50 steps |
| precision | bf16 |
| texture vector | the 25 descriptors of the four families, standardised on the training split |
| concept vector | the 30 released index entries |

The manuscript does not print an optimiser, schedule, batch profile, repair budget or conformance
weighting. Everything in that table except the seed count is an engineering default exposed in
configuration, and each one is listed in the `deviations` array of `claim_to_code.json`.

The unit-test smoke profile is `configs/experiment/_smoke.yaml`. It exists only so the test suite
and the verification pass can run a small loop on a laptop, and it is labelled as such in the
file. Do not use it for reporting.

## Evaluation

```bash
python -m osteoct.cli.evaluate experiment=main
python -m osteoct.cli.evaluate experiment=ablation_without_severity
python -m osteoct.cli.evaluate experiment=supplementary_silent_mode
```

A run writes `evaluation.json` and a plain-text `evaluation_summary.txt` carrying the main
comparison, the constraint-family ablation, the clinical-outcome table with calibration and
decision utility, the reading study, the silent-mode audit, the dissociation analysis and the two
named findings.

Expected values are not stated here. The cohort those numbers would come from is private, so
every cohort-level value is produced from the schema-compatible cohort fixture and labelled as
such; none of them is a manuscript table value. What can be stated is the criterion the
manuscript fixed before the external evaluation and which the code evaluates directly: a
standard-conformance rate of at least 0.90, compared against a 15.0-percentage-point conformance
margin, with a primary utility margin of 5.0 percentage points and a concordance minimum
important difference of 0.030.

## Compute budget

The manuscript does not report a hardware profile. The local verification pass runs the whole
matrix on one CPU-only host: the executed pass encodes 432 fixture examinations, trains six arms
over five seeds each, and completes in minutes at the configured epoch cap. The reported study
itself is not reproducible from this release because its cohort is not distributed; no
accelerator-hour figure is claimed for it, and no figure is invented to stand in for one.

Disk: the fixture volumes are generated in process and written nowhere by default. The auxiliary
resources in `dataset_urls.txt` are downloaded separately and are not needed for the executed
pass; TotalSegmentator and VerSe are several gigabytes each.

## Verification and status

The verification pass resolves the paper-claim mapping and then executes the release:

```bash
python -m osteoct.cli.verify main
python -m osteoct.cli.verify main --probe-live
```

It writes `claim_to_code.json`, `verification_report.json`, `verification_summary.txt` and
`integrity_manifest.json`. The execution pass covers data reading, the forward pass, the loss,
backpropagation, an optimiser step, a checkpoint round trip and a minimal training loop, and it
re-derives every mechanism with independent code: brute-force pairwise counting for the areas
under the curve, pair enumeration with the same censoring rule for the concordance, the
closed-form net benefit, the four-cell reclassification count, a definition-based intraclass
correlation, a brute-force run-length enumeration, the Haar closed form, an independent logistic
fit for the calibration slope, a bootstrap spread against the DeLong standard error, and a
definition-based step-down for the multiplicity correction. The verification code never calls the
routine it is checking to produce its expected value.

`verification_report.json` carries two statuses. `code_status` covers the provenance, execution,
algorithm, measurement, metric, statistic, serialisation, artefact, environment, gating and
hygiene families. `overall_status` also includes the manuscript-arithmetic family, whose one
failing entry records a discrepancy inside the manuscript's own clinical table: the shifted-window
transformer row and the tool-using-agent row report an identical concordance with an identical
95% interval, although the two configurations differ elsewhere in the same manuscript. The
duplication is reported rather than reconciled.

Deliberate departures from the manuscript, and the engineering defaults that stand in where the
manuscript is silent, are listed in the `deviations` array of `claim_to_code.json`; each entry
names its paper location and its justification.

## Licence and third-party material

The code is released under the Apache License 2.0; see `LICENSE`. Third-party datasets and their
licences are named in `NOTICE`, and their access conditions in `dataset_urls.txt`.

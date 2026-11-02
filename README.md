# NeuroTarget

A SQL Server 2025 system that suggests deep brain stimulation (DBS) electrode placement from anatomically similar previous patients and how their operations turned out. It covers cluster headache (posterior hypothalamus) and Parkinson's disease (subthalamic nucleus and globus pallidus internus), and has a Streamlit + Plotly front end.

> **All data is simulated.** This is a research/teaching prototype for exploring the idea. It is not a medical device and must not be used for clinical decisions.

## Why 3D MRI, not X-ray

An X-ray is a flat projection and barely shows brain soft tissue, so deep targets can't be seen on it. Real stereotactic planning is done on a **3D T1/T2 MRI**, often fused with CT. After the operation, a CT (or sometimes X-ray) is used only to confirm where the lead actually ended up. The system follows the same pattern:

| Stage | Imaging | Stored as |
|---|---|---|
| Planning | 3D T1 MRI | 14-number anatomy fingerprint (`VECTOR(14)`), AC/PC landmarks, scaling to AC-PC space |
| After surgery | CT lead localisation | Positions of the 4 contacts (`nt.Contact`) |
| Follow-up | Clinic scores | Active contact, efficacy %, side effects, patient rating |

## How a suggestion is made (`nt.usp_SuggestPlacement`)

1. **Fingerprint.** At ingest, 14 features are measured from the MRI: brain extents and volume, AC-PC length, third-ventricle width and height, lateral-ventricle volume and span, Evans index, some ratios, and age. They are z-scored against the library and stored as `VECTOR(14)` (`nt.usp_IngestScan`).
2. **Find similar brains.** `VECTOR_DISTANCE('euclidean', …)` picks the K nearest previous implants for the same target. Left and right leads are pooled by mirroring X.
3. **Fit the sweet spot.** Each neighbour gives one observation: where its active contact sat, and how well it worked. A grid search finds the point that best explains those outcomes under the target's dose–response curve. Similar neighbours count for more, via a Gaussian kernel on anatomy distance. Candidate points are penalised where similar patients reported side effects. The search runs at 1 mm, then refines at 0.25 mm; it's a `CROSS JOIN` against `nt.GridPoint`.
4. **Map it into this patient's scan** (native mm) and log it in `nt.Suggestion`.
5. **Close the loop.** `nt.usp_RecordImplantOutcome` stores the actual contacts and the 12-month outcome. From then on, that patient is evidence for everyone after them.

`python/neurosim/engine.py` repeats the same maths line for line in numpy. The app uses it when no SQL Server is configured.

## Does it help? (leave-one-out on the simulated library, K = 40)

Each historical implant is hidden in turn and re-planned from everyone else. Because the data is simulated, every patient's true optimal point is known, so both the suggestion and the plain atlas target can be scored against it.

| Target | Implants | Mean error: atlas | Mean error: suggestion | Closer than atlas | Simulated benefit: atlas → suggestion |
|---|---|---|---|---|---|
| Cluster headache – posterior hypothalamus | 240 | 1.99 mm | 1.46 mm | 72% | 53% → 61% |
| Parkinson's – STN | 720 | 1.79 mm | 1.32 mm | 74% | 43% → 50% |
| Parkinson's – GPi | 320 | 1.58 mm | 1.32 mm | 64% | 37% → 39% |

Suggested points sit a little closer to side-effect structures than the atlas target does (for example, CH side-effect score 0.9 → 1.4 out of 10). The "Side-effect caution" slider sets that trade-off.

## Running it

```bash
pip install -r python/requirements.txt

# 1. simulate the cohort (≈ 1–2 min): 760 implanted patients + 20 new referrals with MRI volumes
cd python
python generate_data.py

# 2a. offline: run the app on the Python engine
streamlit run app.py

# 2b. or with SQL Server 2025
export NEUROTARGET_CONN="Driver={ODBC Driver 18 for SQL Server};Server=localhost;UID=sa;PWD=...;TrustServerCertificate=yes"
python load_sql.py          # creates NeuroTarget DB, runs sql/01 + sql/02, loads everything
streamlit run app.py        # choose "SQL Server 2025" in the sidebar
```

Useful T-SQL:

```sql
DECLARE @id INT;
EXEC nt.usp_SuggestPlacement @ScanId = 1761, @TargetId = 1, @Hemisphere = 'R', @K = 40, @SuggestionId = @id OUTPUT;
EXEC nt.usp_EvaluateLOO @TargetId = 2, @K = 40;   -- set-based; ~40M row evaluations, allow a minute
```

## App tabs

- **Plan (MRI):** axial, coronal and sagittal slices through the suggestion, showing the atlas target and the planned lead trajectory.
- **3D:** similar patients' active contacts mapped into this brain, coloured by benefit and sized by similarity, with ventricles, lead, and a 1-SD spread ellipsoid.
- **Similar patients:** the neighbours, their weights and outcomes, and how benefit falls away with distance.
- **Scan fingerprint:** the 14 features and how this brain differs from the library.
- **Validation:** leave-one-out against the known truth.
- **Close the loop:** simulates surgery with realistic error, a 12-month outcome, and a matched atlas-aimed counterfactual, then records it.

## Layout

```
sql/01_schema.sql        tables, VECTOR(14) fingerprint, evidence view, search grid
sql/02_procedures.sql    ingest, suggest, record outcome, leave-one-out
python/neurosim/anatomy.py   phantom MRI renderer + image feature extraction
python/neurosim/cohort.py    simulated patients, surgery, programming, outcomes
python/neurosim/engine.py    numpy mirror of the stored procedures
python/neurosim/backend.py   SQL Server / local backends
python/app.py                Streamlit front end
data/                        generated CSVs + referral volumes
```

## What a real version would need

- **Proper registration.** Nonlinear registration to MNI space (ANTs, or the Lead-DBS toolbox, which already does much of this), not proportional AC-PC scaling.
- **Real lead localisation.** From post-op CT, including directional leads, and modelling the volume of tissue activated rather than a single contact point.
- **Learned models.** Feature weights and the dose–response curve fitted from data, not set by hand. Hemisphere-specific features. Richer image embeddings, which the `VECTOR` column is ready for.
- **Governance.** Ethics approval, information governance, and UK MDR / clinical safety (DCB0129/0160) before any use near patients.

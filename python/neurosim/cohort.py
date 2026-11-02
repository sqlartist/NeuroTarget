"""
Simulated cohort: historical implanted patients (with outcomes) + new referrals (scan only).

For each historical patient:
  scan -> image features -> surgeon plans near atlas -> lead lands with surgical error ->
  clinician programmes the best of 4 contacts -> 12-month outcome depends on how close the
  active contact is to that patient's (hidden) true sweet spot, and on how close it is to a
  side-effect structure.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from datetime import date, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import anatomy as A

CONDITIONS = [
    dict(ConditionId=1, Code="CH", Name="Chronic cluster headache",
         OutcomeMeasure="% reduction in weekly attack frequency at 12 months"),
    dict(ConditionId=2, Code="PD", Name="Parkinson's disease",
         OutcomeMeasure="% improvement in contralateral MDS-UPDRS III (off-medication) at 12 months"),
]

# Per target: efficacy model + side-effect ("no-go") structure relative to the sweet spot
TARGETS = [
    dict(TargetId=1, ConditionId=1, Code="CH-PH", Name="Posterior hypothalamus / VTA",
         Laterality="Ipsilateral", Emax=75.0, Floor=10.0, SigmaMm=2.2,
         NoGo=[0.0, 2.5, 2.5], NoGoName="Autonomic (hypothalamic) - tachycardia, BP rise, vertigo"),
    dict(TargetId=2, ConditionId=2, Code="PD-STN", Name="Subthalamic nucleus",
         Laterality="Bilateral", Emax=62.0, Floor=5.0, SigmaMm=2.0,
         NoGo=[3.0, 1.5, -1.0], NoGoName="Internal capsule - dysarthria, tonic contraction"),
    dict(TargetId=3, ConditionId=2, Code="PD-GPi", Name="Globus pallidus internus",
         Laterality="Bilateral", Emax=45.0, Floor=5.0, SigmaMm=2.5,
         NoGo=[0.0, 0.0, -3.5], NoGoName="Optic tract - phosphenes"),
]
TGT = {t["Code"]: t for t in TARGETS}

CONTACT_PITCH = 2.0  # mm centre to centre (1.5 mm contacts, 0.5 mm gaps)
N_CONTACTS = 4


def trajectory(hemi: str) -> np.ndarray:
    """Unit vector target -> entry (frontal burr hole), template space."""
    d = np.array([0.25, 0.5, 0.83])
    d = d / np.linalg.norm(d)
    return d * np.array([-1 if hemi == "L" else 1, 1, 1])


def lead_contacts(planned: np.ndarray, hemi: str, err: np.ndarray) -> np.ndarray:
    """Contact 1 is aimed at the planned target; tip (contact 0) one pitch deeper."""
    d = trajectory(hemi)
    tip = planned - CONTACT_PITCH * d + err
    return np.array([tip + k * CONTACT_PITCH * d for k in range(N_CONTACTS)])


def response(contact: np.ndarray, sweet: np.ndarray, tgt: dict, hemi: str):
    """Noise-free efficacy (%) and side-effect score (0-10) for a contact position."""
    sgn = np.array([-1 if hemi == "L" else 1, 1, 1])
    d = np.linalg.norm(contact - sweet)
    eff = tgt["Floor"] + (tgt["Emax"] - tgt["Floor"]) * np.exp(-d ** 2 / (2 * tgt["SigmaMm"] ** 2))
    nogo = sweet + np.array(tgt["NoGo"]) * sgn
    se = 10 * np.exp(-np.linalg.norm(contact - nogo) ** 2 / (2 * 1.8 ** 2))
    return float(eff), float(se)


def simulate_outcome(contacts, sweet, tgt, hemi, rng):
    """Clinician tests every contact, picks the best-looking one; 12-month outcome recorded."""
    scores = []
    for c in contacts:
        e, s = response(c, sweet, tgt, hemi)
        scores.append(e - 4 * s + rng.normal(0, 5))
    k = int(np.argmax(scores))
    e, s = response(contacts[k], sweet, tgt, hemi)
    eff = float(np.clip(e + rng.normal(0, 7), -10, 100))
    se = float(np.clip(s + rng.normal(0, 0.6), 0, 10))
    rating = int(np.clip(round(eff / 10 - 0.35 * se + rng.normal(0, 0.8)), 0, 10))
    return k, eff, se, rating


# --------------------------------------------------------------------------------------
@dataclass
class Job:
    patient_id: int
    condition: str
    target: str | None      # None for a referral that could go to either PD target
    status: str             # Historical | Referral
    seed: int


def _one(job: Job):
    rng = np.random.default_rng(job.seed)
    a = A.sample_anatomy(rng, job.condition)
    if job.condition == "CH":
        hemis = ["R" if rng.random() < 0.55 else "L"]
    else:
        hemis = ["L", "R"]
    spots_r = {t: A.sweet_spot(a, t, "R", rng) for t in A.ATLAS}          # right-hemisphere truth
    spots_l = {t: A.sweet_spot(a, t, "L", rng) for t in A.ATLAS}          # independent L asymmetry
    vol = A.render_phantom(a, spots_r, rng)
    ac, pc = A.landmarks(a, rng)
    feats, geom = A.extract(vol, a["age"], ac, pc)

    scan_date = date(2014, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 11)))
    if job.status == "Referral":
        scan_date = date(2026, 6, 1) + timedelta(days=int(rng.integers(0, 110)))

    out = dict(
        patient=dict(PatientId=job.patient_id, ConditionId=1 if job.condition == "CH" else 2,
                     Sex=a["sex"], AgeAtScan=a["age"],
                     PainSide=(hemis[0] if job.condition == "CH" else None),
                     Status=job.status, IsSimulated=1),
        scan=dict(ScanId=job.patient_id, PatientId=job.patient_id, Modality="MRI_T1_3D",
                  AcquiredOn=scan_date.isoformat(), VoxelMm=A.VOXEL_MM,
                  ACX=geom["ac"][0], ACY=geom["ac"][1], ACZ=geom["ac"][2],
                  PCX=geom["pc"][0], PCY=geom["pc"][1], PCZ=geom["pc"][2],
                  ScaleX=geom["scale"][0], ScaleY=geom["scale"][1], ScaleZ=geom["scale"][2],
                  **{f"f_{k}": v for k, v in feats.items()}),
        truth=[], implants=[], contacts=[], programming=[], outcomes=[], volume=None,
    )
    # truth for every target/hemisphere relevant to the condition (sim only)
    tcodes = ["CH-PH"] if job.condition == "CH" else ["PD-STN", "PD-GPi"]
    for t in tcodes:
        for h in (["L", "R"]):
            s = spots_r[t] if h == "R" else spots_l[t]
            out["truth"].append(dict(PatientId=job.patient_id, TargetId=TGT[t]["TargetId"], Hemisphere=h,
                                     SweetX=s[0], SweetY=s[1], SweetZ=s[2]))

    if job.status == "Referral":
        out["volume"] = vol
        return out

    tgt = TGT[job.target]
    for h in hemis:
        sweet = spots_r[job.target] if h == "R" else spots_l[job.target]
        atlas = A.ATLAS[job.target] * np.array([-1 if h == "L" else 1, 1, 1])
        # surgeon: atlas + partial correction from direct visualisation + planning scatter
        planned = atlas + 0.35 * (sweet - atlas) + rng.normal(0, 0.7, 3)
        contacts = lead_contacts(planned, h, rng.normal(0, 0.9, 3))
        k, eff, se, rating = simulate_outcome(contacts, sweet, tgt, h, rng)
        iid = job.patient_id * 10 + (1 if h == "L" else 2)
        out["implants"].append(dict(ImplantId=iid, PatientId=job.patient_id, TargetId=tgt["TargetId"],
                                    Hemisphere=h, ImplantedOn=(scan_date + timedelta(days=int(rng.integers(20, 90)))).isoformat(),
                                    PlannedX=planned[0], PlannedY=planned[1], PlannedZ=planned[2]))
        for n, c in enumerate(contacts):
            out["contacts"].append(dict(ImplantId=iid, ContactNo=n, X=c[0], Y=c[1], Z=c[2]))
        out["programming"].append(dict(ImplantId=iid, ActiveContact=k,
                                       AmplitudeMa=round(float(np.clip(rng.normal(2.6, 0.7), 0.8, 5)), 1),
                                       PulseWidthUs=int(rng.choice([60, 60, 90])) if job.condition == "PD" else 60,
                                       FrequencyHz=int(rng.choice([130, 130, 160])) if job.condition == "PD" else 185))
        out["outcomes"].append(dict(ImplantId=iid, MonthsPostOp=12, EfficacyPct=round(eff, 1),
                                    SideEffectScore=round(se, 1), PatientRating=rating))
    return out


def _encode_volume(vol: np.ndarray) -> bytes:
    return zlib.compress(np.round(vol * 255).astype(np.uint8).tobytes(order="C"), 6)


def decode_volume(blob: bytes, dims=A.DIMS) -> np.ndarray:
    return np.frombuffer(zlib.decompress(blob), np.uint8).reshape(dims).astype(np.float32) / 255


def generate(out_dir: Path, n_ch=240, n_stn=360, n_gpi=160, n_ref_ch=8, n_ref_pd=12, seed=2026, workers=8):
    out_dir = Path(out_dir)
    (out_dir / "volumes").mkdir(parents=True, exist_ok=True)
    jobs, pid = [], 1000
    for cond, tgt, n, status in [("CH", "CH-PH", n_ch, "Historical"), ("PD", "PD-STN", n_stn, "Historical"),
                                 ("PD", "PD-GPi", n_gpi, "Historical"), ("CH", None, n_ref_ch, "Referral"),
                                 ("PD", None, n_ref_pd, "Referral")]:
        for _ in range(n):
            pid += 1
            jobs.append(Job(pid, cond, tgt, status, seed * 100000 + pid))
    with Pool(workers) as p:
        res = p.map(_one, jobs, chunksize=8)

    tables = {k: [] for k in ["patient", "scan", "truth", "implants", "contacts", "programming", "outcomes"]}
    for r in res:
        tables["patient"].append(r["patient"])
        tables["scan"].append(r["scan"])
        for k in ["truth", "implants", "contacts", "programming", "outcomes"]:
            tables[k].extend(r[k])
        if r["volume"] is not None:
            (out_dir / "volumes" / f"scan_{r['scan']['ScanId']}.bin").write_bytes(_encode_volume(r["volume"]))

    scans = pd.DataFrame(tables["scan"])
    fcols = [f"f_{k}" for k, _, _ in A.FEATURES]
    scans[fcols] = scans[fcols].round(4)   # exactly what RawFeatures JSON carries into SQL
    # population normalisation from the historical library only
    hist_ids = {p["PatientId"] for p in tables["patient"] if p["Status"] == "Historical"}
    hist = scans[scans.PatientId.isin(hist_ids)]
    mu, sd = hist[fcols].mean(), hist[fcols].std(ddof=0)
    fdef = pd.DataFrame([dict(FeatureNo=i, Code=k, Description=d, Unit=u, PopMean=mu[f"f_{k}"],
                              PopStd=sd[f"f_{k}"], Weight=1.0) for i, (k, d, u) in enumerate(A.FEATURES)])
    z = (scans[fcols] - mu) / sd
    scans["RawFeatures"] = [json.dumps({k: round(float(r[f"f_{k}"]), 4) for k, _, _ in A.FEATURES}) for _, r in scans.iterrows()]
    # same rounding as nt.usp_IngestScan (DECIMAL(18,6)) so both engines pick the same neighbours
    scans["FeatureVector"] = [json.dumps([round(float(v), 6) for v in row]) for row in z.values]
    scans = scans.drop(columns=fcols)

    pd.DataFrame(CONDITIONS).to_csv(out_dir / "conditions.csv", index=False)
    tg = pd.DataFrame(TARGETS)
    tg["AtlasX"] = [A.ATLAS[c][0] for c in tg.Code]
    tg["AtlasY"] = [A.ATLAS[c][1] for c in tg.Code]
    tg["AtlasZ"] = [A.ATLAS[c][2] for c in tg.Code]
    tg["NoGo"] = tg["NoGo"].apply(json.dumps)
    tg.to_csv(out_dir / "targets.csv", index=False)
    fdef.to_csv(out_dir / "feature_def.csv", index=False)
    pd.DataFrame(tables["patient"]).to_csv(out_dir / "patients.csv", index=False)
    scans.to_csv(out_dir / "scans.csv", index=False)
    pd.DataFrame(tables["truth"]).to_csv(out_dir / "sim_truth.csv", index=False)
    pd.DataFrame(tables["implants"]).to_csv(out_dir / "implants.csv", index=False)
    pd.DataFrame(tables["contacts"]).to_csv(out_dir / "contacts.csv", index=False)
    pd.DataFrame(tables["programming"]).to_csv(out_dir / "programming.csv", index=False)
    pd.DataFrame(tables["outcomes"]).to_csv(out_dir / "outcomes.csv", index=False)
    return {k: len(v) for k, v in tables.items()}

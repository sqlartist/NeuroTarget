"""
Pure-Python mirror of nt.usp_SuggestPlacement / nt.usp_EvaluateLOO.

Used (a) as the offline backend for the app when no SQL Server is reachable and
(b) to test the algorithm. The maths is kept line-for-line equivalent to the T-SQL.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class Library:
    def __init__(self, data_dir: Path):
        d = Path(data_dir)
        self.dir = d
        self.conditions = pd.read_csv(d / "conditions.csv")
        self.targets = pd.read_csv(d / "targets.csv")
        self.fdef = pd.read_csv(d / "feature_def.csv")
        self.patients = pd.read_csv(d / "patients.csv")
        self.scans = pd.read_csv(d / "scans.csv")
        self.truth = pd.read_csv(d / "sim_truth.csv")
        self.implants = pd.read_csv(d / "implants.csv")
        self.contacts = pd.read_csv(d / "contacts.csv")
        self.programming = pd.read_csv(d / "programming.csv")
        self.outcomes = pd.read_csv(d / "outcomes.csv")
        self.vectors = {int(r.ScanId): np.array(json.loads(r.FeatureVector)) for r in self.scans.itertuples()}
        self._build_evidence()

    # nt.vHistoricalEvidence
    def _build_evidence(self):
        oc = (self.outcomes.sort_values("MonthsPostOp", kind="stable").groupby("ImplantId").tail(1))
        ev = (self.implants.merge(self.programming, on="ImplantId")
              .merge(self.contacts, left_on=["ImplantId", "ActiveContact"], right_on=["ImplantId", "ContactNo"])
              .merge(oc, on="ImplantId")
              .merge(self.scans[["ScanId", "PatientId"]], on="PatientId"))
        ev["X"] = ev["X"].abs()  # fold to right hemisphere
        ev["NetBenefit"] = ev["EfficacyPct"] / 100.0 - 0.04 * ev["SideEffectScore"]
        self.evidence = ev.reset_index(drop=True)
        self._evX = {}
        for tid, g in self.evidence.groupby("TargetId"):
            self._evX[tid] = (g.reset_index(drop=True), np.stack([self.vectors[s] for s in g.ScanId]))

    def target(self, target_id):
        return self.targets.set_index("TargetId").loc[target_id]

    def scan(self, scan_id):
        return self.scans.set_index("ScanId").loc[scan_id]


def neighbours(lib: Library, q: np.ndarray, target_id: int, exclude_patient: int, k: int) -> pd.DataFrame:
    ev, X = lib._evX[target_id]
    dist = np.sqrt(((X - q) ** 2).sum(1))              # VECTOR_DISTANCE('euclidean', ...)
    mask = ev.PatientId.values != exclude_patient
    order = np.lexsort((ev.ImplantId.values[mask], dist[mask]))[:k]   # ORDER BY Dist, ImplantId
    nb = ev[mask].iloc[order].copy()
    nb["Dist"] = dist[mask][order]
    nb.insert(0, "Rn", np.arange(1, len(nb) + 1))
    return nb


def _grid(half: float, step: float) -> np.ndarray:
    g = np.arange(-half, half + 1e-9, step)
    return np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T


GRID_COARSE = _grid(5.0, 1.0)    # nt.GridPoint  Pass = 1  (11^3 = 1331 offsets)
GRID_FINE = _grid(1.0, 0.25)     # nt.GridPoint  Pass = 2  ( 9^3 =  729 offsets)


def _score(cand, P, eff, se, wk, t):
    """Kernel-weighted misfit of the dose-response model at each candidate sweet spot (Fit),
    and how much side-effect burden similar patients reported near it (Risk)."""
    d2 = ((cand[:, None, :] - P[None]) ** 2).sum(2)
    pred = t.Floor + (t.Emax - t.Floor) * np.exp(-d2 / (2 * t.SigmaMm ** 2))
    fit = (wk * (pred - eff) ** 2).sum(1) / wk.sum()
    risk = (wk * se * np.exp(-d2 / (2 * 1.5 ** 2))).sum(1) / wk.sum()
    return fit, risk


def aggregate(nb: pd.DataFrame, t, se_penalty: float = 100.0) -> dict:
    """Fit the sweet spot that best explains the anatomically-similar patients' outcomes.

    Each neighbour gives one observation: (active contact position, efficacy, side effects).
    Model: efficacy = Floor + (Emax-Floor) * exp(-|contact - s|^2 / 2 sigma^2).
    s is found by grid search (1 mm coarse, then 0.25 mm fine) around the atlas target.
    """
    atlas = np.array([t.AtlasX, t.AtlasY, t.AtlasZ], float)
    h = nb.Dist.mean() or 1.0
    wk = np.exp(-(nb.Dist.values / h) ** 2)
    nb["KernelW"] = wk
    P = nb[["X", "Y", "Z"]].values
    eff, se = nb.EfficacyPct.values, nb.SideEffectScore.values
    c1 = atlas + GRID_COARSE
    f1, r1 = _score(c1, P, eff, se, wk, t)
    j = np.lexsort((np.abs(GRID_COARSE).sum(1), f1 + se_penalty * r1))[0]
    c2 = c1[j] + GRID_FINE
    f2, r2 = _score(c2, P, eff, se, wk, t)
    j2 = np.lexsort((np.abs(GRID_FINE).sum(1), f2 + se_penalty * r2))[0]
    best = c2[j2]
    # spread of the good placements among neighbours, and local outcome estimate
    good = wk * np.clip(nb.NetBenefit.values, 0, None)
    sd = np.sqrt((good[:, None] * (P - best) ** 2).sum(0) / good.sum()) if good.sum() > 0 else np.full(3, np.nan)
    near = wk * np.exp(-((P - best) ** 2).sum(1) / (2 * 1.5 ** 2))
    nb["NearW"] = near
    return dict(X=best[0], Y=best[1], Z=best[2], SdX=sd[0], SdY=sd[1], SdZ=sd[2],
                EffectiveN=float(wk.sum() ** 2 / (wk ** 2).sum()),
                PredictedEfficacy=float((near * eff).sum() / near.sum()) if near.sum() > 1e-9 else np.nan,
                PredictedSideEffect=float((near * se).sum() / near.sum()) if near.sum() > 1e-9 else np.nan,
                FitRmse=float(np.sqrt(f2[j2])), Risk=float(r2[j2]), Bandwidth=float(h))


def suggest(lib: Library, scan_id: int, target_id: int, hemi: str, k: int = 40, se_penalty: float = 100.0):
    s = lib.scan(scan_id)
    t = lib.target(target_id)
    atlas = np.array([t.AtlasX, t.AtlasY, t.AtlasZ], float)
    nb = neighbours(lib, lib.vectors[scan_id], target_id, int(s.PatientId), k)
    r = aggregate(nb, t, se_penalty)
    sgn = -1.0 if hemi == "L" else 1.0
    mcp = np.array([(s.ACX + s.PCX) / 2, (s.ACY + s.PCY) / 2, (s.ACZ + s.PCZ) / 2])
    scale = np.array([s.ScaleX, s.ScaleY, s.ScaleZ])
    tpl = np.array([sgn * r["X"], r["Y"], r["Z"]])
    nat = tpl / scale + mcp
    atlas_s = np.array([sgn * atlas[0], atlas[1], atlas[2]])
    r.update(ScanId=scan_id, TargetId=target_id, Hemisphere=hemi, K=k,
             TemplateX=tpl[0], TemplateY=tpl[1], TemplateZ=tpl[2],
             NativeX=nat[0], NativeY=nat[1], NativeZ=nat[2],
             AtlasX=atlas_s[0], AtlasY=atlas_s[1], AtlasZ=atlas_s[2],
             AtlasShiftMm=float(np.linalg.norm(tpl - atlas_s)))
    return r, nb


def evaluate_loo(lib: Library, target_id: int, k: int = 40, se_penalty: float = 100.0) -> pd.DataFrame:
    """Leave-one-patient-out: suggestion vs atlas, both scored against simulated ground truth."""
    ev, _ = lib._evX[target_id]
    t = lib.target(target_id)
    atlas = np.array([t.AtlasX, t.AtlasY, t.AtlasZ], float)
    tr = lib.truth[lib.truth.TargetId == target_id].copy()
    tr["SweetX"] = tr.SweetX.abs()
    tr = tr.set_index(["PatientId", "Hemisphere"])
    rows = []
    for r in ev.itertuples():
        nb = neighbours(lib, lib.vectors[r.ScanId], target_id, r.PatientId, k)
        a = aggregate(nb, t, se_penalty)
        truth = tr.loc[(r.PatientId, r.Hemisphere), ["SweetX", "SweetY", "SweetZ"]].values.astype(float)
        sug = np.array([a["X"], a["Y"], a["Z"]])
        se_s, se_a = _sim_response(sug, truth, t), _sim_response(atlas, truth, t)
        rows.append(dict(ImplantId=r.ImplantId, PatientId=r.PatientId, Hemisphere=r.Hemisphere,
                         SuggestEfficacy=se_s[0], AtlasEfficacy=se_a[0],
                         SuggestSideEffect=se_s[1], AtlasSideEffect=se_a[1],
                         SuggestErrMm=float(np.linalg.norm(sug - truth)),
                         AtlasErrMm=float(np.linalg.norm(atlas - truth)),
                         ActualContactErrMm=float(np.linalg.norm(np.array([r.X, r.Y, r.Z]) - truth)),
                         EffectiveN=a["EffectiveN"]))
    return pd.DataFrame(rows)


def _sim_response(p, truth, t):
    """Noise-free simulated response at p (right-hemisphere frame). Evaluation only."""
    d2 = ((p - truth) ** 2).sum()
    eff = t.Floor + (t.Emax - t.Floor) * np.exp(-d2 / (2 * t.SigmaMm ** 2))
    nogo = truth + np.array(json.loads(t.NoGo))
    se = 10 * np.exp(-((p - nogo) ** 2).sum() / (2 * 1.8 ** 2))
    return float(eff), float(se)

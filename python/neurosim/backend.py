"""
Two interchangeable backends for the app:
  SqlBackend   - SQL Server 2025 via pyodbc; all the thinking happens in the stored procs.
  LocalBackend - the same algorithm in numpy over the generated CSVs (offline demo / tests).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import anatomy as A
from . import engine as E
from .cohort import decode_volume


class LocalBackend:
    name = "Local (Python mirror of the stored procedures)"

    def __init__(self, data_dir: Path):
        self.lib = E.Library(data_dir)
        self._next_implant = 5_000_000

    def conditions(self):
        return self.lib.conditions

    def targets(self, condition_id):
        return self.lib.targets[self.lib.targets.ConditionId == condition_id]

    def referrals(self, condition_id):
        p = self.lib.patients
        p = p[(p.ConditionId == condition_id) & (p.Status == "Referral")]
        return p.merge(self.lib.scans[["ScanId", "PatientId", "AcquiredOn"]], on="PatientId")

    def library_size(self, target_id):
        return int((self.lib.evidence.TargetId == target_id).sum())

    def scan(self, scan_id):
        return self.lib.scan(scan_id)

    def volume(self, scan_id):
        f = self.lib.dir / "volumes" / f"scan_{scan_id}.bin"
        return decode_volume(f.read_bytes()) if f.exists() else None

    def feature_table(self, scan_id):
        raw = json.loads(self.lib.scan(scan_id).RawFeatures)
        z = self.lib.vectors[scan_id]
        fd = self.lib.fdef.copy()
        fd["Value"] = [raw[c] for c in fd.Code]
        fd["Z"] = z
        return fd

    def suggest(self, scan_id, target_id, hemi, k, penalty, save=False):
        r, nb = E.suggest(self.lib, scan_id, target_id, hemi, k, penalty)
        return r, nb.reset_index(drop=True)

    def evaluate(self, target_id, k, penalty):
        return E.evaluate_loo(self.lib, target_id, k, penalty)

    def truth(self, patient_id, target_id, hemi):
        t = self.lib.truth
        r = t[(t.PatientId == patient_id) & (t.TargetId == target_id) & (t.Hemisphere == hemi)].iloc[0]
        return np.array([r.SweetX, r.SweetY, r.SweetZ])

    def record_outcome(self, patient_id, target_id, hemi, contacts, active, eff, se, rating, suggestion_id=None):
        lib = self.lib
        self._next_implant += 1
        iid = self._next_implant
        lib.implants = pd.concat([lib.implants, pd.DataFrame([dict(
            ImplantId=iid, PatientId=patient_id, TargetId=target_id, Hemisphere=hemi, ImplantedOn="2026-09-30",
            PlannedX=np.nan, PlannedY=np.nan, PlannedZ=np.nan)])], ignore_index=True)
        lib.contacts = pd.concat([lib.contacts, pd.DataFrame(
            [dict(ImplantId=iid, ContactNo=n, X=c[0], Y=c[1], Z=c[2]) for n, c in enumerate(contacts)])],
            ignore_index=True)
        lib.programming = pd.concat([lib.programming, pd.DataFrame([dict(
            ImplantId=iid, ActiveContact=active, AmplitudeMa=2.5, PulseWidthUs=60, FrequencyHz=130)])],
            ignore_index=True)
        lib.outcomes = pd.concat([lib.outcomes, pd.DataFrame([dict(
            ImplantId=iid, MonthsPostOp=12, EfficacyPct=eff, SideEffectScore=se, PatientRating=rating)])],
            ignore_index=True)
        lib.patients.loc[lib.patients.PatientId == patient_id, "Status"] = "Historical"
        lib._build_evidence()
        return iid


class SqlBackend:
    name = "SQL Server 2025"

    def __init__(self, conn_str: str):
        import pyodbc
        self.pyodbc = pyodbc
        self.cn = pyodbc.connect(conn_str, autocommit=True)

    def _df(self, sql, *params):
        cur = self.cn.cursor()
        cur.execute(sql, *params)
        cols = [c[0] for c in cur.description]
        return pd.DataFrame.from_records(cur.fetchall(), columns=cols)

    def conditions(self):
        return self._df("SELECT ConditionId, Code, Name, OutcomeMeasure FROM nt.Condition ORDER BY ConditionId")

    def targets(self, condition_id):
        return self._df("SELECT * FROM nt.Target WHERE ConditionId = ? ORDER BY TargetId", condition_id)

    def referrals(self, condition_id):
        return self._df("""SELECT p.*, s.ScanId, s.AcquiredOn FROM nt.Patient p JOIN nt.Scan s ON s.PatientId = p.PatientId
                           WHERE p.ConditionId = ? AND p.Status = 'Referral' ORDER BY p.PatientId""", condition_id)

    def library_size(self, target_id):
        return int(self._df("SELECT COUNT(*) n FROM nt.vHistoricalEvidence WHERE TargetId = ?", target_id).n[0])

    def scan(self, scan_id):
        return self._df("""SELECT ScanId, PatientId, Modality, AcquiredOn, VoxelMm, ACX, ACY, ACZ, PCX, PCY, PCZ,
                                  ScaleX, ScaleY, ScaleZ, RawFeatures FROM nt.Scan WHERE ScanId = ?""", scan_id).iloc[0]

    def volume(self, scan_id):
        df = self._df("SELECT DimX, DimY, DimZ, Data FROM nt.ScanVolume WHERE ScanId = ?", scan_id)
        if df.empty:
            return None
        r = df.iloc[0]
        return decode_volume(bytes(r.Data), (int(r.DimX), int(r.DimY), int(r.DimZ)))

    def feature_table(self, scan_id):
        fd = self._df("SELECT FeatureNo, Code, Description, Unit, PopMean, PopStd, Weight FROM nt.FeatureDef ORDER BY FeatureNo")
        s = self._df("SELECT RawFeatures, CAST(Features AS NVARCHAR(MAX)) AS Vec FROM nt.Scan WHERE ScanId = ?", scan_id).iloc[0]
        raw = json.loads(s.RawFeatures)
        fd["Value"] = [raw[c] for c in fd.Code]
        fd["Z"] = json.loads(s.Vec)
        return fd

    def suggest(self, scan_id, target_id, hemi, k, penalty, save=False):
        cur = self.cn.cursor()
        cur.execute("""SET NOCOUNT ON; DECLARE @id INT;
                       EXEC nt.usp_SuggestPlacement @ScanId=?, @TargetId=?, @Hemisphere=?, @K=?,
                            @SideEffectPenalty=?, @Save=?, @SuggestionId=@id OUTPUT;""",
                    scan_id, target_id, hemi, k, penalty, 1 if save else 0)
        cols = [c[0] for c in cur.description]
        r = dict(zip(cols, cur.fetchone()))
        cur.nextset()
        cols = [c[0] for c in cur.description]
        nb = pd.DataFrame.from_records(cur.fetchall(), columns=cols)
        # mirror the local backend's key names
        r.update(X=abs(r["TemplateX"]), Y=r["TemplateY"], Z=r["TemplateZ"])
        return r, nb

    def evaluate(self, target_id, k, penalty):
        cur = self.cn.cursor()
        cur.execute("EXEC nt.usp_EvaluateLOO @TargetId=?, @K=?, @SideEffectPenalty=?", target_id, k, penalty)
        cur.nextset()
        cols = [c[0] for c in cur.description]
        return pd.DataFrame.from_records(cur.fetchall(), columns=cols)

    def truth(self, patient_id, target_id, hemi):
        r = self._df("SELECT SweetX, SweetY, SweetZ FROM nt.SimTruth WHERE PatientId=? AND TargetId=? AND Hemisphere=?",
                     patient_id, target_id, hemi).iloc[0]
        return np.array([r.SweetX, r.SweetY, r.SweetZ], float)

    def record_outcome(self, patient_id, target_id, hemi, contacts, active, eff, se, rating, suggestion_id=None):
        cj = json.dumps([dict(n=n, x=float(c[0]), y=float(c[1]), z=float(c[2])) for n, c in enumerate(contacts)])
        cur = self.cn.cursor()
        cur.execute("""SET NOCOUNT ON; DECLARE @id INT, @today DATE = CAST(SYSUTCDATETIME() AS DATE);
                       EXEC nt.usp_RecordImplantOutcome ?,?,?,@today,?,?,2.5,60,130,12,?,?,?,?,@id OUTPUT;
                       SELECT @id;""", patient_id, target_id, hemi, cj, active, eff, se, rating, suggestion_id)
        return cur.fetchval()

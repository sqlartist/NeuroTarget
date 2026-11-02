"""
Create the NeuroTarget database on SQL Server 2025 and load the simulated cohort.

    python load_sql.py --conn "Driver={ODBC Driver 18 for SQL Server};Server=localhost;UID=sa;PWD=...;TrustServerCertificate=yes"

or set NEUROTARGET_CONN. Scans go in through nt.usp_IngestScan, so the feature vectors
are built by SQL Server itself from the raw image features.
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import pyodbc

from neurosim import anatomy as A

ROOT = Path(__file__).resolve().parent.parent


def run_script(cn, path: Path):
    batches = re.split(r"^\s*GO\s*$", path.read_text(encoding="utf-8"), flags=re.M | re.I)
    cur = cn.cursor()
    for b in batches:
        if b.strip():
            cur.execute(b)
            while cur.nextset():
                pass
    cn.commit()


def none_if_nan(df: pd.DataFrame) -> list[tuple]:
    return [tuple(None if (isinstance(v, float) and pd.isna(v)) else v for v in r)
            for r in df.itertuples(index=False, name=None)]


def bulk(cn, table, df):
    cur = cn.cursor()
    cur.fast_executemany = True
    cols = ", ".join(df.columns)
    qs = ", ".join("?" * len(df.columns))
    cur.executemany(f"INSERT {table} ({cols}) VALUES ({qs})", none_if_nan(df))
    cn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default=os.environ.get("NEUROTARGET_CONN"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    a = ap.parse_args()
    if not a.conn:
        raise SystemExit("Pass --conn or set NEUROTARGET_CONN")
    d = Path(a.data)

    # schema script creates the database, so connect to master first
    master = re.sub(r"(?i)database=[^;]*;?", "", a.conn)
    cn = pyodbc.connect(master, autocommit=True)
    print("schema ..."); run_script(cn, ROOT / "sql" / "01_schema.sql")
    print("procedures ..."); run_script(cn, ROOT / "sql" / "02_procedures.sql")
    cn.close()

    cn = pyodbc.connect(master.rstrip(";") + ";Database=NeuroTarget;", autocommit=False)
    bulk(cn, "nt.Condition", pd.read_csv(d / "conditions.csv"))
    t = pd.read_csv(d / "targets.csv")
    bulk(cn, "nt.Target", t[["TargetId", "ConditionId", "Code", "Name", "Laterality", "AtlasX", "AtlasY", "AtlasZ",
                              "Emax", "Floor", "SigmaMm", "NoGoName", "NoGo"]])
    bulk(cn, "nt.FeatureDef", pd.read_csv(d / "feature_def.csv"))
    p = pd.read_csv(d / "patients.csv")
    p["PainSide"] = p["PainSide"].where(p["PainSide"].notna(), None)
    bulk(cn, "nt.Patient", p)

    print("scans (via nt.usp_IngestScan) ...")
    cur = cn.cursor()
    for s in pd.read_csv(d / "scans.csv").itertuples():
        cur.execute("EXEC nt.usp_IngestScan ?,?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?",
                    s.ScanId, s.PatientId, s.Modality, s.AcquiredOn, s.VoxelMm,
                    s.ACX, s.ACY, s.ACZ, s.PCX, s.PCY, s.PCZ, s.ScaleX, s.ScaleY, s.ScaleZ, s.RawFeatures)
    cn.commit()

    bulk(cn, "nt.Implant", pd.read_csv(d / "implants.csv"))
    bulk(cn, "nt.Contact", pd.read_csv(d / "contacts.csv"))
    bulk(cn, "nt.Programming", pd.read_csv(d / "programming.csv"))
    bulk(cn, "nt.Outcome", pd.read_csv(d / "outcomes.csv"))
    bulk(cn, "nt.SimTruth", pd.read_csv(d / "sim_truth.csv"))

    print("referral volumes ...")
    for f in sorted((d / "volumes").glob("scan_*.bin")):
        sid = int(f.stem.split("_")[1])
        cur.execute("INSERT nt.ScanVolume (ScanId, DimX, DimY, DimZ, OriginX, OriginY, OriginZ, Encoding, Data) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", sid, *A.DIMS, *map(float, A.FOV_MIN), "zlib-uint8",
                    pyodbc.Binary(f.read_bytes()))
    cn.commit()
    n = cn.cursor().execute("SELECT COUNT(*) FROM nt.vHistoricalEvidence").fetchval()
    print(f"done - {n} implants of evidence")


if __name__ == "__main__":
    main()

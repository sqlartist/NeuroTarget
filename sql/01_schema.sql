/* =====================================================================================
   NeuroTarget — outcome-driven electrode placement suggestions
   Schema. Requires SQL Server 2025 (native VECTOR type, VECTOR_DISTANCE).

   SIMULATED DATA ONLY. Research / teaching prototype, not a medical device.

   Coordinate conventions (all mm, RAS: +X right, +Y anterior, +Z superior)
     Native   : the patient's own MRI space.
     Template : AC-PC space, origin at the mid-commissural point (MCP), proportionally
                scaled to a reference brain. Template = Scale * (Native - MCP).
                Atlas targets are stored for the RIGHT hemisphere; left mirrors X.
   ===================================================================================== */
IF DB_ID(N'NeuroTarget') IS NULL
    CREATE DATABASE NeuroTarget;
GO
USE NeuroTarget;
GO
IF SCHEMA_ID(N'nt') IS NULL EXEC (N'CREATE SCHEMA nt AUTHORIZATION dbo;');
GO

/* ---------- drop in dependency order (re-runnable) ---------- */
DROP VIEW  IF EXISTS nt.vHistoricalEvidence;
DROP TABLE IF EXISTS nt.Suggestion, nt.Outcome, nt.Programming, nt.Contact, nt.Implant,
                     nt.SimTruth, nt.ScanVolume, nt.Scan, nt.Patient, nt.GridPoint,
                     nt.FeatureDef, nt.Target, nt.Condition;
DROP SEQUENCE IF EXISTS nt.seqImplant;
GO

/* ---------- reference data ---------- */
CREATE TABLE nt.Condition
(
    ConditionId     TINYINT       NOT NULL CONSTRAINT PK_Condition PRIMARY KEY,
    Code            VARCHAR(10)   NOT NULL CONSTRAINT UQ_Condition_Code UNIQUE,
    Name            NVARCHAR(100) NOT NULL,
    OutcomeMeasure  NVARCHAR(200) NOT NULL
);

CREATE TABLE nt.Target
(
    TargetId        TINYINT       NOT NULL CONSTRAINT PK_Target PRIMARY KEY,
    ConditionId     TINYINT       NOT NULL CONSTRAINT FK_Target_Condition REFERENCES nt.Condition,
    Code            VARCHAR(20)   NOT NULL CONSTRAINT UQ_Target_Code UNIQUE,
    Name            NVARCHAR(100) NOT NULL,
    Laterality      VARCHAR(12)   NOT NULL,              -- Ipsilateral | Bilateral
    AtlasX          FLOAT         NOT NULL,              -- right hemisphere, template mm
    AtlasY          FLOAT         NOT NULL,
    AtlasZ          FLOAT         NOT NULL,
    -- dose-response model used by the fit: Eff = Floor + (Emax-Floor)*exp(-d^2 / 2 Sigma^2)
    Emax            FLOAT         NOT NULL,
    Floor           FLOAT         NOT NULL,
    SigmaMm         FLOAT         NOT NULL,
    NoGoName        NVARCHAR(200) NULL,                  -- the structure that causes side effects
    NoGo            NVARCHAR(100) NULL                   -- sim only: JSON offset from sweet spot
);

/* One row per image-derived feature. PopMean/PopStd are learned from the historical library
   and used to z-score any new scan; Weight lets you emphasise features that matter. */
CREATE TABLE nt.FeatureDef
(
    FeatureNo       TINYINT       NOT NULL CONSTRAINT PK_FeatureDef PRIMARY KEY,
    Code            VARCHAR(40)   NOT NULL CONSTRAINT UQ_FeatureDef_Code UNIQUE,
    Description     NVARCHAR(200) NOT NULL,
    Unit            VARCHAR(10)   NOT NULL,
    PopMean         FLOAT         NOT NULL,
    PopStd          FLOAT         NOT NULL,
    Weight          FLOAT         NOT NULL CONSTRAINT DF_FeatureDef_Weight DEFAULT (1.0)
);

/* Candidate offsets (mm) for the sweet-spot search. Pass 1 = 1 mm over +-5 mm (1331 pts),
   Pass 2 = 0.25 mm over +-1 mm around the pass-1 winner (729 pts). */
CREATE TABLE nt.GridPoint
(
    GridId  INT      NOT NULL IDENTITY CONSTRAINT PK_GridPoint PRIMARY KEY,
    Pass    TINYINT  NOT NULL,
    DX      FLOAT    NOT NULL,
    DY      FLOAT    NOT NULL,
    DZ      FLOAT    NOT NULL,
    L1      AS (ABS(DX) + ABS(DY) + ABS(DZ)) PERSISTED    -- tie-break: prefer closer to centre
);
CREATE INDEX IX_GridPoint_Pass ON nt.GridPoint (Pass) INCLUDE (DX, DY, DZ, L1);

/* ---------- patients and scans ---------- */
CREATE TABLE nt.Patient
(
    PatientId       INT           NOT NULL CONSTRAINT PK_Patient PRIMARY KEY,
    ConditionId     TINYINT       NOT NULL CONSTRAINT FK_Patient_Condition REFERENCES nt.Condition,
    Sex             CHAR(1)       NOT NULL,
    AgeAtScan       TINYINT       NOT NULL,
    PainSide        CHAR(1)       NULL,                  -- cluster headache: side of attacks
    Status          VARCHAR(12)   NOT NULL CONSTRAINT CK_Patient_Status CHECK (Status IN ('Historical','Referral')),
    IsSimulated     BIT           NOT NULL CONSTRAINT DF_Patient_IsSimulated DEFAULT (1)
);

CREATE TABLE nt.Scan
(
    ScanId          INT           NOT NULL CONSTRAINT PK_Scan PRIMARY KEY,
    PatientId       INT           NOT NULL CONSTRAINT FK_Scan_Patient REFERENCES nt.Patient,
    Modality        VARCHAR(20)   NOT NULL,              -- MRI_T1_3D
    AcquiredOn      DATE          NOT NULL,
    VoxelMm         FLOAT         NOT NULL,
    ACX FLOAT NOT NULL, ACY FLOAT NOT NULL, ACZ FLOAT NOT NULL,   -- landmarks, native mm
    PCX FLOAT NOT NULL, PCY FLOAT NOT NULL, PCZ FLOAT NOT NULL,
    MCPX AS ((ACX + PCX) / 2.0) PERSISTED,
    MCPY AS ((ACY + PCY) / 2.0) PERSISTED,
    MCPZ AS ((ACZ + PCZ) / 2.0) PERSISTED,
    ScaleX FLOAT NOT NULL, ScaleY FLOAT NOT NULL, ScaleZ FLOAT NOT NULL,  -- native -> template
    RawFeatures     NVARCHAR(MAX) NOT NULL CONSTRAINT CK_Scan_RawFeatures CHECK (ISJSON(RawFeatures) = 1),
    Features        VECTOR(14)    NOT NULL               -- z-scored, weighted anatomy fingerprint
);
CREATE INDEX IX_Scan_Patient ON nt.Scan (PatientId);

/* The image itself normally stays in PACS; for the demo the new referrals' volumes live here. */
CREATE TABLE nt.ScanVolume
(
    ScanId          INT            NOT NULL CONSTRAINT PK_ScanVolume PRIMARY KEY
                                   CONSTRAINT FK_ScanVolume_Scan REFERENCES nt.Scan,
    DimX SMALLINT NOT NULL, DimY SMALLINT NOT NULL, DimZ SMALLINT NOT NULL,
    OriginX FLOAT NOT NULL, OriginY FLOAT NOT NULL, OriginZ FLOAT NOT NULL,  -- native mm of voxel (0,0,0) corner
    Encoding        VARCHAR(20)    NOT NULL,              -- zlib-uint8, C order (x,y,z)
    Data            VARBINARY(MAX) NOT NULL
);

/* ---------- surgery, programming, outcome ---------- */
CREATE SEQUENCE nt.seqImplant AS INT START WITH 1000000;

CREATE TABLE nt.Implant
(
    ImplantId       INT           NOT NULL CONSTRAINT PK_Implant PRIMARY KEY
                                  CONSTRAINT DF_Implant_Id DEFAULT (NEXT VALUE FOR nt.seqImplant),
    PatientId       INT           NOT NULL CONSTRAINT FK_Implant_Patient REFERENCES nt.Patient,
    TargetId        TINYINT       NOT NULL CONSTRAINT FK_Implant_Target REFERENCES nt.Target,
    Hemisphere      CHAR(1)       NOT NULL CONSTRAINT CK_Implant_Hemi CHECK (Hemisphere IN ('L','R')),
    ImplantedOn     DATE          NOT NULL,
    PlannedX FLOAT NULL, PlannedY FLOAT NULL, PlannedZ FLOAT NULL,          -- template mm (signed X)
    SuggestionId    INT           NULL,
    CONSTRAINT UQ_Implant UNIQUE (PatientId, TargetId, Hemisphere)
);

/* Post-op CT-localised contact centres, template mm (signed X). Contact 0 = deepest. */
CREATE TABLE nt.Contact
(
    ImplantId       INT     NOT NULL CONSTRAINT FK_Contact_Implant REFERENCES nt.Implant,
    ContactNo       TINYINT NOT NULL,
    X FLOAT NOT NULL, Y FLOAT NOT NULL, Z FLOAT NOT NULL,
    CONSTRAINT PK_Contact PRIMARY KEY (ImplantId, ContactNo)
);

CREATE TABLE nt.Programming
(
    ImplantId       INT     NOT NULL CONSTRAINT PK_Programming PRIMARY KEY
                            CONSTRAINT FK_Programming_Implant REFERENCES nt.Implant,
    ActiveContact   TINYINT NOT NULL,
    AmplitudeMa     DECIMAL(4,1) NOT NULL,
    PulseWidthUs    SMALLINT     NOT NULL,
    FrequencyHz     SMALLINT     NOT NULL
);

CREATE TABLE nt.Outcome
(
    OutcomeId       INT          NOT NULL IDENTITY CONSTRAINT PK_Outcome PRIMARY KEY,
    ImplantId       INT          NOT NULL CONSTRAINT FK_Outcome_Implant REFERENCES nt.Implant,
    MonthsPostOp    TINYINT      NOT NULL,
    EfficacyPct     FLOAT        NOT NULL,   -- condition-specific, see nt.Condition.OutcomeMeasure
    SideEffectScore FLOAT        NOT NULL,   -- 0 (none) .. 10 (intolerable)
    PatientRating   TINYINT      NOT NULL,   -- 0 .. 10 patient-reported
    NetBenefit      AS (EfficacyPct / 100.0 - 0.04 * SideEffectScore) PERSISTED,
    RecordedAt      DATETIME2(0) NOT NULL CONSTRAINT DF_Outcome_RecordedAt DEFAULT (SYSUTCDATETIME())
);
CREATE INDEX IX_Outcome_Implant ON nt.Outcome (ImplantId, MonthsPostOp DESC);

/* Every suggestion made, so the surgeon's decision and the eventual result can be audited. */
CREATE TABLE nt.Suggestion
(
    SuggestionId        INT          NOT NULL IDENTITY CONSTRAINT PK_Suggestion PRIMARY KEY,
    ScanId              INT          NOT NULL CONSTRAINT FK_Suggestion_Scan REFERENCES nt.Scan,
    TargetId            TINYINT      NOT NULL CONSTRAINT FK_Suggestion_Target REFERENCES nt.Target,
    Hemisphere          CHAR(1)      NOT NULL,
    K                   INT          NOT NULL,
    SideEffectPenalty   FLOAT        NOT NULL,
    TemplateX FLOAT NOT NULL, TemplateY FLOAT NOT NULL, TemplateZ FLOAT NOT NULL,
    NativeX   FLOAT NOT NULL, NativeY   FLOAT NOT NULL, NativeZ   FLOAT NOT NULL,
    SdX FLOAT NULL, SdY FLOAT NULL, SdZ FLOAT NULL,
    AtlasShiftMm        FLOAT        NOT NULL,
    EffectiveN          FLOAT        NOT NULL,
    PredictedEfficacy   FLOAT        NULL,
    PredictedSideEffect FLOAT        NULL,
    FitRmse             FLOAT        NOT NULL,
    CreatedAt           DATETIME2(0) NOT NULL CONSTRAINT DF_Suggestion_CreatedAt DEFAULT (SYSUTCDATETIME()),
    AcceptedBySurgeon   BIT          NULL
);

/* SIMULATION ONLY: each patient's hidden true optimal point, used to score the method. */
CREATE TABLE nt.SimTruth
(
    PatientId   INT     NOT NULL CONSTRAINT FK_SimTruth_Patient REFERENCES nt.Patient,
    TargetId    TINYINT NOT NULL CONSTRAINT FK_SimTruth_Target REFERENCES nt.Target,
    Hemisphere  CHAR(1) NOT NULL,
    SweetX FLOAT NOT NULL, SweetY FLOAT NOT NULL, SweetZ FLOAT NOT NULL,   -- template mm, signed X
    CONSTRAINT PK_SimTruth PRIMARY KEY (PatientId, TargetId, Hemisphere)
);
GO

/* Evidence = every implant with a known active contact and its latest outcome.
   X is folded to the right hemisphere so left and right leads pool together. */
CREATE VIEW nt.vHistoricalEvidence
AS
SELECT  i.ImplantId, i.PatientId, s.ScanId, i.TargetId, i.Hemisphere,
        ABS(c.X) AS X, c.Y, c.Z,
        o.EfficacyPct, o.SideEffectScore, o.PatientRating, o.NetBenefit,
        s.Features
FROM    nt.Implant      AS i
JOIN    nt.Programming  AS p ON p.ImplantId = i.ImplantId
JOIN    nt.Contact      AS c ON c.ImplantId = i.ImplantId AND c.ContactNo = p.ActiveContact
JOIN    nt.Scan         AS s ON s.PatientId = i.PatientId
CROSS APPLY (SELECT TOP (1) o.EfficacyPct, o.SideEffectScore, o.PatientRating, o.NetBenefit
             FROM nt.Outcome AS o
             WHERE o.ImplantId = i.ImplantId
             ORDER BY o.MonthsPostOp DESC, o.OutcomeId DESC) AS o;
GO

/* Grid of candidate offsets */
WITH n AS (SELECT TOP (21) CAST(ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1 AS FLOAT) AS i
           FROM sys.all_objects)
INSERT nt.GridPoint (Pass, DX, DY, DZ)
SELECT 1, x.i - 5, y.i - 5, z.i - 5
FROM   n AS x CROSS JOIN n AS y CROSS JOIN n AS z
WHERE  x.i <= 10 AND y.i <= 10 AND z.i <= 10
UNION ALL
SELECT 2, (x.i - 4) * 0.25, (y.i - 4) * 0.25, (z.i - 4) * 0.25
FROM   n AS x CROSS JOIN n AS y CROSS JOIN n AS z
WHERE  x.i <= 8 AND y.i <= 8 AND z.i <= 8;
GO

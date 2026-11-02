/* =====================================================================================
   NeuroTarget — procedures
     nt.usp_IngestScan            z-score raw image features -> VECTOR(14), store scan
     nt.usp_SuggestPlacement      nearest-anatomy neighbours -> fitted sweet spot
     nt.usp_RecordImplantOutcome  close the loop: implant + contacts + programming + outcome
     nt.usp_EvaluateLOO           leave-one-out accuracy vs atlas (simulation ground truth)
   The maths is mirrored line-for-line in python/neurosim/engine.py.
   ===================================================================================== */
USE NeuroTarget;
GO

/* -------------------------------------------------------------------------------------
   Ingest: RawFeatures is the JSON object produced by the image pipeline, e.g.
   {"brain_w_mm":136.2, "acpc_mm":24.1, ... "age":58}
   ------------------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE nt.usp_IngestScan
    @ScanId      INT,
    @PatientId   INT,
    @Modality    VARCHAR(20),
    @AcquiredOn  DATE,
    @VoxelMm     FLOAT,
    @ACX FLOAT, @ACY FLOAT, @ACZ FLOAT,
    @PCX FLOAT, @PCY FLOAT, @PCZ FLOAT,
    @ScaleX FLOAT, @ScaleY FLOAT, @ScaleZ FLOAT,
    @RawFeatures NVARCHAR(MAX)
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    DECLARE @missing NVARCHAR(400) =
        (SELECT STRING_AGG(f.Code, N', ')
         FROM nt.FeatureDef AS f
         WHERE JSON_VALUE(@RawFeatures, N'$.' + f.Code) IS NULL);
    IF @missing IS NOT NULL
    BEGIN
        DECLARE @msg NVARCHAR(500) = N'RawFeatures is missing: ' + @missing;
        THROW 50010, @msg, 1;
    END;

    -- z-score each feature against the library population, apply its weight, and build the
    -- JSON array literal that SQL Server 2025 casts straight to VECTOR(n)
    DECLARE @vec NVARCHAR(MAX) =
        N'[' + (SELECT STRING_AGG(CONVERT(VARCHAR(40), CAST(
                        f.Weight * (CAST(JSON_VALUE(@RawFeatures, N'$.' + f.Code) AS FLOAT) - f.PopMean)
                        / NULLIF(f.PopStd, 0) AS DECIMAL(18,6))), ',')
                       WITHIN GROUP (ORDER BY f.FeatureNo)
                FROM nt.FeatureDef AS f) + N']';

    INSERT nt.Scan (ScanId, PatientId, Modality, AcquiredOn, VoxelMm,
                    ACX, ACY, ACZ, PCX, PCY, PCZ, ScaleX, ScaleY, ScaleZ, RawFeatures, Features)
    VALUES (@ScanId, @PatientId, @Modality, @AcquiredOn, @VoxelMm,
            @ACX, @ACY, @ACZ, @PCX, @PCY, @PCZ, @ScaleX, @ScaleY, @ScaleZ, @RawFeatures,
            CAST(@vec AS VECTOR(14)));
END;
GO

/* -------------------------------------------------------------------------------------
   Suggest a placement for one scan / target / hemisphere.

   1. Find the @K implants (same target, other patients) whose scans are nearest in the
      anatomy vector space (VECTOR_DISTANCE euclidean).
   2. Weight them with a Gaussian kernel on that distance (bandwidth = mean distance).
   3. Each neighbour is one observation: active contact position -> efficacy, side effects.
      Grid-search the sweet spot s that best explains those outcomes under the target's
      dose-response curve, penalised by side-effect burden reported near s.
      Coarse 1 mm pass around the atlas, then a 0.25 mm pass around the winner.
   4. Map the answer into the patient's own scan (native mm) and log it.

   Result set 1: the suggestion. Result set 2: the neighbours and their weights.
   ------------------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE nt.usp_SuggestPlacement
    @ScanId             INT,
    @TargetId           TINYINT,
    @Hemisphere         CHAR(1) = 'R',
    @K                  INT     = 40,
    @SideEffectPenalty  FLOAT   = 100,
    @Save               BIT     = 1,
    @SuggestionId       INT     = NULL OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    IF @Hemisphere NOT IN ('L','R') THROW 50020, N'@Hemisphere must be L or R', 1;

    DECLARE @q VECTOR(14), @PatientId INT,
            @mx FLOAT, @my FLOAT, @mz FLOAT, @sx FLOAT, @sy FLOAT, @sz FLOAT;
    SELECT  @q = s.Features, @PatientId = s.PatientId,
            @mx = s.MCPX, @my = s.MCPY, @mz = s.MCPZ,
            @sx = s.ScaleX, @sy = s.ScaleY, @sz = s.ScaleZ
    FROM    nt.Scan AS s
    WHERE   s.ScanId = @ScanId;
    IF @PatientId IS NULL THROW 50021, N'Unknown @ScanId', 1;

    DECLARE @ax FLOAT, @ay FLOAT, @az FLOAT, @Floor FLOAT, @Emax FLOAT, @Sigma FLOAT;
    SELECT  @ax = AtlasX, @ay = AtlasY, @az = AtlasZ, @Floor = Floor, @Emax = Emax, @Sigma = SigmaMm
    FROM    nt.Target WHERE TargetId = @TargetId;
    IF @ax IS NULL THROW 50022, N'Unknown @TargetId', 1;

    /* 1. nearest anatomy */
    CREATE TABLE #nb
    (
        Rn INT NOT NULL, ImplantId INT NOT NULL, PatientId INT NOT NULL, ScanId INT NOT NULL,
        Hemisphere CHAR(1) NOT NULL, X FLOAT NOT NULL, Y FLOAT NOT NULL, Z FLOAT NOT NULL,
        EfficacyPct FLOAT NOT NULL, SideEffectScore FLOAT NOT NULL, PatientRating TINYINT NOT NULL,
        NetBenefit FLOAT NOT NULL, Dist FLOAT NOT NULL, KernelW FLOAT NULL, NearW FLOAT NULL
    );

    INSERT #nb (Rn, ImplantId, PatientId, ScanId, Hemisphere, X, Y, Z,
                EfficacyPct, SideEffectScore, PatientRating, NetBenefit, Dist)
    SELECT TOP (@K)
           ROW_NUMBER() OVER (ORDER BY d.Dist, d.ImplantId),
           d.ImplantId, d.PatientId, d.ScanId, d.Hemisphere, d.X, d.Y, d.Z,
           d.EfficacyPct, d.SideEffectScore, d.PatientRating, d.NetBenefit, d.Dist
    FROM ( SELECT e.ImplantId, e.PatientId, e.ScanId, e.Hemisphere, e.X, e.Y, e.Z,
                  e.EfficacyPct, e.SideEffectScore, e.PatientRating, e.NetBenefit,
                  VECTOR_DISTANCE('euclidean', e.Features, @q) AS Dist
           FROM   nt.vHistoricalEvidence AS e
           WHERE  e.TargetId = @TargetId
             AND  e.PatientId <> @PatientId ) AS d
    ORDER BY d.Dist, d.ImplantId;

    IF NOT EXISTS (SELECT 1 FROM #nb) THROW 50023, N'No historical evidence for this target', 1;

    /* 2. similarity kernel */
    DECLARE @h FLOAT = (SELECT AVG(Dist) FROM #nb);
    IF @h IS NULL OR @h = 0 SET @h = 1.0;
    UPDATE #nb SET KernelW = EXP(-SQUARE(Dist / @h));
    DECLARE @sumW FLOAT = (SELECT SUM(KernelW) FROM #nb);
    DECLARE @twoS2 FLOAT = 2.0 * @Sigma * @Sigma, @twoR2 FLOAT = 2.0 * 1.5 * 1.5;

    /* 3a. coarse pass around the atlas target */
    DECLARE @cx FLOAT, @cy FLOAT, @cz FLOAT;
    SELECT TOP (1) @cx = @ax + g.DX, @cy = @ay + g.DY, @cz = @az + g.DZ
    FROM   nt.GridPoint AS g
    CROSS APPLY (
        SELECT SUM(n.KernelW * SQUARE(@Floor + (@Emax - @Floor) * EXP(-dd.D2 / @twoS2) - n.EfficacyPct)) / @sumW AS Fit,
               SUM(n.KernelW * n.SideEffectScore * EXP(-dd.D2 / @twoR2)) / @sumW AS Risk
        FROM   #nb AS n
        CROSS APPLY (SELECT SQUARE(@ax + g.DX - n.X) + SQUARE(@ay + g.DY - n.Y) + SQUARE(@az + g.DZ - n.Z) AS D2) AS dd
    ) AS s
    WHERE  g.Pass = 1
    ORDER BY s.Fit + @SideEffectPenalty * s.Risk, g.L1, g.DX, g.DY, g.DZ;   -- = np.lexsort + grid order

    /* 3b. fine pass around the coarse winner */
    DECLARE @bx FLOAT, @by FLOAT, @bz FLOAT, @Fit FLOAT;
    SELECT TOP (1) @bx = @cx + g.DX, @by = @cy + g.DY, @bz = @cz + g.DZ, @Fit = s.Fit
    FROM   nt.GridPoint AS g
    CROSS APPLY (
        SELECT SUM(n.KernelW * SQUARE(@Floor + (@Emax - @Floor) * EXP(-dd.D2 / @twoS2) - n.EfficacyPct)) / @sumW AS Fit,
               SUM(n.KernelW * n.SideEffectScore * EXP(-dd.D2 / @twoR2)) / @sumW AS Risk
        FROM   #nb AS n
        CROSS APPLY (SELECT SQUARE(@cx + g.DX - n.X) + SQUARE(@cy + g.DY - n.Y) + SQUARE(@cz + g.DZ - n.Z) AS D2) AS dd
    ) AS s
    WHERE  g.Pass = 2
    ORDER BY s.Fit + @SideEffectPenalty * s.Risk, g.L1, g.DX, g.DY, g.DZ;

    /* spread of good placements, and outcomes seen close to the answer */
    UPDATE #nb SET NearW = KernelW * EXP(-(SQUARE(X - @bx) + SQUARE(Y - @by) + SQUARE(Z - @bz)) / @twoR2);

    DECLARE @SdX FLOAT, @SdY FLOAT, @SdZ FLOAT, @Neff FLOAT, @PredEff FLOAT, @PredSE FLOAT;
    SELECT  @SdX = SQRT(SUM(g.W * SQUARE(X - @bx)) / NULLIF(SUM(g.W), 0)),
            @SdY = SQRT(SUM(g.W * SQUARE(Y - @by)) / NULLIF(SUM(g.W), 0)),
            @SdZ = SQRT(SUM(g.W * SQUARE(Z - @bz)) / NULLIF(SUM(g.W), 0)),
            @Neff = SQUARE(SUM(KernelW)) / SUM(SQUARE(KernelW)),
            @PredEff = CASE WHEN SUM(NearW) > 1e-9 THEN SUM(NearW * EfficacyPct) / SUM(NearW) END,
            @PredSE  = CASE WHEN SUM(NearW) > 1e-9 THEN SUM(NearW * SideEffectScore) / SUM(NearW) END
    FROM    #nb
    CROSS APPLY (SELECT KernelW * CASE WHEN NetBenefit > 0 THEN NetBenefit ELSE 0 END AS W) AS g;

    /* 4. into the patient's own scan */
    DECLARE @sgn FLOAT = CASE @Hemisphere WHEN 'L' THEN -1.0 ELSE 1.0 END;
    DECLARE @tx FLOAT = @sgn * @bx, @ty FLOAT = @by, @tz FLOAT = @bz;
    DECLARE @nx FLOAT = @tx / @sx + @mx, @ny FLOAT = @ty / @sy + @my, @nz FLOAT = @tz / @sz + @mz;
    DECLARE @shift FLOAT = SQRT(SQUARE(@bx - @ax) + SQUARE(@by - @ay) + SQUARE(@bz - @az));

    IF @Save = 1
    BEGIN
        INSERT nt.Suggestion (ScanId, TargetId, Hemisphere, K, SideEffectPenalty,
                              TemplateX, TemplateY, TemplateZ, NativeX, NativeY, NativeZ,
                              SdX, SdY, SdZ, AtlasShiftMm, EffectiveN,
                              PredictedEfficacy, PredictedSideEffect, FitRmse)
        VALUES (@ScanId, @TargetId, @Hemisphere, @K, @SideEffectPenalty,
                @tx, @ty, @tz, @nx, @ny, @nz, @SdX, @SdY, @SdZ, @shift, @Neff,
                @PredEff, @PredSE, SQRT(@Fit));
        SET @SuggestionId = SCOPE_IDENTITY();
    END;

    SELECT  @SuggestionId AS SuggestionId, @ScanId AS ScanId, @TargetId AS TargetId,
            @Hemisphere AS Hemisphere, @K AS K,
            @tx AS TemplateX, @ty AS TemplateY, @tz AS TemplateZ,
            @nx AS NativeX, @ny AS NativeY, @nz AS NativeZ,
            @sgn * @ax AS AtlasX, @ay AS AtlasY, @az AS AtlasZ,
            @SdX AS SdX, @SdY AS SdY, @SdZ AS SdZ,
            @shift AS AtlasShiftMm, @Neff AS EffectiveN,
            @PredEff AS PredictedEfficacy, @PredSE AS PredictedSideEffect,
            SQRT(@Fit) AS FitRmse, @h AS Bandwidth;

    SELECT  Rn, ImplantId, PatientId, ScanId, Hemisphere, X, Y, Z,
            EfficacyPct, SideEffectScore, PatientRating, NetBenefit, Dist, KernelW, NearW
    FROM    #nb
    ORDER BY Rn;
END;
GO

/* -------------------------------------------------------------------------------------
   Close the loop. @ContactsJson: [{"n":0,"x":..,"y":..,"z":..}, ...] template mm, signed X
   (post-op CT localisation). The patient becomes part of the evidence for everyone after.
   ------------------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE nt.usp_RecordImplantOutcome
    @PatientId        INT,
    @TargetId         TINYINT,
    @Hemisphere       CHAR(1),
    @ImplantedOn      DATE,
    @ContactsJson     NVARCHAR(MAX),
    @ActiveContact    TINYINT,
    @AmplitudeMa      DECIMAL(4,1),
    @PulseWidthUs     SMALLINT,
    @FrequencyHz      SMALLINT,
    @MonthsPostOp     TINYINT,
    @EfficacyPct      FLOAT,
    @SideEffectScore  FLOAT,
    @PatientRating    TINYINT,
    @SuggestionId     INT = NULL,
    @ImplantId        INT = NULL OUTPUT
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;
    BEGIN TRAN;

    SELECT @ImplantId = ImplantId FROM nt.Implant
    WHERE PatientId = @PatientId AND TargetId = @TargetId AND Hemisphere = @Hemisphere;

    IF @ImplantId IS NULL
    BEGIN
        SET @ImplantId = NEXT VALUE FOR nt.seqImplant;
        DECLARE @px FLOAT, @py FLOAT, @pz FLOAT;
        SELECT @px = TemplateX, @py = TemplateY, @pz = TemplateZ
        FROM nt.Suggestion WHERE SuggestionId = @SuggestionId;

        INSERT nt.Implant (ImplantId, PatientId, TargetId, Hemisphere, ImplantedOn,
                           PlannedX, PlannedY, PlannedZ, SuggestionId)
        VALUES (@ImplantId, @PatientId, @TargetId, @Hemisphere, @ImplantedOn, @px, @py, @pz, @SuggestionId);

        INSERT nt.Contact (ImplantId, ContactNo, X, Y, Z)
        SELECT @ImplantId, j.n, j.x, j.y, j.z
        FROM OPENJSON(@ContactsJson) WITH (n TINYINT '$.n', x FLOAT '$.x', y FLOAT '$.y', z FLOAT '$.z') AS j;

        IF @SuggestionId IS NOT NULL
            UPDATE nt.Suggestion SET AcceptedBySurgeon = 1 WHERE SuggestionId = @SuggestionId;
    END;

    MERGE nt.Programming AS t
    USING (SELECT @ImplantId AS ImplantId) AS s ON t.ImplantId = s.ImplantId
    WHEN MATCHED THEN UPDATE SET ActiveContact = @ActiveContact, AmplitudeMa = @AmplitudeMa,
                                 PulseWidthUs = @PulseWidthUs, FrequencyHz = @FrequencyHz
    WHEN NOT MATCHED THEN INSERT (ImplantId, ActiveContact, AmplitudeMa, PulseWidthUs, FrequencyHz)
                          VALUES (@ImplantId, @ActiveContact, @AmplitudeMa, @PulseWidthUs, @FrequencyHz);

    INSERT nt.Outcome (ImplantId, MonthsPostOp, EfficacyPct, SideEffectScore, PatientRating)
    VALUES (@ImplantId, @MonthsPostOp, @EfficacyPct, @SideEffectScore, @PatientRating);

    UPDATE nt.Patient SET Status = 'Historical' WHERE PatientId = @PatientId;
    COMMIT;
END;
GO

/* -------------------------------------------------------------------------------------
   Leave-one-patient-out evaluation for a whole target, set-based.
   For every historical implant: rebuild its suggestion from everyone else, then score
   suggestion and plain atlas against the simulated true sweet spot.
   Result set 1: summary. Result set 2: per implant.
   ------------------------------------------------------------------------------------- */
CREATE OR ALTER PROCEDURE nt.usp_EvaluateLOO
    @TargetId           TINYINT,
    @K                  INT   = 40,
    @SideEffectPenalty  FLOAT = 100
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @ax FLOAT, @ay FLOAT, @az FLOAT, @Floor FLOAT, @Emax FLOAT, @Sigma FLOAT, @NoGo NVARCHAR(100);
    SELECT  @ax = AtlasX, @ay = AtlasY, @az = AtlasZ, @Floor = Floor, @Emax = Emax, @Sigma = SigmaMm, @NoGo = NoGo
    FROM    nt.Target WHERE TargetId = @TargetId;
    DECLARE @gx FLOAT = CAST(JSON_VALUE(@NoGo, '$[0]') AS FLOAT),
            @gy FLOAT = CAST(JSON_VALUE(@NoGo, '$[1]') AS FLOAT),
            @gz FLOAT = CAST(JSON_VALUE(@NoGo, '$[2]') AS FLOAT);
    DECLARE @twoS2 FLOAT = 2.0 * @Sigma * @Sigma, @twoR2 FLOAT = 2.0 * 1.5 * 1.5;

    SELECT e.ImplantId AS QId, e.PatientId, e.Hemisphere, e.X, e.Y, e.Z
    INTO   #q
    FROM   nt.vHistoricalEvidence AS e
    WHERE  e.TargetId = @TargetId;

    SELECT q.QId, nb.ImplantId, nb.X, nb.Y, nb.Z, nb.EfficacyPct, nb.SideEffectScore, nb.NetBenefit, nb.Dist,
           CAST(NULL AS FLOAT) AS KernelW
    INTO   #nb
    FROM   #q AS q
    JOIN   nt.vHistoricalEvidence AS qe ON qe.ImplantId = q.QId
    CROSS APPLY (
        SELECT TOP (@K) e.ImplantId, e.X, e.Y, e.Z, e.EfficacyPct, e.SideEffectScore, e.NetBenefit,
               VECTOR_DISTANCE('euclidean', e.Features, qe.Features) AS Dist
        FROM   nt.vHistoricalEvidence AS e
        WHERE  e.TargetId = @TargetId AND e.PatientId <> q.PatientId
        ORDER BY Dist, e.ImplantId
    ) AS nb;

    WITH b AS (SELECT QId, KernelW, Dist, AVG(Dist) OVER (PARTITION BY QId) AS H FROM #nb)
    UPDATE b SET KernelW = EXP(-SQUARE(Dist / CASE WHEN H > 0 THEN H ELSE 1.0 END));

    SELECT QId, SUM(KernelW) AS SumW, SQUARE(SUM(KernelW)) / SUM(SQUARE(KernelW)) AS Neff
    INTO #w FROM #nb GROUP BY QId;

    /* coarse */
    WITH s AS (
        SELECT n.QId, g.DX, g.DY, g.DZ, g.L1,
               SUM(n.KernelW * SQUARE(@Floor + (@Emax - @Floor) * EXP(-d.D2 / @twoS2) - n.EfficacyPct)) AS FitW,
               SUM(n.KernelW * n.SideEffectScore * EXP(-d.D2 / @twoR2)) AS RiskW
        FROM   #nb AS n
        CROSS JOIN nt.GridPoint AS g
        CROSS APPLY (SELECT SQUARE(@ax + g.DX - n.X) + SQUARE(@ay + g.DY - n.Y) + SQUARE(@az + g.DZ - n.Z) AS D2) AS d
        WHERE  g.Pass = 1
        GROUP BY n.QId, g.GridId, g.DX, g.DY, g.DZ, g.L1
    ), r AS (
        SELECT s.QId, @ax + s.DX AS CX, @ay + s.DY AS CY, @az + s.DZ AS CZ,
               ROW_NUMBER() OVER (PARTITION BY s.QId
                                  ORDER BY (s.FitW + @SideEffectPenalty * s.RiskW) / w.SumW, s.L1, s.DX, s.DY, s.DZ) AS rn
        FROM s JOIN #w AS w ON w.QId = s.QId
    )
    SELECT QId, CX, CY, CZ INTO #c FROM r WHERE rn = 1;

    /* fine */
    WITH s AS (
        SELECT n.QId, c.CX + g.DX AS FX, c.CY + g.DY AS FY, c.CZ + g.DZ AS FZ, g.L1,
               SUM(n.KernelW * SQUARE(@Floor + (@Emax - @Floor) * EXP(-d.D2 / @twoS2) - n.EfficacyPct)) AS FitW,
               SUM(n.KernelW * n.SideEffectScore * EXP(-d.D2 / @twoR2)) AS RiskW
        FROM   #nb AS n
        JOIN   #c  AS c ON c.QId = n.QId
        CROSS JOIN nt.GridPoint AS g
        CROSS APPLY (SELECT SQUARE(c.CX + g.DX - n.X) + SQUARE(c.CY + g.DY - n.Y) + SQUARE(c.CZ + g.DZ - n.Z) AS D2) AS d
        WHERE  g.Pass = 2
        GROUP BY n.QId, g.GridId, c.CX, c.CY, c.CZ, g.DX, g.DY, g.DZ, g.L1
    ), r AS (
        SELECT s.QId, s.FX, s.FY, s.FZ,
               ROW_NUMBER() OVER (PARTITION BY s.QId
                                  ORDER BY (s.FitW + @SideEffectPenalty * s.RiskW) / w.SumW, s.L1, s.FX, s.FY, s.FZ) AS rn
        FROM s JOIN #w AS w ON w.QId = s.QId
    )
    SELECT QId, FX, FY, FZ INTO #b FROM r WHERE rn = 1;

    /* score against the hidden truth (simulation only) */
    SELECT  q.QId AS ImplantId, q.PatientId, q.Hemisphere,
            SQRT(SQUARE(b.FX - t.TX) + SQUARE(b.FY - t.SweetY) + SQUARE(b.FZ - t.SweetZ)) AS SuggestErrMm,
            SQRT(SQUARE(@ax  - t.TX) + SQUARE(@ay  - t.SweetY) + SQUARE(@az  - t.SweetZ)) AS AtlasErrMm,
            SQRT(SQUARE(q.X  - t.TX) + SQUARE(q.Y  - t.SweetY) + SQUARE(q.Z  - t.SweetZ)) AS ActualContactErrMm,
            @Floor + (@Emax - @Floor) * EXP(-(SQUARE(b.FX - t.TX) + SQUARE(b.FY - t.SweetY) + SQUARE(b.FZ - t.SweetZ)) / @twoS2) AS SuggestEfficacy,
            @Floor + (@Emax - @Floor) * EXP(-(SQUARE(@ax  - t.TX) + SQUARE(@ay  - t.SweetY) + SQUARE(@az  - t.SweetZ)) / @twoS2) AS AtlasEfficacy,
            10 * EXP(-(SQUARE(b.FX - t.TX - @gx) + SQUARE(b.FY - t.SweetY - @gy) + SQUARE(b.FZ - t.SweetZ - @gz)) / (2 * 1.8 * 1.8)) AS SuggestSideEffect,
            10 * EXP(-(SQUARE(@ax  - t.TX - @gx) + SQUARE(@ay  - t.SweetY - @gy) + SQUARE(@az  - t.SweetZ - @gz)) / (2 * 1.8 * 1.8)) AS AtlasSideEffect,
            w.Neff AS EffectiveN
    INTO    #res
    FROM    #q AS q
    JOIN    #b AS b ON b.QId = q.QId
    JOIN    #w AS w ON w.QId = q.QId
    CROSS APPLY (SELECT ABS(st.SweetX) AS TX, st.SweetY, st.SweetZ
                 FROM nt.SimTruth AS st
                 WHERE st.PatientId = q.PatientId AND st.TargetId = @TargetId AND st.Hemisphere = q.Hemisphere) AS t;

    SELECT  @TargetId AS TargetId, @K AS K, @SideEffectPenalty AS SideEffectPenalty,
            COUNT(*) AS Implants,
            AVG(SuggestErrMm) AS MeanSuggestErrMm, AVG(AtlasErrMm) AS MeanAtlasErrMm,
            AVG(ActualContactErrMm) AS MeanActualContactErrMm,
            AVG(CASE WHEN SuggestErrMm < AtlasErrMm THEN 1.0 ELSE 0.0 END) AS FractionCloserThanAtlas,
            AVG(SuggestEfficacy) AS MeanSuggestEfficacy, AVG(AtlasEfficacy) AS MeanAtlasEfficacy,
            AVG(SuggestSideEffect) AS MeanSuggestSideEffect, AVG(AtlasSideEffect) AS MeanAtlasSideEffect
    FROM    #res;

    SELECT * FROM #res ORDER BY ImplantId;
END;
GO

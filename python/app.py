"""
NeuroTarget - outcome-driven electrode placement explorer.

    streamlit run app.py

Uses SQL Server 2025 when NEUROTARGET_CONN is set (and reachable), otherwise the local
Python mirror over ../data. SIMULATED DATA - research/teaching prototype, not a medical device.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import ndimage
from skimage import measure

from neurosim import anatomy as A
from neurosim import cohort as C
from neurosim.backend import LocalBackend, SqlBackend

ROOT = Path(__file__).resolve().parent.parent

# Palette (validated reference instance): categorical slots + one sequential ramp
SUGGEST = "#eb6834"     # slot 2 orange - the answer
ATLAS = "#1baf7a"       # slot 3 aqua   - textbook atlas target
LEAD = "#4a3aa7"        # slot 7 violet - planned lead
SEQ = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]
S1, S2 = "#2a78d6", "#eb6834"

st.set_page_config(page_title="NeuroTarget", layout="wide")


# ------------------------------------------------------------------------------ backend
@st.cache_resource
def get_backend(kind: str, conn: str | None):
    if kind == "sql":
        return SqlBackend(conn)
    return LocalBackend(ROOT / "data")


conn = os.environ.get("NEUROTARGET_CONN")
with st.sidebar:
    st.header("NeuroTarget")
    options = ["Local (Python mirror)"] + (["SQL Server 2025"] if conn else [])
    choice = st.radio("Engine", options, index=len(options) - 1,
                      help="Set NEUROTARGET_CONN to an ODBC connection string to use SQL Server.")
    try:
        be = get_backend("sql" if choice.startswith("SQL") else "local", conn)
    except Exception as ex:  # fall back gracefully
        st.error(f"SQL Server unavailable: {ex}")
        be = get_backend("local", None)

    conds = be.conditions()
    cond_name = st.radio("Condition", conds.Name.tolist())
    cond = conds[conds.Name == cond_name].iloc[0]
    tg = be.targets(int(cond.ConditionId))
    tname = st.selectbox("Target", tg.Name.tolist())
    tgt = tg[tg.Name == tname].iloc[0]
    refs = be.referrals(int(cond.ConditionId))
    if refs.empty:
        st.warning("No referrals waiting for this condition.")
        st.stop()
    labels = [f"#{r.PatientId}  {r.Sex}, {r.AgeAtScan}y" + (f", pain {r.PainSide}" if isinstance(r.PainSide, str) else "")
              for r in refs.itertuples()]
    pick = st.selectbox("New referral", range(len(refs)), format_func=lambda i: labels[i])
    ref = refs.iloc[pick]
    if cond.Code == "CH":
        hemis = [ref.PainSide if isinstance(ref.PainSide, str) else "R"]
        st.caption(f"Cluster headache: ipsilateral to pain side ({hemis[0]}).")
    else:
        hemis = st.multiselect("Hemispheres", ["L", "R"], default=["L", "R"]) or ["R"]
    k = st.slider("Neighbours (K)", 10, 100, 40, 5,
                  help="How many anatomically similar previous implants inform the suggestion.")
    pen = st.slider("Side-effect caution", 0, 300, 100, 25,
                    help="0 = chase efficacy only. Higher = steer away from where similar patients had side effects.")
    st.caption(f"Engine: {be.name}")

st.warning("Simulated patients and outcomes. A research prototype for exploring the idea - not for clinical use.",
           icon="⚠️")

scan_id = int(ref.ScanId)
s = be.scan(scan_id)
mcp = np.array([(s.ACX + s.PCX) / 2, (s.ACY + s.PCY) / 2, (s.ACZ + s.PCZ) / 2])
scale = np.array([s.ScaleX, s.ScaleY, s.ScaleZ])


def to_native(tpl):
    return np.asarray(tpl) / scale + mcp


results = {h: be.suggest(scan_id, int(tgt.TargetId), h, k, float(pen)) for h in hemis}


@st.cache_data(show_spinner=False)
def load_volume(_be, sid):
    return _be.volume(sid)


vol = load_volume(be, scan_id)

# ------------------------------------------------------------------------------ header metrics
st.subheader(f"{tgt.Name} — patient #{ref.PatientId}")
for h in hemis:
    r, nb = results[h]
    side = "Left" if h == "L" else "Right"
    c = st.columns(4)
    c[0].metric(f"{side}: shift from atlas", f"{r['AtlasShiftMm']:.1f} mm")
    c[1].metric(f"{side}: expected benefit", f"{r['PredictedEfficacy']:.0f}%",
                help=cond.OutcomeMeasure + " - seen in similar patients stimulated near this point")
    c[2].metric(f"{side}: side-effect score", f"{r['PredictedSideEffect']:.1f} / 10")
    c[3].metric(f"{side}: effective evidence", f"{r['EffectiveN']:.0f} implants",
                help=f"Kernel-weighted count from K={k} of {be.library_size(int(tgt.TargetId))} in the library")
if len(hemis) > 1:
    st.caption("Left and right share one whole-brain fingerprint, so they mirror each other. "
               "Hemisphere-specific features would let them differ.")

tab_plan, tab_3d, tab_nb, tab_feat, tab_val, tab_loop = st.tabs(
    ["Plan (MRI)", "3D", "Similar patients", "Scan fingerprint", "Validation", "Close the loop"])


def trajectory_native(tpl_target, h, length=45):
    d = C.trajectory(h)
    pts = np.array([tpl_target + t * d for t in np.linspace(-C.CONTACT_PITCH, length, 30)])
    return np.array([to_native(p) for p in pts])


# ------------------------------------------------------------------------------ plan tab
with tab_plan:
    if vol is None:
        st.info("No image stored for this scan (only its features). Showing coordinates only.")
    else:
        h0 = st.radio("Centre slices on", hemis, horizontal=True, format_func=lambda h: f"{h} suggestion") if len(hemis) > 1 else hemis[0]
        r0 = results[h0][0]
        p = np.array([r0["NativeX"], r0["NativeY"], r0["NativeZ"]])
        ijk = np.clip(np.round((p - A.FOV_MIN) / A.VOXEL_MM - 0.5).astype(int), 0, np.array(A.DIMS) - 1)
        ax = A.voxel_centres()
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Axial", "Coronal", "Sagittal"),
                            horizontal_spacing=0.05)
        views = [(vol[:, :, ijk[2]].T, ax[0], ax[1], 0, 1), (vol[:, ijk[1], :].T, ax[0], ax[2], 0, 2),
                 (vol[ijk[0], :, :].T, ax[1], ax[2], 1, 2)]
        for c, (img, xa, ya, a0, a1) in enumerate(views, start=1):
            fig.add_trace(go.Heatmap(z=img, x=xa, y=ya, colorscale="gray", showscale=False, zmin=0, zmax=1,
                                     hoverinfo="skip"), 1, c)
            for h in hemis:
                r, _ = results[h]
                tpl = np.array([r["TemplateX"], r["TemplateY"], r["TemplateZ"]])
                tr = trajectory_native(tpl, h)
                fig.add_trace(go.Scatter(x=tr[:, a0], y=tr[:, a1], mode="lines", line=dict(color=LEAD, width=2),
                                         name="Planned lead", showlegend=(c == 1 and h == hemis[0]),
                                         hoverinfo="skip"), 1, c)
                at = to_native([r["AtlasX"], r["AtlasY"], r["AtlasZ"]])
                fig.add_trace(go.Scatter(x=[at[a0]], y=[at[a1]], mode="markers", name="Atlas target",
                                         marker=dict(color=ATLAS, size=10, symbol="x", line=dict(width=1, color="white")),
                                         showlegend=(c == 1 and h == hemis[0]),
                                         hovertemplate="Atlas<br>%{x:.1f}, %{y:.1f} mm<extra></extra>"), 1, c)
                sp = np.array([r["NativeX"], r["NativeY"], r["NativeZ"]])
                fig.add_trace(go.Scatter(x=[sp[a0]], y=[sp[a1]], mode="markers", name="Suggested",
                                         marker=dict(color=SUGGEST, size=11, line=dict(width=2, color="white")),
                                         showlegend=(c == 1 and h == hemis[0]),
                                         hovertemplate="Suggested<br>%{x:.1f}, %{y:.1f} mm<extra></extra>"), 1, c)
            fig.update_xaxes(range=[p[a0] - 45, p[a0] + 45], row=1, col=c, showgrid=False, zeroline=False,
                             title_text=["x mm (+R)", "x mm (+R)", "y mm (+A)"][c - 1])
            fig.update_yaxes(range=[p[a1] - 45, p[a1] + 45], row=1, col=c, showgrid=False, zeroline=False,
                             scaleanchor=f"x{'' if c == 1 else c}", scaleratio=1,
                             title_text=["y mm (+A)", "z mm (+S)", "z mm (+S)"][c - 1])
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=60, b=10),
                          legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig, width="stretch")

    rows = []
    for h in hemis:
        r, _ = results[h]
        rows.append(dict(Hemisphere=h, Space="Suggested - AC-PC (mm from MCP)",
                         X=r["TemplateX"], Y=r["TemplateY"], Z=r["TemplateZ"]))
        rows.append(dict(Hemisphere=h, Space="Suggested - this scan (native mm)",
                         X=r["NativeX"], Y=r["NativeY"], Z=r["NativeZ"]))
        rows.append(dict(Hemisphere=h, Space="Atlas - AC-PC (mm from MCP)", X=r["AtlasX"], Y=r["AtlasY"], Z=r["AtlasZ"]))
        rows.append(dict(Hemisphere=h, Space="Spread of good placements (SD)", X=r["SdX"], Y=r["SdY"], Z=r["SdZ"]))
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    st.caption(f"Side-effect structure to respect: {tgt.NoGoName}.")


# ------------------------------------------------------------------------------ 3D tab
@st.cache_data(show_spinner="Building surfaces ...")
def meshes(sid, _vol):
    v = ndimage.gaussian_filter(_vol, 1.0)
    brain = ndimage.binary_fill_holes(v > 0.33)
    lab, _ = ndimage.label(brain)
    c = tuple(d // 2 for d in A.DIMS)
    brain = lab == lab[c] if lab[c] else brain
    out = {}
    verts, faces, _, _ = measure.marching_cubes(brain.astype(np.float32), 0.5, step_size=2)
    out["brain"] = (A.FOV_MIN + (verts + 0.5) * A.VOXEL_MM, faces)
    core = ndimage.binary_erosion(brain, iterations=3)
    vent = (v < 0.3) & core
    if vent.any():
        verts, faces, _, _ = measure.marching_cubes(vent.astype(np.float32), 0.5, step_size=1)
        out["ventricles"] = (A.FOV_MIN + (verts + 0.5) * A.VOXEL_MM, faces)
    return out


with tab_3d:
    zoom = st.toggle("Zoom to target region", value=True)
    fig = go.Figure()
    if vol is not None:
        m = meshes(scan_id, vol)
        (vb, fb) = m["brain"]
        fig.add_trace(go.Mesh3d(x=vb[:, 0], y=vb[:, 1], z=vb[:, 2], i=fb[:, 0], j=fb[:, 1], k=fb[:, 2],
                                color="#b9b8b2", opacity=0.08, name="Brain", hoverinfo="skip", showlegend=True,
                                visible=("legendonly" if zoom else True)))
        if "ventricles" in m:
            (vv, fv) = m["ventricles"]
            fig.add_trace(go.Mesh3d(x=vv[:, 0], y=vv[:, 1], z=vv[:, 2], i=fv[:, 0], j=fv[:, 1], k=fv[:, 2],
                                    color="#86b6ef", opacity=0.35, name="Ventricles", hoverinfo="skip",
                                    showlegend=True))
    for n_h, h in enumerate(hemis):
        r, nb = results[h]
        sgn = -1 if h == "L" else 1
        P = np.array([to_native([sgn * x, y, z]) for x, y, z in nb[["X", "Y", "Z"]].values])
        size = 5 + 10 * nb.KernelW.values / nb.KernelW.max()
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name="Similar patients' active contacts",
            showlegend=n_h == 0,
            marker=dict(size=size, color=nb.EfficacyPct, colorscale=SEQ, cmin=0, cmax=100,
                        line=dict(width=1, color="white"),
                        colorbar=dict(title="Benefit %", len=0.5, thickness=12) if n_h == 0 else None),
            customdata=np.c_[nb.PatientId, nb.EfficacyPct, nb.SideEffectScore, nb.Dist],
            hovertemplate="Patient %{customdata[0]}<br>Benefit %{customdata[1]:.0f}%<br>"
                          "Side effects %{customdata[2]:.1f}<br>Anatomy distance %{customdata[3]:.2f}<extra></extra>"))
        tpl = np.array([r["TemplateX"], r["TemplateY"], r["TemplateZ"]])
        tr = trajectory_native(tpl, h)
        fig.add_trace(go.Scatter3d(x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode="lines", line=dict(color=LEAD, width=6),
                                   name="Planned lead", showlegend=n_h == 0, hoverinfo="skip"))
        at = to_native([r["AtlasX"], r["AtlasY"], r["AtlasZ"]])
        fig.add_trace(go.Scatter3d(x=[at[0]], y=[at[1]], z=[at[2]], mode="markers", name="Atlas target",
                                   showlegend=n_h == 0, marker=dict(size=6, color=ATLAS, symbol="x")))
        sp = np.array([r["NativeX"], r["NativeY"], r["NativeZ"]])
        fig.add_trace(go.Scatter3d(x=[sp[0]], y=[sp[1]], z=[sp[2]], mode="markers", name="Suggested",
                                   showlegend=n_h == 0,
                                   marker=dict(size=9, color=SUGGEST, symbol="diamond", line=dict(width=2, color="white"))))
        # uncertainty ellipsoid (spread of good placements)
        if np.isfinite([r["SdX"], r["SdY"], r["SdZ"]]).all():
            u, v_ = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
            ex = sp[0] + r["SdX"] / scale[0] * np.cos(u) * np.sin(v_)
            ey = sp[1] + r["SdY"] / scale[1] * np.sin(u) * np.sin(v_)
            ez = sp[2] + r["SdZ"] / scale[2] * np.cos(v_)
            fig.add_trace(go.Surface(x=ex, y=ey, z=ez, opacity=0.25, showscale=False, hoverinfo="skip",
                                     colorscale=[[0, SUGGEST], [1, SUGGEST]], name="Spread (1 SD)"))
    scene = dict(aspectmode="data", xaxis_title="x (+R)", yaxis_title="y (+A)", zaxis_title="z (+S)",
                 camera=dict(eye=dict(x=1.5, y=-1.3, z=0.8)))
    if zoom:
        cx = np.mean([results[h][0]["NativeX"] for h in hemis])
        cy = np.mean([results[h][0]["NativeY"] for h in hemis])
        cz = np.mean([results[h][0]["NativeZ"] for h in hemis])
        xs = [results[h][0]["NativeX"] for h in hemis]
        half = (max(xs) - min(xs)) / 2 + 12
        scene.update(xaxis=dict(range=[cx - half, cx + half], title="x (+R)"),
                     yaxis=dict(range=[cy - 12, cy + 12], title="y (+A)"),
                     zaxis=dict(range=[cz - 10, cz + 20], title="z (+S)"), aspectmode="data")
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=0),
                      scene=scene)
    st.plotly_chart(fig, width="stretch")
    st.caption("Each dot is where a previous, anatomically similar patient was actually stimulated, mapped into "
               "this patient's brain. Darker = better outcome, larger = more similar anatomy.")


# ------------------------------------------------------------------------------ neighbours tab
with tab_nb:
    h = st.radio("Hemisphere", hemis, horizontal=True, key="nbh") if len(hemis) > 1 else hemis[0]
    r, nb = results[h]
    c1, c2 = st.columns([3, 2])
    with c1:
        d = np.sqrt((nb.X - r["X"]) ** 2 + (nb.Y - r["Y"]) ** 2 + (nb.Z - r["Z"]) ** 2)
        fig = go.Figure(go.Scatter(
            x=d, y=nb.EfficacyPct, mode="markers",
            marker=dict(size=6 + 12 * nb.KernelW / nb.KernelW.max(), color=S1, opacity=0.8,
                        line=dict(width=1, color="white")),
            customdata=np.c_[nb.PatientId, nb.SideEffectScore, nb.Dist],
            hovertemplate="Patient %{customdata[0]}<br>%{x:.1f} mm from suggestion<br>Benefit %{y:.0f}%<br>"
                          "Side effects %{customdata[1]:.1f}<br>Anatomy distance %{customdata[2]:.2f}<extra></extra>"))
        fig.update_layout(title="Benefit falls away with distance from the suggested point",
                          xaxis_title="Active contact distance from suggestion (mm)",
                          yaxis_title="Benefit (%)", height=380, margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
    with c2:
        show = nb[["Rn", "PatientId", "Hemisphere", "Dist", "KernelW", "EfficacyPct", "SideEffectScore",
                   "PatientRating"]].rename(columns={"Rn": "#", "Dist": "Anatomy dist", "KernelW": "Weight",
                                                     "EfficacyPct": "Benefit %", "SideEffectScore": "Side eff.",
                                                     "PatientRating": "Rating"})
        st.dataframe(show.round(2), hide_index=True, width="stretch", height=380)


# ------------------------------------------------------------------------------ features tab
with tab_feat:
    ft = be.feature_table(scan_id)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(ft[["Description", "Value", "Unit", "Z"]].round(2), hide_index=True, width="stretch",
                     height=520)
    with c2:
        fig = go.Figure(go.Bar(y=ft.Description, x=ft.Z, orientation="h", marker_color=S1,
                               hovertemplate="%{y}<br>z = %{x:.2f}<extra></extra>"))
        fig.update_layout(title="How this brain differs from the library (z-scores)", height=520,
                          margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(autorange="reversed"))
        fig.update_xaxes(zeroline=True, zerolinecolor="#8a8984")
        st.plotly_chart(fig, width="stretch")
    st.caption("These 14 numbers are measured from the MRI at ingest and stored as a VECTOR(14). "
               "Similarity search runs on this fingerprint, not on the raw image.")


# ------------------------------------------------------------------------------ validation tab
@st.cache_data(show_spinner="Running leave-one-out over the whole library ...")
def loo(_be, key, target_id, k, pen):
    return _be.evaluate(target_id, k, pen)


with tab_val:
    st.write("Every historical implant is hidden in turn and re-planned from everyone else. "
             "Because the data is simulated, each patient's true best point is known, so we can score "
             "the suggestion against the plain atlas target.")
    if st.button("Run validation for this target"):
        st.session_state["val"] = (int(tgt.TargetId), k, pen)
    if st.session_state.get("val") == (int(tgt.TargetId), k, pen):
        ev = loo(be, be.name, int(tgt.TargetId), k, float(pen))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean error: suggestion", f"{ev.SuggestErrMm.mean():.2f} mm")
        m2.metric("Mean error: atlas", f"{ev.AtlasErrMm.mean():.2f} mm")
        m3.metric("Closer than atlas", f"{100 * (ev.SuggestErrMm < ev.AtlasErrMm).mean():.0f}%")
        m4.metric("Simulated benefit", f"{ev.SuggestEfficacy.mean():.0f}% vs {ev.AtlasEfficacy.mean():.0f}%")
        bins = dict(start=0, end=float(max(ev.AtlasErrMm.max(), ev.SuggestErrMm.max())) + 0.25, size=0.25)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ev.AtlasErrMm, name="Atlas target", xbins=bins, marker_color=ATLAS))
        fig.add_trace(go.Histogram(x=ev.SuggestErrMm, name="Suggested", xbins=bins, marker_color=SUGGEST))
        fig.update_layout(barmode="group", bargap=0.15, title="Distance from each patient's true sweet spot",
                          xaxis_title="Error (mm)", yaxis_title="Implants", height=400,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.08, x=0.6))
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Per-implant results"):
            st.dataframe(ev.round(2), hide_index=True, width="stretch")


# ------------------------------------------------------------------------------ close the loop tab
with tab_loop:
    st.write("Simulate the operation: a lead is placed at the suggestion (with realistic surgical error), "
             "the best contact is programmed, and the 12-month outcome is generated from this patient's hidden "
             "true sweet spot. Recording it adds the patient to the evidence for everyone after them.")
    if st.button("Simulate surgery and 12-month outcome"):
        rng = np.random.default_rng()
        rows = []
        for h in hemis:
            r, _ = be.suggest(scan_id, int(tgt.TargetId), h, k, float(pen), save=True)  # logged suggestion
            planned = np.array([r["TemplateX"], r["TemplateY"], r["TemplateZ"]])
            err = rng.normal(0, 0.9, 3)
            seed = int(rng.integers(1 << 31))
            truth = be.truth(int(ref.PatientId), int(tgt.TargetId), h)
            contacts = C.lead_contacts(planned, h, err)
            kk, eff, se, rating = C.simulate_outcome(contacts, truth, C.TGT[tgt.Code], h, np.random.default_rng(seed))
            # counterfactual: identical surgery and follow-up, aimed at the atlas instead
            atlas_c = C.lead_contacts(np.array([r["AtlasX"], r["AtlasY"], r["AtlasZ"]]), h, err)
            _, eff_a, _, _ = C.simulate_outcome(atlas_c, truth, C.TGT[tgt.Code], h, np.random.default_rng(seed))
            iid = be.record_outcome(int(ref.PatientId), int(tgt.TargetId), h, contacts, kk, round(eff, 1),
                                    round(se, 1), rating, r.get("SuggestionId"))
            rows.append(dict(Hemisphere=h, ImplantId=iid, ActiveContact=kk, BenefitPct=round(eff, 1),
                             SideEffects=round(se, 1), Rating=rating,
                             ErrorMm=round(float(np.linalg.norm(contacts[kk] - truth)), 2),
                             SameOpAtAtlasBenefitPct=round(eff_a, 1)))
        st.success("Recorded. This patient is now part of the historical library.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.cache_data.clear()

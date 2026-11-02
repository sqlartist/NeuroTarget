"""
Anatomy model, phantom renderer and feature extractor.

Coordinate conventions
----------------------
* All coordinates are millimetres, RAS: +x = patient right, +y = anterior, +z = superior.
* "Template" space is AC-PC space: origin at the mid-commissural point (MCP),
  AC-PC line along +y, proportionally scaled to a reference brain
  (width 136 mm, length 170 mm, height 124 mm, AC-PC 24 mm).
  This is the classic stereotactic frame surgeons actually plan in.
* "Native" space is the patient's own scanner space (mm). The mapping is
      template = S * (native - MCP_native)      S = diag(ScaleX, ScaleY, ScaleZ)
  i.e. Talairach-style proportional scaling. No rotation: we assume the
  sequence was acquired / resliced AC-PC aligned, which is standard pre-op.

Everything here is SIMULATED. It is a teaching / research toy, not a medical device.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

VOXEL_MM = 2.0
# field of view in native mm (x, y, z)
FOV_MIN = np.array([-84.0, -104.0, -76.0])
DIMS = (84, 104, 80)  # voxels

TEMPLATE_BRAIN = np.array([136.0, 170.0, 124.0])  # W, L, H  (mm)
TEMPLATE_ACPC = 24.0

# Atlas targets, RIGHT hemisphere, template (AC-PC, MCP-origin) mm.
# Values follow the classic indirect-targeting rules of thumb.
ATLAS = {
    "CH-PH":  np.array([2.0, -3.0, -5.0]),   # posterior hypothalamus (Leone): 2 lat, 3 post, 5 inf
    "PD-STN": np.array([12.0, -2.0, -4.0]),  # subthalamic nucleus: 12 lat, 2 post, 4 inf
    "PD-GPi": np.array([20.0, 2.0, -4.0]),   # globus pallidus internus: 20 lat, 2 ant, 4 inf
}
# How strongly each target's true sweet spot is pushed around by individual anatomy
ANAT_GAIN = {"CH-PH": 1.2, "PD-STN": 1.4, "PD-GPi": 1.2}

# Intensities, T1-weighted look
I_SCALP, I_SKULL, I_CSF, I_GM, I_WM = 0.70, 0.10, 0.15, 0.50, 0.80
I_THAL, I_GPI, I_STN, I_HYPO, I_VENT, I_COMM = 0.62, 0.58, 0.40, 0.55, 0.12, 1.00

FEATURES = [
    ("brain_w_mm", "Brain width (L-R extent)", "mm"),
    ("brain_l_mm", "Brain length (A-P extent)", "mm"),
    ("brain_h_mm", "Brain height (S-I extent)", "mm"),
    ("brain_vol_ml", "Intracranial brain volume", "ml"),
    ("acpc_mm", "AC-PC distance", "mm"),
    ("v3_width_mm", "Third ventricle width at MCP", "mm"),
    ("lv_vol_ml", "Lateral ventricle volume", "ml"),
    ("lv_span_mm", "Frontal horn span", "mm"),
    ("evans_index", "Evans index (horn span / brain width)", "ratio"),
    ("v3_ratio", "3rd ventricle width / brain width x100", "%"),
    ("lv_frac", "Ventricle volume / brain volume x100", "%"),
    ("acpc_ratio", "AC-PC / brain length x100", "%"),
    ("v3_height_mm", "Third ventricle height at MCP", "mm"),
    ("age", "Age at scan", "years"),
]
N_FEATURES = len(FEATURES)


# --------------------------------------------------------------------------------------
# Latent anatomy
# --------------------------------------------------------------------------------------
def sample_anatomy(rng: np.random.Generator, condition: str) -> dict:
    if condition == "CH":
        age = float(np.clip(rng.normal(42, 10), 20, 70))
        sex = "M" if rng.random() < 0.75 else "F"
    else:
        age = float(np.clip(rng.normal(62, 8), 40, 80))
        sex = "M" if rng.random() < 0.6 else "F"
    male = 1.0 if sex == "M" else 0.0
    k = 1.0 + 0.03 * (male - 0.5) + rng.normal(0, 0.045, 3)
    lv = float(np.clip(1 + 0.012 * (age - 50) + rng.normal(0, 0.15), 0.7, 1.8))
    w3 = float(np.clip(3.5 + 0.08 * (age - 40) + 1.5 * (lv - 1) + rng.normal(0, 0.8), 2.0, 11.0))
    acpc = float(TEMPLATE_ACPC * k[1] * (1 + rng.normal(0, 0.04)))
    mcp = rng.normal(0, 4, 3)  # head position in scanner
    return dict(age=round(age), sex=sex, kx=k[0], ky=k[1], kz=k[2], lv=lv, w3=w3, acpc=acpc,
                mcp=mcp.tolist())


def true_scale(a: dict) -> np.ndarray:
    brain = TEMPLATE_BRAIN * np.array([a["kx"], a["ky"], a["kz"]])
    return np.array([TEMPLATE_BRAIN[0] / brain[0], TEMPLATE_ACPC / a["acpc"], TEMPLATE_BRAIN[2] / brain[2]])


def sweet_spot(a: dict, target: str, hemi: str, rng: np.random.Generator) -> np.ndarray:
    """True optimal stimulation point, template space (x signed by hemisphere).
    Deviates from the atlas as a function of anatomy that proportional scaling does not remove."""
    zv3 = (a["w3"] - 5.5) / 1.8
    zlv = (a["lv"] - 1.2) / 0.25
    zac = (a["acpc"] / a["ky"] - TEMPLATE_ACPC) / 1.0
    delta = np.array([1.4 * zv3, -0.7 * zlv + 0.5 * zac, 0.9 * zlv - 0.4 * zv3]) * ANAT_GAIN[target]
    p = ATLAS[target] + delta + rng.normal(0, 0.4, 3) + rng.normal(0, 0.25, 3)
    if hemi == "L":
        p = p * np.array([-1, 1, 1])
    return p


def template_to_native(p, scale, mcp):
    return np.asarray(p) / np.asarray(scale) + np.asarray(mcp)


def native_to_template(p, scale, mcp):
    return (np.asarray(p) - np.asarray(mcp)) * np.asarray(scale)


# --------------------------------------------------------------------------------------
# Phantom renderer
# --------------------------------------------------------------------------------------
def _grid():
    ax = [FOV_MIN[i] + VOXEL_MM * (np.arange(DIMS[i]) + 0.5) for i in range(3)]
    return ax


_AX = _grid()


def voxel_centres():
    return _AX


def _paint(vol, centre, axes, intensity, edge=VOXEL_MM):
    """Soft-edged ellipsoid composited into vol (partial-volume aware), bounding-box only."""
    centre = np.asarray(centre, float)
    axes = np.asarray(axes, float)
    lo = [max(0, int((centre[i] - axes[i] - 2 * edge - FOV_MIN[i]) / VOXEL_MM)) for i in range(3)]
    hi = [min(DIMS[i], int((centre[i] + axes[i] + 2 * edge - FOV_MIN[i]) / VOXEL_MM) + 2) for i in range(3)]
    if any(h <= l for l, h in zip(lo, hi)):
        return
    x = _AX[0][lo[0]:hi[0], None, None] - centre[0]
    y = _AX[1][None, lo[1]:hi[1], None] - centre[1]
    z = _AX[2][None, None, lo[2]:hi[2]] - centre[2]
    r = np.sqrt((x / axes[0]) ** 2 + (y / axes[1]) ** 2 + (z / axes[2]) ** 2)
    m = np.clip(0.5 + (1 - r) * axes.mean() / edge, 0, 1)
    sub = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub *= (1 - m)
    sub += intensity * m


def render_phantom(a: dict, spots: dict, rng: np.random.Generator) -> np.ndarray:
    """Synthetic 3D T1 MRI (float32, DIMS) for one patient.
    spots: {target_code: right-hemisphere template sweet spot} used to place the deep nuclei."""
    vol = np.zeros(DIMS, np.float32)
    mcp = np.array(a["mcp"])
    scale = true_scale(a)
    half = TEMPLATE_BRAIN / 2 * np.array([a["kx"], a["ky"], a["kz"]])
    c = mcp + np.array([0, -5, 12])  # brain centre sits above/behind the MCP
    _paint(vol, c, half + 11, I_SCALP)
    _paint(vol, c, half + 7, I_SKULL)
    _paint(vol, c, half + 2, I_CSF)
    _paint(vol, c, half, I_GM)
    _paint(vol, c, half - 6, I_WM)

    def nat(t):
        return template_to_native(t, scale, mcp)

    for s in (1, -1):
        m = np.array([s, 1, 1])
        _paint(vol, nat(np.array([a["w3"] / 2 * scale[0] + 8, -10, 3]) * m), [8, 13, 8], I_THAL)
        _paint(vol, nat((spots["PD-GPi"] + [-1.0, 0, 0.5]) * m), [3, 6, 4], I_GPI)
        _paint(vol, nat((spots["PD-STN"] + [-1.0, -0.5, -1.0]) * m), [3, 5, 2.5], I_STN)
        _paint(vol, nat((spots["CH-PH"] + [1.0, 0, -1.0]) * m), [2.5, 4, 3], I_HYPO)
        # lateral ventricles (bodies + frontal horns)
        _paint(vol, mcp + [s * (3.5 + 4.5 * a["lv"]), 4, 20], [4.2 * a["lv"], 26, 5.5 * a["lv"]], I_VENT)
    # third ventricle: thin midline slit between AC and PC
    _paint(vol, mcp + [0, -1, 3], [a["w3"] / 2, a["acpc"] / 2 - 3.5, 7.0], I_VENT, edge=1.0)
    # commissures (bright white-matter bundles)
    for yy in (a["acpc"] / 2, -a["acpc"] / 2):
        _paint(vol, mcp + [0, yy, 0], [3.0, 3.0, 3.0], I_COMM, edge=0.5)

    vol = ndimage.gaussian_filter(vol, 0.6)
    vol += rng.normal(0, 0.018, vol.shape).astype(np.float32)
    return np.clip(vol, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Feature extraction: works ONLY from the image (plus age), like a real ingest pipeline
# --------------------------------------------------------------------------------------
def landmarks(a: dict, rng: np.random.Generator):
    """Simulated AC/PC landmark clicks (native mm), ~0.5 mm operator error."""
    mcp = np.array(a["mcp"])
    ac = mcp + [0, a["acpc"] / 2, 0] + rng.normal(0, 0.5, 3)
    pc = mcp - [0, a["acpc"] / 2, 0] + rng.normal(0, 0.5, 3)
    return ac, pc


def extract(vol: np.ndarray, age: float, ac, pc) -> tuple[dict, dict]:
    v = VOXEL_MM
    ctr = tuple(d // 2 for d in DIMS)
    fg = vol > 0.3
    lab, _ = ndimage.label(fg)
    # brain = largest component not touching the FOV border (scalp does touch/encircle)
    sizes = ndimage.sum(fg, lab, range(1, lab.max() + 1))
    order = np.argsort(sizes)[::-1] + 1
    brain = None
    for l in order[:4]:
        comp = lab == l
        sl = ndimage.find_objects(comp.astype(int))[0]
        w = [s.stop - s.start for s in sl]
        if w[0] < DIMS[0] - 4:  # scalp shell nearly spans the FOV
            if brain is None or comp.sum() > brain.sum():
                inside = ndimage.binary_fill_holes(comp)
                if inside[ctr]:
                    brain = inside
    if brain is None:
        brain = ndimage.binary_fill_holes(lab == order[0])
    idx = np.array(np.nonzero(brain))
    ext = (idx.max(1) - idx.min(1) + 1) * v
    brain_vol = brain.sum() * v ** 3 / 1000

    # AC / PC come from the landmarking step (clinician-verified clicks in the
    # planning software, as in real stereotactic workflows), passed in by the caller.
    ac, pc = np.asarray(ac, float), np.asarray(pc, float)
    mcp = (ac + pc) / 2
    acpc = float(np.linalg.norm(ac - pc))
    mi = np.round((mcp - FOV_MIN) / v - 0.5).astype(int)

    # third ventricle width: integrate "CSF-ness" across midline at MCP level
    band = vol[mi[0] - 6:mi[0] + 7, mi[1] - 1:mi[1] + 2, mi[2] - 1:mi[2] + 2]
    csf3 = np.clip((I_THAL - band) / (I_THAL - I_VENT), 0, 1)
    v3w = float(csf3.sum(0).mean() * v)
    colz = vol[mi[0], mi[1] - 1:mi[1] + 2, mi[2] - 10:mi[2] + 11]
    v3h = float(np.clip((I_WM - colz) / (I_WM - I_VENT), 0, 1).sum(1).mean() * v)

    # lateral ventricles: CSF inside an eroded brain mask, above the MCP
    core = ndimage.binary_erosion(brain, iterations=3)
    core[:, :, : mi[2] + 4] = False
    csf = np.clip((I_WM - vol) / (I_WM - I_VENT), 0, 1) * core
    lv_vol = float(csf.sum() * v ** 3 / 1000)
    xs = np.nonzero((csf > 0.5).any(axis=(1, 2)))[0]
    lv_span = float((xs.max() - xs.min() + 1) * v) if len(xs) else 0.0

    f = dict(
        brain_w_mm=float(ext[0]), brain_l_mm=float(ext[1]), brain_h_mm=float(ext[2]),
        brain_vol_ml=float(brain_vol), acpc_mm=acpc, v3_width_mm=v3w, lv_vol_ml=lv_vol,
        lv_span_mm=lv_span, evans_index=lv_span / ext[0], v3_ratio=100 * v3w / ext[0],
        lv_frac=100 * lv_vol / brain_vol, acpc_ratio=100 * acpc / ext[1], v3_height_mm=v3h,
        age=float(age),
    )
    geom = dict(ac=ac.tolist(), pc=pc.tolist(), mcp=mcp.tolist(),
                scale=[TEMPLATE_BRAIN[0] / ext[0], TEMPLATE_ACPC / acpc, TEMPLATE_BRAIN[2] / ext[2]])
    return f, geom

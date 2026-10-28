"""Keep the two people's clothing colours stable along the chain of generated beats.

Chained image-to-video drifts towards higher saturation and contrast (the bottle-green jacket turns teal, black
jeans turn blue). Correction: SAM3 segments each garment in every frame of a locked beat (sam3_garments.py);
each garment's median Lab colour is compared with its reference colour (A05, frames 20-40, reference.json) and
75 % of the difference is added back inside that garment's own mask (soft edge). Offsets are smoothed over time.
Skin, hair, shoes, the bag and the background are not touched.

  python colorfix.py measure A10     # SAM3 garments for all frames of chain/A10.mkv (queued on the worker GPU)
  python colorfix.py seed A10        # chain/A10_last_cc.png = corrected last frame (start image of the next beat)
In assemble.py: colorfix.correct_beat(frames, beat).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import chain as ch  # noqa: E402
from assemble import read_frames  # noqa: E402
from lockjob import SAM3_PY  # noqa: E402
from sg_paths import SCRIPTS  # noqa: E402

ROOT = ch.ROOT
CF = os.path.join(ROOT, "work", "colorfix")
REF_BEAT, REF_IDX = "A05", [20, 30, 40]
GARMENTS = ["green puffer jacket", "red coat", "jeans", "khaki trousers", "white knitted beanie", "short grey hair", "white trainers", "black handbag", "head of the man in the green jacket"]
STRENGTH = 0.75
CLIP = np.array([30.0, 18.0, 18.0])     # max |offset| per Lab channel (8-bit Lab units)
MIN_PX = 400


def run_garments(frames, tag, labels=False):
    os.makedirs(CF, exist_ok=True)
    npy = os.path.join(CF, f"{tag}_frames.npy")
    np.save(npy, frames)
    out = os.path.join(CF, f"{tag}_stats.json")
    lab = os.path.join(CF, f"{tag}_labels.npz")
    q = os.path.join(ROOT, "work", "queue")
    name = f"garm_{tag}_{int(time.time())}"
    cmd = (f"{SAM3_PY} {SCRIPTS}/sam3_garments.py --frames {npy} --idx {' '.join(map(str, range(len(frames))))} "
           f"--out {out}" + (f" --labels {lab}" if labels else ""))
    tmp = os.path.join(q, f".{name}.tmp")
    json.dump(dict(name=name, cmd=cmd, priority=-1), open(tmp, "w"))
    os.rename(tmp, os.path.join(q, name + ".json"))
    while True:
        if any(p.endswith(f"__{name}.json") for p in os.listdir(os.path.join(q, "done"))):
            break
        if any(p.endswith(f"__{name}.json") for p in os.listdir(os.path.join(q, "failed"))):
            raise RuntimeError("garment stats failed")
        time.sleep(3)
    os.remove(npy)
    return json.load(open(out))


# reference colours on the real GP02 view: her garments from A01 (first seconds, closest to the opening frame);
# his garments from the first beat where he is fully visible near the bench (A04 end / A05 start)
REF_SOURCES = [("A01", [2, 8, 14], ["red coat", "jeans", "white knitted beanie", "white trainers", "black handbag"]),
               ("A05", [2, 8, 14], ["green puffer jacket", "khaki trousers", "short grey hair",
                                    "head of the man in the green jacket"])]


def reference():
    p = os.path.join(CF, "reference.json")
    if os.path.exists(p):
        return json.load(open(p))
    ref = {}
    for beat, idx, garms in REF_SOURCES:
        clip = os.path.join(ch.CH, f"{beat}.mkv")
        if not os.path.exists(clip):
            continue
        st = run_garments(read_frames(clip)[idx], f"ref_{beat}")
        for g in garms:
            meds = [st[k][g]["med"] for k in st if g in st[k]]
            if meds:
                ref[g] = np.median(np.array(meds), 0).tolist()
    if all(any(g in ref for g in garms) for _, _, garms in REF_SOURCES):
        json.dump(ref, open(p, "w"), indent=1)    # cache only once both people are in
    return ref


def offsets_from_stats(st, n, ref):
    """(n, G, 3) Lab offsets per frame and garment: interpolate gaps, median-then-box smooth over time."""
    D = np.full((n, len(GARMENTS), 3), np.nan, np.float32)
    for f in range(n):
        for gi, g in enumerate(GARMENTS):
            v = st.get(str(f), {}).get(g)
            if v and v["n"] >= MIN_PX and g in ref:
                clip = np.array([30.0, 30.0, 30.0]) if g == "short grey hair" else CLIP
                D[f, gi] = np.clip(STRENGTH * (np.array(ref[g]) - np.array(v["med"])), -clip, clip)
    t = np.arange(n)
    for gi in range(len(GARMENTS)):
        for c in range(3):
            y = D[:, gi, c]
            ok = ~np.isnan(y)
            D[:, gi, c] = np.interp(t, t[ok], y[ok]) if ok.any() else 0.0
    M = D.copy()
    for f in range(n):
        M[f] = np.median(D[max(0, f - 4): f + 5], 0)
    S = M.copy()
    for f in range(n):
        S[f] = M[max(0, f - 3): f + 4].mean(0)
    return S


def measure(beat):
    fr = read_frames(os.path.join(ch.CH, f"{beat}.mkv"))
    st = run_garments(fr, beat, labels=True)
    D = offsets_from_stats(st, len(fr), reference())
    np.save(os.path.join(CF, f"{beat}_offsets.npy"), D)
    return D


HEAD = "head of the man in the green jacket"


def apply(frame, labels, D):
    """Add each garment's Lab offset inside its own (soft-edged) mask; inside the man's head, turn blue hair grey."""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    hi = GARMENTS.index(HEAD) + 1
    hm = cv2.GaussianBlur(cv2.dilate((labels == hi).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32),
                          (0, 0), 1.5)
    if hm.any():
        blue = np.clip((124.0 - lab[..., 2]) / 10.0, 0, 1) * hm      # only bluish pixels of the head
        lab[..., 1] += (128.0 - lab[..., 1]) * 0.9 * blue
        lab[..., 2] += (128.0 - lab[..., 2]) * 0.9 * blue
        lab[..., 0] += 10.0 * blue
    for gi in range(min(len(GARMENTS), len(D))):
        if GARMENTS[gi] == HEAD:
            continue
        m = (labels == gi + 1).astype(np.float32)
        if not m.any() or not np.any(D[gi]):
            continue
        m = cv2.GaussianBlur(m, (0, 0), 1.5)
        lab += m[..., None] * D[gi][None, None, :]
    return cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def correct_beat(frames, beat):
    po, pl = os.path.join(CF, f"{beat}_offsets.npy"), os.path.join(CF, f"{beat}_labels.npz")
    if not (os.path.exists(po) and os.path.exists(pl)):
        return frames
    D, L = np.load(po), np.load(pl)["labels"]
    return np.stack([apply(f, L[i], D[i]) for i, f in enumerate(frames)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "seed"])
    ap.add_argument("beat")
    a = ap.parse_args()
    po = os.path.join(CF, f"{a.beat}_offsets.npy")
    stale = (not os.path.exists(po)) or os.path.getmtime(po) < os.path.getmtime(os.path.join(ch.CH, f"{a.beat}.mkv"))
    D = measure(a.beat) if (a.cmd == "measure" or stale) else np.load(po)
    for f in (1, 40, 80):
        print(a.beat, f, {g[:14]: np.round(D[min(f, len(D) - 1), gi], 1).tolist() for gi, g in enumerate(GARMENTS)})
    if a.cmd == "seed":
        L = np.load(os.path.join(CF, f"{a.beat}_labels.npz"))["labels"]
        fr = read_frames(os.path.join(ch.CH, f"{a.beat}.mkv"))
        out = apply(fr[-1], L[-1], D[-1])
        p = os.path.join(ch.CH, f"{a.beat}_last_cc.png")
        cv2.imwrite(p, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print("wrote", p)


if __name__ == "__main__":
    main()


L_MID = 0.35 / STRENGTH   # brightness part applied mid-beat at assembly (lighting/pose make L unreliable)


def blended_offsets(ids, resets, n=12, ramp=16):
    """Per-beat offsets, made continuous across ordinary beat boundaries (frame 80 of one beat and frame 1 of the
    next show the same people, so their corrections must agree). Boundaries where the next beat started from a
    colour-corrected frame (`resets`) are left alone: there the correction is already in the new beat's pixels."""
    D = {}
    for b in ids:
        f = os.path.join(CF, f"{b}_offsets.npy")
        if os.path.exists(f):
            x = np.load(f)
            if x.shape[1] < len(GARMENTS):       # older measurement with fewer garments: zero offsets for the rest
                x = np.concatenate([x, np.zeros((x.shape[0], len(GARMENTS) - x.shape[1], 3), x.dtype)], 1)
            D[b] = x
    for a, b in zip(ids, ids[1:]):
        if b in resets or a not in D or b not in D:
            continue
        m = 0.5 * (D[a][80] + D[b][1])
        da, db = m - D[a][80], m - D[b][1]
        for j in range(n):
            w = (j + 1) / n                      # 0 -> 1 towards the boundary
            D[a][80 - n + 1 + j] += w * da
            D[b][1 + j] += (1 - j / n) * db      # 1 -> 0 away from the boundary
    # brightness: reduced mid-beat; full value at the end of a beat whose last frame seeded the next beat
    for k, b in enumerate(ids):
        if b not in D:
            continue
        f = np.full(len(D[b]), L_MID, np.float32)
        nxt = ids[k + 1] if k + 1 < len(ids) else None
        if nxt in resets:
            f[-ramp:] = np.linspace(L_MID, 1.0, ramp)
        D[b][:, :, 0] *= f[:, None]
    return D


def correct_with(frames, beat, D):
    L = np.load(os.path.join(CF, f"{beat}_labels.npz"))["labels"]
    return np.stack([apply(f, L[i], D[i]) for i, f in enumerate(frames)])

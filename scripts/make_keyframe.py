"""Compose an end keyframe in the fixed camera view from people cut out of locked chain frames.

Used where the video model will not hold the camera (A17: she backs away up the platform, he stands up).
Both ends of the beat are then given to Wan2.2 (first + last frame), so the camera has no reason to move.

Each cut-out = person mask (SAM3, warped to the plate) + their shadow (pixels near the person that differ from
the plate), taken from a locked frame (already in plate coordinates), optionally scaled about the feet and moved.
Perspective (measured on the man's walk-in, A02-A03): focal ~1570 px; a 1.65 m person at depth D is 1570*1.65/D px.

  python make_keyframe.py --out work/keyframes/A17_end.png \
      --cut A04:80:man --cut A16:80:woman:0.70:540:545
  (--cut beat:frame:label[:scale:foot_x:foot_y]; the cut-out's bottom-centre is placed at foot_x, foot_y)
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from assemble import read_frames  # noqa: E402

from sg_paths import ROOT  # noqa: E402


def people_mask(beat, f, shape):
    pm = np.load(os.path.join(ROOT, "work", "lock", beat, "people.npz"))["masks"][f]
    H = np.load(os.path.join(ROOT, "work", "lock", beat, "homographies.npy"))[f]
    return cv2.warpPerspective(pm.astype(np.uint8), H, (shape[1], shape[0]), flags=cv2.INTER_NEAREST) > 0


def cutout(frame, plate, pmask, pick, inst_mask=None, grow_mask=True):
    """Person component chosen by `pick` ('left' / 'right' / 'largest' / 'inst') + attached shadow; soft alpha."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(pmask.astype(np.uint8))
    comps = [(c, st[c]) for c in range(1, n) if st[c][4] > 2000]
    if pick == "inst":
        person = inst_mask
        lab = np.where(person, 1, 0); c = 1
    elif pick == "left":
        c = min(comps, key=lambda t: t[1][0] + t[1][2] / 2)[0]
    elif pick == "right":
        c = max(comps, key=lambda t: t[1][0] + t[1][2] / 2)[0]
    else:
        c = max(comps, key=lambda t: t[1][4])[0]
    if pick != "inst":
        person = (lab == c)
    d = np.abs(frame.astype(np.int16) - plate.astype(np.int16)).max(-1) > 6
    if pick == "inst" and grow_mask:
        # chain frames are background-locked, so anything clearly different from the plate right next to the
        # instance mask belongs to that person (bag, hands, hair the mask missed), unless it is the other person
        oth = cv2.dilate((pmask & ~person).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        strong = np.abs(frame.astype(np.int16) - plate.astype(np.int16)).max(-1) > 14
        grow = cv2.dilate(person.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
        hull = np.zeros(person.shape, np.uint8)
        cnts, _ = cv2.findContours(person.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(cnts))], 1)
        add = strong & (grow | (hull > 0)) & ~oth
        add = cv2.morphologyEx(add.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        person = person | add
        person = cv2.morphologyEx(person.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    near = cv2.dilate(person.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (181, 181))) > 0
    other = cv2.dilate((pmask & ~person).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    shadow = d & near & ~other & ~person
    shadow = cv2.morphologyEx(shadow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mp = cv2.GaussianBlur(cv2.dilate(person.astype(np.float32), np.ones((3, 3), np.uint8)), (0, 0), 1.2)
    ms = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), 1.5) * (1 - mp)
    ys, xs = np.where(person)
    foot = (float(xs.mean()), float(ys.max()))
    return (mp, ms), foot, person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut", action="append", required=True)
    ap.add_argument("--plate", default=os.path.join(ROOT, "work", "plates", "plate.png"))
    a = ap.parse_args()
    plate = cv2.cvtColor(cv2.imread(a.plate), cv2.COLOR_BGR2RGB)
    out = plate.astype(np.float32).copy()
    H, W = plate.shape[:2]
    for spec in a.cut:
        p = spec.split(":")
        beat, f, label = p[0], int(p[1]), p[2]
        src = os.path.join(ROOT, "chain", f"{beat}_last_cc.png")
        if f == 80 and os.path.exists(src):
            frame = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        else:
            frame = read_frames(os.path.join(ROOT, "chain", f"{beat}.mkv"))[f]
            cf = os.path.join(ROOT, "work", "colorfix")
            if os.path.exists(os.path.join(cf, f"{beat}_offsets.npy")):   # same garment colour fix as the chain
                import colorfix
                frame = colorfix.apply(frame, np.load(os.path.join(cf, f"{beat}_labels.npz"))["labels"][f],
                                       np.load(os.path.join(cf, f"{beat}_offsets.npy"))[f])
                print(f"  colour-corrected {beat} f{f}")
        pm = people_mask(beat, f, plate.shape)
        inst = os.path.join(ROOT, "work", "keyframes", f"{beat}_f{f}_{label}.png")
        if os.path.exists(inst):            # instance mask from sam3_instance.py (separates overlapping people)
            im = cv2.imread(inst, 0) > 127
            pm = im | (pm & ~cv2.dilate(pm.astype(np.uint8) & ~im.astype(np.uint8), np.ones((1, 1), np.uint8)).astype(bool))
            # "other" = people-mask blobs that do not touch this instance (her handbag touches her, so it stays hers);
            # for a person who is moved, everything that is not their instance counts as other
            allp = people_mask(beat, f, plate.shape)
            moved_spec = len(p) > 3 and p[3] != ""
            touch = cv2.dilate(im.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            nb, lb = cv2.connectedComponents((allp & ~touch).astype(np.uint8))
            pm_other = np.zeros_like(allp)
            for c in range(1, nb):
                comp = lb == c
                near_inst = (cv2.dilate(comp.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & im
                if comp.sum() > 1500 and near_inst.sum() < 0.02 * comp.sum():
                    pm_other |= comp
            if moved_spec:
                pm_other = allp & ~cv2.dilate(im.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            pm = im
            moved_spec = len(p) > 3 and p[3] != ""
            m, foot, person = cutout(frame, plate, pm | pm_other, "inst", inst_mask=im, grow_mask=not moved_spec)
        else:
            pick = "left" if label == "woman" else "largest"
            m, foot, person = cutout(frame, plate, pm, pick)
        mp, ms = m
        flip = len(p) > 6 and p[6] == "flip"
        if len(p) > 3 and p[3] != "":
            s, fx, fy = float(p[3]), float(p[4]), float(p[5])
            M = np.array([[s, 0, fx - s * foot[0]], [0, s, fy - s * foot[1]]], np.float32)
        else:
            M = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
        fl = cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_LINEAR
        fw = cv2.warpAffine(frame.astype(np.float32), M, (W, H), flags=fl)
        msw = cv2.warpAffine(ms, M, (W, H), flags=cv2.INTER_LINEAR)
        moved = abs(M[0, 2]) > 0.5 or abs(M[1, 2]) > 0.5 or abs(M[0, 0] - 1) > 1e-3
        if moved:
            # shadow as a darkening of whatever is at the destination (never copy source background pixels)
            gf = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) + 1
            gp = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY).astype(np.float32) + 1
            r = np.clip(cv2.GaussianBlur(gf, (0, 0), 2) / cv2.GaussianBlur(gp, (0, 0), 2), 0.3, 1.0)
            sm = ms * (r < 0.85)
            sm = cv2.GaussianBlur(sm, (0, 0), 1.5)
            smw = cv2.warpAffine(sm, M, (W, H), flags=cv2.INTER_LINEAR)
            rw = cv2.warpAffine(r, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=1.0)
            out = out * (1 - smw * (1 - rw))[..., None]
        else:
            out = msw[..., None] * fw + (1 - msw[..., None]) * out      # shadow first, never mirrored
        Mp = M.copy()
        if flip:                                                       # mirror the person about the foot x
            fx2 = M[0, 0] * foot[0] + M[0, 2]
            Mp = np.array([[-M[0, 0], 0, 2 * fx2 - M[0, 2]], [0, M[1, 1], M[1, 2]]], np.float32)
            Mp[0, 2] = 2 * fx2 - (M[0, 0] * 0 + M[0, 2]) - 2 * M[0, 0] * 0
            Mp = np.array([[-M[0, 0], 0, fx2 + M[0, 0] * foot[0]], [0, M[1, 1], M[1, 2]]], np.float32)
        fpw = cv2.warpAffine(frame.astype(np.float32), Mp, (W, H), flags=fl)
        mpw = cv2.warpAffine(mp, Mp, (W, H), flags=cv2.INTER_LINEAR)
        out = mpw[..., None] * fpw + (1 - mpw[..., None]) * out
        print(f"{label} from {beat} f{f}: foot {np.round(foot, 1).tolist()} -> {np.round(M @ [foot[0], foot[1], 1], 1).tolist()}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, cv2.cvtColor(out.round().clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

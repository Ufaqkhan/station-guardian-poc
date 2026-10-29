"""Build the plate (empty platform view) from the real GP02 camera frame by removing its burned-in timestamp.

The camera draws its clock as light-grey text on a translucent dark box at the top right (x >= 1391, y < 25 in the
1600x900 frame). Inside the box the scene is still visible: observed = g * scene + o. The gain and offset are fitted
from the box's edge rows/columns against the pixels just outside it, the scene is recovered as (observed - o) / g,
the text strokes are inpainted, colour noise (amplified by the inversion) is smoothed in Lab, and the patch is matched
to the rows just below the box. Every pixel outside the box is the original.

  python scripts/make_real_plate.py --src View-Display.jpeg

Writes (in $SG_ROOT/work/plates/):
  real/view_original.png    the source frame (1600x900)
  real/view_clean_1600.png  timestamp removed
  real/osd_removed_zoom.png before/after zoom of the corner (check it)
  plate_1536.png            full-size plate (legacy file name; it is the 1600x900 view)
  plate.png                 1280x720 plate used by the whole pipeline
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

from sg_paths import ROOT

X0, Y1 = 1391, 25          # timestamp box: x >= X0, y < Y1 (1600x900 frame)


def remove_osd(im):
    im = im.astype(np.float32)
    x0, y1 = X0, Y1
    inside = np.concatenate([im[23:25, x0:1600].reshape(-1, 3), im[0:25, x0 + 1:x0 + 3].reshape(-1, 3)])
    outside = np.concatenate([im[26:28, x0:1600].reshape(-1, 3), im[0:25, x0 - 3:x0 - 1].reshape(-1, 3)])
    keep = inside.mean(1) < 95                                  # ignore text pixels
    A = np.stack([outside[keep].ravel(), np.ones(keep.sum() * 3)], 1)
    g, o = np.linalg.lstsq(A, inside[keep].ravel(), rcond=None)[0]
    print(f"overlay fit: observed = {g:.3f} x scene + {o:.1f}")
    box = im[0:y1, x0:1600].copy()
    text = cv2.dilate((box.mean(-1) > 95).astype(np.uint8), np.ones((3, 3), np.uint8))
    rec = np.clip((box - o) / g, 0, 255).astype(np.uint8)
    rec = cv2.inpaint(rec, text, 3, cv2.INPAINT_TELEA)
    lab = cv2.cvtColor(rec, cv2.COLOR_BGR2LAB).astype(np.float32)   # chroma denoise: L lightly, a/b more
    lab[..., 0] = cv2.GaussianBlur(lab[..., 0], (0, 0), 0.7)
    lab[..., 1:] = cv2.GaussianBlur(lab[..., 1:], (0, 0), 2.0)
    rec = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    ref = im[26:40, x0:1600].reshape(-1, 3)                       # match the rows just below the box
    cur = rec[14:25].reshape(-1, 3)
    rec = (rec - cur.mean(0)) * (ref.std(0) / np.maximum(cur.std(0), 1)) + ref.mean(0)
    out = im.copy()
    out[0:y1, x0:1600] = rec
    out[25, x0:1600] = 0.5 * out[24, x0:1600] + 0.5 * out[26, x0:1600]     # partial boundary row / column
    out[0:26, x0 - 1] = 0.5 * out[0:26, x0 - 2] + 0.5 * out[0:26, x0]
    return out.round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "work", "plates", "real", "view_original.png"),
                    help="real camera frame, 1600x900 (e.g. View-Display.jpeg)")
    a = ap.parse_args()
    real = os.path.join(ROOT, "work", "plates", "real")
    os.makedirs(real, exist_ok=True)
    im = cv2.imread(a.src)
    assert im is not None and im.shape[:2] == (900, 1600), f"expected a 1600x900 frame: {a.src}"
    orig = os.path.join(real, "view_original.png")
    if os.path.abspath(a.src) != os.path.abspath(orig):
        cv2.imwrite(orig, im)
    out = remove_osd(im)
    cv2.imwrite(os.path.join(real, "view_clean_1600.png"), out)
    z = np.concatenate([im[0:60, 1300:1600], out[0:60, 1300:1600]], 0)
    cv2.imwrite(os.path.join(real, "osd_removed_zoom.png"),
                cv2.resize(z, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    pil.save(os.path.join(ROOT, "work", "plates", "plate_1536.png"))
    pil.resize((1280, 720), Image.LANCZOS).save(os.path.join(ROOT, "work", "plates", "plate.png"))
    print("wrote", os.path.join(ROOT, "work", "plates", "plate.png"))


if __name__ == "__main__":
    main()

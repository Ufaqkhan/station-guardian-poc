"""1080p stage: SeedVR2 (7B) upscaling of the 720p master, then a background re-lock at 1080p.

  python scripts/sr_stage.py queue  --master work/master/story_720.mkv --tag story   # queues SeedVR2 on a worker GPU
  python scripts/sr_stage.py relock --master work/master/story_720.mkv --tag story   # -> work/master/story_1080.mkv

Re-lock: out = a * SR(frame) + (1 - a) * SR(plate) * gain, with the people matte `a` (upsampled from the 720p
composite and refined against the upscaled frame) and the shadow map from compose.py. The platform is therefore one
upscaled still for the whole video: no texture shimmer between upscaling batches, exactly like a bolted camera.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compose import RATIO, guided  # noqa: E402

from sg_paths import ROOT, SEEDVR2_DIR as TOOL, SEEDVR2_PY as PY  # noqa: E402
W1, H1 = 1920, 1080


def sr_cmd(inp, out, batch=25, fmt="png"):
    return (f"cd {TOOL} && {PY} inference_cli.py {inp} --output {out} --output_format {fmt} "
            f"--model_dir {TOOL}/models/SEEDVR2 --dit_model seedvr2_ema_7b_fp16.safetensors --resolution 1080 "
            f"--batch_size {batch} --uniform_batch_size --temporal_overlap 4 --color_correction lab "
            f"--attention_mode sdpa --tensor_offload_device cpu")


def queue_cmd(name, cmd):
    q = os.path.join(ROOT, "work", "queue")
    n = f"{name}_{int(time.time())}"
    p = os.path.join(q, n + ".json")
    json.dump(dict(name=n, cmd=cmd, priority=-2), open(p + ".tmp", "w"))
    os.rename(p + ".tmp", p)
    return n


def wait(n):
    q = os.path.join(ROOT, "work", "queue")
    while True:
        for sub in ("done", "failed"):
            hit = [p for p in os.listdir(os.path.join(q, sub)) if p.endswith(f"__{n}.json")]
            if hit:
                d = json.load(open(os.path.join(q, sub, hit[0])))
                if sub == "failed" or d.get("returncode"):
                    raise RuntimeError(d.get("stderr", "")[-2000:])
                return d
        time.sleep(10)


def do_queue(a):
    wd = os.path.join(ROOT, "work", "master")
    mp4 = os.path.join(wd, f"{a.tag}_720_sr_in.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.master, "-c:v", "libx264", "-crf", "2", "-preset",
                    "slow", "-pix_fmt", "yuv444p", mp4], check=True)
    jobs = [queue_cmd(f"sr_{a.tag}", sr_cmd(mp4, os.path.join(wd, f"{a.tag}_sr")))]
    plate_sr = os.path.join(ROOT, "work", "plates", "plate_sr")
    if not glob.glob(os.path.join(plate_sr, "*.png")):
        jobs.append(queue_cmd("sr_plate", sr_cmd(os.path.join(ROOT, "work", "plates", "plate_1536.png"), plate_sr,
                                                 batch=1)))
    print("queued", jobs)
    for j in jobs:
        wait(j)
        print("done", j, flush=True)


def load_mattes(beats):
    out = {}
    for b in beats:
        m = np.load(os.path.join(ROOT, "work", "lock", b, "matte.npz"))
        out[b] = (m["alpha"], m["shadow"])
    return out


def do_relock(a):
    wd = os.path.join(ROOT, "work", "master")
    meta = json.load(open(a.master + ".json"))
    beats, n = meta["beats"], meta["frames"]
    srs = sorted(glob.glob(os.path.join(wd, f"{a.tag}_sr", "**", "*.png"), recursive=True))
    assert len(srs) == n, (len(srs), n)
    pf = sorted(glob.glob(os.path.join(ROOT, "work", "plates", "plate_sr", "**", "*.png"), recursive=True))[0]
    plate = cv2.cvtColor(cv2.imread(pf), cv2.COLOR_BGR2RGB)
    if plate.shape[:2] != (H1, W1):
        plate = cv2.resize(plate, (W1, H1), interpolation=cv2.INTER_LANCZOS4)
    plate = plate.astype(np.float32)
    mattes = load_mattes(beats)
    out_path = os.path.join(wd, f"{a.tag}_1080.mkv")
    wr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W1}x{H1}",
                           "-r", "16", "-i", "-", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", out_path],
                          stdin=subprocess.PIPE)
    TAIL = 8
    for k in range(n):
        bi, i = k // 80, k % 80 + 1
        b = beats[bi]
        al, sh = mattes[b]
        if i > 80 - TAIL and bi + 1 < len(beats):         # same easing into the next beat as assemble.py
            j = i - (80 - TAIL + 1)
            w = 0.5 - 0.5 * np.cos(np.pi * (j + 1) / TAIL)
            nal, nsh = mattes[beats[bi + 1]]
            al = {i: ((1 - w) * al[i] + w * nal[0]).astype(np.uint8)}
            sh = {i: ((1 - w) * sh[i] + w * nsh[0]).astype(np.uint8)}
        fr = cv2.cvtColor(cv2.imread(srs[k]), cv2.COLOR_BGR2RGB)
        if fr.shape[:2] != (H1, W1):
            fr = cv2.resize(fr, (W1, H1), interpolation=cv2.INTER_LANCZOS4)
        a0 = cv2.resize(al[i].astype(np.float32) / 255.0, (W1, H1), interpolation=cv2.INTER_LINEAR)
        a1 = np.clip(cv2.GaussianBlur((a0 > 0.5).astype(np.float32), (0, 0), 1.3), 0, 1)   # crisp 1080p edge
        s = cv2.resize(sh[i].astype(np.float32) / 255.0, (W1, H1), interpolation=cv2.INTER_LINEAR)
        gain = 1.0 - s[..., None] * (1.0 - RATIO[None, None, :])
        o = a1[..., None] * fr.astype(np.float32) + (1 - a1[..., None]) * plate * gain
        wr.stdin.write(o.round().clip(0, 255).astype(np.uint8).tobytes())
        if k % 160 == 0:
            print(f"relock {k}/{n}", flush=True)
    wr.stdin.close()
    wr.wait()
    json.dump(meta, open(out_path + ".json", "w"))
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["queue", "relock"])
    ap.add_argument("--master", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "work", "master"), exist_ok=True)
    (do_queue if a.cmd == "queue" else do_relock)(a)


if __name__ == "__main__":
    main()

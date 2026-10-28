"""Persistent Wan2.2-I2V-A14B worker for the Station Guardian PoC.

Loads the two 14B experts once (fp16 Comfy repackage -> bf16), adds the LightX2V 4-step adapters, then
takes generation requests from a queue folder until it has been idle for --idle_min minutes or a file
named STOP appears in the queue folder.

Request = one JSON file in <queue>/  (claimed by an atomic rename into <queue>/running/):
  {"name": "A04_s3", "image": "start.png", "prompt": "...", "seed": 3, "out_dir": "...",
   "last_image": null, "num_frames": 81, "sigmas": [1.0, 0.75, 0.5, 0.25], "guidance": 1.0,
   "negative_prompt": "..."}
Outputs per request in out_dir:
  <name>.mkv        lossless (FFV1, RGB) 16 fps
  <name>.mp4        preview (x264 crf 18)
  <name>_last.png   last frame, lossless (start image for the next beat)
  <name>_sheet.jpg  3x3 contact sheet of frames 0,10,...,80
  <name>.json       the request plus timing
"""
import argparse
import glob
import json
import os
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw

from sg_paths import HF_HUB as HF  # noqa: E402
BASE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DEFAULT_NEG = (
    "camera movement, camera shake, zoom, pan, cut, scene change, text, subtitles, watermark, logo, blurry faces, "
    "deformed hands, extra limbs, extra people, duplicated people, distorted body, morphing, flicker, cartoon, "
    "illustration, CGI, oversaturated, overexposed, low quality"
)


def snap(repo, *parts):
    d = glob.glob(os.path.join(HF, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    assert d, f"{repo} not downloaded"
    return os.path.join(d[0], *parts)


def merged_transformer(ckpt, lora, subfolder, scale=1.0, dev="cuda"):
    """Original-format Wan checkpoint + LightX2V adapter merged ComfyUI-style, then converted to diffusers.

    diffusers' own LoRA converter drops the adapter's `diff` / `diff_m` tensors (norm, patch-embedding, head and
    modulation offsets), so the merge is done here in the original key space:
      X.lora_down/up -> X.weight += up @ down   (no alpha key -> scale 1)
      X.diff_b -> X.bias += ;  X.diff -> X.weight += ;  X.diff_m -> X.modulation +=
    """
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
    from safetensors.torch import load_file

    sd = load_file(ckpt)
    n = 0
    skipped = []
    if lora:
        ad = {k.replace("diffusion_model.", "", 1): v for k, v in load_file(lora).items()}
        def target(k):
            for suf, rep in ((".lora_down.weight", ".weight"), (".diff_b", ".bias"), (".diff_m", ".modulation"),
                             (".diff", ".weight")):
                if k.endswith(suf):
                    return k[: -len(suf)] + rep
            raise KeyError(f"unhandled adapter key {k}")

        for k, v in ad.items():
            if k.endswith(".lora_up.weight"):
                continue
            if target(k) not in sd:
                skipped.append(k)   # e.g. k_img / v_img: Wan2.1 image branch, absent in Wan2.2 I2V
                continue
            if k.endswith(".lora_down.weight"):
                base = k[: -len(".lora_down.weight")]
                up = ad[base + ".lora_up.weight"].to(dev).float()
                dw = (up @ v.to(dev).float()) * scale
                tgt = base + ".weight"
                sd[tgt] = (sd[tgt].to(dev).float() + dw.reshape(sd[tgt].shape)).to(torch.bfloat16).cpu()
            elif k.endswith(".diff_b"):
                tgt = k[: -len(".diff_b")] + ".bias"
                sd[tgt] = sd[tgt].float() + scale * v.float()
            elif k.endswith(".diff_m"):
                tgt = k[: -len(".diff_m")] + ".modulation"
                sd[tgt] = sd[tgt].float() + scale * v.float()
            elif k.endswith(".diff"):
                tgt = k[: -len(".diff")] + ".weight"
                sd[tgt] = sd[tgt].float() + scale * v.float().reshape(sd[tgt].shape)
            else:
                raise KeyError(f"unhandled adapter key {k}")
            n += 1
    conv = convert_wan_transformer_to_diffusers(sd)
    keep32 = WanTransformer3DModel._keep_in_fp32_modules or []
    conv = {k: (v.float() if any(m in k.split(".") or k.startswith(m) for m in keep32) else v.to(torch.bfloat16))
            for k, v in conv.items()}
    cfg = WanTransformer3DModel.load_config(snap(BASE), subfolder=subfolder)
    with init_empty_weights():
        model = WanTransformer3DModel.from_config(cfg)
    missing, unexpected = model.load_state_dict(conv, strict=False, assign=True)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    n32 = sum(p.dtype == torch.float32 for p in model.parameters())
    print(f"{subfolder}: merged {n} adapter tensors, skipped {len(skipped) if lora else 0}, fp32 params {n32}", flush=True)
    return model.eval()


def load_pipe(use_lora=True):
    from diffusers import FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline

    comfy = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    lx = "lightx2v/Wan2.2-Distill-Loras"
    t0 = time.time()
    hi = merged_transformer(snap(comfy, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
                            snap(lx, "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors")
                            if use_lora else None, "transformer").to("cuda")
    lo = merged_transformer(snap(comfy, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
                            snap(lx, "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors")
                            if use_lora else None, "transformer_2").to("cuda")
    pipe = WanImageToVideoPipeline.from_pretrained(snap(BASE), transformer=hi, transformer_2=lo,
                                                   torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipe.to("cuda")
    print(f"pipeline ready in {time.time() - t0:.0f}s; boundary_ratio={pipe.config.boundary_ratio}", flush=True)
    return pipe


def set_sigmas(pipe, sigmas):
    """Make the pipeline's internal set_timesteps call use our explicit sigma list (then shift 5)."""
    sched = pipe.scheduler
    orig = type(sched).set_timesteps

    def patched(self, num_inference_steps=None, device=None, **kw):
        return orig(self, sigmas=list(sigmas), device=device)

    sched.set_timesteps = patched.__get__(sched)


def write_video(frames_u8, path_mkv, path_mp4, fps=16):
    h, w = frames_u8.shape[1:3]
    raw = frames_u8.tobytes()
    base = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", "-"]
    subprocess.run(base + ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", path_mkv], input=raw, check=True)
    subprocess.run(base + ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", path_mp4],
                   input=raw, check=True)


def contact_sheet(frames_u8, path, idx=(0, 10, 20, 30, 40, 50, 60, 70, 80), tw=640, th=360):
    idx = [i for i in idx if i < len(frames_u8)]
    sheet = Image.new("RGB", (3 * tw, ((len(idx) + 2) // 3) * th))
    for k, i in enumerate(idx):
        t = Image.fromarray(frames_u8[i]).resize((tw, th), Image.LANCZOS)
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 60, 16], fill="black")
        d.text((3, 2), f"f{i:02d}", fill="yellow")
        sheet.paste(t, ((k % 3) * tw, (k // 3) * th))
    sheet.save(path, quality=90)


def run_request(pipe, req):
    out = req["out_dir"]
    os.makedirs(out, exist_ok=True)
    name = req["name"]
    img = Image.open(req["image"]).convert("RGB")
    W, H = req.get("width", 1280), req.get("height", 720)
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    last = None
    if req.get("last_image"):
        last = Image.open(req["last_image"]).convert("RGB").resize((W, H), Image.LANCZOS)
    set_sigmas(pipe, req.get("sigmas", [1.0, 0.75, 0.5, 0.25]))
    g = torch.Generator("cuda").manual_seed(int(req["seed"]))
    t0 = time.time()
    kw = dict(image=img, prompt=req["prompt"], negative_prompt=req.get("negative_prompt", DEFAULT_NEG),
              height=H, width=W, num_frames=req.get("num_frames", 81), num_inference_steps=len(req.get("sigmas", [0] * 4)),
              guidance_scale=req.get("guidance", 1.0), guidance_scale_2=req.get("guidance2", req.get("guidance", 1.0)),
              generator=g, output_type="np")
    if last is not None:
        kw["last_image"] = last
    frames = pipe(**kw).frames[0]
    gen_s = time.time() - t0
    u8 = (np.clip(frames, 0, 1) * 255 + 0.5).astype(np.uint8)
    write_video(u8, os.path.join(out, name + ".mkv"), os.path.join(out, name + ".mp4"))
    Image.fromarray(u8[-1]).save(os.path.join(out, name + "_last.png"))
    contact_sheet(u8, os.path.join(out, name + "_sheet.jpg"))
    req.update(gen_seconds=round(gen_s, 1), host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"),
               frames=int(len(u8)), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(req, open(os.path.join(out, name + ".json"), "w"), indent=1)
    print(f"done {name} in {gen_s:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--idle_min", type=float, default=25)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--max_hours", type=float, default=3.8)
    args = ap.parse_args()
    for sub in ("running", "done", "failed"):
        os.makedirs(os.path.join(args.queue, sub), exist_ok=True)
    pipe = load_pipe(use_lora=not args.no_lora)
    tag = f"{socket.gethostname()}_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    t_start, t_idle = time.time(), time.time()
    while True:
        if os.path.exists(os.path.join(args.queue, "STOP")):
            print("STOP file found", flush=True)
            break
        if time.time() - t_start > args.max_hours * 3600:
            print("time budget reached", flush=True)
            break
        reqs = sorted(glob.glob(os.path.join(args.queue, "*.json")),
                      key=lambda p: (json.load(open(p)).get("priority", 5), os.path.getmtime(p)))
        claimed = None
        for p in reqs:
            dst = os.path.join(args.queue, "running", f"{tag}__{os.path.basename(p)}")
            try:
                os.rename(p, dst)
                claimed = dst
                break
            except OSError:
                continue
        if claimed is None:
            if time.time() - t_idle > args.idle_min * 60:
                print("idle timeout", flush=True)
                break
            time.sleep(5)
            continue
        req = json.load(open(claimed))
        try:
            if req.get("cmd"):
                # auxiliary GPU work on this worker's GPU (e.g. SAM3 person masks in the sam3 env)
                t0 = time.time()
                r = subprocess.run(req["cmd"], shell=True, capture_output=True, text=True)
                req.update(returncode=r.returncode, stdout=r.stdout[-3000:], stderr=r.stderr[-3000:],
                           seconds=round(time.time() - t0, 1))
                json.dump(req, open(claimed, "w"), indent=1)
                print(f"cmd {req['name']} rc={r.returncode} in {time.time() - t0:.0f}s", flush=True)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-1500:])
            else:
                run_request(pipe, req)
            os.rename(claimed, os.path.join(args.queue, "done", os.path.basename(claimed)))
        except Exception:
            traceback.print_exc()
            req["error"] = traceback.format_exc()[-2000:]
            json.dump(req, open(claimed, "w"), indent=1)
            os.rename(claimed, os.path.join(args.queue, "failed", os.path.basename(claimed)))
        t_idle = time.time()


if __name__ == "__main__":
    main()

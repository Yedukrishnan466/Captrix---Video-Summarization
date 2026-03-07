import os, json, argparse
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
import random

def sample_frames(video_path, out_dir, samples=32):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(str(video_path))
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames=[]
        while True:
            ret, f = vid.read()
            if not ret: break
            frames.append(f)
        total = len(frames)
        if total == 0:
            vid.release()
            return []
        indices = np.linspace(0, total-1, samples, dtype=int)
        imgs=[]
        for i, idx in enumerate(indices):
            f = frames[idx]
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            outp = Path(out_dir)/f"{Path(video_path).stem}_frame_{i:04d}.jpg"
            Image.fromarray(rgb).save(outp)
            imgs.append(str(outp))
        vid.release()
        return imgs

    indices = np.linspace(0, max(0,total-1), samples, dtype=int)
    imgs = []
    for i, idx in enumerate(indices):
        vid.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = vid.read()
        if not ret:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outp = Path(out_dir)/f"{Path(video_path).stem}_frame_{i:04d}.jpg"
        Image.fromarray(img).save(outp)
        imgs.append(str(outp))
    vid.release()
    return imgs

def convert(input_json, videos_dir, out_meta="data/meta.json", samples=32, create_val=False, val_size=500, seed=42):
    with open(input_json, "r", encoding="utf8") as f:
        items = json.load(f)

    if create_val:
        random.seed(seed)
        random.shuffle(items)
        val_items = items[:val_size]
        train_items = items[val_size:]
    else:
        train_items = items
        val_items = []

    meta={}
    processed=0
    total = len(train_items)+len(val_items)
    print(f"Processing {total} videos (train {len(train_items)} + val {len(val_items)})")

    for dataset_part, rows in (("train", train_items), ("val", val_items)):
        for row in rows:
            vidid = row.get("video_id")
            vidname = row.get("video")
            caps = row.get("caption") or []
            if isinstance(caps, str):
                caps = [caps]
            caps = list(dict.fromkeys(caps))  # remove exact duplicates
            video_path = os.path.join(videos_dir, vidname)
            if not os.path.exists(video_path):
                print(f"[WARN] Missing video {video_path} - skipping")
                continue
            out_frames_dir = os.path.join("data","frames",vidid)
            frames = sample_frames(video_path, out_frames_dir, samples=samples)
            if len(frames)==0:
                print(f"[WARN] No frames for {vidid}")
                continue
            meta[vidid] = {"video": os.path.relpath(video_path), "frames": frames, "captions": caps}
            processed += 1
            if processed%200==0:
                print(f"Processed {processed}/{total}")

    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    with open(out_meta, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta to {out_meta}. Total processed: {processed}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--videos_dir", default="dataset/video")
    p.add_argument("--out_meta", default="data/meta.json")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--create_val", action="store_true")
    p.add_argument("--val_size", type=int, default=500)
    args=p.parse_args()
    convert(args.input_json, args.videos_dir, args.out_meta, samples=args.samples, create_val=args.create_val, val_size=args.val_size)

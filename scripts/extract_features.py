import os, json, argparse
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

def extract(meta_path, out_dir, device="cuda"):
    with open(meta_path,"r",encoding="utf8") as f:
        meta = json.load(f)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    os.makedirs(out_dir, exist_ok=True)
    for vidid, info in meta.items():
        frames = info["frames"]
        imgs = [Image.open(p).convert("RGB") for p in frames]
        inputs = proc(images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_feats = model.get_image_features(**inputs)  # (N,D)
        img_feats = img_feats.cpu().numpy()
        caps = info.get("captions", [])
        if len(caps)>0:
            token = proc(text=caps, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                txt_feats = model.get_text_features(**token)  # (C,D)
            txt_feats = txt_feats.cpu().numpy()
        else:
            txt_feats = np.zeros((0, img_feats.shape[1]), dtype=np.float32)
        np.savez_compressed(os.path.join(out_dir, f"{vidid}.npz"), img_feats=img_feats, txt_feats=txt_feats)
        print("Saved features:", vidid)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--meta", default="data/meta.json")
    p.add_argument("--out", default="data/features")
    p.add_argument("--device", default="cuda")
    args=p.parse_args()
    extract(args.meta, args.out, args.device)

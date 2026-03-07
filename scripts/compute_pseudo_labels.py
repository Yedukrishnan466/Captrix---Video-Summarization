import os, argparse, numpy as np, json

def compute_all(features_dir, out_json):
    out={}
    for f in os.listdir(features_dir):
        if not f.endswith(".npz"): continue
        data = np.load(os.path.join(features_dir,f))
        img_feats = data["img_feats"]  # (N,D)
        txt_feats = data["txt_feats"]  # (C,D)
        if txt_feats.shape[0] == 0:
            labels = [0.0]*img_feats.shape[0]
        else:
            img_norm = img_feats / (np.linalg.norm(img_feats,axis=1,keepdims=True)+1e-8)
            txt_norm = txt_feats / (np.linalg.norm(txt_feats,axis=1,keepdims=True)+1e-8)
            sims = img_norm.dot(txt_norm.T)  # (N,C)
            labels = sims.mean(axis=1)
            minv, maxv = labels.min(), labels.max()
            if maxv-minv > 1e-6:
                labels = (labels - minv) / (maxv - minv)
            else:
                labels = np.zeros_like(labels)
        vidid = f.replace(".npz","")
        out[vidid] = labels.tolist()
    with open(out_json,"w",encoding="utf8") as fp:
        json.dump(out, fp, indent=2)
    print("Saved pseudo labels to", out_json)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--out", default="data/pseudo_labels.json")
    args=p.parse_args()
    compute_all(args.features_dir, args.out)

import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FrameDataset(Dataset):
    def __init__(self, features_dir, labels_json):
        self.items=[]
        with open(labels_json,'r',encoding='utf8') as f:
            labels = json.load(f)
        for fn in os.listdir(features_dir):
            if not fn.endswith(".npz"): continue
            vidid = fn.replace(".npz","")
            arr = np.load(os.path.join(features_dir,fn))
            feats = arr['img_feats']  # (N,D)
            lbls = labels.get(vidid, [0]*feats.shape[0])
            for i in range(feats.shape[0]):
                self.items.append((feats[i].astype(np.float32), float(lbls[i])))
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        f,l = self.items[idx]
        return torch.from_numpy(f), torch.tensor(l, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x).squeeze(-1)

def train(features_dir, labels_json, out_model, epochs=8, batch=64, lr=1e-3, device='cuda'):
    ds = FrameDataset(features_dir, labels_json)
    num_workers = 0  # safe for Windows
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=num_workers)
    # find input dim
    sample_np = next(x for x in os.listdir(features_dir) if x.endswith(".npz"))
    in_dim = np.load(os.path.join(features_dir, sample_np))['img_feats'].shape[1]
    model = MLP(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        total_loss=0.0
        for xb,yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()*xb.size(0)
        print(f"Epoch {ep+1}/{epochs} loss={total_loss/len(ds):.6f}")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    torch.save(model.state_dict(), out_model)
    print("Saved model to", out_model)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--labels_json", default="data/pseudo_labels.json")
    p.add_argument("--out_model", default="models/selector.pth")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    args=p.parse_args()
    train(args.features_dir, args.labels_json, args.out_model, epochs=args.epochs, batch=args.batch, lr=args.lr, device=args.device)

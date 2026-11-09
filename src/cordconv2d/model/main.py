from pathlib import Path
import torch as t

from cordconv2d.dataset.datasets.replay_shard import ReplayShards
from cordconv2d.dataset.datasets.utils import read_dataset_meta
from cordconv2d.model.train import train
from cordconv2d.model.model import Iwmymdosr

def init(m):
    if isinstance(m, (t.nn.Linear, t.nn.Conv2d)):
        t.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

if __name__ == "__main__":
    device = t.device("cuda")
    data_path = Path(rf"E:\iwmymdosr_dataset/shards")
    train_path = data_path / "train"
    val_path = data_path / "val"
    train_meta = read_dataset_meta(train_path)
    W, H = train_meta.resolution
    model = Iwmymdosr().to(device)
    model(t.zeros((1, 1, H, W), dtype=t.float32).to(device))

    model.apply(init)
    t.nn.init.normal_(model.head[5].weight, std=1e-2)
    trainset = t.utils.data.DataLoader(ReplayShards(train_path, shuffle=True, seed=42), batch_size=128)
    valset = t.utils.data.DataLoader(ReplayShards(val_path, shuffle=False), batch_size=128)
    train(model, trainset, valset, device, "./models", lr=1e-4, report_every=50)

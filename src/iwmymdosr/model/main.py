from pathlib import Path
import torch as t

from iwmymdosr.dataset.datasets.replay_shard import ReplayShards
from iwmymdosr.dataset.datasets.utils import read_dataset_meta
from iwmymdosr.model.baseline import get_baselines
from iwmymdosr.model.criterions.weighted_mse import weighted_mse
from iwmymdosr.model.early_stopping import EarlyStopping
from iwmymdosr.model.train import train
from iwmymdosr.model.model import Iwmymdosr

def init(m):
    if isinstance(m, (t.nn.Linear, t.nn.Conv2d)):
        t.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

if __name__ == "__main__":
    device = t.device("cuda")
    data_path = Path(rf"D:\datasets\iwmymdosr_dataset\shards")
    train_path = data_path / "train"
    val_path = data_path / "val"

    model = Iwmymdosr().to(device)
    model.apply(init)

    criterion = weighted_mse(k=35)
    early_stopping = EarlyStopping(model, "./models", patience=5, min_delta=1e-4)

    t.nn.init.normal_(model.head.weight, std=1e-2)
    trainset = t.utils.data.DataLoader(ReplayShards(train_path, shuffle=True, seed=42), batch_size=128, persistent_workers=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    valset = t.utils.data.DataLoader(ReplayShards(val_path, shuffle=False), batch_size=128, persistent_workers=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    mean_loss, zeros_loss = get_baselines(trainset, valset, criterion)

    print(f"Predicting mean loss: {mean_loss}")
    print(f"Predicting zeros loss: {zeros_loss}")

    train(model, trainset, valset, device, criterion, early_stopping, lr=3e-4, report_every=50)

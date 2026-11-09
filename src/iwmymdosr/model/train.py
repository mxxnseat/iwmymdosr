from os import PathLike

import torch as t
from iwmymdosr.dataset.datasets.objects import Meta
from iwmymdosr.dataset.datasets.replay_shard import ReplayShards
from iwmymdosr.model.early_stopping import EarlyStopping
from iwmymdosr.model.model import Iwmymdosr

def get_pos_weight(dataset: t.utils.data.DataLoader):
    total = 0
    pos = 0
    neg = 0
    for _, y in dataset:
        clicks = y[:, -1]
        total += clicks.size(0)
        pos += t.sum(clicks[t.where(clicks == 1)])

    neg = total - pos

    return t.tensor(neg/pos)

def train(model: Iwmymdosr, trainset: t.utils.data.DataLoader, valset: t.utils.data.DataLoader, device, output_dir: PathLike[str] | str, epoches = 100, lr=1e-3, report_every = 20):
    optim = t.optim.Adam(model.parameters(), lr)
    mse_criteria = t.nn.MSELoss()
    early_stopping = EarlyStopping(model, output_dir, min_delta=1e-3)

    for epoch_n in range(1, epoches+1):
        train_loss = 0.0
        model.train()
        for batch_n, (x, y) in enumerate(trainset, 1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            model_dxdy_out = model.forward(x)
            mse_loss = mse_criteria(model_dxdy_out, y[:, :2])
            loss = mse_loss
            loss.backward()
            optim.step()
            train_loss += loss.item() / len(trainset)
            if batch_n % report_every == 0:
                print(f"{epoch_n}/{batch_n} train loss {(train_loss / batch_n):.6f}")

        val_loss = 0.0
        model.eval()
        with t.no_grad():
            for x, y in valset:
                x, y = x.to(device), y.to(device)
                model_dxdy_out = model.forward(x)
                mse_loss = mse_criteria(model_dxdy_out, y[:, :2])
                loss = mse_loss
                val_loss += loss.item() / len(valset)

        print(f"Epoch {epoch_n}/{epoches}   train_loss {train_loss:.6f}")
        print(f"                            val_loss {val_loss:.6f}")
        print()
        print()
        if early_stopping.step(val_loss):
            print("stopping...")
            print(f"Best loss: {early_stopping.min_loss:.6f}")
            break

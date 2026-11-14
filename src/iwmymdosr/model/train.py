import torch as t
from iwmymdosr.model.model import Iwmymdosr
from iwmymdosr.model.utils import transfer_to_device

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

def train(model: Iwmymdosr, trainset: t.utils.data.DataLoader, valset: t.utils.data.DataLoader, device, criterion, early_stopping, epoches = 100, lr=1e-3, report_every = 20):
    optim = t.optim.Adam(model.parameters(), lr)

    for epoch_n in range(1, epoches+1):
        train_loss = 0.0
        model.train()
        samples_n = 0
        for batch_n, (x, y) in enumerate(trainset, 1):
            x, y = transfer_to_device(x, device), transfer_to_device(y, device)
            optim.zero_grad()
            with t.autocast("cuda", dtype=t.bfloat16):
                out = model.forward(x)
                loss = criterion(out, y)
            loss.backward()
            optim.step()
            train_loss += loss.item() * x.size(0)
            samples_n += x.size(0)
            if batch_n % report_every == 0:
                print(f"{epoch_n}/{batch_n} train loss {(train_loss / samples_n):.6f}")

        train_loss /= samples_n
        
        samples_n = 0
        val_loss = 0.0
        model.eval()
        with t.no_grad():
            for x, y in valset:
                x, y = transfer_to_device(x, device), transfer_to_device(y, device)
                with t.autocast("cuda", dtype=t.bfloat16):
                    out = model.forward(x)
                    loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                samples_n += x.size(0)
        
        val_loss /= samples_n

        print(f"Epoch {epoch_n}/{epoches}   train_loss {train_loss:.6f}")
        print(f"                            val_loss {val_loss:.6f}")
        print()
        print()
        if early_stopping.step(val_loss):
            print("stopping...")
            print(f"Best loss: {early_stopping.min_loss:.6f}")
            break

import torch as t

def get_baselines(trainset: t.utils.data.DataLoader, valset: t.utils.data.DataLoader, criterion):
    zeros_loss = 0.0
    mean_loss = 0.0
    samples_n = 0

    total, count = 0.0, 0
    for _, y in trainset:
        total += y.sum().item()
        count += y.numel()
    train_mean = total / count

    for _, y in valset:
        zeros_loss += criterion(t.zeros_like(y), y).item() * y.size(0)
        mean_loss += criterion(t.full_like(y, train_mean), y).item() * y.size(0)

        samples_n += y.size(0)

    mean_loss /= samples_n
    zeros_loss /= samples_n

    return mean_loss, zeros_loss

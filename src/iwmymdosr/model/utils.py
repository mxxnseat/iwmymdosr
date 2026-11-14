def log_grad_norms(model):
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = (p.grad.norm() / (p.norm() + 1e-12)).item()

    return norms

def transfer_to_device(data, device, non_blocking=True):
    return data.to(device, non_blocking=non_blocking)
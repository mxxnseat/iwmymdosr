from pynput import keyboard
import torch as t

from iwmymdosr.dataset.datasets.utils import read_dataset_meta
from iwmymdosr.iwmymdosr.runners.runner import Runner
from iwmymdosr.model.model import Iwmymdosr

is_playing = False

def on_press(key):
    global is_playing
    if hasattr(key, 'char') and key.char == 'o':
        is_playing = not is_playing

def main():
    device = t.device("cuda")
    meta = read_dataset_meta(r"E:\iwmymdosr_dataset\shards\train")
    model = Iwmymdosr().to(device)
    W,H = meta.resolution
    # lazy_layer_warmup = t.zeros((1, 2*meta.frames_n, H, W), dtype=t.float32, device=device)
    # model.forward(lazy_layer_warmup)
    model.load_state_dict(t.load(r"models\Iwmymdosr_1794180186.075168_best.pt", weights_only=True))
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # shard_loader = ShardLoader(r"E:\iwmymdosr_dataset\shards\val")
    # debug_runner = DebugRunner(model, meta, device, shard_loader, debug_resolution=(128, 72), debug_dir="./data/debug")
    # debug_runner.step()
    while not is_playing:
        ...
        
    runner = Runner(model, meta, device)
    runner.run()

if __name__ == "__main__":
    main()
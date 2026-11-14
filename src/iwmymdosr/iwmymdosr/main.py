from pathlib import Path

from pynput import keyboard
import torch as t

from iwmymdosr.dataset.datasets.replay_shard import ReplayShards
from iwmymdosr.dataset.datasets.utils import read_dataset_meta
from iwmymdosr.iwmymdosr.runners.debug_runner import DebugRunner
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
    data_path = Path(rf"D:\datasets\iwmymdosr_dataset\shards")
    val_path = data_path / "val"
    model.load_state_dict(t.load(r"models\Iwmymdosr_1794605511.2214444_best.pt", weights_only=True))
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not is_playing:
        ...
        
    runner = Runner(model, meta, device)
    # valset = t.utils.data.DataLoader(ReplayShards(val_path, shuffle=False), batch_size=128, persistent_workers=True, num_workers=8, pin_memory=True, prefetch_factor=8)

    # runner = DebugRunner(model, meta, device, valset)
    runner.run()

if __name__ == "__main__":
    main()
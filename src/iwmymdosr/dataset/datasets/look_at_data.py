import argparse

import torch as t
import matplotlib.pyplot as plt

from iwmymdosr.dataset.datasets.replay_shard import ReplayShards

def plot_sample(frames, heatmap):
    # frames: (3, C, H, W) tensor, heatmap: (H, W) tensor
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 3)
    frames = frames.squeeze(0)
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames[i].cpu())   # CHW -> HWC
        ax.set_title(f"F{i + 1}")
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])                    # middle column, bottom row
    im = ax.imshow(heatmap.squeeze(0).cpu(), cmap="brg", vmin=0, vmax=1)
    ax.set_title("H")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()

def main(args):
    data = t.utils.data.DataLoader(ReplayShards(args.shards_dir, args.shards_limit), batch_size=1)
    for ind, (x, y) in enumerate(data, 0):
        if ind == 0 or ind % args.step == 0:
            plot_sample(x, y)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--shards-dir", default="./data/source")
    arg_parser.add_argument("--shards-limit", default=None)
    arg_parser.add_argument("--step", type=int, default=5)

    args = arg_parser.parse_args()
    main(args)
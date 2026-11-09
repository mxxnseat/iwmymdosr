from cordconv2d.dataset.datasets.replay_shard_builder import ReplayShardBuilder
import argparse

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source", default="./data/source")
    arg_parser.add_argument("--output", default="./data/output")
    arg_parser.add_argument("--val-size", type=float, default=0.1)
    arg_parser.add_argument("--past-frames-n", default=3)
    arg_parser.add_argument("--max-replays", type=int)
    arg_parser.add_argument("--debug-output", default=None)
    arg_parser.add_argument("--specific", default=None)
    args = arg_parser.parse_args()
    ReplayShardBuilder(args.source, args.output, args.past_frames_n, val_frac=args.val_size, max_replays=args.max_replays, debug_folder=args.debug_output, specific=args.specific).build()
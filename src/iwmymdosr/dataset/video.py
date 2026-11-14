import argparse
import asyncio
from asyncio import subprocess
from os import PathLike
import os
from pathlib import Path
import sys
import time
import iwmymdosr.dataset.danser as danser
import ffmpeg

from iwmymdosr.dataset.danser import get_download_replay_cmd
from iwmymdosr.dataset.osu.scrapper.unzip import lookup_specific_diff
from iwmymdosr.utils.main import log

async def pump(stream):
    while True:
        line = await stream.readline()
        if not line:
            break
        log(pump, line.decode(errors='replace'))

async def kill_tree(proc):
    if sys.platform == "win32":
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/F", "/T", "/PID", str(proc.pid),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        proc.kill()

async def run(cmd, sem):
    async with sem:
        start = time.time()
        log(run, "Start")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(pump(proc.stdout), proc.wait()),
                timeout=3600,
            )
        except asyncio.TimeoutError:
            await kill_tree(proc)
            print(f"TIMEOUT after {time.time() - start:.0f}s", flush=True)
            return cmd, -1
        log(run, f"DONE code={proc.returncode} in {time.time() - start:.0f}s")
        return cmd, proc.returncode
def is_replay_has_video(replay: Path):
    return (replay.parent / f"{replay.stem}.mp4").exists()

def get_danser_no_objects_conf_name(danser_config):
    return danser_config+"no_objects"

def extract_video_audio(video_path: PathLike[str] | str):
    video_path = Path(video_path)

    audio_file = f"{video_path.parent}/{video_path.stem}.mp3"
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream, audio_file)
    ffmpeg.run(stream)

def extract_videos_audio(replays_dir: PathLike[str]):
    replays_dir = Path(replays_dir)

    for video in replays_dir.glob("*.mp4"):
        extract_video_audio(video)

async def get_video_from_replays(replays_dir: PathLike[str], max_replays = None, specific = None, danser_conf = "cordconv2d"):
    replays_dir = Path(replays_dir)
    commands = []

    replays = []

    if specific is None:
        candidates = replays_dir.rglob("*.osr")
        for candidate in candidates:
            if not max_replays is None:
                if max_replays <= 0:
                    break
                max_replays -= 1
            replays.append(candidate.resolve())
    else:
        replays.append(next(replays_dir.rglob(f"{specific}.osr")).resolve())


    for replay in replays:
        if is_replay_has_video(replay):
            continue
        commands.append(get_download_replay_cmd(replay, replay.stem, danser_conf))
    if not commands:
        print("commands are empty.")
        return
    
    semaphore = asyncio.Semaphore(2)
    results = await asyncio.gather(*(run(c, semaphore) for c in commands))
    for cmd, code in results:
        print(cmd[1], code)

def pipeline(args):
    if args.clean:
        cleanup_replays_without_beatmap(args.output_dir)
        cleanup_videos(args.output_dir)
        cleanup_video_audio(args.output_dir)
    if args.replays:
        danser.make_profile(args.danser_dir, Path(args.output_dir).resolve(), Path(args.source_dir).resolve(), args.danser_config)
        asyncio.run(get_video_from_replays(args.source_dir, args.max_replays, args.specific,  args.danser_config))
        extract_videos_audio(args.source_dir)

def cleanup_videos(source_dir: PathLike[str]):
    source_dir = Path(source_dir)
    files_to_delete = source_dir.rglob("*.mp4")

    for file in files_to_delete:
        os.remove(file)

def cleanup_video_audio(source_dir: PathLike[str]):
    source_dir = Path(source_dir)
    files_to_delete = source_dir.rglob("*.mp3")

    for file in files_to_delete:
        os.remove(file)

def cleanup_replays_without_beatmap(source_dir: PathLike[str]):
    source_dir = Path(source_dir)

    for replay_file in source_dir.rglob("*.osr"):
        if lookup_specific_diff(replay_file, source_dir) is None:
            os.remove(replay_file)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replays", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--danser-config", default="cordconv2d")
    p.add_argument("--output-dir", default="./data/source")
    p.add_argument("--source-dir", default="./data/source")
    p.add_argument("--danser-dir", default=r"D:\danser")
    p.add_argument("--max-replays", type=int, default=None)
    p.add_argument("--specific")
    args = p.parse_args()
    pipeline(args)

if __name__ == "__main__":
   main()
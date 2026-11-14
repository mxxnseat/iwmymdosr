from collections import deque
import math
import random
import shutil
import traceback
from os import PathLike
from pathlib import Path

import cv2
import librosa
import numpy as np

from iwmymdosr.dataset.datasets.preprocess import process_frame
from iwmymdosr.dataset.datasets.shard_writer import ShardWriter
from iwmymdosr.dataset.osu.map_parser import get_hit_objects, get_preempt, hit_windows
from iwmymdosr.dataset.osu.objects import HitObject, MapDifficulty, MapGeneral, Replay
from iwmymdosr.dataset.osu.replay_parser import load_replay
from iwmymdosr.dataset.osu.scrapper.unzip import lookup_specific_diff
from iwmymdosr.dataset.osu.utils import derive_transform, osu_to_px
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
from bisect import bisect_left
from intervaltree import IntervalTree

class ReplayShardBuilder:
    def __init__(
        self,
        source_dir: PathLike[str] | str,
        out_dir: PathLike[str] | str,
        frames_number: int,
        samples_per_shard: int = 128,
        val_frac: float = 0.1,
        seed: int = 42,
        max_replays: int | None = None,
        debug_output=None,
        specific=None
    ):
        self.source_dir = Path(source_dir)
        self.out_dir = Path(out_dir)
        self.frames_number = frames_number
        self.samples_per_shard = samples_per_shard
        self.val_frac = val_frac
        self.seed = seed
        self.max_replays = max_replays
        self.debug_output = debug_output
        self.specific = specific
        self.video_resolution = self.get_videos_resolution(self.source_dir)
        self.resize_video_resolution = (128, int(128 / (self.video_resolution[0]/self.video_resolution[1])))

    def build(self):
        W, H = self.resize_video_resolution
        mb = self.samples_per_shard * self.frames_number * W * H / 1e6
        print(f"~{mb:.0f} MB per shard ({self.samples_per_shard} samples, {W}x{H})")

        self._inputs = self._collect_inputs()
        files = sorted(self._inputs)
        random.Random(self.seed).shuffle(files)
        if self.max_replays:
            files = files[:self.max_replays]

        n_val = int(round(len(files) * self.val_frac))
        splits = {"val": files[:n_val], "train": files[n_val:]}

        for name, replay_files in splits.items():
            if not replay_files:
                continue
            meta = {"frames_number": self.frames_number, "resolution": [W, H]}
            writer = ShardWriter(self.out_dir / name, self.samples_per_shard, meta)
            for replay_file in replay_files:
                try:
                    self._process_replay(replay_file, writer)
                except Exception:
                    traceback.print_exc()
            writer.close()
            total = sum(writer.counts.values())
            print(f"{name}: {len(replay_files)} replays -> {total} samples, "
                  f"{len(writer.counts)} shards")

    def _collect_inputs(self):
        inputs: dict[Path, tuple[Path, Path, Path]] = {}
        skipped = 0
        for replay_file in sorted(self.source_dir.glob(f"{self.specific}.osr" if self.specific else "*.osr")):
            video_file = replay_file.with_suffix(".mp4")
            video_audio_file = replay_file.with_suffix(".mp3")
            reason = None
            osu = None

            if not replay_file.is_file():
                reason = "replay (.osr) missing"
            elif not video_file.is_file():
                reason = "video (.mp4) missing"
            else:
                try:
                    osu = lookup_specific_diff(replay_file, replay_file.parent)
                except Exception as e:
                    reason = f"beatmap lookup failed ({e})"
                else:
                    if osu is None or not Path(osu).is_file():
                        reason = "beatmap (.osu) missing"

            if reason:
                skipped += 1
                print(f"skip {replay_file.name}: {reason}")
            else:
                inputs[replay_file] = (video_file, video_audio_file, Path(osu)) # type: ignore

        print(f"{len(inputs)} usable replays, {skipped} skipped")
        return inputs

    def _process_replay(self, replay_file: Path, writer: ShardWriter):
        _, _, osu = self._inputs[replay_file]

        replay = load_replay(replay_file)
        diff, map_general, hit_objects = get_hit_objects(osu, replay)
        self._extract(replay_file, map_general, hit_objects, diff, writer)

    def show_debug(self, frames: deque, labels: np.ndarray):
        frames_len = len(frames)
        for i in range(frames_len):
            frame = cv2.cvtColor(frames[i].copy(), cv2.COLOR_GRAY2BGR)

            for label in labels:
                is_down = label[-1] == 0
                color = (0,0,255) if is_down else (0,255,0)
                cv2.circle(frame, (int(label[0]), int(label[1])), 2, color, -1)

            plt.subplot(1, frames_len, i+1)
            plt.imshow(frame, cmap="brg")

        plt.show()    


    def get_videos_resolution(self, source_dir: Path):
        video_path = next(source_dir.glob("*.mp4"), None)
        if video_path is None:
            raise FileNotFoundError(f"No .mp4 files in {source_dir}")
        cap = cv2.VideoCapture(str(video_path))
        try:
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (W, H)
        finally:
            cap.release()

    def object_time_to_frame(self, t, video_offset, frame_ms):
        return max(0, round((t + video_offset) / frame_ms))
    
    def frame_to_object_time(self, frame_idx, video_offset, frame_ms):
        return frame_idx * frame_ms - video_offset

    def get_danser_video_offset_by_audio(self, replay_file: Path, map_general: MapGeneral):
        _, video_audio_file, osu_file = self._inputs[replay_file]
        original_audio_file = osu_file.parent / map_general.audio_filename
        sr = 16000
        original_audio, _ = librosa.load(original_audio_file, sr=sr, mono=True)
        video_audio, _ = librosa.load(video_audio_file, sr=sr, mono=True)

        original_audio = stats.zscore(original_audio)
        video_audio = stats.zscore(video_audio)

        corr = signal.correlate(video_audio, original_audio, mode="full", method="fft")
        lags = signal.correlation_lags(video_audio.size, original_audio.size, mode="full")

        best = np.argmax(corr)
        lag_samples = lags[best]
        lag_ms = int(lag_samples*1000 / sr)

        confidence = corr[best] / (np.linalg.norm(original_audio) * np.linalg.norm(video_audio)) # type: ignore
        
        return lag_ms, confidence

    def get_hit_objects_hit_times(self, hit_objects: list[HitObject], video_offset: int, frame_ms, diff: MapDifficulty):
        result = IntervalTree()
        preempt = get_preempt(diff.ar)
        preempt_in_frames = self.object_time_to_frame(preempt, 0, frame_ms)

        for hit_object in hit_objects:
            end_time = hit_object.slider.end_time if hit_object.slider else hit_object.time
            start_frame = self.object_time_to_frame(hit_object.time, video_offset, frame_ms)
            end_frame = self.object_time_to_frame(end_time, video_offset, frame_ms)

            result.addi(start_frame-preempt_in_frames, end_frame, hit_object)

        return result

    def skip(self, capture: cv2.VideoCapture, amount):
        for _ in range(amount):
            if not capture.grab():
                break

    def _extract(self, replay_file: Path, map_general: MapGeneral, hit_objects: list[HitObject], diff: MapDifficulty, writer: ShardWriter):
        video_path, _, _ = self._inputs[replay_file]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
        
        try:
            VIDEO_OFFSET, video_offset_confidence = self.get_danser_video_offset_by_audio(replay_file, map_general)  # danser offset
            if video_offset_confidence < 0.5:
                print(RuntimeWarning(f"Confidence for {replay_file} is {video_offset_confidence}"))

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_ms = 1000 / fps
            sorted_objects = sorted(hit_objects, key=lambda o: o.time)

            resize_W, resize_H = self.resize_video_resolution
            output_transform = derive_transform(resize_W, resize_H)

            hit_object_by_frame = self.get_hit_objects_hit_times(sorted_objects, VIDEO_OFFSET, frame_ms, diff)
            frames = deque(maxlen=self.frames_number)

            skip_amount = self.object_time_to_frame(sorted_objects[0].time, VIDEO_OFFSET, frame_ms)

            self.skip(cap, skip_amount)
            frame_idx = skip_amount

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                
                grayscaled_frame, _ = process_frame(frame, self.resize_video_resolution)
                if not frames:
                    frames.extend([grayscaled_frame] * (self.frames_number - 1))
                frames.append(grayscaled_frame)
                t_ms = self.frame_to_object_time(frame_idx, VIDEO_OFFSET, frame_ms)
                visible_objects = hit_object_by_frame[frame_idx]
                labels = []
                for visible_object in visible_objects:
                    hit_object: HitObject = visible_object.data

                    visible_object_hit_frame = self.object_time_to_frame(hit_object.time, VIDEO_OFFSET, frame_ms)
                    frames_idx_diff = visible_object_hit_frame-frame_idx
                    x, y = osu_to_px(hit_object.x, hit_object.y, output_transform)
                    
                    if hit_object.slider:
                        if t_ms >= hit_object.time:
                            bx, by = hit_object.slider.ball_pos(t_ms)
                            x, y = osu_to_px(bx, by, output_transform)
                            frames_idx_diff = 0
                    labels.append([x,y, frames_idx_diff])

                labels = np.array(labels, dtype=np.float32)

                if self.debug_output:
                    self.show_debug(frames, labels)
                
                writer.add(np.stack(frames), labels)
                
                frame_idx += 1

        finally:
            cap.release()

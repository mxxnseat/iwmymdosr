import math
import random
import shutil
import traceback
from os import PathLike
from pathlib import Path

import cv2
import librosa
import numpy as np

from cordconv2d.dataset.datasets.preprocess import process_frame
from cordconv2d.dataset.datasets.shard_writer import ShardWriter
from cordconv2d.dataset.osu.map_parser import get_hit_objects, get_preempt
from cordconv2d.dataset.osu.objects import HitObject, MapDifficulty, MapGeneral, Replay
from cordconv2d.dataset.osu.replay_parser import load_replay
from cordconv2d.dataset.osu.scrapper.unzip import lookup_specific_diff
from cordconv2d.dataset.osu.utils import derive_transform, osu_to_px
import scipy.signal as signal
import scipy.stats as stats

class ReplayShardBuilder:
    def __init__(
        self,
        source_dir: PathLike[str] | str,
        out_dir: PathLike[str] | str,
        frames_n: int,
        samples_per_shard: int = 128,
        val_frac: float = 0.1,
        seed: int = 42,
        max_replays: int | None = None,
        debug_folder=None,
        specific=None
    ):
        self.source_dir = Path(source_dir)
        self.out_dir = Path(out_dir)
        self.frames_n = frames_n
        self.samples_per_shard = samples_per_shard
        self.val_frac = val_frac
        self.seed = seed
        self.max_replays = max_replays
        self.debug_folder = debug_folder
        self.specific = specific
        self.video_resolution = self.get_videos_resolution(self.source_dir)
        self.resize_video_resolution = (128, int(128 / (self.video_resolution[0]/self.video_resolution[1])))

    def build(self):
        if self.debug_folder:
            shutil.rmtree(self.debug_folder, ignore_errors=True)

        W, H = self.resize_video_resolution
        mb = self.samples_per_shard * self.frames_n * W * H / 1e6
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
            meta = {"frames_n": self.frames_n, "resolution": [W, H]}
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
        self._extract(replay_file, map_general, diff, hit_objects, writer)

    def cursor_at(self, data, step: float):
        ts = np.arange(data[0, 0], data[-1, 0] - step, step)

        x = np.interp(ts, data[:, 0], data[:, 1])
        y = np.interp(ts, data[:, 0], data[:, 2])
        idx = np.searchsorted(data[:, 0], ts, side="right") - 1
        key_down = data[np.clip(idx, 0, len(data) - 1), 3]

        return ts, x, y, key_down

    def get_debug_screenshots_from_video(self, out_dir, video_path, is_down, frames, screen_px, frame_no):
        out_dir = Path(out_dir)
        (out_dir / video_path.name).mkdir(exist_ok=True, parents=True)
        color = (0, 0, 255) if is_down else (255, 0, 0)

        grayscale, resized, orig = frames
        rec_width = 5
        rec_height = 5
        tr = derive_transform(orig.shape[1], orig.shape[0])
        screen_px = osu_to_px(screen_px[0], screen_px[1], tr)

        def draw_rect(frame, suffix = ""):
            cv2.rectangle(
                frame,
                (int(screen_px[0] - rec_width), int(screen_px[1] - rec_height)),
                (int(screen_px[0] + rec_width), int(screen_px[1] + rec_height)),
                color,
                -1,
            )
            cv2.imwrite(str(out_dir / video_path.name / f"{frame_no}{suffix}.jpg"), frame)

        draw_rect(grayscale, "_resized_gray")
        draw_rect(resized, "_resized")
        draw_rect(orig, "_orig")
        cv2.imshow("", orig)
        cv2.waitKey(0)

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

    def dervive_max_step(self, replay_files: list[Path]):
        cords = []
        # filter replays by video existence
        for replay_file in replay_files:
            try:
                if not Path(f"{replay_file.parent}/{replay_file.stem}.mp4").exists():
                    continue
                replay = load_replay(replay_file)
                cords.append(replay.replay_data[:, 1:3])
            except Exception as e:
                print(e)

        cords = np.concatenate(cords, axis=0)
        transform = derive_transform(self.resize_video_resolution[0], self.resize_video_resolution[1])
        cords = [osu_to_px(x, y, transform) for x, y in cords]
        diff = np.diff(cords, axis=0)
        steps = np.linalg.norm(diff, axis=1)

        return np.percentile(steps, 99.5)
    
    def lookup_cursor_position_in_frame(self, frame, template):
        result = cv2.matchTemplate(
            frame,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, confidence, _, location = cv2.minMaxLoc(result)

        h, w = template.shape[:2]

        center = (
            location[0] + w // 2,
            location[1] + h // 2
        )

        return center, confidence

    def get_cursor_positions(self, video_path, seek_pos, n = 20):
        temp = cv2.imread("./cursor-temp.png")
        cap = cv2.VideoCapture(f"{video_path.parent}/{video_path.stem}_no_objects.mp4")
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
        
        for _ in range(int(seek_pos)):
            cap.grab()

        try:
            positions = []
            W, H = self.video_resolution
            for _ in range(n):
                _, frame = cap.read()
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                pos, _ = self.lookup_cursor_position_in_frame(frame, temp)
                positions.append(pos)

            return np.array(positions)
        finally:
            cap.release()

    def find_best_match_index_in_replay(self, replay, cursor_positions, n = 20, window = 200):
        best_i = -1
        m = float("inf")
        for i in range(window):
            data = replay[i:i+n]
            dists = np.linalg.norm(data - cursor_positions, axis=1)
            s = dists.sum()
            if s < m:
                m = s
                best_i = i
        return best_i

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

    def _extract(self, replay_file: Path, map_general: MapGeneral, difficulty: MapDifficulty,
                 hit_objects: list[HitObject], writer: ShardWriter):
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
            sorted_objects = sorted_objects

            next_pos = 0
            last_idx = -1
            last_frame = None

            for i, hit_object in enumerate(sorted_objects, 1):
                target = max(0, round((hit_object.time + VIDEO_OFFSET) / frame_ms))

                if target == last_idx:
                    frame = last_frame
                else:
                    for _ in range(target - next_pos):
                        if not cap.grab():
                            return
                    ok, frame = cap.read()
                    if not ok:
                        break
                    next_pos = target + 1
                    last_idx, last_frame = target, frame

                scr = np.array(
                    osu_to_px(hit_object.x, hit_object.y, output_transform),
                    dtype=np.float32,
                )

                grayscaled_frame, resized_frame = process_frame(frame, self.resize_video_resolution)
                data = np.array([grayscaled_frame])
                target = np.array([scr[0], scr[1], float(True)], dtype=np.float32)

                writer.add(data, target)
                if self.debug_folder:
                    frame_no = i
                    assert frame is not None
                    self.get_debug_screenshots_from_video(
                        self.debug_folder, video_path, True, (grayscaled_frame.copy(), resized_frame.copy(), frame.copy()),
                        (hit_object.x, hit_object.y), frame_no,
                    )
        finally:
            cap.release()

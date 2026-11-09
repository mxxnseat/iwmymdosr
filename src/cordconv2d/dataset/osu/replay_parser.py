from osrparse import Replay, Key, ReplayEventOsu, Mod
import cordconv2d.dataset.osu.objects as osu_objects
import numpy as np

def is_key_pressed(keys: Key) -> bool:
    return (keys & (Key.K1 | Key.K2)) != 0

def load_replay(path):
    r = Replay.from_path(path)
    data: list[ReplayEventOsu] = r.replay_data # type: ignore
    t = 0
    rows: list[tuple[int, float, float, bool]] = []
    for e in data:
        t += e.time_delta

        if e.time_delta < 0:
            continue

        if e.y < -100:
            continue
        
        rows.append((t, e.x, e.y, is_key_pressed(e.keys)))

    replay_data = np.array(rows, dtype=float)
    _, unique_idx = np.unique(replay_data[:, 0], return_index=True)
    replay_data = replay_data[unique_idx]
    return osu_objects.Replay(replay_data, r.mods)

import hashlib
from pathlib import Path
from osrparse import Replay
import zipfile

def unzip(osu_file: Path, output_dir: Path):
    """
        unzip .osu file
    """
    with zipfile.ZipFile(osu_file, "r") as z:
        z.extractall(output_dir)

def compute_md5_hash(file: Path):
    return hashlib.md5(Path(file).read_bytes()).hexdigest()

def lookup_specific_diff(replay_path: Path, osu_song_dir: Path):
    """
        find an appropriate map difficulty from replay
    """
    replay = Replay.from_path(replay_path)
    for song in osu_song_dir.rglob("*.osu"):
        if compute_md5_hash(song) == replay.beatmap_hash:
            return song
    return None
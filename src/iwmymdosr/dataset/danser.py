import json
from os import PathLike
from pathlib import Path

def common_cfg(danser_dir: Path, output_dir: Path, songs_dir: PathLike[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = danser_dir / "settings" / "default.json"
    cfg = json.loads(base.read_text(encoding="utf-8"))
    cfg["Recording"]["OutputDir"] = str(output_dir)
    cfg["General"]["OsuSongsDir"] = str(songs_dir)
    cfg["Graphics"]["Width"] = 1280
    cfg["Graphics"]["Height"] = 720
    cfg["Recording"]["FrameWidth"] = 1280
    cfg["Recording"]["FrameHeight"] = 720
    cfg["Recording"]["FPS"] = 30

    return cfg

def make_profile_no_objects(danser_dir: PathLike[str], output_dir: PathLike[str], songs_dir: PathLike[str], profile: str):
    danser_dir = Path(danser_dir)
    output_dir = Path(output_dir)
    cfg = common_cfg(danser_dir, output_dir, songs_dir)
    cfg["Playfield"]["DrawObjects"] = False
    cfg["Playfield"]["DrawCursors"] = True
    cfg["Objects"]["DrawFollowPoints"] = False


    (danser_dir / "settings" / f"{profile}.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

def make_profile(danser_dir: PathLike[str], output_dir: PathLike[str], songs_dir: PathLike[str], profile: str):
    danser_dir = Path(danser_dir)
    output_dir = Path(output_dir)
    cfg = common_cfg(danser_dir, output_dir, songs_dir)


    (danser_dir / "settings" / f"{profile}.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )

def get_download_replay_cmd(replay_path: str, output: str, profile = "local.denser") -> list[str]:
    return ["danser-cli", f"-replay={replay_path}", "-record", f"-out={output}", f"-settings={profile}"]

from pathlib import Path
from osrparse import Mod
from iwmymdosr.dataset.osu.objects import HitObject, MapDifficulty, MapGeneral, Replay, Transform

def apply_stacking(hit_objects: list[HitObject], difficulty: MapDifficulty) -> list[HitObject]:
    cs = difficulty.cs
    stack_leniency = difficulty.leniency

    thr = get_preempt(difficulty.ar) * stack_leniency
    stack = [0] * len(hit_objects)

    for i in range(len(hit_objects) - 1, -1, -1):
        if hit_objects[i].is_spinner or stack[i] != 0:
            continue
        cur = i
        for j in range(i - 1, -1, -1):
            if hit_objects[j].is_spinner:
                continue
            if hit_objects[cur].time - hit_objects[j].time > thr:
                break
            if abs(hit_objects[j].x - hit_objects[cur].x) < 3 and abs(hit_objects[j].y - hit_objects[cur].y) < 3:
                stack[j] = stack[cur] + 1
                cur = j

    scale = (1 - 0.7 * (cs - 5) / 5) / 2
    result: list[HitObject] = []
    
    for hit_object, stacking in zip(hit_objects, stack):
        transformed_hit_object = HitObject(
            round(hit_object.x - stacking * scale * 6.4),
            round(hit_object.y - stacking * scale * 6.4),
            hit_object.radius,
            hit_object.time,
            hit_object.type,
            hit_object.is_circle,
            hit_object.is_slider,
            hit_object.is_spinner
        )
        result.append(transformed_hit_object)

    return result

def get_ar(ar_base: float, replay: Replay | None):
    if replay and replay.mods & Mod.HardRock:
        return min(10, ar_base*1.4)
    return ar_base
    
def get_preempt(ar: float) -> float:
    if ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    return 1200 - 750 * (ar - 5) / 5

def transform_object(hit_object: HitObject, t: Transform) -> HitObject:
    return HitObject(
            round(t.s * hit_object.x + t.ox),
            round(t.s * hit_object.y + t.oy),
            hit_object.radius*t.s,
            hit_object.time,
            hit_object.type,
            hit_object.is_circle,
            hit_object.is_slider,
            hit_object.is_slider
    )

def get_circle_radius(cs: float):
    return (54.4 - 4.48 * cs) * 1.00041

def parse_type(t: int):
    """
        The hit object's type parameter is an 8-bit integer where each bit is a flag with special meaning.
        0	Marks the object as a hit circle
        1	Marks the object as a slider
        2	Marks the start of a new combo
        3	Marks the object as a spinner
        4, 5, 6	A 3-bit integer specifying how many combo colours to skip, a practice referred to as "colour hax". Only relevant if the object starts a new combo.
        7	Marks the object as an osu!mania hold note. (does not present)
    """
    return {
        "circle":    bool(t & 1),    # bit 0
        "slider":    bool(t & 2),    # bit 1
        "new_combo": bool(t & 4),    # bit 2
        "spinner":   bool(t & 8),    # bit 3
        "color_skip": (t >> 4) & 0b111,  # bits 4-6
    }

def remap_raw_map_difficulty(difficulty: dict) -> MapDifficulty:
    return MapDifficulty(
        cs=float(difficulty["CircleSize"]),
        ar=float(difficulty["ApproachRate"]),
        leniency=float(difficulty["StackLeniency"]),
        od=float(difficulty["OverallDifficulty"])
    )

def remap_raw_map_general(general: dict) -> MapGeneral:
    return MapGeneral(
        audio_filename=general["AudioFilename"],
    )

def read_metadata(osu_path: Path) -> dict[str, str]:
    meta = {}
    in_section = False
    with open(osu_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line == "[Metadata]":
                in_section = True
            elif line.startswith("[") and in_section:
                break
            elif in_section and ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip().lower()] = value
    return meta

def apply_hardrock(hit_object: HitObject, replay: Replay):
    if not replay.mods & Mod.HardRock:
        return hit_object
    hit_object.y = 384 - hit_object.y
    return hit_object

def get_hit_objects(filepath: Path, replay: Replay) -> tuple[MapDifficulty, MapGeneral, list[HitObject]]:
    difficulty = {}
    general = {}
    objects: list[HitObject] = []
    section = None

    print(filepath)
    with open(filepath, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section == "Difficulty" or section == "General":
                key, _, value = line.partition(":")
                difficulty[key.strip()] = value
                general[key.strip()] = value.strip()

            elif section == "HitObjects":
                parameter = line.split(",")
                x, y, time, typ = int(parameter[0]), int(parameter[1]), int(parameter[2]), int(parameter[3])

                types = parse_type(typ)

                is_circle = types["circle"]
                is_slider = types["slider"]
                is_spinner = types["spinner"]

                objects.append(
                    apply_hardrock(
                        HitObject(
                            x=x, 
                            y=y,
                            radius=0, 
                            time=time, 
                            type=typ, 
                            is_circle=is_circle, 
                            is_slider=is_slider, 
                            is_spinner=is_spinner,
                    ), replay)
                )

    remapped_difficulty = remap_raw_map_difficulty(difficulty)
    remapped_general = remap_raw_map_general(general)
    remapped_difficulty.ar = get_ar(remapped_difficulty.ar, replay)
    for hit_object in objects:
        hit_object.radius = get_circle_radius(remapped_difficulty.cs)
    objects = apply_stacking(objects, remapped_difficulty)

    return remapped_difficulty, remapped_general, objects
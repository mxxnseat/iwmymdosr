from pathlib import Path
from osrparse import Mod
import dataclasses
from iwmymdosr.dataset.osu.objects import HitObject, MapDifficulty, MapGeneral, Replay, SliderProperties, Transform
from iwmymdosr.dataset.osu.slider import Slider
from bisect import bisect_right

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
        transformed = dataclasses.replace(hit_object, 
            x=round(hit_object.x - stacking * scale * 6.4),
            y=round(hit_object.y - stacking * scale * 6.4)
        )
        if transformed.slider:
            transformed.slider.set_stack_offset(transformed.x, transformed.y)
        result.append(transformed)

    return result

def get_ar(ar_base: float, replay: Replay | None):
    if replay and replay.mods & Mod.HardRock:
        return min(10, ar_base*1.4)
    return ar_base

def get_od(od_base: float, replay: Replay | None):
    if replay and replay.mods & Mod.HardRock:
        return min(10, od_base*1.4)
    return od_base
    
def get_preempt(ar: float) -> float:
    if ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    return 1200 - 750 * (ar - 5) / 5

def transform_object(hit_object: HitObject, t: Transform) -> HitObject:
    return dataclasses.replace(
        hit_object,
        x=round(t.s * hit_object.x + t.ox),
        y=round(t.s * hit_object.y + t.oy),
        radius=hit_object.radius * t.s,
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
        od=float(difficulty["OverallDifficulty"]),
        slider_multiplier=float(difficulty["SliderMultiplier"])
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

def apply_hardrock(cords: tuple[int, int], replay: Replay):
    x, y = cords
    if not replay.mods & Mod.HardRock:
        return x, y
    return x, 384 - y

def hit_windows(od)->tuple[float,float,float]:
    return (80 - 6 * od, 140 - 8 * od, 200 - 10 * od)

def build_curve_points(points: str, replay: Replay):
    result = []
    splited_points = list(filter(lambda x: x, points.split("|")))

    for point in splited_points:
        x, y = point.split(":")
        result.append(apply_hardrock((int(x), int(y)), replay))

    return result

def build_timing_table(timing):
    """timing: list of (time, beatLength, uninherited), sorted by time."""
    times, states = [], []
    beat_length = next(bl for t, bl, u in timing if u)   # fallback for early sliders
    sv = 1.0
    for t, bl, uninh in timing:
        if uninh:
            beat_length = bl
            sv = 1.0                                     # red line resets SV
        else:
            sv = max(0.1, min(10.0, -100.0 / bl))
        times.append(t)
        states.append((beat_length, sv))                 # state in effect from t onward
    return times, states

def timing_at(table, T):
    times, states = table
    i = bisect_right(times, T) - 1
    if i < 0:                                            # before the first point
        return states[0][0], 1.0
    return states[i]

def get_hit_objects(filepath: Path, replay: Replay) -> tuple[MapDifficulty, MapGeneral, list[HitObject]]:
    difficulty = {}
    general = {}
    timing_points = []
    objects: list[HitObject] = []
    sliders_properties: list[SliderProperties | None] = []
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

            elif section == "TimingPoints":
                p = line.split(',')
                t, bl = float(p[0]), float(p[1])
                uninherited = int(p[6]) == 1 if len(p) > 6 else bl > 0
                timing_points.append((t, bl, uninherited))

            elif section == "HitObjects":
                parameter = line.split(",")
                x, y, time, typ = int(parameter[0]), int(parameter[1]), int(parameter[2]), int(parameter[3])
                x, y = apply_hardrock((x,y), replay)
                types = parse_type(typ)

                is_circle = types["circle"]
                is_slider = types["slider"]
                is_spinner = types["spinner"]

                if is_slider:
                    slides, length = int(parameter[6]), float(parameter[7])
                    curve_type, *raw = parameter[5].split("|")
                    pts = [(x, y)] + build_curve_points("|".join(raw), replay)
                    sliders_properties.append(
                        SliderProperties(
                            curve_type,  # type: ignore
                            pts, 
                            slides, 
                            length
                        )
                    )
                else:
                    sliders_properties.append(None)

                objects.append(
                    HitObject(
                        x=x, 
                        y=y,
                        radius=0, 
                        time=time, 
                        type=typ, 
                        is_circle=is_circle, 
                        is_slider=is_slider, 
                        is_spinner=is_spinner,
                        slider=None
                    )
                )

    timing_table = build_timing_table(timing_points)
    remapped_difficulty = remap_raw_map_difficulty(difficulty)
    remapped_general = remap_raw_map_general(general)
    remapped_difficulty.ar = get_ar(remapped_difficulty.ar, replay)
    remapped_difficulty.od = get_od(remapped_difficulty.od, replay)
    
    for i, hit_object in enumerate(objects):
        hit_object.radius = get_circle_radius(remapped_difficulty.cs)
        if hit_object.is_slider:
            slider_properties = sliders_properties[i]
            beat_length, sv = timing_at(timing_table, hit_object.time)
            assert slider_properties is not None
            slider = Slider(
                start_time=hit_object.time, 
                curve_type=slider_properties.curve_type, 
                pts=slider_properties.curve_points, 
                slides=slider_properties.slides, 
                length=slider_properties.length,
                slider_multiplier=remapped_difficulty.slider_multiplier,
                beat_length=beat_length,
                sv_multiplier=sv
            )
            hit_object.slider = slider
                    

    objects = apply_stacking(objects, remapped_difficulty)

    return remapped_difficulty, remapped_general, objects
import argparse
import asyncio
import contextlib
import functools
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Optional, cast
import aiohttp
import anyio
from osrparse import Replay
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from dotenv import load_dotenv
from osrparse import Mod
import iwmymdosr.dataset.osu.map_parser as map_parser
from iwmymdosr.dataset.osu.scrapper.monkey_patches import apply_monkey_patches
import iwmymdosr.dataset.osu.scrapper.unzip as scrapper_unzip
from osu import AsynchronousAuthHandler, AsynchronousClient, Beatmap, BeatmapsetSearchSort, GameModeInt, GameModeStr, LazerMod, LegacyScore, Mods, RankStatus, RequestException, Scope, BeatmapsetSearchFilter
from iwmymdosr.dataset.osu.scrapper.network import RateLimiter, retry, network_policy, io_policy, any_of
from iwmymdosr.utils.main import log
from iwmymdosr.dataset.osu.scrapper.watchdog import watchdog

MIRRORS = [
    lambda beatmap: f"https://osu.direct/api/d/{beatmap.beatmapset_id}"
]

API_LIMITER: RateLimiter
MIRROR_LIMITER: RateLimiter

pending_scores: dict[int, list[int]] = {}   # beatmapset_id -> score ids waiting for its download
downloaded_beatmapsets: set[int] = set()
dest_locks: dict[Path, LockEntry] = {}
claimed_scores: set[int] = set()   # score ids that are committed or in flight in this run

class LockEntry:
    __slots__ = ("lock", "users")

    def __init__(self):
        self.lock = anyio.Lock()
        self.users = 0

@contextlib.asynccontextmanager
async def dest_lock(dest: Path):
    entry = dest_locks.get(dest)
    if entry is None:
        entry = dest_locks[dest] = LockEntry()
    entry.users += 1
    try:
        async with entry.lock:
            yield
    finally:
        entry.users -= 1
        if entry.users == 0:
            del dest_locks[dest]

@retry(network_policy, max_attempts=10, base_delay=5, limiter=lambda: API_LIMITER)
async def get_replay_data_by_id_only(client: AsynchronousClient, score_id: int):
    await API_LIMITER.wait()
    return await client.get_replay_data_by_id_only(score_id, use_osrparse=False)

@retry(network_policy, max_attempts=10, base_delay=5, limiter=lambda: API_LIMITER)
async def get_beatmap_scores(client: AsynchronousClient, beatmap_id: int):
    await API_LIMITER.wait()
    return await client.get_beatmap_scores(beatmap_id, legacy_only=True)

@retry(network_policy, max_attempts=10, base_delay=5, limiter=lambda: API_LIMITER)
async def search_beatmapsets(client: AsynchronousClient, filt: BeatmapsetSearchFilter, pagination_cursor: Optional[int]):
    await API_LIMITER.wait()
    return await client.search_beatmapsets(filt, pagination_cursor)

async def download_scores_worker(
    receive_stream: MemoryObjectReceiveStream,
    progress_send: MemoryObjectSendStream,
    download_map_send: MemoryObjectSendStream,
    client: AsynchronousClient,
    args,
):
    async with receive_stream, progress_send, download_map_send:
        async for beatmap in receive_stream:
            await download_scores(
                client, beatmap, download_map_send, progress_send, args
            )

async def download_beatmaps_worker(download_map_recieve: MemoryObjectReceiveStream, progress_send: MemoryObjectSendStream):
    session = create_aiohttp_session()

    async with session, progress_send:
        async with download_map_recieve:
            async for data in download_map_recieve:
                with anyio.CancelScope(shield=True):
                    set_id = data["beatmap"].beatmapset_id
                    ok = await download_beatmapset(data["beatmap"], session, data["args"])
                    score_ids = pending_scores.pop(set_id, [])
                    if ok:
                        downloaded_beatmapsets.add(set_id)
                        with contextlib.suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                            for score_id in score_ids:
                                await progress_send.send({
                                    "type": "score_processed",
                                    "payload": {"score_id": score_id},
                                })
                    else:
                        claimed_scores.difference_update(score_ids)

def save_json(path: Path, state: dict) -> None:
    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f)

    os.replace(temp_path, path)

async def state_manager(receive_stream: MemoryObjectReceiveStream, args, cancel_scope: anyio.CancelScope):
    save_every = int(args.save_every)
    state_path = Path(args.temp_dir) / "state.json"

    state = {
        "target": args.replays_target,
        "processed_scores": [],
    }

    if state_path.exists():
        with state_path.open(encoding="utf-8") as f:
            stored_json = json.load(f)

        if stored_json["target"] != state["target"]:
            raise RuntimeError(
                "State target does not match requested target"
            )

        state.update(stored_json)

    dirty = False

    async def save():
        await anyio.to_thread.run_sync(
            save_json, state_path, state
        )

    try:
        async with receive_stream:
            async for event in receive_stream:
                event_type = event["type"]
                payload = event["payload"]

                if event_type == "score_processed":
                    score_id = payload["score_id"]
                    if score_id not in state["processed_scores"]:
                        state["processed_scores"].append(score_id)
                        dirty = True


                    count = len(state["processed_scores"])
                    log(state_manager, f"{count}/{state["target"]}")

                    if count >= state["target"]:
                        await save()
                        dirty = False

                        log(state_manager, "Replay target reached")
                        cancel_scope.cancel()
                        return

                    if count and count % save_every == 0:
                        await save()
                        dirty = False
                        log(
                            state_manager,
                            f"Saved: {count}/{state['target']}",
                        )

                dirty = True

    except anyio.get_cancelled_exc_class():
        with anyio.CancelScope(shield=True):
            if dirty:
                await save()
        raise
    except Exception:
        log(state_manager, "Critical state-manager failure")
        raise
    else:
        if dirty:
            await save()

@retry(io_policy, max_attempts=5, base_delay=0.05)
async def save_replay(client: AsynchronousClient, beatmap: Beatmap, score: LegacyScore, args):
    replay_data: bytes = cast(bytes, await get_replay_data_by_id_only(client, score.id))
    replay = Replay.from_string(replay_data)
    replay_filename = Path(args.output_dir) / f"{replay.beatmap_hash}_{beatmap.beatmapset_id}_{score.id}.osr"

    with open(replay_filename, "wb") as replay:
        replay.write(replay_data)

def is_slop_replay(score: LegacyScore):
    # FL is not a slop but it is among all other slop mods, hehe
    skip_mods = {"HD", "FL", "DT", "NC"}
    return any(m.mod.value in skip_mods for m in score.mods)

@retry(any_of(io_policy, network_policy), max_attempts=3, base_delay=2, limiter=lambda:MIRROR_LIMITER)
async def download_beatmap_from_mirror(mirror: str, session: aiohttp.ClientSession, dest: Path, chunk_size = 64 * 1024):
    await MIRROR_LIMITER.wait()
    tmp_file = dest.with_suffix(dest.suffix + ".part")
    try:
        async with session.get(mirror) as response:
            response.raise_for_status()
            downloaded = 0
            with open(tmp_file, "wb") as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
            tmp_file.replace(dest)
    except Exception as e:
        tmp_file.unlink(missing_ok=True)
        log(download_beatmap_from_mirror, f"Download failed: {e}")
        raise

def safe_name(name: str, repl: str = "") -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', repl, name)

    name = re.sub(r" {2,}", " ", name)

    return name.rstrip(" .")

def get_osu_song_directory(osu_file: Path, beatmap: Beatmap, args):
    meta = map_parser.read_metadata(osu_file)
    return Path(args.output_dir) / safe_name(f"{beatmap.beatmapset_id} {meta["artist"]} - {meta["title"]}")

@retry(io_policy, max_attempts=5, base_delay=0.05)
def save_songs(osu_zip: Path, beatmap: Beatmap, args):
    preprocess_dir = Path(args.temp_dir) / str(beatmap.beatmapset_id)
    try:
        scrapper_unzip.unzip(osu_zip, preprocess_dir)
        osu = next(preprocess_dir.rglob("*.osu"))
        dest = get_osu_song_directory(osu, beatmap, args)
        shutil.copytree(preprocess_dir, dest)
    except Exception as ex:
        print(ex)
    finally:
        osu_zip.unlink(True)
        shutil.rmtree(preprocess_dir, ignore_errors=True)

@retry(network_policy, max_attempts=5, base_delay=2)
async def download_beatmapset(beatmap: Beatmap, session: aiohttp.ClientSession, args):
    # We download file to temp directory because we need to unzip it before adding it to the source directory
    # .osu file is in fact a .zip file
    beatmapset_tmp_filename = Path(args.temp_dir) / f"{beatmap.beatmapset_id}.zip"

    async with dest_lock(beatmapset_tmp_filename):
        if beatmapset_tmp_filename.exists():
            return True

        errors: list[Exception] = []
        success = False

        for mirror in MIRRORS:
            try:
                await download_beatmap_from_mirror(mirror(beatmap), session, beatmapset_tmp_filename)
                save_songs(beatmapset_tmp_filename, beatmap, args)
                success = True
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                errors.append(e)
        if not success:
            log(download_beatmapset, f"All mirrors failed: {errors!r}")
        return success

async def download_scores(
    client: AsynchronousClient, beatmap: Beatmap, download_map_send: MemoryObjectSendStream,  progress_send: MemoryObjectSendStream, args
):
    try:
        stored_json = {
            "processed_scores": []
        }
        state_path = Path(args.temp_dir) / "state.json"
        if state_path.exists():
            with state_path.open(encoding="utf-8") as f:
                stored_json = json.load(f)

        top_scores = await get_beatmap_scores(client, beatmap.id)
        count = 0
        for score in top_scores.scores:
            score = cast(LegacyScore, score)
            if is_slop_replay(score):
                continue
            if count == args.replays_per_map:
                break
            if score.id in stored_json["processed_scores"] or score.id in claimed_scores:
                log(download_scores, f"{score.id} already processed. skipping...")
                continue

            # No await between this check and the add, so workers can't both take the last slot
            if len(claimed_scores | set(stored_json["processed_scores"])) >= args.replays_target:
                break
            claimed_scores.add(score.id)

            try:
                await save_replay(client, beatmap, score, args)
            except BaseException:
                claimed_scores.discard(score.id)
                raise

            set_id = beatmap.beatmapset_id
            if set_id in downloaded_beatmapsets:
                await progress_send.send({
                    "type": "score_processed",
                    "payload": {"score_id": score.id},
                })
            elif set_id in pending_scores:
                pending_scores[set_id].append(score.id)
            else:
                pending_scores[set_id] = [score.id]
                await download_map_send.send({"beatmap": beatmap, "args": args})
            count += 1
                
    except TypeError as e:
        if "BeatmapUserScore.__init__" not in str(e):
            raise
        log(download_scores, f"Skipping beatmap {beatmap.beatmapset_id}: osu.py parsing bug")
    except anyio.get_cancelled_exc_class():
        raise
    except Exception:
        log(download_scores, f"Critical failure on beatmap {beatmap.beatmapset_id}")
        raise

def form_star_query_filter(min_sr: float, max_sr: float) -> str:
    return f"star>={min_sr} star<={max_sr}"

def is_has_leaderboard(rank: RankStatus) -> bool:
    return rank == RankStatus.RANKED or rank == RankStatus.LOVED

def is_map_inside_sr_range(map_sr: float, min_sr: float, max_sr: float):
    return min_sr <= map_sr <= max_sr

def create_aiohttp_session() -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=30)
    return aiohttp.ClientSession(timeout=timeout)

def get_state_path(args):
    return Path(args.temp_dir) / "state.json"

async def download_maps(client: AsynchronousClient, job_send: MemoryObjectSendStream, args):
    async with job_send:
        log(download_maps, "Start process")
        
        state_path = get_state_path(args)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        if state_path.exists():
            with state_path.open(encoding="utf-8") as f:
                stored_json = json.load(f)

            if len(stored_json["processed_scores"]) == args.replays_target:
                raise RuntimeError(
                    f"We are already done. Check {state_path}"
                )

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        seen_beatmaps: set[int] = set()
        min_sr = int(args.min_stars)
        max_sr = int(args.max_stars)

        filt = BeatmapsetSearchFilter() \
            .set_mode(GameModeInt.STANDARD) \
            .set_query(form_star_query_filter(min_sr, max_sr)) \

        pagination_cursor: int | None = None
        new_in_pass = 0
        pass_number = 1

        while True:
            beatmapset_result = await search_beatmapsets(client, filt, pagination_cursor)
            next_cursor = beatmapset_result.cursor["id"] if beatmapset_result.cursor else None
            
            for beatmapset in beatmapset_result.beatmapsets:
                if beatmapset.beatmaps is None:
                    continue
                for beatmap in beatmapset.beatmaps:
                    if beatmap.id in seen_beatmaps:
                        continue
                    if not is_has_leaderboard(beatmap.ranked):
                        continue
                    if not is_map_inside_sr_range(beatmap.difficulty_rating, min_sr, max_sr):
                        continue

                    seen_beatmaps.add(beatmap.id)
                    new_in_pass += 1
                    await job_send.send(beatmap)
            if next_cursor is None:
                log(download_maps, f"End of results, restarting (found {new_in_pass} new)")
                pass_number += 1
                new_in_pass = 0
                pagination_cursor = None
                continue

            if next_cursor == pagination_cursor:
                log(download_maps, f"Cursor repeated ({next_cursor}), stopping")
                break

            pagination_cursor = next_cursor

async def run_download_pipeline(client, args):
    global API_LIMITER, MIRROR_LIMITER
    API_LIMITER = RateLimiter(args.api_rps)
    MIRROR_LIMITER = RateLimiter(args.mirror_rps)
    job_send, job_receive = anyio.create_memory_object_stream(10)
    download_beatmap_send, download_beatmap_receive = anyio.create_memory_object_stream(30)
    progress_send, progress_receive = anyio.create_memory_object_stream(60)

    worker_progress = [
        progress_send.clone() for _ in range(3)
    ]
    download_beatmap_streams = [
        download_beatmap_send.clone() for _ in range(3)
    ]
    beatmap_progress = [
        progress_send.clone() for _ in range(3)
    ]

    async with anyio.create_task_group() as outer_tg:
        outer_tg.start_soon(watchdog, {"jobs": job_send, "maps": download_beatmap_send, "progress": progress_send}, 60)

        async with anyio.create_task_group() as tg:
            tg.start_soon(state_manager, progress_receive, args, tg.cancel_scope)
            for i in range(3):
                tg.start_soon(
                    download_beatmaps_worker,
                    download_beatmap_receive.clone(),
                    beatmap_progress[i]
                )
                tg.start_soon(
                    download_scores_worker,
                    job_receive.clone(),
                    worker_progress[i],
                    download_beatmap_streams[i],
                    client,
                    args,
                )
                

            await job_receive.aclose()
            await download_beatmap_receive.aclose()
            await download_beatmap_send.aclose()
            await progress_send.aclose()

            tg.start_soon(
                download_maps,
                client,
                job_send,
                args,
            )

        outer_tg.cancel_scope.cancel() 

async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-stars", type=float, default=5.5)
    p.add_argument("--max-stars", type=float, default=7.5)
    p.add_argument("--api-rps", type=float, default=1)
    p.add_argument("--mirror-rps", type=float, default=1)
    p.add_argument("--replays-per-map", type=int, default=3)
    p.add_argument("--replays-target", type=int, default=1000, help="top replays per set (uses its hardest in-range diff)")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--output-dir", default="osu_data")
    p.add_argument("--temp-dir", default="./tmp")
    p.add_argument("--callback", default="http://localhost:7272")
    args = p.parse_args()

    cid, secret = os.environ.get("OSU_CLIENT_ID"), os.environ.get("OSU_CLIENT_SECRET")
    
    if not cid or not secret:
        raise RuntimeError("Provide client_id and client_secret for osu api")
    if not cid.isnumeric():
        raise RuntimeError("Client id for osu client must be numeric")

    apply_monkey_patches()

    auth = AsynchronousAuthHandler(int(cid), secret, args.callback, Scope.identify())
    print(auth.get_auth_url())
    await auth.get_auth_token(input("Code: "))
    client = AsynchronousClient(auth)

    user = await client.get_own_data(GameModeStr.STANDARD)
    print(user.username)

    await run_download_pipeline(client, args)

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
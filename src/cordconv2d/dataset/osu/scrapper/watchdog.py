import asyncio
import anyio

from cordconv2d.utils.main import log

def where(t: asyncio.Task) -> str:
    c = t.get_coro()
    chain = []
    while c is not None:
        frame = (getattr(c, "cr_frame", None)
                 or getattr(c, "gi_frame", None)
                 or getattr(c, "ag_frame", None))
        if frame is None:
            break
        chain.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
        c = getattr(c, "cr_await", None) or getattr(c, "gi_yieldfrom", None)
    return " > ".join(chain)

async def watchdog(streams: dict, interval=30):
    while True:
        await anyio.sleep(interval)
        for name, s in streams.items():
            st = s.statistics()
            log(watchdog, f"{name}: buf={st.current_buffer_used}/{st.max_buffer_size} "
                          f"wait_send={st.tasks_waiting_send} wait_recv={st.tasks_waiting_receive} "
                          f"open_send={getattr(st, 'open_send_streams', '-')} "
                          f"open_recv={getattr(st, 'open_receive_streams', '-')}")
        for t in asyncio.all_tasks():
            if not t.done():
                if not t.done():
                    log(watchdog, f"{t.get_name()} -> {where(t)}")

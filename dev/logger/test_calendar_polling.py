# test_calendar_polling.py

import os
import asyncio
import aiohttp
from ics import Calendar as IcsCalendar
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional

POLL_INTERVAL = 30  # seconds between polls

async def poll_forever(ics_url: str, tz_name: Optional[str] = None):
    """
    tz_name: optional, e.g. "America/Los_Angeles"; if None, local system tz
    """
    if tz_name:
        tz = ZoneInfo(tz_name)
    else:
        tz = datetime.now().astimezone().tzinfo or timezone.utc

    last_map: dict[str, dict] = {}  # uid -> { "begin_local": str, "description": str }

    while True:
        now_local = datetime.now(tz)
        print(f"[{now_local.isoformat(sep=' ', timespec='seconds')}] Polling…")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(ics_url) as resp:
                    resp.raise_for_status()
                    body = await resp.text()
        except Exception as e:
            print("Fetch error:", e)
            await asyncio.sleep(POLL_INTERVAL)
            continue

        try:
            cal = IcsCalendar(body)
        except Exception as e:
            print("Parse error:", e)
            await asyncio.sleep(POLL_INTERVAL)
            continue

        new_map: dict[str, dict] = {}
        for ev in cal.events:
            uid = ev.uid
            # compute local begin
            if ev.begin:
                begin_dt = ev.begin.datetime
                begin_local = begin_dt.astimezone(tz)
                begin_str = begin_local.isoformat()
            else:
                begin_str = None

            new_map[uid] = {
                "begin_local": begin_str,
                "description": ev.description or "",
            }

        # additions
        for uid, info in new_map.items():
            if uid not in last_map:
                print("+++ New event:", uid, "begin:", info.get("begin_local"))
                if info.get("description"):
                    print("    desc:", info["description"])

        # removals
        for uid in list(last_map.keys()):
            if uid not in new_map:
                print("--- Event removed:", uid)

        # modifications
        for uid in new_map.keys() & last_map.keys():
            old = last_map[uid]
            new = new_map[uid]
            diffs = {}
            if new.get("begin_local") != old.get("begin_local"):
                diffs["begin"] = (old.get("begin_local"), new.get("begin_local"))
            if new.get("description") != old.get("description"):
                diffs["description"] = (old.get("description"), new.get("description"))
            if diffs:
                print("*** Event modified:", uid, "diffs:", diffs)

        last_map = new_map

        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    ics_url = os.getenv("CALENDAR_ICS")
    if not ics_url:
        raise RuntimeError("Please set CALENDAR_ICS env var first.")
    # optionally pass your timezone name
    asyncio.run(poll_forever(ics_url, tz_name=None))
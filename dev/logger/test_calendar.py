import os
import asyncio
import aiohttp
from ics import Calendar as IcsCalendar
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
load_dotenv()

def human_delta(seconds: float) -> str:
    neg = seconds < 0
    s = int(abs(seconds))
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    return f"{days}d {hours}h {minutes}m{' ago' if neg else ''}"

async def test_ics_fetch(ics_url: str, n: int = 5) -> None:
    print(f"🔍 Testing ICS URL: {ics_url}")

    async with aiohttp.ClientSession() as session:
        async with session.get(ics_url) as resp:
            resp.raise_for_status()
            body = await resp.text()

    cal = IcsCalendar(body)
    events = list(cal.events)
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime.now(tz)
    week_ago = now - timedelta(weeks=1)

    print(f"✅ Parsed {len(events)} events. Showing up to {n}:")
    count = 0

    for ev in events:
        if count >= n or not ev.begin:
            break

        # Always interpret ICS times as local wall clock times
        start = ev.begin.datetime.replace(tzinfo=tz)
        end = ev.end.datetime.replace(tzinfo=tz) if ev.end else None

        # Ignore very old events
        if start < week_ago:
            continue

        delta = start - now
        delta_str = human_delta(delta.total_seconds())

        print("----------")
        print(f"Title       : {ev.name or '<no title>'}")
        print(f"When        : {start} → {end}")
        print(f"Location    : {ev.location or '<no location>'}")
        if ev.description:
            print("Description :")
            for line in ev.description.splitlines():
                print("  " + line)
        print(f"Now         : {now}")
        print(f"Time delta  : {delta_str}")
        count += 1

    print("----------")

if __name__ == "__main__":
    ics_url = os.getenv("CALENDAR_ICS")
    if not ics_url:
        raise RuntimeError("Set CALENDAR_ICS or hardcode your ICS URL.")
    asyncio.run(test_ics_fetch(ics_url))
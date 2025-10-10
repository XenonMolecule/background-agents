# test_calendar_polling_diagnostics.py
"""
Diagnostic polling tester for the Calendar observer.
Runs manual poll cycles (no background Observer task),
so that we can debug whether polling stops or fails silently.
"""

import os
import asyncio
import traceback
from datetime import datetime
from cal import Calendar
from dotenv import load_dotenv

load_dotenv()


async def debug_poll(observer: Calendar, interval: int = 60, cycles: int = 5):
    """Run multiple manual poll cycles and show detailed debug output."""
    print(f"\n[Diagnostics] Starting manual polling test ({cycles} cycles, {interval}s interval)")
    for i in range(1, cycles + 1):
        print(f"\n========== CYCLE {i} / {cycles} ==========")
        print(f"[{datetime.now(observer.local_tz).strftime('%H:%M:%S')}] Polling once...")

        try:
            changed = await observer._poll_once()
            print(f"[Diagnostics] Poll result: changed={changed}")
        except Exception as e:
            print(f"[Diagnostics] ERROR during poll cycle {i}: {type(e).__name__} - {e}")
            traceback.print_exc()

        # Drain the update queue (if any)
        while not observer.update_queue.empty():
            update = await observer.update_queue.get()
            print("\n--- Update emitted ---")
            print(update.content)
            print("----------------------\n")

        print(f"[{datetime.now(observer.local_tz).strftime('%H:%M:%S')}] Sleeping {interval}s before next poll...")
        await asyncio.sleep(interval)

    print("\n[Diagnostics] Completed all cycles.")


async def main():
    ics_url = os.getenv("CALENDAR_ICS")
    if not ics_url:
        raise RuntimeError("Set CALENDAR_ICS before running diagnostics.")

    # Instantiate Calendar observer manually (no background thread)
    cal = Calendar(ics_url=ics_url, debug=True)
    await debug_poll(cal, interval=5, cycles=5)


if __name__ == "__main__":
    asyncio.run(main())
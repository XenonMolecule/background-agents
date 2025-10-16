import asyncio
import csv
import json
import os
from pathlib import Path
from gum import gum
from gum.observers import Screen
from cal import Calendar
from dotenv import load_dotenv
from objective_inducer import ObjectiveInducer
import dspy

load_dotenv()

user_name = "Michael Ryan"
model = "gpt-4o-mini-2024-07-18"
max_batch_size = 15

dspy_model = dspy.LM('openai/gpt-4o-mini-2024-07-18')
dspy.configure(lm=dspy_model)

# CSV logging setup
CSV_PATH = os.path.join(os.path.dirname(__file__), "objectives_log.csv")
_csv_lock = asyncio.Lock()

CSV_COLUMNS = [
    "reasoning",
    "objectives",
    "rationale",
    "agent",
    "start_listener",
    "report_listener",
]

def _serialize_for_csv(value):
    """Serialize complex values (pydantic models, lists, dicts) to JSON strings."""
    try:
        # Pydantic v2
        if hasattr(value, "model_dump"):
            return json.dumps(value.model_dump(), ensure_ascii=False)
        # Pydantic v1
        if hasattr(value, "dict"):
            return json.dumps(value.dict(), ensure_ascii=False)
        # Lists of models or primitives
        if isinstance(value, list):
            def _to_plain(v):
                if hasattr(v, "model_dump"):
                    return v.model_dump()
                if hasattr(v, "dict"):
                    return v.dict()
                return v
            return json.dumps([_to_plain(v) for v in value], ensure_ascii=False)
        # Dicts
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    except Exception:
        # Fallback to string if anything goes wrong
        pass
    return str(value)

async def _append_objectives_csv_row(row_data: dict):
    """Append a single row to the objectives CSV in a concurrency-safe way."""
    # Normalize/serialize values
    row = {col: _serialize_for_csv(row_data.get(col, "")) for col in CSV_COLUMNS}
    # Ensure directory exists
    Path(os.path.dirname(CSV_PATH)).mkdir(parents=True, exist_ok=True)
    async with _csv_lock:
        write_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

async def main():
    cal = Calendar()

    async def log_observations(observer, update):
        print("=== Logging observations ===")
        print(observer, update)
        print("=== === === === === === ===")
        print(cal.query_str())
        print("=== === === === === === ===")

    async with gum(
        user_name, 
        model, 
        # Screen(model),
        cal,
        max_batch_size=max_batch_size
    ) as gum_instance:

        objective_inducer = ObjectiveInducer(gum_instance, cal)

        async def log_objectives(observer, update):
            print("=== Logging objectives ===")
            print(observer, update)
            print("=== === === === === === ===")
            rationale, agent, start_listener, report_listener, reasoning, objectives = await objective_inducer.select_agent_and_triggers(context=update.content, goal_limit=3)
            # print("Reasoning")
            # print(reasoning)
            # print("Objectives")
            # print(objectives)
            # print("Rationale")
            # print(rationale)
            # print("Agent")
            # print(agent)
            # print("Start Listener")
            # print(start_listener)
            # print("Report Listener")
            # print(report_listener)
            # print("=== === === === === === ===")

            # Append to CSV with concurrency protection
            await _append_objectives_csv_row({
                "reasoning": reasoning,
                "objectives": objectives,
                "rationale": rationale,
                "agent": agent,
                "start_listener": start_listener,
                "report_listener": report_listener,
            })

        gum_instance.register_update_handler(log_objectives)
        await asyncio.Future()  # run forever (Ctrl-C to stop)

if __name__ == "__main__":
    asyncio.run(main())
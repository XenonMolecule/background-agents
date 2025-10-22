import asyncio
from datetime import timedelta, datetime
import os
import subprocess
from gum import gum
from gum.observers import Screen, Calendar
from dotenv import load_dotenv
from objective_inducer import ObjectiveInducer
import dspy
from pynput.keyboard import Listener, Key, KeyCode

load_dotenv()

user_name = "Michael Ryan"
model = "gpt-4o-mini-2024-07-18"
max_batch_size = 15

dspy_model = dspy.LM('openai/gpt-4o-mini-2024-07-18')
dspy.configure(lm=dspy_model)

CSV_PATH = os.path.join(os.path.dirname(__file__), "context_log.csv")
SCREENSHOT_DIRECTORY = os.path.join(os.path.dirname(__file__), "screenshots")

async def main():
    cal = Calendar()

    last_objective_induction_time = datetime.now()
    last_survey_time = datetime.now()

    objective_induction_interval = timedelta(minutes=3)
    survey_interval = timedelta(minutes=30)

    # Helper: launch the SwiftUI survey app (expects prebuilt binary)
    def _launch_survey_app():
        pkg_dir = os.path.join(os.path.dirname(__file__), "swiftui-survey")
        exe_path = os.path.join(pkg_dir, ".build", "release", "SurveyApp")
        if not os.path.exists(exe_path):
            raise RuntimeError(
                "SurveyApp binary not found. Build it first: (cd dev/survey/swiftui-survey && swift build -c release)"
            )
        subprocess.Popen([exe_path], cwd=pkg_dir)

    # Global hotkey: Cmd + Shift + \\ (rarely used by system defaults)
    # Sets last times far in the past to naturally trigger induction and survey on next update
    def _start_hotkey_listener():
        nonlocal last_objective_induction_time, last_survey_time

        pressed = set()
        trigger_key = KeyCode.from_char('\\')

        def on_press(key):
            nonlocal last_objective_induction_time, last_survey_time
            pressed.add(key)
            if Key.cmd in pressed and Key.shift in pressed and trigger_key in pressed:
                last_objective_induction_time = datetime.now() - timedelta(days=3650)
                last_survey_time = datetime.now() - timedelta(days=3650)

        def on_release(key):
            try:
                pressed.discard(key)
            except Exception:
                pass

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    _start_hotkey_listener()

    async with gum(
        user_name, 
        model, 
        Screen(model),
        cal,
        max_batch_size=max_batch_size
    ) as gum_instance:

        objective_inducer = ObjectiveInducer(gum_instance, cal)

        async def log_objectives(observer, update):
            nonlocal last_objective_induction_time, last_survey_time
            now = datetime.now()
            if now - last_objective_induction_time < objective_induction_interval:
                return
            last_objective_induction_time = now

            goals, reasoning = await objective_inducer.induce_and_log(context=update.content, limit=3, csv_path=CSV_PATH, screenshot_directory=SCREENSHOT_DIRECTORY)

            if now - last_survey_time < survey_interval:
                return
            last_survey_time = now

            # Launch the SwiftUI survey app
            _launch_survey_app()


        gum_instance.register_update_handler(log_objectives)
        await asyncio.Future()  # run forever (Ctrl-C to stop)

if __name__ == "__main__":
    asyncio.run(main())
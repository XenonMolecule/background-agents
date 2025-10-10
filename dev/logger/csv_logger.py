import asyncio
from gum import gum
from gum.observers import Screen
from cal import Calendar
from dotenv import load_dotenv

load_dotenv()

user_name = "Michael Ryan"
model = "gpt-4o-mini-2024-07-18"
max_batch_size = 15

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
        gum_instance.register_update_handler(log_observations)
        await asyncio.Future()  # run forever (Ctrl-C to stop)

if __name__ == "__main__":
    asyncio.run(main())
from dotenv import load_dotenv
import os

load_dotenv()  # or load_dotenv(dotenv_path='path_to_your_env')

print("TELEGRAM_API_ID =", os.getenv("TELEGRAM_API_ID"))
print("TELEGRAM_API_HASH =", os.getenv("TELEGRAM_API_HASH"))

"""Step 2 smoke test: can we talk to the LLM?

Run with:  uv run python hello_llm.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# 1. Read OPENAI_API_KEY and OPENAI_MODEL from the .env file into the environment.
load_dotenv()

model = os.getenv("OPENAI_MODEL")
if not model:
    raise SystemExit("OPENAI_MODEL is missing from your .env file.")

# 2. Create the client. It picks up OPENAI_API_KEY from the environment automatically,
#    so the key never appears in our code.
client = OpenAI()

# 3. Send one question and print the answer.
messages = [
    {
        "role": "user",
        "content": "In one sentence: what type is Torchic, and what does it evolve into?",
    }
]
response = client.chat.completions.create(
    model=model,
    messages=messages,
)

print(f"Model: {model}")
print(f"Answer: {response.choices[0].message.content}")

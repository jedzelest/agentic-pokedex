"""Step 2 smoke test: can we talk to the LLM?

Run with:  uv run python hello_llm.py
"""

import os
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_developer_message_param import ChatCompletionDeveloperMessageParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam

# 1. Read OPENAI_API_KEY and OPENAI_MODEL from the .env file into the environment.
load_dotenv()

model = os.getenv("OPENAI_MODEL")
if not model:
    raise SystemExit("OPENAI_MODEL is missing from your .env file.")

# 2. Create the client. It picks up OPENAI_API_KEY from the environment automatically,
#    so the key never appears in our code.
client = OpenAI()

# 3. Send one question and print the answer.
messages: Iterable[ChatCompletionDeveloperMessageParam | ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam | ChatCompletionToolMessageParam | ChatCompletionFunctionMessageParam] = [
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

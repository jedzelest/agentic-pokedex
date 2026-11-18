"""The agent loop: the LLM decides which tools to call, we run them, repeat until it answers."""

import json

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from agentic_pokedex.tools import ToolRegistry

SYSTEM_PROMPT = """\
You are an agentic Pokédex: an expert assistant for Pokémon questions, backed by PokéAPI.

Scope:
- Only answer questions about Pokémon game data: species, moves, abilities, types, evolutions.
- For anything else (other anime, general chat, writing tasks), don't answer and don't call tools.
  Reply in one sentence that you're a Pokédex, and suggest a Pokémon question instead.
- For Pokémon topics your tools can't verify (anime plot, trivia, card games),
  say your data only covers game data.

How to work:
- Look facts up with your tools instead of relying on memory.
- Prefer the specialized tools (get_pokemon_species, get_ability). Use pokeapi_lookup for
  everything else, e.g. resource "pokemon" for types, base stats and abilities, "move" for
  moves, "type" for type matchups, "evolution-chain" for full evolution lines.
- You may call several tools, one after another, to answer a single question.
- If a lookup returns an error, check the spelling and try again (PokéAPI uses lowercase,
  hyphenated names like "mr-mime" or "speed-boost").
- If the data isn't available, say so rather than guessing.
- Answer clearly and concisely.
"""

MAX_ROUNDS = 8  # safety limit: max LLM round-trips for one question


class PokedexAgent:
    def __init__(self, model: str, registry: ToolRegistry, client: OpenAI | None = None,
                 verbose: bool = True):
        self.model = model
        self.registry = registry
        self.client = client or OpenAI()  # reads OPENAI_API_KEY from the environment
        self.verbose = verbose  # print each tool call so you can watch the agent think
        self.messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def ask(self, question: str) -> str:
        """Answer one question. The history is kept, so follow-up questions work."""
        self.messages.append({"role": "user", "content": question})

        for _ in range(MAX_ROUNDS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.registry.openai_schemas(),
            )
            message = response.choices[0].message

            # No tool calls means the model has its final answer.
            if not message.tool_calls:
                answer = message.content or ""
                self.messages.append({"role": "assistant", "content": answer})
                return answer

            # The model wants tools. First record its request in the history...
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                    if call.type == "function"
                ],
            })

            # ...then run each tool and send back one "tool" message per call.
            for call in message.tool_calls:
                if call.type != "function":  # we only define function tools
                    continue
                self._log(call.function.name, call.function.arguments)
                result = self.registry.execute(call.function.name, call.function.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
            # Loop: the model now reads the tool results and decides what to do next.

        return "Sorry, I couldn't finish looking that up. Try asking a more specific question."

    def reset(self) -> None:
        """Forget the conversation (keeps the system prompt)."""
        del self.messages[1:]

    def _log(self, name: str, arguments: str) -> None:
        if not self.verbose:
            return
        try:
            args = ", ".join(f"{k}={v!r}" for k, v in json.loads(arguments).items())
        except (json.JSONDecodeError, AttributeError):
            args = arguments
        print(f"  → {name}({args})")

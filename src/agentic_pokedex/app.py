"""Web chat UI for the Pokédex, built with Gradio.

Run with:  uv run pokedex-web     (then open http://127.0.0.1:7860)
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
import openai
from openai.types.chat import ChatCompletionMessageParam

from agentic_pokedex.agent import FinalAnswer, PokedexAgent, ToolStep, format_call
from agentic_pokedex.config import Settings
from agentic_pokedex.pokeapi import PokeAPIClient
from agentic_pokedex.tools import ToolRegistry

MAX_RESULT_PREVIEW = 1500  # characters of tool output shown in the UI
CSS_PATH = Path(__file__).parent / "assets" / "pokedex.css"

EXAMPLES = [
    "What does the ability Speed Boost do, and which starter Pokémon can have it?",
    "Who is faster, Pikachu or Jolteon?",
    "What is Eevee's full evolution line?",
    "Which types are super effective against Dragon?",
]


def _text(content: Any) -> str:
    """Gradio message content may be a plain string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_text(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return ""


def _to_openai_history(history: list[dict]) -> list[ChatCompletionMessageParam]:
    """Convert the chat shown in the browser back into OpenAI messages.

    Only the user questions and final answers are kept. The tool-call boxes are skipped
    (they have a metadata title); the agent can simply look things up again if needed.
    """
    messages: list[ChatCompletionMessageParam] = []
    for message in history:
        if (message.get("metadata") or {}).get("title"):
            continue
        text = _text(message.get("content"))
        if message.get("role") == "user":
            messages.append({"role": "user", "content": text})
        elif message.get("role") == "assistant":
            messages.append({"role": "assistant", "content": text})
    return messages


def _tool_box(step: ToolStep) -> gr.ChatMessage:
    """A collapsible box in the chat showing one tool call and what it returned."""
    try:
        preview = json.dumps(json.loads(step.result), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        preview = step.result
    if len(preview) > MAX_RESULT_PREVIEW:
        preview = preview[:MAX_RESULT_PREVIEW] + "\n... (shortened)"
    return gr.ChatMessage(
        role="assistant",
        content=f"```json\n{preview}\n```",
        metadata={"title": f"🔍 {format_call(step.name, step.arguments)}", "status": "done"},
    )


def build_demo() -> gr.Blocks:
    settings = Settings.from_env()
    registry = ToolRegistry.default(PokeAPIClient(settings.pokeapi_base_url))

    def respond(message: str, history: list[dict]) -> Iterator[list[gr.ChatMessage]]:
        # A fresh agent per question, seeded with the conversation from the browser.
        # This keeps each browser tab's conversation separate.
        agent = PokedexAgent(
            model=settings.openai_model,
            registry=registry,
            verbose=False,
            history=_to_openai_history(history),
        )
        shown: list[gr.ChatMessage] = [
            gr.ChatMessage(role="assistant", content="",
                           metadata={"title": "🤔 Thinking...", "status": "pending"})
        ]
        yield shown
        try:
            for event in agent.run(message):
                if isinstance(event, ToolStep):
                    # Insert the tool box before the "Thinking..." indicator.
                    shown.insert(len(shown) - 1, _tool_box(event))
                    yield shown
                elif isinstance(event, FinalAnswer):
                    shown[-1] = gr.ChatMessage(role="assistant", content=event.text)
                    yield shown
        except openai.APIError as exc:
            shown[-1] = gr.ChatMessage(role="assistant", content=f"⚠️ OpenAI error: {exc}")
            yield shown

    # gr.Blocks lets us put our own styled header above the chat.
    with gr.Blocks(title="Agentic Pokédex", fill_height=True) as demo:
        gr.HTML(
            f"""
            <div id="pokedex-header">
              <span class="pokeball"></span>
              <div>
                <h1>Agentic Pokédex</h1>
                <p>Ask anything about Pokémon. Answers are looked up live from PokéAPI
                   (model: {settings.openai_model}). Expand the 🔍 boxes to see each lookup.</p>
              </div>
            </div>
            """
        )
        gr.ChatInterface(
            fn=respond,
            examples=EXAMPLES,
            cache_examples=False,  # don't call the LLM at startup
        )
    return demo


def main() -> None:
    # In Gradio 6, custom CSS is passed to launch() rather than to gr.Blocks().
    build_demo().launch(css_paths=[CSS_PATH])


if __name__ == "__main__":
    main()

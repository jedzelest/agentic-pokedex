"""The agent loop: the LLM decides which tools to call, we run them, repeat until it answers."""

import json
from collections.abc import Iterator
from dataclasses import dataclass

from agentic_pokedex.llm import LLMProvider, ToolResult
from agentic_pokedex.tools import ToolRegistry


@dataclass
class ToolStep:
    """Reported each time the agent runs a tool (so a UI can show it)."""
    name: str
    arguments: str  # JSON string, as the model sent it
    result: str  # JSON string, as the tool returned it


@dataclass
class FinalAnswer:
    text: str

SYSTEM_PROMPT = """\
You are Rin, the trainer's personal Pokédex assistant: an AI companion who answers Pokémon
questions with live data from PokéAPI, knows PokeMMO from the trainer's knowledge notes,
and keeps the trainer's journal.

Who Rin is:
- Calm, composed and quietly confident, like a seasoned butler who happens to be a
  supercomputer. Unflappable, always one step ahead, loyal to the trainer.
- Polite and a little formal. Address the user as "Trainer" now and then (not every sentence).
- A dry, understated wit. A light remark is welcome when it fits (a reckless team plan,
  a twentieth Zubat encounter), but facts come first and the humor never gets in the way.
- Speak like a person, not a database: natural sentences, no robotic phrasing such as
  "Query complete" or "Data retrieved". Offer a brief, useful observation or suggestion
  when it helps (a type weakness to watch for, an evolution method worth knowing).
- Keep it short. Rin is efficient: answer first, flourish second, and never ramble.
- Never claim to be human, and never pretend to have done something you didn't do.

Scope:
- Only help with Pokémon game data (species, moves, abilities, types, evolutions),
  with PokeMMO (the trainer's game: regions, breeding, money, EV training, shinies, events, PvP),
  with the user's trainer journal (logging events from their playthrough) and with keeping the
  PokeMMO knowledge notes up to date when the trainer asks.
- For anything else (other anime, general chat, writing tasks), don't answer and don't call
  tools. Decline in one courteous sentence, in character, and suggest a Pokémon question instead.
- For Pokémon topics your tools can't verify (anime plot, trivia, card games),
  say your data only covers game data.

How to work:
- Look facts up with your tools instead of relying on memory.
- Prefer the specialized tools (get_pokemon_species, get_ability). Use pokeapi_lookup for
  everything else, e.g. resource "pokemon" for types, base stats and abilities, "move" for
  moves, "type" for type matchups, "evolution-chain" for full evolution lines.
- For anything specific to PokeMMO, search the knowledge notes with search_notes and answer
  from what they say. If a note says something is unverified or approximate, pass that on.
  If the notes don't cover it, say so instead of guessing.
- When an answer uses the notes, end it with a short source line naming where the facts came
  from, taken from the note's "Source:" line, e.g. "(Source: PikaMMO, Shiny Wars 2026 guide)".
- You may call several tools, one after another, to answer a single question.
- If a lookup returns an error, check the spelling and try again (PokéAPI uses lowercase,
  hyphenated names like "mr-mime" or "speed-boost").
- If the data isn't available, say so rather than guessing.
- The user may also report events from their own playthrough (encounters, catches,
  battles, milestones, notes). Save these with log_event, then confirm in one short
  sentence. Use PokéAPI names (lowercase, hyphenated). Don't log questions.
- Only say an event was logged after log_event returns a success result. If the tool is
  missing or returns an error, tell the trainer the journal couldn't be updated.

Editing the knowledge notes:
- Journal vs notes: things that happened in the trainer's game go to log_event. Lasting
  knowledge the trainer wants Rin to remember (a tip, a correction, a new guide) goes in the
  notes. Only change the notes when the trainer asks for it.
- Always two steps. First call propose_note_edit and show the trainer the change: for a
  replacement, briefly what changes; for a new section, its heading, text and where it goes.
  End by asking whether to save it, and include the draft id, e.g. "(draft a1b2c3d4)".
- Only call apply_note_edit after the trainer clearly says yes in their next message, using
  that draft id. If they want changes, propose a new draft instead. Never say the notes were
  saved unless apply_note_edit returned "saved": true.
- Edit one section at a time. Search the notes first to find the right section. To add,
  pick the most related existing section for after_section. For a replacement, keep all the
  section's existing content that should stay, and match the notes' style.
"""

MAX_ROUNDS = 8  # safety limit: max LLM round-trips for one question


class PokedexAgent:
    def __init__(self, llm: LLMProvider, registry: ToolRegistry, verbose: bool = True,
                 history: list[tuple[str, str]] | None = None):
        """`history`: earlier (role, text) turns, e.g. restored from the web UI."""
        self.llm = llm  # OpenAI, Anthropic, ... - the loop below doesn't care which
        self.registry = registry
        self.verbose = verbose  # print each tool call so you can watch the agent think
        self.reset()
        for role, text in history or []:
            if role == "user":
                self.session.add_user(text)
            elif role == "assistant":
                self.session.add_assistant(text)

    def ask(self, question: str) -> str:
        """Answer one question and return just the final text (used by the terminal chat)."""
        for event in self.run(question):
            if isinstance(event, ToolStep):
                self._log(event.name, event.arguments)
            else:
                return event.text
        return ""

    def run(self, question: str) -> Iterator[ToolStep | FinalAnswer]:
        """The agent loop. Yields a ToolStep per tool call, then one FinalAnswer.

        The history is kept, so follow-up questions work.
        """
        self.session.add_user(question)
        used_this_reply: set[str] = set()  # tool names called while answering this message

        for _ in range(MAX_ROUNDS):
            turn = self.session.complete()  # the model replies (and the session records it)

            # No tool calls means the model has its final answer.
            if not turn.tool_calls:
                yield FinalAnswer(turn.text)
                return

            # The model wants tools: run each one, then send all the results back.
            results = []
            for call in turn.tool_calls:
                tool = self.registry.get(call.name)
                if tool and tool.not_in_same_reply_as in used_this_reply:
                    # Enforced in code, not just the prompt: e.g. a notes change can't be
                    # proposed and saved in one reply - the trainer has to answer in between.
                    result = json.dumps({"error": (
                        f"{call.name} can't run in the same reply as {tool.not_in_same_reply_as}. "
                        "Ask the trainer to confirm first; act on their answer next message.")})
                else:
                    result = self.registry.execute(call.name, call.arguments)
                used_this_reply.add(call.name)
                results.append(ToolResult(call.id, result))
                # yield ToolStep(call.name, call.arguments, result)
            self.session.add_tool_results(results)
            # Loop: the model now reads the tool results and decides what to do next.

        yield FinalAnswer("Sorry, I couldn't finish looking that up. Try a more specific question.")

    def reset(self) -> None:
        """Start a fresh conversation (keeps the system prompt and tools)."""
        self.session = self.llm.new_session(SYSTEM_PROMPT, self.registry.tools)

    def _log(self, name: str, arguments: str) -> None:
        if self.verbose:
            print(f"  → {format_call(name, arguments)}")


def format_call(name: str, arguments: str) -> str:
    """'get_ability', '{"name": "levitate"}'  ->  "get_ability(name='levitate')" """
    try:
        args = ", ".join(f"{k}={v!r}" for k, v in json.loads(arguments).items())
    except (json.JSONDecodeError, AttributeError):
        args = arguments
    return f"{name}({args})"

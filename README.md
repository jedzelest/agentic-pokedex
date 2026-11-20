# Agentic Pokédex: Rin

Rin is an AI assistant for Pokémon and PokeMMO, and my first project with agentic AI.
She answers questions with live data from [PokéAPI](https://pokeapi.co), searches my own
PokeMMO knowledge notes, keeps a journal of my playthrough, and can update the notes
herself, but only after I confirm.

- **Live app:** https://agentic-pokedex.onrender.com (behind a login)
- **Stack:** Python 3.12, uv, OpenAI (Anthropic-ready), Supabase (Postgres + pgvector +
  Storage), Gradio, Render

This README is written for **me, reviewing this project later**. It explains what each
part does, how the parts connect, and why things were built this way.

---

## Contents

1. [What makes it "agentic"](#1-what-makes-it-agentic)
2. [The big picture](#2-the-big-picture)
3. [Folder and file guide](#3-folder-and-file-guide)
4. [Following one question through the code](#4-following-one-question-through-the-code)
5. [Supabase: what lives where](#5-supabase-what-lives-where)
6. [Running it locally](#6-running-it-locally)
7. [Configuration (environment variables)](#7-configuration-environment-variables)
8. [Deploying and day-to-day tasks](#8-deploying-and-day-to-day-tasks)
9. [Design decisions worth remembering](#9-design-decisions-worth-remembering)
10. [Glossary](#10-glossary)
11. [Known limitations and ideas](#11-known-limitations-and-ideas)
12. [Project timeline](#12-project-timeline)

---

## 1. What makes it "agentic"

A plain chatbot answers from memory, in one step. An **agent** can **decide to use tools**,
look at the results, and decide again, looping until it can answer.

```
            ┌───────────────────────────────────────────────┐
            ▼                                               │
 question ─► LLM thinks ─► wants a tool? ── yes ─► run the tool(s) ─► send results back
                                 │
                                 no
                                 ▼
                           final answer
```

Example: *"Who is faster, Pikachu or Jolteon?"*. The model calls
`pokeapi_lookup(pokemon, pikachu)`, then `pokeapi_lookup(pokemon, jolteon)`, reads both
results, and answers *"Jolteon (130) is faster than Pikachu (90)."* I never told it which
lookups to make; **the model chose them.** That's the core idea.

The loop lives in [`agent.py`](src/agentic_pokedex/agent.py) (`PokedexAgent.run`), and it's
only about 30 lines. Everything else in the project is either a **tool** the model can call,
or **plumbing** around the loop: model APIs, the database, and the web UI.

---

## 2. The big picture

```
                       ┌──────────────────────────────┐
  Browser (Gradio) ───►│ app.py        web chat UI    │
  Terminal         ───►│ __init__.py   terminal chat  │
                       └──────────────┬───────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │ agent.py   PokedexAgent      │  ← the agent loop + Rin's
                       │            SYSTEM_PROMPT     │    personality and rules
                       └──────┬────────────────┬──────┘
                 talks to     │                │   runs tools through
                              ▼                ▼
          ┌──────────────────────────┐  ┌─────────────────────────────────┐
          │ llm/  (model providers)  │  │ tools/registry.py  ToolRegistry  │
          │  base.py      interface  │  └──────────────┬──────────────────┘
          │  openai_provider.py      │                 │
          │  anthropic_provider.py   │   ┌─────────────┼──────────────┬──────────────────┐
          └──────────────────────────┘   ▼             ▼              ▼                  ▼
                                    PokéAPI tools   log_event    search_notes     propose/apply
                                    species,ability,  (journal)    (RAG search)     _note_edit
                                    pokeapi_lookup        │              │                │
                                         │                ▼              ▼                ▼
                                         ▼             db/ ──► Supabase   rag/ ──► Supabase
                                    pokeapi/client.py      log_event table    doc_chunks + Storage
                                         │
                                         ▼
                                    pokeapi.co
```

**The rule that keeps this organized:** each layer only talks to the one below it.
- `agent.py` never imports OpenAI or Anthropic directly; it talks to `llm/`.
- Tools never call each other; the agent loop calls them through the registry.
- Only `db/` creates the Supabase connection, and only `pokeapi/` makes HTTP calls to PokéAPI.

---

## 3. Folder and file guide

```
agentic-pokedex/
├── src/agentic_pokedex/        ← all application code (one Python package)
│   ├── __init__.py             ← terminal chat (`uv run agentic-pokedex`)
│   ├── app.py                  ← web chat (`uv run pokedex-web`)
│   ├── agent.py                ← THE AGENT LOOP + system prompt
│   ├── config.py               ← settings from .env
│   ├── assets/                 ← CSS theme for the web UI
│   ├── llm/                    ← talking to the language model (OpenAI / Anthropic)
│   ├── tools/                  ← everything the model can DO
│   ├── pokeapi/                ← HTTP client for pokeapi.co
│   ├── db/                     ← Supabase connection + journal table access
│   └── rag/                    ← knowledge notes: chunk, embed, index, search, edit
├── hello_llm.py                ← the very first experiment (one LLM call)
├── pyproject.toml / uv.lock    ← dependencies (managed by uv)
├── requirements.txt            ← same dependencies, exported for Render
├── render.yaml                 ← Render deployment blueprint
├── .env.example                ← template for secrets/settings (.env itself is git-ignored)
├── .gitignore, .python-version, .vscode/settings.json
└── README.md
```

### How the code is grouped

The folders follow **one job per folder**:

| Folder | Answers the question… | Depends on |
|---|---|---|
| top-level `agent.py` | *How does Rin think and loop?* | `llm/`, `tools/` |
| `llm/` | *How do we talk to a model API?* | nothing project-specific (just `tools/base.py` for tool definitions) |
| `tools/` | *What can Rin do?* | `pokeapi/`, `db/`, `rag/` |
| `pokeapi/` | *How do we fetch Pokémon data?* | `requests` |
| `db/` | *How do we reach Supabase, and write journal rows?* | `supabase` |
| `rag/` | *How do the knowledge notes become searchable and editable?* | `db/`, OpenAI embeddings |
| `app.py`, `__init__.py` | *How does a human talk to Rin?* | everything above |

---

### Entry points and core

#### [`src/agentic_pokedex/agent.py`](src/agentic_pokedex/agent.py): the heart of the project
- **`SYSTEM_PROMPT`**: Rin's instructions. Who she is (calm, JARVIS-like, calls me
  "Trainer"), what's in scope, how to use each tool, and the rules for the journal and for
  editing notes. Changing Rin's behaviour usually means editing this text, not code.
- **`PokedexAgent`**: holds one conversation (`session`) and the tools (`registry`).
  - `run(question)`: **the agent loop**. It sends the conversation to the model. If the reply
    has tool calls, it runs them, sends back the results, and loops. If not, that's the final
    answer. It's a *generator*: it `yield`s a `ToolStep` per tool call (so a UI can show it)
    and finally a `FinalAnswer`.
  - Safety limits:
    - `MAX_ROUNDS = 8` caps how many model calls one question can make.
    - The **same-reply guard** refuses a tool if its `not_in_same_reply_as` tool already ran
      while answering the same message. This stops Rin from proposing and saving a notes edit
      without me answering in between.
  - `ask(question)`: a simpler wrapper for the terminal; it returns just the text.
  - `history=`: earlier turns as `(role, text)` pairs, used by the web app.
- **`ToolStep` / `FinalAnswer`**: the events `run()` yields.
- **`format_call()`**: turns `get_ability`, `{"name": "levitate"}` into
  `get_ability(name='levitate')` for display.
- ⚠️ The line `# yield ToolStep(...)` is **commented out**, which hides the 🔍 tool boxes in
  the web UI and the `→` lines in the terminal. Remove the `#` to see which tools Rin uses.
  That's very useful when debugging.

#### [`src/agentic_pokedex/app.py`](src/agentic_pokedex/app.py): the web UI (Gradio)
- `build_demo()` builds the page: the red Poké Ball header (`gr.HTML`) and the chat
  (`gr.ChatInterface`), plus example questions.
- `respond()` runs for every message. It creates a **fresh `PokedexAgent`** seeded with that
  browser tab's visible chat history (`_to_history`), so different tabs and users never share
  a conversation. It streams a "🤔 Thinking..." placeholder, then the answer.
- `_tool_box()` renders a tool call as a collapsible 🔍 box (when `ToolStep` is enabled).
- `_print_startup_status()` prints the model and which features are ON/OFF, then re-indexes
  the notes if the file in the bucket changed. These lines show up in Render's **Logs**
  (`flush=True` makes sure they appear right away).
- `main()` handles launch settings:
  - **Locally:** `127.0.0.1:7860`.
  - **On Render:** `0.0.0.0` on Render's `PORT`.
  - **Optional login:** turned on when `APP_USERNAME` and `APP_PASSWORD` are set.
  - **Theme:** loads the CSS.

#### [`src/agentic_pokedex/__init__.py`](src/agentic_pokedex/__init__.py): the terminal chat
- `main()` is a simple input loop that also understands `reset` and `quit`. It's the same agent
  as the web app, just without the UI. Good for quick tests.

#### [`src/agentic_pokedex/config.py`](src/agentic_pokedex/config.py): settings
- **`Settings`**: which model provider and model to use (`LLM_PROVIDER`, `OPENAI_MODEL` /
  `ANTHROPIC_MODEL`), plus the PokéAPI URL.
- **`RagSettings`**: where the notes live (bucket and file names) and the embedding model and
  size. All of these have defaults, so `.env` only needs them to override.
- API keys are **not** stored here; each SDK reads its own key from the environment.

#### [`src/agentic_pokedex/assets/pokedex.css`](src/agentic_pokedex/assets/pokedex.css): the theme
1. **Palette tokens** (`--pdx-red`, `--pdx-white`, `--pdx-black`, …), with a dark-mode variant.
2. **Gradio mapping:** Gradio's own CSS variables (`--button-primary-background-fill`,
   `--color-accent`, …) pointed at those tokens, which recolors all of Gradio at once.
3. **Hooks:** `#pokedex-header` (the red banner) and `.pokeball` (a pure-CSS Poké Ball icon).

---

### `llm/`: talking to the language model

**Why it exists:** at first `agent.py` called OpenAI directly. I wanted to be able to switch to
Anthropic (Claude) later without rewriting the loop, so the model-specific code moved here
behind a neutral interface.

| File | Purpose |
|---|---|
| [`base.py`](src/agentic_pokedex/llm/base.py) | The **neutral interface**. `ChatSession` holds one conversation (`add_user`, `add_assistant`, `complete()`, `add_tool_results`). `LLMProvider` creates sessions and holds the API client. `ModelTurn`, `ToolCall` and `ToolResult` are the neutral data shapes. `LLMError` is the one error type the UIs catch. |
| [`openai_provider.py`](src/agentic_pokedex/llm/openai_provider.py) | OpenAI **Chat Completions**: the system prompt is the first message, tools use the `{"type": "function", ...}` format, and each tool result is its own `role: "tool"` message. |
| [`anthropic_provider.py`](src/agentic_pokedex/llm/anthropic_provider.py) | Anthropic **Messages API**: the system prompt is a separate argument, tools use `input_schema`, and all tool results go back together in **one** user message. Claude's replies contain **thinking blocks** that must be sent back unchanged, so the session stores the full reply content. It also turns on **server-side refusal fallbacks** on models that support them, and handles `refusal` / `max_tokens` stop reasons. Written and tested with fake clients, but **not yet run against the real API.** |
| [`__init__.py`](src/agentic_pokedex/llm/__init__.py) | `create_llm(settings)`: picks the provider from `LLM_PROVIDER`, importing only the SDK it needs. |

**Lesson:** each provider keeps the conversation in **its own native format**. Converting
back and forth through one shared format would lose details, like Claude's thinking blocks.

---

### `tools/`: what Rin can do

Every tool is a class with a `name`, a `description` (the model reads this to decide when to
use the tool), `parameters` (a JSON Schema for the arguments) and `run(**kwargs)`, which
returns a dict.

**Class hierarchy** (in [`base.py`](src/agentic_pokedex/tools/base.py)):
```
BaseTool                         anything the LLM can call
├── PokeAPITool                  + a shared PokeAPIClient
│   ├── PokeAPILookupTool        general: any of PokéAPI's ~51 resources
│   └── ResourceTool             specialized: fixed resource + curated summary
│       ├── PokemonSpeciesTool   resource = "pokemon-species"
│       └── AbilityTool          resource = "ability"
├── LogEventTool                 writes to Supabase (not PokéAPI)
├── SearchNotesTool              RAG search
├── ProposeNoteEditTool          drafts a notes change
└── ApplyNoteEditTool            saves a confirmed draft  (not_in_same_reply_as = propose_note_edit)
```

| File | Tool name (what the model sees) | Purpose |
|---|---|---|
| [`base.py`](src/agentic_pokedex/tools/base.py) | — | `BaseTool`, `PokeAPITool`, `ResourceTool`, plus helpers `english()` (picks the English entry from PokéAPI's many languages) and `names()`. Tools are **provider-neutral**: `llm/` converts them to each API's format. |
| [`general.py`](src/agentic_pokedex/tools/general.py) | `pokeapi_lookup` | **Generalization:** any resource (`pokemon`, `move`, `type`, `evolution-chain`, …). Its list of valid resources comes from PokéAPI's main endpoint at startup, so the model can't invent one. Raw responses are huge (a `pokemon` entry is about 70,000 characters), so it removes non-English text, sprites and per-game data, shortens lists to 15 items, and caps the result at 8,000 characters by dropping whole fields, so the JSON stays valid. |
| [`species.py`](src/agentic_pokedex/tools/species.py) | `get_pokemon_species` | **Specialization:** a clean species summary (category, Pokédex entry, generation, legendary/mythical, evolves-from, `evolution_chain_id`, egg groups, …). |
| [`ability.py`](src/agentic_pokedex/tools/ability.py) | `get_ability` | **Specialization:** what an ability does, and which Pokémon have it (hidden or not). |
| [`log_event.py`](src/agentic_pokedex/tools/log_event.py) | `log_event` | Saves a playthrough event (encounter, catch, battle, milestone, note) to the `log_event` table. `LOG_EVENT_PARAMETERS` defines the fields. |
| [`search_notes.py`](src/agentic_pokedex/tools/search_notes.py) | `search_notes` | RAG: finds the most relevant sections of my PokeMMO notes (default 4, max 6). |
| [`note_edits.py`](src/agentic_pokedex/tools/note_edits.py) | `propose_note_edit`, `apply_note_edit` | Two-step editing of the notes. **Propose** builds a draft and returns a preview plus a `draft_id`; nothing is saved. **Apply** saves that exact draft, and only in a *later* message. |
| [`registry.py`](src/agentic_pokedex/tools/registry.py) | — | **`ToolRegistry`**, the toolbox. `default()` builds the standard set. **Optional tools are added only if their settings exist** (Supabase keys, `OPENAI_API_KEY`), and `notices` records ON/OFF for the startup log. `execute(name, args_json)` runs a call and **returns errors as data instead of raising them**, so the model can read the error and retry (for example, after misspelling a Pokémon). |
| [`__init__.py`](src/agentic_pokedex/tools/__init__.py) | — | Re-exports the classes. |

**Lesson:** the tool `description` is a prompt. Rin chooses between `search_notes` and the
PokéAPI tools largely because of how their descriptions are worded.

---

### `pokeapi/`: fetching Pokémon data

| File | Purpose |
|---|---|
| [`client.py`](src/agentic_pokedex/pokeapi/client.py) | **`PokeAPIClient`**, the only code that calls pokeapi.co. Reuses one HTTP connection (`requests.Session`), **caches** every response (asking about Torchic twice means one HTTP call), and **normalizes names** (`"Mr. Mime"` → `mr-mime`). Raises `PokeAPINotFound` on a 404. `list_resources()` reads the main endpoint (`/api/v2/`). |
| [`__init__.py`](src/agentic_pokedex/pokeapi/__init__.py) | Re-exports. |

---

### `db/`: Supabase

| File | Purpose |
|---|---|
| [`client.py`](src/agentic_pokedex/db/client.py) | `get_db()` creates **one shared Supabase client**, on first use. `is_configured()` checks the env vars. Uses the **service role key**: it bypasses Row Level Security, so it must stay server-side. |
| [`events.py`](src/agentic_pokedex/db/events.py) | `log_event(...)` inserts a row into the `log_event` table. Empty fields are left out, so the table's defaults apply. |
| [`__init__.py`](src/agentic_pokedex/db/__init__.py) | Re-exports. |

---

### `rag/`: the knowledge notes

**RAG (Retrieval-Augmented Generation)** means: before answering, *retrieve* relevant text
from my own documents and give it to the model. The notes are one Markdown file,
`rin_pokemmo_notes.md`, in a Supabase Storage bucket.

```
 rin_pokemmo_notes.md (Storage)
        │  download
        ▼
 chunker.py: split at "## " headings ─► 55 chunks (one per section)
        │
        ▼
 embeddings.py: each chunk ─► 1,024 numbers (OpenAI text-embedding-3-small)
        │
        ▼
 index.py: store in doc_chunks (pgvector), with the file's SHA-256 fingerprint
        │
        ▼
 search: embed the question ─► match_chunks() SQL finds the closest chunks
```

| File | Purpose |
|---|---|
| [`chunker.py`](src/agentic_pokedex/rag/chunker.py) | `chunk_markdown()` splits the file into **one chunk per `##` section**. Each chunk starts with its heading, so it makes sense alone. Sections over 600 words are split by paragraph. |
| [`embeddings.py`](src/agentic_pokedex/rag/embeddings.py) | `Embedder` calls OpenAI embeddings with `dimensions=1024`, to match the `vector(1024)` column. It always uses OpenAI, even if the chat model is Claude, because **Anthropic has no embeddings API**. |
| [`index.py`](src/agentic_pokedex/rag/index.py) | **`NotesIndex`**. `download()` reads the file from Storage. `fingerprint()` is a SHA-256 of the text. `reindex()` re-chunks and re-embeds a file, replacing its rows, and **skips if the fingerprint is unchanged**. `sync_on_startup()` re-indexes changed files when the app starts and never crashes the app. `search()` calls the `match_chunks` SQL function. |
| [`reindex.py`](src/agentic_pokedex/rag/reindex.py) | The command line: `uv run python -m agentic_pokedex.rag.reindex` (add `--force` to rebuild everything). |
| [`notes.py`](src/agentic_pokedex/rag/notes.py) | **`NotesDocument`**: the file as a list of `Section(heading, body)`. `parse()` and `render()` round-trip the file **byte-for-byte**, so sections nobody touched never change. `replace()` and `add_after()` edit one section and append *"Edited by Rin on <date>"* or *"Added by Rin on <date>"*. `find()` matches headings loosely and suggests close matches. Raises `NoteEditError`. |
| [`editor.py`](src/agentic_pokedex/rag/editor.py) | **`NotesEditor`**: propose → apply. `propose_*` builds the whole new file in memory as a **draft** (expires after 15 minutes). `apply()` refuses if the file changed since the draft was made (fingerprint check), **backs up** the current file to `history/<name>.<UTC time>.md` in the bucket, saves the draft (as `bytes`: a `str` would be treated as a file path!) and re-indexes. |
| [`__init__.py`](src/agentic_pokedex/rag/__init__.py) | Re-exports. |

---

### Project-root files

| File | Purpose |
|---|---|
| [`hello_llm.py`](hello_llm.py) | The **first experiment**: one OpenAI call with no tools, to check the key works. Kept as a reminder of the starting point. Run it with `uv run python hello_llm.py`. |
| [`pyproject.toml`](pyproject.toml) | Project metadata and dependencies, plus two commands: `agentic-pokedex` (terminal) and `pokedex-web` (web). |
| [`uv.lock`](uv.lock) | Exact versions of every package, so installs are reproducible. Managed by uv; don't edit it. |
| [`requirements.txt`](requirements.txt) | The same pinned list in pip's format, **for Render** (Render installs with pip). The `-e .` line installs this package itself. **Re-export it after every `uv add`** (command below). |
| [`render.yaml`](render.yaml) | The Render **Blueprint**: build and start commands, the free plan, and env vars. Secrets are marked `sync: false`, so their values live only in Render's dashboard. |
| [`.env.example`](.env.example) | A template listing every setting. Copy it to `.env` and fill it in. |
| `.env` | My real secrets. **Git-ignored; never commit it.** |
| [`.gitignore`](.gitignore) | Keeps `.env`, `.venv/`, caches and build output out of git. |
| [`.python-version`](.python-version) | `3.12`, so uv uses Python 3.12 for this project. |
| [`.vscode/settings.json`](.vscode/settings.json) | Makes VS Code's terminal load `.env`. |

---

## 4. Following one question through the code

**"What does my notes say about Shiny Wars?"** in the web app:

1. **`app.py` → `respond()`** gets the message and the tab's chat history, and creates a
   `PokedexAgent` with the shared `llm` and `registry`.
2. **`agent.py` → `run()`** adds the question to the session and calls `session.complete()`.
3. **`llm/openai_provider.py`** sends the system prompt, the history, the question and the
   tool definitions to OpenAI. The model replies with a tool call:
   `search_notes(query="Shiny Wars")`.
4. **`agent.py`** checks the same-reply guard, then calls
   `registry.execute("search_notes", ...)`.
5. **`tools/search_notes.py`** → **`rag/index.py` → `search()`**:
   1. `rag/embeddings.py` turns the query into 1,024 numbers.
   2. Supabase's `match_chunks` returns the closest sections, with "Events: Shiny Wars" first.
6. The results go back to the model through `session.add_tool_results()`, and the loop runs
   again.
7. Now the model has what it needs, so it replies with text and no tool calls. `run()`
   yields a `FinalAnswer`.
8. **`app.py`** replaces the "🤔 Thinking..." placeholder with Rin's answer.

Editing works the same way, but takes **two messages**:
- **Message 1:** `propose_note_edit` returns a preview, and Rin asks *"Shall I save this?
  (draft xxxx)"*.
- **Message 2:** after my "yes", `apply_note_edit(draft_id)` saves it.

The draft ID appears in Rin's message because the web app rebuilds each new turn from the
**visible** chat text only.

---

## 5. Supabase: what lives where

| Thing | Type | Used by | Notes |
|---|---|---|---|
| `log_event` | table | `db/events.py` | Journal rows: `event_type`, `region`, `location`, `pokemon`, `opponent`, `outcome`, `quantity`, `is_shiny`, `details` (jsonb), `note`, `occurred_at`. **RLS on.** |
| `doc_chunks` | table | `rag/index.py` | `source`, `chunk_index`, `content`, `embedding vector(1024)`, `metadata` (jsonb: `heading`, `sha256`). **RLS on.** |
| `match_chunks(query_embedding, match_count)` | SQL function | `rag/index.py` | Vector similarity search. Returns `id`, `source`, `content`, `similarity`. |
| `RAG Files` | Storage bucket (private) | `rag/` | `rin_pokemmo_notes.md`, plus backups in `history/`. |

**Keys:** the app uses the **service role** key, which bypasses RLS. That's safe only
because it never leaves the server. With RLS on and no policies, the public "anon" key can't
read or write anything.

---

## 6. Running it locally

```powershell
uv sync                       # install everything into .venv
Copy-Item .env.example .env   # then fill in the real values
uv run pokedex-web            # web UI on http://127.0.0.1:7860
uv run agentic-pokedex        # or: chat in the terminal
```

Useful extras:
```powershell
uv run python -m agentic_pokedex.rag.reindex           # re-index notes that changed
uv run python -m agentic_pokedex.rag.reindex --force   # rebuild the whole index
uv run python hello_llm.py                             # the original one-call smoke test
```

If `uv add` fails with *"file is being used by another process"*, the app is still running.
Stop it with `Ctrl+C` first.

---

## 7. Configuration (environment variables)

| Variable | Required? | Purpose |
|---|---|---|
| `LLM_PROVIDER` | no (default `openai`) | `openai` or `anthropic` |
| `OPENAI_API_KEY` | yes | Chat (when `openai`) **and** embeddings (always) |
| `OPENAI_MODEL` | yes for `openai` | e.g. `gpt-5.4-mini` |
| `ANTHROPIC_API_KEY` | for `anthropic` | Claude key from platform.claude.com |
| `ANTHROPIC_MODEL` | no (default `claude-opus-5-5`) | `claude-sonnet-5-5` / `claude-haiku-5-5` are cheaper |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | for journal + notes | Without them, those features are switched **off** (the startup log says so) |
| `APP_USERNAME`, `APP_PASSWORD` | recommended when deployed | Turns on the login screen |
| `RAG_BUCKET`, `RAG_FILES` | no | Defaults: `RAG Files`, `rin_pokemmo_notes.md` |
| `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | no | Defaults: `text-embedding-3-small`, `1024` (must match `vector(1024)`) |

`.env` is for local runs only. On Render, set the same names in **Environment**. Render never
sees `.env`.

---

## 8. Deploying and day-to-day tasks

**Deploying:** Render is connected to this repo through the Blueprint (`render.yaml`).
**Every push or merge to `main` redeploys automatically.** I work on `feature/*` branches and
merge through pull requests.

Checklist when something changes:

| I changed… | Also do… |
|---|---|
| dependencies (`uv add …`) | `uv export --no-hashes --no-dev --format requirements-txt -o requirements.txt`, then commit |
| a secret / API key | Render → Environment (and my local `.env`) |
| a non-secret setting | `render.yaml` (it's applied on the next Blueprint sync) |
| the notes file | upload it to the `RAG Files` bucket → restart / redeploy (it re-indexes automatically), or run the reindex command |
| Rin's personality or rules | `SYSTEM_PROMPT` in `agent.py` |

**Switching to Claude:** set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` (and optionally
`ANTHROPIC_MODEL`). No code changes needed. Embeddings stay on OpenAI.

**After a deploy, check Render's Logs** for:
```
Model: openai / gpt-5.4-mini
Journal logging: ON
Knowledge notes search + editing: ON
Notes rin_pokemmo_notes.md: up to date (55 chunks)
```

**Free plan notes:**
- The app sleeps after about 15 minutes idle; the first visit after that takes about a minute.
- Login sessions are kept in memory, so I log in again after every sleep or redeploy.

---

## 9. Design decisions worth remembering

- **Specialized tools plus one general tool.** A few curated tools (species, ability) give
  clean, small answers. `pokeapi_lookup` covers everything else. The model prefers the
  specialized ones because the descriptions and prompt say to.
- **Shrink tool output.** Raw API data is huge and costs tokens. Tools should return what the
  model needs, not everything.
- **Errors as data.** Tool errors go back to the model as `{"error": ...}`, so it can fix a
  typo and retry, instead of the app crashing.
- **Features turn off loudly, not silently.** An early bug: without Supabase keys on Render,
  `log_event` silently disappeared, and the model **pretended** it had logged things. The fix
  had three parts: startup ON/OFF notices, a prompt rule ("only say it was logged if the tool
  returned success"), and the lesson that **models will claim success unless told not to**.
- **Provider-neutral core.** `agent.py` doesn't know which LLM it's using, so switching
  providers is a configuration change.
- **Markdown, not PDF, for the knowledge base.** Markdown is easy for the model to read and
  write, splits cleanly at headings, and can still be exported to PDF later.
- **One chunk per section.** The notes were written as self-contained `## Topic: subtopic`
  sections of 200–440 words, so chunking is simple and search results make sense on their own.
- **Fingerprints instead of polling.** The index stores a SHA-256 of the file. A restart or
  the reindex command re-embeds only when the file actually changed.
- **Writes need human confirmation, enforced in code.** Editing is propose → I answer →
  apply. The prompt asks for it, **and** the agent loop's same-reply guard enforces it. A
  backup is saved before every write, and stale drafts are refused.
- **Small, safe edits.** Rin edits one section at a time, and the byte-for-byte round trip
  guarantees untouched sections stay identical.
- **Secrets stay on the server.** `.env` is git-ignored, Render secrets use `sync: false`,
  the service role key is never sent to the browser, and RLS is on for every table.

---

## 10. Glossary

| Term | Meaning here |
|---|---|
| **Agent** | An LLM in a loop that can call tools and decide what to do next |
| **Tool** | A Python function the model may call, described by a name, a description and a JSON Schema |
| **Tool call** | The model's request to run a tool with specific arguments |
| **System prompt** | The standing instructions sent before the conversation (Rin's personality and rules) |
| **Provider** | The company and API serving the model (OpenAI, Anthropic) |
| **RAG** | Retrieval-Augmented Generation: look up relevant text from my documents, then answer |
| **Embedding** | A list of numbers (here 1,024) representing a text's meaning; similar texts have nearby vectors |
| **Chunk** | One piece of a document that gets embedded and retrieved (here, one `##` section) |
| **pgvector** | The Postgres extension that stores vectors and finds the nearest ones |
| **Fingerprint (SHA-256)** | A short hash of the file's text; if the text changes, the hash changes |
| **RLS** | Row Level Security: Postgres rules about who can read and write rows |
| **Service role key** | Supabase's admin key; it bypasses RLS, so it's server-only |
| **Blueprint** | Render's `render.yaml`: infrastructure described as code |
| **uv** | The tool that manages Python versions, the virtual environment and dependencies |

---

## 11. Known limitations and ideas

**Limitations**
- **Anthropic is wired up but not tested** against the real API yet.
- **Tool steps are hidden:** `yield ToolStep` is commented out in `agent.py`.
- **Drafts live in memory:** they're lost on restart or after 15 minutes, which is fine for a
  quick "yes".
- **Notes editing is replace or add only;** there's no delete yet.
- **The web history keeps only the visible text,** so follow-up questions may repeat lookups.
- **One shared journal:** `log_event` has no per-user column; it's fine for a single-user app.
- **The login page uses Gradio's default orange:** it doesn't load `pokedex.css`.

**Ideas**
- Delete a section, or show a diff view in the edit preview.
- Export the notes to PDF.
- A tool to *read* the journal ("what did I catch this week?").
- Show tool steps in the UI with a toggle.
- Move the chat to Claude once the budget allows.

---

## 12. Project timeline

| Step | What was added |
|---|---|
| 1 | uv project, `.env`, `hello_llm.py`: the first LLM call |
| 2 | OOP tool structure: one general PokéAPI tool plus specialized species and ability tools |
| 3 | The agent loop (`agent.py`) and the terminal chat |
| 4 | Gradio web UI and the Poké Ball CSS theme |
| 5 | Render deployment (Blueprint, login, `PORT` / `0.0.0.0`) |
| 6 | Supabase journal: `log_event` table and tool |
| 7 | Persona: the assistant becomes **Rin** |
| 8 | Provider-neutral LLM layer (OpenAI now, Anthropic ready) |
| 9 | RAG Phase 1: notes in Storage → chunks → embeddings → `search_notes` |
| 10 | RAG Phase 2: Rin edits the notes (propose → confirm → apply, with backups) |

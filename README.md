# LangGraph Lab: Building AI Agents with LangGraph

A step-by-step implementation of LangGraph agents, from a single node to a multi-agent
**University Assistant** with routing, tools, memory and error handling.

## Repository contents

| File / folder | Description |
|---|---|
| `langgraph_lab.ipynb` | The executed notebook, with outputs |
| `README.md` | This file (setup, how to run, design choices) |
| `report.docx` | Short report answering the conceptual questions |
| `screenshots/` | Execution screenshots and graph diagrams |
| `screenshots/Architecture_diagram.png` | Architecture of the final assistant (`show_graph(assistant)`) |

## Setup

Tested on Windows with Python 3.14, LangGraph 1.2.18 and LangChain 1.4.7.

```bash
pip install -U langgraph langchain langchain-openai langchain-community pandas requests jupyter
```

### API key (OpenRouter)

The notebook uses **OpenRouter** with a **free model** instead of the OpenAI API. OpenRouter is
OpenAI-compatible, so `ChatOpenAI` is used with a different `base_url`.

- Create a key at https://openrouter.ai (set a low credit limit).
- Either set the environment variable `OPENROUTER_API_KEY` before starting Jupyter, or paste the key
  when the notebook prompts for it (`getpass`, so it is never displayed or stored in the notebook).
- Never write the key in a notebook cell or commit it to a repository.

## How to run

1. Start Jupyter: `jupyter notebook`, then open `langgraph_lab.ipynb`.
2. Run the cells **in order from top to bottom** (several cells redefine `AgentState` and `graph`).
3. Avoid *Restart & Run All* repeatedly: free OpenRouter models have a **daily request limit**, and
   the full notebook makes many LLM calls.

## Design choices

**LLM.** `ChatOpenAI` pointed at OpenRouter, model `nvidia/nemotron-3.5-super-vl:free` (backup:
`poolside/laguna-xs-2.1:free`), `temperature=0`, `max_retries=6`. The model must support **tool
calling**. Free models change often, so the model name may need to be updated.

**State.** `TypedDict` states for simple graphs. For conversational graphs, `messages` uses the
`add_messages` reducer, which appends messages instead of overwriting them.

**Routing.** An LLM classifier writes a category into the state. A routing function reads it and
`add_conditional_edges` sends the flow to the matching agent node. Invalid labels, empty input and
API errors all fall back to a safe `fallback` node instead of crashing.

**Tools.**
- `calculator`: built on `ast` and `operator`, allowing arithmetic only. The lab's `eval()` version
  is kept in the notebook as `calculator_unsafe` for comparison, because `eval()` can execute
  arbitrary code (for example `__import__('os')`).
- `current_datetime`: returns the current date and time.
- `wikipedia_search` (optional extension): a custom `@tool` that calls the Wikipedia search API and the
  REST summary endpoint with `requests` (10 s timeout, descriptive User-Agent, errors returned as text).
  LangChain's `WikipediaQueryRun` was tried first but failed with `JSONDecodeError`: it relies on the
  unmaintained `wikipedia` package, whose hard-coded User-Agent gets rate-limited by Wikimedia.

**Tool loop.** The agent node binds the tools with `bind_tools`. After each agent step, a conditional
edge goes to a `ToolNode` (with `handle_tool_errors=True`) if the last message contains tool calls,
otherwise to `END`. After the tool runs, the flow returns to the agent that requested it.

**Memory.** `MemorySaver` checkpointer with a `thread_id`. The same thread remembers earlier messages;
a different thread has separate memory.

**Final assistant.** `classifier` routes to one of five specialized agents (`administration`,
`courses`, `general`, `mathematics`, `programming`), each with its own system prompt and access to the
tools, plus a `fallback` node. See `screenshots/Architecture_diagram.png`.

## Results

- **Routing evaluation:** 6/6 correct (100%) on the 6 test questions of the notebook (2 per category
  for mathematics, programming and artificial intelligence), with 3 to 17 seconds per question.
- **Memory:** with a checkpointer, the agent answers "Ahmed" to "What is my name?" on the same thread;
  without it, or on another thread, it does not know.
- **Capstone demo:** six queries covering courses, programming, mathematics, administration, general
  (date) and empty input were all routed to the expected category.
- **Tool use:** the calculator trace for `(15 + 27) * 3` shows the full cycle (tool call, tool result `126`,
  final answer). The Wikipedia question calls `wikipedia_search` twice and the answer is based on the
  returned summaries.
- **Error handling:** when the free-model daily quota was exceeded (HTTP 429), the classifier caught the
  error and the assistant returned a polite message instead of crashing. A failing tool (the first Wikipedia
  attempt) was reported to the agent as a tool error without crashing the graph.

## Limitations and known issues

- Free OpenRouter models are rate limited (per minute and per day), which can return HTTP 429 errors.
- The evaluation set is small and unambiguous, and the check `expected in got` is lenient, so the 100%
  result should not be read as proof of robustness. Ambiguous questions were not tested.
- LLM outputs vary between runs and between models, and small free models are less reliable at
  routing and tool calling than larger ones.
- When a tool keeps failing, the agent retries the same call many times (about ten in our first Wikipedia
  attempt), which wastes requests. Passing `recursion_limit` in the run config caps this.
- Section 5 (extra tool) changes the tool list, so the graph assembly cell must be run again after it.
- The Wikipedia tool needs internet access.

## Possible improvements

- Use `with_structured_output` with a `Literal` or enum so that the classifier must return a valid
  category, and add few-shot examples for borderline questions.
- Extend the test set with ambiguous and out-of-scope questions, use exact matching, and repeat runs.
- Replace `MemorySaver` with a persistent checkpointer (for example SQLite) to keep conversations
  between sessions.

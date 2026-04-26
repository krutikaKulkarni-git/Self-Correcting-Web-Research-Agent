# Self-Correcting Web Research Agent

A multi-agent AI system that autonomously researches topics on the web, self-evaluates the quality of its own answers, and corrects itself when the quality falls short — all powered by Claude, LangGraph, SerpAPI, and TruLens.

## How It Works

The agent follows a looped pipeline:

```
User Prompt → Agent → Tools (Search + Browse) → Verifier → [Pass / Retry] → Human Input
```

1. **Agent Node** — Claude receives the user query and decides which tools to call.
2. **Tool Node** — Executes `search_google_serpapi` (structured Google results via SerpAPI) and `navigate_to_url` (page content via Playwright).
3. **Verifier Node** — A second Claude instance audits the agent's answer against the raw tool output, scoring it on groundedness, answer relevancy, and context precision.
4. **Self-Correction** — If the groundedness score is below 80% (4/5), the graph routes back to the agent with a corrective instruction to search more deeply.
5. **Human-in-the-Loop** — Once quality passes, execution pauses and waits for the user's next command, enabling multi-turn research sessions.
6. **TruLens Evaluation** — Every run is recorded with TruLens feedback metrics (groundedness, relevance) for observability.

## Features

- Self-correcting RAG loop with automatic retry on low-quality answers
- Real-time web search via SerpAPI
- Full page content extraction via Playwright (headless Chromium)
- Human-in-the-Loop (HITL) for multi-turn sessions
- TruLens integration for RAG triad evaluation (groundedness, relevance, context precision)
- Persistent conversation state via LangGraph's `InMemorySaver`

## Prerequisites

- Python 3.10+
- A SerpAPI account and API key
- An Anthropic API key

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/krutikakulkarni-git/self-correcting-web-research-agent.git
   cd self-correcting-web-research-agent
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers**

   ```bash
   playwright install chromium
   ```

4. **Configure environment variables**

   Copy the example file and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   SERPAPI_API_KEY=your_serpapi_api_key_here
   ```

## Running the Agent

```bash
python SerpApiTruEval.py
```

You will be prompted for an initial research question. After the agent completes a cycle, you can enter follow-up commands or type `exit` / `quit` to end the session.

## Viewing Evaluation Results

After the session ends, launch the TruLens dashboard:

```bash
trulens-eval browse
```

## Project Structure

```
Self-Correcting-Web-Research-Agent/
├── SerpApiTruEval.py   # Main agent: graph definition, tools, and execution loop
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── .gitignore
```

## Architecture

| Component | Role |
|-----------|------|
| LangGraph | Orchestrates the agent, tool, verifier, and HITL nodes |
| Claude (Haiku) | Primary agent LLM and verifier LLM |
| SerpAPI | Structured Google search results |
| Playwright | Headless browser for full page extraction |
| TruLens + LiteLLM | RAG evaluation and observability dashboard |

## License

MIT

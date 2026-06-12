LangChain Research Assistant Agent

I built this LangChain Research Assistant Agent to practice making AI Agents with Python and the various features it can offer. It provides memory through short-term, long-term with FAISS, and episodic, orchestration with retries and guardrails, and an automated evaluation framework.

Disclaimer that I did use AI to help compile everything I did for this project but I used my own mind for the final writing down of the details here in this README.md file. The second half was written with AI but mainly for the formatting.

Files

- agent_setup.py
- agent_tools.py
- agent_memory.py
- memory_agent.py
- agent_orchestrator.py
- agent_evaluation.py

Features

Three-Layer Memory Architecture

- Short-term through MemorySaver within a session
- Long-term through FAISS vector store that persists across sessions
- Episodic through conversation exchanges saved to disk that is recalled via semantic search

Tools

- Web search via DuckDuckGo
- Web content fetching for full-page retrieval
- Calculator for math and unit conversions
- Text summarization for long-form content

Thread-Bases Session Management

- Allows for easier managing of memory across sessions

Production Orchestration Layer

- Retry logic through configurable max_retries that automatically re-attempts on failure
- Timeout enforcement - timeout_seconds prevents runaway queries
- Guardrails - validates inputs (e.g. rejects empty queries)
- Metrics tracking via internal AgentMetrics dataclass

Automated Evaluation Framework

- AgentEvaluator class takes an AgentOrchestrator instance and runs structured test suites
- Tracks tool accuracy - whether agent used expected tools for each query
- Sorts test runs by difficulty levels
- Resets tool_usage metrics before each eval run for clean scorng
- Outputs summary report with per-case results and overall statistics

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install langchain langchain-community langchain-openai langchain-huggingface
pip install langgraph faiss-cpu sentence-transformers duckduckgo-search
```

### Environment Setup

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Run the Agent (CLI)

```bash
python memory_agent.py
```

On first run, a `faiss_index/` directory is created. Subsequent runs load from it automatically.

### Run Evaluations

```bash
python agent_evaluation.py
```

Runs the full test suite and prints a per-case report with tool accuracy and overall scores.

### Running It on Your Web Browser

There is now an option to run this program on your web browser through Streamlit! Simply run this command:

```bash
streamlit run app.py
```

and click on the localhost link it gives you. This will allow you to play around with the chatbot as you enter in commands and on the side the metrics will show up as well.

---

## 🔄 How It All Fits Together

```
User Query
    │
    ▼
AgentOrchestrator (agent_orchestrator.py)
    ├─ Validates input (guardrails)
    ├─ Searches long-term FAISS index for relevant past context
    ├─ Injects context + query into agent prompt
    │
    ▼
memory_agent.py (LangGraph ReAct loop)
    ├─ Reasons over tools
    ├─ Calls: search_web / calculate / fetch / summarize
    ├─ Returns response
    │
    ▼
AgentOrchestrator (post-processing)
    ├─ Saves exchange to FAISS (long-term memory)
    ├─ Updates metrics (response time, tool usage, success rate)
    └─ Returns {"success": True, "output": "..."}
```

**Session continuity** is handled by `thread_id`:
```python
config = {"configurable": {"thread_id": "session_1"}}
# Same thread_id = agent remembers everything from this session
# New thread_id = fresh conversation, but long-term memory still available
```

---

## 🧪 Evaluation Example

```python
orchestrator = AgentOrchestrator(max_retries=3, timeout_seconds=60, enable_guardrails=True)
evaluator = AgentEvaluator(orchestrator)

evaluator.add_test_case(
    query="What is the population of Paris?",
    expected_tools=["search_web"],
    difficulty="easy",
    description="Basic web search"
)
evaluator.add_test_case(
    query="Calculate the square root of 144",
    expected_tools=["calculate"],
    difficulty="easy",
    description="Calculator tool"
)

results = evaluator.run_evaluation(thread_id="eval_session")
```

Sample output:
```
🧪 Running 2 test cases...

  Test 1 [easy]: Basic web search     ✔ 5.41s
  Test 2 [easy]: Calculator tool      ✔ 1.30s

--- Metrics ---
success_rate:      100.00%
avg_response_time: 3.36s
tool_usage:        {'search_web': 1, 'calculate': 1}
```

---

## 🔩 Debugging Notes

A few real gotchas encountered building this:

- **`langchain.agents.create_agent` import** — moved in recent versions; use `from langchain.agents import create_agent`
- **DuckDuckGo missing dependency** — requires `duckduckgo-search` installed separately
- **FAISS deserialization** — requires `allow_dangerous_deserialization=True` when loading from disk
- **`InMemoryChatMessageHistory` vs `MemorySaver`** — the former doesn't integrate with LangGraph's checkpointer system; use `MemorySaver` + `thread_id` instead

---

## 🗺️ Roadmap

- [x] Short-term memory with `MemorySaver`
- [x] Long-term FAISS memory with disk persistence
- [x] Orchestration layer with retries, guardrails, and metrics
- [x] Automated evaluation framework
- [ ] Streamlit chat frontend
- [ ] Swap GPT for Claude (`ChatAnthropic` — one line in `agent_setup.py`)
- [ ] Multi-agent workflows with LangGraph
- [ ] Episodic memory with timestamps and decay
- [ ] Tool result caching to reduce redundant web searches

---

## 🧱 Built With

- [LangChain](https://github.com/langchain-ai/langchain) — agent framework and tooling
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful agent orchestration
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) — free web search tool

---

## 📄 License

MIT — feel free to fork, extend, and build on it.
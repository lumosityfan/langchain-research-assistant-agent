# agent_orchestrator.py
import time
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from langchain.agents import create_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from agent_setup import llm
from agent_tools import research_tools
from agent_memory import short_term_memory, memory_system_prompt

logging.getLogger("ddgs.ddgs").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    response_times: list = field(default_factory=list)

    def success_rate(self) -> str:
        if self.total_queries == 0:
            return "0.00%"
        return f"{(self.successful_queries / self.total_queries) * 100:.2f}%"

    def avg_response_time(self) -> str:
        if not self.response_times:
            return "0.00s"
        return f"{sum(self.response_times) / len(self.response_times):.2f}s"


class ToolTrackingCallback(BaseCallbackHandler):
    def __init__(self, metrics: AgentMetrics):
        self.metrics = metrics

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.metrics.tool_usage[tool_name] = self.metrics.tool_usage.get(tool_name, 0) + 1
        logger.info(f"Tool called: {tool_name}")


class AgentOrchestrator:
    def __init__(self, max_retries: int = 3, timeout_seconds: int = 120, enable_guardrails: bool = True):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_guardrails = enable_guardrails
        self.metrics = AgentMetrics()
        self.callback = ToolTrackingCallback(self.metrics)
        self.agent = create_agent(
            model=llm,
            tools=research_tools,
            system_prompt=memory_system_prompt,
            checkpointer=short_term_memory
        )

    def validate_input(self, query: str):
        if not query or not query.strip():
            return False, "Query cannot be empty"
        if len(query) > 5000:
            return False, "Query too long (max 5000 characters)"
        for pattern in ["ignore previous instructions", "disregard all", "system:", "___"]:
            if pattern in query.lower():
                return False, f"Potentially unsafe input detected: {pattern}"
        return True, None

    def validate_output(self, output: str):
        for pattern in ["api_key", "password", "secret", "token"]:
            if pattern in output.lower():
                logger.warning(f"Output contains sensitive pattern: {pattern}")
        return True, None

    def execute(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        self.metrics.total_queries += 1
        start_time = time.time()

        if self.enable_guardrails:
            is_valid, error_msg = self.validate_input(query)
            if not is_valid:
                self.metrics.failed_queries += 1
                return {"success": False, "error": error_msg, "output": None}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.max_retries} for query: {query[:50]}...")
                result = self.agent.invoke(
                    {"messages": [("user", query)]},
                    config={
                        "configurable": {"thread_id": thread_id},
                        "callbacks": [self.callback]
                    }
                )
                output = result['messages'][-1].content

                if self.enable_guardrails:
                    self.validate_output(output)

                elapsed = time.time() - start_time
                self.metrics.successful_queries += 1
                self.metrics.response_times.append(elapsed)
                logger.info(f"Query completed in {elapsed:.2f}s")

                return {
                    "success": True,
                    "output": output,
                    "metadata": {
                        "attempts": attempt + 1,
                        "elapsed_time": f"{elapsed:.2f}s",
                        "thread_id": thread_id
                    }
                }
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        self.metrics.failed_queries += 1
        return {"success": False, "error": last_error, "output": None}

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_queries": self.metrics.total_queries,
            "successful_queries": self.metrics.successful_queries,
            "failed_queries": self.metrics.failed_queries,
            "success_rate": self.metrics.success_rate(),
            "avg_response_time": self.metrics.avg_response_time(),
            "tool_usage": self.metrics.tool_usage
        }


if __name__ == "__main__":
    orchestrator = AgentOrchestrator(
        max_retries=3,
        timeout_seconds=60,
        enable_guardrails=True
    )

    test_queries = [
        ("What is the population of Paris?", "session_1"),
        ("Calculate the square root of 144", "session_1"),
        ("", "session_1"),
        ("What are the main features of AI agents?", "session_2"),
    ]

    for query, thread_id in test_queries:
        result = orchestrator.execute(query, thread_id=thread_id)
        if result["success"]:
            print(f"✔ [{thread_id}] {result['output'][:100]}...")
        else:
            print(f"✘ [{thread_id}] {result['error']}")

    print("\n--- Metrics ---")
    for key, value in orchestrator.get_metrics().items():
        print(f"{key}: {value}")
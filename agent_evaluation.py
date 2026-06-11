# agent_evaluation.py
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from agent_orchestrator import AgentOrchestrator

logging.getLogger("ddgs.ddgs").setLevel(logging.ERROR)
logging.getLogger("primp").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class AgentEvaluator:
    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.test_cases: List[Dict] = []
        self.results: List[Dict] = []

    def add_test_case(
        self,
        query: str,
        expected_tools: List[str],
        difficulty: str = "medium",
        description: str = ""
    ):
        self.test_cases.append({
            "query": query,
            "expected_tools": expected_tools,
            "difficulty": difficulty,
            "description": description
        })

    def run_evaluation(self, thread_id: str = "eval_session") -> Dict:
        self.results = []
        # Reset tool usage before eval run
        self.orchestrator.metrics.tool_usage = {}

        print(f"\n🧪 Running {len(self.test_cases)} test cases...\n")

        for i, tc in enumerate(self.test_cases, 1):
            print(f"  Test {i} [{tc['difficulty']}]: {tc['description'] or tc['query'][:50]}")

            # Track tool usage before this query
            tools_before = dict(self.orchestrator.metrics.tool_usage)

            start = datetime.now()
            result = self.orchestrator.execute(tc["query"], thread_id=f"{thread_id}_{i}")
            elapsed = (datetime.now() - start).total_seconds()

            # Figure out which tools were called for this query
            tools_after = self.orchestrator.metrics.tool_usage
            tools_called = [
                tool for tool in tools_after
                if tools_after[tool] > tools_before.get(tool, 0)
            ]

            # Check if expected tools were used
            tools_matched = all(t in tools_called for t in tc["expected_tools"])

            evaluation = {
                "test_case": tc,
                "result": result,
                "elapsed_time": elapsed,
                "passed": result["success"],
                "tools_called": tools_called,
                "tools_matched": tools_matched,
            }
            self.results.append(evaluation)

            status = "✔ PASS" if evaluation["passed"] else "✘ FAIL"
            tool_status = "✔ tools matched" if tools_matched else f"✘ expected {tc['expected_tools']}, got {tools_called}"
            print(f"         {status} ({elapsed:.2f}s) | {tool_status}")
            if result["success"]:
                print(f"         Output: {result['output'][:80]}...")
            else:
                print(f"         Error: {result['error']}")
            print()

        return self._generate_report()

    def _generate_report(self) -> Dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        tools_matched = sum(1 for r in self.results if r["tools_matched"])
        avg_time = sum(r["elapsed_time"] for r in self.results) / total

        # Break down by difficulty
        by_difficulty = {}
        for r in self.results:
            d = r["test_case"]["difficulty"]
            if d not in by_difficulty:
                by_difficulty[d] = {"total": 0, "passed": 0}
            by_difficulty[d]["total"] += 1
            if r["passed"]:
                by_difficulty[d]["passed"] += 1

        recommendations = []
        if avg_time > 30:
            recommendations.append("⚠ Avg response time > 30s — consider reducing complexity or switching models")
        if passed < total:
            recommendations.append(f"⚠ {total - passed} test(s) failed — review error logs")
        if tools_matched < total:
            recommendations.append(f"⚠ {total - tools_matched} test(s) used unexpected tools — review tool descriptions")
        if not recommendations:
            recommendations.append("✔ All metrics within acceptable ranges!")

        report = {
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": f"{(passed / total) * 100:.2f}%",
                "tool_match_rate": f"{(tools_matched / total) * 100:.2f}%",
                "avg_response_time": f"{avg_time:.2f}s",
            },
            "by_difficulty": by_difficulty,
            "recommendations": recommendations
        }

        print("--- Evaluation Report ---")
        print(json.dumps(report["summary"], indent=2))
        print("\nBy difficulty:")
        for diff, stats in by_difficulty.items():
            print(f"  {diff}: {stats['passed']}/{stats['total']} passed")
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  {rec}")

        return report


if __name__ == "__main__":
    orchestrator = AgentOrchestrator(
        max_retries=3,
        timeout_seconds=60,
        enable_guardrails=True
    )

    evaluator = AgentEvaluator(orchestrator)

    evaluator.add_test_case(
        query="What is 15 multiplied by 23?",
        expected_tools=["calculate"],
        difficulty="easy",
        description="Basic math calculation"
    )
    evaluator.add_test_case(
        query="What is the current population of Tokyo?",
        expected_tools=["search_web"],
        difficulty="easy",
        description="Simple web search"
    )
    evaluator.add_test_case(
        query="Search for the latest LangChain features and summarize them",
        expected_tools=["search_web", "summarize_text"],
        difficulty="medium",
        description="Search and summarize"
    )
    evaluator.add_test_case(
        query="Find Tokyo's population, then calculate what percentage it is of Japan's total population of 125 million",
        expected_tools=["search_web", "calculate"],
        difficulty="hard",
        description="Multi-step search and calculation"
    )
    evaluator.add_test_case(
        query="",
        expected_tools=[],
        difficulty="easy",
        description="Empty query guardrail test"
    )

    report = evaluator.run_evaluation()
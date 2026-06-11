from agent_setup import llm
from agent_tools import research_tools
from agent_memory import (
    short_term_memory, 
    memory_system_prompt,
    save_to_long_term,
    search_long_term,
    episodic_memory
)
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=research_tools,
    system_prompt=memory_system_prompt,
    checkpointer=short_term_memory
)

config = {"configurable": {"thread_id": "session_1"}}

def run_memory_agent(query: str) -> str:
    # Inject long-term and episodic context into the query
    long_term_context = search_long_term(query)
    episodic_context = episodic_memory.retrieve_similar_episodes(query)

    augmented_query = f"""{query}

[IMPORTANT - Use this information from your memory before searching the web]:
Long-term memory: {long_term_context}
Past episodes: {episodic_context}

If the above memory contains a relevant answer, use it directly and acknowledge 
that you recall it from a previous conversation. Only search the web if memory 
is insufficient."""

    result = agent.invoke(
        {"messages": [("user", augmented_query)]},
        config=config
    )
    response = result['messages'][-1].content

    # Save to long-term and episodic memory
    save_to_long_term(query, response)
    episodic_memory.add_episode(query, response)

    return response

if __name__ == "__main__":
    response0 = run_memory_agent("What do you know about LangChain from our previous conversations?")
    print(f"Response 0 (from long-term memory): {response0}\n")
    response1 = run_memory_agent("What are the key features of LangChain?")
    print(f"Response 1: {response1}\n")

    response2 = run_memory_agent("How does it compare to CrewAI?")
    print(f"Response 2: {response2}\n")

    # Should reference the first answer since it's the same thread
    response3 = run_memory_agent("What are the key features of LangChain?")
    print(f"Response 3 (references past): {response3}")
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os, datetime

# Short-term memory via LangGraph checkpointer
short_term_memory = MemorySaver()

# Long-term memory (vector store for semantic retrieval)
embeddings = OpenAIEmbeddings()

if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("Loaded existing long-term memory from disk")
else:
    vectorstore = FAISS.from_texts(["Agent initialized."], embeddings)
    print("Created new long-term memory")

def save_to_long_term(query: str, response: str):
    """Save a conversation exchange to long-term memory."""
    vectorstore.add_texts([f"Q: {query}\nA: {response}"])
    vectorstore.save_local("faiss_index") # persist after every save

def search_long_term(query: str, k: int = 3) -> str:
    """Retrieve relevant past exchanges from long-term memory."""
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return "No relevant long-term memories found."
    return "\n".join([doc.page_content for doc in results])

# Episodic memory (event-based experience tracking)
class EpisodicMemory:
    """Stores specific episodes with timestamp and outcome."""
    def __init__(self):
        self.episodes = []

    def add_episode(self, query: str, outcome: str):
        episode = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "outcome": outcome
        }
        self.episodes.append(episode)
        if len(self.episodes) > 50: # Keep only last 50 episodes
            self.episodes = self.episodes[-50:]

    def retrieve_similar_episodes(self, query: str, top_k: int = 3) -> str:
        if not self.episodes:
            return "No past episodes found."
        query_words = set(query.lower().split())
        scored = []
        for ep in self.episodes:
            ep_words = set(ep['query'].lower().split())
            scored.append((len(query_words.intersection(ep_words)), ep))
        scored.sort(reverse=True, key=lambda x:x[0])
        if scored[0][0] == 0:
            return "No relevant past episodes found."
        result = "Similar past episodes:\n"
        for score, ep in scored[:top_k]:
            if score > 0:
                result += f"- [{ep['timestamp']}] {ep['query'][:50]}... -> {ep['outcome'][:50]}...\n"
        return result
    
episodic_memory = EpisodicMemory()

# System prompt — no need for {chat_history} or {agent_scratchpad}, LangGraph handles those
memory_system_prompt = """You are a helpful research assistant with memory capabilities.
You can search the web, fetch content, perform calculations, and summarize information.
You remember previous messages in the conversation and refer back to them when relevant."""

print("Memory systems initialized")
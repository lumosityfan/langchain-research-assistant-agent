import streamlit as st
from agent_orchestrator import AgentOrchestrator

st.title("Research Assistant Agent")

# Initialize orchestrator once and store in session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator(
        max_retries=3,
        timeout_seconds=60,
        enable_guardrails=True
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history onapp rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.orchestrator.execute(
                prompt,
                thread_id="streamlit_session"
            )
        if result["success"]:
            st.markdown(result["output"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["output"]
            })
        else:
            st.error(f"Error: {result['error']}")

# Sidebar with metrics
with st.sidebar:
    st.header("Agent Metrics")
    metrics = st.session_state.orchestrator.get_metrics()
    st.metric("Total Queries", metrics["total_queries"])
    st.metric("Success Rate", metrics["success_rate"])
    st.metric("Avg Response Time", metrics["avg_response_time"])

    if metrics["tool_usage"]:
        st.subheader("Tool Usage")
        for tool, count in metrics["tool_usage"].items():
            st.metric(tool, count)
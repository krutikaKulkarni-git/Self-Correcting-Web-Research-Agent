import os
import asyncio
import time
import re
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# Load environment variables FIRST before anything else
load_dotenv()

# --- TRULENS IMPORTS ---
from trulens.core import Select, TruSession, Metric
from trulens.apps.langchain import TruChain
from trulens.feedback import LLMProvider
from trulens.providers.litellm import LiteLLM
from trulens.core import Select



# --- LANGCHAIN & LANGGRAPH IMPORTS ---
from serpapi import GoogleSearch
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from playwright.async_api import async_playwright


import warnings
import logging

# Filter out the specific Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
# Optionally mute LiteLLM logging if it gets too chatty
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Failed to process and remove trivial statements")

# --- 1. STATE DEFINITION ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. BROWSER & TOOLS ---
browser_session = {"playwright": None, "browser": None, "page": None}

async def get_session():
    if browser_session["page"] is None:
        browser_session["playwright"] = await async_playwright().start()
        browser_session["browser"] = await browser_session["playwright"].chromium.launch(headless=True)
        context = await browser_session["browser"].new_context(viewport={'width': 1280, 'height': 720})
        browser_session["page"] = await context.new_page()
    return browser_session["page"]

@tool
async def search_google_serpapi(query: str):
    """Search Google via SerpApi for clean, structured results."""
    params = {"q": query, "api_key": os.getenv("SERPAPI_API_KEY"), "num": 5}
    search = await asyncio.to_thread(GoogleSearch(params).get_dict)
    organic = search.get("organic_results", [])
    return [{"title": r.get("title"), "link": r.get("link")} for r in organic] if organic else "No results."

@tool
async def navigate_to_url(url: str):
    """Navigates to a URL and extracts the first 3000 characters of text."""
    page = await get_session()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        content = await page.evaluate("() => document.body.innerText.substring(0, 3000)")
        return f"Content from {url}:\n\n{content}"
    except Exception as e:
        return f"Error: {str(e)}"

tools = [search_google_serpapi, navigate_to_url]
tool_node = ToolNode(tools)

# --- 3. LLM INITIALIZATION ---
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0
).bind_tools(tools)

verifier_llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0
)

# --- 4. NODES ---

async def agent_node(state: State):
    """Agent: Executes tasks and handles auditor feedback."""
    clean_messages = [m for m in state["messages"] if "--- VERIFICATION REPORT ---" not in str(m.content)]

    last_msg = state["messages"][-1]
    if "--- VERIFICATION REPORT ---" in str(last_msg.content):
        instruction = HumanMessage(content=(
            f"The auditor found issues: {last_msg.content}. "
            "Please fix these by searching more deeply or extracting missing data."
        ))
        clean_messages.append(instruction)

    response = await llm.ainvoke(clean_messages)
    return {"messages": [response]}

async def verifier_node(state: State):
    user_query = state["messages"][0].content
    # Get the last AI summary
    last_ai_msg = next((m.content for m in reversed(state["messages"]) 
                       if isinstance(m, AIMessage) and not m.tool_calls), "")
    # Get all tool outputs
    tool_results = "\n".join([str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)])

    prompt = f"""
    You are an objective QA Auditor evaluating a RAG system.
    
    USER QUERY: {user_query}
    CONTEXT (Tool Results): {tool_results}
    RESPONSE (Agent Summary): {last_ai_msg}

    TASK 1: Extract every factual claim from the RESPONSE.
    TASK 2: For each claim, check if it is explicitly supported by the CONTEXT.
    TASK 3: Calculate the Groundedness Score (Supported Claims / Total Claims).
    TASK 4: Rate how directly the RESPONSE answers the USER QUERY (1-5).

    Format exactly:
    --- VERIFICATION REPORT ---
    ### 📏 RAG Triad Metrics
    - **Groundedness Score:** [Numerator]/[Denominator] (e.g., 4/5)
    - **Answer Relevancy:** [1-5]
    - **Context Precision:** [High/Medium/Low]

    ### 🔍 Evidence Audit
    [List each claim and whether it passed or failed]
    """
    report = await verifier_llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=report.content)]}

def human_input_node(state: State):
    """HITL Node: Pauses for user feedback."""
    user_input = interrupt("Agent logic complete. Enter next command:")
    return {"messages": [HumanMessage(content=user_input)]}

# --- 5. ROUTING LOGIC ---

def route_after_agent(state: State):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "verify"

def route_after_verify(state: State):
    last_report = state["messages"][-1].content
    
    # Standard Regex to find the fraction score (e.g., 3/5)
    match = re.search(r"Groundedness Score:\s*(\d)/(\d)", last_report)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        score = (numerator / denominator) * 5 # Scale to 5
    else:
        score = 5 # Default to pass if parsing fails

    # If Groundedness is less than 80% (4/5), retry
    if score < 4:
        return "agent"
    return "human_input"

# --- 6. GRAPH CONSTRUCTION ---
workflow = StateGraph(State)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("verify", verifier_node)
workflow.add_node("human_input", human_input_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", route_after_agent)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("verify", route_after_verify)
workflow.add_edge("human_input", "agent")

app = workflow.compile(checkpointer=InMemorySaver())

# --- 7. TRULENS EVALUATION SETUP ---
# --- 7. TRULENS EVALUATION SETUP ---


tru = TruSession()
feedback_provider = LiteLLM(model_engine="claude-haiku-4-5-20251001")

# Standard Relevancy: Compares the very first input to the final output
f_relevance = Metric(feedback_provider.relevance).on_input_output()

# Standard Groundedness: 
# Note: In OTEL mode, 'on_input_output' is the safest default. 
# It will attempt to find the 'context' automatically from your tool calls.
f_groundedness = (
    Metric(feedback_provider.groundedness_measure_with_cot_reasons)
    .on_input_output()
)

tru_recorder = TruChain(
    app,
    app_id='Capstone_Accident_Agent_V2',
    feedbacks=[f_groundedness, f_relevance]
)

# --- 8. MAIN EXECUTION LOOP ---
async def main():
    config = {"configurable": {"thread_id": "capstone-multi-agent"}}
    print("\n" + "═"*50 + "\n 🚀 MULTI-AGENT CAPSTONE SYSTEM ACTIVE \n" + "═"*50)

    initial_prompt = input("\n[Prompt]: ")

    # Initial execution
    start_time = time.perf_counter()
    with tru_recorder as recording:
        async for _ in app.astream({"messages": [HumanMessage(content=initial_prompt)]}, config):
            pass
    latency = time.perf_counter() - start_time

    while True:
        # Retrieve current state
        state = await app.aget_state(config)
        messages = state.values["messages"]

        agent_summary = ""
        verification_report = ""
        total_tokens = 0

        # Extract most recent summary and audit report
        for msg in reversed(messages):
            if "--- VERIFICATION REPORT ---" in str(msg.content):
                if not verification_report:
                    verification_report = str(msg.content).replace("--- VERIFICATION REPORT ---", "").strip()
            elif isinstance(msg, AIMessage) and not msg.tool_calls and "--- VERIFICATION REPORT ---" not in str(msg.content):
                if not agent_summary:
                    agent_summary = str(msg.content)

        # Calculate total tokens
        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                total_tokens += msg.usage_metadata.get("total_tokens", 0)

        # UI DISPLAY
        print("\n🟢" + "─"*48)
        print(f" STEP 1: AGENT SUMMARY")
        print(f" [Latency: {latency:.2f}s | Session Tokens: {total_tokens}]")
        print("─"*50)
        print(agent_summary if agent_summary else "Collecting data...")

        print("\n🛡️" + "─"*48)
        print(f" STEP 2: QUALITY ASSURANCE AUDIT (TRULENS INTEGRATED)")
        print("─"*50)
        print(verification_report if verification_report else "Audit in progress.")

        print("\n💡" + "─"*48)
        print(f" STEP 3: NEXT ACTION")
        print("─"*50)

        user_cmd = input("\n[Next Command/Exit]: ")
        if user_cmd.lower() in ['exit', 'quit']:
            break

        # Follow-up steps
        start_time = time.perf_counter()
        with tru_recorder as recording:
            async for _ in app.astream(Command(resume=user_cmd), config):
                pass
        latency = time.perf_counter() - start_time

    # Cleanup
    if browser_session["browser"]:
        await browser_session["browser"].close()
        await browser_session["playwright"].stop()

    print("\nSession ended. To view TruLens Dashboard, run: trulens-eval browse")

if __name__ == "__main__":
    asyncio.run(main())
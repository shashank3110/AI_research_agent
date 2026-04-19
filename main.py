import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import hashlib
from langgraph.prebuilt import create_react_agent
from tools_literature import search_arxiv, search_wikipedia
from tools_implementation import write_python_script, execute_python_script, ask_human
from tools_evaluation import read_file
from tools_writing import write_document, compile_latex

# Load environment variables (e.g. GOOGLE_API_KEY)
load_dotenv()

# Define the state of our research workflow
class ResearchState(TypedDict):
    topic: str
    run_dir: str
    literature_review: str
    codebase: str
    evaluation_results: str
    draft_article: str

# Initialize the LLM (Gemini)
# Requires GOOGLE_API_KEY to be set in the environment
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2, max_retries=6)

# Load the system prompt
def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.md")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r") as f:
            return f.read()
    return "You are an autonomous AI research agent."

system_prompt = load_system_prompt()

# Setup the Literature Discovery Agent
literature_tools = [search_arxiv, search_wikipedia]
literature_agent = create_react_agent(
    llm, 
    tools=literature_tools, 
    prompt=f"{system_prompt}\n\nYou are the Literature Discovery & Review Agent."
)

# Node 1: Literature Discovery
def literature_discovery(state: ResearchState) -> dict:
    # Check cache first to save massive token limits on successive runs for the same topic!
    topic_hash = hashlib.md5(state['topic'].strip().lower().encode()).hexdigest()
    cache_path = os.path.join("logs", f"cache_literature_{topic_hash}.md")
    
    if os.path.exists(cache_path):
        print("\n--- Loaded Literature Discovery from Cache (Skipping LLM Tokens!) ---")
        with open(cache_path, "r") as f:
            return {"literature_review": f.read()}
            
    print("--- Running Literature Discovery ---")
    prompt = f"Conduct a rigorous and extremely detailed literature review for the topic: '{state['topic']}'. Use your tools to search arXiv and Wikipedia.\nFor EVERY paper you find, you MUST explicitly output a comprehensive section explicitly highlighting its: 1. Goal, 2. Methodologies, 3. Strengths, 4. Shortcomings, and 5. Conclusions.\nDo not be superficial; extract as much deep technical detail as possible."
    response = literature_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    review_content = response["messages"][-1].content
    
    # Save to dynamic static cache to avoid API hits next time
    os.makedirs("logs", exist_ok=True)
    with open(cache_path, "w") as f:
        f.write(review_content)
        
    return {"literature_review": review_content}

# Setup the Implementation Agent
implementation_tools = [write_python_script, execute_python_script, ask_human]
implementation_agent_node = create_react_agent(
    llm,
    tools=implementation_tools,
    prompt=f"{system_prompt}\n\nYou are the Implementation & Experimentation Agent."
)

# Node 2: Implementation & Experimentation
def implementation_agent(state: ResearchState) -> dict:
    print("--- Running Implementation & Experimentation ---")
    prompt = f"Based on the following literature review, autonomously write the implementation scripts in Python using your tools, execute them, and return the codebase details and execution summary. Do not ask the user for permission or wait for inputs.\n\nIMPORTANT: Save all generated code files strictly inside this directory: {state.get('run_dir', '.')}\n\nCRITICAL: In your final response, list the exact filenames you created so the evaluation agent can find and test them.\n\nLiterature Review:\n{state.get('literature_review', '')}"
    response = implementation_agent_node.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"codebase": response["messages"][-1].content}

# Setup the Evaluation Agent
evaluation_tools = [write_python_script, execute_python_script, read_file]
evaluation_agent_node = create_react_agent(
    llm,
    tools=evaluation_tools,
    prompt=f"{system_prompt}\n\nYou are the Evaluation Agent."
)

# Node 3: Evaluation
def evaluation_agent(state: ResearchState) -> dict:
    print("--- Running Evaluation ---")
    prompt = f"The implementation agent has finished writing the codebase to the directory: {state.get('run_dir', '.')}. Use your recursive `read_file` tools to inspect the generated python scripts.\nBased on those codebase files, autonomously write and execute evaluation scripts to benchmark the models against standardized datasets or synthetic data. Read the logs/results and synthesize them. \n\nIMPORTANT: Save any evaluation scripts and logs strictly inside this directory: {state.get('run_dir', '.')}\n\nImplementation Agent Summary:\n{state.get('codebase', '')}"
    response = evaluation_agent_node.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"evaluation_results": response["messages"][-1].content}

# Setup the Scientific Writing Agent
writing_tools = [write_document, compile_latex]
writing_agent_node = create_react_agent(
    llm,
    tools=writing_tools,
    prompt=f"{system_prompt}\n\nYou are the Scientific Writing Agent."
)

# Node 4: Scientific Writing
def writing_agent(state: ResearchState) -> dict:
    print("--- Running Scientific Writing ---")
    
    evaluation_text = state.get('evaluation_results', 'No empirical evaluations or code implementations were performed. Write the scientific article purely as a Literature Review / Survey paper based strictly on theoretical findings.')
    
    prompt = f"Based on the provided context, draft a massive, comprehensive, and highly detailed scientific article.\n1. The LaTeX source MUST be strictly Overleaf compatible. DO NOT use external .bib files (like \\bibliography{{...}}); you must embed all references natively using the \\begin{{thebibliography}} environment.\n2. You MUST use the `write_document` tool to explicitly save the `.tex` file.\n3. CRITICAL: For your FINAL text response, you MUST output the entire article formatted in clean Markdown. Do NOT use the write tool for the markdown; just output the markdown text directly in your final response message!\n\nThe output must be extremely detailed. In the literature review section, dedicate entire paragraphs to each reviewed paper, highlighting their explicit Goal, Methodologies, Strengths, Shortcomings, and Conclusions as extracted by the Literature agent. Do not be superficial.\n\nIMPORTANT: Save your .tex documents and compiled pdf strictly inside this directory: {state.get('run_dir', '.')}\n\nLiterature Review:\n{state.get('literature_review', '')}\n\nEvaluation/Empirical Context:\n{evaluation_text}"
    response = writing_agent_node.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"draft_article": response["messages"][-1].content}

def build_graph(run_experiments: bool):
    workflow = StateGraph(ResearchState)
    workflow.add_node("literature_discovery", literature_discovery)
    workflow.add_node("writing_agent", writing_agent)

    workflow.set_entry_point("literature_discovery")
    if run_experiments:
        workflow.add_node("implementation_agent", implementation_agent)
        workflow.add_node("evaluation_agent", evaluation_agent)
        workflow.add_edge("literature_discovery", "implementation_agent")
        workflow.add_edge("implementation_agent", "evaluation_agent")
        workflow.add_edge("evaluation_agent", "writing_agent")
    else:
        # Skip implementation and evaluation directly to survey writing
        workflow.add_edge("literature_discovery", "writing_agent")

    workflow.add_edge("writing_agent", END)
    return workflow.compile()

if __name__ == "__main__":
    import datetime
    
    print("\n" + "="*50)
    user_topic = input("Enter the research topic (or press Enter for default): ")
    topic = user_topic.strip() if user_topic.strip() else "Self-supervised feature learning for time series anomaly detection"
    
    mode = input("\nDo you want the agent to write code and perform empirical evaluations to compare against SOTA?\n(Type 'y' for Full Empirical Research, 'n' for Literature Review Only): ")
    run_experiments = mode.lower().startswith('y')
    print("="*50 + "\n")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Inject absolute isolated path into environment for all IO tools!
    import os
    os.environ["CURRENT_RUN_DIR"] = os.path.abspath(run_dir)
    
    print(f"Starting research workflow for topic: '{topic}'\nLogging to: {run_dir}\n")
    
    app = build_graph(run_experiments)
    
    # Run the graph (with a recursion limit of 25 to prevent infinite ReAct loops)
    final_state = app.invoke({"topic": topic, "run_dir": run_dir}, config={"recursion_limit": 25})
    
    print("\nSaving raw state texts to log folder...")
    with open(os.path.join(run_dir, "1_literature_log.md"), "w") as f:
        f.write(final_state.get("literature_review", ""))
    
    with open(os.path.join(run_dir, "2_implementation_log.txt"), "w") as f:
        f.write(final_state.get("codebase", ""))
        
    with open(os.path.join(run_dir, "3_evaluation_summary.txt"), "w") as f:
        f.write(final_state.get("evaluation_results", ""))
        
    with open(os.path.join(run_dir, "4_final_article_draft.md"), "w") as f:
        f.write(final_state.get("draft_article", ""))
        
    print("\n" + "="*50)
    print(f"WORKFLOW COMPLETE. All scripts, plots, documents, and logs are saved in: {run_dir}")
    print("="*50)

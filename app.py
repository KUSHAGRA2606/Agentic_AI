import streamlit as st
import time
import json
import os

from workflow import app as langgraph_app
from rag_core import COLLECTIONS

st.set_page_config(
    page_title="Agentic Hackathon Framework",
    layout="wide"
)

st.title("Agentic Framework")
st.markdown("Automated Research Synthesis & PRD Generation Pipeline")

if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False


with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("NVIDIA NIM API Key", type="password", value=os.getenv("NVIDIA_API_KEY", ""))
    if api_key:
        os.environ["NVIDIA_API_KEY"] = api_key
        st.success("API Key loaded!")
    else:
        st.warning("Please enter your NVIDIA API key to run LLM operations.")
        
    st.divider()
    st.markdown("### System Agents")
    st.checkbox("Orchestrator", value=True, disabled=True)
    st.checkbox("Librarian (Research)", value=True, disabled=True)
    st.checkbox("GitHub Agent", value=True, disabled=True)
    st.checkbox("Web Agent", value=True, disabled=True)


st.header("1. Define the Problem")
problem_statement = st.text_area(
    "Enter your Hackathon Problem Statement (PS):",
    height=150,
    placeholder="e.g., Build a multi-agent automated scientific literature review system that outputs a structured PRD."
)

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Run Pipeline", type="primary", use_container_width=True):
        if not problem_statement:
            st.error("Please enter a Problem Statement.")
        elif not os.getenv("NVIDIA_API_KEY"):
            st.error("NVIDIA API Key is required.")
        else:
            for collection in COLLECTIONS.values():
                try:
                    all_items = collection.get()
                    if all_items and all_items.get("ids"):
                        collection.delete(ids=all_items["ids"])
                except Exception as e:
                    st.sidebar.error(f"Error clearing vector DB: {e}")
            st.session_state.is_running = True
            st.session_state.final_state = None

if st.session_state.is_running:
    st.header("2. Agentic Execution Logs")
    
    initial_state = {
        "ps": problem_statement,
        "prd_version": 0,
        "confidence_scores": {}
    }
    
    with st.status("Initializing Agentic Pipeline...", expanded=True) as status:
        try:
            for event in langgraph_app.stream(initial_state):
                for node_name, state_update in event.items():
                    st.write(f"**{node_name}** completed.")
                    time.sleep(0.5) 
                    
                    if state_update and isinstance(state_update, dict):
                        if st.session_state.final_state is None:
                            st.session_state.final_state = {}
                        
                        st.session_state.final_state.update(state_update)
            
            status.update(label="Pipeline Execution Complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Pipeline Failed", state="error", expanded=True)
            st.error(f"Error during execution: {str(e)}")
            
    st.session_state.is_running = False

if st.session_state.final_state:
    state = st.session_state.final_state
    
    st.divider()
    st.header("3. Review & Evaluation Dashboard")
    
    tab1, tab2, tab3 = st.tabs([
        "Orchestrator Intent", 
        "Subagent Reports", 
        "Generated PRD", 
    ])
    
    with tab1:
        st.subheader("Parsed Intent")
        st.info(state.get("intent", "No intent parsed."))
        
        st.subheader("Generated Search Queries")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            st.markdown("**Research Queries**")
            st.write(state.get("research_queries", []))
        with col_q2:
            st.markdown("**GitHub Queries**")
            st.write(state.get("github_search_terms", []))
        with col_q3:
            st.markdown("**Web Queries**")
            st.write(state.get("web_search_queries", []))
            

    with tab2:
        st.subheader("🔍 Source Retrieval Analysis")
    
        scores = state.get("confidence_scores", {})
        if scores:
            col_c1, col_c2, col_c3 = st.columns(3)
            with col_c1: st.metric("Librarian Confidence", f"{scores.get('research', 0.0):.2f}")
            with col_c2: st.metric("GitHub Core Confidence", f"{scores.get('github', 0.0):.2f}")
            with col_c3: st.metric("Web Tracker Confidence", f"{scores.get('web', 0.0):.2f}")
        st.divider()

        res_report = state.get("research_report", {})
        papers = res_report.get("fetched_items", [])
    
        with st.expander(f"Academic Literature Verified ({len(papers)} Papers)", expanded=True):
            if not papers:
                st.info("No research papers were fetched.")
            for paper in papers:
                st.markdown(f"##### {paper.get('title')}")
        
                platform_info = f" | Platform: `{paper.get('platform', 'ArXiv')}`"
                citation_info = f" | Citations: **{paper.get('citations', 0)}**" if "citations" in paper else ""
        
                st.caption(f"**Year:** {paper.get('year')}{platform_info}{citation_info}")
                st.markdown(f"*{paper.get('abstract')}*")
                st.markdown("---")

        git_report = state.get("github_report", {})
        repos = git_report.get("fetched_items", [])
    
        with st.expander(f"Open-Source Repositories Scoped ({len(repos)} Repos)", expanded=False):
            if not repos:
                st.info("No repositories were found.")
            for repo in repos:
                st.markdown(f"##### [{repo.get('repo_url').split('/')[-1]}]({repo.get('repo_url')})")
                st.markdown(f"**Stars:** {repo.get('stars')} | **Primary Stack:** `{', '.join(repo.get('tech_stack', []))}`")
                st.markdown(f"> {repo.get('description')}")
                st.markdown("---")

        web_report = state.get("web_report", {})
        articles = web_report.get("fetched_items", [])
    
        with st.expander(f"Industry & Web Slices Contextualized ({len(articles)} Articles)", expanded=False):
            if not articles:
                st.info("No web articles were processed.")
            for art in articles:
                st.markdown(f"##### [{art.get('title')}]({art.get('url')})")
                snippet = art.get('content', '')[:300] + '...' if len(art.get('content', '')) > 300 else art.get('content','')
                st.write(snippet)
                st.markdown("---")


        hf_report = state.get("hf_report")
        if hf_report is None:
            hf_report = {}

        if isinstance(hf_report, dict):
            datasets = hf_report.get("fetched_items")
            if datasets is None:
                datasets = []
        else:
            datasets = []

        with st.expander(f"Hugging Face Datasets Discovered ({len(datasets)})", expanded=False):
            if not datasets:
                st.info("No free Hugging Face datasets were matched for this specific architecture requirement.")
            else:
                for ds in datasets:
                    hub_url = f"https://huggingface.co/datasets/{ds.get('name')}"
                    st.markdown(f"##### [{ds.get('name')}]({hub_url})")
            
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.caption(f"**Author:** {ds.get('author')}")
                    with col_d2:
                        st.caption(f"**License Tag:** `{ds.get('license').upper()}`")
        
                    st.metric(label="Community Downloads", value=f"{ds.get('downloads_count', 0):,}")
                    st.markdown("---")
            
    with tab3:
        st.subheader(f"Product Requirements Document (v{state.get('prd_version', 1)})")
        prd_sections = state.get("prd_sections", {})
        
        if not prd_sections:
            st.warning("No PRD sections were generated.")
        else:
            for sec_title, sec_content in prd_sections.items():
                st.markdown(f"### {sec_title}")
                st.markdown(sec_content)
                st.divider()
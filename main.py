import argparse
import json
import sys
import os

from workflow import run_pipeline

from dotenv import load_dotenv
load_dotenv()

def format_output(final_state: dict):
    """Utility to print the final output in a readable format."""
    print("PIPELINE EXECUTION COMPLETE 🎉")
    
    print(f"\n[Orchestrator Intent]: {final_state.get('intent')}")
    
    print("\n[Agent Confidence Scores]:")
    scores = final_state.get("confidence_scores", {})
    for agent, score in scores.items():
        print(f"  - {agent.capitalize()}: {score:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Agentic Framework for Automated Research Synthesis")
    
    parser.add_argument(
        "--ps", 
        type=str, 
        help="The Hackathon Problem Statement to process (enclose in quotes)."
    )
    
    args = parser.parse_args()

    if args.ps:
        print(f"\nInitializing Agentic pipeline for Problem Statement:\n\"{args.ps}\"\n")
        try:
            final_state = run_pipeline(args.ps)
            format_output(final_state)
                
        except Exception as e:
            print(f"\nPipeline failed with error: {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    if not os.getenv("NVIDIA_API_KEY"):
        print("WARNING: NVIDIA_API_KEY environment variable is not set. LLM calls will fail.")
    main()
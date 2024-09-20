import sys
from scripts import arxiv_search, select_papers, summarize_papers, podcast, cleanup
from utils.utils import resolve_config

def main():
    config = resolve_config()
    pipeline_steps = [step.strip() for step in config.get('pipeline', 'steps').split(',')]
    
    step_functions = {
        'arxiv_search': arxiv_search.search_papers,
        'select_papers': select_papers.select_top_papers,
        'summarize_papers': summarize_papers.summarize_papers,
        'podcast': podcast.generate_podcast,
        'cleanup': cleanup.cleanup_and_send_to_obsidian
    }
    
    for step in pipeline_steps:
        print(f"Executing step: {step}")
        if step in step_functions:
            step_functions[step](config=config)
        else:
            print(f"Warning: Unknown pipeline step '{step}'")
    
    print("Pipeline execution completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
        sys.exit(1)
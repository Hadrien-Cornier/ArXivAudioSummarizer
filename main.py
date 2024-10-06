import sys
from scripts import (
    arxiv_search,
    select_papers,
    summarize_papers,
    podcast,
    cleanup,
)  # , benchmark_pdf_extraction
from utils.utils import resolve_config
import os


def main():
    config = resolve_config()
    pipeline_steps = [
        step.strip() for step in config.get("pipeline", "steps").split(",")
    ]

    step_functions = {
        "arxiv_search": arxiv_search.search_papers,
        "select_papers": select_papers.select_top_papers,
        "summarize_papers": summarize_papers.summarize_papers,
        "podcast": podcast.generate_podcast,
        "cleanup": cleanup.cleanup_and_send_to_obsidian,
        # "benchmark": benchmark_pdf_extraction.main,  # Add this line
    }

    for step in pipeline_steps:
        print(f"Executing step: {step}")
        if step in step_functions:
            step_functions[step](config=config)

            # Check if papers were found after arxiv_search
            if step == "arxiv_search":
                papers_found_csv = os.path.join(
                    config.get("arxiv_search", "output_dir"), "papers_found.csv"
                )
                if (
                    not os.path.exists(papers_found_csv)
                    or os.path.getsize(papers_found_csv) == 0
                ):
                    print("No new papers found. Stopping pipeline.")
                    break
        else:
            print(f"Warning: Unknown pipeline step '{step}'")

    print("Pipeline execution completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
        sys.exit(1)

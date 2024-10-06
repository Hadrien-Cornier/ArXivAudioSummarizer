import configparser
import arxiv
import os
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.retry import retry_with_exponential_backoff
from openai import OpenAI
from utils.weaviate_client import get_or_create_class, get_weaviate_client


@retry_with_exponential_backoff
def fetch_arxiv_results(
    search: arxiv.Search, client: arxiv.Client
) -> List[arxiv.Result]:
    return list(client.results(search))


def search_papers(config: configparser.ConfigParser) -> None:
    arxiv_config: Dict[str, Any] = config["arxiv_search"]
    weaviate_config: Dict[str, Any] = config["weaviate"]
    output_dir: str = arxiv_config.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_file: str = os.path.join(
        os.path.dirname(output_dir), "most_recent_day_searched.txt"
    )

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            most_recent_day_searched = datetime.strptime(f.read().strip(), "%Y-%m-%d")
    else:
        most_recent_day_searched = datetime.now() - timedelta(
            days=arxiv_config.getint("date_range")
        )

    date_range = arxiv_config.getint("date_range")
    end_date = datetime.now().date()
    start_date = (most_recent_day_searched - timedelta(days=date_range)).date()

    query: str = (
        f"({arxiv_config.get('categories')}) AND submittedDate:[{start_date:%Y%m%d}000000 TO {end_date:%Y%m%d}235959]"
    )

    client: arxiv.Client = arxiv.Client(
        page_size=arxiv_config.getint("max_results"), delay_seconds=5.0, num_retries=3
    )
    search: arxiv.Search = arxiv.Search(
        query=query,
        max_results=arxiv_config.getint("max_results"),
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: List[Dict[str, Any]] = []

    try:
        results = fetch_arxiv_results(search, client)
    except Exception as e:
        print(f"Failed to fetch results from arXiv: {e}")
        return

    for result in results:
        if start_date <= result.published.date() <= end_date:
            paper = {
                "paper_id": result.get_short_id(),
                "title": result.title,
                "arxiv_url": result.entry_id,
                "pdf_url": result.pdf_url,
                "published_date": str(result.published.date()),
                "abstract": result.summary,
                "full_text": "",
            }
            papers.append(paper)
        else:
            print(
                f"Skipping result published on {result.published.date()} as it falls outside the selected date range."
            )
            print(f"Date range: {start_date} to {end_date}")

        if result.published.date() > most_recent_day_searched.date():
            most_recent_day_searched = result.published

        # Add paper to Weaviate
        paper_class = get_or_create_class(config["weaviate"].get("papers_class_name"))

        existing_papers = (
            get_weaviate_client()
            .collections.get(config["weaviate"].get("papers_class_name"))
            .query.fetch_object_by_id(paper["paper_id"])
        )

        # Extract the results
        existing_papers_data = existing_papers["data"]["Get"][
            config["weaviate"].get("papers_class_name")
        ]

        if not existing_papers_data:
            paper_class.data_object().create({**paper})
        else:
            print(
                f"Paper with ID {paper['paper_id']} already exists. Skipping insertion."
            )

    if papers:
        with open(
            os.path.join(output_dir, "papers_found.csv"),
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer: csv.writer = csv.writer(file)
            writer.writerow(
                ["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Abstract"]
            )
            writer.writerows([list(paper.values()) for paper in papers])

        # Update the checkpoint file with the most recent date
        with open(checkpoint_file, "w") as f:
            f.write(most_recent_day_searched.strftime("%Y-%m-%d"))

    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper['title']}")

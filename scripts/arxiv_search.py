# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configparser
import arxiv
import os
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.utils import read_lines_from_file
from utils.retry import retry_with_exponential_backoff
from chromadb import Client
from chromadb.config import Settings
from openai import OpenAI


@retry_with_exponential_backoff
def fetch_arxiv_results(
    search: arxiv.Search, client: arxiv.Client
) -> List[arxiv.Result]:
    return list(client.results(search))


def search_papers(config: configparser.ConfigParser) -> None:
    arxiv_config: Dict[str, Any] = config["arxiv_search"]
    output_dir: str = arxiv_config.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_file: str = os.path.join(output_dir, "most_recent_day_searched.txt")

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_date = datetime.strptime(f.read().strip(), "%Y-%m-%d")
    else:
        start_date = datetime.now()

    begin_date = start_date - timedelta(days=arxiv_config.getint("date_range"))
    end_date = datetime.now().date()
    query: str = (
        f"({arxiv_config.get('categories')}) AND submittedDate:[{start_date:%Y%m%d}000000 TO {datetime.now():%Y%m%d}235959]"
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
    most_recent_date = start_date

    # Initialize Chroma client
    chroma_client = Client(
        Settings(persist_directory=arxiv_config.get("chroma_persist_dir"))
    )
    chroma_collection = chroma_client.get_or_create_collection(name="papers")

    openai_client = OpenAI(
        api_key=open(config.get("openai", "api_key_location")).read().strip()
    )

    try:
        results = fetch_arxiv_results(search, client)
    except Exception as e:
        print(f"Failed to fetch results from arXiv: {e}")
        return

    for result in results:
        if result.published.date() <= start_date.date():
            print(
                f"Skipping result published on {result.published.date()} as it falls outside the selected date range."
            )
            begin_date = start_date - timedelta(days=arxiv_config.getint("date_range"))
            end_date = datetime.now().date()
            print(f"Date range: {begin_date.date()} to {end_date}")
            continue

        paper = {
            "id": result.get_short_id(),
            "title": result.title,
            "arxiv_url": result.entry_id,
            "pdf_url": result.pdf_url,
            "published_date": str(result.published.date()),
            "abstract": result.summary,
        }
        papers.append(paper)

        if result.published.date() > most_recent_date.date():
            most_recent_date = result.published

        # Get embedding for the paper
        paper_text = f"{paper['title']}\n{paper['abstract']}"
        embedding = (
            openai_client.embeddings.create(
                input=paper_text, model=arxiv_config.get("embedding_model")
            )
            .data[0]
            .embedding
        )

        # Add paper to Chroma
        chroma_collection.add(
            ids=[paper["id"]], embeddings=[embedding], metadatas=[paper]
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
            f.write(most_recent_date.strftime("%Y-%m-%d"))

    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper['title']}")

import configparser
import csv
import os
import requests
from configparser import ConfigParser
from datetime import datetime
from typing import List

import numpy as np

from openai import OpenAI

from utils.retry import retry_with_exponential_backoff
from utils.weaviate_client import get_or_create_class, get_weaviate_client
from weaviate.classes.query import MetadataQuery


@retry_with_exponential_backoff
def compute_relevance_score(
    title: str,
    abstract: str,
    include_terms: List[str],
    exclude_terms: List[str],
    config: ConfigParser,
) -> float:
    client = OpenAI(
        api_key=open(config.get("openai", "api_key_location")).read().strip()
    )

    # Combine title and abstract
    paper_text = f"{title}\n{abstract}"

    # Get embeddings for paper, include terms, and exclude terms
    paper_embedding = get_embedding_or_cache(paper_text, client)
    include_embeddings = [
        get_embedding_or_cache(term, client) for term in include_terms
    ]
    exclude_embeddings = [
        get_embedding_or_cache(term, client) for term in exclude_terms
    ]

    # Calculate similarities
    include_similarities = [
        np.dot(paper_embedding, inc_emb) for inc_emb in include_embeddings
    ]
    exclude_similarities = [
        np.dot(paper_embedding, exc_emb) for exc_emb in exclude_embeddings
    ]

    # Compute score
    score = sum(include_similarities) - sum(exclude_similarities)

    return score


def select_top_papers(config: ConfigParser) -> None:
    weaviate_config = config["weaviate"]
    weaviate_client = get_weaviate_client()
    paper_class = get_or_create_class(
        weaviate_client, weaviate_config.get("papers_class_name")
    )
    # Print the number of documents currently indexed in Weaviate
    paper_count = paper_class.aggregate.over_all(total_count=True)

    print(f"Number of documents currently indexed in Weaviate: {paper_count}")

    if paper_count == 0:
        print("Cannot select from an empty collection. Skipping paper selection.")
        return

    queries = config.get("select_papers", "queries").split(",")
    results = []

    for query_name in queries:
        query_name = query_name.strip()
        query_terms = config.get("select_papers", query_name).split(",")
        query_text = " ".join(query_terms)

        # Perform hybrid search
        hybrid_results = paper_class.query.hybrid(
            query=query_text,
            limit=config.getint("select_papers", "number_of_papers_to_summarize"),
        )

        for o in hybrid_results.objects:
            print(o.properties)
            results.append(o.properties)

    output_dir = config.get("select_papers", "output_dir")
    os.makedirs(output_dir, exist_ok=True)

    with open(
        os.path.join(output_dir, "papers_to_summarize.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "ID",
                "Title",
                "ArXivURL",
                "PublishedDate",
                "Abstract",
                "Filename",
            ]
        )

        for paper in results:
            # paper is a dict with the following keys:
            # ['arxiv_url', 'full_text', 'published_date', 'pdf_url', 'arxiv_id', 'title', 'abstract']
            published_date = paper["published_date"].strftime("%Y-%m-%d")

            # potentially rewrite this title to look nicer
            filename = f"{published_date}-{paper['title'].replace(' ', '_').replace(':', '').replace(',', '')[:50]}"

            with open(os.path.join(output_dir, f"{filename}.pdf"), "wb") as f:
                f.write(requests.get(paper["pdf_url"]).content)

            writer.writerow(
                [
                    paper["arxiv_id"],
                    paper["title"],
                    paper["arxiv_url"],
                    paper["published_date"],
                    paper["abstract"],
                    filename,
                ]
            )
            print(f"Downloaded {filename}.pdf")

    print(f"Selected top {len(results)} papers.")
    weaviate_client.close()


def run(config: configparser.ConfigParser) -> None:
    select_top_papers(config)

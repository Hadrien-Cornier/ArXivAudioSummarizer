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
from utils.weaviate_client import get_weaviate_client, get_or_create_class


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
    # Print the number of documents currently indexed in Weaviate
    paper_count = (
        weaviate_client.query.aggregate(weaviate_config.get("papers_class_name"))
        .with_meta_count()
        .do()
    )["data"]["Aggregate"][weaviate_config.get("papers_class_name")][0]["meta"]["count"]

    print(f"Number of documents currently indexed in Weaviate: {paper_count}")

    if paper_count == 0:
        print("Cannot select from an empty collection. Skipping paper selection.")
        return

    queries = config.get("select_papers", "queries").split(",")
    results = []
    openai_client = OpenAI(
        api_key=open(config.get("openai", "api_key_location")).read().strip()
    )

    for query_name in queries:
        query_name = query_name.strip()
        query_terms = config.get("select_papers", query_name).split(",")
        query_text = " ".join(query_terms)

        # Get embedding for the query
        query_embedding = get_embedding_or_cache(query_text, openai_client)

        # Perform hybrid search
        hybrid_results = (
            weaviate_client.query.get(
                weaviate_config.get("papers_class_name"),
                ["id", "title", "arxiv_url", "pdf_url", "published_date", "abstract"],
            )
            .with_hybrid(
                query=query_text,
                alpha=0.5,  # Adjust this value to balance between vector and keyword search
                vector=query_embedding,
            )
            .with_limit(config.getint("select_papers", "number_of_papers_to_summarize"))
            .do()
        )

        results.extend(
            hybrid_results["data"]["Get"][weaviate_config.get("papers_class_name")]
        )

    # Sort results by score (if available in the response)
    top_papers = sorted(
        results, key=lambda x: x.get("_additional", {}).get("score", 0), reverse=True
    )

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
                "ArXiv URL",
                "PDF URL",
                "Published Date",
                "Abstract",
                "Score",
                "Filename",
            ]
        )

        for paper in top_papers:
            published_date = datetime.strptime(
                paper["published_date"], "%Y-%m-%d"
            ).date()
            filename = f"{published_date}-{paper['title'].replace(' ', '_').replace(':', '').replace(',', '')[:50]}"

            with open(os.path.join(output_dir, f"{filename}.pdf"), "wb") as f:
                f.write(requests.get(paper["pdf_url"]).content)

            writer.writerow(list(paper.values()) + [filename])
            print(f"Downloaded {filename}.pdf")

    print(f"Selected top {len(top_papers)} papers.")


def run(config: configparser.ConfigParser) -> None:
    select_top_papers(config)

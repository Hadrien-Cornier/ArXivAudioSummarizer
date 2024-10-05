import csv
import os
import requests
from configparser import ConfigParser
from datetime import datetime
from typing import Any, DefaultDict, List, Dict
from collections import defaultdict

import numpy as np
import json

from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings

from utils.retry import retry_with_exponential_backoff


def load_embedding_cache(cache_file: str) -> Dict[str, Any]:
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_embedding_cache(cache: Dict[str, Any], cache_file: str) -> None:
    with open(cache_file, "w") as f:
        json.dump(cache, f)


@retry_with_exponential_backoff
def get_embedding(
    text: str,
    client: OpenAI,
    model: str = "text-embedding-ada-002",
    cache: Dict[str, Any] = None,
) -> List[float]:
    if cache is not None and text in cache:
        return cache[text]

    response = client.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding

    if cache is not None:
        cache[text] = embedding

    return embedding


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

    # Load embedding caches
    include_cache_file = os.path.join("data", "include_embedding_cache.json")
    exclude_cache_file = os.path.join("data", "exclude_embedding_cache.json")
    include_cache = load_embedding_cache(include_cache_file)
    exclude_cache = load_embedding_cache(exclude_cache_file)

    # Combine title and abstract
    paper_text = f"{title}\n{abstract}"

    # Get embeddings for paper, include terms, and exclude terms
    paper_embedding = get_embedding(paper_text, client)
    include_embeddings = [
        get_embedding(term, client, cache=include_cache) for term in include_terms
    ]
    exclude_embeddings = [
        get_embedding(term, client, cache=exclude_cache) for term in exclude_terms
    ]

    # Save updated caches
    save_embedding_cache(include_cache, include_cache_file)
    save_embedding_cache(exclude_cache, exclude_cache_file)

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
    chroma_client = Client(
        Settings(persist_directory=config.get("arxiv_search", "chroma_persist_dir"))
    )
    chroma_collection = chroma_client.get_collection(name="papers")

    # Print the number of documents currently indexed in Chroma
    paper_count = chroma_collection.count()
    print(f"Number of documents currently indexed in Chroma: {paper_count}")

    if paper_count == 0:
        print("Cannot select from an empty collection. Skipping paper selection.")
        return

    queries = config.get("select_papers", "queries").split(",")
    results: DefaultDict[str, List[Any]] = defaultdict(list)
    openai_client = OpenAI(
        api_key=open(config.get("openai", "api_key_location")).read().strip()
    )

    for query_name in queries:
        query_name = query_name.strip()  # Remove any leading/trailing whitespace
        query_terms = config.get("select_papers", query_name).split(",")

        # Get embeddings for query terms
        query_embeddings = [
            get_embedding(term.strip(), openai_client) for term in query_terms
        ]

        # Query Chroma with vector search
        vector_results = chroma_collection.query(
            query_embeddings=query_embeddings,
            n_results=config.getint("select_papers", "number_of_papers_to_summarize"),
        )

        # Query Chroma with BM25 search
        bm25_results = chroma_collection.query(
            query_texts=[term.strip() for term in query_terms],
            n_results=config.getint("select_papers", "number_of_papers_to_summarize"),
        )

        # Merge the results
        for key in vector_results.keys():
            results[key].append(vector_results[key])
            results[key].append(bm25_results[key])

    print(results)
    # Sort results by combined BM25 and vector score
    top_papers = sorted(query_results, key=lambda x: x["score"], reverse=True)[
        : config.getint("select_papers", "number_of_papers_to_summarize")
    ]

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

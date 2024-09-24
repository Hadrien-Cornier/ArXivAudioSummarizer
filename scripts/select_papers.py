import csv
import os
import requests
from configparser import ConfigParser
from utils.utils import read_papers_from_csv, read_lines_from_file
from datetime import datetime
from typing import List
import openai
import numpy as np
import json

def load_embedding_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_embedding_cache(cache, cache_file):
    with open(cache_file, 'w') as f:
        json.dump(cache, f)

def get_embedding(text, model="text-embedding-ada-002", cache=None):
    if cache is not None and text in cache:
        return cache[text]
    
    embedding = openai.Embedding.create(input=text, model=model)['data'][0]['embedding']
    
    if cache is not None:
        cache[text] = embedding
        
    return embedding

def compute_relevance_score(title: str, abstract: str, include_terms: List[str], exclude_terms: List[str], config: ConfigParser) -> float:
    openai.api_key = open(config.get('openai', 'api_key_location')).read().strip()
    
    # Load embedding caches
    include_cache_file = os.path.join('data', 'include_embedding_cache.json')
    exclude_cache_file = os.path.join('data', 'exclude_embedding_cache.json')
    include_cache = load_embedding_cache(include_cache_file)
    exclude_cache = load_embedding_cache(exclude_cache_file)
    
    # Combine title and abstract
    paper_text = f"{title}\n{abstract}"
    
    # Get embeddings for paper, include terms, and exclude terms
    paper_embedding = get_embedding(paper_text)
    include_embeddings = [get_embedding(term, cache=include_cache) for term in include_terms]
    exclude_embeddings = [get_embedding(term, cache=exclude_cache) for term in exclude_terms]
    
    # Save updated caches
    save_embedding_cache(include_cache, include_cache_file)
    save_embedding_cache(exclude_cache, exclude_cache_file)
    
    # Calculate similarities
    include_similarities = [np.dot(paper_embedding, inc_emb) for inc_emb in include_embeddings]
    exclude_similarities = [np.dot(paper_embedding, exc_emb) for exc_emb in exclude_embeddings]
    
    # Compute score
    score = sum(include_similarities) - sum(exclude_similarities)
    
    return score

def select_top_papers(config: ConfigParser) -> None:
    papers = read_papers_from_csv(config.get('select_papers', 'input_file'))
    include_terms = read_lines_from_file(config.get('arxiv_search', 'include_terms_file', fallback='config/search_terms_include.txt'))
    exclude_terms = read_lines_from_file(config.get('arxiv_search', 'exclude_terms_file', fallback='config/search_terms_exclude.txt'))
    
    for paper in papers:
        paper['Score'] = compute_relevance_score(paper['Title'], paper['Abstract'], include_terms, exclude_terms, config)
    
    top_papers = sorted(papers, key=lambda x: float(x['Score']), reverse=True)[:config.getint('select_papers', 'number_of_papers_to_summarize')]
    
    output_dir = config.get('select_papers', 'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'papers_to_summarize.csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Abstract", "Score", "Filename"])
        
        for paper in top_papers:
            published_date = datetime.strptime(paper['Published Date'], '%Y-%m-%d').date()
            filename = f"{published_date}-{paper['Title'].replace(' ', '_').replace(':', '').replace(',', '')[:50]}"
            
            with open(os.path.join(output_dir, f"{filename}.pdf"), "wb") as f:
                f.write(requests.get(paper['PDF URL']).content)
            
            writer.writerow(list(paper.values()) + [filename])
            print(f"Downloaded {filename}.pdf")
    
    print(f"Selected top {len(top_papers)} papers.")
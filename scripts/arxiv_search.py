import configparser
import arxiv
import os
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.utils import read_lines_from_file

def compute_relevance_score(title: str, abstract: str, include_terms: List[str]) -> int:
    """Compute relevance score based on term occurrences in title and abstract."""
    return sum(2 if term.lower() in title.lower() else 1 if term.lower() in abstract.lower() else 0 for term in include_terms)

def search_papers(config: configparser.ConfigParser) -> None:
    """Search for papers on arXiv based on given configuration."""
    arxiv_config = config['arxiv_search']
    output_dir = arxiv_config.get('output_dir')
    
    restrict_to_most_recent: bool = arxiv_config.getboolean('restrict_to_most_recent')
    max_results: int = arxiv_config.getint('max_results')
    categories: str = arxiv_config.get('categories')
    date_range: int = arxiv_config.getint('date_range')
    include_terms: List[str] = read_lines_from_file(arxiv_config.get('tags_file'))

    start_date: datetime = datetime.now() - timedelta(days=date_range)
    query: str = f"({categories}) AND submittedDate:[{start_date.strftime('%Y%m%d')} TO {datetime.now().strftime('%Y%m%d')}]"

    client: arxiv.Client = arxiv.Client(page_size=max_results, delay_seconds=5.0, num_retries=3)
    search: arxiv.Search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers: List[Dict[str, Any]] = []
    for result in client.results(search):
        if restrict_to_most_recent and (result.published.date() <= start_date.date()):
            with open(os.path.join(output_dir, 'most_recent_day_searched.txt'), 'w') as file:
                file.write(result.published.date().strftime('%Y-%m-%d'))
            break
        
        score: int = compute_relevance_score(result.title, result.summary, include_terms)
        papers.append({
            "id": result.get_short_id(),
            "title": result.title,
            "arxiv_url": result.entry_id,
            "pdf_url": result.pdf_url,
            "published_date": result.published.date(),
            "score": score
        })

    papers.sort(key=lambda x: x['score'], reverse=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'papers_found.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Score"])
        for paper in papers:
            writer.writerow([paper[key] for key in ["id", "title", "arxiv_url", "pdf_url", "published_date", "score"]])

    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper['title']}")
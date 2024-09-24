import configparser
import arxiv
import os
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.utils import read_lines_from_file

def search_papers(config: configparser.ConfigParser) -> None:
    arxiv_config: Dict[str, Any] = config['arxiv_search']
    output_dir: str = arxiv_config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_file: str = os.path.join(output_dir, 'most_recent_day_searched.txt')
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_date = datetime.strptime(f.read().strip(), '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=arxiv_config.getint('date_range'))

    query: str = f"({arxiv_config.get('categories')}) AND submittedDate:[{start_date:%Y%m%d}000000 TO {datetime.now():%Y%m%d}235959]"

    client: arxiv.Client = arxiv.Client(page_size=arxiv_config.getint('max_results'), delay_seconds=5.0, num_retries=3)
    search: arxiv.Search = arxiv.Search(query=query, max_results=arxiv_config.getint('max_results'), sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending)

    include_terms: List[str] = read_lines_from_file(arxiv_config.get('tags_file'))
    papers: List[Dict[str, Any]] = []
    most_recent_date = start_date

    for result in client.results(search):
        if result.published.date() <= start_date.date():
            break
        
        papers.append({
            "id": result.get_short_id(),
            "title": result.title,
            "arxiv_url": result.entry_id,
            "pdf_url": result.pdf_url,
            "published_date": result.published.date(),
            "abstract": result.summary
        })
        
        if result.published.date() > most_recent_date.date():
            most_recent_date = result.published

    if papers:
        with open(os.path.join(output_dir, 'papers_found.csv'), mode='w', newline='', encoding='utf-8') as file:
            writer: csv.writer = csv.writer(file)
            writer.writerow(["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Abstract"])
            writer.writerows([list(paper.values()) for paper in papers])

        # Update the checkpoint file with the most recent date
        with open(checkpoint_file, 'w') as f:
            f.write(most_recent_date.strftime('%Y-%m-%d'))

    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper['title']}")
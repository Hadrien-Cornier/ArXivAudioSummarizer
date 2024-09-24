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
    
    start_date: datetime = datetime.now() - timedelta(days=arxiv_config.getint('date_range'))
    query: str = f"({arxiv_config.get('categories')}) AND submittedDate:[{start_date:%Y%m%d} TO {datetime.now():%Y%m%d}]"

    client: arxiv.Client = arxiv.Client(page_size=arxiv_config.getint('max_results'), delay_seconds=5.0, num_retries=3)
    search: arxiv.Search = arxiv.Search(query=query, max_results=arxiv_config.getint('max_results'), sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending)

    include_terms: List[str] = read_lines_from_file(arxiv_config.get('tags_file'))
    papers: List[Dict[str, Any]] = []
    for result in client.results(search):
        if arxiv_config.getboolean('restrict_to_most_recent') and result.published.date() <= start_date.date():
            with open(os.path.join(output_dir, 'most_recent_day_searched.txt'), 'w') as file:
                file.write(result.published.date().strftime('%Y-%m-%d'))
            break
        
        papers.append({
            "id": result.get_short_id(),
            "title": result.title,
            "arxiv_url": result.entry_id,
            "pdf_url": result.pdf_url,
            "published_date": result.published.date(),
            "abstract": result.summary
        })

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'papers_found.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer: csv.writer = csv.writer(file)
        writer.writerow(["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Abstract"])
        writer.writerows([list(paper.values()) for paper in papers])

    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper['title']}")
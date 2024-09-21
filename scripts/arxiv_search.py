import configparser, arxiv, os, csv
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.utils import read_lines_from_file

def compute_relevance_score(title: str, abstract: str, include_terms: List[str]) -> int:
    return sum(2 if term.lower() in title.lower() else 1 if term.lower() in abstract.lower() else 0 for term in include_terms)

def search_papers(config: configparser.ConfigParser) -> None:
    arxiv_config = config['arxiv_search']
    output_dir = arxiv_config.get('output_dir')
    
    start_date = datetime.now() - timedelta(days=arxiv_config.getint('date_range'))
    query = f"({arxiv_config.get('categories')}) AND submittedDate:[{start_date:%Y%m%d} TO {datetime.now():%Y%m%d}]"

    client = arxiv.Client(page_size=arxiv_config.getint('max_results'), delay_seconds=5.0, num_retries=3)
    search = arxiv.Search(query=query, max_results=arxiv_config.getint('max_results'), sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending)

    include_terms = read_lines_from_file(arxiv_config.get('tags_file'))
    papers = []
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
            "score": compute_relevance_score(result.title, result.summary, include_terms)
        })

    papers.sort(key=lambda x: x['score'], reverse=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'papers_found.csv'), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Score"])
        writer.writerows([paper.values() for paper in papers])

    print(f"Found {len(papers)} papers:")
    for paper in papers:
        print(f"- {paper['title']}")
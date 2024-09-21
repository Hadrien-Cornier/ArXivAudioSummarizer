import csv
import os
import requests
from configparser import ConfigParser
from utils.utils import read_papers_from_csv, compute_relevance_score, read_lines_from_file
from datetime import datetime

def select_top_papers(config: ConfigParser) -> None:
    papers = read_papers_from_csv(config.get('select_papers', 'input_file'))
    include_terms = read_lines_from_file(config.get('arxiv_search', 'include_terms_file', fallback='config/search_terms_include.txt'))
    
    for paper in papers:
        paper['Score'] = compute_relevance_score(paper['Title'], paper.get('Abstract', ''), include_terms)
    
    top_papers = sorted(papers, key=lambda x: int(x['Score']), reverse=True)[:config.getint('select_papers', 'number_of_papers_to_summarize')]
    
    output_dir = config.get('select_papers', 'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'papers_to_summarize.csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Score", "Filename"])
        
        for paper in top_papers:
            published_date = datetime.strptime(paper['Published Date'], '%Y-%m-%d').date()
            filename = f"{published_date}-{paper['Title'].replace(' ', '_').replace(':', '').replace(',', '')[:50]}"
            
            with open(os.path.join(output_dir, f"{filename}.pdf"), "wb") as f:
                f.write(requests.get(paper['PDF URL']).content)
            
            writer.writerow(list(paper.values()) + [filename])
            print(f"Downloaded {filename}.pdf")
    
    print(f"Selected top {len(top_papers)} papers.")
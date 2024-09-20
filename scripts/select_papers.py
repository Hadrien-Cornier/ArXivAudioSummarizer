import csv
import os
import requests
from configparser import ConfigParser
from utils.utils import read_papers_from_csv

def select_top_papers(config: ConfigParser) -> None:
    """Select top papers, download PDFs, and save info to CSV."""
    input_file = config.get('select_papers', 'input_file')
    output_dir = config.get('select_papers', 'output_dir')
    number_of_papers = config.getint('select_papers', 'number_of_papers_to_summarize')
    
    papers = read_papers_from_csv(input_file)
    top_papers = sorted(papers, key=lambda x: int(x['Score']), reverse=True)[:number_of_papers]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for paper in top_papers:
        pdf_path = os.path.join(output_dir, f"{paper['ID']}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(requests.get(paper['PDF URL']).content)
        print(f"Downloaded {paper['ID']}.pdf")
    
    with open(os.path.join(output_dir, 'papers_to_summarize.csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Title", "ArXiv URL", "PDF URL", "Published Date", "Score"])
        writer.writerows([paper.values() for paper in top_papers])
    
    print(f"Selected top {len(top_papers)} papers.")
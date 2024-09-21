import os
import csv
import configparser
from typing import List, Dict, Any, Tuple
import PyPDF2

# Initialize configuration
def resolve_config() -> configparser.ConfigParser:
    config: configparser.ConfigParser = configparser.ConfigParser()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config', 'config.ini')
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return config

# File Operations
def save_file(filepath: str, content: str) -> None:
    """Save content to a file."""
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath: str) -> str:
    """Read and return the content of a file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()

def read_lines_from_file(filename: str) -> List[str]:
    """Read lines from a file and return as a list."""
    lines: List[str] = []
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"File not found: {filename}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return lines

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader: PyPDF2.PdfReader = PyPDF2.PdfReader(file)
            paper: str = ''
            for page_num in range(len(pdf_reader.pages)):
                page: PyPDF2.PageObject = pdf_reader.pages[page_num]
                try:
                    paper += page.extract_text()
                except Exception as e:
                    print(f"Skipping page due to error: {e}")
                    continue
        return paper[:176000] if len(paper) > 176000 else paper
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading file {pdf_path}: {e}")
        return ''

# CSV Operations
def get_link(base_filename: str, csv_path: str) -> str:
    """Get ArXiv URL for a given base filename from CSV."""
    with open(csv_path, mode='r', newline='') as file:
        reader: csv.DictReader = csv.DictReader(file)
        for row in reader:
            if row['ID'] == base_filename:
                return row['ArXiv URL']
    return ''

def read_papers_from_csv(input_file: str) -> List[Dict[str, Any]]:
    """Read paper information from CSV file."""
    papers: List[Dict[str, Any]] = []
    with open(input_file, mode='r', newline='', encoding='utf-8') as file:
        reader: csv.DictReader = csv.DictReader(file)
        papers = [row for row in reader]
    return papers

# Folder Operations
def make_folder_if_none(path: str) -> None:
    """Create a folder if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def delete_all_files_in_folder(folder_path: str) -> None:
    """Delete all files in a given folder."""
    for filename in os.listdir(folder_path):
        file_path: str = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Couldn't delete {filename} file from {folder_path} folder bc Error occurred: \n{e}")

# String Operations
def cut_off_string(input_string: str, cutoff_string: str) -> Tuple[str, str]:
    """Cut off a string at a specified substring."""
    cutoff_index: int = input_string.find(cutoff_string)
    
    if cutoff_index != -1:
        return input_string[:cutoff_index + len(cutoff_string)], input_string[cutoff_index + len(cutoff_string):]
    else:
        return input_string, ''

def compute_relevance_score(title: str, abstract: str, include_terms: List[str]) -> int:
    """Compute relevance score based on term occurrences in title and abstract."""
    return sum(2 if term.lower() in title.lower() else 1 if term.lower() in abstract.lower() else 0 for term in include_terms)
import os
import csv
import configparser
from typing import List, Dict, Any, Tuple, Optional
import PyPDF2
import subprocess
import backoff
import openai
import json

# from marker import Marker


# Initialize configuration
def resolve_config() -> configparser.ConfigParser:
    config: configparser.ConfigParser = configparser.ConfigParser()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config", "config.ini")
    if os.path.exists(config_path):
        config.read(config_path)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return config


# File Operations
def save_file(filepath: str, content: str) -> None:
    """Save content to a file."""
    with open(filepath, "w", encoding="utf-8") as outfile:
        outfile.write(content)


def open_file(filepath: str) -> str:
    """Read and return the content of a file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as infile:
        return infile.read()


def read_lines_from_file(filename: str) -> List[str]:
    """Read lines from a file and return as a list."""
    lines: List[str] = []
    try:
        with open(filename, "r") as file:
            lines = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"File not found: {filename}.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return lines


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file"""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader: PyPDF2.PdfReader = PyPDF2.PdfReader(file)
            paper: str = ""
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
        return ""


# CSV Operations
def get_link(base_filename: str, csv_path: str) -> str:
    """Get ArXiv URL for a given base filename from CSV."""
    with open(csv_path, mode="r", newline="") as file:
        reader: csv.DictReader = csv.DictReader(file)
        for row in reader:
            if row["ID"] == base_filename:
                return row["ArXiv URL"]
    return ""


def read_papers_from_csv(input_file: str) -> List[Dict[str, Any]]:
    """Read paper information from CSV file."""
    papers: List[Dict[str, Any]] = []
    with open(input_file, mode="r", newline="", encoding="utf-8") as file:
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
            print(
                f"Couldn't delete {filename} file from {folder_path} folder bc Error occurred: \n{e}"
            )


# String Operations
def cut_off_string(input_string: str, cutoff_string: str) -> Tuple[str, str]:
    """Cut off a string at a specified substring."""
    cutoff_index: int = input_string.find(cutoff_string)

    if cutoff_index != -1:
        return (
            input_string[: cutoff_index + len(cutoff_string)],
            input_string[cutoff_index + len(cutoff_string) :],
        )
    else:
        return input_string, ""


def compute_relevance_score(title: str, abstract: str, include_terms: List[str]) -> int:
    """Compute relevance score based on term occurrences in title and abstract."""
    return sum(
        (
            2
            if term.lower() in title.lower()
            else 1 if term.lower() in abstract.lower() else 0
        )
        for term in include_terms
    )


def convert_pdfs_to_markdown_with_marker(input_folder: str, output_folder: str) -> None:
    """Convert PDFs in the specified folder to markdown using the marker CLI."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Command to run the marker CLI
    command = [
        "marker",
        input_folder,
        output_folder,
        "--workers",
        "4",  # Adjust the number of workers as needed
        "--max",
        "10",  # Adjust the maximum number of PDFs to convert
    ]

    try:
        # Call the marker CLI
        subprocess.run(command, check=True)
        print(
            f"Successfully converted PDFs in {input_folder} to markdown in {output_folder}."
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while converting PDFs: {e}")


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_batch_responses_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.75,
    n_responses: int = 1,
) -> Tuple[List[str], List[List[Dict[str, str]]]]:
    if msg_history is None:
        msg_history = []

    if model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif "claude" in model:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def get_response_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.75,
) -> Tuple[str, List[Dict[str, str]]]:
    if msg_history is None:
        msg_history = []

    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=3000,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        ]
    elif model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=3000,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> Optional[Dict[str, Any]]:
    json_start_marker = "```json"
    json_end_marker = "```"

    start_index = llm_output.find(json_start_marker)
    if start_index != -1:
        start_index += len(json_start_marker)
        end_index = llm_output.find(json_end_marker, start_index)
    else:
        return None

    if end_index == -1:
        return None

    json_string = llm_output[start_index:end_index].strip()
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError:
        return None


def get_review_model_settings() -> Tuple[str, float]:
    config = resolve_config()
    model = config.get("review", "model", fallback="gpt-4o-2024-08-06")
    temperature = config.getfloat("review", "temperature", fallback=0.75)
    return model, temperature

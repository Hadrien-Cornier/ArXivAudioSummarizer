import os
from typing import List, Dict, Optional
from openai import OpenAI
from utils.utils import read_lines_from_file, get_link, open_file, extract_text_from_pdf
from time import sleep
from configparser import ConfigParser
import time


def summarize_papers(config: ConfigParser) -> None:
    input_folder: str = config.get("summarize_papers", "input_folder")
    output_folder: str = config.get("summarize_papers", "output_folder")
    csv_path: str = config.get(
        "summarize_papers",
        "csv_path",
        fallback="data/pdfs-to-summarize/papers_to_summarize.csv",
    )

    print("Starting summarization process...")
    os.makedirs(output_folder, exist_ok=True)
    pdf_files: List[str] = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files to process")

    summaries: str = ""
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i}/{len(pdf_files)}: {pdf_file}")
        start_time: float = time.time()

        base_filename: str = pdf_file.replace(".pdf", "")
        filename: str = f"{output_folder}/{base_filename}.md"
        if os.path.exists(filename):
            print("File already processed, skipping...")
            continue

        paper: Optional[str] = extract_text_from_pdf(f"{input_folder}/{pdf_file}")
        if paper:
            summaries += (
                f"\n\n\n\n# {base_filename}\n{get_link(base_filename, csv_path)}"
            )
            summaries += generate_summary(paper, config)

        with open(filename, "w", encoding="utf-8") as summary_file:
            summary_file.write(summaries)

        if config.getboolean("Obsidian", "send_to_obsidian", fallback=False):
            try:
                write_to_obsidian(base_filename, paper, summaries, config)
            except Exception as e:
                print(f"Error writing to Obsidian: {e}")

        print(f"Processed in {time.time() - start_time:.2f} seconds")
        print_progress_bar(i, len(pdf_files))

    print("\nAll files processed.")
    with open("data/txt-summaries/newsletter.md", "w", encoding="utf-8") as outfile:
        outfile.write(summaries)


def chatbot(
    conversation: List[Dict[str, str]],
    config: ConfigParser,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    client: OpenAI = OpenAI(
        api_key=open(config.get("openai", "api_key_location")).read().strip()
    )
    for _ in range(3):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=conversation,
                temperature=temperature,
                max_tokens=1500,
                n=1,
                stop=None,
                stream=True,
            )
            return "".join(
                chunk.choices[0].delta.content or "" for chunk in stream
            ).strip()
        except Exception as e:
            sleep(1)
    return f"Error: {str(e)}"


def determine_tags(abstract: str, config: ConfigParser) -> List[str]:
    include_terms: List[str] = read_lines_from_file(
        config.get(
            "arxiv_search",
            "include_terms_file",
            fallback="config/search_terms_include.txt",
        )
    )
    return [term for term in include_terms if term.lower() in abstract.lower()][:10]


def generate_summary(paper: str, config: ConfigParser) -> str:
    all_messages: List[Dict[str, str]] = [{"role": "system", "content": paper}]
    for prompt in config.get("summarize_papers", "prompts").split(","):
        all_messages.append({"role": "user", "content": prompt})
        all_messages.append(
            {"role": "assistant", "content": chatbot(all_messages, config)}
        )

    all_messages.append(
        {
            "role": "user",
            "content": "Synthesize the above information into a concise summary of the paper's key contributions and significance.",
        }
    )
    return chatbot(all_messages, config)


def write_to_obsidian(
    base_filename: str, paper: Optional[str], summaries: str, config: ConfigParser
) -> None:
    tags: List[str] = determine_tags(paper or "", config)
    obsidian_content: str = f"---\ntags: {', '.join(tags)}\n---\n\n{summaries}"

    # Use the vault_attachments_location for writing the Markdown file
    obsidian_attachments_location: str = config.get(
        "Obsidian", "vault_attachments_location"
    )

    # Instead of creating the directory, just ensure it exists
    if not os.path.exists(obsidian_attachments_location):
        print(
            f"Warning: Attachments directory does not exist: {obsidian_attachments_location}"
        )
        return

    obsidian_filename: str = os.path.join(
        obsidian_attachments_location, f"{base_filename}.md"
    )

    # Check if the file already exists
    if os.path.exists(obsidian_filename):
        print(f"Warning: File already exists: {obsidian_filename}")
        return

    try:
        with open(obsidian_filename, "w", encoding="utf-8") as f:
            f.write(obsidian_content)
        print(f"Wrote Obsidian file: {obsidian_filename}")
    except IOError as e:
        print(f"Error writing file {obsidian_filename}: {e}")


def print_progress_bar(current: int, total: int, bar_length: int = 20) -> None:
    progress: float = current / total
    filled_length: int = int(bar_length * progress)
    bar: str = "=" * filled_length + "-" * (bar_length - filled_length)
    print(f"\rProgress: [{bar}] {progress:.0%}", end="")

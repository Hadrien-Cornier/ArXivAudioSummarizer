import os
from typing import List, Dict
from openai import OpenAI
from utils.utils import read_lines_from_file, get_link, open_file, extract_text_from_pdf
from time import sleep
from configparser import ConfigParser
import time

def summarize_papers(config: ConfigParser) -> None:
    """Process PDF files, generate summaries, and save them to output folder and optionally Obsidian."""
    input_folder = config.get('summarize_papers', 'input_folder')
    output_folder = config.get('summarize_papers', 'output_folder')
    
    print("Starting summarization process...")
    print(config)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print("Step 1: Output folder checked/created")

    pdf_files: List[str] = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    print(f"Step 2: Found {len(pdf_files)} PDF files to process")

    summaries: str = ''

    for i, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing file {i+1}/{len(pdf_files)}: {pdf_file}")
        start_time = time.time()

        base_filename: str = pdf_file.replace('.pdf', '')
        link: str = get_link(base_filename, config.get('summarize_papers', 'csv_path', fallback='data/pdfs-to-summarize/papers_to_summarize.csv'))
        summaries += f"\n\n\n\n# {base_filename}\n{link}"

        filename: str = f'{output_folder}/{base_filename}.txt'
        print("Step 3: Preparing output file")

        if os.path.exists(filename):
            print("File already processed, skipping...")
            continue

        print("Step 4: Extracting text from PDF")
        paper: str = extract_text_from_pdf(f'{input_folder}/{pdf_file}')
        if paper:
            print("Step 5: Generating summary")
            summaries += generate_summary(paper, config)

        with open(filename, 'w', encoding='utf-8') as summary_file:
            summary_file.write(summaries)
        print("Step 6: Summary saved to file")

        if config.getboolean('Obsidian', 'send_to_obsidian', fallback=False):
            print("Step 7: Writing to Obsidian")
            write_to_obsidian(base_filename, paper, summaries, config)

        end_time = time.time()
        print(f"Processed in {end_time - start_time:.2f} seconds")

        # Update progress bar
        progress = (i + 1) / len(pdf_files)
        bar_length = 20
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f'\rProgress: [{bar}] {progress:.0%}', end='')

    print("\nAll files processed.")

    with open('data/txt-summaries/newsletter.txt', 'w', encoding='utf-8') as outfile:
        outfile.write(f"{summaries}")
    print("Step 8: Newsletter saved to file")

def chatbot(conversation: List[Dict[str, str]], config: ConfigParser, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """Interact with the OpenAI chatbot to generate responses."""
    max_retry: int = 3
    retry: int = 0
    client = OpenAI(api_key=open_file(config.get('summarize_papers', 'openai_api_key_location')))
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=conversation,
                temperature=temperature,
                max_tokens=1500,
                n=1,
                stop=None
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            retry += 1
            if retry >= max_retry:
                print(f"Error: {e}")
                return "Error: " + str(e)
            sleep(1)

def determine_tags(abstract: str, config: ConfigParser) -> List[str]:
    """Determine tags for a paper based on its abstract."""
    include_terms: List[str] = read_lines_from_file(config.get('arxiv_search', 'include_terms_file', fallback='config/search_terms_include.txt'))
    tags: List[str] = []
    for term in include_terms:
        if term.lower() in abstract.lower() and term not in tags:
            tags.append(term)
        if len(tags) >= 3:
            break
    return tags

def generate_summary(paper: str, config: ConfigParser) -> str:
    """Generate a summary for a paper using the chatbot."""
    all_messages: List[Dict[str, str]] = [{'role': 'system', 'content': paper}]
    prompts = config.get('summarize_papers', 'prompts').split(',')
    
    # Process all prompts and accumulate responses
    for p in prompts:
        all_messages.append({'role': 'user', 'content': p})
        print(f"\nPrompt: {p}")
        print("Chatbot response:")
        answer: str = chatbot(all_messages, config)
        print(answer)
        all_messages.append({'role': 'assistant', 'content': answer})
    
    # Add a final prompt to summarize all previous responses
    final_prompt = "Synthesize the above information into a concise summary of the paper's key contributions and significance."
    all_messages.append({'role': 'user', 'content': final_prompt})
    print(f"\nFinal Prompt: {final_prompt}")
    print("Final summary:")
    final_summary: str = chatbot(all_messages, config)
    print(final_summary)
    
    return final_summary

def write_to_obsidian(base_filename: str, paper: str, summaries: str, config: ConfigParser) -> None:
    """Write the summary to an Obsidian vault."""
    tags: List[str] = determine_tags(paper, config)
    obsidian_content: str = f"---\ntags: {', '.join(tags)}\n---\n\n{summaries}"
    obsidian_vault_location: str = config.get('Obsidian', 'vault_location')
    obsidian_filename: str = os.path.join(obsidian_vault_location, f"{base_filename}.md")
    with open(obsidian_filename, 'w', encoding='utf-8') as f:
        f.write(obsidian_content)
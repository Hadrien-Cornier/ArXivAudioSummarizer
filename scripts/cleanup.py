from configparser import ConfigParser
import os
import shutil
import glob
import csv
from utils.utils import delete_all_files_in_folder, get_link

def update_papers_kept_csv(base_filename: str) -> None:
    downloaded_csv, kept_csv = "papers_downloaded.csv", "papers_kept.csv"
    
    if not os.path.isfile(kept_csv):
        with open(kept_csv, 'w', newline='') as file:
            csv.writer(file).writerow(["Title", "ArXiv Link", "Paper Date", "Date Added"])

    with open(downloaded_csv, 'r', newline='') as file:
        row_to_add = next((row for row in csv.reader(file) if row and row[0] == base_filename), None)
    
    if row_to_add:
        with open(kept_csv, 'a', newline='') as file:
            csv.writer(file).writerow(row_to_add)
        print(f"Added to {kept_csv}: {base_filename}")
    else:
        print(f"Error: Could not find {base_filename} in {downloaded_csv}")

def process_files(pdf_folder: str, md_final_folder: str, pdf_final_folder: str, csv_path: str) -> None:
    count = 0
    for pdf_file in glob.glob(os.path.join(pdf_folder, '*.pdf')):
        count += 1
        base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
        link = get_link(base_filename, csv_path)
        
        md_content = f"{'Link: [' + link + '](' + link + ')' if link else ''}\n\n![[{base_filename}.pdf]]"
        md_file = os.path.join(pdf_folder, f"{base_filename.title()} (pdf).md")
        
        with open(md_file, 'w') as f_out:
            f_out.write(md_content)
        
        for src, dst in [(md_file, md_final_folder), (pdf_file, pdf_final_folder)]:
            try:
                shutil.move(src, dst)
            except shutil.Error as e:
                print(f"Error: {e}. Skipping file {src} because it already exists in the destination.")

        update_papers_kept_csv(base_filename)

    print(f'{count} files added to vault assuming no skip errors')

def cleanup_files(folders_to_clean: list, files_to_remove: list) -> None:
    for folder in folders_to_clean:
        delete_all_files_in_folder(folder)

    for file in files_to_remove:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Couldn't delete {file} due to error: {e}")

def cleanup_and_send_to_obsidian(config: ConfigParser):
    if config.getboolean('Obsidian', 'send_to_obsidian'):
        process_files(
            config.get('select_papers', 'output_dir'),
            config.get('Obsidian', 'vault_location'),
            config.get('Obsidian', 'vault_attachments_location'),
            config.get('summarize_papers', 'csv_path', fallback='data/pdfs-to-summarize/papers_to_summarize.csv')
        )

    cleanup_files(
        [config.get('select_papers', 'output_dir'), config.get('arxiv_search', 'output_dir')],
        ['links.txt', 'timestamps.txt', 'trimmed_timestamps.txt', 
         'timestamps_adjusted.txt', 'newsletter.txt', 'newsletter_podcast.mp3']
    )

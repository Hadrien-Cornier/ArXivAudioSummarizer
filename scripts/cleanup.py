from configparser import ConfigParser
import configparser
import os
import shutil
import glob
from utils.utils import get_link


def process_files(
    pdf_folder: str, md_final_folder: str, pdf_final_folder: str, csv_path: str
) -> None:
    count = 0
    for pdf_file in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        count += 1
        base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
        link = get_link(base_filename, csv_path)

        md_content = f"{'Link: [' + link + '](' + link + ')' if link else ''}\n\n![[{base_filename}.pdf]]"
        md_file = os.path.join(pdf_folder, f"{base_filename.title()} (pdf).md")

        with open(md_file, "w") as f_out:
            f_out.write(md_content)

        for src, dst in [(md_file, md_final_folder), (pdf_file, pdf_final_folder)]:
            try:
                shutil.move(src, dst)
            except shutil.Error as e:
                print(
                    f"Error: {e}. Skipping file {src} because it already exists in the destination."
                )

    print(f"{count} files added to vault assuming no skip errors")


def cleanup_files(
    folders_to_clean: list, files_to_remove: list, files_to_preserve: list
) -> None:
    for folder in folders_to_clean:
        files_in_folder = os.listdir(folder)
        for file in files_in_folder:
            file_path = os.path.join(folder, file)
            if file not in files_to_preserve:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    for file in files_to_remove:
        if file not in files_to_preserve:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Couldn't delete {file} due to error: {e}")


def cleanup_and_send_to_obsidian(config: ConfigParser):
    if config.getboolean("Obsidian", "send_to_obsidian"):
        process_files(
            config.get("select_papers", "output_dir"),
            config.get("Obsidian", "vault_location"),
            config.get("Obsidian", "vault_attachments_location"),
            config.get(
                "summarize_papers",
                "csv_path",
                fallback="data/pdfs-to-summarize/papers_to_summarize.csv",
            ),
        )

    files_to_preserve = ["papers_to_summarize.csv", "most_recent_day_searched.txt"]

    cleanup_files(
        [
            config.get("select_papers", "output_dir"),
            config.get("arxiv_search", "output_dir"),
        ],
        [
            "links.txt",
            "timestamps.txt",
            "trimmed_timestamps.txt",
            "timestamps_adjusted.txt",
            "newsletter.txt",
            "newsletter_podcast.mp3",
        ],
        files_to_preserve,
    )


def run(config: configparser.ConfigParser) -> None:
    cleanup_and_send_to_obsidian(config)

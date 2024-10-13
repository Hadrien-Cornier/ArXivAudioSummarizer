import configparser
import os
import time
from typing import Dict
from utils.utils import delete_all_files_in_folder
from utils.utils import (
    extract_text_from_pdf,
    resolve_config,
    convert_pdfs_to_markdown_with_marker,
)


# extract_text_with_marker
def benchmark_extraction(
    pdf_folder: str, output_folder: str
) -> Dict[str, Dict[str, float]]:
    results = {}

    # Wipe the output folder
    delete_all_files_in_folder(output_folder)

    # benchmark Marker
    start_time = time.time()
    convert_pdfs_to_markdown_with_marker(pdf_folder, output_folder)
    marker_time = time.time() - start_time

    results["marker_time"] = marker_time

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            # benchmark extract_text_from_pdf
            start_time = time.time()
            pypdf_text = extract_text_from_pdf(pdf_path)
            pypdf_time = time.time() - start_time

            # Save extracted texts
            with open(
                os.path.join(output_folder, f"{filename}_pypdf.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(pypdf_text)

            results[filename] = {
                "pypdf_time": pypdf_time,
                "pypdf_length": len(pypdf_text),
            }

    return results


def main():
    config = resolve_config()
    pdf_folder = config.get("benchmark", "pdf_folder")
    output_folder = config.get("benchmark", "output_folder")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = benchmark_extraction(pdf_folder, output_folder)

    total_pypdf_time = sum(result["pypdf_time"] for result in results.values())
    print(f"Total PyPDF2 Time: {total_pypdf_time:.2f}s")
    print(f"Total Marker Time: {results['marker_time']:.2f}s")


def run(config: configparser.ConfigParser) -> None:
    main()


if __name__ == "__main__":
    main()

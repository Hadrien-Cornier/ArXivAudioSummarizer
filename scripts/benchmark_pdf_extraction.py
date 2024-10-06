import os
import time
from typing import Dict
from utils.utils import extract_text_from_pdf, resolve_config


# extract_text_with_marker
def benchmark_extraction(
    pdf_folder: str, output_folder: str
) -> Dict[str, Dict[str, float]]:
    results = {}

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)

            # benchmark extract_text_from_pdf
            start_time = time.time()
            pypdf_text = extract_text_from_pdf(pdf_path)
            pypdf_time = time.time() - start_time

            # benchmark Marker
            start_time = time.time()
            marker_text = extract_text_with_marker(pdf_path)
            marker_time = time.time() - start_time

            # Save extracted texts
            with open(
                os.path.join(output_folder, f"{filename}_pypdf.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(pypdf_text)
            with open(
                os.path.join(output_folder, f"{filename}_marker.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(marker_text)

            results[filename] = {
                "pypdf_time": pypdf_time,
                "marker_time": marker_time,
                "pypdf_length": len(pypdf_text),
                "marker_length": len(marker_text),
            }

    return results


def main():
    config = resolve_config()
    pdf_folder = config.get("benchmark", "pdf_folder")
    output_folder = config.get("benchmark", "output_folder")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    results = benchmark_extraction(pdf_folder, output_folder)

    # Print results
    for filename, data in results.items():
        print(f"File: {filename}")
        print(f"PyPDF2 Time: {data['pypdf_time']:.2f}s, Length: {data['pypdf_length']}")
        print(
            f"Marker Time: {data['marker_time']:.2f}s, Length: {data['marker_length']}"
        )
        print("--------------------")


if __name__ == "__main__":
    main()

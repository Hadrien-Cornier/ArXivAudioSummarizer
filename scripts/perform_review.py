import os
import json
import configparser
from typing import Dict, Any
from openai import OpenAI
from pypdf import PdfReader
import pymupdf
import pymupdf4llm
from utils.utils import (
    extract_text_from_pdf,
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
    get_review_model_settings,
    resolve_config,
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def create_load_prompt() -> callable:
    config = resolve_config()
    prompts_dir = config["review"].get("prompts_dir", "scripts/prompts/review")

    def load_prompt(filename: str) -> str:
        with open(os.path.join(prompts_dir, filename), "r") as file:
            return file.read().strip()

    return load_prompt


load_prompt = create_load_prompt()

reviewer_base_prompt = load_prompt("reviewer_base_prompt.txt")
reviewer_system_prompt_base = load_prompt("reviewer_system_prompt_base.txt")
reviewer_system_prompt_neg = load_prompt("reviewer_system_prompt_neg.txt")
reviewer_system_prompt_pos = load_prompt("reviewer_system_prompt_pos.txt")
reviewer_template_instructions = load_prompt("reviewer_template_instructions.txt")
reviewer_neurips_form = (
    load_prompt("reviewer_neurips_form.txt") + reviewer_template_instructions
)
reviewer_reviews_aggregation = load_prompt("reviewer_reviews_aggregation.txt")
reviewer_reflection_prompt = load_prompt("reviewer_reflection_prompt.txt")
reviewer_meta_system_prompt = load_prompt("reviewer_meta_system_prompt.txt")
reviewer_improvement_prompt = load_prompt("reviewer_improvement_prompt.txt")

del load_prompt


def perform_review(config: configparser.ConfigParser) -> None:
    review_config = config["review"]
    input_folder = review_config.get("input_folder")
    output_folder = review_config.get("output_folder")
    model, temperature = get_review_model_settings()
    num_reflections = review_config.getint("num_reflections", 1)
    num_fs_examples = review_config.getint("num_fs_examples", 1)
    num_reviews_ensemble = review_config.getint("num_reviews_ensemble", 1)

    os.makedirs(output_folder, exist_ok=True)

    client = OpenAI(
        api_key=open(config.get("openai", "api_key_location")).read().strip()
    )

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}_review.json"
            )

            if os.path.exists(output_path):
                print(f"Review for {filename} already exists. Skipping...")
                continue

            print(f"Reviewing {filename}...")
            text = extract_text_from_pdf(pdf_path)

            review = perform_single_review(
                text,
                model,
                client,
                num_reflections,
                num_fs_examples,
                num_reviews_ensemble,
                temperature,
            )

            with open(output_path, "w") as f:
                json.dump(review, f, indent=2)

            print(f"Review for {filename} completed and saved.")

    print("All reviews completed.")


def perform_single_review(
    text: str,
    model: str,
    client: Any,
    num_reflections: int,
    num_fs_examples: int,
    num_reviews_ensemble: int,
    temperature: float,
) -> Dict[str, Any]:
    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples)
        base_prompt = reviewer_neurips_form + fs_prompt
    else:
        base_prompt = reviewer_neurips_form

    base_prompt += reviewer_base_prompt.format(text=text)

    if num_reviews_ensemble > 1:
        llm_review, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt_neg,
            print_debug=False,
            msg_history=msg_history,
            # Higher temperature to encourage diversity.
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_review):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews)

        # take first valid in case meta-reviewer fails
        if review is None:
            review = parsed_reviews[0]

        # Replace numerical scores with the average of the ensemble.
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[1] >= r[score] >= limits[0]:
                    scores.append(r[score])
            review[score] = int(round(np.mean(scores)))

        # Rewrite the message history with the valid one and new aggregated review.
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": reviewer_reviews_aggregation.format(
                    num_reviews_ensemble=num_reviews_ensemble,
                    aggregated_review=json.dumps(review),
                ),
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt_neg,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            print(f"Relection: {j + 2}/{num_reflections}")
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt.format(
                    current_round=j + 2, num_reflections=num_reflections
                ),
                client=client,
                model=model,
                system_message=reviewer_system_prompt_neg,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"

            if "I am done" in text:
                print(f"Review generation converged after {j + 2} iterations.")
                break
    return review


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:
                text = text + page.get_text()
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")

    return text


def load_review(path):
    with open(path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]


fewshot_dir = os.path.join(project_root, "scripts/prompts/review/fewshot_examples")
fewshot_papers = [
    os.path.join(fewshot_dir, "132_automated_relational.pdf"),
    os.path.join(fewshot_dir, "attention.pdf"),
    os.path.join(fewshot_dir, "2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(fewshot_dir, "132_automated_relational.json"),
    os.path.join(fewshot_dir, "attention.json"),
    os.path.join(fewshot_dir, "2_carpe_diem.json"),
]


def get_review_fewshot_examples(num_fs_examples=1):
    fewshot_prompt = "\nBelow are some sample reviews, copied from previous machine learning conferences.\nNote that while each review is formatted differently according to each reviewer's style, the reviews are well-structured and therefore easy to navigate.\n"

    for paper, review in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        txt_path = paper.replace(".pdf", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                paper_text = f.read()
        else:
            paper_text = load_paper(paper)
        review_text = load_review(review)
        fewshot_prompt += (
            f"Paper:\n\n```\n{paper_text}\n```\n\nReview:\n\n```\n{review_text}\n```\n"
        )

    return fewshot_prompt


def get_meta_review(model, client, temperature, reviews):
    # Write a meta-review from a set of individual reviews
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"Review {i + 1}/{len(reviews)}:\n```\n{json.dumps(r)}\n```\n"
    base_prompt = reviewer_neurips_form + review_text

    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=reviewer_meta_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review


def perform_improvement(review, coder):
    improvement_prompt = improvement_prompt.format(review=json.dumps(review))
    coder_out = coder.run(improvement_prompt)

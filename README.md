# ArXiv Paper Summarizer and Podcast Generator

This repository contains a comprehensive pipeline that automates the process of searching for academic papers on arXiv, downloading them, generating summaries, and converting those summaries into an audio format. Here's an overview of the main components:

1. **arXiv Search**: The pipeline begins by searching arXiv for papers based on specified keywords and criteria.

2. **Paper Selection and Download**: From the search results, it selects the top-rated papers and downloads their PDFs.

3. **Summarization**: Using advanced natural language processing techniques, it generates concise summaries of the downloaded papers.

4. **Audio Generation**: Finally, it converts the compiled summaries into an audio format, essentially creating a podcast of the latest research findings.

This automated workflow allows researchers, students, and enthusiasts to stay up-to-date with the latest scientific literature in their field of interest, all while providing the convenience of audio consumption.

The project utilizes Weaviate as a vector database for efficient storage and retrieval of paper information and summaries.

## Running the Pipeline

Before executing the pipeline, you need to set up Weaviate:

1. **Set up Weaviate**:
   - Navigate to the `docker-compose` directory.
   - Rename the `.env-template` file to `.env`.
   - Open the `.env` file and replace `<REPLACE_ME_WITH_YOUR_OPENAI_APIKEY>` with your actual OpenAI API key.
   - Run Weaviate locally using Docker Compose:
     ```
     docker-compose up -d
     ```

To execute the entire workflow as defined in the pipeline configuration:

1. **Ensure Configuration is Set**: 
   - Verify that all necessary settings in `config/config.ini` are correctly configured, including paths and pipeline steps.
   - Set up your search terms in `config/search_terms.txt`. Each line should contain a keyword or phrase you want to search for in arXiv papers.
   - Configure your OpenAI API key in `config/config.ini` under the `[OpenAI]` section.
   - In `config/config.ini`, ensure the following sections are properly set up:
     - `[pipeline]`: Define the order of steps to be executed (e.g., arxiv_search, select_papers, etc.).
     - `[arxiv_search]`: Set input and output directories for the arXiv search process.
     - `[select_papers]`: Configure the number of papers to summarize and related settings.
     - `[summarize_papers]`: Set up input and output folders for the summarization process.
     - `[podcast]`: Define paths for the newsletter text and audio files.
     - `[cleanup]`: Set whether to send results to Obsidian and specify the vault location.
   - If you're using replacements for certain terms, set them up in `config/replacements.txt` with each line in the format: `original_term:replacement_term`.
2. **Activate Virtual Environment**: If not already active, activate your Python virtual environment.
3. **Run the Pipeline**: Execute the main script which will process the steps in the order defined in `config.ini`:
    ```
    python main.py
    ```
4. **Monitor Execution**: The script will print out each step as it executes. Ensure each step completes successfully.
5. **Completion**: Once all steps are executed, you will see "Pipeline execution completed." in the console.

## Acknowledgements

This repository is heavily inspired by [Tunador's arvix-workflow](https://github.com/evintunador).
import os
import json
import fitz  # PyMuPDF
import logging
from core.quiz_engine.quiz_generator import QuizGenerator

# --- Configuration ---
PDF_DIRECTORY = "chapters_to_process"
QUIZ_OUTPUT_DIRECTORY = "core/quiz_content"
QUESTIONS_PER_QUIZ = 5
SUBJECT = "Science"
# ---------------------

logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Uses PyMuPDF to get clean text from a PDF."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        logging.info(f"Extracted {len(full_text)} chars from {pdf_path}")
        return full_text
    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""

def process_all_chapters():
    """
    Loops through the PDF directory, generates quizzes, and saves them.
    """
    logging.info("--- Starting Quiz Generation Pipeline ---")
    
    # We only need one generator instance
    try:
        generator = QuizGenerator()
    except ValueError as e:
        logging.error(f"Failed to start QuizGenerator: {e}")
        return

    for pdf_filename in os.listdir(PDF_DIRECTORY):
        if not pdf_filename.endswith(".pdf"):
            continue
            
        chapter_id = pdf_filename.replace(".pdf", "")
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_filename)
        output_path = os.path.join(QUIZ_OUTPUT_DIRECTORY, f"{chapter_id}.json")

        # Don't re-process files that already have a quiz
        if os.path.exists(output_path):
            logging.warning(f"Skipping {pdf_filename}, quiz already exists.")
            continue

        # 1. Read PDF Text
        chapter_text = extract_text_from_pdf(pdf_path)
        if not chapter_text:
            continue
            
        # 2. Generate Quiz with Gemini
        logging.info(f"Generating quiz for {chapter_id}...")
        quiz_data = generator.generate_quiz_from_text(
            chapter_text, 
            num_questions=QUESTIONS_PER_QUIZ, 
            subject=SUBJECT
        )

        # 3. Save Quiz JSON
        if quiz_data:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(quiz_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully created quiz: {output_path}")
        else:
            logging.error(f"Failed to generate quiz for {chapter_id}.")

if __name__ == "__main__":
    process_all_chapters()
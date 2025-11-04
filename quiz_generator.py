import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Load API keys from .env
load_dotenv()
logger = logging.getLogger(__name__)

class QuizGenerator:
    """
    Uses the Gemini API to generate quiz questions from text.
    """
    def __init__(self):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            logger.error("GEMINI_API_KEY not found in .env file.")
            raise ValueError("GEMINI_API_KEY is not set.")
            
        genai.configure(api_key=gemini_key)
        # NEW LINE:
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        logger.info("QuizGenerator initialized with Gemini model.")

    def _build_prompt(self, chapter_text: str, num_questions: int = 5, subject: str = "STEM"):
        """Creates the detailed prompt for the Gemini API."""
        
        # This prompt is engineered to return clean JSON
        return f"""
        You are an expert quiz designer for a {subject} education app.
        Based on the following chapter text, generate {num_questions} high-quality multiple-choice quiz questions.
        The questions should test the most important concepts from the text.

        RULES:
        1. Return ONLY a valid JSON list.
        2. Do not include any other text, markdown, or explanations outside of the JSON.
        3. Each question object must have these exact keys: "question_text", "options", "correct_answer", "explanation".
        4. "options" must be a list of 4 strings.
        5. "correct_answer" must be one of the strings from the "options" list.
        6. "explanation" should be a brief reason why the answer is correct.

        CHAPTER TEXT:
        ---
        {chapter_text[:8000]} 
        ---
        """
        # Note: Sliced text to [:8000] to respect token limits.
        # You may need to adjust this or send text in chunks.

    def _parse_gemini_response(self, response_text: str) -> list:
        """Safely parses the JSON response from the API."""
        try:
            # Clean up potential markdown formatting
            clean_text = response_text.strip().lstrip("```json").rstrip("```")
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from Gemini: {e}")
            logger.error(f"Raw response was: {response_text}")
            return None

    def generate_quiz_from_text(self, chapter_text: str, num_questions: int = 5, subject: str = "STEM") -> list | None:
        """
        Main function to generate a quiz.
        
        Args:
            chapter_text (str): The text from the PDF chapter.
            num_questions (int): Number of questions to generate.
            subject (str): The subject (e.g., "Physics", "Biology").

        Returns:
            list: A list of quiz question dictionaries, or None if failed.
        """
        if not chapter_text:
            logger.warning("No text provided to QuizGenerator.")
            return None
            
        prompt = self._build_prompt(chapter_text, num_questions, subject)
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_gemini_response(response.text)
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return None
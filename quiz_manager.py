import json
import os
import logging
from ..nlp_engine.tts_service import TextToSpeechService

logger = logging.getLogger(__name__)

class QuizManager:
    """
    Loads pre-generated quiz JSON files and runs the quiz for the student.
    Includes accessibility features like Text-to-Speech.
    """
    def __init__(self, content_path="core/quiz_content/"):
        self.content_path = content_path
        
        # Use your existing TTS service for accessibility
        try:
            self.tts = TextToSpeechService()
            logger.info("QuizManager initialized with TTS service.")
        except ValueError as e:
            logger.warning(f"TTS service failed to load: {e}. QuizManager will run without audio.")
            self.tts = None

    def load_quiz_for_chapter(self, chapter_id: str) -> dict | None:
        """
        Loads the quiz JSON file for a specific chapter.
        
        Args:
            chapter_id (str): The identifier (filename) for the chapter, 
                              e.g., "science_g9_ch1"
        
        Returns:
            dict: The quiz data, or None if not found.
        """
        filepath = os.path.join(self.content_path, f"{chapter_id}.json")
        
        if not os.path.exists(filepath):
            logger.error(f"Quiz file not found: {filepath}")
            return None
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                quiz_data = json.load(f)
            return {"chapter_id": chapter_id, "questions": quiz_data}
        except Exception as e:
            logger.error(f"Failed to load quiz file {filepath}: {e}")
            return None

    def check_answer(self, question: dict, user_answer: str) -> bool:
        """
        Checks if the user's answer is correct.
        (This is a simple check; you'd expand this with memo_service later)
        """
        return user_answer.lower() == question.get("correct_answer", "").lower()

    def speak_question(self, question: dict, language: str = "en-US"):
        """
        Uses the TTS service to read the question and options aloud.
        """
        if not self.tts:
            logger.warning("Speak request ignored: TTS service not available.")
            return

        text_to_speak = question.get("question_text", "")
        if not text_to_speak:
            return

        # Set appropriate voice (you can make this logic smarter)
        voice = "hi-IN-SwaraNeural" if language == "hi-IN" else "en-US-AvaNeural"
        self.tts.set_voice(voice)
        
        # Speak the question
        logger.info(f"Speaking: {text_to_speak}")
        self.tts.synthesize(text_to_speak)
        
        # Speak the options
        option_letter = 'A'
        for opt in question.get("options", []):
            option_text = f"Option {option_letter}: {opt}"
            logger.info(f"Speaking: {option_text}")
            self.tts.synthesize(option_text)
            option_letter = chr(ord(option_letter) + 1)
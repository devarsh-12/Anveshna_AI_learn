import os
import uuid
import logging
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Import your existing modules from nlp_engine
from ..nlp_engine.tts_service import TextToSpeechService
# --- FIX 1: Import the correct classes ---
from ..nlp_engine.adaptive_translator import AdaptiveSTEMTranslator, TranslationConfig, Language

load_dotenv()
logger = logging.getLogger(__name__)

# --- FIX 2: Add a helper to map language codes to your Enums ---
# This maps simple language codes (like 'hi') to the Enums
# your translator class requires.
LANGUAGE_MAP = {
    "hi": Language.HINDI,
    "bn": Language.BENGALI,
    "ta": Language.TAMIL,
    "te": Language.TELUGU,
    "mr": Language.MARATHI,
    "gu": Language.GUJARATI,
    "kn": Language.KANNADA,
    "ml": Language.MALAYALAM,
    "pa": Language.PUNJABI,
    "ur": Language.URDU,
}

class ContextExplainer:
    """
    Handles on-demand explanation and translation of selected text.
    Integrates Gemini, AdaptiveSTEMTranslator, and TextToSpeechService.
    """
    
    def __init__(self, audio_output_dir="static/audio/"):
        # 1. Init Gemini Model
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        genai.configure(api_key=gemini_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.gemini_api_key = gemini_key # Save key for translator

        # 2. Init your existing TTS Service
        try:
            self.tts = TextToSpeechService()
        except Exception as e:
            logger.error(f"Failed to initialize TextToSpeechService: {e}")
            self.tts = None
            
        # 3. Define where to save temporary audio files
        self.audio_output_dir = audio_output_dir
        if not os.path.exists(self.audio_output_dir):
            os.makedirs(self.audio_output_dir)
        
        # NOTE: We no longer initialize the translator here, 
        # as we need to configure it per-language.
            
        logger.info("ContextExplainer initialized successfully.")

    def simplify_and_translate(self, text: str, target_language: str) -> str | None:
        """
        ACTION 1: Get a more understandable translation.
        This will now use your AdaptiveSTEMTranslator.
        """
        
        # --- FIX 3: Initialize and use AdaptiveSTEMTranslator correctly ---
        try:
            # 1. Find the Language Enum
            lang_enum = LANGUAGE_MAP.get(target_language)
            if not lang_enum:
                logger.error(f"Unsupported target language: {target_language}")
                return None

            # 2. Create a config for the translator
            # This uses the built-in Gemini refinement in your translator
            config = TranslationConfig(
                model_name="Helsinki-NLP/opus-mt-en-hi",
                beam_size=4,
                max_length=512,
                target_language=lang_enum,
                grade_level=8, # Use a default grade
                use_gemini=True, # We want the translator to use Gemini
                gemini_api_key=self.gemini_api_key,
                glossary_base_dir=Path("glossaries"),
                reset_glossary=False,
                cultural_adaptation=True,
                simplify_for_grade=True,
                show_comparison=False
            )
            
            # 3. Initialize the translator with the config
            translator = AdaptiveSTEMTranslator(config)
            
            # 4. Translate and get the result object
            result = translator.translate(text)
            
            # 5. Return the final translation string
            return result.final_translation

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None

    def explain_and_get_audio(self, text: str, subject: str = "Science") -> str | None:
        """
        ACTION 2: Explain the text with audio.
        (This function remains the same as it was correct)
        """
        if not self.tts:
            logger.error("TTS service is not available for this action.")
            return None

        # 1. Generate explanation with Gemini
        prompt = f"""
        You are an expert {subject} teacher. A student has selected a term
        and wants to understand it better.
        
        Explain the following concept in 2-3 simple sentences, as if 
        you are talking to a 10th grader. Do not just rephrase it; 
        explain what it is and why it's important.
        
        CONCEPT: "{text}"
        
        EXPLANATION:
        """
        try:
            response = self.model.generate_content(prompt)
            explanation_text = response.text.strip()
            logger.info(f"Gemini Explanation: {explanation_text}")
        except Exception as e:
            logger.error(f"Gemini explanation generation failed: {e}")
            return None

        # 2. Convert explanation to audio
        try:
            audio_filename = f"{uuid.uuid4()}.wav"
            output_path = os.path.join(self.audio_output_dir, audio_filename)
            
            self.tts.set_voice("en-US-AvaNeural") 
            success = self.tts.synthesize(explanation_text, output_filename=output_path)

            if success:
                web_path = f"/static/audio/{audio_filename}"
                logger.info(f"Audio file saved. Returning web path: {web_path}")
                return web_path
            else:
                logger.error("TTS synthesis failed to save file.")
                return None
        except Exception as e:
            logger.error(f"TTS synthesis process failed: {e}")
            return None
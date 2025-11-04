import logging
import os
# This ensures your file paths work on any OS
from pathlib import Path 
from core.comprehension_engine.context_explainer import ContextExplainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use Path for robust directory handling
AUDIO_DIR = Path("static/audio")

if not AUDIO_DIR.exists():
    logger.warning(f"Creating directory: {AUDIO_DIR}")
    AUDIO_DIR.mkdir(parents=True)

def run_test():
    logger.info("--- 🧪 TESTING COMPREHENSION ENGINE 🧪 ---")
    
    try:
        # Initialize the engine
        explainer = ContextExplainer(audio_output_dir=str(AUDIO_DIR))
    except Exception as e:
        logger.error(f"Failed to initialize ContextExplainer: {e}")
        logger.error("Please check all your API keys in the .env file.")
        return

    # --- Test 1: Simplify and Translate ---
    translated_text = None # Define here to use in Test 1b
    try:
        logger.info("\n--- Test 1: Simplify and Translate ---")
        complex_text = "Photosynthesis is the metabolic process by which green plants and some other organisms synthesize nutrients from carbon dioxide and water, utilizing light energy."
        target_lang = "hi" # Test with Hindi
        
        logger.info(f"Input Text: {complex_text}")
        logger.info(f"Target Language: {target_lang}")
        
        translated_text = explainer.simplify_and_translate(complex_text, target_lang)
        
        if translated_text:
            logger.info("✅ Test 1 Success!")
            logger.info(f"Simplified/Translated Result: {translated_text}")
        else:
            logger.error("❌ Test 1 FAILED: Received no translated text.")
            
    except Exception as e:
        logger.error(f"❌ Test 1 FAILED with error: {e}")

    # --- [NEW] Test 1b: Speak Hindi Translation ---
    # This new test runs only if Test 1 succeeded
    if translated_text and explainer.tts:
        try:
            logger.info("\n--- Test 1b: Speak Hindi Translation ---")
            hindi_voice = "hi-IN-SwaraNeural"
            output_file = AUDIO_DIR / "hindi_translation.wav" # Use Path to join

            logger.info(f"Speaking Hindi text to {output_file}...")
            
            # Set the voice on the tts service (which is inside the explainer)
            explainer.tts.set_voice(hindi_voice)
            
            # Build the SSML string for robust Hindi synthesis
            ssml_for_hindi = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='hi-IN'>
                <voice name='{hindi_voice}'>
                    {translated_text}
                </voice>
            </speak>
            """
            
            # Call the synthesize_ssml method
            success = explainer.tts.synthesize_ssml(ssml_for_hindi, output_filename=str(output_file))
            
            if success:
                logger.info(f"✅ Test 1b Success! File '{output_file}' created.")
            else:
                logger.error("❌ Test 1b FAILED: SSML synthesis failed.")

        except Exception as e:
            logger.error(f"❌ Test 1b FAILED with error: {e}")
    elif not explainer.tts:
            logger.warning("Skipping Test 1b: TTS service not available.")
    
    # --- Test 2: Explain and Get Audio (English) ---
    try:
        logger.info("\n--- Test 2: Explain and Get Audio (English) ---")
        concept_text = "Mitochondria"
        
        logger.info(f"Input Concept: {concept_text}")
        
        # This function returns a web path like "/static/audio/file.wav"
        audio_url = explainer.explain_and_get_audio(concept_text)
        
        if audio_url:
            # Check if the file was *actually* created
            local_file_path = Path(audio_url.lstrip("/"))
            
            if local_file_path.exists():
                logger.info("✅ Test 2 Success!")
                logger.info(f"Web URL: {audio_url}")
                logger.info(f"Local File: {local_file_path}")
            else:
                logger.error(f"❌ Test 2 FAILED: File not found at {local_file_path}")
        else:
            logger.error("❌ Test 2 FAILED: No audio URL was returned.")
            
    except Exception as e:
        logger.error(f"❌ Test 2 FAILED with error: {e}")

    logger.info("\n--- ✅ COMPREHENSION TEST COMPLETE ---")
    logger.info("Check your 'static/audio' folder for the new audio files.")

if __name__ == "__main__":
    run_test()
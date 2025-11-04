# -*- coding: utf-8 -*-
import logging
from core.nlp_engine.tts_service import TextToSpeechService

logging.basicConfig(level=logging.INFO)
print("--- Starting Robust TTS Synthesis Test ---")

try:
    tts = TextToSpeechService()

    # --- Test 1: Hindi (Using new SSML method) ---
    print("\nSynthesizing Hindi (SSML)...")
    
    # We must match the voice name with the language in the SSML
    hindi_voice = "hi-IN-SwaraNeural"
    tts.set_voice(hindi_voice)
    
    hindi_text = "नमस्ते, यह एक परीक्षण है।"
    
    # This is the SSML string. It explicitly defines the language.
    ssml_for_hindi = f"""
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='hi-IN'>
        <voice name='{hindi_voice}'>
            {hindi_text}
        </voice>
    </speak>
    """
    
    # Call the new SSML method
    tts.synthesize_ssml(ssml_for_hindi, output_filename="hindi_test.wav")

    # --- Test 2: English (Using original method) ---
    # The original method is fine for English
    print("\nSynthesizing English (standard)...")
    
    tts.set_voice("en-US-AvaNeural")
    english_text = "This is a test in English. It should work perfectly."
    tts.synthesize(english_text, output_filename="english_test.wav")
    
    print("\n--- Test complete! ---")
    print("Check 'hindi_test.wav' and 'english_test.wav'")

except Exception as e:
    print(f"\n--- An Error Occurred ---")
    print(f"Error: {e}")
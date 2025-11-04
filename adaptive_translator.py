#!/usr/bin/env python3
"""
adaptive_translator.py

Enhanced Adaptive STEM Translator with side-by-side comparison of NLLB and Gemini translations.
"""

import argparse
import logging
import os  # ← Make sure this import is present
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple



import nltk
import torch
from textstat import flesch_kincaid_grade
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.download("punkt", quiet=True)
except:
    logger.warning("NLTK punkt download failed")

class Language(Enum):
    HINDI = "hin_Deva"
    BENGALI = "ben_Beng"
    TAMIL = "tam_Taml"
    TELUGU = "tel_Telu"
    MARATHI = "mar_Deva"
    GUJARATI = "guj_Gujr"
    KANNADA = "kan_Knda"
    MALAYALAM = "mal_Mlym"
    PUNJABI = "pan_Guru"
    URDU = "urd_Arab"

@dataclass
class TranslationConfig:
    model_name: str
    beam_size: int
    max_length: int
    target_language: Language
    grade_level: int
    use_gemini: bool
    gemini_api_key: Optional[str]
    glossary_base_dir: Path
    reset_glossary: bool = False
    cultural_adaptation: bool = True
    simplify_for_grade: bool = True
    show_comparison: bool = True  # New option to show comparison

@dataclass
class TranslationResult:
    original_text: str
    raw_nllb_translation: str  # Changed from raw_translation
    glossary_applied_translation: str  # New field
    gemini_refined_translation: str  # New field
    final_translation: str
    confidence_score: float
    grade_level: float
    domain: str
    terms: List[str]
    time_taken: float
    cultural_adaptations: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class Domain(Enum):
    MATHEMATICS = "mathematics"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    ECONOMICS = "economics"
    AGRICULTURE = "agriculture"
    BIOLOGY = "biology"
    GENERAL = "general"

DOMAIN_KEYWORDS = {
    Domain.MATHEMATICS: ["algebra", "geometry", "calculus", "equation", "function", "derivative", "integral", "theorem"],
    Domain.CHEMISTRY: ["element", "compound", "reaction", "molecule", "acid", "base", "atom", "bond"],
    Domain.PHYSICS: ["motion", "force", "energy", "velocity", "acceleration", "momentum", "gravity", "wave"],
    Domain.ECONOMICS: ["gdp", "inflation", "market", "trade", "investment", "demand", "supply", "currency"],
    Domain.AGRICULTURE: ["crop", "soil", "harvest", "irrigation", "livestock", "fertilizer", "yield"],
    Domain.BIOLOGY: ["cell", "organism", "dna", "evolution", "ecosystem", "photosynthesis", "respiration"],
}

class MultiSubjectGlossaryManager:
    """Manages multiple subject-specific glossaries."""
    
    def __init__(self, glossary_base_dir: Path, reset: bool = False):
        self.glossary_base_dir = glossary_base_dir
        self.glossary_base_dir.mkdir(parents=True, exist_ok=True)
        self.connections = {}
        self._initialize_glossaries()

    def _initialize_glossaries(self):
        """Initialize all subject-specific glossaries."""
        glossary_files = {
            'economics': 'economics_glossary.db',
            'mathematics': 'maths_glossary.db', 
            'chemistry': 'chemistry_glossary.db',
            'agriculture': 'agriculture_glossary.db',
            'physics': 'physics_glossary.db',
            'biology': 'biology_glossary.db'
        }
        
        for subject, filename in glossary_files.items():
            db_path = self.glossary_base_dir / filename
            self.connections[subject] = sqlite3.connect(db_path, check_same_thread=False)
            self._create_table(self.connections[subject], subject)
            
            if self._is_database_empty(self.connections[subject], subject):
                self._load_sample_data(self.connections[subject], subject)
                logger.info(f"Loaded sample data for {subject} glossary")

    def _create_table(self, conn, subject: str):
        """Create table for a specific subject glossary."""
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {subject}_glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                english TEXT NOT NULL,
                lang TEXT NOT NULL,
                term TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                UNIQUE(english, lang)
            )
        """)
        conn.commit()

    def _is_database_empty(self, conn, subject: str) -> bool:
        """Check if glossary database is empty."""
        cursor = conn.execute(f"SELECT COUNT(*) FROM {subject}_glossary")
        return cursor.fetchone()[0] == 0

    def _load_sample_data(self, conn, subject: str):
        """Load sample data for a subject glossary."""
        sample_data = {
            'economics': [
                ("GDP", Language.HINDI.value, "सकल घरेलू उत्पाद"),
                ("inflation", Language.HINDI.value, "मुद्रास्फीति"),
                ("investment", Language.HINDI.value, "निवेश"),
                ("export", Language.HINDI.value, "निर्यात"),
            ],
            'mathematics': [
                ("equation", Language.HINDI.value, "समीकरण"),
                ("function", Language.HINDI.value, "फलन"),
                ("derivative", Language.HINDI.value, "अवकलज"),
                ("integral", Language.HINDI.value, "समाकलन"),
            ],
            'physics': [
                ("force", Language.HINDI.value, "बल"),
                ("energy", Language.HINDI.value, "ऊर्जा"),
                ("velocity", Language.HINDI.value, "वेग"),
                ("acceleration", Language.HINDI.value, "त्वरण"),
            ],
            'chemistry': [
                ("element", Language.HINDI.value, "तत्व"),
                ("compound", Language.HINDI.value, "यौगिक"),
                ("reaction", Language.HINDI.value, "अभिक्रिया"),
                ("molecule", Language.HINDI.value, "अणु"),
            ],
            'agriculture': [
                ("crop", Language.HINDI.value, "फसल"),
                ("soil", Language.HINDI.value, "मिट्टी"),
                ("irrigation", Language.HINDI.value, "सिंचाई"),
                ("harvest", Language.HINDI.value, "फसल कटाई"),
            ],
            'biology': [
                ("cell", Language.HINDI.value, "कोशिका"),
                ("organism", Language.HINDI.value, "जीव"),
                ("dna", Language.HINDI.value, "डीएनए"),
                ("evolution", Language.HINDI.value, "विकास"),
            ]
        }
        
        if subject in sample_data:
            conn.executemany(
                f"INSERT INTO {subject}_glossary (english, lang, term) VALUES (?, ?, ?)",
                [(e.lower(), l, t) for e, l, t in sample_data[subject]]
            )
            conn.commit()

    def get_term(self, word: str, lang: Language, subject: str) -> Optional[str]:
        """Get translated term from subject-specific glossary."""
        if subject not in self.connections:
            return None
            
        try:
            cursor = self.connections[subject].execute(
                f"SELECT term FROM {subject}_glossary WHERE english = ? AND lang = ? LIMIT 1",
                (word.lower(), lang.value)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Error querying {subject} glossary: {e}")
            return None

class TranslatorModel:
    """Wraps NLLB model for translation."""
    
    def __init__(self, model_name: str, device: str, beam_size: int, max_length: int):
        logger.info(f"Loading model {model_name} on {device}")
        try:
            # --- START OF FIX ---
            # Store the model_name to check what kind it is
            self.model_name = model_name
            # --- END OF FIX ---
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model = self.model.to(device)
            self.device = device
            self.beam_size = beam_size
            self.max_length = max_length
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def translate(self, text: str, tgt: Language) -> str:
        """Translate text to target language."""
        try:
            # --- START OF FIX ---
            # The Helsinki model doesn't use (or need) src_lang
            if "nllb" in self.model_name.lower():
                self.tokenizer.src_lang = "eng_Latn"
            # --- END OF FIX ---
            
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
            ).to(self.device)
            
            # --- START OF FIX ---
            # Build generation arguments
            # This is the generic way that works for Helsinki
            gen_kwargs = {
                "num_beams": self.beam_size,
                "max_length": self.max_length,
                "early_stopping": True
            }
            
            # If we're using an NLLB model, add the special token
            if "nllb" in self.model_name.lower():
                gen_kwargs['forced_bos_token_id'] = self.tokenizer.lang_code_to_id[tgt.value]
            
            generated_tokens = self.model.generate(**inputs, **gen_kwargs)
            # --- END OF FIX ---
            
            return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

class GeminiRefiner:
    """Uses Gemini API for refinement."""
    
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package not available")
        
        if not api_key:
            raise ValueError("Gemini API key is required")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        logger.info("Gemini refiner initialized")

    def refine(self, original_english: str, nllb_translation: str, glossary_applied: str, 
               terms: List[str], domain: str, grade_level: int, lang: Language) -> str:
        """Refine translation using Gemini with detailed comparison context."""
        try:
            glossary_ctx = "\n".join(f"- {t}" for t in terms) if terms else "None"
            
            prompt = f"""
You are an expert educational translator specializing in STEM content for rural Indian students.

COMPARISON TASK:
I will provide you with:
1. Original English text
2. Raw NLLB model translation
3. Glossary-applied translation (technical terms replaced)

Your task is to create a FINAL REFINED VERSION that:

IMPROVEMENTS NEEDED:
- Use simpler, more natural language appropriate for grade {grade_level} students
- Incorporate rural Indian cultural context and relatable examples
- Ensure technical accuracy while making concepts accessible
- Improve flow and readability
- Maintain consistency with glossary terms

ORIGINAL ENGLISH: {original_english}

RAW NLLB TRANSLATION: {nllb_translation}

GLOSSARY-APPLIED TRANSLATION: {glossary_applied}

DOMAIN: {domain}
TARGET LANGUAGE: {lang.name}
TECHNICAL TERMS (must preserve): {glossary_ctx}

RURAL CONTEXT GUIDELINES:
- Use examples from farming, local markets, daily village life
- Simplify complex sentences
- Avoid academic jargon when possible
- Make it sound natural and conversational

Provide ONLY the final refined translation without any explanations:
"""
            
            response = self.model.generate_content(prompt)
            refined_text = response.text.strip()
            
            return refined_text if refined_text else glossary_applied
            
        except Exception as e:
            logger.error(f"Gemini refinement error: {e}")
            return glossary_applied

class AdaptiveSTEMTranslator:
    """Main translation pipeline with comparison feature."""
    
    def __init__(self, cfg: TranslationConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = TranslatorModel(
            cfg.model_name, self.device, cfg.beam_size, cfg.max_length
        )
        self.glossary = MultiSubjectGlossaryManager(cfg.glossary_base_dir)
        
        # Initialize Gemini refiner if API key is available
        self.gemini_refiner = None
        if cfg.use_gemini and cfg.gemini_api_key:
            if GEMINI_AVAILABLE:
                try:
                    self.gemini_refiner = GeminiRefiner(cfg.gemini_api_key)
                    logger.info("Gemini refiner initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini: {e}")
            else:
                logger.warning("Gemini package not available")

    def _detect_domain(self, text: str) -> Domain:
        """Detect the subject domain of the text."""
        scores = defaultdict(int)
        text_lower = text.lower()
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                    scores[domain] += 1
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return Domain.GENERAL

    def _extract_terms(self, text: str, domain: Domain) -> List[str]:
        """Extract technical terms from text."""
        found_terms = []
        text_lower = text.lower()
        
        for keyword in DOMAIN_KEYWORDS[domain]:
            if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
                found_terms.append(keyword)
        
        return found_terms

    def _apply_glossary_terms(self, translation: str, terms: List[str], domain: Domain) -> Tuple[str, List[str]]:
        """Apply glossary terms and track changes."""
        result = translation
        adaptations = []
        
        for term in terms:
            glossary_term = self.glossary.get_term(term, self.cfg.target_language, domain.value)
            if glossary_term:
                # Count occurrences to show what was replaced
                pattern = rf'\b{re.escape(term)}\b'
                before_count = len(re.findall(pattern, result, flags=re.IGNORECASE))
                result = re.sub(pattern, glossary_term, result, flags=re.IGNORECASE)
                after_count = len(re.findall(pattern, result, flags=re.IGNORECASE))
                
                if before_count > after_count:
                    adaptations.append(f"Glossary: '{term}' → '{glossary_term}'")
        
        return result, adaptations

    def translate(self, text: str) -> TranslationResult:
        """Main translation pipeline with detailed comparison."""
        start_time = time.time()
        errors = []
        cultural_adaptations = []
        
        if not text.strip():
            return TranslationResult(text, "", "", "", "", 0.0, 0.0, Domain.GENERAL.value, [], 0.0)
        
        try:
            # Step 1: Domain detection
            domain = self._detect_domain(text)
            logger.info(f"Detected domain: {domain.value}")
            
            # Step 2: Term extraction
            terms = self._extract_terms(text, domain)
            logger.info(f"Extracted {len(terms)} technical terms: {terms}")
            
            # Step 3: Raw NLLB translation
            raw_nllb_translation = self.model.translate(text, self.cfg.target_language)
            
            # Step 4: Apply glossary terms
            glossary_applied_translation, glossary_adaptations = self._apply_glossary_terms(
                raw_nllb_translation, terms, domain
            )
            cultural_adaptations.extend(glossary_adaptations)
            
            # Step 5: Gemini refinement (if available)
            gemini_refined_translation = glossary_applied_translation  # Default to glossary version
            
            if self.gemini_refiner:
                try:
                    gemini_refined_translation = self.gemini_refiner.refine(
                        text, raw_nllb_translation, glossary_applied_translation,
                        terms, domain.value, self.cfg.grade_level, self.cfg.target_language
                    )
                    cultural_adaptations.append("Gemini refinement applied")
                except Exception as e:
                    errors.append(f"Gemini refinement failed: {e}")
                    logger.error(f"Gemini refinement error: {e}")
            
            # Step 6: Determine final translation
            final_translation = (
                gemini_refined_translation if self.gemini_refiner 
                else glossary_applied_translation
            )
            
            # Step 7: Calculate metrics
            try:
                grade_level = flesch_kincaid_grade(final_translation)
            except:
                grade_level = self.cfg.grade_level
            
            # Confidence based on processing steps
            confidence = 0.7  # Base
            confidence += 0.1 if terms else 0
            confidence += 0.1 if glossary_adaptations else 0
            confidence += 0.1 if self.gemini_refiner else 0
            
            return TranslationResult(
                original_text=text,
                raw_nllb_translation=raw_nllb_translation,
                glossary_applied_translation=glossary_applied_translation,
                gemini_refined_translation=gemini_refined_translation,
                final_translation=final_translation,
                confidence_score=min(confidence, 1.0),
                grade_level=grade_level,
                domain=domain.value,
                terms=terms,
                time_taken=time.time() - start_time,
                cultural_adaptations=cultural_adaptations,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Translation pipeline error: {e}")
            errors.append(str(e))
            return TranslationResult(
                text, "", "", "", "", 0.0, 0.0, Domain.GENERAL.value, [], 
                time.time() - start_time, errors=errors
            )

def display_comparison(result: TranslationResult, config: TranslationConfig):
    """Display detailed comparison of translation stages."""
    print("\n" + "="*80)
    print("🧪 TRANSLATION COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n📖 ORIGINAL ENGLISH:")
    print(f"   {result.original_text}")
    
    print(f"\n🔤 DETECTED DOMAIN: {result.domain.upper()}")
    print(f"   Technical terms: {', '.join(result.terms) if result.terms else 'None'}")
    
    print(f"\n1️⃣ RAW NLLB TRANSLATION:")
    print(f"   {result.raw_nllb_translation}")
    
    print(f"\n2️⃣ GLOSSARY-APPLIED TRANSLATION:")
    print(f"   {result.glossary_applied_translation}")
    
    if hasattr(result, 'gemini_refined_translation') and result.gemini_refined_translation != result.glossary_applied_translation:
        print(f"\n3️⃣ GEMINI-REFINED TRANSLATION:")
        print(f"   {result.gemini_refined_translation}")
    
    print(f"\n🎯 FINAL TRANSLATION:")
    print(f"   {result.final_translation}")
    
    print(f"\n📊 QUALITY METRICS:")
    print(f"   Confidence: {result.confidence_score:.2f}")
    print(f"   Grade Level: {result.grade_level:.1f}")
    print(f"   Processing Time: {result.time_taken:.2f}s")
    
    if result.cultural_adaptations:
        print(f"\n🔧 ADAPTATIONS APPLIED:")
        for adaptation in result.cultural_adaptations:
            print(f"   ✓ {adaptation}")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for error in result.errors:
            print(f"   - {error}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Adaptive STEM Translator with Comparison")
    
    parser.add_argument("text", nargs="+", help="Text to translate")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--lang", type=str, required=True, choices=[lang.value for lang in Language])
    parser.add_argument("--grade", type=int, default=8)
    parser.add_argument("--gemini", action="store_true", help="Use Gemini refinement")
    parser.add_argument("--key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API key")
    parser.add_argument("--gloss-dir", default="glossaries", help="Glossary directory")
    parser.add_argument("--no-comparison", action="store_false", dest="comparison", 
                       help="Disable detailed comparison")
    
    args = parser.parse_args()
    
    # Convert language string to Enum
    try:
        target_lang = next(lang for lang in Language if lang.value == args.lang)
    except StopIteration:
        print(f"Error: Invalid language code {args.lang}")
        sys.exit(1)
    
    # Check Gemini availability
    if args.gemini and not args.key:
        print("Warning: Gemini requested but no API key provided. Use --key or set GEMINI_API_KEY env var.")
        args.gemini = False
    
    config = TranslationConfig(
        model_name=args.model,
        beam_size=args.beam,
        max_length=args.maxlen,
        target_language=target_lang,
        grade_level=args.grade,
        use_gemini=args.gemini,
        gemini_api_key=args.key,
        glossary_base_dir=Path(args.gloss_dir),
        show_comparison=args.comparison
    )
    
    try:
        translator = AdaptiveSTEMTranslator(config)
    except Exception as e:
        print(f"Error initializing translator: {e}")
        sys.exit(1)
    
    input_text = " ".join(args.text)
    result = translator.translate(input_text)
    
    if config.show_comparison:
        display_comparison(result, config)
    else:
        # Simple output
        print(f"\nFinal Translation: {result.final_translation}")

if __name__ == "__main__":
    main()
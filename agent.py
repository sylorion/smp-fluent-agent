"""
Services CEO wants to provide a new service to the customers through its Services Platform technically called "SMP". 
1. The service is called "Fluent Agent". 
2. The service is a translation service that can translate documents using various models and optical character recognition. 
3. The service should be able to detect the language of the document and translate it to the desired language. 
4. The service should also be able to handle various document formats such as PDF, text, and images. 
5. The service should be able to preserve the formatting and layout of the document during translation. 
6. The service should also be able to handle multiple translation requests and provide statistics about the translation operations.
This module provides functionality for translating documents
using various models and optical character recognition.
"""
import re
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import pytesseract
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.units import inch
from reportlab.lib.colors import Color

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('translation_agent.log')
    ]
)

logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Enumeration of supported document types"""
    PDF = "pdf"
    TEXT = "txt"
    IMAGE = "image"
    UNKNOWN = "unknown"

@dataclass
class TranslationRequest:
    """Data class for translation requests"""
    source_path: str
    target_language: str
    source_language: Optional[str] = None
    formatting_options: Optional[Dict] = None
    
    def __post_init__(self):
        if not self.formatting_options:
            self.formatting_options = {}
        if not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

class TranslationResult:
    """Class to hold translation results and metadata"""
    def __init__(self, 
                 original_text: str,
                 translated_text: str,
                 source_language: str,
                 target_language: str,
                 confidence: float,
                 processing_time: float):
        self.original_text = original_text
        self.translated_text = translated_text
        self.source_language = source_language
        self.target_language = target_language
        self.confidence = confidence
        self.processing_time = processing_time
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        """Convert the result to a dictionary"""
        return {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp
        }

class EnhancedReActTranslationAgent:
    def __init__(self,
                 instruct_model_name: str = "meta-llama/Llama-3.3-7B-Instruct",
                 translation_model_name: str = "facebook/nllb-200-distilled-600M",
                 cache_dir: str = "./models",
                 ocr_config: Optional[Dict] = None):
        """
        Initialize the enhanced translation agent with more robust configuration
        and error handling.
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.translation_history: List[TranslationResult] = []
        
        # Enhanced language mapping with more languages and scripts
        self.lang_map = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "zh": "zho_Hans",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "ru": "rus_Cyrl",
            "ar": "ara_Arab",
            "hi": "hin_Deva",
            "pt": "por_Latn"
        }

        # Initialize models with better error handling
        self._initialize_models(instruct_model_name, translation_model_name)
        
        # Configure OCR with custom settings
        self.ocr_config = ocr_config or {
            'lang': 'eng+fra+spa+deu',  # Multiple language support
            'config': '--psm 3',  # Automatic page segmentation
            'timeout': 30
        }
        
        # Initialize image processing parameters
        self.image_processing_config = {
            'default_font_size': 20,
            'min_confidence': 0.7,
            'supported_formats': {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'},
            'max_image_size': (3000, 3000)
        }

    def _initialize_models(self, instruct_model_name: str, translation_model_name: str):
        """Initialize ML models with proper error handling and validation"""
        try:
            # Initialize instruction model
            self.instruct_tokenizer = AutoTokenizer.from_pretrained(
                instruct_model_name, 
                cache_dir=self.cache_dir,
                use_fast=True
            )
            self.instruct_model = AutoModelForSeq2SeqLM.from_pretrained(
                instruct_model_name,
                cache_dir=self.cache_dir,
                device_map="auto"
            )
            self.instruct_pipeline = pipeline(
                "text2text-generation",
                model=self.instruct_model,
                tokenizer=self.instruct_tokenizer,
                device_map="auto"
            )
            
            # Initialize translation model
            self.trans_tokenizer = AutoTokenizer.from_pretrained(
                translation_model_name,
                cache_dir=self.cache_dir,
                use_fast=True
            )
            self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
                translation_model_name,
                cache_dir=self.cache_dir,
                device_map="auto"
            )
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def process_translation_request(self, request: TranslationRequest) -> TranslationResult:
        """
        Process a translation request with enhanced error handling and automatic
        format detection.
        """
        start_time = time.time()
        
        try:
            # Detect document type and validate request
            doc_type = self._detect_document_type(request.source_path)
            self.logger.info(f"Detected document type: {doc_type}")
            
            # Extract text based on document type
            original_text = self._extract_text(request.source_path, doc_type)
            
            # Detect source language if not provided
            if not request.source_language:
                request.source_language = self.detect_language(original_text)
                self.logger.info(f"Detected source language: {request.source_language}")
            
            # Perform translation
            translated_text = self.translate_with_nllb(
                original_text,
                request.target_language,
                request.source_language
            )
            
            # Process output based on document type and formatting options
            output_path = self._process_output(
                request.source_path,
                translated_text,
                doc_type,
                request.formatting_options
            )
            
            # Create translation result
            processing_time = time.time() - start_time
            result = TranslationResult(
                original_text=original_text,
                translated_text=translated_text,
                source_language=request.source_language,
                target_language=request.target_language,
                confidence=0.95,  # This could be calculated based on model confidence
                processing_time=processing_time
            )
            
            # Store in history
            self.translation_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing translation request: {str(e)}")
            raise

    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Detect document type from file extension and content"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext in self.image_processing_config['supported_formats']:
            return DocumentType.IMAGE
        elif ext in {'.txt', '.doc', '.docx', '.rtf'}:
            return DocumentType.TEXT
        else:
            return DocumentType.UNKNOWN

    def _extract_text(self, file_path: str, doc_type: DocumentType) -> str:
        """Extract text from document based on its type"""
        if doc_type == DocumentType.PDF:
            return self._extract_text_from_pdf(file_path)
        elif doc_type == DocumentType.IMAGE:
            return self._extract_text_from_image(file_path)
        elif doc_type == DocumentType.TEXT:
            return self._extract_text_from_text_file(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Enhanced PDF text extraction with better formatting preservation"""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            
            for page in doc:
                # Extract text with blocks to preserve layout
                blocks = page.get_text("blocks")
                for block in blocks:
                    text_blocks.append(block[4])  # block[4] contains the text
                text_blocks.append("\n\n")  # Add spacing between pages
                
            return "\n".join(text_blocks)
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _extract_text_from_image(self, image_path: str) -> str:
        """Enhanced OCR with preprocessing and validation"""
        try:
            # Open and preprocess image
            with Image.open(image_path) as img:
                # Resize if necessary
                if img.size[0] > self.image_processing_config['max_image_size'][0] or \
                   img.size[1] > self.image_processing_config['max_image_size'][1]:
                    img.thumbnail(self.image_processing_config['max_image_size'])
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply image enhancements
                img = img.filter(ImageFilter.SHARPEN)
                
                # Perform OCR
                text = pytesseract.image_to_string(
                    img,
                    lang=self.ocr_config['lang'],
                    config=self.ocr_config['config']
                )
                
                return text.strip()
                
        except Exception as e:
            self.logger.error(f"Error performing OCR: {str(e)}")
            raise

    def _process_output(self, 
                       source_path: str,
                       translated_text: str,
                       doc_type: DocumentType,
                       formatting_options: Dict) -> str:
        """Process and save the translated output with formatting"""
        output_dir = os.path.dirname(source_path)
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        
        if doc_type == DocumentType.PDF:
            output_path = os.path.join(output_dir, f"{base_name}_translated.pdf")
            self._create_formatted_pdf(translated_text, output_path, formatting_options)
        elif doc_type == DocumentType.IMAGE:
            output_path = os.path.join(output_dir, f"{base_name}_translated.png")
            self._create_translated_image(source_path, translated_text, output_path, formatting_options)
        else:
            output_path = os.path.join(output_dir, f"{base_name}_translated.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
                
        return output_path

    def _create_formatted_pdf(self, 
                            text: str,
                            output_path: str,
                            formatting_options: Dict) -> None:
        """Create a well-formatted PDF with the translated text"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=formatting_options.get('pagesize', A4),
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        
        styles = getSampleStyleSheet()
        custom_style = ParagraphStyle(
            'CustomStyle',
            parent=styles['Normal'],
            fontSize=formatting_options.get('font_size', 11),
            leading=formatting_options.get('leading', 14),
            alignment=formatting_options.get('alignment', 0)
        )
        
        story = []
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para, custom_style))
                
        doc.build(story)

    def run_interactive_session(self, user_question: str, max_steps: int = 10) -> None:
        """Enhanced interactive session with better context management"""
        self.conversation_context = {
            "question": user_question,
            "steps_taken": [],
            "current_state": {},
            "max_steps": max_steps
        }
        
        system_prompt = self._generate_system_prompt()
        self.conversation = f"{system_prompt}\nQuestion: {user_question}\n"
        
        step = 0
        while step < max_steps:
            try:
                # Generate next action
                response = self._generate_next_action()
                if "Final Answer:" in response:
                    final_answer = response.split("Final Answer:")[-1].strip()
                    self.logger.info(f"Session completed with answer: {final_answer}")
                    break
                    
                # Extract and execute action
                action_command = self._extract_and_validate_action(response)
                if action_command:
                    observation = self._execute_action_safely(action_command)
                    self._update_conversation_context(action_command, observation)
                else:
                    self.logger.warning("No valid action found in response")
                    break
                    
                step += 1
                
            except Exception as e:
                self.logger.error(f"Error in interactive session: {str(e)}")
                break

    def _generate_system_prompt(self) -> str:
        """Generate an enhanced system prompt with more detailed instructions"""
        return """You are an advanced translation agent capable of:
        1. Understanding and processing various document formats
        2. Detecting languages automatically
        3. Preserving document formatting and layout
        4. Handling complex translation requirements
        5. Managing multiple translation requests
        
        Available actions (specify as JSON):
        1. perform_ocr: {"action": "perform_ocr", "action_input": {"image_path": "path"}}
        2. detect_language: {"action": "detect_language", "action_input": {"text": "text"}}
        3. translate_text: {"action": "translate_text", "action_input": {"text": "text", "tgt_lang": "lang_code"}}
        4. process_document: {"action": "process_document", "action_input": {"path": "path", "options": {}}}
        5. summarize_text: {"action": "summarize_text", "action_input": {"text": "text"}}
        
        Response format:
        Thought: <your analysis>
        Action: ```<json action object>```
        Observation: <result>
        
        End with:
        Final Answer: <complete translation summary>
        """

    def _generate_next_action(self) -> str:
        """Generate next action with enhanced context awareness"""
        try:
            # Add context to help model make better decisions
            context_prompt = f"\nCurrent context: {json.dumps(self.conversation_context['current_state'])}\n"
            full_prompt = self.conversation + context_prompt
            
            # Generate response with temperature adjustment for better reasoning
            result = self.instruct_pipeline(
                full_prompt,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )[0]['generated_text']
            
            return result.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating next action: {str(e)}")
            raise

    def _extract_and_validate_action(self, response: str) -> Optional[Dict]:
        """Extract and validate action JSON with enhanced error checking"""
        try:
            # Extract JSON from markdown block
            match = re.search(r"Action:\s*```(.*?)```", response, re.DOTALL)
            if not match:
                return None
                
            action_json = match.group(1).strip()
            action_command = json.loads(action_json)
            
            # Validate action structure
            if not isinstance(action_command, dict):
                raise ValueError("Action must be a dictionary")
                
            required_fields = {'action', 'action_input'}
            if not all(field in action_command for field in required_fields):
                raise ValueError(f"Action missing required fields: {required_fields}")
                
            return action_command
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in action: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error validating action: {str(e)}")
            return None

    def _execute_action_safely(self, action_command: Dict) -> str:
        """Execute action with enhanced safety checks and error handling"""
        action = action_command.get('action')
        params = action_command.get('action_input', {})
        
        try:
            # Validate action type
            if action not in self.get_supported_actions():
                raise ValueError(f"Unsupported action: {action}")
                
            # Validate parameters
            self._validate_action_parameters(action, params)
            
            # Execute action
            result = self._execute_single_action(action, params)
            
            # Update execution history
            self._update_execution_history(action, params, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing action '{action}': {str(e)}")
            return f"Error: {str(e)}"

    def _validate_action_parameters(self, action: str, params: Dict) -> None:
        """Validate action parameters against defined schemas"""
        action_schemas = {
            'perform_ocr': {'required': ['image_path']},
            'detect_language': {'required': ['text']},
            'translate_text': {'required': ['text', 'tgt_lang']},
            'process_document': {'required': ['path', 'options']},
            'summarize_text': {'required': ['text']}
        }
        
        schema = action_schemas.get(action)
        if not schema:
            raise ValueError(f"No schema defined for action: {action}")
            
        missing_params = [param for param in schema['required'] if param not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters for {action}: {missing_params}")

    def _execute_single_action(self, action: str, params: Dict) -> str:
        """Execute a single action with proper error handling"""
        action_map = {
            'perform_ocr': self._extract_text_from_image,
            'detect_language': self.detect_language,
            'translate_text': self.translate_with_nllb,
            'process_document': self._process_document
        }
        
        if action not in action_map:
            raise ValueError(f"Unsupported action: {action}")
            
        return action_map[action](**params)

    def _update_execution_history(self, action: str, params: Dict, result: str) -> None:
        """Update execution history for better context tracking"""
        history_entry = {
            'timestamp': time.time(),
            'action': action,
            'parameters': params,
            'result_summary': result[:100] + '...' if len(result) > 100 else result
        }
        
        self.conversation_context['steps_taken'].append(history_entry)
        self.conversation_context['current_state'].update({
            'last_action': action,
            'last_result': result[:100] + '...' if len(result) > 100 else result
        })

    def get_supported_actions(self) -> List[str]:
        """Return list of supported actions"""
        return ['perform_ocr', 'detect_language', 'translate_text', 'process_document']

    def get_translation_statistics(self) -> Dict:
        """Get statistics about translation operations"""
        if not self.translation_history:
            return {"message": "No translations performed yet"}
        """ Let save the statistics in a dictionary TODO: save the statistics in a database """    
        stats = {
            "total_translations": len(self.translation_history),
            "average_processing_time": sum(r.processing_time for r in self.translation_history) / len(self.translation_history),
            "language_pairs": {},
            "success_rate": 0.0
        }
        
        # Calculate language pair statistics
        for result in self.translation_history:
            pair = f"{result.source_language}->{result.target_language}"
            if pair not in stats["language_pairs"]:
                stats["language_pairs"][pair] = 0
            stats["language_pairs"][pair] += 1
            
        return stats

if __name__ == '__main__':
    # Example usage with enhanced error handling
    try:
        # Initialize agent
        agent = EnhancedReActTranslationAgent()
        
        # Example translation request
        request = TranslationRequest(
            source_path="sample.pdf",
            target_language="en",
            formatting_options={
                "font_size": 12,
                "preserve_layout": True
            }
        )
        
        # Process request
        result = agent.process_translation_request(request)
        
        # Print results
        print(f"Translation completed in {result.processing_time:.2f} seconds")
        print(f"Source language: {result.source_language}")
        print(f"Confidence: {result.confidence:.2f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        sys.exit(1)
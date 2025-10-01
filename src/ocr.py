import io
import logging
import os
from typing import Optional
from pathlib import Path

import pytesseract
from PIL import Image, UnidentifiedImageError
from PyPDF2 import PageObject, PdfReader

from src.constants import LOG_FILE_PATH
from src.utils import clean_text, setup_logging, secure_file_path

setup_logging()
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF with security and error handling."""
    try:
        # Always use secure path handling
        safe_path = secure_file_path(os.path.dirname(file_path), os.path.basename(file_path))
        
        text_parts = []
        with open(safe_path, "rb") as f:
            pdf_reader = PdfReader(f)
            logger.info(f"Processing PDF with {len(pdf_reader.pages)} pages")

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                        logger.debug(f"Extracted text from page {page_num}")
                    else:
                        logger.info(f"Attempting OCR for page {page_num}")
                        ocr_text = extract_text_from_images(page)
                        if ocr_text:
                            text_parts.append(ocr_text)
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")

        text = " ".join(text_parts)
        cleaned_text = clean_text(text)
        logger.info(f"Extracted {len(cleaned_text)} characters from PDF")
        return cleaned_text
        
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def extract_text_from_images(page: PageObject) -> str:
    """Extract text from images using OCR with specific error handling."""
    text_parts = []
    
    try:
        for image_file_object in page.images:
            try:
                image = Image.open(io.BytesIO(image_file_object.data))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    text_parts.append(ocr_text)
                    logger.debug("OCR text extracted from image")
            except UnidentifiedImageError:
                logger.warning("Could not identify image format for OCR")
            except pytesseract.TesseractError as e:
                logger.error(f"Tesseract OCR failed: {e}")
            except Exception as e:
                logger.error(f"Image processing error: {e}")
                
    except Exception as e:
        logger.error(f"Error accessing page images: {e}")
    
    return " ".join(text_parts)

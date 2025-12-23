"""Preprocessing utilities for Spanish tasks.

These functions replicate legacy preprocessing utilities used in prior evaluations
but are self-contained in Benchy.
"""

import re
from typing import Dict, Any, List


def lowercase_first_letter(text: str) -> str:
    """Lowercase the first letter of text.
    
    Args:
        text: Input text
        
    Returns:
        Text with first letter lowercased
    """
    if not text:
        return text
    return text[0].lower() + text[1:]


def general_detokenize(text: str) -> str:
    """Detokenize text by removing extra whitespace.
    
    This is a simplified version of the legacy general_detokenize utility.
    It normalizes whitespace in the text.
    
    Args:
        text: Input text (may have tokenization artifacts)
        
    Returns:
        Detokenized text with normalized whitespace
    """
    if not text:
        return text
    
    # Normalize whitespace: multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def process_doc_nli(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Process NLI (Natural Language Inference) document.
    
    Applies preprocessing for NLI tasks:
    - Detokenize premise and hypothesis
    - Remove trailing punctuation from premise
    - Lowercase first letter of hypothesis
    - Ensure hypothesis ends with a dot
    
    Args:
        doc: Document dictionary with 'premise' and 'hypothesis' keys
        
    Returns:
        Processed document dictionary
    """
    # Detokenize
    doc["premise"] = general_detokenize(doc.get("premise", "")).strip()
    doc["hypothesis"] = general_detokenize(doc.get("hypothesis", "")).strip()
    
    # Remove last punctuation mark in the premise
    if doc["premise"].endswith((".", ",", "!", "?")):
        doc["premise"] = doc["premise"][:-1]
    
    # Lowercase the first letter in the hypothesis
    doc["hypothesis"] = lowercase_first_letter(doc["hypothesis"])
    
    # Ensure that the hypothesis ends with a dot
    if not doc["hypothesis"].endswith("."):
        doc["hypothesis"] = doc["hypothesis"] + "."
    
    return doc


def process_docs_paraphrases(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Process paraphrase document.
    
    Applies preprocessing for paraphrase tasks:
    - Detokenize sentence1 and sentence2
    - Remove trailing punctuation from sentence1
    - Lowercase first letter of sentence2
    
    Args:
        doc: Document dictionary with 'sentence1' and 'sentence2' keys
        
    Returns:
        Processed document dictionary, or None if sentences are empty
    """
    if doc.get("sentence1") in [None, ""] or doc.get("sentence2") in [None, ""]:
        return None
    
    doc["sentence1"] = general_detokenize(doc["sentence1"]).strip()
    doc["sentence2"] = general_detokenize(doc["sentence2"]).strip()
    
    # Remove final punctuation mark in the first sentence
    if doc["sentence1"].endswith((".", ",", ";")):
        doc["sentence1"] = doc["sentence1"][:-1]
    
    # Start the second sentence in lowercase
    doc["sentence2"] = lowercase_first_letter(doc["sentence2"])
    
    return doc


def process_docs_copa_es(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Process COPA-es document.
    
    Lowercases the first letter of choice1 and choice2.
    
    Args:
        doc: Document dictionary with 'choice1' and 'choice2' keys
        
    Returns:
        Processed document dictionary
    """
    doc["choice1"] = lowercase_first_letter(doc.get("choice1", ""))
    doc["choice2"] = lowercase_first_letter(doc.get("choice2", ""))
    return doc




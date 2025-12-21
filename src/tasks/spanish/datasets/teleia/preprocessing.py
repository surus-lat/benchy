"""Preprocessing utilities for teleia tasks."""

import re
from typing import Dict, Any, List, Optional


def preprocess(text: str) -> str:
    """Preprocess text by removing special markers and normalizing.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_cervantes(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Process cervantes_ave document.
    
    Args:
        doc: Document dictionary with question, option_a, option_b, option_c, option_d, answer
        
    Returns:
        Processed document with query, choices, target
    """
    question = preprocess(doc.get("question", ""))
    query = f"Pregunta: {question}\nRespuesta:"
    
    choices = [
        preprocess(option)
        for option in [
            doc.get("option_a"),
            doc.get("option_b"),
            doc.get("option_c"),
            doc.get("option_d"),
        ]
        if option
    ]
    
    answer = doc.get("answer", "A")
    target = ["A", "B", "C", "D"].index(answer) if answer in ["A", "B", "C", "D"] else 0
    
    return {
        "query": query,
        "choices": choices,
        "target": target,
    }


def process_pce_siele(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Process PCE or SIELE document.
    
    Args:
        doc: Document dictionary with question, option_a, option_b, option_c, answer
        
    Returns:
        Processed document with query, choices, target
    """
    question = preprocess(doc.get("question", ""))
    query = f"Pregunta: {question}\nRespuesta:"
    
    choices = [
        preprocess(option)
        for option in [
            doc.get("option_a"),
            doc.get("option_b"),
            doc.get("option_c"),
        ]
        if option
    ]
    
    answer = doc.get("answer", "A")
    target = ["A", "B", "C"].index(answer) if answer in ["A", "B", "C"] else 0
    
    return {
        "query": query,
        "choices": choices,
        "target": target,
    }




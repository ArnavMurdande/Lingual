import spacy
import re
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from langdetect import detect
from deep_translator import GoogleTranslator
from .models import SentenceScore, NerEntity
import textwrap

# Initialize models lazily or globally
print("Loading models... (This may take a moment on first run)")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

abstractive_model = None 
abstractive_tokenizer = None

def load_abstractive_model():
    global abstractive_model, abstractive_tokenizer
    if abstractive_model is None:
        print("Loading abstractive model (mT5_multilingual_XLSum)...")
        try:
            # Swapped to csebuetnlp/mT5_multilingual_XLSum as requested
            model_name = "csebuetnlp/mT5_multilingual_XLSum"
            abstractive_tokenizer = AutoTokenizer.from_pretrained(model_name)
            abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading abstractive model: {e}")
            abstractive_model = None

def detect_src_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang
    except:
        return "en"

def clean_visual_artifacts(text: str) -> str:
    """
    Removes single newlines (visual line breaks) but preserves double newlines (paragraphs).
    """
    if not text:
        return ""
    # Split by double newlines to isolate paragraphs
    paragraphs = text.replace('\r\n', '\n').split('\n\n')
    # Join lines within each paragraph with a space
    cleaned_paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs]
    # Rejoin paragraphs with double newlines
    return '\n\n'.join([p for p in cleaned_paragraphs if p])

def extract_and_remove_references(text: str):
    """
    Extracts citations/bibliography section based on headers and returns cleaned text + citations list.
    """
    # Common headers for references
    patterns = [
        r'\n(References|Bibliography|Works Cited|Citations)\s*\n', # Standard
        r'\n([0-9]+\.\s*(?:References|Bibliography|Works Cited))\s*\n' # Numbered
    ]
    
    split_index = -1
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # If multiple consistencies found, take the last one usually? 
            # Or the first one near the end? Usually References are at the end.
            # Let's take the last occurrence to avoid false positives in TOC
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                 # Check if the match is in the last 30% of the document to be safe?
                 last_match = matches[-1]
                 if last_match.start() > len(text) * 0.5:
                     split_index = last_match.start()
                     break
    
    citations = []
    cleaned_text = text
    
    if split_index != -1:
        cleaned_text = text[:split_index].strip()
        ref_section = text[split_index:].strip()
        
        # Naive extraction of citations from the section
        # Assuming citations are either numbered [1] or paragraphs
        # Let's just split by newlines for now and filter empty
        raw_citations = ref_section.split('\n')
        # Skip the header itself
        citations = [c.strip() for c in raw_citations[1:] if len(c.strip()) > 10]
        # Further refine: filter out page numbers or short trash
        citations = [c for c in citations if not c.isdigit()]
    
    return cleaned_text, citations

def extract_ner(text: str):
    """
    Extracts NER data using Spacy.
    """
    # Limit text size for Spacy optimization if too large
    safe_text = text[:50000]
    doc = nlp(safe_text)
    
    ner_list = []
    seen = set()
    
    # Pre-compute sentence mapping for faster context lookup
    # Actually doc.ents iteration is fine.
    
    for ent in doc.ents:
        # Filter commonly uninteresting types
        if ent.label_ in ["CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "DATE", "TIME"]:
            continue
            
        key = (ent.text.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        
        relationship = ""
        
        # Heuristic 2.0: Capture Subject - Verb - Object context
        try:
             head = ent.root.head
             
             if head.pos_ == "VERB":
                 # Entity is argument of a verb.
                 # Let's try to reconstruct the clause: [Subj] [Verb] [Obj]
                 
                 # 1. Get Subject(s)
                 subjects = [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")]
                 subj_text = subjects[0].text if subjects else ""
                 
                 # 2. Get Object(s) / Attributes
                 objects = [c for c in head.children if c.dep_ in ("dobj", "attr", "pobj", "acomp")]
                 obj_text = objects[0].text if objects else ""
                 
                 # 3. Construct string based on where the entity is
                 verb_text = head.text
                 
                 if ent.root.dep_ in ("nsubj", "nsubjpass"):
                     # Entity is Subject -> "Verb Object"
                     if obj_text:
                         relationship = f"{verb_text} {obj_text}"
                     else:
                        relationship = f"{verb_text}"
                        
                 elif ent.root.dep_ in ("dobj", "attr", "pobj"):
                     # Entity is Object -> "Subject Verb"
                     if subj_text:
                         relationship = f"{subj_text} {verb_text}"
                     else:
                         relationship = f"action: {verb_text}"
                 else:
                     # Other dependency, just show [Verb] [Object] or [Subj] [Verb]
                     if subj_text and obj_text:
                         relationship = f"{subj_text} {verb_text} {obj_text}"
                     else:
                         relationship = f"{verb_text} {obj_text}"
                         
             elif head.pos_ == "NOUN":
                 relationship = f"related to {head.text}"
             else:
                 # Fallback to a small window if no clear verb structure
                 start = max(ent.start - 3, 0)
                 end = min(ent.end + 3, len(doc))
                 # Exclude entity itself from context string if possible to avoid redundancy?
                 # Actually context is better with just surrounding words.
                 window = doc[start:end].text
                 relationship = f"...{window}..."
                 
        except Exception as e:
            relationship = "mentioned"
            
        # Cleanup
        relationship = relationship.strip()
        if not relationship or relationship == ent.text:
             relationship = "mentioned"
             
        ner_list.append(NerEntity(
            name=ent.text,
            entity=ent.label_,
            relationship=relationship
        ))
        
    return ner_list

def translate_content(text: str, target_lang: str) -> str:
    if not text.strip():
        return ""
    try:
        # Refactored chunking using textwrap to avoid splitting words
        # 2000 chars is a safe limit for Google Translator
        chunks = textwrap.wrap(text, width=2000, break_long_words=False, replace_whitespace=False)
        
        translated_chunks = []
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        for chunk in chunks:
            try:
                res = translator.translate(chunk)
                if isinstance(res, list):
                    res = " ".join(res)
                translated_chunks.append(res)
            except Exception as inner_e:
                print(f"Chunk translation error: {inner_e}")
                translated_chunks.append(chunk) # Fallback to original
            time.sleep(0.2) # Rate limit politeness
            
        return " ".join(translated_chunks)
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

def content_ranking(text: str, top_n: int = 5, ratio: float = None):
    # 0. Clean Artifacts is assumed done BEFORE this function call now in execute_summary
    # But for safety we can ensure no weird newlines break sentence splitting regex
    
    # 1. Segmentation
    sentences = []
    
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
    
    if has_cjk:
        # CJK splitting
        clean_text = text.replace('\n', '') if len(text) < 5000 else text 
        sentences = re.split(r'([。？！.?!])', clean_text)
        new_sentences = []
        current = ""
        for part in sentences:
            if part in ['。', '？', '！', '.', '?', '!']:
                current += part
                if len(current.strip()) > 1:
                    new_sentences.append(current.strip())
                current = ""
            else:
                current += part
        if len(current.strip()) > 0:
            new_sentences.append(current.strip())
        sentences = new_sentences
    else:
        # Non-CJK flow
        try:
            lang = detect(text[:1000]) 
        except:
            lang = "en"
            
        if lang in ['hi', 'mr', 'ne', 'sa', 'gu', 'te', 'kn', 'bn', 'pa']:
            sentences = re.split(r'[।?!.]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        else:
            try:
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            except:
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            
    sentences = [s for s in sentences if s and len(s) > 1]
    
    if not sentences:
        return [], [], [], {}

    n_sentences = len(sentences)
    
    if ratio is not None:
        calculated_n = int(n_sentences * ratio)
        top_n = max(1, calculated_n)
    
    if top_n > n_sentences:
        top_n = n_sentences

    # 2. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words=None) 
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()
    except:
        tfidf_scores = np.zeros(n_sentences)

    # 3. Embeddings & Centrality
    embeddings = embedder.encode(sentences) 
    doc_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    centrality_scores = cosine_similarity(embeddings, doc_embedding).flatten()
    if centrality_scores.max() > 0:
        centrality_scores = centrality_scores / centrality_scores.max()
        
    # 4. TextRank
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    nx_graph = nx.from_numpy_array(sim_matrix)
    try:
        textrank_scores = nx.pagerank(nx_graph)
        textrank_scores = np.array([textrank_scores[i] for i in range(len(sentences))])
        if textrank_scores.max() > 0:
            textrank_scores = textrank_scores / textrank_scores.max()
    except:
        textrank_scores = np.zeros(n_sentences)

    # 5. Composite Score
    final_scores = (0.4 * centrality_scores) + (0.3 * textrank_scores) + (0.3 * tfidf_scores)
    
    ranked_sentences = []
    for i in range(n_sentences):
        ranked_sentences.append(SentenceScore(
            text=sentences[i],
            index=i,
            score=float(final_scores[i]),
            rank=0,
            reasons={
                "centrality": float(centrality_scores[i]),
                "textrank": float(textrank_scores[i]),
                "tfidf": float(tfidf_scores[i])
            }
        ))
        
    ranked_sentences.sort(key=lambda x: x.score, reverse=True)
    
    top_indices = []
    for rank, item in enumerate(ranked_sentences):
        item.rank = rank + 1
        if rank < top_n:
            top_indices.append(item.index)
            
    top_indices.sort()
    summary_sentences = [sentences[i] for i in top_indices]
    
    metrics = {
        "original_length": len(text),
        "summary_length": sum(len(s) for s in summary_sentences),
        "compression_ratio": round(1 - (sum(len(s) for s in summary_sentences) / len(text)), 2) if len(text) > 0 else 0,
        "sentence_count": n_sentences,
        "kept_sentences": top_n
    }
    
    return summary_sentences, ranked_sentences, metrics

def abstractive_refine(text: str, max_length: int = 150):
    # text is already cleaned in execute_summary_logic
    load_abstractive_model()
    if abstractive_model is None:
        return text

    try:
        # Chunking strategy for long documents (Map-Reduce style)
        # csebuetnlp/mT5_multilingual_XLSum supports larger context, but we must respect strict limits.
        # We'll use 3000 chars logical chunks roughly mapped to 512-1024 tokens.
        chunks = textwrap.wrap(text, width=3000, break_long_words=False, replace_whitespace=False)
        
        prefix = "summarize: " 
        chunk_summaries = []
        
        for chunk in chunks:
            input_text = prefix + chunk
            inputs = abstractive_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
            
            summary_ids = abstractive_model.generate(
                inputs, 
                max_length=max_length, 
                min_length=30, 
                length_penalty=1.2, 
                num_beams=4, # High quality beam search
                early_stopping=True
            )
            
            summary = abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(summary)
            
        return " ".join(chunk_summaries)
    except Exception as e:
        print(f"Abstractive generation failed: {e}")
        return text

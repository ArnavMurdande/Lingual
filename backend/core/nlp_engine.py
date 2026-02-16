import spacy
import re
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import time
from langdetect import detect
from deep_translator import GoogleTranslator
from .models import SentenceScore

# Initialize models lazily or globally
# For production, load these once at startup.
# We will load them globally here for simplicity in this demo.

print("Loading models... (This may take a moment on first run)")
try:
    nlp = spacy.load("en_core_web_sm") # Fallback, ideally load multilingual "xx_ent_wiki_sm"
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Multilingual sentence transformer (CPU friendly)
# paraphras-multilingual-MiniLM-L12-v2 is standard good compromise.
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Optional abstractive model (load lazily to save RAM if not used)
abstractive_model = None 
abstractive_tokenizer = None

def load_abstractive_model():
    global abstractive_model, abstractive_tokenizer
    if abstractive_model is None:
        # mT5 small is good for multilingual. 
        # For better quality but slower: "google/mt5-small" or "csebuetnlp/mT5_multilingual_XLSum"
        print("Loading abstractive model (mT5-small)...")
        try:
            model_name = "google/mt5-small"
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

def translate_content(text: str, target_lang: str) -> str:
    if not text.strip():
        return ""
    try:
        # Chunking for long text (Google Translator API limit approx 5000 chars)
        # Use smaller chunks to avoid timeouts and hit limits less
        # Python slices are character-based so safe for CJK chars, but might split a sentence.
        chunk_size = 2000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        translated_chunks = []
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        for chunk in chunks:
            # Add simple retry logic
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
    """
    Hybrid ranking strategy:
    1. TextRank (Graph connectivity)
    2. TF-IDF (Keyword importance)
    3. Semantic Centrality (Embedding similarity to document mean)
    """
    
    # 1. Segmentation
    sentences = []
    
    # Check for CJK characters by unicode range (basic check)
    has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
    
    if has_cjk:
        # CJK splitting by period/ideographic full stop (Chinese, Japanese, Korean)
        # Using a broad set of delimiters: 。 ？ ！ . ? ! 
        # Also clean up potential PDF artifacts (newlines within sentences)
        clean_text = text.replace('\n', '') if len(text) < 5000 else text # Basic cleanup, careful with paragraphs
        sentences = re.split(r'([。？！.?!])', clean_text)
        # Re-attach delimiters?
        # A simple way is to iterate and attach. Alternatively, just take the content.
        # Let's just take the content for now, or re-join. 
        # Actually, split with capturing group returns delimiters too. 
        # ["Sen1", "。", "Sen2", "！", ...]
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
        # Add remainder
        if len(current.strip()) > 0:
            new_sentences.append(current.strip())
            
        sentences = new_sentences
    else:
        # Non-CJK flow
        # Force language detection for other scripts
        try:
            lang = detect(text[:1000]) 
        except:
            lang = "en"
            
        if lang in ['hi', 'mr', 'ne', 'sa', 'gu', 'te', 'kn', 'bn', 'pa']:
            # Devanagari/South Asian splitting by danda (।)
            sentences = re.split(r'[।?!.]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        else:
            # Fallback to Spacy or simple split for Indo-European languages
            try:
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            except:
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            
    # Final cleanup of sentences list
    sentences = [s for s in sentences if s and len(s) > 1]
    
    if not sentences:
        return [], [], [], {
            "original_length": len(text),
            "summary_length": 0,
            "compression_ratio": 0,
            "sentence_count": 0,
            "kept_sentences": 0
        }

    n_sentences = len(sentences)
    
    # Calculate top_n based on ratio if provided
    if ratio is not None:
        calculated_n = int(n_sentences * ratio)
        top_n = max(1, calculated_n) # Ensure at least 1 sentence
    
    if top_n > n_sentences:
        top_n = n_sentences

    # 2. Vectorization (TF-IDF)
    # Use simple heuristic for stop words or none for multilingual
    vectorizer = TfidfVectorizer(stop_words=None) 
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        # Normalize
        if tfidf_scores.max() > 0:
            tfidf_scores = tfidf_scores / tfidf_scores.max()
    except:
        tfidf_scores = np.zeros(n_sentences)

    # 3. Embeddings & Centrality
    embeddings = embedder.encode(sentences) # (n, 384)
    doc_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
    
    # Elastic Net type ranking: how similar is each sent to the document center?
    centrality_scores = cosine_similarity(embeddings, doc_embedding).flatten()
    # Normalize
    if centrality_scores.max() > 0:
        centrality_scores = centrality_scores / centrality_scores.max()
        
    # 4. TextRank (via Similarity Matrix)
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    nx_graph = nx.from_numpy_array(sim_matrix)
    try:
        textrank_scores = nx.pagerank(nx_graph)
        textrank_scores = np.array([textrank_scores[i] for i in range(len(sentences))])
        # Normalize
        if textrank_scores.max() > 0:
            textrank_scores = textrank_scores / textrank_scores.max()
    except:
        textrank_scores = np.zeros(n_sentences)

    # 5. Composite Score
    # Weights: Centrality (40%), TextRank (30%), TF-IDF (30%)
    # Adjust based on preference for "main idea" vs "informative content"
    final_scores = (0.4 * centrality_scores) + (0.3 * textrank_scores) + (0.3 * tfidf_scores)
    
    # Pack data
    ranked_sentences = []
    for i in range(n_sentences):
        ranked_sentences.append(SentenceScore(
            text=sentences[i],
            index=i,
            score=float(final_scores[i]),
            rank=0, # Will assign after sorting
            reasons={
                "centrality": float(centrality_scores[i]),
                "textrank": float(textrank_scores[i]),
                "tfidf": float(tfidf_scores[i])
            }
        ))
        
    # Sort
    ranked_sentences.sort(key=lambda x: x.score, reverse=True)
    
    # Assign Rank
    top_indices = []
    for rank, item in enumerate(ranked_sentences):
        item.rank = rank + 1
        if rank < top_n:
            top_indices.append(item.index)
            
    # Re-order top sentences by original appearance for coherence
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
    load_abstractive_model()
    if abstractive_model is None:
        return text

    try:
        # Preprocess for T5: "summarize: " prefix is standard for T5 but mt5 is raw seq2seq usually. 
        # However, google/mt5-small is not fine-tuned for summarization specifically, 
        # but often works better with a prefix or just by itself if fine-tuned.
        # User requested "mt5-small summarization". Raw mt5-small generates tokens. 
        # It's safer to use the pipeline if possible, or manual generation.
        # Given "CPU" constraint and "mt5-small", we'll do manual generation with token limiting.
        
        inputs = abstractive_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generator args adjusted for CPU speed
        summary_ids = abstractive_model.generate(
            inputs, 
            max_length=max_length, 
            min_length=30, 
            length_penalty=2.0, 
            num_beams=2, # Low beam search for speed
            early_stopping=True
        )
        
        summary = abstractive_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Abstractive generation failed: {e}")
        return text

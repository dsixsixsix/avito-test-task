import os, hashlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from common import mapk, parse_gt

CACHE = "solution/cache"
P = pd.read_pickle(CACHE + "/prep.pkl")
aid = np.array(P["article_id"])


def build_doc_texts(max_body_chars=2000):
    titles = P["art_title"]
    bodies = P["art_body_text"]
    docs = []
    for t, b in zip(titles, bodies):
        b = b[:max_body_chars]
        docs.append(f"{t}. {b}".strip())
    return docs


def encode(model_name, texts, prefix, batch=64):
    payload = model_name + "|" + prefix + "|" + "\x00".join(texts)
    key = hashlib.md5(payload.encode()).hexdigest()[:12]
    fn = f"{CACHE}/emb_{key}.npy"
    if os.path.exists(fn):
        return np.load(fn)
    model = SentenceTransformer(model_name)
    inp = [prefix + t for t in texts]
    emb = model.encode(inp, batch_size=batch, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(fn, emb)
    return emb


def rank_from_scores(scores, k=10):
    out = []
    for row in scores:
        idx = np.argsort(-row)[:k]
        out.append([int(aid[i]) for i in idx])
    return out


def minmax(s):
    lo = s.min(axis=1, keepdims=True)
    hi = s.max(axis=1, keepdims=True)
    return (s - lo) / (hi - lo + 1e-9)


def bm25_scores(corpus, queries, k1=2.0, b=0.5):
    bm = BM25Okapi(corpus, k1=k1, b=b)
    return np.array([bm.get_scores(q) for q in queries])



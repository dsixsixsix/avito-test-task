import os, json, time
import pandas as pd
from common import load_data, clean_html, tokenize

CACHE = "solution/cache"
os.makedirs(CACHE, exist_ok=True)


def build():
    art, cal, test = load_data()
    t0 = time.time()

    art_clean = art.copy()
    art_clean["body_text"] = art_clean["body"].map(clean_html)
    art_clean["full"] = art_clean["title"].fillna("") + ". " + art_clean["body_text"]
    art_tokens = [tokenize(t) for t in art_clean["full"]]
    title_tokens = [tokenize(t) for t in art_clean["title"].fillna("")]
    print(f"articles tokenized {time.time()-t0:.1f}s")

    cal_tokens = [tokenize(t) for t in cal["query_text"]]
    test_tokens = [tokenize(t) for t in test["query_text"]]
    print(f"queries tokenized {time.time()-t0:.1f}s")

    pd.to_pickle({
        "article_id": art["article_id"].tolist(),
        "art_tokens": art_tokens,
        "title_tokens": title_tokens,
        "art_body_text": art_clean["body_text"].tolist(),
        "art_title": art["title"].fillna("").tolist(),
        "cal_id": cal["query_id"].tolist(),
        "cal_tokens": cal_tokens,
        "cal_text": cal["query_text"].tolist(),
        "cal_gt": cal["ground_truth"].tolist(),
        "test_id": test["query_id"].tolist(),
        "test_tokens": test_tokens,
        "test_text": test["query_text"].tolist(),
    }, CACHE + "/prep.pkl")
    print(f"cached total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    build()

import os, hashlib, time
os.environ["OMP_NUM_THREADS"]="4"; os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import re, numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from dense import encode, minmax

CACHE="solution/cache"
P=pd.read_pickle(CACHE+"/prep.pkl")
aid=np.array(P["article_id"]); nA=len(aid)
titles,bodies=P["art_title"],P["art_body_text"]
M="intfloat/multilingual-e5-large"
CHARS,OV,MC=1100,200,8
owner,chunks=[],[]
for i in range(nA):
    b=bodies[i]
    if len(b)<=CHARS: pc=[b]
    else:
        pc,s=[],0
        while s<len(b) and len(pc)<MC: pc.append(b[s:s+CHARS]); s+=CHARS-OV
    for p in pc: chunks.append(f"{titles[i]}. {p}".strip()); owner.append(i)
owner=np.array(owner)
key=hashlib.md5(("chunk|"+M+"|"+str(len(chunks))+chunks[0][:40]+chunks[-1][:40]).encode()).hexdigest()[:12]
cemb=np.load(f"{CACHE}/chunk_{key}.npy")
GRE=re.compile(r"(здравствуйте|добрый день|добрый вечер|доброе утро|привет|пожалуйста|подскажите|скажите|спасибо|доброго времени суток)[\s,!.]*",re.I)
clean_q=[GRE.sub("",q).strip() for q in P["test_text"]]
qL=encode(M,clean_q,"query: ",batch=16); sims=qL@cemb.T; n_q=sims.shape[0]
by_art=[[] for _ in range(nA)]
for c in range(len(owner)): by_art[owner[c]].append(c)
EL=np.zeros((n_q,nA)); best_chunk=np.zeros((n_q,nA),dtype=int)
for a,cols in enumerate(by_art):
    cols=np.array(cols); sub=sims[:,cols]; tk=min(2,sub.shape[1])
    EL[:,a]=np.partition(sub,-tk,axis=1)[:,-tk:].mean(axis=1)
    best_chunk[:,a]=cols[np.argmax(sub,axis=1)]
bm=BM25Okapi(P["art_tokens"],k1=2.0,b=0.5); BM=np.array([bm.get_scores(q) for q in P["test_tokens"]])
HYB=minmax(EL)+0.3*minmax(BM)
CB_CACHE256=f"{CACHE}/cb_test_max_top100.npy"
if os.path.exists(CB_CACHE256):
    CB=np.load(CB_CACHE256)
else:
    dv=np.load(f"{CACHE}/cb_docs_vecs.npy",mmap_mode="r"); doff=np.load(f"{CACHE}/cb_docs_offs.npy")
    qv=np.load(f"{CACHE}/cb_testq_vecs.npy"); qoff=np.load(f"{CACHE}/cb_testq_offs.npy")
    cand100=np.argsort(-HYB,axis=1)[:,:100]
    CB=np.zeros((n_q,nA))
    for qi in range(n_q):
        q=qv[qoff[qi]:qoff[qi+1]].astype(np.float32)
        for a in cand100[qi]:
            best=-1e9
            for c in by_art[a]:
                d=dv[doff[c]:doff[c+1]].astype(np.float32)
                s=(q@d.T).max(axis=1).mean()
                if s>best: best=s
            CB[qi,a]=best
    np.save(CB_CACHE256,CB)
BASE=HYB+0.6*minmax(CB)

TOPK=20
cand=np.argsort(-BASE,axis=1)[:,:TOPK]
CE_CACHE=f"{CACHE}/ce_mini_test_top{TOPK}.npy"
if not os.path.exists(CE_CACHE):
    import torch
    torch.set_num_threads(4)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    D="solution/models/mmarco-minilm"
    tok=AutoTokenizer.from_pretrained(D)
    model=AutoModelForSequenceClassification.from_pretrained(D); model.eval()
    print("mini loaded",flush=True)
    pairs,meta=[],[]
    for qi in range(n_q):
        for a in cand[qi]:
            pairs.append((clean_q[qi],chunks[best_chunk[qi,a]])); meta.append((qi,a))
    sc=[]; t0=time.time()
    with torch.no_grad():
        for i in range(0,len(pairs),32):
            b=pairs[i:i+32]
            enc=tok([x[0] for x in b],[x[1] for x in b],padding=True,truncation=True,max_length=384,return_tensors="pt")
            sc.append(model(**enc).logits.reshape(-1).numpy())
            if (i//32)%50==0: print(f"  {i}/{len(pairs)} {time.time()-t0:.0f}s",flush=True)
    sc=np.concatenate(sc)
    RS=np.full((n_q,nA),-1e9)
    for (qi,a),s in zip(meta,sc): RS[qi,a]=s
    np.save(CE_CACHE,RS)
    print("saved",CE_CACHE,flush=True)
else:
    print("already cached",flush=True)

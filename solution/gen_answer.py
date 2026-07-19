import os, hashlib, time
os.environ["OMP_NUM_THREADS"]="4"; os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import re, numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from dense import encode, minmax

CACHE="solution/cache"
P=pd.read_pickle(CACHE+"/prep.pkl")
aid=np.array(P["article_id"]); nA=len(aid)
test_id=P["test_id"]; test_text=P["test_text"]
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
clean_q=[GRE.sub("",q).strip() for q in test_text]
qL=encode(M,clean_q,"query: ",batch=16); sims=qL@cemb.T; n_q=sims.shape[0]

by_art=[[] for _ in range(nA)]
for c in range(len(owner)): by_art[owner[c]].append(c)
EL=np.zeros((n_q,nA))
for a,cols in enumerate(by_art):
    cols=np.array(cols); sub=sims[:,cols]; tk=min(2,sub.shape[1])
    EL[:,a]=np.partition(sub,-tk,axis=1)[:,-tk:].mean(axis=1)
bm=BM25Okapi(P["art_tokens"],k1=2.0,b=0.5); BM=np.array([bm.get_scores(q) for q in P["test_tokens"]])
HYB=minmax(EL)+0.3*minmax(BM)
print("hybrid built on test",flush=True)

K=100
CB_CACHE=f"{CACHE}/cb384_test_max_top{K}.npy"
if os.path.exists(CB_CACHE):
    CB=np.load(CB_CACHE)
else:
    dv=np.load(f"{CACHE}/cb384_docs_vecs.npy",mmap_mode="r"); doff=np.load(f"{CACHE}/cb384_docs_offs.npy")
    qv=np.load(f"{CACHE}/cb_testq_vecs.npy"); qoff=np.load(f"{CACHE}/cb_testq_offs.npy")
    cand=np.argsort(-HYB,axis=1)[:,:K]
    CB=np.zeros((n_q,nA)); t0=time.time()
    for qi in range(n_q):
        q=qv[qoff[qi]:qoff[qi+1]].astype(np.float32)
        for a in cand[qi]:
            best=-1e9
            for c in by_art[a]:
                d=dv[doff[c]:doff[c+1]].astype(np.float32)
                s=(q@d.T).max(axis=1).mean()
                if s>best: best=s
            CB[qi,a]=best
        if qi%100==0: print(f"  q{qi}/{n_q} {time.time()-t0:.0f}s",flush=True)
    np.save(CB_CACHE,CB)

RS=np.load(f"{CACHE}/ce_mini_test_top20.npy")
RSc=np.where(RS<=-1e8,np.nanmin(np.where(RS<=-1e8,np.nan,RS)),RS)
de=np.load(f"{CACHE}/berta_docs.npy"); qe=np.load(f"{CACHE}/berta_testq.npy")
bsims=qe@de.T
BE=np.zeros((n_q,nA))
for a,cols in enumerate(by_art):
    cols=np.array(cols); sub=bsims[:,cols]; tk=min(2,sub.shape[1])
    BE[:,a]=np.partition(sub,-tk,axis=1)[:,-tk:].mean(axis=1)
FINAL=HYB+0.7*minmax(CB)+0.15*minmax(RSc)+0.8*minmax(BE)
rows=[]
for qi in range(n_q):
    idx=np.argsort(-FINAL[qi])[:10]
    rows.append((test_id[qi]," ".join(str(int(aid[i])) for i in idx)))
out=pd.DataFrame(rows,columns=["query_id","answer"])
out.to_csv("solution/answer.csv",index=False)
print(f"wrote solution/answer.csv rows={len(out)}",flush=True)
print(out.head(3).to_string(),flush=True)

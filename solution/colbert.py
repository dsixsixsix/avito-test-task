import os, sys, hashlib, time
os.environ["OMP_NUM_THREADS"]="4"; os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import re, numpy as np, pandas as pd

CACHE="solution/cache"
D="solution/models/bge-m3"
P=pd.read_pickle(CACHE+"/prep.pkl")
aid=np.array(P["article_id"]); nA=len(aid)
titles,bodies=P["art_title"],P["art_body_text"]

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
GRE=re.compile(r"(здравствуйте|добрый день|добрый вечер|доброе утро|привет|пожалуйста|подскажите|скажите|спасибо|доброго времени суток)[\s,!.]*",re.I)

MAXLEN_DOC=int(os.environ.get("CB_MAXLEN","256")); MAXLEN_Q=64; BATCH=16
DOC_TAG="cb_docs" if MAXLEN_DOC==256 else f"cb{MAXLEN_DOC}_docs"

def load_model():
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(D)
    model=AutoModel.from_pretrained(D); model.eval()
    lin=torch.nn.Linear(1024,1024,bias=True)
    lin.load_state_dict(torch.load(D+"/colbert_linear.pt",map_location="cpu"))
    lin.eval()
    return torch,tok,model,lin

def encode_colbert(texts,maxlen,out_prefix):
    vec_file=f"{CACHE}/{out_prefix}_vecs.npy"; off_file=f"{CACHE}/{out_prefix}_offs.npy"
    if os.path.exists(vec_file) and os.path.exists(off_file):
        print("already cached",flush=True); return
    torch,tok,model,lin=load_model()
    SHARD=400; shard_files=[]; buf=[]; lens=[]
    t0=time.time()
    with torch.no_grad():
        for i in range(0,len(texts),BATCH):
            b=texts[i:i+BATCH]
            enc=tok(b,padding=True,truncation=True,max_length=maxlen,return_tensors="pt")
            out=model(**enc).last_hidden_state
            cb=lin(out)
            cb=torch.nn.functional.normalize(cb,dim=-1)
            mask=enc["attention_mask"].bool()
            for j in range(len(b)):
                v=cb[j][mask[j]][1:]
                buf.append(v.numpy().astype(np.float16)); lens.append(v.shape[0])
            if len(buf)>=SHARD:
                sf=f"{CACHE}/{out_prefix}_shard{len(shard_files)}.npy"
                np.save(sf,np.concatenate(buf,axis=0)); shard_files.append(sf); buf=[]
            if (i//BATCH)%20==0:
                el=time.time()-t0; done=i+len(b)
                print(f"  {done}/{len(texts)} {el:.0f}s ({done/max(el,1e-9):.1f} t/s)",flush=True)
    if buf:
        sf=f"{CACHE}/{out_prefix}_shard{len(shard_files)}.npy"
        np.save(sf,np.concatenate(buf,axis=0)); shard_files.append(sf); buf=[]
    del model,lin; import gc; gc.collect()
    offs=np.zeros(len(lens)+1,dtype=np.int64); offs[1:]=np.cumsum(lens)
    total=int(offs[-1])
    mm=np.lib.format.open_memmap(vec_file,mode="w+",dtype=np.float16,shape=(total,1024))
    pos=0
    for sf in shard_files:
        a=np.load(sf); mm[pos:pos+a.shape[0]]=a; pos+=a.shape[0]; del a; os.remove(sf)
    mm.flush(); del mm
    np.save(off_file,offs)
    print(f"saved {vec_file} ({total},1024) {(total*1024*2)/1e6:.0f}MB",flush=True)

def get_hyb(split):
    from rank_bm25 import BM25Okapi
    from dense import encode, minmax
    M="intfloat/multilingual-e5-large"
    key=hashlib.md5(("chunk|"+M+"|"+str(len(chunks))+chunks[0][:40]+chunks[-1][:40]).encode()).hexdigest()[:12]
    cemb=np.load(f"{CACHE}/chunk_{key}.npy")
    texts=P[f"{split}_text"]; toks=P[f"{split}_tokens"]
    clean=[GRE.sub("",q).strip() for q in texts]
    qL=encode(M,clean,"query: ",batch=16); sims=qL@cemb.T; n_q=sims.shape[0]
    by_art=[[] for _ in range(nA)]
    for c in range(len(owner)): by_art[owner[c]].append(c)
    EL=np.zeros((n_q,nA)); bc=np.zeros((n_q,nA),dtype=int)
    for a,cols in enumerate(by_art):
        cols=np.array(cols); sub=sims[:,cols]; tk=min(2,sub.shape[1])
        EL[:,a]=np.partition(sub,-tk,axis=1)[:,-tk:].mean(axis=1)
        bc[:,a]=cols[np.argmax(sub,axis=1)]
    bm=BM25Okapi(P["art_tokens"],k1=2.0,b=0.5); BM=np.array([bm.get_scores(q) for q in toks])
    HYB=minmax(EL)+0.3*minmax(BM)
    return HYB,bc,clean,n_q

def maxsim(qv,dv):
    s=qv.astype(np.float32)@dv.astype(np.float32).T
    return float(s.max(axis=1).mean())

def encode_dense(texts,maxlen,out_file):
    if os.path.exists(out_file):
        print("already cached",flush=True); return
    torch,tok,model,lin=load_model()
    out=[]; t0=time.time()
    with torch.no_grad():
        for i in range(0,len(texts),BATCH):
            b=texts[i:i+BATCH]
            enc=tok(b,padding=True,truncation=True,max_length=maxlen,return_tensors="pt")
            h=model(**enc).last_hidden_state[:,0]
            h=torch.nn.functional.normalize(h,dim=-1)
            out.append(h.numpy().astype(np.float32))
            if (i//BATCH)%20==0:
                el=time.time()-t0; print(f"  {i+len(b)}/{len(texts)} {el:.0f}s",flush=True)
    np.save(out_file,np.concatenate(out,axis=0))
    print("saved",out_file,flush=True)

TOPK=20
cmd=sys.argv[1]
if cmd=="dense_docs":
    encode_dense(chunks,MAXLEN_DOC,f"{CACHE}/m3d_docs.npy")
elif cmd in ("dense_cal","dense_test"):
    split=cmd.split("_")[1]
    texts=[GRE.sub("",q).strip() for q in P[f"{split}_text"]]
    encode_dense(texts,MAXLEN_Q,f"{CACHE}/m3d_{split}q.npy")
elif cmd=="encode_docs":
    encode_colbert(chunks,MAXLEN_DOC,DOC_TAG)
elif cmd in ("encode_cal","encode_test"):
    split=cmd.split("_")[1]
    texts=[GRE.sub("",q).strip() for q in P[f"{split}_text"]]
    encode_colbert(texts,MAXLEN_Q,f"cb_{split}q")
elif cmd in ("score_cal","score_test"):
    split=cmd.split("_")[1]
    HYB,bc,clean,n_q=get_hyb(split)
    dv=np.load(f"{CACHE}/cb_docs_vecs.npy",mmap_mode="r"); doff=np.load(f"{CACHE}/cb_docs_offs.npy")
    qv=np.load(f"{CACHE}/cb_{split}q_vecs.npy"); qoff=np.load(f"{CACHE}/cb_{split}q_offs.npy")
    cand=np.argsort(-HYB,axis=1)[:,:TOPK]
    CB=np.full((n_q,nA),-1e9)
    t0=time.time()
    for qi in range(n_q):
        q=qv[qoff[qi]:qoff[qi+1]]
        for a in cand[qi]:
            c=bc[qi,a]
            CB[qi,a]=maxsim(q,dv[doff[c]:doff[c+1]])
        if qi%100==0: print(f"  q{qi}/{n_q} {time.time()-t0:.0f}s",flush=True)
    np.save(f"{CACHE}/cb_{split}_scores.npy",CB)
    if split=="cal":
        from common import mapk, parse_gt
        from dense import rank_from_scores
        gt=parse_gt(pd.Series(P["cal_gt"]))
        print("hybrid:",round(mapk(gt,rank_from_scores(HYB)),4),flush=True)
        print("colbert alone:",round(mapk(gt,rank_from_scores(CB)),4),flush=True)
        for w in (0.0,0.2,0.3,0.5,0.8,1.0,1.5):
            out=np.full((n_q,nA),-1e9)
            for qi in range(n_q):
                cols=cand[qi]
                cb=CB[qi,cols]; cb=(cb-cb.mean())/(cb.std()+1e-9)
                hs=HYB[qi,cols]; hs=(hs-hs.mean())/(hs.std()+1e-9)
                out[qi,cols]=cb+w*hs
            print(f"blend CB + {w}*HYB: {mapk(gt,rank_from_scores(out)):.4f}",flush=True)

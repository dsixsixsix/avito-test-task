import os, sys, time, hashlib
os.environ["OMP_NUM_THREADS"]="4"; os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
import re, numpy as np, pandas as pd

CACHE="solution/cache"
D=os.environ.get("ENC_DIR","solution/models/berta")
TAG=os.environ.get("ENC_TAG","berta")
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

def enc(texts,prefix,out_file,maxlen=384,batch=32):
    if os.path.exists(out_file): return np.load(out_file)
    import torch; torch.set_num_threads(4)
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    tok=AutoTokenizer.from_pretrained(D)
    cfg=AutoConfig.from_pretrained(D)
    dtype=torch.bfloat16 if os.environ.get("ENC_BF16")=="1" else torch.float32
    if "T5EncoderModel" in (cfg.architectures or []):
        from transformers import T5EncoderModel
        model=T5EncoderModel.from_pretrained(D,torch_dtype=dtype)
    else:
        model=AutoModel.from_pretrained(D,torch_dtype=dtype)
    model.eval()
    pool=os.environ.get("ENC_POOL","mean")
    out=[]; t0=time.time()
    with torch.no_grad():
        for i in range(0,len(texts),batch):
            b=[prefix+t for t in texts[i:i+batch]]
            e=tok(b,padding=True,truncation=True,max_length=maxlen,return_tensors="pt")
            h=model(**e).last_hidden_state.float()
            if pool=="cls":
                v=h[:,0]
            else:
                m=e["attention_mask"].unsqueeze(-1).float()
                v=(h*m).sum(1)/m.sum(1)
            v=torch.nn.functional.normalize(v,dim=-1)
            out.append(v.numpy().astype(np.float32))
            if (i//batch)%20==0: print(f"  {i+len(b)}/{len(texts)} {time.time()-t0:.0f}s",flush=True)
    E=np.concatenate(out,axis=0); np.save(out_file,E); return E

cmd=sys.argv[1]
if cmd=="docs":
    enc(chunks,"search_document: ",f"{CACHE}/{TAG}_docs.npy")
else:
    texts=[GRE.sub("",q).strip() for q in P[f"{cmd}_text"]]
    enc(texts,"search_query: ",f"{CACHE}/{TAG}_{cmd}q.npy")
print("done",flush=True)

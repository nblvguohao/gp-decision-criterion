"""Minimal reader for R .rda / .RData / .rds files (serialization format 2/3, XDR).

Enough of the format to recover data.frames and named lists: no R install,
no pyreadr build. Handles NULL, symbols, pairlists, character/integer/real/
logical vectors, generic vectors, references, and the compact_intseq ALTREP
used for row.names.
"""
import bz2, gzip, lzma, struct, numpy as np, pandas as pd

NILVALUE,REFSXP,ALTREP = 254,255,238
NILSXP,SYMSXP,LISTSXP,CHARSXP,LGLSXP,INTSXP,REALSXP,STRSXP,VECSXP = 0,1,2,9,10,13,14,16,19
NA_INT=-2**31

def _open(path):
    raw=open(path,"rb").read()
    if raw[:2]==b"\x1f\x8b": raw=gzip.decompress(raw)
    elif raw[:2]==b"BZ":     raw=bz2.decompress(raw)
    elif raw[:5]==b"\xfd7zXZ": raw=lzma.decompress(raw)
    return raw

class _R:
    def __init__(self,buf):
        self.b=buf; self.i=0; self.refs=[]
    def int(self):
        v=struct.unpack(">i",self.b[self.i:self.i+4])[0]; self.i+=4; return v
    def dbl(self,n):
        a=np.frombuffer(self.b,dtype=">f8",count=n,offset=self.i); self.i+=8*n
        return a.astype(np.float64)
    def ints(self,n):
        a=np.frombuffer(self.b,dtype=">i4",count=n,offset=self.i); self.i+=4*n
        return a.astype(np.int64)
    def raw(self,n):
        s=self.b[self.i:self.i+n]; self.i+=n; return s
    def obj(self):
        flags=self.int(); t=flags&0xFF
        has_attr=(flags>>9)&1; has_tag=(flags>>10)&1
        if t in (NILVALUE,NILSXP): return None
        if t==REFSXP:
            idx=flags>>8
            if idx==0: idx=self.int()
            return self.refs[idx-1]
        if t==SYMSXP:
            v=self.obj(); self.refs.append(v); return v
        if t==CHARSXP:
            n=self.int()
            if n==-1: return None
            return self.raw(n).decode("utf-8","replace")
        if t==LISTSXP:
            out={}
            while True:
                tag=self.obj() if has_tag else None
                val=self.obj()
                if tag is not None: out[tag]=val
                flags=self.int(); t=flags&0xFF
                if t in (NILVALUE,NILSXP): return out
                has_tag=(flags>>10)&1
        if t==STRSXP:
            n=self.int(); v=[self.obj() for _ in range(n)]
        elif t==INTSXP:
            n=self.int(); a=self.ints(n).astype(float); a[a==NA_INT]=np.nan; v=a
        elif t==LGLSXP:
            n=self.int(); a=self.ints(n).astype(float); a[a==NA_INT]=np.nan; v=a
        elif t==REALSXP:
            n=self.int(); v=self.dbl(n)
        elif t==VECSXP:
            n=self.int(); v=[self.obj() for _ in range(n)]
        elif t==ALTREP:
            info=self.obj(); state=self.obj(); self.obj()
            if isinstance(state,(list,np.ndarray)) and len(state)==3:
                n,start,step=[float(x) for x in np.asarray(state,dtype=float)]
                v=np.arange(start,start+n*step,step)[:int(n)]
            else: v=state
            return v
        else:
            raise NotImplementedError(f"SEXP type {t}")
        attrs=self.obj() if has_attr else None
        if attrs:
            names=attrs.get("names"); cls=attrs.get("class")
            if isinstance(cls,list) and "data.frame" in cls and isinstance(v,list):
                return pd.DataFrame({n_:c for n_,c in zip(names,v)})
            if names is not None and isinstance(v,list):
                return dict(zip(names,v))
            if attrs.get("dim") is not None:
                d=[int(x) for x in np.asarray(attrs["dim"],dtype=float)]
                v=np.asarray(v).reshape(d,order="F")
        return v

def read(path):
    raw=_open(path)
    j=raw.find(b"X\n")
    r=_R(raw[j+2:])
    ver=r.int(); r.int(); r.int()
    if ver>=3:
        n=r.int(); r.raw(n)
    out={}
    while r.i < len(r.b)-4:
        try:
            flags=r.int(); t=flags&0xFF
            if t in (NILVALUE,NILSXP): break
            r.i-=4
            name=None
            if t==LISTSXP:
                flags=r.int(); has_tag=(flags>>10)&1
                name=r.obj() if has_tag else None
                val=r.obj()
                out[name or f"obj{len(out)}"]=val
                nf=r.int()
                if (nf&0xFF) in (NILVALUE,NILSXP): break
                r.i-=4
            else:
                out[f"obj{len(out)}"]=r.obj(); break
        except Exception: break
    return out

if __name__=="__main__":
    import sys
    for f in sys.argv[1:]:
        print(f"### {f}")
        try:
            d=read(f)
            for k,v in d.items():
                if isinstance(v,pd.DataFrame): print(f"  {k}: DataFrame {v.shape}  cols {list(v.columns)[:10]}")
                elif isinstance(v,dict):       print(f"  {k}: dict keys {list(v)[:12]}")
                elif isinstance(v,np.ndarray): print(f"  {k}: array {v.shape} {v.dtype}")
                else:                          print(f"  {k}: {type(v).__name__} {str(v)[:80]}")
        except Exception as e: print("  FAILED:",type(e).__name__,e)

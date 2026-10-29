"""Pixel/identity overlap quarantine. Similarity is conservative, not proof of duplication."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from scipy.fft import dctn
from .common import sha,write_json,digest
from .data import ISIC_METADATA_POLICY


def image_records(frame):
    rows=[]
    for r in frame.itertuples():
        for view in ('clinical','dermoscopic'):
            for path in json.loads(getattr(r,view)):
                key=digest([r.source,r.case_id,view,Path(path).stem])[:32]
                rows.append({'key':key,'case_id':r.case_id,'group_id':r.group_id,'source':r.source,
                             'split':r.split,'view':view,'path':path,'image_id':Path(path).stem})
    return rows


def fingerprint(record):
    path=Path(record['path']);stat=path.stat()
    with Image.open(path) as im:
        rgb=ImageOps.exif_transpose(im).convert('RGB')
        pixel=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        base=rgb.resize((32,32),Image.Resampling.LANCZOS).convert('L');hashes=[]
        for reflected in [False,True]:
            image=base.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if reflected else base
            for rotation in (0,90,180,270):
                low=dctn(np.asarray(image.rotate(rotation),dtype=float),norm='ortho')[:8,:8].flatten()
                bits=low>np.median(low[1:]);hashes.append(int.from_bytes(np.packbits(bits).tobytes(),'big'))
    return {**record,'sha256':sha(path),'pixel_sha256':pixel,'phash':[str(x) for x in hashes],
            'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def fingerprints(frame, cache, workers=2):
    records=image_records(frame);cache=Path(cache);prior={}
    if cache.exists():prior={r['key']:r for r in json.loads(cache.read_text())}
    def one(r):
        old=prior.get(r['key']);s=Path(r['path']).stat()
        if old and old['path']==r['path'] and old['size']==s.st_size and old['mtime_ns']==s.st_mtime_ns:
            return {**old,**r}
        return fingerprint(r)
    result=[];start=time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1,workers)) as pool:
        for i,r in enumerate(pool.map(one,records),1):
            result.append(r)
            if i%250==0 or i==len(records):
                print(f'Overlap audit {i}/{len(records)} images ({time.monotonic()-start:.1f}s)',flush=True)
                if i%2000==0:write_json(cache,result)
    write_json(cache,result);return result


class FingerprintIndex:
    """Pigeonhole index: <=radius bit changes imply one unchanged segment."""
    def __init__(self, radius=3):
        if not 0<=radius<=8:raise ValueError('pHash radius must be 0..8')
        self.radius=radius;self.exact={};self.ids={};self.buckets={};self.entries=[]
        self.parts=radius+1
    def segments(self,value):
        for i in range(self.parts):
            lo=64*i//self.parts;hi=64*(i+1)//self.parts
            yield i,(value>>lo)&((1<<(hi-lo))-1)
    def add(self,r):
        index=len(self.entries);self.entries.append(r)
        self.exact.setdefault(r['sha256'],set()).add(index)
        self.exact.setdefault(r['pixel_sha256'],set()).add(index)
        self.ids.setdefault(r['image_id'],set()).add(index)
        for v in set(map(int,r['phash'])):
            for segment in self.segments(v):self.buckets.setdefault(segment,set()).add((index,v))
    def matches(self,r):
        found={i:'exact_file_or_pixels' for h in [r['sha256'],r['pixel_sha256']] for i in self.exact.get(h,())}
        found.update({i:'shared_image_identifier' for i in self.ids.get(r['image_id'],())})
        v=int(r['phash'][0]);candidates=set()
        for segment in self.segments(v):candidates.update(self.buckets.get(segment,()))
        for i,old in candidates:
            if (v^old).bit_count()<=self.radius:found.setdefault(i,'conservative_near_duplicate')
        return [(self.entries[i],why) for i,why in found.items()]


def quarantine_development(frame, records, radius=3):
    index=FingerprintIndex(radius);reject=set();audit=[]
    metadata_cases=[]
    if 'metadata_quarantine' in frame:
        flagged=frame[frame.metadata_quarantine.fillna(False).astype(bool)]
        reject.update(flagged.group_id)
        metadata_cases=[{'case_id':r.case_id,'group_id':r.group_id,
                         'reasons':json.loads(r.metadata_quarantine_reasons),
                         'observations':json.loads(r.metadata_observations)}
                        for r in flagged.itertuples()]
    # Audit every MILK partition, including diagnoses excluded from this new endpoint.
    for r in records:
        if r['source']=='MILK':
            for previous,reason in index.matches(r):
                if (previous['group_id']!=r['group_id'] and previous['split']!=r['split']
                    and reason=='exact_file_or_pixels'):
                    reject.update([previous['group_id'],r['group_id']])
                    audit.append({'case':r['case_id'],'matched_case':previous['case_id'],'reason':reason,'action':'Quarantine both groups in the NEW study; frozen split unchanged'})
                elif previous['group_id']!=r['group_id'] and previous['split']!=r['split']:
                    reject.update([previous['group_id'],r['group_id']])
                    audit.append({'case':r['case_id'],'matched_case':previous['case_id'],'reason':reason,'action':'Conservative cross-split MILK quarantine in new study only'})
            index.add(r)
    for r in records:
        if r['source']!='ISIC2019':continue
        for previous,reason in index.matches(r):
            if previous['group_id']==r['group_id']:continue
            reject.add(r['group_id'])
            if previous['source']=='ISIC2019':reject.add(previous['group_id'])
            audit.append({'case':r['case_id'],'matched_case':previous['case_id'],'reason':reason})
        index.add(r)
    result=frame[(frame.binary>=0)&~frame.group_id.isin(reject)].reset_index(drop=True)
    return result,{'quarantined_groups':sorted(reject),'matches':audit,'radius':radius,
                   'metadata_policy':ISIC_METADATA_POLICY,
                   'metadata_conflict_cases':metadata_cases,
                   'metadata_conflict_groups':sorted({r['group_id'] for r in metadata_cases}),
                   'isic_cases_with_varying_age':int(frame.get('metadata_age_varied',pd.Series(False,index=frame.index)).fillna(False).astype(bool).sum()),
                   'excluded_diagnosis_cases':int((frame.binary<0).sum()),
                   'limitations':'Conservative pHash quarantine may exclude similar distinct lesions; cropped duplicates can remain.'}


def quarantine_external(frame, records, development_records, radius=3):
    index=FingerprintIndex(radius)
    for r in development_records:index.add(r)
    reject=set();audit=[]
    for r in records:
        for previous,reason in index.matches(r):
            if previous['source']=='PAD' and previous['group_id']==r['group_id']:continue
            reject.add(r['group_id'])
            if previous['source']=='PAD':reject.add(previous['group_id'])
            audit.append({'case':r['case_id'],'matched_case':previous['case_id'],'reason':reason})
        index.add(r)
    result=frame[(frame.binary>=0)&~frame.group_id.isin(reject)].reset_index(drop=True)
    return result,{'quarantined_patient_groups':sorted(reject),'matches':audit,'radius':radius,
                   'excluded_diagnosis_cases':int((frame.binary<0).sum()),
                   'original_cases':len(frame),'eligible_test_cases':len(result),
                   'policy':'Whole external patient quarantined using pixels/IDs only, never model errors or labels.'}

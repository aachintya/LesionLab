"""Source-specific labels, lesion aggregation and train-only shared metadata."""
from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .common import write_json

CLASSES = ('AKIEC','BCC','BEN_OTH','BKL','DF','INF','MAL_OTH','MEL','NV','SCCKA','VASC')
LABEL_POLICY = 'malignancy-including-scc-in-situ-v1; AK-and-KA-excluded'
SUBTYPES = ('BCC','MEL','SCC','OTHER_MALIGNANT')
SITES = ('head_neck','trunk','upper_limb','lower_limb','acral','genital','other','unknown')
UNKNOWN = {'','nan','none','unk','unknown','na','n/a'}
ISIC_METADATA_POLICY = 'isic-lesion-metadata-v2: valid-age-median; categorical-conflict-group-quarantine'


def milk_label(target, granular):
    target = int(target); text = str(granular).strip()
    if target not in range(11) or not text or text.lower() in UNKNOWN:
        raise ValueError('MILK reference diagnosis is missing/invalid')
    if target == 0:
        if text == 'Solar or actinic keratosis': return -1, -1
        if text == 'Squamous cell carcinoma in situ, Bowens disease': return 1, 2
        raise ValueError(f'Unreviewed AKIEC diagnosis: {text}')
    if target == 9:
        if text == 'Keratoacanthoma': return -1, -1
        if text == 'Squamous cell carcinoma, Invasive': return 1, 2
        raise ValueError(f'Unreviewed SCCKA diagnosis: {text}')
    return (1, {1:0,7:1,6:3}[target]) if target in (1,6,7) else (0,-1)


def source_label(code, source):
    mapping = {'ISIC2019': {'BCC':(1,0),'MEL':(1,1),'SCC':(1,2),'AK':(-1,-1),
                          'NV':(0,-1),'BKL':(0,-1),'DF':(0,-1),'VASC':(0,-1),'UNK':(-1,-1)},
               'PAD': {'BCC':(1,0),'MEL':(1,1),'SCC':(1,2),'ACK':(-1,-1),
                       'NEV':(0,-1),'SEK':(0,-1)}}
    if source not in mapping or code not in mapping[source]:
        raise ValueError(f'Unreviewed {source} diagnosis: {code}')
    return mapping[source][code]


def safe_id(value):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value): raise ValueError(f'Invalid identifier: {value}')
    return value


class ImageIndex:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.paths = {}
        print(f'Indexing source images: {self.root}', flush=True)
        for p in self.root.rglob('*'):
            if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                if not p.resolve().is_relative_to(self.root): raise ValueError('Image escapes dataset root')
                key = p.stem
                if key in self.paths: raise ValueError(f'Ambiguous image basename: {key}')
                self.paths[key] = str(p.resolve())
                if len(self.paths)%5000 == 0:
                    print(f'Indexed {len(self.paths)} source images', flush=True)
        if not self.paths: raise FileNotFoundError(f'No extracted images under {root}')
        print(f'Image index ready: {len(self.paths)} images', flush=True)

    def get(self, name):
        if Path(str(name)).name != str(name):raise ValueError('Image identifiers must be filenames, not paths')
        key = safe_id(Path(str(name)).stem)
        if key not in self.paths: raise FileNotFoundError(f'Image {key} not found in {self.root}')
        return self.paths[key]


def value(group, name):
    if name not in group: return None
    known = group[name].dropna().astype(str).str.strip()
    known = known[~known.str.lower().isin(UNKNOWN)].unique()
    if len(known) > 1: raise ValueError(f'Conflicting lesion metadata: {name}')
    return known[0] if len(known) else None


def isic_metadata(group):
    """Aggregate covariates without labels; quarantine ambiguous lesion identities.

    Repeated images may report different rounded ages. The median of valid
    observations is deterministic. Conflicting sex or coarse site may indicate
    a reused lesion identifier, so the whole group is excluded downstream.
    Raw observations are retained for audit; image/diagnosis checks stay strict.
    """
    ages=pd.to_numeric(group.age_approx,errors='coerce')
    valid=ages[ages.between(0,120)]
    age=float(valid.median()) if len(valid) else None
    unique_ages=sorted(float(x) for x in valid.unique())
    def known(name):
        return sorted({str(x).strip().lower() for x in group[name].dropna()
                       if str(x).strip().lower() not in UNKNOWN})
    sexes=known('sex');raw_sites=known('anatom_site_general')
    sites=sorted({site_code(x) for x in raw_sites}-{ 'unknown' })
    reasons=[]
    if len(sexes)>1:reasons.append('conflicting_sex')
    if set(sexes)-{'male','female'}:reasons.append('unrecognized_sex')
    if len(sites)>1:reasons.append('conflicting_coarse_site')
    return age,(sexes[0] if len(sexes)==1 else None),(sites[0] if len(sites)==1 else None),{
        'metadata_policy':ISIC_METADATA_POLICY,
        'metadata_quarantine':bool(reasons),
        'metadata_quarantine_reasons':json.dumps(reasons),
        'metadata_age_varied':len(unique_ages)>1,
        'metadata_observations':json.dumps({'ages':unique_ages,'valid_age_rows':len(valid),
            'missing_or_invalid_age_rows':int(len(ages)-len(valid)),
            'sex':sexes,'raw_sites':raw_sites,'coarse_sites':sites},sort_keys=True)}


def _case(source, lesion, group, split, clinical, dermoscopic, target, subtype, age, sex, site, **extra):
    return {'case_id':f'{source}:{safe_id(lesion)}','source':source,'group_id':f'{source}:{safe_id(group)}',
            'split':split,'clinical':json.dumps(clinical),'dermoscopic':json.dumps(dermoscopic),
            'binary':int(target),'subtype':int(subtype),'age':age,'sex':sex,'site':site,**extra}


def milk_cases(csv, images):
    frame = pd.read_csv(csv, dtype={'lesion_id':str,'isic_id':str})
    required = {'lesion_id','isic_id','image_type','target','split','diagnosis_full','age_approx','sex','site'}
    if not required <= set(frame): raise ValueError(f'MILK CSV missing {required-set(frame)}')
    if frame.isic_id.duplicated().any(): raise ValueError('Duplicate MILK image ID')
    index = ImageIndex(images); rows = []
    patients = next((c for c in ('patient_id','case_id','subject_id','patient') if c in frame),None)
    for lesion,g in frame.groupby('lesion_id',sort=True):
        if len(g)!=2 or set(g.image_type)!= {'clinical: close-up','dermoscopic'}:
            raise ValueError(f'MILK pair incomplete: {lesion}')
        for c in ('split','target','diagnosis_full','age_approx','sex','site'):
            if g[c].nunique(dropna=False)!=1: raise ValueError(f'Conflicting MILK pair {lesion}: {c}')
        if g['split'].iloc[0] not in ('train','val','fusion','holdout'): raise ValueError('Invalid frozen split')
        target,subtype=milk_label(g.target.iloc[0],g.diagnosis_full.iloc[0])
        patient=value(g,patients) if patients else None
        rows.append(_case('MILK',lesion,patient or lesion,g['split'].iloc[0],
          [index.get(g.loc[g.image_type=='clinical: close-up','isic_id'].iloc[0])],
          [index.get(g.loc[g.image_type=='dermoscopic','isic_id'].iloc[0])],target,subtype,
          value(g,'age_approx'),value(g,'sex'),value(g,'site'),original_target=int(g.target.iloc[0]),
          patient_grouping=bool(patient),diagnosis_confirm_type=value(g,'diagnosis_confirm_type')))
    result=pd.DataFrame(rows); assert_groups(result)
    return result


def isic_cases(ground_truth, metadata, images, seed=42):
    gt=pd.read_csv(ground_truth,dtype={'image':str});meta=pd.read_csv(metadata,dtype={'image':str})
    labels=['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']
    metadata_fields={'image','lesion_id','age_approx','sex','anatom_site_general'}
    if not {'image',*labels} <= set(gt) or not metadata_fields <= set(meta):
        raise ValueError('Use official ISIC 2019 ground truth and metadata, including lesion_id, age_approx, sex and anatom_site_general')
    cols=labels+(['UNK'] if 'UNK' in gt else [])
    a=gt[cols].to_numpy(float)
    if not np.isfinite(a).all() or not np.isin(a,[0,1]).all() or not (a.sum(1)==1).all():
        raise ValueError('ISIC ground truth must be one-hot diagnoses')
    gt=gt.assign(diagnosis=[cols[i] for i in a.argmax(1)])
    if gt.image.duplicated().any() or meta.image.duplicated().any(): raise ValueError('Repeated ISIC image identity')
    merged=gt.merge(meta,on='image',how='left',validate='one_to_one',indicator=True)
    if not merged['_merge'].eq('both').all(): raise ValueError('ISIC metadata rows missing')
    index=ImageIndex(images); merged['lesion']=merged.get('lesion_id',merged.image).fillna(merged.image).astype(str)
    rows=[]
    for lesion,g in merged.groupby('lesion',sort=True):
        if g.diagnosis.nunique()!=1: raise ValueError('ISIC lesion has conflicting labels')
        target,subtype=source_label(g.diagnosis.iloc[0],'ISIC2019')
        patient=value(g,'patient_id')
        age,sex,site,metadata_audit=isic_metadata(g)
        rows.append(_case('ISIC2019',lesion,patient or lesion,'unassigned',[],[index.get(x) for x in g.image],
          target,subtype,age,sex,site,
          original_target=-1,patient_grouping=bool(patient),diagnosis_confirm_type=None,**metadata_audit))
    result=pd.DataFrame(rows)
    # Groups, not images, are assigned. All images of one lesion remain one case.
    groups=result.groupby('group_id').binary.max().clip(lower=0)
    if groups.value_counts().min()<4: raise ValueError('Insufficient groups for stratified ISIC development splits')
    fit,cal=train_test_split(groups.index,test_size=.15,random_state=seed,stratify=groups)
    train,val=train_test_split(fit,test_size=.15/.85,random_state=seed+1,stratify=groups.loc[fit])
    assignments={**{x:'train' for x in train},**{x:'val' for x in val},**{x:'fusion' for x in cal}}
    result['split']=result.group_id.map(assignments); assert_groups(result)
    print(f'ISIC metadata: {len(result)} lesions; {int(result.metadata_age_varied.sum())} with varying ages; '
          f'{int(result.metadata_quarantine.sum())} categorical-conflict cases marked for whole-group quarantine',flush=True)
    return result


def pad_cases(metadata, images):
    frame=pd.read_csv(metadata,dtype={'patient_id':str,'lesion_id':str,'img_id':str})
    required={'patient_id','lesion_id','img_id','diagnostic','age','gender','region'}
    if not required<=set(frame): raise ValueError('PAD metadata must have official patient, lesion, image and diagnosis fields')
    if frame[list(required)].isna().any().any() and frame[['patient_id','lesion_id','img_id','diagnostic']].isna().any().any():
        raise ValueError('PAD identities and diagnoses cannot be missing')
    if frame.img_id.duplicated().any(): raise ValueError('Repeated PAD image identity')
    index=ImageIndex(images); rows=[]
    for (patient,lesion),g in frame.groupby(['patient_id','lesion_id'],sort=True):
        if g.diagnostic.nunique()!=1: raise ValueError('PAD lesion labels disagree')
        target,subtype=source_label(g.diagnostic.iloc[0],'PAD')
        rows.append(_case('PAD',f'{safe_id(patient)}_{safe_id(lesion)}',patient,'external',
          [index.get(x) for x in g.img_id],[],target,subtype,value(g,'age'),value(g,'gender'),value(g,'region'),
          original_target=-1,patient_grouping=True,diagnosis_confirm_type={'true':'histopathology','1':'histopathology','false':'clinical','0':'clinical'}.get(str(value(g,'biopsed')).lower(),'unknown')))
    result=pd.DataFrame(rows);assert_groups(result)
    return result


def assert_groups(frame):
    if frame.case_id.duplicated().any(): raise ValueError('Duplicate case IDs')
    if (frame.groupby('group_id')['split'].nunique()>1).any(): raise ValueError('Patient/lesion group crosses splits')
    if not frame.binary.isin([-1,0,1]).all(): raise ValueError('Invalid binary target')
    owners={}
    for row in frame.itertuples():
        for view in ('clinical','dermoscopic'):
            for path in json.loads(getattr(row,view)):
                if path in owners and owners[path]!=row.case_id: raise ValueError('Image assigned to multiple lesions')
                owners[path]=row.case_id


def site_code(raw):
    s=str(raw).lower().strip().replace('_',' ')
    if s in UNKNOWN:return 'unknown'
    if s in ('acral','palm','palms','soles','sole'):return 'acral'
    if any(t in s for t in ('head','neck','face','scalp','ear','nose','lip')):return 'head_neck'
    if any(t in s for t in ('arm','upper','hand','forearm','wrist')):return 'upper_limb'
    if any(t in s for t in ('leg','lower','thigh','foot','feet','ankle')):return 'lower_limb'
    if any(t in s for t in ('trunk','torso','chest','back','abdomen','breast','buttock')):return 'trunk'
    if any(t in s for t in ('genital','groin')):return 'genital'
    return 'other'


class SharedMetadata:
    """Age, sex, coarse site and per-field observed flags; no labels or source input."""
    width=1+3+len(SITES)+3
    def fit(self, frame):
        if not set(frame['split'])<= {'train'}: raise ValueError('Metadata may fit training cases only')
        ages=pd.to_numeric(frame.age,errors='coerce');ages=ages.where(ages.between(0,120))
        self.mean=float(ages.mean()) if ages.notna().any() else 60.
        self.std=max(float(ages.std(ddof=0)) if ages.notna().any() else 1.,1.)
        return self
    def transform(self, frame):
        x=np.zeros((len(frame),self.width),np.float32); observed=np.zeros((len(frame),3),np.float32)
        for i,r in enumerate(frame.itertuples()):
            age=pd.to_numeric(getattr(r,'age',None),errors='coerce')
            known=bool(np.isfinite(age) and 0<=age<=120)
            x[i,0]=(float(age)-self.mean)/self.std if known else 0.; observed[i,0]=known
            sex=str(getattr(r,'sex',None)).strip().lower();sex={'male':0,'female':1}.get(sex,2)
            x[i,1+sex]=1.;observed[i,1]=sex!=2
            site=site_code(getattr(r,'site',None));x[i,4+SITES.index(site)]=1.;observed[i,2]=site!='unknown'
        x[:,-3:]=observed
        return x,observed
    def save(self,path):write_json(path,{'mean':self.mean,'std':self.std,'width':self.width,'fields':['age','sex','site']})
    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text());obj=cls();obj.mean=data['mean'];obj.std=data['std'];return obj


def mask_metadata(values, flags):
    x=values.copy(); x[:,0]*=flags[:,0];x[:,1:4]*=flags[:,1,None];x[:,4:4+len(SITES)]*=flags[:,2,None];x[:,-3:]=flags
    return x

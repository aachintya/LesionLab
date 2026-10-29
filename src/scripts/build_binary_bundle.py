"""Build a minimal standalone binary training bundle from immutable baseline bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def build(root,baseline,destination):
    from utils.baselines import verify_baseline,sha256_file
    import pandas as pd
    root,baseline,destination=map(Path,(root,baseline,destination))
    verify_baseline(baseline)
    members={}
    with zipfile.ZipFile(baseline/'payload/training_artifacts.zip') as original:
        for view in ('clinical','dermoscopic'):
            members[f'reference/{view}.pt']=original.read(f'src/checkpoints/best_{view}.pt')
    members['reference/split.csv']=(baseline/'payload/final_metadata_with_split.csv').read_bytes()
    members['reference/baseline_metrics.json']=(baseline/'metrics.json').read_bytes()
    legacy=root/'demo/evaluation/baseline_holdout_predictions.csv'
    frame=pd.read_csv(legacy);split=pd.read_csv(baseline/'payload/final_metadata_with_split.csv')
    held=split.query("split == 'holdout'").drop_duplicates('lesion_id').set_index('lesion_id')
    if len(frame)!=len(held) or set(frame.lesion_id)!=set(held.index):raise ValueError('Archived prediction identities do not match frozen holdout')
    aligned=frame.set_index('lesion_id').loc[held.index]
    if not (aligned.target.to_numpy()==held.target.to_numpy()).all():raise ValueError('Archived reference labels differ')
    expected=json.loads(members['reference/baseline_metrics.json'])['accuracy']
    if abs(float((frame.target==frame.predicted).mean())-expected)>1e-12:raise ValueError('Archived baseline accuracy differs')
    members['reference/legacy_holdout.csv']=legacy.read_bytes()
    reference={'format':'binary-reference-v1','historical_baseline':baseline.name,
               'historical_manifest_sha256':sha256_file(baseline/'manifest.json'),
               'note':'Only copies of original encoders are provided. No original files are modified.',
               'sha256':{name.removeprefix('reference/'):hashlib.sha256(data).hexdigest() for name,data in members.items()}}
    members['reference/manifest.json']=(json.dumps(reference,indent=2)+'\n').encode()
    files=[*root.glob('src/binary_study/*.py'),root/'src/scripts/binary_study.py',
           root/'src/scripts/build_binary_bundle.py',root/'src/scripts/binary_kaggle_repairs.py',
           root/'src/models/efficientnet_v2_image.py',root/'src/components/preprocessing.py',
           root/'src/components/image_cache.py',root/'tests/test_binary_study.py',
           root/'tests/test_binary_study_metadata.py',root/'README.md',
           root/'notebooks/kaggle_binary_v1.ipynb']
    for path in files:
        if not path.is_file():raise FileNotFoundError(path)
        members[path.relative_to(root).as_posix()]=path.read_bytes()
    manifest={'format':'binary-code-bundle-v1','sha256':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(members.items())}}
    members['binary_bundle_manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.parent.mkdir(parents=True,exist_ok=True);temp=destination.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,data in sorted(members.items()):archive.writestr(name,data)
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip():raise ValueError('Bundle CRC verification failed')
    temp.replace(destination)
    print(f'Built {destination} ({destination.stat().st_size/1e6:.1f} MB); SHA256 {sha256_file(destination)}')
    return destination

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=root/'baselines/baseline-v1-2026-10-27')
    p.add_argument('--output',type=Path,default=root/'dist/skin_cancer_binary_v1.zip')
    args=p.parse_args();build(root,args.baseline,args.output)

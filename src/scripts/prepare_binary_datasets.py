"""Download official ISIC2019/PAD releases and build flat Kaggle upload archives.

Only dataset packaging is performed; no image model, split or outcome is fitted.
Publisher PAD SHA256 values and original CSV bytes are preserved.
"""
from __future__ import annotations
import concurrent.futures as futures
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request
import zipfile
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/external_downloads'
AUDIT=ROOT/'output/binary_dataset_preparation'
HEADERS={'User-Agent':'Mozilla/5.0','Accept':'*/*','Accept-Encoding':'identity'}
ISIC_BASE='https://isic-archive.s3.amazonaws.com/challenges/2019/'
ISIC_PAGE='https://challenge.isic-archive.com/data/'
PAD_PAGE='https://data.mendeley.com/datasets/zr7vgbcyr2/1'
CHUNK=256*1024*1024


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def request(url,method='GET',headers=None):
    # Same publisher S3 bucket/object via its regional endpoint; the global
    # endpoint intermittently stalls TLS on this network. If-Match pins ranges.
    if url.startswith('https://isic-archive.s3.amazonaws.com/'):
        url=url.replace('https://isic-archive.s3.amazonaws.com/','https://isic-archive.s3.us-east-1.amazonaws.com/',1)
    return urllib.request.urlopen(urllib.request.Request(url,method=method,headers={**HEADERS,**(headers or {})}),timeout=60)


def specifications():
    with request('https://data.mendeley.com/public-api/datasets/zr7vgbcyr2') as stream:pad=json.load(stream)
    if pad['id']!='zr7vgbcyr2' or pad['version']!=1:raise ValueError('Unexpected PAD release identity')
    write_json(AUDIT/'PAD_publisher_release.json',pad)
    items=[]
    for name in ['ISIC_2019_Training_Input.zip','ISIC_2019_Training_GroundTruth.csv','ISIC_2019_Training_Metadata.csv']:
        url=ISIC_BASE+name
        with request(url,'HEAD') as response:size=int(response.headers['Content-Length']);etag=response.headers['ETag']
        items.append({'dataset':'isic2019','name':name,'url':url,'bytes':size,'etag':etag,'source_page':ISIC_PAGE,
                      'transport_url':url.replace('isic-archive.s3.amazonaws.com','isic-archive.s3.us-east-1.amazonaws.com')})
    for file in pad['files']:
        url=f"https://data.mendeley.com/public-api/datasets/zr7vgbcyr2/files/{file['id']}/file_downloaded?version=1"
        with request(url) as response:download=json.load(response)['url']
        with request(download,'HEAD') as response:
            if int(response.headers['Content-Length'])!=file['size']:raise ValueError('Publisher PAD file size disagrees')
            etag=response.headers['ETag']
        items.append({'dataset':'pad_ufes20','name':file['filename'],'url':download,'bytes':file['size'],'etag':etag,
                      'publisher_sha256':file['content_details']['sha256_hash'],'source_page':PAD_PAGE})
    write_json(AUDIT/'download_sources.json',items)
    return items


def download_all(items):
    jobs=[];states=[]
    for item in items:
        directory=RAW/item['dataset'];directory.mkdir(parents=True,exist_ok=True)
        destination=directory/item['name'];parts=directory/(item['name']+'.chunks');parts.mkdir(exist_ok=True)
        identity={k:item[k] for k in ('url','bytes','etag')};marker=parts/'identity.json'
        if marker.exists() and json.loads(marker.read_text())!=identity:raise ValueError('Remote download identity changed')
        write_json(marker,identity)
        state={'item':item,'destination':destination,'parts':[]};states.append(state)
        if destination.exists():
            if destination.stat().st_size!=item['bytes']:raise ValueError('Completed source has an unexpected size')
            continue
        prefix=directory/(item['name']+'.part');offset=0
        if prefix.exists():
            # Reuse the sequential prefix already downloaded from the same official URL.
            saved=parts/'prefix.bin'
            if saved.exists():raise ValueError('Two existing sequential prefixes')
            prefix.rename(saved)
        else:saved=parts/'prefix.bin'
        if saved.exists():
            offset=saved.stat().st_size
            if offset>item['bytes']:raise ValueError('Partial download is longer than source')
            state['parts'].append(saved)
        for start in range(offset,item['bytes'],CHUNK):
            end=min(start+CHUNK,item['bytes'])-1;path=parts/f'{start:012d}-{end:012d}.bin'
            state['parts'].append(path);jobs.append((item,start,end,path))
    lock=threading.Lock();transferred={str(p):min(p.stat().st_size if p.exists() else 0,end-start+1)
                                      for _item,start,end,p in jobs};errors=[]
    def one(job):
        item,start,end,path=job;wanted=end-start+1
        for attempt in range(8):
            have=path.stat().st_size if path.exists() else 0
            if have==wanted:return
            if have>wanted:raise ValueError('Partial chunk is too large')
            try:
                with request(item['url'],headers={'Range':f'bytes={start+have}-{end}','If-Match':item['etag']}) as response:
                    expected=f"bytes {start+have}-{end}/{item['bytes']}"
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('Server did not honor the requested byte range')
                    with path.open('ab') as out:
                        while block:=response.read(1024*1024):
                            out.write(block)
                            with lock:transferred[str(path)]=out.tell()
                if path.stat().st_size!=wanted:raise IOError('Incomplete range')
                return
            except Exception as exc:
                if attempt==7:raise
                print(f"Retry {item['name']} range {start}: {type(exc).__name__}: {exc}",flush=True)
                time.sleep(min(2**attempt,20))
    # Interleave sources so PAD and ISIC both advance within the same worker budget.
    queues={}
    for job in jobs:queues.setdefault(job[0]['name'],[]).append(job)
    ordered=[]
    while any(queues.values()):
        for queue in queues.values():
            if queue:ordered.append(queue.pop(0))
    def finish_state(state):
        item=state['item'];destination=state['destination']
        if state.get('verified'):return
        if not destination.exists():
            temp=destination.with_name(destination.name+'.assembling')
            with temp.open('wb') as out:
                for part in state['parts']:
                    with part.open('rb') as stream:shutil.copyfileobj(stream,out,8*1024*1024)
            if temp.stat().st_size!=item['bytes']:raise ValueError('Assembled source length mismatch')
            temp.replace(destination)
        digest=sha(destination)
        if item.get('publisher_sha256') and digest!=item['publisher_sha256']:raise ValueError(f'Publisher SHA256 mismatch: {item["name"]}')
        item['downloaded_sha256']=digest;state['verified']=True
        print(f"Verified source: {item['name']} ({item['bytes']/1e9:.2f} GB)",flush=True)
    packaged={};started=time.monotonic();initial=sum(transferred.values());last=0
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending={pool.submit(one,job) for job in ordered}
        while pending:
            done,pending=futures.wait(pending,timeout=10,return_when=futures.FIRST_EXCEPTION)
            for result in done:result.result()
            now=time.monotonic()
            if now-last>=10 or not pending:
                with lock:downloaded=sum(transferred.values())
                by_dataset={}
                for state in states:
                    got=state['item']['bytes'] if state['destination'].exists() else sum(min(p.stat().st_size if p.exists() else 0,state['item']['bytes']) for p in state['parts'])
                    entry=by_dataset.setdefault(state['item']['dataset'],{'downloaded_bytes':0,'total_bytes':0})
                    entry['downloaded_bytes']+=got;entry['total_bytes']+=state['item']['bytes']
                speed=(downloaded-initial)/max(now-started,1)
                status={'stage':'downloading','datasets':by_dataset,'network_MB_s':speed/1e6,'elapsed_seconds':now-started}
                write_json(AUDIT/'progress.json',status)
                print('Download: '+'; '.join(f"{name} {d['downloaded_bytes']/1e9:.2f}/{d['total_bytes']/1e9:.2f} GB" for name,d in by_dataset.items())+f'; network {speed/1e6:.1f} MB/s',flush=True)
                last=now
            for dataset in ('pad_ufes20','isic2019'):
                if dataset in packaged:continue
                members=[s for s in states if s['item']['dataset']==dataset]
                if all(s['destination'].exists() or sum(p.stat().st_size if p.exists() else 0 for p in s['parts'])==s['item']['bytes'] for s in members):
                    for state in members:finish_state(state)
                    packaged[dataset]=package_dataset(dataset,items)
    for state in states:
        finish_state(state)
    write_json(AUDIT/'verified_sources.json',items)
    return packaged


def csv_rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream);rows=list(reader);return reader.fieldnames,rows


def image_members(archives,suffix):
    result={}
    for archive in archives:
        for member in archive.infolist():
            path=Path(member.filename)
            if member.is_dir() or path.suffix.lower()!=suffix or '__MACOSX' in path.parts or path.name.startswith('._'):continue
            if path.name in result:raise ValueError(f'Duplicate image filename: {path.name}')
            result[path.name]=(archive,member)
    return result


def package_dataset(dataset,items):
    isic=dataset=='isic2019';directory=RAW/dataset
    name='ISIC_2019_Training_Kaggle.zip' if isic else 'PAD_UFES_20_External_Kaggle.zip'
    output=ROOT/'dist'/name;temp=output.with_name(name+'.part');output.parent.mkdir(exist_ok=True)
    sources=[item for item in items if item['dataset']==dataset]
    receipt=AUDIT/f'{dataset}_package.json'
    if output.exists() and receipt.exists():
        saved=json.loads(receipt.read_text())
        with zipfile.ZipFile(output) as existing:prior=json.loads(existing.read('dataset_provenance.json'))
        expected={item['name']:item['downloaded_sha256'] for item in sources}
        previous={item['name']:item['downloaded_sha256'] for item in prior['sources']}
        if expected==previous and sha(output)==saved['sha256']:
            print(f'READY: reusing verified {output}',flush=True);return saved
    csvs=['ISIC_2019_Training_GroundTruth.csv','ISIC_2019_Training_Metadata.csv'] if isic else ['metadata.csv']
    if isic:
        gt_fields,ground=csv_rows(directory/csvs[0]);meta_fields,meta=csv_rows(directory/csvs[1])
        if not {'image','MEL','NV','BCC','AK','BKL','DF','VASC','SCC'}<=set(gt_fields):raise ValueError('ISIC diagnosis CSV schema mismatch')
        if not {'image','lesion_id','age_approx','sex','anatom_site_general'}<=set(meta_fields):raise ValueError('ISIC metadata schema mismatch')
        ids=[r['image']+'.jpg' for r in ground];other=[r['image']+'.jpg' for r in meta]
        if len(ids)!=25331 or len(set(ids))!=len(ids) or set(ids)!=set(other) or len(other)!=len(ids):raise ValueError('ISIC metadata identities do not match')
        image_folder='ISIC_2019_Training_Input';expected_count=25331
    else:
        fields,meta=csv_rows(directory/'metadata.csv')
        if not {'patient_id','lesion_id','img_id','diagnostic','age','gender','region'}<=set(fields):raise ValueError('PAD metadata schema mismatch')
        # Identity and packaging checks only; do not inspect PAD diagnosis values.
        ids=[r['img_id'] for r in meta];image_folder='PAD_UFES_20_Input';expected_count=2298
        if len(ids)!=expected_count or len(set(ids))!=len(ids):raise ValueError('PAD metadata identities are repeated/missing')
        if len({r['patient_id'] for r in meta})!=1373 or len({r['lesion_id'] for r in meta})!=1641:
            raise ValueError('PAD source patient/lesion identity counts differ from release')
    opened=[];member_hashes={};dimensions=[];started=time.monotonic()
    try:
        for item in sources:
            if item['name'].endswith('.zip'):opened.append(zipfile.ZipFile(directory/item['name']))
        images=image_members(opened,'.jpg' if isic else '.png')
        if len(images)!=expected_count or set(images)!=set(ids):
            raise ValueError(f'{dataset}: metadata/image mismatch; metadata-only {list(set(ids)-set(images))[:5]}, image-only {list(set(images)-set(ids))[:5]}')
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as out:
            for index,(filename,(archive,member)) in enumerate(sorted(images.items()),1):
                data=archive.read(member) # Reads the complete member and validates the source ZIP CRC.
                with Image.open(io.BytesIO(data)) as im:
                    dimensions.append(im.size);im.verify()
                target=f'{image_folder}/{filename}';out.writestr(target,data);member_hashes[target]=hashlib.sha256(data).hexdigest()
                if index%500==0 or index==len(images):
                    print(f'{dataset}: checked and packaged {index}/{len(images)} images; {time.monotonic()-started:.1f}s',flush=True)
                    write_json(AUDIT/'progress.json',{'stage':'packaging','dataset':dataset,'images_done':index,'images_total':len(images)})
            for filename in csvs:
                data=(directory/filename).read_bytes();out.writestr(filename,data);member_hashes[filename]=hashlib.sha256(data).hexdigest()
            # Preserve attribution/license text distributed inside the official image archives.
            for archive in opened:
                for member in archive.infolist():
                    if not member.is_dir() and Path(member.filename).suffix.lower() in ('.txt','.md') and '__MACOSX' not in member.filename:
                        target='source_documents/'+Path(member.filename).name;data=archive.read(member)
                        if target in member_hashes:raise ValueError('Conflicting source-document filename')
                        out.writestr(target,data);member_hashes[target]=hashlib.sha256(data).hexdigest()
            record={'dataset':'ISIC 2019 Training' if isic else 'PAD-UFES-20 Version 1','source_page':ISIC_PAGE if isic else PAD_PAGE,
                    'intended_use':'Binary-model development' if isic else 'External testing only; never train or select thresholds on PAD',
                    'images':expected_count,'metadata_rows':len(meta),'original_image_and_csv_bytes_preserved':True,
                    'sources':sources,'sha256':member_hashes,'validation':'All image members read with source ZIP CRC and PIL structural verification; exact image/metadata ID match'}
            if not isic:
                owners={}
                for row in meta:owners.setdefault(row['lesion_id'],set()).add(row['patient_id'])
                record['identity_audit']={'patient_ids':len({r['patient_id'] for r in meta}),
                    'distinct_lesion_id_values':len(owners),'distinct_patient_lesion_pairs':len({(r['patient_id'],r['lesion_id']) for r in meta}),
                    'lesion_ids_shared_between_patients':sum(len(v)>1 for v in owners.values()),
                    'grouping_rule':'Use patient_id + lesion_id for cases; bootstrap/split by patient_id. Never merge patients by lesion_id alone.'}
            out.writestr('dataset_provenance.json',json.dumps(record,indent=2)+'\n')
            readme=f"# {record['dataset']} — Kaggle upload package\n\nSource: {record['source_page']}\n\n{record['intended_use']}\n\nUpload this ZIP as a separate Kaggle dataset and allow it to extract.\nImages are directly under {image_folder}/, with the original metadata CSV(s) at the root.\nOriginal image bytes, diagnoses and metadata are unchanged. No train/test splits have been rewritten.\n\n"
            if isic:
                readme+='The official release lists CC BY-NC terms. Attribute the BCN20000, HAM10000 and MSK contributors as specified on the official download page.\n'
            else:
                readme+='License: CC BY 4.0. Cite Pacheco et al., PAD-UFES-20, Data in Brief (2020), and dataset DOI 10.17632/zr7vgbcyr2.1.\n'
                readme+='\nIdentity audit: the release has 1,641 distinct lesion_id values but 1,891 patient+lesion pairs; 250 lesion IDs occur for different patients. The supplied binary adapter already groups cases by the compound patient+lesion identity. Metadata are preserved exactly.\n'
            out.writestr('README_KAGGLE.md',readme)
        print(f'{dataset}: checking final upload ZIP integrity...',flush=True)
        with zipfile.ZipFile(temp) as checked:
            if checked.testzip():raise ValueError('Final upload ZIP CRC failed')
            for filename in csvs:
                if checked.read(filename)!=(directory/filename).read_bytes():raise ValueError('Original CSV bytes changed during packaging')
        temp.replace(output)
        result={'file':str(output),'bytes':output.stat().st_size,'sha256':sha(output),'images':expected_count,'metadata_rows':len(meta),
                'safe_zip_paths':True,'source_crc_verified':True,'image_structures_verified':True,'csv_and_image_ids_match':True,
                'width_range':[min(w for w,h in dimensions),max(w for w,h in dimensions)],
                'height_range':[min(h for w,h in dimensions),max(h for w,h in dimensions)]}
        write_json(AUDIT/f'{dataset}_package.json',result);print(f'READY: {output} ({result["bytes"]/1e9:.2f} GB)',flush=True)
        return result
    finally:
        for archive in opened:archive.close()


def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    items=specifications();packaged=download_all(items)
    # Package the smaller external dataset first so it is ready sooner.
    packages=[packaged.get(dataset) or package_dataset(dataset,items) for dataset in ('pad_ufes20','isic2019')]
    write_json(AUDIT/'ready.json',{'packages':packages,'baseline_modified':False,'training_run':False,'PAD_outcome_analysis_performed':False})
    print('Both Kaggle upload datasets are ready. No model or saved baseline was modified.',flush=True)

if __name__=='__main__':main()

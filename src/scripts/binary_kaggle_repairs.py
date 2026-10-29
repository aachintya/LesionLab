"""Generate a pinned metadata repair for the already uploaded v1 code bundle."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL={
    'src/binary_study/data.py':'7995b8848815cff6c0b7c95d1336dd001ad18fdd4e0180bfb44f4c9e67b9c2d6',
    'src/binary_study/overlap.py':'85f65a257da752b00857ae8f8171bd7664693afcdd0b0047f5bbfc55d878a894',
    'tests/test_binary_study_metadata.py':None,
}


def metadata_repair_source():
    payload={name:{'original_sha256':old,'source':(ROOT/name).read_text(),
                   'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
             for name,old in ORIGINAL.items()}
    source='# ISIC_METADATA_REPAIR_V2: patch only the verified writable code copy.\n'
    source+='METADATA_REPAIR_PAYLOAD = json.loads('+repr(json.dumps(payload))+')\n'
    source+=textwrap.dedent('''
    repair_receipt = {'format':'isic-metadata-repair-v2','files':{},
                      'policy':'Median of valid within-lesion ages; categorical identity conflicts quarantine the whole group before fitting.'}
    reference_before = {n:file_sha(PROJECT/n) for n in manifest['sha256'] if n.startswith('reference/')}
    for name, item in METADATA_REPAIR_PAYLOAD.items():
        destination = (PROJECT/name).resolve()
        if not destination.is_relative_to(PROJECT.resolve()) or not name.startswith(('src/binary_study/','tests/')):
            raise ValueError('Unexpected metadata repair destination')
        current = file_sha(destination) if destination.exists() else None
        if current not in (item['original_sha256'],item['sha256']):
            raise ValueError('Unrecognized source for pinned metadata repair: '+name)
        compile(item['source'],name,'exec')
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(item['source'])
        if file_sha(destination) != item['sha256']:
            raise ValueError('Metadata repair write failed: '+name)
        repair_receipt['files'][name] = {'original_sha256':item['original_sha256'],'repaired_sha256':item['sha256']}
    if reference_before != {n:file_sha(PROJECT/n) for n in reference_before}:
        raise ValueError('Reference changed during metadata repair')
    repair_receipt['unchanged_reference_sha256'] = reference_before
    (PROJECT/'metadata_repair.json').write_text(json.dumps(repair_receipt,indent=2)+'\\n')
    print('ISIC metadata repair v2 installed and verified. Reference weights and frozen splits unchanged.',flush=True)
    ''')
    compile(source,'metadata-repair-bootstrap','exec')
    return source

import json
import glob

notebooks = glob.glob('*.ipynb')
for nb in notebooks:
    if 'split_dataset' in nb: continue
    print(f"Extracting {nb}...")
    d = json.load(open(nb, encoding='utf-8'))
    with open(nb.replace('.ipynb', '_code.py'), 'w', encoding='utf-8') as f:
        f.write('\n'.join([''.join(c['source']) for c in d['cells'] if c['cell_type'] == 'code']))

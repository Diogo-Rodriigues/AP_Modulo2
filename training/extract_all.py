import json
import glob
import os

os.makedirs('codepy', exist_ok=True)

notebooks = glob.glob('*.ipynb')
for nb in notebooks:
    if 'split_dataset' in nb: continue
    print(f"Extracting {nb}...")
    d = json.load(open(nb, encoding='utf-8'))
    out_path = os.path.join('codepy', nb.replace('.ipynb', '_code.py'))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([''.join(c['source']) for c in d['cells'] if c['cell_type'] == 'code']))

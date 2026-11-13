# Builds the Colab notebooks from src/*.py ("# %%" = new cell), and runs the same cells locally at tiny scale.
#   python build.py                      -> write notebooks next to this folder
#   python build.py test 01_prep.py '{"ROOT": "/tmp/x", ...}'   -> exec the cells here; the JSON overrides the [config] cell
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC, OUT = os.path.join(HERE, 'src'), os.path.dirname(HERE)
NOTEBOOKS = {'01_prep.py': '01_prep_alania.ipynb', '02_pretrain.py': '02_pretrain.ipynb', '03_my_voice.py': '03_my_voice.ipynb',
             '04_finetune.py': '04_finetune_me.ipynb', '05_eval.py': '05_freya_eval.ipynb'}

def cells(name):
    out, buf, tag = [], [], 'code'
    def flush():
        src = ''.join(buf).strip('\n')
        if src: out.append((tag, src))
    for line in open(os.path.join(SRC, name), encoding='utf-8'):
        m = re.match(r'# %%\s*(?:\[(.+?)\])?\s*$', line)
        if not m: buf.append(line); continue
        flush(); buf, tag = [], (m.group(1) or 'code')
        if tag.startswith('include '): out.extend(cells(tag.split()[1])); tag = 'code'
    flush()
    return out

def write(name, target):
    nb = {'cells': [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src.splitlines(keepends=True)}
                    for _, src in cells(name)],
          'metadata': {'accelerator': 'GPU', 'colab': {'gpuType': 'A100', 'machine_shape': 'hm', 'provenance': []},
                       'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 0}
    json.dump(nb, open(os.path.join(OUT, target), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(target, len(nb['cells']), 'cells')

def test(name, overrides):
    g = {'__name__': '__main__'}
    for tag, src in cells(name):
        if tag == 'colab': continue
        exec(compile(src, f'{name}[{tag}]', 'exec'), g)
        if tag == 'config': g.update(overrides)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test': test(sys.argv[2], json.loads(sys.argv[3]))
    else:
        for n, t in NOTEBOOKS.items():
            if os.path.exists(os.path.join(SRC, n)): write(n, t)

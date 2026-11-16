"""Validate, reproduce and package the paper without Git-history dependencies."""
import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/camera_ready_standardized'
PAPER = ROOT / 'paper'
CONTRACT = json.loads((ROOT / 'release_contract.json').read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args, cwd=ROOT):
    subprocess.run(list(args), cwd=cwd, check=True)


def verify_sources():
    for group in ['computational_source_sha256', 'numerical_archive_sha256']:
        for name, expected in CONTRACT[group].items():
            assert sha(ROOT / name) == expected, f'Release fingerprint differs: {name}'
    print('PASS: computational source, checkpoint archive and data fingerprints', flush=True)


def restore():
    """Expand only the archived run records; retain any already-restored records."""
    archive = RESULTS / 'checkpoints.zip'
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.startswith('runs/')]
        assert sum(n.endswith('.json') for n in names) == 1210
        assert sum(n.endswith('.npz') for n in names) == 1210
        assert z.testzip() is None
        for name in names:
            path = RESULTS / name
            assert path.resolve().is_relative_to((RESULTS / 'runs').resolve())
            if not path.exists():
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(z.read(name))
    print('Restored checkpoint archive (expanded files are Git-ignored).', flush=True)


def fetch_data():
    path = ROOT / 'data/Raman_Oils_raw.csv'
    expected = json.loads((RESULTS / 'protocol.json').read_text())['data_sha256']
    if path.exists():
        assert sha(path) == expected, 'Raw data hash differs from the released dataset'
        return
    url = 'https://www.data.gouv.fr/api/1/datasets/r/b02203bc-7c07-4f7c-803e-ac2470f54aa0'
    request = urllib.request.Request(url, headers={'User-Agent': 'Raman-research-reproducibility/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        content = response.read()
    assert hashlib.sha256(content).hexdigest() == expected, 'Downloaded dataset hash differs'
    path.write_bytes(content)
    print('Downloaded and hash-verified the attributed public dataset.', flush=True)


def check():
    verify_sources()
    restore()
    run(sys.executable, 'experiments/check_grouped_vip.py')
    run(sys.executable, 'experiments/check_protocol.py')
    run(sys.executable, 'paper/make_standardized.py', '--audit-only')


def figures():
    verify_sources()
    restore()
    run(sys.executable, 'paper/make_standardized.py')


def validate_paper():
    import pymupdf
    source = (PAPER / 'paper.tex').read_text(encoding='utf-8')
    body = (PAPER / 'camera_ready_body.tex').read_text(encoding='utf-8')
    assert re.search(r'\\title\{([^}]+)\}', source).group(1) == CONTRACT['title']
    assert body.split('\\section*{Acknowledgment}\n')[1].strip() == CONTRACT['acknowledgment']
    with pymupdf.open(PAPER / 'paper.pdf') as doc:
        assert len(doc) <= 6, 'Conference page limit exceeded'
        assert all('??' not in p.get_text() for p in doc)
        assert all(f[2] != 'Type3' and bool(doc.extract_font(f[0])[3])
                   for page in doc for f in page.get_fonts()), 'Unembedded or Type 3 font'
        pages = len(doc)
    log = (PAPER / 'paper.log').read_text(errors='replace')
    assert 'Overfull' not in log and 'undefined references' not in log
    print(f'PASS: {pages}-page PDF, fonts, references, original title and acknowledgment.', flush=True)
    return pages


def build():
    for _ in range(2):
        run('pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'paper.tex', cwd=PAPER)
    return validate_paper()


def package(pages):
    out = ROOT / 'dist'
    out.mkdir(exist_ok=True)
    pdf = out / 'camera-ready-paper.pdf'
    shutil.copy2(PAPER / 'paper.pdf', pdf)
    source = out / 'camera-ready-source.zip'
    paper_files = [PAPER / n for n in ['paper.tex', 'camera_ready_body.tex', 'final_numbers.tex', 'paper.pdf']]
    paper_files += list((PAPER / 'tables').glob('final_*.tex'))
    paper_files += list((PAPER / 'figs').glob('fig_final_*.pdf'))
    with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in paper_files:
            z.write(path, path.relative_to(PAPER).as_posix())
        cls = subprocess.check_output(['kpsewhich', 'IEEEtran.cls'], text=True).strip()
        assert cls, 'IEEEtran class not found'
        z.write(cls, 'IEEEtran.cls')
    code = out / 'camera-ready-research-code.zip'
    top = ['README.md', 'NOTICE.md', 'requirements.txt', 'release_contract.json',
           'reproduce.py', '.gitignore', '.gitattributes']
    files = [ROOT / n for n in top]
    for directory, patterns in [('src', ['*.py']), ('experiments', ['*.py']),
                                ('data', ['README.md']), ('docs', ['*.md']),
                                ('paper', ['*.tex', '*.py', '*.pdf']),
                                ('paper/tables', ['final_*.tex']),
                                ('paper/figs', ['fig_final_*.pdf', 'fig_final_*.png']),
                                ('results/camera_ready_standardized', ['*.json', '*.csv', '*.npz', 'checkpoints.zip'])]:
        for pattern in patterns:
            files += list((ROOT / directory).glob(pattern))
    with zipfile.ZipFile(code, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(files)):
            z.write(path, path.relative_to(ROOT).as_posix())
    for archive in [source, code]:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
    audit = json.loads((RESULTS / 'audit.json').read_text())
    assert audit['all_complete'] and audit['completed'] == 1210
    manifest = dict(repository=CONTRACT['repository'], pages=pages, checkpoints=1210,
                    fitted_repetitions=15, timing_repetitions=3,
                    files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in [pdf, source, code]})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Packaged final submission artifacts in', out, flush=True)


def rerun(jobs):
    verify_sources()
    assert not list((RESULTS / 'runs').glob('*.json')), (
        'For an independent numerical rerun, use a fresh clone before restoring archived runs.')
    fetch_data()
    (RESULTS / 'rerun_environment.json').write_text(json.dumps(dict(
        recorded=time.strftime('%Y-%m-%dT%H:%M:%S%z'), python=platform.python_version(),
        platform=platform.platform(), note='Original environment.json describes released measurements.'), indent=2))
    for stage in ['search', 'references', 'full', 'region', 'controls']:
        run(sys.executable, 'experiments/camera_ready_final.py', '--stage', stage, '--jobs', str(jobs))
    run(sys.executable, 'experiments/lv_cap20_standardized.py')
    run(sys.executable, 'experiments/camera_ready_final.py', '--stage', 'refit', '--jobs', str(jobs))
    run(sys.executable, 'experiments/camera_ready_final.py', '--stage', 'timing', '--jobs', '1')
    run(sys.executable, 'paper/make_standardized.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'figures', 'build', 'package', 'all', 'fetch-data', 'rerun'])
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'check': check()
    elif args.command == 'figures': figures()
    elif args.command == 'build': build()
    elif args.command == 'fetch-data': fetch_data()
    elif args.command == 'rerun': rerun(args.jobs)
    elif args.command == 'package': check(); package(build())
    else:
        check(); figures(); package(build())

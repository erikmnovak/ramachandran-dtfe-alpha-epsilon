#!/usr/bin/env python3
"""Compile the standalone exposition; optionally regenerate its data figures.

The distributed source bundle already includes all figures and tables. A
normal build therefore needs only Python's standard library, latexmk, a TeX
installation, and pdfinfo/pdftotext. --figures additionally requires the full
project's saved scientific artifacts, NumPy, SciPy, and Matplotlib.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
IN_PROJECT = (PROJECT/'documents/DTFE_ALPHA_EPSILON_POLICY.json').exists()
NAME = 'dtfe_alpha_epsilon'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    if args.figures:
        if not IN_PROJECT:
            parser.error('--figures needs the full project and its saved inputs')
        subprocess.run([sys.executable, str(HERE/'generate_figures.py')], check=True)
    manifest = json.loads((HERE/'figure_generation.json').read_text())
    for name, expected in manifest['outputs'].items():
        if sha(HERE/name) != expected:
            raise RuntimeError(f'Figure differs from generation manifest: {name}')
    build = PROJECT/'tmp/pdfs'/NAME if IN_PROJECT else HERE/'build'
    output = PROJECT/'output/pdf' if IN_PROJECT else HERE
    build.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with (build/'latexmk_console.log').open('w') as log:
        result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode',
            '-halt-on-error', f'-outdir={build}', f'{NAME}.tex'],
            cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((build/'latexmk_console.log').read_text()[-7000:])
        raise SystemExit(result.returncode)
    log = (build/f'{NAME}.log').read_text(errors='replace')
    problems = [line for line in log.splitlines() if re.search(
        r'Overfull|Underfull|undefined|Citation.+Warning|Rerun to get|multiply defined', line)]
    if problems:
        print('\n'.join(problems))
    pdf = output/f'{NAME}.pdf'
    shutil.copy2(build/f'{NAME}.pdf', pdf)
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    text = subprocess.check_output(['pdftotext', str(pdf), '-'], text=True)
    assert '??' not in text, 'Possible unresolved cross-reference in extracted PDF text'
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
    record = dict(status='compiled', pages=pages,
        protocol='dtfe-alpha-epsilon-span125-v1',
        pdf=str(pdf), pdf_sha256=sha(pdf), typesetting_messages=problems,
        source_sha256={str(p.relative_to(HERE)):sha(p) for p in
            [HERE/f'{NAME}.tex', HERE/'references.bib', HERE/'generate_figures.py',
             HERE/'build.py', HERE/'figure_generation.json', HERE/'numerical_results.json',
             *sorted((HERE/'tables').glob('*.tex'))]},
        figure_hashes_verified=True, unresolved_question_marks=False,
        visual_review='Separate rendered-page review required after this build')
    (HERE/'build_validation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(f'Built {pages} pages: {pdf}')
    print(f'Typesetting messages needing inspection: {len(problems)}')


if __name__ == '__main__':
    main()

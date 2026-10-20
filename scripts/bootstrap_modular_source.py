"""One-time migration of tested helpers; legacy files are retained intact."""
from pathlib import Path
import re, shutil
ROOT=Path(__file__).resolve().parents[1]
pkg=ROOT/'src'/'leakbench'; pkg.mkdir(parents=True, exist_ok=True)
mapping={'benchmark_adapters':'adapters','benchmark_ocr_adapters':'ocr_adapters',
         'benchmark_pipeline':'pipeline','benchmark_metrics':'metrics',
         'benchmark_workflow':'legacy_workflow','benchmark_environment':'environment'}
for old,new in mapping.items():
    target=pkg/f'{new}.py'
    if target.exists(): continue
    source=(ROOT/f'{old}.py').read_text(encoding='utf-8')
    for a,b in mapping.items(): source=re.sub(r'from '+a+r' import', 'from .'+b+' import', source)
    target.write_text(source,encoding='utf-8')
(pkg/'__init__.py').write_text('__version__ = "2.0.0"\n',encoding='utf-8')
(ROOT/'configs').mkdir(exist_ok=True)
shutil.copy2(ROOT/'taxonomy_crosswalk.csv',ROOT/'configs'/'taxonomy_crosswalk.csv')
print('New package prepared without moving legacy files.')

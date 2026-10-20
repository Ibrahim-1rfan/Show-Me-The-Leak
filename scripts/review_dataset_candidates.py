"""Generate review sheets from the previous cohort; does not assign categories."""
from pathlib import Path
import json, io, zipfile
from PIL import Image, ImageOps, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
sample=json.loads(next((ROOT/'Leak_Benchmark_Results').rglob('sample.json')).read_text())
out=ROOT/'tmp'/'category_review';out.mkdir(parents=True,exist_ok=True)
archive=zipfile.ZipFile(ROOT/'train2017.zip')
for category in ['document','general']:
    rows=[r for r in sample['images'] if r['stratum']==category]
    for start in range(0,len(rows),10):
        canvas=Image.new('RGB',(1500,1250),'white'); draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(rows[start:start+10]):
            with Image.open(io.BytesIO(archive.read(row['image_path']))) as im:
                im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((290,540))
                x=(j%5)*300; y=(j//5)*625
                canvas.paste(im,(x+(290-im.width)//2,y+42))
                draw.text((x+4,y+3),f"{start+j+1}: {row['image_id']}",fill='black')
                draw.text((x+4,y+20),'safe' if row['ground_truth_attributes']==['a0_safe'] else 'privacy labels',fill='black')
        canvas.save(out/f'{category}-{start//10+1}.jpg')
print('Contact sheets:',out)

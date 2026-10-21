"""Explicit OCR experiments; tiles remain in original upright pixel coordinates."""
import math
from .common import log

def tile_boxes(size,edge=1024,overlap=.15,max_tiles=16):
    if edge<128 or not 0<=overlap<.5:raise ValueError('Invalid tile edge/overlap')
    w,h=size;step=max(1,round(edge*(1-overlap)))
    def starts(length):
        values=list(range(0,max(1,length-edge+1),step))
        if values[-1]!=max(0,length-edge):values.append(max(0,length-edge))
        return values
    boxes=[(x,y,min(w,x+edge),min(h,y+edge)) for y in starts(h) for x in starts(w)]
    if len(boxes)>max_tiles:raise ValueError(f'Tile plan needs {len(boxes)} tiles, over explicit limit {max_tiles}; none silently skipped')
    return boxes

def map_tile_region(region,box):
    x,y,r,b=box;local=region.get('bbox_pixels');result=dict(region)
    if local is not None:
        if (len(local)!=4 or any(not math.isfinite(v) for v in local)
            or not 0<=local[0]<local[2]<=r-x or not 0<=local[1]<local[3]<=b-y):raise ValueError('Invalid tile-local coordinates')
        result['bbox_pixels']=[local[0]+x,local[1]+y,local[2]+x,local[3]+y]
    if region.get('quad_pixels') is not None:
        result['quad_pixels']=[v+(x if i%2==0 else y) for i,v in enumerate(region['quad_pixels'])]
    return result

def tiled_extract(extract,image,edge=1024,overlap=.15,max_tiles=16):
    boxes=tile_boxes(image.size,edge,overlap,max_tiles);regions=[];calls=[];issues=[]
    for index,box in enumerate(boxes):
        log('OCR tile',tile=index+1,total=len(boxes),original_crop=list(box))
        native=extract(image.crop(box));calls.append({'tile_box':list(box),'output':native})
        if native.get('parse_success') is False or native.get('metadata',{}).get('truncated'):
            issues.append({'tile':index,'reason':'invalid_or_truncated_output'})
        for region in native.get('text_regions',[]):
            try:regions.append({**map_tile_region(region,box),'tile_index':index})
            except ValueError as exc:issues.append({'tile':index,'reason':str(exc)})
    # Keep overlaps for audit. Attribute scoring is set-based; redaction unions masks.
    return {'raw_text':'','text_regions':regions,'tile_outputs':calls,
        'predicted_text':'\n'.join(r['text'] for r in regions),'parse_success':not issues,'parse_warnings':issues,
        'metadata':{'preprocessing_policy':'overlapping_original_resolution_tiles','original_size':list(image.size),
            'tile_edge':edge,'overlap':overlap,'tile_count':len(boxes),'truncated':bool(issues),
            'overlapping_regions_preserved':True}}

"""Editable table assembly and explicitly confirmed local OCR labels."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from cell_pipeline import COLUMNS, boxes, field_warning
from preprocessing import write_image


def assemble_table(image, regions, rows, entries):
    estimated = not regions
    if estimated:
        regions=[{'row':r,'col':c,'bbox':list(box)} for r,c,box in boxes(image,rows,COLUMNS)]
    cells=[dict(region,text='',original='',confidence=0,warning='',
                estimated=region.get('estimated',estimated),crossing=False) for region in regions]
    unplaced=[]
    for entry in entries:
        points=np.asarray(entry['polygon'],dtype=np.float32)
        area=max(1,cv2.contourArea(points))
        matches=[]
        for index,cell in enumerate(cells):
            if 'polygon' in cell:
                polygon=np.asarray(cell['polygon'],dtype=np.float32)
            else:
                x1,y1,x2,y2=cell['bbox']
                polygon=np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            overlap,_=cv2.intersectConvexConvex(points,polygon)
            if overlap/area>.15:
                matches.append((overlap,index))
        if not matches:
            unplaced.append(entry)
            continue
        _,index=max(matches)
        cell=cells[index]
        cell['text']=' '.join(filter(None,[cell['text'],entry['text']]))
        cell['original']=cell['text']
        cell['confidence']=min(cell['confidence'],entry['confidence']) if cell.get('has_text') else entry['confidence']
        cell['has_text']=True
        cell['crossing']=cell['crossing'] or len(matches)>1
        if entry.get('candidates'):
            cell.setdefault('candidates',[]).extend(entry['candidates'])
        if '多版本辨識不一致' in entry.get('warning',''):
            cell.setdefault('recognition_warnings',[]).append(entry['warning'])
    for cell in cells:
        reasons=[field_warning(cell['col'],cell['text'])]
        reasons.extend(cell.get('recognition_warnings',[]))
        if cell['estimated']:
            reasons.append('格線不完整，位置為估計，請對照原圖。')
        if cell['crossing']:
            reasons.append('完整文字框跨格，暫放在重疊最多的格子，請核對。')
        cell['warning']=' '.join(filter(None,reasons))
    return cells,unplaced


def save_labels(path, image, cells, confirmed, employee_id=''):
    if len(confirmed)!=len(cells) or any(type(value) is not bool for value in confirmed):
        raise ValueError('Invalid confirmation flags.')
    samples=path/'samples'
    samples.mkdir(exist_ok=True)
    records=[]
    image_id=hashlib.sha256(image.tobytes()).hexdigest()
    for cell,checked in zip(cells,confirmed):
        if not checked or cell.get('estimated') or cell.get('crossing'):
            continue
        x1,y1,x2,y2=cell['bbox']
        crop=image[y1:y2,x1:x2]
        name=f"r{cell['row']+1}-c{cell['col']+1}.png"
        write_image(samples/name,crop)
        records.append({'image':name,'text':cell['text'],'prediction':cell['original'],
                        'row':cell['row'],'col':cell['col'],'form_id':path.name,
                        'source_sha256':image_id,'employee_id':employee_id,
                        'blank':not bool(cell['text'])})
    (samples/'labels.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records),encoding='utf-8')
    return len(records)

"""Whole text boxes with optional grid suggestions and explicit review mapping."""
import cv2
import numpy as np
from cell_pipeline import HEADERS, field_warning, workbook


def date_candidates(crop, engine):
    """Compare gentle variants without removing rules or inventing strokes."""
    factor = max(2, 64 / crop.shape[0])
    enlarged = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4)).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    candidates = []
    for name, pixels in (('原圖', crop), ('放大', enlarged), ('對比增強', enhanced)):
        readings = engine.ocr(pixels, det=False, cls=False)
        items = readings[0] if readings and readings[0] else []
        candidates.append({'variant': name, 'text': ' '.join(item[0] for item in items),
                           'confidence': min((float(item[1]) for item in items), default=0)})
    return candidates


def recognize_text(image, engine, regions, row_edges, first_row=0, latin_engine=None):
    result = engine.ocr(image, cls=False)
    lines = result[0] if result and result[0] else []
    height, width = image.shape[:2]
    entries = []
    for polygon, (text, confidence) in lines:
        detector_text=text
        points = np.asarray(polygon,dtype=np.float32)
        x,y,w,h = cv2.boundingRect(points)
        center_y = float(points[:,1].mean())
        if row_edges and center_y < row_edges[first_row]:
            continue
        area = max(1,cv2.contourArea(points))
        hits = []
        for cell in regions:
            overlap,_ = cv2.intersectConvexConvex(points,np.asarray(cell['polygon'],dtype=np.float32))
            if overlap/area > .15:
                hits.append(cell)
        row,col,end_col = -1,-1,-1
        warning = '位置待確認；請指定列與欄，或保留在待確認工作表。'
        if hits and len({cell['row'] for cell in hits}) == 1:
            row = hits[0]['row']-first_row
            if len(hits)==1:
                col=end_col=hits[0]['col']
                warning=field_warning(col,text)
            else:
                warning='文字框跨越多格，請確認欄位或合併範圍。'
        elif row_edges:
            source_row = next((i for i in range(len(row_edges)-1)
                               if row_edges[i] <= center_y < row_edges[i+1]),-1)
            row=source_row-first_row if source_row>=first_row else -1
        # These fields occasionally contain Chinese notes. Preserve that candidate
        # for review instead of forcing it through an English-only recognizer.
        chinese_note = col in (3,4,5,6) and any('\u3400' <= ch <= '\u9fff' for ch in text)
        candidates = []
        if latin_engine is not None and 1<=col<=12 and not chinese_note:
            crop=image[max(0,y-3):min(height,y+h+3),max(0,x-3):min(width,x+w+3)]
            if col == 1:
                candidates = date_candidates(crop, latin_engine)
                readings = [[(candidates[0]['text'], candidates[0]['confidence'])]] if candidates[0]['text'] else []
            else:
                readings=latin_engine.ocr(crop,det=False,cls=False)
            if readings and readings[0]:
                text=' '.join(item[0] for item in readings[0])
                confidence=min(float(item[1]) for item in readings[0])
                warning=field_warning(col,text)
            if candidates and len({c['text'] for c in candidates}) > 1:
                comparison = '；'.join(c['variant']+'：'+(c['text'] or '未讀出') for c in candidates)
                warning = ' '.join(filter(None, [warning, '日期多版本辨識不一致，請核對。'+comparison]))
        entries.append({'row':row,'col':col,'end_col':end_col,'text':text,'original':text,
                        'detector_text':detector_text,
                        'candidates':candidates,
                        'confidence':float(confidence),'warning':warning,
                        'bbox':[max(0,x),max(0,y),min(width,x+w),min(height,y+h)],
                        'polygon':polygon})
    return sorted(entries,key=lambda entry:(entry['bbox'][1],entry['bbox'][0]))


def export_entries(originals, edits, rows):
    if not isinstance(edits,list) or len(edits)!=len(originals):
        raise ValueError('Review entry count changed.')
    cells, unresolved, occupied, merges = [], [], set(), []
    reviewed=[]
    for original,edit in zip(originals,edits):
        text=edit.get('text')
        if not isinstance(text,str) or len(text)>2000:
            raise ValueError('Invalid reviewed text.')
        row,col,end=int(edit.get('row',-1)),int(edit.get('col',-1)),int(edit.get('end_col',-1))
        entry=dict(original,text=text,row=row,col=col,end_col=end)
        reviewed.append(entry)
        if row<0 or col<0:
            unresolved.append(entry)
            continue
        if not 0<=row<rows or not 0<=col<=end<14:
            raise ValueError('列或合併欄位超出範圍。')
        addresses={(row,c) for c in range(col,end+1)}
        if occupied & addresses:
            raise ValueError('多筆文字指定到相同格子。請修改位置；不會覆蓋內容。')
        occupied.update(addresses)
        cells.append(entry)
        if end>col:
            merges.append((row,col,end))
    book=workbook(cells,rows)
    for row,col,end in merges:
        book.worksheets[0].merge_cells(start_row=row+2,end_row=row+2,
                                     start_column=col+1,end_column=end+1)
    pending=book.create_sheet('Unassigned')
    pending.append(['Original','Reviewed','Reason','Image box'])
    for entry in unresolved:
        pending.append([entry['original'],entry['text'],entry['warning'],str(entry['bbox'])])
        pending.cell(pending.max_row,1).data_type='s'
        pending.cell(pending.max_row,2).data_type='s'
    return book,reviewed

"""Local photo → confirmed grid → OCR → review → Excel application."""
import json
import hashlib
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from cell_pipeline import COLUMNS, HEADERS, boxes, preview, recognize, rectify, workbook
from preprocessing import crop_document, read_image, write_image
from grid_detection import detect_grid, suggest_corners, extract_tracks
from text_review import recognize_text, export_entries
from review_data import assemble_table, save_labels

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs' / 'web'
app = Flask(__name__, template_folder=str(ROOT / 'python' / 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True
_engine = None
_latin_engine = None


def folder(job):
    if len(job) != 32 or any(c not in '0123456789abcdef' for c in job):
        raise ValueError('Invalid job identifier.')
    path = RUNS / job
    if not path.is_dir():
        raise ValueError('Upload an image first.')
    return path


def engine():
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR
        _engine = PaddleOCR(lang='ch', use_gpu=False, use_angle_cls=True,
                            show_log=False, enable_mkldnn=False,
                            det_model_dir=str(ROOT / '.models' / 'det'),
                            rec_model_dir=str(ROOT / '.models' / 'rec'),
                            cls_model_dir=str(ROOT / '.models' / 'cls'))
    return _engine


def latin_engine():
    global _latin_engine
    if _latin_engine is None:
        from paddleocr import PaddleOCR
        _latin_engine = PaddleOCR(lang='en', use_gpu=False, use_angle_cls=False,
                                 show_log=False, enable_mkldnn=False,
                                 det_model_dir=str(ROOT/'.models'/'det'),
                                 rec_model_dir=str(ROOT/'.models'/'rec-en'),
                                 cls_model_dir=str(ROOT/'.models'/'cls'))
    return _latin_engine


@app.errorhandler(ValueError)
def invalid(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(RuntimeError)
def inference_error(error):
    return jsonify(error='OCR 執行失敗，請查看終端機訊息與模型下載狀態。'), 500


@app.get('/')
def home():
    return render_template('index.html', headers=HEADERS, columns=COLUMNS)


@app.get('/reviews')
def saved_reviews():
    reviews = []
    if RUNS.exists():
        for path in RUNS.iterdir():
            if not path.is_dir() or not (path/'reviewed.json').exists():
                continue
            try:
                data = restore_review(path.name)
                reviews.append({'job': path.name, 'rows': data['rows'],
                                'saved_at': (path/'reviewed.json').stat().st_mtime})
            except (ValueError, OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
    return jsonify(reviews=sorted(reviews, key=lambda r:r['saved_at'], reverse=True))


def restore_review(job):
    path = folder(job)
    required = ('reviewed.json', 'ocr.json', 'selection.json', 'geometry.json', 'grid.png')
    if any(not (path/name).is_file() for name in required):
        raise ValueError('這筆校對缺少必要檔案，無法恢復。')
    settings = json.loads((path/'selection.json').read_text(encoding='utf-8'))
    if settings.get('mode') == 'text':
        raise ValueError('舊版文字清單暫不支援恢復，請選擇表格校對。')
    cells = json.loads((path/'reviewed.json').read_text(encoding='utf-8'))
    originals = json.loads((path/'ocr.json').read_text(encoding='utf-8'))
    if len(cells) != len(originals):
        raise ValueError('校對與原始辨識的格子數不一致。')
    restored = []
    for saved, original in zip(cells, originals):
        if (saved['row'], saved['col']) != (original['row'], original['col']):
            raise ValueError('校對與原始辨識的位置不一致。')
        if not isinstance(saved['text'], str):
            raise ValueError('校對文字格式無效。')
        restored.append(dict(original, text=saved['text']))
    metadata_path = path/'review_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    flags = metadata.get('confirmed', [])
    if len(flags) != len(cells) or any(type(flag) is not bool for flag in flags):
        flags = [False]*len(cells)
    # Older saves predate metadata; recover only explicitly saved, matching labels.
    labels_path = path/'samples'/'labels.jsonl'
    if not metadata_path.exists() and labels_path.exists():
        source_hash = hashlib.sha256(read_image(path/'grid.png').tobytes()).hexdigest()
        labels = [json.loads(line) for line in labels_path.read_text(encoding='utf-8').splitlines() if line]
        matches = {(r['row'],r['col'],r['text']) for r in labels
                   if r.get('form_id') == job and r.get('source_sha256') == source_hash}
        flags = [(c['row'],c['col'],c['text']) in matches
                 and not c.get('estimated') and not c.get('crossing') for c in restored]
        employees = {r.get('employee_id','') for r in labels if r.get('source_sha256') == source_hash}
        if len(employees) == 1:
            metadata['employee_id'] = employees.pop()
    pending = path/'unplaced.json'
    geometry = json.loads((path/'geometry.json').read_text(encoding='utf-8'))
    return {'job': job, 'cells': restored, 'rows': settings['rows'], 'mode': 'cells',
            'confirmed': flags, 'employee_id': metadata.get('employee_id', ''),
            'first_row': settings.get('first_row', 1), 'corners': geometry['corners'],
            'grid_rows': geometry['rows'],
            'unplaced': json.loads(pending.read_text(encoding='utf-8')) if pending.exists() else []}


@app.get('/review/<job>')
def load_review(job):
    return jsonify(restore_review(job))


@app.post('/upload')
def upload():
    photo = request.files.get('photo')
    if photo is None:
        raise ValueError('Select a photograph.')
    image = cv2.imdecode(np.frombuffer(photo.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode this image.')
    if image.shape[0] * image.shape[1] > 30_000_000:
        raise ValueError('Image is too large; resize below 30 megapixels.')
    cropped, detected = crop_document(image)
    job = uuid.uuid4().hex
    path = RUNS / job
    path.mkdir(parents=True)
    write_image(path / 'original.png', image)
    write_image(path / 'document.png', cropped)
    corners, suggested = suggest_corners(cropped)
    return jsonify(job=job, detected=detected, corners=corners, suggested=suggested)


@app.get('/image/<job>/<name>')
def image(job, name):
    if name not in ('original.png', 'document.png', 'grid.png', 'preview.png'):
        raise ValueError('Invalid image name.')
    return send_file(folder(job) / name)


@app.post('/grid/<job>')
def grid(job):
    path = folder(job)
    data = request.get_json()
    image = rectify(read_image(path / 'document.png'), data['corners'])
    # Preserve exterior rules at the warp edge during morphological extraction.
    margin = max(2,min(20,min(image.shape[:2])//100))
    image = cv2.copyMakeBorder(image,margin,margin,margin,margin,
                              cv2.BORDER_CONSTANT,value=(255,255,255))
    cells, drawing, diagnostics = detect_grid(image)
    write_image(path / 'grid.png', image)
    write_image(path / 'preview.png', drawing)
    # A changed grid invalidates previous recognition and review.
    for name in ('ocr.json', 'reviewed.json', 'selection.json', 'review_metadata.json', 'unplaced.json'):
        (path / name).unlink(missing_ok=True)
    tracks,_ = extract_tracks(image)
    row_edges = [float(np.polyval(track,image.shape[1]/2)) for track in tracks]
    if not cells:
        rows = max(1,len(row_edges)-1)
        if len(row_edges)<2:
            row_edges=[0,image.shape[0]]
    else:
        rows = max(cell['row'] for cell in cells)+1
    for row in range(rows):
        first = next((cell for cell in cells if cell['row']==row and cell['col']==0),None)
        x1,y1,x2,y2 = first['bbox'] if first else [0,row_edges[row],40,row_edges[row+1]]
        cv2.putText(drawing,str(row+1),(int(x1+4),int((y1+y2)//2)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,220),2)
    write_image(path/'preview.png',drawing)
    data.update(rows=rows, cells=cells, diagnostics=diagnostics,row_edges=row_edges)
    (path / 'geometry.json').write_text(json.dumps(data), encoding='utf-8')
    return jsonify(ok=True, rows=rows, diagnostics=diagnostics,grid_complete=bool(cells),
                   message='格線可供定位。' if cells else '格線不完整，仍可辨識完整文字框；欄位需要人工指定。')


@app.post('/recognize/<job>')
def ocr(job):
    path = folder(job)
    if not (path / 'geometry.json').exists():
        raise ValueError('Preview and confirm the grid first.')
    data = json.loads((path / 'geometry.json').read_text(encoding='utf-8'))
    options = request.get_json(silent=True) or {}
    start = int(options.get('first_row',1))-1
    if not 0 <= start < data['rows']:
        raise ValueError('第一筆資料列號超出範圍。')
    regions = [dict(cell,row=cell['row']-start) for cell in data['cells'] if cell['row']>=start]
    if options.get('mode') in ('text','table') or not data['cells']:
        entries=recognize_text(read_image(path/'grid.png'),engine(),data['cells'],data['row_edges'],start,
                               latin_engine() if data['cells'] else None)
        if options.get('mode')=='table':
            if not regions:
                regions=[{'row':r-start,'col':c,'bbox':list(box),'estimated':True}
                         for r,c,box in boxes(read_image(path/'grid.png'),data['rows'],COLUMNS) if r>=start]
            table,unplaced=assemble_table(read_image(path/'grid.png'),regions,data['rows']-start,entries)
            (path/'ocr.json').write_text(json.dumps(table,ensure_ascii=False),encoding='utf-8')
            (path/'unplaced.json').write_text(json.dumps(unplaced,ensure_ascii=False),encoding='utf-8')
            (path/'selection.json').write_text(json.dumps({'rows':data['rows']-start,'first_row':start+1,'mode':'table'}),encoding='utf-8')
            return jsonify(cells=table,rows=data['rows']-start,mode='cells',unplaced=unplaced)
        (path/'ocr.json').write_text(json.dumps(entries,ensure_ascii=False),encoding='utf-8')
        (path/'selection.json').write_text(json.dumps({'rows':data['rows']-start,'first_row':start+1,'mode':'text'}),encoding='utf-8')
        return jsonify(entries=entries,rows=data['rows']-start,mode='text')
    cells = recognize(read_image(path / 'grid.png'), data['rows']-start, None, engine(), regions, latin_engine())
    (path / 'ocr.json').write_text(json.dumps(cells, ensure_ascii=False), encoding='utf-8')
    (path/'selection.json').write_text(json.dumps({'rows':data['rows']-start,'first_row':start+1}),encoding='utf-8')
    return jsonify(cells=cells, rows=data['rows']-start)


@app.post('/export/<job>')
@app.post('/save/<job>')
def export(job):
    make_excel=request.path.startswith('/export/')
    path = folder(job)
    if not (path / 'ocr.json').exists():
        raise ValueError('Run recognition before exporting.')
    originals = json.loads((path / 'ocr.json').read_text(encoding='utf-8'))
    payload=request.get_json()
    selection = path/'selection.json'
    settings=json.loads(selection.read_text()) if selection.exists() else {}
    if settings.get('mode')=='text':
        book,cells=export_entries(originals,payload.get('entries'),settings['rows'])
        if make_excel:
            book.save(path/'reviewed.xlsx')
        (path/'reviewed.json').write_text(json.dumps(cells,ensure_ascii=False),encoding='utf-8')
        if not make_excel:
            return jsonify(saved=True,samples=0)
        return send_file(path/'reviewed.xlsx',as_attachment=True,download_name='reviewed-form.xlsx')
    edits = payload['texts']
    if len(edits) != len(originals) or any(not isinstance(t, str) or len(t) > 2000 for t in edits):
        raise ValueError('Invalid review data.')
    cells = [dict(cell, text=text) for cell, text in zip(originals, edits)]
    data = json.loads((path / 'geometry.json').read_text(encoding='utf-8'))
    selection = path/'selection.json'
    rows = json.loads(selection.read_text())['rows'] if selection.exists() else data['rows']
    book = workbook(cells, int(rows))
    count=0
    if settings.get('mode')=='table':
        employee=str(payload.get('employee_id',''))[:80]
        count=save_labels(path,read_image(path/'grid.png'),cells,payload.get('confirmed',[False]*len(cells)),employee)
        pending=path/'unplaced.json'
        if pending.exists():
            sheet=book.create_sheet('Unplaced text')
            sheet.append(['OCR text','Image box'])
            for entry in json.loads(pending.read_text(encoding='utf-8')):
                sheet.append([entry['text'],str(entry['bbox'])])
                sheet.cell(sheet.max_row,1).data_type='s'
    if make_excel:
        book.save(path / 'reviewed.xlsx')
    (path / 'reviewed.json').write_text(json.dumps(cells, ensure_ascii=False), encoding='utf-8')
    (path/'review_metadata.json').write_text(json.dumps({'confirmed':payload.get('confirmed',[]),
            'employee_id':payload.get('employee_id',''),'samples':count},ensure_ascii=False),encoding='utf-8')
    if not make_excel:
        return jsonify(saved=True,samples=count,nonblank_samples=sum(
            bool(cell['text']) and checked and not cell.get('estimated') and not cell.get('crossing')
            for cell,checked in zip(cells,payload.get('confirmed',[]))))
    return send_file(path / 'reviewed.xlsx', as_attachment=True, download_name='reviewed-form.xlsx')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5000, help='Local HTTP port (default: 5000).')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535.')
    # One request at a time: Paddle predictors are shared and recognition is CPU-heavy.
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=False)

"""Fixed-form cell recognition. Geometry is confirmed by the user first."""
import cv2
import re
from datetime import datetime
import numpy as np
from openpyxl import Workbook

HEADERS = ['星期', '日期', '品種', '母瓶數', 'A1', 'A2', 'A3', '時數',
           '標準瓶', '增減瓶數－計件用', '增減瓶數－補瓶用', '單價', '總計', '異常備註']
# Approximate column proportions of this form; the editor allows adjustment.
COLUMNS = [0, .06, .12, .21, .28, .35, .42, .48, .54, .61, .68, .75, .81, .87, 1]


def rectify(image, corners):
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Select four corners: top-left, top-right, bottom-right, bottom-left.')
    h, w = image.shape[:2]
    if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
        raise ValueError('Corners must be inside the image.')
    if len(np.unique(points, axis=0)) != 4:
        raise ValueError('Select four different corners.')
    # Click order is not geometry. Sort around the centre, starting at top-left.
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1]-centre[1], points[:, 0]-centre[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    if not cv2.isContourConvex(points.reshape(4, 1, 2)) or abs(cv2.contourArea(points)) < 100:
        raise ValueError('Corners must form a convex area.')
    tl, tr, br, bl = points
    width = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    height = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    matrix = cv2.getPerspectiveTransform(points, np.float32([[0, 0], [width-1, 0],
                                                        [width-1, height-1], [0, height-1]]))
    return cv2.warpPerspective(image, matrix, (width, height))


def boxes(image, rows, columns):
    columns = np.asarray(columns, dtype=float)
    if not 1 <= rows <= 60 or len(columns) != 15 or not np.isfinite(columns).all():
        raise ValueError('Use 1–60 rows and 15 column boundaries.')
    if columns[0] != 0 or columns[-1] != 1 or (np.diff(columns) < .015).any():
        raise ValueError('Column boundaries must increase from 0 to 1, at least .015 apart.')
    h, w = image.shape[:2]
    for row in range(rows):
        for col in range(14):
            yield row, col, (int(columns[col]*w), round(row*h/rows),
                             int(columns[col+1]*w), round((row+1)*h/rows))


def preview(image, rows, columns):
    result = image.copy()
    for row, col, (x1, y1, x2, y2) in boxes(image, rows, columns):
        cv2.rectangle(result, (x1, y1), (x2-1, y2-1), (0, 180, 0), 1)
        cv2.putText(result, f'{row+1}:{col+1}', (x1+2, y1+12), cv2.FONT_HERSHEY_SIMPLEX,
                    .3, (0, 0, 200), 1)
    return result


def field_warning(col, text):
    if not text:
        return ''
    if col == 2 and not re.fullmatch(r'[A-Za-z0-9]+', text):
        return '品種應為英文與數字，請核對原圖。'
    if col == 1:
        valid = bool(re.fullmatch(r'\d{1,2}',text) and 1 <= int(text) <= 31)
        for pattern in ('%m/%d', '%Y/%m/%d', '%Y-%m-%d'):
            try:
                datetime.strptime(text,pattern)
                valid = True
            except ValueError:
                pass
        if not valid:
            return '日期格式不符，請核對日數或月/日。'
    if col in (3,4,5,6) and not re.fullmatch(r'\d+', text):
        return '此欄通常為數字，也可能有中文備註；請核對原圖。'
    return ''


def recognize(image, rows, columns, engine, geometry=None, latin_engine=None):
    cells = []
    regions = geometry if geometry is not None else [
        {'row': row, 'col': col, 'bbox': box} for row, col, box in boxes(image, rows, columns)]
    for region in regions:
        row, col = region['row'], region['col']
        x1, y1, x2, y2 = region['bbox']
        # Trim only the edge; aggressive line removal can erase handwriting.
        pad = max(1, min(3, (y2-y1)//10))
        aligned = rectify(image, region['polygon']) if 'polygon' in region else image[y1:y2, x1:x2]
        cell = aligned[pad:-pad, pad:-pad]
        if cell.size == 0:
            raise ValueError('A cell is too small; check the grid.')
        scale = max(1, 64 / cell.shape[0])
        cell = cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cell = cv2.copyMakeBorder(cell, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # Short single-line numeric/code fields often fail the text detector.
        # Read their confirmed crop directly; free-text fields retain detection.
        selected = latin_engine if latin_engine is not None and 1 <= col <= 12 else engine
        if 1 <= col <= 12:
            result = selected.ocr(cell, det=False, cls=False)
            readings = result[0] if result and result[0] else []
        else:
            result = engine.ocr(cell, cls=True)
            lines = result[0] if result and result[0] else []
            lines = sorted(lines, key=lambda line: (min(p[1] for p in line[0]), min(p[0] for p in line[0])))
            readings = [line[1] for line in lines]
        text = ' '.join(item[0] for item in readings)
        confidence = min((float(item[1]) for item in readings), default=0)
        cells.append(dict(region, text=text, original=text, confidence=confidence,
                          warning=field_warning(col,text)))
    return cells


def workbook(cells, rows):
    book = Workbook()
    sheet = book.active
    sheet.title = 'Reviewed form'
    sheet.append(HEADERS)
    for cell in cells:
        row, col = int(cell['row']), int(cell['col'])
        if not (0 <= row < rows and 0 <= col < 14):
            raise ValueError('Invalid cell position.')
        value = str(cell['text'])
        target = sheet.cell(row+2, col+1, value)
        # OCR/user strings must not become executable Excel formulas.
        target.data_type = 's'
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width = 16
    sheet.freeze_panes = 'A2'
    audit = book.create_sheet('OCR comparison')
    audit.append(['Row', 'Column', 'OCR original', 'Reviewed', 'OCR confidence'])
    for cell in cells:
        audit.append([cell['row']+1, HEADERS[cell['col']], cell['original'],
                      cell['text'], cell['confidence']])
        for target in (audit.cell(audit.max_row, 3), audit.cell(audit.max_row, 4)):
            target.data_type = 's'
    return book

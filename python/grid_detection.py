"""Extract actual ruled-form boundaries; never substitute evenly spaced rows."""
import cv2
import numpy as np


def line_tracks(mask, horizontal):
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    groups = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w < width * .25) if horizontal else (h < height * .65):
            continue
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, -1)
        ys, xs = np.where((component > 0) & (mask > 0))
        independent, dependent = (xs, ys) if horizontal else (ys, xs)
        # Fit a gently curved track, rather than assuming every line is axis-aligned.
        positions = np.unique(independent)
        centers = np.array([np.median(dependent[independent == p]) for p in positions])
        coefficients = np.polyfit(positions, centers, 1)
        residual = float(np.median(np.abs(np.polyval(coefficients, positions)-centers)))
        if residual > 4:
            continue
        midpoint = width / 2 if horizontal else height / 2
        location = float(np.polyval(coefficients, midpoint))
        matching = next((group for group in groups if abs(group[0]-location) < 6), None)
        if matching is None:
            groups.append([location, positions, centers])
        else:
            matching[1] = np.concatenate((matching[1], positions))
            matching[2] = np.concatenate((matching[2], centers))
    tracks = []
    for _, positions, centers in groups:
        required = (width if horizontal else height) * .65
        if np.ptp(positions) < required:
            continue
        tracks.append(np.polyfit(positions, centers, 2).tolist())
    midpoint = width / 2 if horizontal else height / 2
    return sorted(tracks, key=lambda p: np.polyval(p, midpoint))


def extract_tracks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    height, width = gray.shape
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width//35), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, height//45))))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))
    rows, columns = line_tracks(horizontal, True), line_tracks(vertical, False)
    # A table touching the crop edge can have its exterior contour rejected.
    # Recover it only when the vertical-line mask supplies long pixel support.
    support = np.count_nonzero(vertical, axis=0)
    for start, end in ((0, max(1,width//12)), (width-width//12,width)):
        x = int(start+np.argmax(support[start:end]))
        if support[x] < height*.65:
            continue
        if any(abs(np.polyval(line,height/2)-x) < 8 for line in columns):
            continue
        ys, xs = np.where(vertical[:,max(0,x-3):min(width,x+4)] > 0)
        xs = xs+max(0,x-3)
        columns.append(np.polyfit(ys,xs,1).tolist())
    columns.sort(key=lambda line: np.polyval(line,height/2))
    return rows, columns


def suggest_corners(image):
    """A form-specific suggestion, not a confirmed crop. Keep detected boundaries."""
    height, width = image.shape[:2]
    rows, columns = extract_tracks(image)
    if len(rows) < 5 or len(columns) < 10:
        return [[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], False
    # This form has title/year/header bands above the data. Pick the boundary
    # near their expected location, then let the user drag it if necessary.
    candidates = [line for line in rows if height*.06 <= np.polyval(line,width/2) <= height*.17]
    top = candidates[0] if candidates else rows[0]
    bottom = rows[-1]
    left, right = columns[0], columns[-1]
    corners = []
    for horizontal, vertical in ((top,left),(top,right),(bottom,right),(bottom,left)):
        x = float(np.polyval(vertical,height/2))
        for _ in range(6):
            y = float(np.polyval(horizontal,x))
            x = float(np.polyval(vertical,y))
        corners.append([x,y])
    # Include a narrow margin so the exterior lines survive rectification.
    centre = np.mean(corners,axis=0)
    corners = centre+(np.array(corners)-centre)*1.025
    corners[:,0] = np.clip(corners[:,0],0,width-1)
    corners[:,1] = np.clip(corners[:,1],0,height-1)
    return corners.tolist(), True


def detect_grid(image):
    height, width = image.shape[:2]
    rows, columns = extract_tracks(image)
    diagnostics = {'horizontal_lines': len(rows), 'vertical_lines': len(columns)}
    drawing = image.copy()
    for tracks, axis in ((rows, True), (columns, False)):
        for coefficients in tracks:
            positions = np.arange(width if axis else height)
            dependent = np.polyval(coefficients, positions)
            points = np.column_stack((positions, dependent) if axis else (dependent, positions))
            cv2.polylines(drawing, [np.round(points).astype(np.int32)], False,
                          (0, 160, 0) if axis else (220, 80, 0), 2)
    if len(columns) != 15 or not 2 <= len(rows) <= 61:
        return [], drawing, diagnostics
    row_positions = np.array([np.polyval(line, width/2) for line in rows])
    gaps = np.diff(row_positions)
    if (gaps < 8).any() or (len(gaps) > 2 and max(gaps) > np.median(gaps)*1.6):
        diagnostics['warning'] = 'Possible missing or duplicate horizontal boundary.'
        return [], drawing, diagnostics
    if row_positions[0] > height*.08 or row_positions[-1] < height*.92:
        diagnostics['warning'] = 'Selected area includes unbounded rows.'
        return [], drawing, diagnostics
    intersections = []
    for row in rows:
        points = []
        for column in columns:
            x = float(np.polyval(column, height/2))
            for _ in range(6):
                y = float(np.polyval(row, x))
                x = float(np.polyval(column, y))
            points.append([float(np.clip(x, 0, width-1)), float(np.clip(y, 0, height-1))])
        intersections.append(points)
    cells = []
    for row in range(len(rows)-1):
        for col in range(14):
            polygon = [intersections[row][col], intersections[row][col+1],
                       intersections[row+1][col+1], intersections[row+1][col]]
            points = np.asarray(polygon, dtype=np.float32)
            if not cv2.isContourConvex(points.reshape(4, 1, 2)) or cv2.contourArea(points) < 40:
                return [], drawing, diagnostics
            x, y, w, h = cv2.boundingRect(points)
            cells.append({'row': row, 'col': col, 'polygon': polygon,
                          'bbox': [x, y, min(width, x+w), min(height, y+h)]})
    return cells, drawing, diagnostics

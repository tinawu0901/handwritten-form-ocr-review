"""OpenCV helpers adapted from the original document-processing experiments."""
import cv2
import numpy as np


def read_image(path):
    # imdecode also supports Windows paths containing non-ASCII characters.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def crop_document(image):
    """Find a large convex quadrilateral; fall back to the original image."""
    ratio = min(1.0, 1200 / image.shape[0])
    small = cv2.resize(image, None, fx=ratio, fy=ratio)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < small.shape[0] * small.shape[1] * 0.2:
            break
        corners = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(corners) != 4 or not cv2.isContourConvex(corners):
            continue
        points = corners.reshape(4, 2).astype(np.float32) / ratio
        sums, differences = points.sum(axis=1), np.diff(points, axis=1).ravel()
        ordered = points[[np.argmin(sums), np.argmin(differences),
                          np.argmax(sums), np.argmax(differences)]]
        if len(np.unique(ordered, axis=0)) != 4:
            continue
        tl, tr, br, bl = ordered
        width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        if min(width, height) < 2:
            continue
        destination = np.float32([[0, 0], [width - 1, 0],
                                  [width - 1, height - 1], [0, height - 1]])
        matrix = cv2.getPerspectiveTransform(ordered, destination)
        return cv2.warpPerspective(image, matrix, (width, height)), True
    return image.copy(), False


def preprocess(image, mode):
    if mode == "none":
        return image.copy()
    if mode == "shadow":
        # Estimate the background separately in each colour channel.
        planes = []
        for plane in cv2.split(image):
            background = cv2.medianBlur(cv2.dilate(plane, np.ones((7, 7), np.uint8)), 21)
            planes.append(255 - cv2.absdiff(plane, background))
        image = cv2.merge(planes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

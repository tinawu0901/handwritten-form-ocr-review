import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import app as web
from cell_pipeline import COLUMNS, boxes, rectify, workbook, field_warning
from grid_detection import detect_grid, suggest_corners


class CellPipelineTests(unittest.TestCase):
    def test_field_rules_flag_errors_without_rewriting_them(self):
        self.assertFalse(field_warning(2,'DC3115'))
        self.assertTrue(field_warning(2,'移植'))
        self.assertFalse(field_warning(1,'31'))
        self.assertTrue(field_warning(1,'32'))
        self.assertFalse(field_warning(1,'1/1'))
        self.assertTrue(field_warning(1,'13/35'))
        self.assertTrue(field_warning(4,'一'))
        self.assertFalse(field_warning(6,'93'))
    def test_no_lines_suggestion_is_only_a_bounded_fallback(self):
        image = np.full((200,300,3),255,dtype=np.uint8)
        corners, detected = suggest_corners(image)
        self.assertFalse(detected)
        self.assertEqual(corners, [[0,0],[299,0],[299,199],[0,199]])

    def test_edge_rules_are_detected_with_white_warp_margin(self):
        image = np.full((600,800,3),255,dtype=np.uint8)
        for x in np.linspace(0,799,15).astype(int):
            cv2.line(image,(x,0),(x,599),(0,0,0),2)
        for y in np.linspace(0,599,13).astype(int):
            cv2.line(image,(0,y),(799,y),(0,0,0),2)
        padded = cv2.copyMakeBorder(image,6,6,6,6,cv2.BORDER_CONSTANT,value=(255,255,255))
        cells,_,diagnostics = detect_grid(padded)
        self.assertEqual(diagnostics['vertical_lines'],15)
        self.assertEqual(len(cells),12*14)
    def test_real_lines_preserve_unequal_row_heights(self):
        image = np.full((400, 700, 3), 255, dtype=np.uint8)
        for x in np.linspace(10, 689, 15).astype(int):
            cv2.line(image, (x, 10), (x, 389), (0, 0, 0), 2)
        for y in [10, 95, 190, 280, 389]:
            cv2.line(image, (10, y), (689, y), (0, 0, 0), 2)
        cells, _, diagnostics = detect_grid(image)
        self.assertEqual(diagnostics['vertical_lines'], 15)
        self.assertEqual(len(cells), 56)
        heights = [cell['bbox'][3]-cell['bbox'][1] for cell in cells if cell['col']==0]
        self.assertGreater(max(heights)-min(heights), 15)
        self.assertEqual(detect_grid(np.full_like(image,255))[0], [])
        # A missing interior row line must not silently merge two data rows.
        missing = np.full_like(image, 255)
        for x in np.linspace(10, 689, 15).astype(int):
            cv2.line(missing, (x, 10), (x, 389), (0, 0, 0), 2)
        for y in [10, 95, 280, 389]:
            cv2.line(missing, (10, y), (689, y), (0, 0, 0), 2)
        self.assertEqual(detect_grid(missing)[0], [])

    def test_invalid_geometry_is_rejected(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            rectify(image, [[0, 0], [199, 0], [199, 99], [199, 99]])
        with self.assertRaises(ValueError):
            list(boxes(image, 10, [0]*14+[1]))

    def test_all_click_orders_keep_content_upright(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)
        image[5:25, 5:25] = (0, 0, 255)
        image[70:90, 170:190] = (255, 0, 0)
        corners = [[0, 0], [199, 0], [199, 99], [0, 99]]
        expected = rectify(image, corners)
        for order in itertools.permutations(corners):
            np.testing.assert_array_equal(rectify(image, order), expected)

    def test_review_export_preserves_edits_originals_and_formula_text(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(web, 'RUNS', Path(directory)):
            client = web.app.test_client()
            image = np.full((200, 300, 3), 255, dtype=np.uint8)
            for x in np.linspace(10, 289, 15).astype(int):
                cv2.line(image, (x, 10), (x, 189), (0, 0, 0), 2)
            for y in [10, 95, 189]:
                cv2.line(image, (10, y), (289, y), (0, 0, 0), 2)
            _, encoded = cv2.imencode('.png', image)
            with patch.object(web, 'crop_document', return_value=(image, False)):
                response = client.post('/upload', data={'photo': (io.BytesIO(encoded.tobytes()), 'sample.png')})
            job = response.json['job']
            self.assertEqual(len(response.json['corners']),4)
            response = client.post('/grid/'+job, json={'corners': [[0,0],[299,0],[299,199],[0,199]],
                                                       'rows': 1, 'columns': COLUMNS})
            self.assertEqual(response.status_code, 200)
            cells = [{'row':0, 'col':col, 'text':'old', 'original':'old',
                      'confidence':.2, 'bbox':[0,0,10,10]} for col in range(14)]
            with patch.object(web, 'engine', return_value=None), patch.object(web, 'latin_engine', return_value=None), patch.object(web, 'recognize', return_value=cells) as recognition:
                self.assertEqual(client.post('/recognize/'+job,json={'first_row':2}).status_code, 200)
                regions = recognition.call_args.args[4]
                self.assertTrue(all(c['row']==0 and c['bbox'][1]>=90 for c in regions))
                self.assertEqual(client.post('/recognize/'+job,json={'first_row':3}).status_code,400)
            edits = ['=1+1', 'corrected'] + ['']*12
            flags=[True,False]+[False]*12
            saved=client.post('/save/'+job,json={'texts':edits,'confirmed':flags,'employee_id':'worker-a'})
            self.assertEqual(saved.status_code,200)
            self.assertTrue(saved.json['saved'])
            self.assertTrue((Path(directory)/job/'reviewed.json').exists())
            self.assertFalse((Path(directory)/job/'reviewed.xlsx').exists())
            resumed=client.get('/review/'+job)
            self.assertEqual(resumed.status_code,200)
            self.assertEqual([c['text'] for c in resumed.json['cells']],edits)
            self.assertEqual(resumed.json['cells'][0]['original'],'old')
            self.assertEqual(resumed.json['confirmed'],flags)
            self.assertEqual(resumed.json['employee_id'],'worker-a')
            self.assertEqual(resumed.json['first_row'],2)
            self.assertEqual(client.get('/reviews').json['reviews'][0]['job'],job)
            self.assertEqual(client.get('/review/'+'f'*32).status_code,400)
            # A damaged positional mapping must not restore edits into wrong cells.
            reviewed_path=Path(directory)/job/'reviewed.json'
            content=reviewed_path.read_text(encoding='utf-8')
            damaged=json.loads(content)
            damaged[0]['col']=8
            reviewed_path.write_text(json.dumps(damaged),encoding='utf-8')
            self.assertEqual(client.get('/review/'+job).status_code,400)
            self.assertEqual(client.get('/reviews').json['reviews'],[])
            reviewed_path.write_text(content,encoding='utf-8')
            metadata_path=Path(directory)/job/'review_metadata.json'
            metadata_content=metadata_path.read_text(encoding='utf-8')
            metadata_path.unlink()
            samples=Path(directory)/job/'samples'
            samples.mkdir()
            import hashlib
            from preprocessing import read_image
            source_hash=hashlib.sha256(read_image(Path(directory)/job/'grid.png').tobytes()).hexdigest()
            labels=[{'row':0,'col':0,'text':edits[0],'form_id':job,'source_sha256':source_hash,'employee_id':'old-worker'},
                    {'row':0,'col':1,'text':edits[1],'form_id':job,'source_sha256':'wrong-source','employee_id':'wrong'}]
            (samples/'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in labels),encoding='utf-8')
            legacy=client.get('/review/'+job).json
            self.assertEqual(legacy['confirmed'],[True]+[False]*13)
            self.assertEqual(legacy['employee_id'],'old-worker')
            metadata_path.write_text(metadata_content,encoding='utf-8')
            response = client.post('/export/'+job, json={'texts': edits})
            self.assertEqual(response.status_code, 200)
            book = load_workbook(io.BytesIO(response.data))
            self.assertEqual(book.worksheets[0]['A2'].value, '=1+1')
            self.assertEqual(book.worksheets[0]['A2'].data_type, 's')
            self.assertEqual(book.worksheets[0]['B2'].value, 'corrected')
            self.assertEqual(book.worksheets[1]['C2'].value, 'old')
            self.assertEqual(book.worksheets[1]['D2'].value, '=1+1')
            self.assertTrue((Path(directory)/job/'reviewed.json').exists())
            response.close()
            book.close()


if __name__ == '__main__':
    unittest.main()

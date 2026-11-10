import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from openpyxl import load_workbook

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import app as web
from text_review import recognize_text,export_entries


class FakeOCR:
    def ocr(self,image,**kwargs):
        return [[[[[20,30],[100,30],[100,60],[20,60]],('2/1',.9)]]]


class TextReviewTests(unittest.TestCase):
    def test_date_variants_preserve_prediction_and_surface_disagreement(self):
        from unittest.mock import Mock
        from review_data import assemble_table
        image=np.full((100,120,3),255,dtype=np.uint8)
        regions=[{'row':0,'col':1,'bbox':[0,0,120,100],
                  'polygon':[[0,0],[119,0],[119,99],[0,99]]}]
        latin=Mock()
        latin.ocr.side_effect=[[[('26',.8)]],[[('2/6',.9)]],[[('2/6',.92)]]]
        entries=recognize_text(image,FakeOCR(),regions,[0,99],latin_engine=latin)
        self.assertEqual(entries[0]['text'],'26')
        self.assertEqual([c['text'] for c in entries[0]['candidates']],['26','2/6','2/6'])
        cells,_=assemble_table(image,regions,1,entries)
        self.assertIn('多版本辨識不一致',cells[0]['warning'])
        self.assertIn('2/6',cells[0]['warning'])
        self.assertEqual(cells[0]['text'],'26')
        self.assertGreater(latin.ocr.call_args_list[1].args[0].shape[0],latin.ocr.call_args_list[0].args[0].shape[0])

    def test_chinese_note_in_numeric_field_is_not_overwritten(self):
        image=np.full((100,120,3),255,dtype=np.uint8)
        from unittest.mock import Mock
        regions=[{'row':0,'col':4,'polygon':[[0,0],[119,0],[119,99],[0,99]]}]
        chinese=Mock()
        chinese.ocr.return_value=[[[[[20,30],[100,30],[100,60],[20,60]],('休',.9)]]]
        latin=Mock()
        latin.ocr.return_value=[[('11',.95)]]
        result=recognize_text(image,chinese,regions,[0,99],latin_engine=latin)
        self.assertEqual(result[0]['text'],'休')
        self.assertIn('中文',result[0]['warning'])
        latin.ocr.assert_not_called()
        chinese.ocr.return_value=[[[[[20,30],[100,30],[100,60],[20,60]],('17',.9)]]]
        result=recognize_text(image,chinese,regions,[0,99],latin_engine=latin)
        self.assertEqual(result[0]['text'],'11')
        latin.ocr.assert_called_once()

    def test_cross_cell_text_stays_whole_and_unassigned(self):
        image=np.full((100,120,3),255,dtype=np.uint8)
        regions=[{'row':0,'col':col,'polygon':[[col*60,0],[(col+1)*60,0],[(col+1)*60,99],[col*60,99]]}
                 for col in range(2)]
        result=recognize_text(image,FakeOCR(),regions,[0,99])
        self.assertEqual(result[0]['original'],'2/1')
        self.assertEqual(result[0]['col'],-1)
        self.assertIn('跨',result[0]['warning'])

    def test_merge_date_text_pending_and_collision(self):
        originals=[{'original':'2/1','text':'2/1','warning':'','confidence':.9,'bbox':[0,0,10,10]},
                   {'original':'ABC','text':'ABC','warning':'cross','confidence':.5,'bbox':[10,0,30,10]},
                   {'original':'unknown','text':'unknown','warning':'pending','confidence':.2,'bbox':[0,20,10,30]}]
        edits=[{'row':0,'col':1,'end_col':1,'text':'2/1'},
               {'row':0,'col':2,'end_col':3,'text':'ABC'},
               {'row':-1,'col':-1,'end_col':-1,'text':'unknown'}]
        book,_=export_entries(originals,edits,2)
        self.assertEqual(book.worksheets[0]['B2'].value,'2/1')
        self.assertEqual(book.worksheets[0]['B2'].data_type,'s')
        self.assertIn('C2:D2',str(book.worksheets[0].merged_cells))
        self.assertEqual(book['Unassigned']['B2'].value,'unknown')
        edits[2]={'row':0,'col':3,'end_col':3,'text':'conflict'}
        with self.assertRaises(ValueError):
            export_entries(originals,edits,2)

    def test_no_grid_still_recognizes_and_exports_pending(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(web,'RUNS',Path(directory)):
            client=web.app.test_client()
            image=np.full((200,300,3),255,dtype=np.uint8)
            _,encoded=cv2.imencode('.png',image)
            job=client.post('/upload',data={'photo':(io.BytesIO(encoded.tobytes()),'blank.png')}).json['job']
            response=client.post('/grid/'+job,json={'corners':[[0,0],[299,0],[299,199],[0,199]]})
            self.assertTrue(response.json['ok'])
            self.assertFalse(response.json['grid_complete'])
            with patch.object(web,'engine',return_value=FakeOCR()):
                response=client.post('/recognize/'+job,json={'mode':'text','first_row':1})
            self.assertEqual(response.json['entries'][0]['text'],'2/1')
            response=client.post('/export/'+job,json={'entries':[{'row':-1,'col':-1,'end_col':-1,'text':'2/1'}]})
            self.assertEqual(response.status_code,200)
            book=load_workbook(io.BytesIO(response.data))
            self.assertEqual(book['Unassigned']['B2'].value,'2/1')
            book.close()
            response.close()
            with patch.object(web,'engine',return_value=FakeOCR()):
                response=client.post('/recognize/'+job,json={'mode':'table','first_row':1})
            cells=response.json['cells']
            self.assertEqual(len(cells),14)
            response=client.post('/export/'+job,json={'texts':[c['text'] for c in cells],
                                                    'confirmed':[True]*len(cells)})
            self.assertEqual(response.status_code,200)
            # Estimated positions must not become handwriting training labels.
            self.assertEqual((Path(directory)/job/'samples/labels.jsonl').read_text(),'')
            response.close()

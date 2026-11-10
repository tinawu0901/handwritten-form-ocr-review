import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from review_data import assemble_table,save_labels
from evaluate_excel import compare,value_text


class ReviewDataTests(unittest.TestCase):
    def test_blank_positions_and_crossing_text_are_preserved(self):
        image=np.full((100,1400,3),255,dtype=np.uint8)
        regions=[{'row':0,'col':c,'bbox':[c*100,0,(c+1)*100,100],
                  'polygon':[[c*100,0],[(c+1)*100,0],[(c+1)*100,99],[c*100,99]]} for c in range(14)]
        entries=[{'text':'2/1','confidence':.8,'polygon':[[110,20],[190,20],[190,60],[110,60]]},
                 {'text':'ABC','confidence':.6,'polygon':[[210,20],[370,20],[370,60],[210,60]]}]
        cells,pending=assemble_table(image,regions,1,entries)
        self.assertEqual(len(cells),14)
        self.assertEqual(cells[1]['text'],'2/1')
        self.assertEqual(cells[0]['text'],'')
        self.assertTrue(any(c['crossing'] for c in cells))
        self.assertEqual(sum(c['text']=='ABC' for c in cells),1)
        self.assertFalse(pending)

    def test_only_confirmed_reliable_labels_are_saved(self):
        image=np.full((100,400,3),255,dtype=np.uint8)
        cells=[{'row':0,'col':i,'bbox':[i*100,0,(i+1)*100,100],'text':'2/1','original':'21'} for i in range(4)]
        cells[2]['estimated']=True;cells[3]['crossing']=True
        with tempfile.TemporaryDirectory() as directory:
            count=save_labels(Path(directory),image,cells,[True,False,True,True],'employee-1')
            self.assertEqual(count,1)
            record=json.loads((Path(directory)/'samples/labels.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(record['text'],'2/1')
            self.assertEqual(record['prediction'],'21')
            self.assertTrue((Path(directory)/'samples'/record['image']).is_file())

    def test_evaluation_does_not_hide_errors_behind_blank_cells(self):
        truth=[['2/1','ABC']+['']*12]
        predicted=[['21','ABC','invented']+['']*11]
        report=compare(truth,predicted)
        self.assertEqual(report['nonblank_exact_accuracy'],.5)
        self.assertEqual(report['blank_false_positives'],1)
        with self.assertRaises(ValueError):
            compare(truth,[])
        from datetime import datetime
        with self.assertRaises(ValueError):
            value_text(datetime(2026,2,1))

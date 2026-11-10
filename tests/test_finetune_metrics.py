import sys
import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
from finetune_recognizer import edit_distance, metrics
import finetune_recognizer as training


class FineTuneMetricTests(unittest.TestCase):
    def test_different_crops_of_same_original_remain_one_photo_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            image=np.full((40,60,3),255,dtype=np.uint8)
            for job in ('first','second'):
                path=root/'runs/web'/job
                samples=path/'samples'
                samples.mkdir(parents=True)
                cv2.imwrite(str(path/'original.png'),image)
                cv2.imwrite(str(samples/'cell.png'),image)
                record={'image':'cell.png','text':'2/6','row':0,'col':1,
                        'form_id':job,'source_sha256':'different-grid-'+job}
                (samples/'labels.jsonl').write_text(json.dumps(record))
            output=root/'experiment'
            (output/'images').mkdir(parents=True)
            with patch.object(training,'ROOT',root):
                records,excluded=training.prepare(output,set('2/6'))
            self.assertEqual(len(records),2)
            self.assertFalse(excluded)
            self.assertEqual(len({r['photo_sha256'] for r in records}),1)
            self.assertEqual(len({r['source_sha256'] for r in records}),2)

    def test_blank_cells_do_not_dilute_recognition_errors(self):
        records = [{'text':'2/6','predicted':'26','col':1},
                   {'text':'ABC','predicted':'ABC','col':2},
                   {'text':'','predicted':'1','col':1},
                   {'text':'','predicted':'','col':1}]
        result = metrics(records)
        self.assertEqual(result['nonblank'],2)
        self.assertEqual(result['exact_accuracy'],.5)
        self.assertAlmostEqual(result['character_error_rate'],1/6)
        self.assertEqual(result['blank_false_positives'],1)
        self.assertEqual(result['columns']['1'],{'count':1,'correct':0})
        self.assertEqual(edit_distance('', '123'),3)
        self.assertEqual(edit_distance('ABC',''),3)


if __name__ == '__main__':
    unittest.main()

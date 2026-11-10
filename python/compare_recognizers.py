"""Compare pretrained recognizers on identical confirmed local crops, no training."""
import argparse
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def edit_distance(left,right):
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def variant(image,mode):
    if mode=='original':
        return image
    pad=min(3,max(1,min(image.shape[:2])//12))
    crop=image[pad:-pad,pad:-pad].copy()
    if mode=='tight':
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,12)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        keep=[]
        h,w=gray.shape
        for x,y,cw,ch,area in stats[1:]:
            if area<4 or (cw>w*.8 and ch<4) or (ch>h*.8 and cw<4):
                continue
            keep.append((x,y,x+cw,y+ch))
        if keep:
            x1=max(0,min(b[0] for b in keep)-3);y1=max(0,min(b[1] for b in keep)-3)
            x2=min(w,max(b[2] for b in keep)+3);y2=min(h,max(b[3] for b in keep)+3)
            crop=crop[y1:y2,x1:x2]
    factor=max(1,48/crop.shape[0])
    crop=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(crop,4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--backend',choices=['v4','v5','trocr'],required=True)
    parser.add_argument('--variant',choices=['original','trimmed','tight'],default='trimmed')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--blank-count',type=int,default=20)
    parser.add_argument('--trocr-model',default='microsoft/trocr-base-handwritten',
                        choices=['microsoft/trocr-base-handwritten','microsoft/trocr-small-handwritten'])
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path.')
    records=[json.loads(line) for line in (args.samples/'labels.jsonl').read_text(encoding='utf-8').splitlines() if line]
    # A manifest copied from an evaluation must not carry another model's confidence.
    records=[{key:value for key,value in record.items() if key not in {'predicted','correct','confidence'}}
             for record in records]
    nonblank=[r for r in records if r['text']]
    blanks=[r for r in records if not r['text']]
    selected=[blanks[int(i)] for i in np.linspace(0,len(blanks)-1,min(max(0,args.blank_count),len(blanks)))] if blanks and args.blank_count else []
    records=nonblank+selected
    if not records:
        parser.error('No labelled samples.')
    if args.backend=='v4':
        from paddleocr import PaddleOCR
        model=PaddleOCR(lang='en',use_gpu=False,use_angle_cls=False,show_log=False,enable_mkldnn=False,
                       det_model_dir=str(ROOT/'.models/det'),rec_model_dir=str(ROOT/'.models/rec-en'),
                       cls_model_dir=str(ROOT/'.models/cls'))
        def predict(image):
            results=model.ocr(image,det=False,cls=False)
            return ' '.join(item[0] for item in (results[0] or []))
        model_name='en_PP-OCRv4_rec'
    elif args.backend=='v5':
        os.environ.setdefault('PADDLE_PDX_MODEL_SOURCE','BOS')
        from paddleocr import TextRecognition
        model=TextRecognition(model_name='en_PP-OCRv5_mobile_rec',device='cpu',enable_mkldnn=False,cpu_threads=4)
        def predict(image):
            result=next(iter(model.predict(input=image,batch_size=1)))
            return str(result['rec_text'])
        model_name='en_PP-OCRv5_mobile_rec'
    else:
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor,VisionEncoderDecoderModel
        torch.set_num_threads(4)
        model_name=args.trocr_model
        processor=TrOCRProcessor.from_pretrained(model_name,cache_dir=str(ROOT/'.models/huggingface'),use_fast=False)
        model=VisionEncoderDecoderModel.from_pretrained(model_name,cache_dir=str(ROOT/'.models/huggingface'),
                                                        use_safetensors=model_name.endswith('base-handwritten')).eval()
        def predict(image):
            pixels=processor(images=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),return_tensors='pt').pixel_values
            with torch.inference_mode():
                result=model.generate(pixels,max_new_tokens=24)
            return processor.batch_decode(result,skip_special_tokens=True)[0]
    results=[]
    start=time.perf_counter()
    for index,record in enumerate(records):
        image=cv2.imdecode(np.fromfile(str(args.samples/record['image']),dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Missing image: '+record['image'])
        text=predict(variant(image,args.variant)).strip()
        results.append(dict(record,predicted=text,correct=text==record['text']))
        if (index+1)%10==0:
            print(f'Processed {index+1}/{len(records)}',flush=True)
    positive=[r for r in results if r['text']]
    blank=[r for r in results if not r['text']]
    report={'backend':args.backend,'model':model_name,'variant':args.variant,'nonblank':len(positive),
            'correct':sum(r['correct'] for r in positive),
            'character_error_rate':sum(edit_distance(r['text'],r['predicted']) for r in positive)/max(1,sum(len(r['text']) for r in positive)),
            'case_insensitive_correct':sum(r['predicted'].casefold()==r['text'].casefold() for r in positive),
            'blank_tested':len(blank),'blank_false_positives':sum(bool(r['predicted']) for r in blank),
            'inference_seconds':time.perf_counter()-start,'source_images':len({r['source_sha256'] for r in records}),
            'scope':'Calibration on confirmed crops, not independent unseen-form validation.',
            'results':results}
    if args.backend=='trocr':
        report['model_revision']=getattr(model.config,'_commit_hash',None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()

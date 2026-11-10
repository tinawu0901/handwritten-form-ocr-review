"""Small CPU CTC fine-tuning experiment, evaluated by held-out source photograph.

Uses the official PP-OCRv4 architecture and training weights. The visual backbone
is frozen; only the final CTC classifier is updated. Results and
copied labels stay under the ignored output folder. No web model is replaced.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(ca != cb)))
        previous = current
    return previous[-1]


def prepare(output, characters):
    records, seen, excluded = [], {}, []
    for manifest in sorted((ROOT/'runs/web').glob('*/samples/labels.jsonl')):
        photo_path = manifest.parent.parent/'original.png'
        original = cv2.imdecode(np.fromfile(str(photo_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if original is None:
            raise ValueError('Original photograph missing: '+str(photo_path))
        photo_hash = hashlib.sha256(original.tobytes()).hexdigest()
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if not line:
                continue
            record = json.loads(line)
            key = (record['source_sha256'], record['row'], record['col'])
            if key in seen:
                if seen[key] != record['text']:
                    raise ValueError('Conflicting confirmed answers for the same source cell.')
                continue
            seen[key] = record['text']
            if record['text'] and any(char not in characters for char in record['text']):
                excluded.append(dict(record, reason='Characters outside pretrained English dictionary'))
                continue
            image_path = manifest.parent/record['image']
            pixels = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError('Cannot read sample: '+str(image_path))
            name = f"{record['form_id']}-{record['image']}"
            shutil.copy2(image_path, output/'images'/name)
            record = dict(record, image=name, image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
            record['photo_sha256'] = photo_hash
            records.append(record)
    (output/'dataset.json').write_text(json.dumps({'records':records, 'excluded':excluded},ensure_ascii=False,indent=2),encoding='utf-8')
    return records, excluded


def preprocess(path, brightness=1):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    pad = min(3, max(1, min(image.shape[:2])//12))
    image = image[pad:-pad, pad:-pad]
    if brightness != 1:
        image = np.clip(image.astype(np.float32)*brightness, 0, 255).astype(np.uint8)
    image = cv2.copyMakeBorder(image, 4,4,4,4,cv2.BORDER_CONSTANT,value=(255,255,255))
    height, width = image.shape[:2]
    resized_width = min(320, int(np.ceil(48*width/height)))
    image = cv2.resize(image,(resized_width,48),interpolation=cv2.INTER_LINEAR)
    normalized = image.astype('float32').transpose(2,0,1)/127.5-1
    padded = np.zeros((3,48,320),dtype='float32')
    padded[:,:,:resized_width] = normalized
    return padded


def metrics(results):
    positive = [r for r in results if r['text']]
    blank = [r for r in results if not r['text']]
    correct = sum(r['text'] == r['predicted'] for r in positive)
    errors = sum(edit_distance(r['text'],r['predicted']) for r in positive)
    count = sum(len(r['text']) for r in positive)
    columns = {}
    for col in sorted({r['col'] for r in positive}):
        items = [r for r in positive if r['col']==col]
        columns[str(col)] = {'count':len(items),'correct':sum(r['text']==r['predicted'] for r in items)}
    return {'nonblank':len(positive),'correct':correct,'exact_accuracy':correct/len(positive) if positive else None,
            'character_error_rate':errors/count if count else None,'blank_tested':len(blank),
            'blank_false_positives':sum(bool(r['predicted']) for r in blank),'columns':columns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--learning-rate',type=float,default=0.00001)
    parser.add_argument('--batch-size',type=int,default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory.')
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('Training parameters must be positive.')
    args.source = args.source.resolve()
    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    sys.path.insert(0,str(args.source))
    import paddle
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess.rec_postprocess import CTCLabelDecode
    paddle.set_device('cpu')
    decoder = CTCLabelDecode(character_dict_path=str(args.source/'ppocr/utils/en_dict.txt'),use_space_char=True)
    character_index = {char:i for i,char in enumerate(decoder.character) if i}
    config_path = args.source/'configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    architecture = config['Architecture']
    architecture['Head']['out_channels_list'] = {'CTCLabelDecode':len(decoder.character),
        'SARLabelDecode':len(decoder.character)+2,'NRTRLabelDecode':len(decoder.character)+3}
    pretrained = paddle.load(str(args.weights))
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir()
    shutil.copy2(Path(__file__), args.output/'training-script.py')
    records, excluded = prepare(args.output, character_index)
    sources = sorted({r['photo_sha256'] for r in records if r['text']})
    if len(sources) < 3:
        raise ValueError('Need at least three source photographs for grouped evaluation.')
    recipe = {'epochs':args.epochs,'learning_rate':args.learning_rate,'batch_size':args.batch_size,
              'seed':42,'scope':'ASCII/English-dictionary confirmed crops; Chinese excluded',
              'split':'Leave-one-source-photograph-out; fixed final epoch, no held-out checkpoint selection',
              'trained_layers':'Final CTC classifier only; backbone and sequence encoder frozen in eval mode',
              'augmentation':'Training only: cached brightness variants 0.9, 1.0, 1.1; no digit rotation or synthetic labels',
              'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              'source_provenance':json.loads((args.source.parent/'source.json').read_text()),
              'architecture':architecture,'versions':{'paddle':paddle.__version__,'numpy':np.__version__},
              'source_images':len(sources),'nonblank':sum(bool(r['text']) for r in records),
              'excluded_nonblank':len(excluded)}
    (args.output/'recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2),encoding='utf-8')
    all_before, all_after, folds = [], [], []
    start = time.perf_counter()
    cache_records = [r for r in records if r['text']]
    for source in sources:
        blanks = [r for r in records if not r['text'] and r['photo_sha256']==source]
        cache_records += [blanks[int(i)] for i in np.linspace(0,len(blanks)-1,min(20,len(blanks)))] if blanks else []
    feature_cache = {}
    for fold, source in enumerate(sources):
        random.seed(42); np.random.seed(42); paddle.seed(42)
        model = build_model(copy.deepcopy(architecture))
        state = model.state_dict()
        missing = [key for key,value in state.items() if key not in pretrained or list(pretrained[key].shape)!=list(value.shape)]
        if missing:
            raise ValueError('Pretrained architecture mismatch: '+str(missing[:10]))
        model.set_state_dict({key:pretrained[key] for key in state})
        for parameter in model.parameters():
            parameter.stop_gradient = True
        trainable = list(model.head.ctc_head.parameters())
        for parameter in trainable:
            parameter.stop_gradient = False
        train = [r for r in records if r['text'] and r['photo_sha256'] != source]
        heldout = [r for r in records if r['text'] and r['photo_sha256'] == source]
        blanks = [r for r in records if not r['text'] and r['photo_sha256'] == source]
        heldout += [blanks[int(i)] for i in np.linspace(0,len(blanks)-1,min(20,len(blanks)))] if blanks else []
        fold_dir = args.output/f'fold-{fold+1}'
        fold_dir.mkdir()
        (fold_dir/'split.json').write_text(json.dumps({'train':[r['image'] for r in train],
            'heldout':[r['image'] for r in heldout],'heldout_source':source},indent=2))

        if not feature_cache:
            model.eval()
            for offset in range(0,len(cache_records),args.batch_size):
                batch = cache_records[offset:offset+args.batch_size]
                for brightness in (.9,1.0,1.1):
                    pixels = paddle.to_tensor(np.stack([preprocess(args.output/'images'/r['image'],brightness) for r in batch]))
                    with paddle.no_grad():
                        features = model.head.ctc_encoder(model.backbone(pixels)).numpy()
                    for r,features_row in zip(batch,features):
                        feature_cache[(r['image'],brightness)] = features_row
                if offset % (args.batch_size*2) == 0:
                    print(f'Cached features {min(offset+len(batch),len(cache_records))}/{len(cache_records)}',flush=True)

        def evaluate(items):
            results = []
            model.eval()
            for offset in range(0,len(items),args.batch_size):
                batch = items[offset:offset+args.batch_size]
                features = paddle.to_tensor(np.stack([feature_cache[(r['image'],1.0)] for r in batch]))
                with paddle.no_grad():
                    predictions = decoder(model.head.ctc_head(features).numpy())
                results.extend(dict(r,predicted=p[0],confidence=float(p[1])) for r,p in zip(batch,predictions))
            return results

        before = evaluate(heldout)
        print(f'Fold {fold+1}/{len(sources)}: train {len(train)}, heldout {len(heldout)}; baseline {metrics(before)}',flush=True)
        optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate,parameters=trainable,
                                         grad_clip=paddle.nn.ClipGradByGlobalNorm(5))
        history = []
        for epoch in range(args.epochs):
            indices = np.random.permutation(len(train))
            losses = []
            for offset in range(0,len(indices),args.batch_size):
                batch = [train[int(i)] for i in indices[offset:offset+args.batch_size]]
                features = paddle.to_tensor(np.stack([feature_cache[(r['image'],random.choice((.9,1.0,1.1)))] for r in batch]))
                targets = [[character_index[char] for char in r['text']] for r in batch]
                lengths = np.array([len(t) for t in targets],dtype='int64')
                labels = np.zeros((len(batch),int(lengths.max())),dtype='int32')
                for i,target in enumerate(targets): labels[i,:len(target)] = target
                model.head.ctc_head.train()
                logits = model.head.ctc_head(features)
                loss = paddle.nn.functional.ctc_loss(paddle.nn.functional.log_softmax(logits,axis=2).transpose([1,0,2]),
                    paddle.to_tensor(labels),paddle.to_tensor(np.full(len(batch),logits.shape[1],dtype='int64')),
                    paddle.to_tensor(lengths),blank=0,reduction='mean')
                if not bool(paddle.isfinite(loss).item()):
                    raise ValueError('Non-finite training loss; stopping.')
                loss.backward();optimizer.step();optimizer.clear_grad()
                losses.append(float(loss.item()))
            history.append({'epoch':epoch+1,'loss':float(np.mean(losses))})
            print(f'Fold {fold+1} epoch {epoch+1}/{args.epochs}: loss {history[-1]["loss"]:.4f}',flush=True)
        paddle.save(model.state_dict(),str(fold_dir/'final.pdparams'))
        after = evaluate(heldout)
        result = {'fold':fold+1,'train_nonblank':len(train),'heldout_source':source,
                  'before':metrics(before),'after':metrics(after),'history':history,
                  'predictions_before':before,'predictions_after':after}
        (fold_dir/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        folds.append({k:v for k,v in result.items() if not k.startswith('predictions')})
        all_before.extend(before);all_after.extend(after)
        print('Fold result',json.dumps({'before':result['before'],'after':result['after']}),flush=True)
    report = {'recipe':recipe,'before':metrics(all_before),'after':metrics(all_after),'folds':folds,
              'seconds':time.perf_counter()-start,'limitations':'Three source photos; no independent final test or known writer split. Measures recognition on confirmed crops, not full-page detection/placement.'}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'before':report['before'],'after':report['after'],'seconds':report['seconds']}),flush=True)


if __name__ == '__main__':
    main()

"""Compare aligned manual/OCR Excel tables; never assume an arbitrary layout."""
import argparse
import json
import hashlib
from datetime import date, datetime
from pathlib import Path
from openpyxl import Workbook, load_workbook
from cell_pipeline import HEADERS


def value_text(value):
    if value is None:
        return ''
    if isinstance(value,(date,datetime)):
        raise ValueError('Excel has converted a date. Save dates such as 2/1 as text first.')
    if isinstance(value,float) and value.is_integer():
        value=int(value)
    return str(value).strip()


def read_table(path):
    book=load_workbook(path,data_only=False)
    try:
        sheet=book['Reviewed form'] if 'Reviewed form' in book.sheetnames else book.worksheets[0]
        headers=[value_text(sheet.cell(1,col+1).value) for col in range(14)]
        if headers!=HEADERS:
            raise ValueError(f'{path.name}: headers do not match. Copy data into the supplied template; do not silently shift columns.')
        if sheet.max_column>14:
            raise ValueError(f'{path.name}: unexpected extra columns.')
        rows=[[value_text(sheet.cell(row,col+1).value) for col in range(14)]
              for row in range(2,sheet.max_row+1)]
        return rows
    finally:
        book.close()


def compare(manual,predicted):
    if len(manual)!=len(predicted):
        raise ValueError('Row counts differ. Check alignment and blank rows before evaluating.')
    differences=[]
    total=correct=nonblank=nonblank_correct=false_positive=0
    fields={header:{'nonblank':0,'correct_nonblank':0} for header in HEADERS}
    for row,(expected,actual) in enumerate(zip(manual,predicted),start=2):
        if len(expected)!=14 or len(actual)!=14:
            raise ValueError('Expected 14 columns.')
        for col,(truth,guess) in enumerate(zip(expected,actual)):
            total+=1
            correct+=truth==guess
            if truth:
                nonblank+=1
                nonblank_correct+=truth==guess
                fields[HEADERS[col]]['nonblank']+=1
                fields[HEADERS[col]]['correct_nonblank']+=truth==guess
            elif guess:
                false_positive+=1
            if truth!=guess:
                differences.append({'excel_row':row,'column':HEADERS[col],'expected':truth,'predicted':guess})
    return {'rows':len(manual),'total_cells':total,'correct_cells':correct,
            'all_cell_exact_accuracy':correct/total if total else None,
            'nonblank_cells':nonblank,'correct_nonblank_cells':nonblank_correct,
            'nonblank_exact_accuracy':nonblank_correct/nonblank if nonblank else None,
            'blank_false_positives':false_positive,'fields':fields,'differences':differences}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manual',type=Path)
    parser.add_argument('--prediction',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--image',type=Path,help='Optional matching photograph, recorded by path and hash')
    parser.add_argument('--make-template',type=Path)
    parser.add_argument('--rows',type=int,default=30)
    args=parser.parse_args()
    try:
        if args.make_template:
            if args.make_template.exists():
                raise ValueError('Template already exists; choose another path.')
            if not 1<=args.rows<=60:
                raise ValueError('Use 1–60 data rows.')
            book=Workbook();sheet=book.active;sheet.title='Reviewed form';sheet.append(HEADERS)
            for _ in range(args.rows):
                sheet.append(['']*14)
            for row in sheet:
                for cell in row:
                    cell.number_format='@'
            args.make_template.parent.mkdir(parents=True,exist_ok=True)
            book.save(args.make_template)
            print(args.make_template.resolve())
        else:
            if not all((args.manual,args.prediction,args.output)):
                raise ValueError('Provide --manual, --prediction and --output.')
            if args.output.exists():
                raise ValueError('Report exists; choose another output path.')
            report=compare(read_table(args.manual),read_table(args.prediction))
            report['manual']=str(args.manual.resolve());report['prediction']=str(args.prediction.resolve())
            if args.image:
                report['image']=str(args.image.resolve())
                report['image_sha256']=hashlib.sha256(args.image.read_bytes()).hexdigest()
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({key:value for key,value in report.items() if key not in ('differences','fields')},ensure_ascii=True))
    except (ValueError,OSError) as error:
        parser.exit(1,f'Error: {error}\n')


if __name__=='__main__':
    main()

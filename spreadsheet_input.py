"""Convert a first-sheet .xlsx upload to CSV using Python's standard library."""
import base64
import binascii
import csv
import io
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

MAIN = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
DOCREL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
PKGREL = '{http://schemas.openxmlformats.org/package/2006/relationships}'
MAX_EXPANDED = 100 * 1024 * 1024


def as_csv(data):
    name = str(data.get('name', ''))
    if not name.lower().endswith('.xlsx'):
        return data
    try:
        raw = base64.b64decode(data.get('content_base64', ''), validate=True)
        with ZipFile(io.BytesIO(raw)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_EXPANDED:
                raise ValueError('Excel workbook expands beyond the 100 MiB parsing limit.')
            workbook = ET.fromstring(z.read('xl/workbook.xml'))
            sheets = workbook.find(MAIN + 'sheets')
            if sheets is None or len(sheets) == 0:
                raise ValueError('Excel workbook has no worksheets.')
            rel_id = sheets[0].get(DOCREL + 'id')
            rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
            rel = next((r for r in rels.findall(PKGREL + 'Relationship') if r.get('Id') == rel_id), None)
            if rel is None or rel.get('TargetMode') == 'External':
                raise ValueError('Cannot find the first worksheet in the Excel workbook.')
            target = rel.get('Target', '')
            path = posixpath.normpath(target.lstrip('/') if target.startswith('/') else posixpath.join('xl', target))
            if not path.startswith('xl/'):
                raise ValueError('Invalid Excel worksheet path.')
            shared = []
            if 'xl/sharedStrings.xml' in z.namelist():
                strings = ET.fromstring(z.read('xl/sharedStrings.xml'))
                shared = [''.join(t.text or '' for t in si.iter(MAIN + 't')) for si in strings.findall(MAIN + 'si')]
            out = io.StringIO(newline='')
            writer = csv.writer(out, lineterminator='\n')
            next_row = 1
            with z.open(path) as sheet:
                for _, row in ET.iterparse(sheet, events=('end',)):
                    if row.tag != MAIN + 'row':
                        continue
                    number = int(row.get('r', next_row))
                    if number < next_row or number > 1_048_576:
                        raise ValueError('Invalid row number in Excel workbook.')
                    while next_row < number:
                        writer.writerow([])
                        next_row += 1
                    cells = []
                    for cell in row.findall(MAIN + 'c'):
                        ref = cell.get('r', '')
                        match = re.match(r'^([A-Z]{1,3})[0-9]+$', ref)
                        if not match:
                            raise ValueError('Invalid cell reference in Excel workbook.')
                        column = 0
                        for letter in match.group(1):
                            column = column * 26 + ord(letter) - 64
                        if column > 16384:
                            raise ValueError('Invalid column in Excel workbook.')
                        while len(cells) < column:
                            cells.append('')
                        if cell.find(MAIN + 'f') is not None:
                            raise ValueError('Excel formulas are not supported. Copy and paste values into the first worksheet before uploading.')
                        kind = cell.get('t', '')
                        value = cell.findtext(MAIN + 'v', default='')
                        if kind == 's':
                            value = shared[int(value)]
                        elif kind == 'inlineStr':
                            value = ''.join(t.text or '' for t in cell.iter(MAIN + 't'))
                        elif kind == 'e':
                            raise ValueError('Excel worksheet contains an error cell. Fix it before uploading.')
                        cells[column - 1] = value
                    writer.writerow(cells)
                    next_row += 1
                    row.clear()
        return {**data, 'name': name[:-5] + '.csv', 'content': out.getvalue()}
    except (BadZipFile, KeyError, ET.ParseError, binascii.Error, IndexError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name}: invalid or unsupported .xlsx workbook.') from exc

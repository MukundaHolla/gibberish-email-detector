import base64
import io
import unittest
from zipfile import ZipFile

from domain_rules import parse_domain_file
from local_ui import parse_rows


def workbook(rows):
    buf = io.BytesIO()
    with ZipFile(buf, 'w') as z:
        z.writestr('xl/workbook.xml', '''<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="First" sheetId="1" r:id="rId1"/></sheets></workbook>''')
        z.writestr('xl/_rels/workbook.xml.rels', '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>''')
        z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + rows + '</sheetData></worksheet>')
    return {'name': 'test.xlsx', 'content_base64': base64.b64encode(buf.getvalue()).decode()}


def cell(ref, text):
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


class SpreadsheetInputTests(unittest.TestCase):
    def test_email_custom_column_and_optional_name_with_sparse_rows(self):
        data = workbook('<row r="1">' + cell('B1', 'contact') + cell('C1', 'full_name') + '</row><row r="3">' + cell('B3', 'mukunda@gmail.com') + cell('C3', 'Mukunda') + '</row>')
        data['column'] = 'contact'
        self.assertEqual(parse_rows(data, with_names=True), [(3, 'mukunda@gmail.com', 'Mukunda')])
        with self.assertRaisesRegex(ValueError, 'was not found'):
            parse_rows({**data, 'column':'missing'})

    def test_domain_custom_column(self):
        data = workbook('<row r="1">' + cell('A1','name') + cell('B1','website') + '</row><row r="2">' + cell('A2','one') + cell('B2','example.com') + '</row>')
        data['column'] = 'website'
        self.assertEqual(parse_domain_file(data)['domains'], ['example.com'])

    def test_browser_training_with_email_and_domain_workbooks(self):
        from pathlib import Path
        from web import browser_backend
        def make_file(header, values):
            rows = '<row r="1">' + cell('A1', header) + '</row>'
            rows += ''.join(f'<row r="{i}">' + cell(f'A{i}', value) + '</row>' for i, value in enumerate(values, 2))
            return {**workbook(rows), 'column': header}
        ordinary = Path('examples/ordinary.txt').read_text().splitlines()
        flagged = Path('examples/flagged.txt').read_text().splitlines()
        data = {'ordinary': make_file('contact', ordinary), 'flagged': make_file('contact', flagged),
                'domain_files': {'ordinary': make_file('host_name', [f'ordinary{i}.com' for i in range(30)]),
                                 'flagged': make_file('host_name', [f'qzxv{i}.com' for i in range(30)])},
                'options': {'min_coverage': 0}}
        result = browser_backend.handle('train', data, lambda _: None)
        self.assertIn('domain_model', browser_backend.handle('artifacts', {'model': result['id']}, lambda _: None)['model.json'])
        self.assertGreater(len(browser_backend.handle('batch', {'model': result['id'], 'file': data['ordinary']}, lambda _: None)['results']), 0)

    def test_reject_formulas_and_invalid_data(self):
        data = workbook('<row r="1">' + cell('A1','email') + '</row><row r="2"><c r="A2"><f>1+1</f><v>2</v></c></row>')
        with self.assertRaisesRegex(ValueError, 'formulas are not supported'):
            parse_rows(data)
        with self.assertRaisesRegex(ValueError, 'invalid or unsupported'):
            parse_rows({'name':'bad.xlsx', 'content_base64':'bad!'})

if __name__ == '__main__':
    unittest.main()

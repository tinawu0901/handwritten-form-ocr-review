# Validation

Checked on 10 November 2026 using 64-bit Python 3.12 on Windows.

- All 20 existing unit tests passed.
- The included synthetic form produced a six-row, fourteen-column grid after manual adjustment of the upper corners.
- A Flask test-client workflow passed upload, grid generation, mocked text recognition, review saving, restoration and Excel export. All 84 synthetic cell labels were saved.
- The test environment used NumPy 1.26.4, OpenCV headless 4.11.0.86, Flask 3.1.2, openpyxl 3.1.5 and PyYAML 6.0.2. Its dependency check passed.
- Application source files and original tests match the reviewed source snapshot. The changes in this distribution are documentation, synthetic fixtures, their generator, ignore rules and test setup.
- Both interface screenshots and the generated example were visually inspected.
- A GitHub Actions workflow is included but has not yet run on a hosted repository.

Pretrained OCR inference, weight downloads, model training and browser interaction were not rerun in this validation. Mocked recognition checks data flow, not recognition quality. Historical experiment figures in the technical notes are retained from the project documentation and were not independently reproduced for this distribution.

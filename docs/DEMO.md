# Synthetic demo walkthrough

The included image contains six printed rows in the application's fixed 14-column layout. All values are invented. It contains no real names, employee IDs, signatures or payroll records.

The expected values are in `examples/expected.json`; they describe the intended contents, not predictions or a completed review. OCR can differ between environments. The existing screenshots use a separately prepared synthetic form and are illustrative, not screenshots of this exact new fixture.

## Run the application

Install the application dependencies and start `python/app.py` as described in the root README. Open the localhost address in a browser.

## Upload and locate the table

1. Upload `examples/synthetic-form.png`.
2. In **確認資料區** (confirm data area), check the red corner handles. Include the entire grid and exclude the title above it. The form has no header row inside the grid.
   For this fixture, automatic corner suggestions can mistake the first printed row for a header and start at the second row. Move both upper red handles to the top outer grid border so that the row beginning with `1 / 2/1 / AB100` is included.
3. Press **產生切格預覽** (generate grid preview).
4. Check for seven horizontal boundaries and fifteen vertical boundaries, forming six rows and fourteen columns.
5. Leave **第一筆資料在預覽第幾列？** (first data row) at **1**.
6. Press **選區確認，辨識文字** (confirm region and recognise text). The first request may download models.

If detection is wrong, adjust the corners and regenerate the preview before recognition.

## Review and export

- In **人工校對** (manual review), select a cell to inspect the source crop.
- Yellow cells are review prompts; a non-yellow cell can still be incorrect.
- Use arrow keys to move between fields. Edit incorrect values; dates and codes should stay as text.
- Compare all rows with `examples/expected.json`. The optional employee alias field can remain blank.
- Check each confirmed cell, or use confirm-all only after checking the entire table.
- Press **儲存校對結果與樣本** to save corrections and eligible crop labels.
- Press **匯出 Excel（選用）** to download the workbook.

Saved reviews and generated files remain in `runs/web/`, which is excluded from version control.

## Regenerate the example

From the repository root, after installing the application dependencies:

```powershell
python scripts/generate_demo.py
```

The script draws the printed grid with OpenCV and writes deterministic intended values as JSON. It does not use ImageGen, a real form image, or model inference. The generated files overwrite only the two named synthetic example files.

This walkthrough demonstrates interaction and data flow. It makes no claim about handwriting accuracy.

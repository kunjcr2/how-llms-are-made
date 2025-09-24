# RAG Lec-1 Vizuara AI

## 1. Data Ingestion

- Downloading and reading PDFs.
- If we have a ONLY text in the docs, we can use - `PyMuPDF`, can save images but not read the image, for that we need OCR. And a famous one is `tesseract`, it is used to get texts from the image. For tabular data, we use `docling`. It is really good at working with tables, it saves the tables as ACTUAL tables with rows and cols, and can be joined with OCR where it can get tables through the image too by preserving the tables. `docling` can handle markdown tables and joining with an OCR tool.

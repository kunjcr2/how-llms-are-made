# RAG Lec-1 Vizuara AI

## 1. Data Ingestion

- Downloading and reading PDFs.
- If we have a ONLY text in the docs, we can use - `PyMuPDF`, can save images but not read the image, for that we need OCR. And a famous one is `tesseract`, it is used to get texts from the image from hand written stuff or bills image. For tabular data, we use `docling`. It is really good at working with tables, it saves the tables as ACTUAL tables with rows and cols, and can be joined with OCR where it can get tables through the image too by preserving the tables. `docling` can handle markdown tables and joining with an OCR tool. `Mistral AI OCR` is a good one as well ! 
- for scraping websites we use `Firecrawl`,  `beautifulsoup4` for html, `pyppeteer` is good as well for titles and stuff by mentioning styles of what you want.

### Lets start with simple text. **PyMuPDF**. it is 10-15 times faster than docling.

1. We install packages.
2a. We download the pdf and use `fitz` which is pymupdf basically and open it. we fo some preprocess on it and maintain a list of each page with text, nymber of chars; etc in a form of dictionary.
2b. we can get some stats of it as well.
3. 
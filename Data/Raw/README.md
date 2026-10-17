# Raw documents (not in the repository)

The source documents (`STUDENT HANDOUT DOCUMENTS/…`, about 1.3 GB of PDFs, Word files, slides and videos) are excluded from Git
because of their size. To rebuild the corpus from scratch place the folder here as

```
Data/Raw/STUDENT HANDOUT DOCUMENTS/<Academic Calendar | Attendance Rules | … | Student Resource Book>/…
```

and run `python src/preprocess.py`. You do **not** need them to run the project: `Data/processed/` already contains the
rebuilt corpus, the FAISS index and the confidence calibration.

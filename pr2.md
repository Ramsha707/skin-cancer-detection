## Week 2 - backend, database, dataset splits and metrics

FastAPI service layer, SQLAlchemy models and SQLite schema.

### Dataset
Reads the real HAM10000 layout (9,873 usable images across 7,372 lesions). Extensionless metadata \image_id\ values resolve to \.jpg\ files via a cached stem index, so 10k rows resolve in one pass.

### Taxonomy correction
HAM10000 ships **no squamous cell carcinoma images** - its seven \dx\ values are nv/mel/bkl/bcc/akiec/vasc/df. Actinic keratosis is a UV precursor and clinically distinct from SCC, so it is kept as its own class rather than being relabelled \scc\ on 327 images, which would have fabricated ground truth. \asc\ (142 benign vascular lesions) is excluded.

### Splits
Grouped by \lesion_id\ and stratified by dominant class, so no lesion leaks across splits and the ~80% benign majority cannot swamp val/test. Verified zero patient overlap across the three splits.

### Agent partitions
Non-IID via Dirichlet sampling, capped at 1.5x the mean size. Unconstrained sampling left one site with 205 images next to one with 3,462; sizes are now 1,413-2,050 while class mix stays skewed (one site 93% benign, another 23% melanoma).

### Metrics
Macro-F1 and per-class recall are the primary report fields, accuracy demoted to a diagnostic. A real zero-shot MedSigLIP run scored 0.30 accuracy against 0.21 macro-F1 on this data - the gap these fields exist to expose. Malignant-vs-benign AUC now sums malignant-class probabilities instead of thresholding hard labels.

## Verification
- 38 tests pass
- \uff check\ clean
- \
pm run build\ passes
- All 7 API routes return 200 against the real dataset

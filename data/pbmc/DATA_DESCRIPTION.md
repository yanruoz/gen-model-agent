# DATA_DESCRIPTION: pbmc3k

## What this is
Peripheral blood mononuclear cells (PBMCs) from a healthy donor, profiled with
10x Genomics single-cell RNA sequencing (scRNA-seq). This is the "pbmc3k"
dataset used throughout the Scanpy tutorials and widely used as a benchmark for
single-cell analysis methods. Stored as the raw h5ad file from the Scanpy
tutorials.

## Loading
file: pbmc3k_raw.h5ad
url: https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad
md5: e15ea74a89b3bf86022dd0b2eb3d0021

## Structure
- 2,700 cells (samples) x 32,738 genes (features).
- Feature values are raw UMI counts (non-negative integers), not normalized or
  log-transformed.
- Matrix is highly sparse (~97.4% zero entries), typical of scRNA-seq: most
  genes are not detected in most cells (technical and biological dropout).
- Mean ~847 genes detected per cell (range 212 to 3,422), mean ~2,367 total
  counts per cell.
- This is the **raw, unfiltered** dataset: no quality-control filtering (e.g.
  removing low-quality cells, doublets, or lowly-expressed genes) has been
  applied.

## Labels
None. `adata.obs` is empty; there is no cell-type or other annotation attached.
Any objective that needs labels (class-conditional generation, class
balancing) cannot be run as stated on this dataset.

## Feature identity
Each feature is one gene, identified by a gene symbol (e.g. `MIR1302-10`,
`SAMD11`) and an Ensembl gene ID (`var['gene_ids']`). The same 32,738 genes are
measured, in the same order, for every cell.

## Known caveats
- scRNA-seq count matrices are extremely sparse and heavy-tailed; a small
  number of highly-expressed genes can dominate raw totals. Count likelihoods
  (Poisson, negative binomial, zero-inflated variants) are the usual modeling
  choice for raw counts.
- Library size (total counts per cell) varies about 10x between cells.
- No QC has been applied: some cells may be low-quality (very few genes
  detected) or potential doublets (unusually high counts).
- With only 2,700 cells, large generative models can memorize training cells;
  any evaluation should compare against held-out real cells.

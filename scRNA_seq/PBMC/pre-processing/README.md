# Pre-processing for AnnData Export

This directory contains several Jupyter notebooks that outline the preprocessing steps necessary for exporting single-cell RNA sequencing data as an AnnData object. The following notebooks are included:

1. **Export GSE138266 to anndata.ipynb**: This notebook details the preprocessing steps for the GSE138266 dataset, including data loading, normalization, and integration of metadata.

2. **Export GSE144744.ipynb**: This notebook focuses on the GSE144744 dataset. It includes steps such as reading raw data, filtering cells, and adding metadata, which are essential for preparing the dataset for export to AnnData format.

3. **GSE227954.ipynb**: Similar to the previous notebooks, this file outlines the preprocessing steps for the GSE227954 dataset, including quality control and normalization.

4. **GSE194078.ipynb**: This notebook describes the preprocessing steps for the GSE194078 dataset, detailing how to handle raw counts and prepare the data for integration.

## Preprocessing Steps Overview

The preprocessing steps generally include:

- **Data Loading**: Importing raw count data and associated metadata.
- **Quality Control**: Filtering out low-quality cells based on metrics such as the number of features detected and total counts.
- **Normalization**: Adjusting the raw counts to account for differences in sequencing depth and other technical variations.
- **Metadata Addition**: Incorporating relevant metadata for each cell, which is crucial for downstream analysis and interpretation.

These steps are critical for ensuring that the data is in the correct format and quality for export as an AnnData object, which can be used for further analysis in Python environments.

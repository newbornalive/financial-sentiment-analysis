# Upload Guide

This package is intended to replace the current contents of the
`financial-sentiment-analysis` repository.

## Option A: GitHub web interface

1. Open the repository.
2. Remove the existing empty placeholder files.
3. Choose **Add file → Upload files**.
4. Upload the contents of this folder, preserving the directory structure.
5. Commit with:

   `Rebuild financial sentiment project with reproducible pipeline and tests`

## Option B: Git command line

```bash
git clone https://github.com/newbornalive/financial-sentiment-analysis.git
cd financial-sentiment-analysis

# Copy the contents of this package into the cloned repository.
git add .
git commit -m "Rebuild financial sentiment project with reproducible pipeline and tests"
git push origin main
```

## Repository settings to update

Add this description:

`Reproducible financial-text sentiment classification with VADER, FinBERT, model evaluation, tests, and CI.`

Add these topics:

`nlp`, `finbert`, `sentiment-analysis`, `pytorch`, `transformers`,
`financial-data`, `machine-learning`, `model-evaluation`, `python`

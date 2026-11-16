# Frozen Representations, Latent Signal

Code for *"Frozen Representations, Latent Signal: Benchmarking and Unlocking
Tumor-Cellularity Structure in Medical Foundation Models"* (Filip Ręka, Bartosz Minch;
submitted to *Medical Image Analysis*).

We test whether frozen embeddings from eight pretrained models (MedSigLIP,
MedImageInsights, BiomedCLIP, NeuroVFM, Merlin, Med3D, and two generic baselines,
CLIP and an ImageNet-21k ViT) are spatially coherent with total cell count (TCC), an
ADC-derived tumor-cellularity proxy, on the CFB-GBM glioblastoma MRI collection.
Coherence is Geary's *C* on a k-nearest-neighbor graph in embedding space with
same-patient neighbors excluded, against a permutation null. We then train lightweight
adapters (LoRA or a residual projection head) against a differentiable relaxation of
Geary's *C* and evaluate them with nested, patient-grouped cross-validation.

## Repository layout

```
data/datasets/cfb_gbm.py     CFB-GBM loader (T1Gd volumes, ADC, GTV mask, RANO, slice selection)
embeddings/backends/         one file per pretrained model
embeddings/pipeline.py       dataset -> backend -> HDF5 store, with per-scan caching
embeddings/store.py          HDF5 embedding store (one file per backend)
analysis/cellularity.py      TCC (via TumorTwin), k-NN graphs, Geary's C, Moran's I, permutation nulls
analysis/reduction.py        UMAP / PCA / t-SNE
finetune/                    coherence loss, LoRA and projection-head adapters, training loop
scripts/                     entry points for every table and figure (see below)
tests/                       unit tests (data-dependent tests skip without the dataset)
third_party/MedImageInsights git submodule with the MedImageInsights model code and weights
```

## Setup

Python 3.11 with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras        # everything except NeuroVFM
git submodule update --init --recursive
(cd third_party/MedImageInsights && git lfs pull)
```

Each backbone also has its own extra (`merlin`, `med3d`, `biomedclip`, `clip`,
`medsiglip`, `imagenet_vit`, `medimageinsights`), plus `cellularity` (TumorTwin, for
TCC) and `finetune` (peft). MedSigLIP is a gated model on Hugging Face: accept its
license and run `huggingface-cli login` first.

**NeuroVFM** pins its own torch/Python versions and needs `flash-attn`, so it runs in
a separate environment. The setup steps are in the docstring of
`scripts/embed_neurovfm_cfb_gbm.py`.

Embedding extraction and fine-tuning need a GPU; all analysis scripts run on CPU.

## Data

Download CFB-GBM from TCIA (DOI [10.7937/v9pn-2f72](https://doi.org/10.7937/v9pn-2f72))
and point `CFB_GBM_ROOT` at the download folder, the one that contains the `CFB-GBM/`
imaging directory and the collection's clinical TSV files:

```bash
export CFB_GBM_ROOT=/path/to/CFB-GBMv4
```

## Reproducing the paper

Run everything from the repository root (`uv run python scripts/<name>.py`).
Embeddings are written to `stores/` and results to `results/`.

**1. Frozen embeddings** (GPU). The analysis uses every qualifying scan, which is the
`scan` pairing:

```bash
for b in merlin biomedclip medsiglip medimageinsights clip med3d imagenet_vit; do
    CFB_GBM_PAIRING=scan uv run python scripts/embed_cfb_gbm.py $b
done
python scripts/embed_neurovfm_cfb_gbm.py              # in the NeuroVFM environment
for b in biomedclip medsiglip clip imagenet_vit; do  # mid-volume slice (Table 7)
    uv run python scripts/embed_cfb_gbm_midslice.py $b
done
```

**2. Frozen benchmark** (CPU). Run `cellularity_coherence.py` first: it computes the
TCC table (`results/tcc_table.csv`) that every later script reads.

| Paper | Script |
|---|---|
| Table 3, frozen Geary's *C* | `cellularity_coherence.py` |
| Table 4, Moran's *I* | `validate_morans_i.py` |
| Table 5, global pairwise-distance test | `validate_pairwise_distance_target.py` |
| Section 4.1, TCC vs. volume and ADC | `validate_tcc_decomposition.py` |
| Table 6, volume and ADC targets (after the line above) | `validate_volume_residual_coherence.py` |
| Table 7, slice-selection sensitivity | `validate_slice_selection_sensitivity.py` |
| Figure 3, Geary's *C* vs. *k* | `coherence_k_sweep.py` |
| Patient-block permutation null (Limitations) | `validate_cluster_permutation.py`, `validate_cluster_permutation_pairwise.py` |
| ImageNet ViT baseline, side by side with BiomedCLIP and CLIP | `evaluate_imagenet_vit_frozen.py` |
| BiomedCLIP vs. ImageNet ViT paired bootstrap | `compare_biomedclip_imagenet_vit.py` (after step 3) |
| Cohort sex and age | `cohort_demographics.py` |
| Figure 1, example scan | `plot_example_patient.py` |
| Figure 4, frozen UMAP | `plot_tcc_umap.py` |

**3. Fine-tuning** (GPU, then CPU).

| Paper | Script |
|---|---|
| Nested cross-validation, all eight backbones | `finetune_nested_cv.py [backend ...]` (GPU) |
| Tables 8 and 9, collapse checks | `analyze_nested_cv.py` |
| Figure 5, fine-tuned UMAP | `plot_finetuned_umap.py` |
| Merlin/Med3D same-split selection comparison | `finetune_projection_head_robustness_check.py <merlin\|med3d>` |

The single-split training scripts (`finetune_cfb_gbm.py`, `finetune_clip_cfb_gbm.py`,
`finetune_medsiglip_cfb_gbm.py`, `finetune_neurovfm_cfb_gbm.py`,
`finetune_projection_head_cfb_gbm.py`) hold each backbone's model setup and
hyperparameters, which the nested cross-validation and the fine-tuned UMAP reuse.

## Tests

```bash
uv run pytest
```

Tests that need the dataset are skipped unless `CFB_GBM_ROOT` is set.

## License

MIT, see [LICENSE](LICENSE). Model weights and the CFB-GBM data are subject to their own licenses.

## Citation

If you use this code, please cite the paper (citation to be added on publication).

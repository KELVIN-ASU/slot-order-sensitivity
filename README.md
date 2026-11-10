# Slot-Order Sensitivity of Downstream Predictors

Experiments testing whether downstream predictors trained on Slot Attention representations depend on slot order and slot initialization.

## Overview

Slot Attention is permutation-equivariant with respect to its initial slots, so a downstream predictor of scene properties should be permutation-invariant to the order of the final slots. A predictor that concatenates the slots can still learn to rely on their order.

Two predictors are trained on the final slots of a Slot Attention backbone and evaluated on how their predictions change when the slot order or the slot initialization changes:

- **FlatMLP**, which concatenates the four slot vectors in their input order;
- **DeepSets**, which applies a shared network to each slot and averages the resulting features.

The prediction targets are the ten generative factors of the Causal3DIdent dataset (object position, rotation, spotlight position, and three hues). Predictors output a Gaussian distribution for each linear factor and a wrapped normal distribution for each hue, and are trained with the continuous ranked probability score (CRPS).

## Repository structure

```text
slot-order-sensitivity/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── config.json                 configuration of the reported run
├── run_all.py                  entry point
│
├── pipeline.py                 experiment stages
├── data.py                     dataset loading, splits, corruptions
├── models.py                   Slot Attention backbone, FlatMLP, DeepSets
├── training.py                 backbone and predictor training
├── inference.py                evaluation under initializations and orderings
├── probability.py              Gaussian and wrapped-normal scores
├── mixtures.py                 distribution averaging over orderings
├── calibration.py              scale, isotonic, and conformal calibration
├── diagnostics.py              alignment, equivariance, Jacobian, numerical checks
├── statistical_analysis.py     bootstrap intervals and sign-flip tests
├── result_summaries.py, reporting.py, visualization.py
├── permutation_scoring.py, evaluation_cache.py
├── configuration.py, reproducibility.py, artifacts.py
│
├── Backbone Training/
│   └── 1_Baseline_slotAE.ipynb  trains the Slot Attention backbone
│
└── tests/                      unit tests
```

## Experiments

### 1. Slot permutations

For each test image, the final slots are computed once and the predictor is evaluated on all 24 orderings of the same four slots. The experiment measures the RMSE for each ordering and the change in each individual prediction across orderings, relative to its predicted uncertainty.

### 2. Initialization interventions and alignment

Initial slots are a learned mean plus Gaussian noise times a learned standard deviation. Fixed initialization reuses the same noise for every image; resampled initialization draws new noise for each image. Each predictor is evaluated with the reference fixed initialization, an alternative fixed initialization, resampled initialization, and random reassignment of the reference initial slots to slot indices. The resulting final slots are evaluated before and after Hungarian alignment to the reference slots of the same image. The experiment is repeated on three independently trained backbones.

### 3. Training initialization

With a frozen backbone, FlatMLP is trained with fixed initialization and with resampled initialization, and each model is evaluated under both.

### 4. Training procedures and invariant predictors

Predictors are trained with the backbone frozen and jointly with the backbone (CRPS plus reconstruction loss). FlatMLP is also trained with the final slots randomly reordered for each training example (permutation augmentation).

### 5. Distribution averaging

For each predictor, the predictive distributions from all 24 orderings are averaged into one mixture, which is exactly permutation-invariant for a given set of slots.

### Additional analyses

Jacobian symmetry, calibration, image corruptions, numerical validation of the circular CRPS, network timing, and an audit evaluation on a held-out half of the test set.

## Data and checkpoints

Download Causal3DIdent (`trainset.tar` and `testset.tar`) from [Zenodo](https://zenodo.org/records/4784282).

The [v1.0 release](https://github.com/KELVIN-ASU/slot-order-sensitivity/releases/tag/v1.0) contains:

| File | Content |
| --- | --- |
| [`slot_attention_epoch_50.pth`](https://github.com/KELVIN-ASU/slot-order-sensitivity/releases/download/v1.0/slot_attention_epoch_50.pth) | Slot Attention backbone used in all experiments |
| [`20261029_091948_264375_73b2786e.zip`](https://github.com/KELVIN-ASU/slot-order-sensitivity/releases/download/v1.0/20261029_091948_264375_73b2786e.zip) | Complete reported run: trained predictor and replica-backbone checkpoints, result tables, figures, and run manifest |

SHA-256 of the backbone checkpoint:

```text
6266af9ccc42cc336954d8e24cecbd9f57e5b5df257b6039bc47d744b53da326
```

`config.json` expects the data and the backbone checkpoint in the parent folder of the code:

```text
parent_folder/
├── slot-order-sensitivity/        code
├── causal3DIdent_dataset/         trainset.tar, testset.tar
└── SlotAttentionCheckpoints/      slot_attention_epoch_50.pth
```

To use other locations, edit the paths in `config.json`. Relative paths are resolved against that file.

## Reproducibility

All main experiments use five predictor seeds. The three additional backbones each use one predictor seed.

Each run writes a manifest (`run_manifest.json`) with the configuration, software versions, random seeds, and hashes of the inputs and source code.

The reported run used Python 3.12, PyTorch 2.10 with CUDA 13.0, and one NVIDIA RTX 5090 GPU, and took about 25 hours.

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies: numpy, scipy, pandas, Pillow, torch, torchvision, scikit-learn, matplotlib, joblib, tqdm.

## Running the experiments

```bash
python run_all.py --config config.json
```

For a shorter trial, reduce the seed lists and the epoch counts in `config.json`.

Each run writes to a new folder:

```text
results/<date>_<time>_<microseconds>_<config hash>/flatmlp_mechanism/
├── run_manifest.json
├── checkpoints/          trained predictors and additional backbones
└── evaluation/           result tables (CSV) and figures
```

Run the unit tests with

```bash
pytest tests
```

## Summary of empirical findings

With a frozen backbone, FlatMLP trained with fixed initialization has an RMSE of 0.288. Its RMSE rises to 0.724 when new initial slots are drawn for each test image, and returns to 0.290 after the final slots are aligned to the reference slots of the same image. Every one of the 23 other orderings of the final slots also increases its RMSE.

FlatMLP trained with resampled initialization has an RMSE of 0.293 under fixed and 0.294 under resampled initialization. Its average RMSE is the same across orderings, but for about 80% of image-factor predictions the prediction changes by more than half the predicted Gaussian scale when the slots are reordered. Permutation augmentation shows the same pattern.

DeepSets predictions do not change with slot order. Averaging the predictive distributions over all orderings also removes the dependence on order, and gives lower RMSE than DeepSets for FlatMLP trained with resampled initialization or permutation augmentation.

## License

See `LICENSE`.

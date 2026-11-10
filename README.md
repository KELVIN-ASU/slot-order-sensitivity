# Slot-index dependence

This package trains the readouts and additional backbones, runs inference,
and writes scientific summaries under `evaluation/`. It includes the original
experiment families and additional tests of circular scoring, initialization,
permutation sensitivity, and permutation-averaged predictive distributions.

## Run

Use the existing environment that runs the v6 pipeline. Dependency versions
are not pinned, and no package upgrades are required. In particular, keep the
working PyTorch and torchvision installations. If packages are missing, install
only those packages; `requirements.txt` lists the dependencies. The executable
entry point is:

```bash
python run_all.py --config config.json
```

Before running, edit the paths in `config.json`. Relative paths are resolved
against that file. Point `slot_checkpoint` at the supplied pretrained Slot
Attention checkpoint. It is loaded in place, never copied or moved. Set the
training and test archive paths, extraction directory, and separate base paths
for `output_root` and `array_root`. Each launch inserts a parent folder named
with the local date, time, microseconds, and the first eight characters of the
configuration SHA-256 hash. Results and arrays share the same run identifier;
the internal folder names remain unchanged. The hash excludes `output_root`
and `array_root`. Existing parent folders are never reused.

Check the dataset's image-label pairing and the ten factor definitions in
`data.py`, then set `factor_schema_confirmed` to `true`. The final three hue
factors use period pi on the support `[-pi/2, pi/2)`. This explicit setting
prevents running the experiment on an unverified factor schema. Dataset checks
also inspect factor ranges, shard ordering, partition overlap, duplicate image
bytes, and optional supplied split records. Extracted archives must contain
`trainset/` and `testset/` with
`latents_*.npy` shards and their corresponding image folders.

For a shorter trial, reduce seeds, `backbone_epochs`, `predictor_epochs`, and
`score_comparison_epochs` in the same configuration. Keep `predictor_epochs`
greater than `warmup_epochs`. Each launch creates new result and array parents.
There is one execution path; there are no execution modes. Existing results
are protected rather than overwritten or resumed.

## Experiment coverage

| Experiment | Computation |
| --- | --- |
| Primary readouts | Frozen FlatMLP, stochastic-initialization FlatMLP, permutation-augmented FlatMLP, and DeepSets; joint FlatMLP, permutation-augmented FlatMLP, and DeepSets, across all configured seeds |
| Additional backbones | Independent reconstruction training, reconstruction and slot diagnostics, and frozen FlatMLP readouts |
| Initialization | Fixed reference vectors, alternative fixed draws, per-example stochastic draws, and reassignment of the same fixed vectors |
| Permutations | All 24 permutations of slots extracted once for each image and initialization draw |
| Individual sensitivity | Per-image and per-factor prediction changes, uncertainty changes, median, upper quantiles, maxima, and tolerance exceedances |
| Training versus evaluation initialization | Frozen-backbone fixed-trained and stochastic-trained FlatMLP under both fixed and stochastic evaluation, with paired seed-level contrasts |
| Invariant baseline | Equal-weight mixture of the 24 full predictive distributions, including marginal scores, joint mixture likelihood, and distribution intervals |
| Slot mechanism | Cosine-distance Hungarian alignment, slot equivariance, masks and attention diagnostics, initialization stability, Jacobian and readout-weight analyses |
| Calibration | Raw Gaussian, scalar temperature, per-factor temperature, isotonic PIT, absolute conformal, and Gaussian split conformal intervals |
| Other evaluations | PIT, calibration error, factor metrics, paired permutation bootstraps, orbit risk, corruption sweeps, backbone replication, and audit split comparisons |
| Circular scoring | Independent numerical reference, finite-difference and autograd checks, predicted-scale audits, parameter-gradient comparisons, and checkpoint-selection comparisons |
| Scoring and training | Matched validation-only training comparisons using Fourier and quantile scoring estimators |
| Across-seed analysis | Paired effects, seed-level bootstrap intervals, sign-flip comparisons, and multiple-comparison adjustment |

The default configuration trains 35 primary readouts, three additional
backbones, three backbone-replication readouts, and four scoring-comparison
readouts. The 38 primary/replication readouts receive working-test evaluation;
the five stochastic-trained readouts do not receive audit-subset evaluation.
The four scoring-comparison readouts use validation data only. Counts are
derived from the training conditions, including when seed lists are shortened.
Initialization-training conclusions concern the frozen-backbone setting.

Circular CRPS used for main training and evaluation is computed through its
Fourier representation. The original 16- and 64-quantile approximations remain
available for numerical, gradient, training, and model-selection comparisons.
These comparisons produce results; the package does not assume what the
predicted-scale discrepancies or scientific conclusions will be.

`crps_reporting_changes.csv` compares 16- and 64-node scores at reference fixed
initialization, seed 1234 by default, with raw and per-factor-temperature
variances across all 24 permutations. It does not measure reporting changes
for alternative draws or scalar-temperature variances. Independent numerical
references, gradient checks, predicted-scale validation audits, matched training
comparisons, and checkpoint-selection checks remain enabled.

Image corruptions are evaluated in identity slot order with all calibration
methods. Permutation-mixture scoring remains enabled for initialization and
permutation experiments; it is not run for image corruptions.

`corruption_severities` accepts any nonempty list of unique integers from 1 to
5, including a single severity such as `[3]`. Evaluation tables and figures use
only those severities. The parameters defining each severity are unchanged.

Mixtures and their paired probability-mass interval comparisons are evaluated
only for reference fixed initialization and per-example stochastic draws, using
raw and per-factor-temperature variances. With four stochastic draws, this is
ten 24-component mixture evaluations per model. Ordinary identity and orbit
metrics still cover every initialization condition and all three variance
variants. Data-loading workers default to four. Permutation scoring is sequential.

Cosine-distance Hungarian matching retains normalization with epsilon
`1e-12`. Permuting the same initial vectors and recovering their alignment
tests implementation behavior; it alone does not establish a learned mechanism.

Diagnostic tolerance exceedances are recorded as experimental results and do
not terminate execution. `pretraining_unit_tests.csv` and
`empirical_crps_gradients.csv` retain their measurements and add tolerances
and pass/fail flags. Gradients outside the variance clamp are marked unchecked.
`numerical_diagnostics.csv` records autograd agreement, periodicity, clamp
derivatives, the uniform limit, quadrature, and DeepSets invariance.
`slot_equivariance.csv` records all 24 permutations, both errors, the tolerance,
and separate pass/fail flags; `backbone_seed` is retained for trained backbones.

Diagnostic tables are written after each diagnostic step, before subsequent
training or evaluation. Failed comparisons print `WARNING:` messages and are
summarized in the manifest's `warnings` list, with links by filename to their
detailed tables. Nonfinite warning measurements use explicit text in JSON;
their numerical values remain in the scientific tables. Calculations,
comparison tolerances, matching, training, random streams, and configuration
are unchanged. Input validation and errors that prevent computation still
raise exceptions.

## Computation

Images are decoded and resized once, stored as RGB `uint8` pixels, and converted
to floating-point tensors using the same conversion as direct image loading.
Image-based training and evaluation share this storage; corruptions are applied
afterward. `image_cache_max_gib` defaults to 24 GiB. Shared RAM is used only
within this budget and the detected available/shared-memory limits. Larger
caches use atomic disk-backed arrays, which loader workers reopen without
copying the complete cache. Keep `array_root` on a fast local disk when possible.

Frozen readouts reuse evaluation slots for matching backbone, dataset, ordered
indices, preprocessing, initialization condition and draw, device, and extraction
batch size. Mask and attention diagnostics are computed once per matching
frozen-backbone condition and draw, while each model still receives its
scientific rows and distributions. Jointly trained backbones remain separate.
The cache implementation is in `evaluation_cache.py`; callers are in
`inference.py`.

`permutation_scoring.py` scores permutations sequentially in their specified
order, with a `tqdm` progress bar. There is no `scoring_workers` setting.
Training and evaluation show batch progress bars; initialization conditions
and corruption severities are labelled. Console stage messages report start,
completion or failure, and elapsed time. Stage status remains recorded only
in `run_manifest.json`. CPU thread settings are not changed.

## Outputs and reproducibility

```text
results/<date>_<time>_<microseconds>_<hash>/flatmlp_mechanism/
    run_manifest.json
    checkpoints/
        primary/<regime>_<architecture>/seed_<n>/best.pth
        backbone_<n>/frozen_flatmlp/seed_<n>/best.pth
        score_<estimator>/frozen_flatmlp/seed_<n>/best.pth
        backbones/seed_<n>/best.pth
    evaluation/
        training/
        stochastic_init_mechanism/
        figures/
        <summary tables>.csv
experiment_arrays/<date>_<time>_<microseconds>_<hash>/flatmlp_mechanism/
    calibration/
    raw_predictions/
    audit_predictions/
    validation_predictions/  # Selected score-comparison epochs only.
    permutation_evaluation/  # Per-image prediction-change arrays.
    initialization_mechanism/
    symmetry_diagnostics/
    <split arrays and dataset checks>
```

Only `best.pth` is saved for each readout and newly trained backbone. All
epoch-level training and validation history CSVs remain available, including
both CRPS estimators used for checkpoint-selection comparisons. Alternative
epoch weights are not retained. `backbone_checkpoint_every` remains an accepted
configuration field but does not cause periodic checkpoint writes.

Validation prediction arrays are saved only for the selected epoch of each
score-comparison readout: four files with the default configuration. These
arrays supply `score_training_comparison.csv`.

Working-test prediction parameters are retained for reference fixed
initialization and per-example stochastic draws, including all 24 permutation
means and variances, targets, full image indices, and permutation order. Slots
are not included. Calibration predictions and audit predictions are also
retained without slots. Per-image prediction-change arrays, draw variation,
mask/attention distributions, Jacobian diagnostics, and calibration bundles
remain available.

Mixture metrics, aligned predictions, uncertainty changes, and corruption
metrics are computed and summarized in CSVs. Their mixture-distribution arrays,
aligned-prediction arrays, uncertainty-change arrays, raw corruption predictions,
and per-image readout-contribution arrays are not written. Some additional
analyses therefore require rerunning inference.

After successful final validation, the pipeline closes memory mappings and removes only `slot_cache/`, `evaluation_slots/`,
`image_cache/`, and `scoring_arrays/` from `array_root`. A run that fails before
this stage retains its caches. The manifest's final output inventory is written
after cleanup and lists only retained files.

`run_manifest.json` is the sole run-provenance record. It contains the resolved
configuration, run identifier, timestamp, configuration hash, Git commit and working-tree status when available, source hash,
library and device information, stage statuses, and checkpoint paths and
SHA-256 hashes. A package extracted outside a Git checkout has no commit hash;
run from a committed checkout when a Git commit is required. Scientific CSVs
contain experimental condition labels, measurements, and statistics.

Writes are atomic. Existing run parents are refused, and exclusive run locks
prevent concurrent reuse. The resolved result and array paths are printed at
startup and recorded in `run_manifest.json`. Checkpoints, caches, raw arrays,
and calibration bundles are ignored by Git. Code and small summary tables may
be committed. Full per-image exports are outside the output directory.

The global seeding, deterministic settings, device generators, and slot-noise
streams follow the source `flatmlp_permutation_v6.py`, including enabled TF32,
`cudnn.deterministic=True`, `cudnn.benchmark=True`, and deterministic algorithms
without silent fallback. CPU random streams are tested against the source
functions. Identical seeds do not establish identical arithmetic across
different hardware or PyTorch versions.

Gaussian corruption noise uses base seed 100047, plus the full dataset image
index. Each image's noise pattern is scaled across severity levels.

## Reading the results

Dataset-average RMSE is reported alongside individual prediction changes.
Draw averages are computed within a training seed before across-seed
comparisons. A one-seed trial supports descriptive results only.

`prediction_variation_by_factor.csv` also reports
`sigma_normalized_change_p50`, `sigma_normalized_change_p95`, and
`fraction_above_sigma_tolerance`. For each image and factor, the maximum
pairwise mean change over all 24 permutations is divided by the square root
of the mean raw predicted variance over those permutations. The tolerance is
set by `sigma_change_tolerance`, which defaults to `0.5`. Circular mean changes
use wrapped distances; their denominator is the underlying wrapped-Gaussian
scale, not a circular standard deviation. Mean variances are floored at
`1e-15` before taking the square root. Read these measurements alongside the
absolute changes: a large predicted variance can make the normalized change
small without making the prediction stable.

`draw_variation_by_factor.csv` reports the median and 95th percentile of each
image's maximum pairwise identity-order mean change across stochastic draws,
normalized by training SD for linear factors and pi for circular factors.
It includes the number of draws. With fewer than two draws, the new quantiles
are unavailable and labelled `insufficient_draws`; existing draw summaries
and arrays are unchanged.

`permutation_change_vs_initialization_variation.csv` averages normalized
per-factor permutation-change medians and 95th percentiles over stochastic
draws within each model and training seed. It divides each statistic by the
corresponding DeepSets identity-order draw-change statistic for the same group,
regime, training seed, split, and factor. Both statistics and their ratio are
reported. Missing DeepSets matches, fewer than two reference draws, and
reference changes at or below `1e-12` normalized units produce unavailable
ratios with explicit status labels. Replication groups currently have no
DeepSets match. Ratios near or below one indicate changes no larger than the
matched DeepSets initialization variability; they do not establish approximate
invariance. Joint models have separately trained backbones, which also limits
this comparison. The comparison depends on the configured draw set.

Permutation averaging mixes predictive distributions, including their
between-component variation. Circular point predictions use trigonometric
moments; near-zero resultant lengths are flagged rather than interpreted as
well-defined circular means. Inspect these flags when reporting point accuracy.

Original Gaussian-arc interval reports and the original conservative conformal
quantile convention are retained. Mixture intervals contain the requested
predictive probability mass. `distribution_interval_comparison.csv` additionally
compares ordinary and mixture predictions using the same probability-mass
interval definition. Marginal NLL and joint mixture NLL are distinctly labeled.

The initialization and uncertainty summaries include error-conditioned
measurements to help distinguish error growth from a separate uncertainty
effect. They do not automatically establish either interpretation. Timing
tables synchronize device work and measure backbone/readout computation;
they exclude transfers, serialization, and metric computation and are not
complete deployment-latency measurements.
Frozen-model timing rows mark `backbone_timing_shared`: the measured extraction
cost is shared across readouts, while head timings are measured per readout.
Do not sum shared extraction costs to estimate actual pipeline runtime.

The original test partition is reproduced. Its audit subset is not a new,
previously unseen dataset if it was inspected in prior work. Backbone validation
retains the original full validation partition, including images later used
for readout calibration. This limits strict calibration independence and is
recorded by the split diagnostics. Results remain specific to the dataset and
four-slot architecture. No additional dataset or Set Transformer is silently
introduced.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover numerical values and gradients, source random streams,
cosine matching, mixtures, all six calibration methods, paired initialization
contrasts, pixel and noise equivalence for both image cache forms, direct versus
cached slot predictions, sequential permutation order and scores, atomic writes, and
output protection. A separate integration test
actually trains and evaluates the complete pipeline on a small synthetic CPU
dataset. This verifies orchestration and saved artifacts; it does not replace
the full CUDA run on the real dataset.

The integration test checks model counts, excluded audit evaluations, CRPS
reporting scope, diagnostic reuse, and checkpoint hashes. Real-dataset training
and CUDA execution have not been run in this environment.

The integration test additionally checks selected-only checkpoints and
score-comparison arrays, retained prediction fields, omitted intermediate
files, complete training histories, and cache cleanup. A focused cleanup test
checks that memory mappings close and retained files remain intact.

The tests also cover uncertainty-scaled permutation changes, circular wrapping,
factorwise draw summaries, matched DeepSets comparisons, unavailable ratios,
and configurable corruption severities. A second complete CPU integration run
uses just one corruption severity.

The diagnostic tests deliberately exceed tolerances for scoring values and
gradients, score identities, quadrature, DeepSets, and backbone equivariance.
An additional complete CPU integration run verifies that failed measurements
are saved before subsequent stages, warnings are recorded, and execution
reaches completion.

All 36 tests passed on CPU, including the complete pipeline integration tests.
Cached and direct predictions matched exactly, and sequential
permutation scores matched individual scoring calls exactly in the CPU checks. Formatting and lint
checks passed.
Matched synthetic CPU runs preserved retained scientific metrics while omitting
mixture evaluations outside the documented scope. Elapsed-time measurements
are excluded from this comparison. Selected model and optimizer states and retained
prediction-array values matched exactly.

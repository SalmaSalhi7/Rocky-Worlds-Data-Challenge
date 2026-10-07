# Eclipse Pipeline

This repository packages a three-step workflow for JWST eclipse analysis:

1. Pre-processing with Eureka!
2. Individual eclipse fitting
3. Joint eclipse fitting

The code is designed so each eclipse gets its own config file and its own output directory, while shared helper logic lives in `pipeline/utils/`.

## Layout

```text
github_pipeline/
├── configs/
│   ├── preprocessing_template.yaml
│   ├── individual_fit_template.yaml
│   ├── joint_fit_template.yaml
│   └── eureka_templates/
│       ├── S1_miri_lhs1140_eclipse6.ecf
│       ├── S2_miri_lhs1140_eclipse6.ecf
│       └── S3_miri_lhs1140_eclipse6.ecf
├── pipeline/
│   ├── utils/
│   └── ...
├── step1_preprocessing/
├── step2_individual_fit/
└── step3_joint_fit/
```

## Dependencies

Install the scientific stack used by the pipeline:

```bash
pip install -r requirements.txt
```

You will also need Eureka! available in the environment for step 1.

## Step 1: Pre-processing

Edit `configs/preprocessing_template.yaml` for a specific eclipse.

Key fields:

- `preprocessing.backend`: leave blank for now, or set to `eureka`
- `planet_name` and `eclipse_number`
- `paths.raw_input_dir`
- `paths.output_root`
- per-stage paths and ECF parameter overrides
- `stages.stage1.enabled`, `stages.stage2.enabled`, `stages.stage3.enabled`

Run:

```bash
python step1_preprocessing/run_step1.py --config /path/to/preprocessing.yaml
```

This will:

- create `main_path/planet_name/step1/eclipseX/stageY/`
- render the ECF files into each stage folder
- optionally run Eureka! stage 1, 2, and 3 in sequence

## Step 2: Individual Eclipse Fit

Edit `configs/individual_fit_template.yaml`.

Important fields:

- `planet_name`, `eclipse_number`
- `data.path`
- `transit.expected_midtime_mjd`
- `transit.fixed`
- `detrending.model_type`
- `detrending.initial_guess`
- `fit.nsteps`, `fit.nwalkers`
- `plots.*`

Run:

```bash
python step2_individual_fit/run_step2.py --config /path/to/individual_fit.yaml
```

Outputs go to:

`main_path/planet_name/step2/eclipseX/individual_fit_YYYYMMDD_HHMMSS/`

The folder contains:

- `config_resolved.yaml`
- MCMC chains and log probabilities
- text summary
- plots

## Step 3: Joint Eclipse Fit

Edit `configs/joint_fit_template.yaml`.

This config accepts any number of eclipses. Add one entry per eclipse under `eclipses:`.

Run:

```bash
python step3_joint_fit/run_step3.py --config /path/to/joint_fit.yaml
```

Outputs go to:

`main_path/planet_name/joint_fit_YYYYMMDD_HHMMSS/`

## Notes

- The fitting code uses the same detrending model family as the notebook prototype.
- Step 2 and step 3 both fit `dt_s` and `fp` while holding the rest of the transit shape fixed from the YAML.
- The YAML templates are meant to be copied and edited per eclipse or per fit attempt.
- The preprocessing backend field is ready for future expansion; Eureka! is implemented now.


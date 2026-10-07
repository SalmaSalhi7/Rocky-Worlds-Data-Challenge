# Eclipse Pipeline

**This code was created with the help of Codex model GPT-5.6 Sol**.

This repository packages a three-step workflow for analyzing the raw eclipse data, from pre-processing to producing a joint fit of X lightcurves:

1. Pre-processing with Eureka!
2. Individual eclipse fitting
3. Joint eclipse fitting

The code is designed so each eclipse gets its own config file and its own output directory, while shared helper logic lives in `pipeline/utils/`.


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

To run the pre-processing step with Eureka!:

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

You have the option of applying a detrending model to the data, which is fit in conjuction with the eclipse model, as well as any GPs you wish to add. For my data challenge results, I only used a `linear` model for each eclipse, because I opted to trim the ramp instead of attempting to fit it. No GPs were used for the final results. 

To run the individual fit:

```bash
python step2_individual_fit/run_step2.py --config /path/to/individual_fit.yaml
```

You'll find the outputs in:

`main_path/planet_name/step2/eclipseX/individual_fit_YYYYMMDD_HHMMSS/`

The folder contains the following outputs:

- `config_resolved.yaml`
- `binned_fit.png`: shows the binned data with the fitted model
- `corner.png`: shows the unbinned data with the corrected unbinned data on top, as well as the fitted model
- `rednoise.png`: shows the RMS as a function of bin size
- `walkers.png`: shows the progress of all walkers for all parameters
- `summary.json`: displays the fitted parameters and their uncertainties

## Step 3: Joint Eclipse Fit

Edit `configs/joint_fit_template.yaml`.

This config accepts any number of eclipses. Add one entry per eclipse under `eclipses:`.

Important fields:

- `planet_name`, `eclipse_number`
- `data.path`
- `submission.enabled`: if True, then you have to provide it a json file for the submission
- `fit.sampler`
- `eclipses.eclipse_number`
- `eclipses.transit`
- `eclipses.detrending`

You have the option of adding detrending models to each eclipse. The detrending models are fit separately, as is the eclipse, but the likelihoods are combined when sampling. 

To run the joint fit:

```bash
python step3_joint_fit/run_step3.py --config /path/to/joint_fit.yaml
```

You'll find the outputs in:

`main_path/planet_name/joint_fit_YYYYMMDD_HHMMSS/`

The folder contains the following outputs:

- `config_resolved.yaml`
- `combined_binned_fit.png`: shows the combined data across all eclipses and the binned fit on top
- `corner.png`: a corner plot for all fitted parameters 
- `stacked_binned_fit.png`: shows all the eclipses stacked in one plot
- `best_params.npy`: shows all the best fit values for each of the fitted parameters


# TO RECREATE OUR RESULTS FOR THE DATA CHALLENGE: 

1. Run step 1 using the `preprocessing_template.yaml` provided. 
2. Using the `trim_eclipse_h5.ipynb` notebook, trim each outputted `stage3/*.h5` file to remove the ramp. 
3. (Optional): Use the provided `individual_fit_eclipse.yaml` for each eclipse. You can modify the parameters however you like for each eclipse. 
4. Run step 3 to get the joint fit that was submitted to the Rocky Worlds Data Challenge. Use the provided `joint_fit_template.yaml`--the parameters are already set to exactly what I used to produce the result I submitted. Change the file paths depending on where you stored the trimmed cleaned data. 

I've provided the submission forms for all of our submissions; the one that produced the highest score is `form_LHS1140b_ecl4times_20260827_164049.json`, which contains the results from the fits using the `joint_fit_template.yaml` provided. 

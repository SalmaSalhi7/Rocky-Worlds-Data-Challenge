from __future__ import annotations
import argparse
from pathlib import Path

import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pipeline.utils.config import copy_yaml, load_yaml, dump_yaml
from pipeline.utils.ecf import render_ecf_template
from pipeline.utils.paths import build_step1_root, ensure_dir, get_planet_name




def resolve_stage_paths(config: dict, planet_name: str) -> dict:
    preprocessing = config.get("preprocessing", {})
    output_root = Path(config["paths"]["output_root"])
    eclipse_number = config["eclipse_number"]
    step1_root = build_step1_root(output_root, planet_name, eclipse_number)
    ensure_dir(step1_root)

    resolved = {}
    previous_output = None
    for stage_name in ("stage1", "stage2", "stage3"):
        stage_cfg = dict(preprocessing.get("stages", {}).get(stage_name, {}))
        stage_dir = ensure_dir(step1_root / stage_name)
        template_path = Path(stage_cfg.get("template_path") or config["paths"]["template_dir"]) / stage_cfg['template']
        if not template_path.name:
            template_path = Path(config["paths"]["template_dir"]) / stage_cfg['template']
            print('changed template name')
        if not template_path.exists():
            raise FileNotFoundError(f"Missing template for {stage_name}: {template_path}")

        inputdir = stage_cfg.get("inputdir")
        if not inputdir and previous_output is not None:
            inputdir = str(previous_output)
        outputdir = stage_cfg.get("outputdir") or str(stage_dir)
        updates = dict(stage_cfg.get("params", {}))
        updates["topdir"] = config["paths"].get("topdir", "/")
        if inputdir:
            updates["inputdir"] = inputdir
        updates["outputdir"] = outputdir

        ecf_name = stage_cfg.get("ecf_name") or template_path.name
        ecf_name = ecf_name.replace('template', 'eclipse'+str(eclipse_number))
        ecf_path = stage_dir / ecf_name
        render_ecf_template(template_path, ecf_path, updates)
        resolved[stage_name] = {
            "enabled": bool(stage_cfg.get("enabled", True)),
            "stage_dir": str(stage_dir),
            "ecf_path": str(ecf_path),
            "inputdir": inputdir,
            "outputdir": outputdir,
        }
        previous_output = outputdir

    return resolved


def run_eureka_stage(stage_name: str, eventlabel: str, stage_dir: Path):
    if stage_name == "stage1":
        import eureka.S1_detector_processing.s1_process as s1
        return s1.rampfitJWST(eventlabel, ecf_path=str(stage_dir))
    if stage_name == "stage2":
        import eureka.S2_calibrations.s2_calibrate as s2

        return s2.calibrateJWST(eventlabel, ecf_path=str(stage_dir))
    if stage_name == "stage3":
        import eureka.S3_data_reduction.s3_reduce as s3

        return s3.reduce(eventlabel, ecf_path=str(stage_dir))
    raise ValueError(f"Unknown Eureka stage: {stage_name}")

def run_exotedrf_stage(config: Path, output_root: Path):
    # try:
    #     config_file = config_dir
    # except IndexError:
    #     raise FileNotFoundError('Config file must be provided')
    # config = parse_config(config_file)
    # save_config(config)
    input_files = unpack_files(config)
    if 1 in config['run_stages']:
        from exotedrf.stage1 import run_stage1 
        # Determine which steps to run and which to skip.
        steps = ['DQInitStep', 'EmiCorrStep', 'ResetStep', 'SuperBiasStep', 'RefPixStep',
                 'DarkCurrentStep', 'OneOverFStep_grp', 'LinearityStep', 'JumpStep', 'RampFitStep',
                 'GainScaleStep']
        stage1_skip = []
        for step in steps:
            if config[step] == 'skip':
                if step == 'OneOverFStep_grp':
                    stage1_skip.append('OneOverFStep')
                else:
                    stage1_skip.append(step)
        # Run stage 1.
        stage1_results = run_stage1(input_files,
                                    mode=config['observing_mode'],
                                    soss_background_model=config['soss_background_file'],
                                    baseline_ints=config['baseline_ints'],
                                    oof_method=config['oof_method'],
                                    superbias_method=config['superbias_method'],
                                    soss_timeseries=config['soss_timeseries'],
                                    soss_timeseries_o2=config['soss_timeseries_o2'],
                                    save_results=config['save_results'],
                                    pixel_masks=config['outlier_maps'],
                                    force_redo=config['force_redo'],
                                    flag_up_ramp=config['flag_up_ramp'],
                                    rejection_threshold=config['jump_threshold'],
                                    flag_in_time=config['flag_in_time'],
                                    time_rejection_threshold=config['time_jump_threshold'],
                                    output_tag=output_root + config['output_tag_stage1'],
                                    skip_steps=stage1_skip,
                                    do_plot=config['do_plots'],
                                    soss_inner_mask_width=config['soss_inner_mask_width'],
                                    soss_outer_mask_width=config['soss_outer_mask_width'],
                                    nirspec_mask_width=config['nirspec_mask_width'],
                                    centroids=config['centroids'],
                                    hot_pixel_map=config['hot_pixel_map'],
                                    miri_drop_groups=config['miri_drop_groups'],
                                    saturation_threshold=config['saturation_threshold'],
                                    f277w=config['f277w'], **config['stage1_kwargs'])
    else:
        stage1_results = input_files
        
    if 2 in config['run_stages']:
        from exotedrf.stage1 import run_stage2
        # Determine which steps to run and which to skip.
        steps = ['AssignWCSStep', 'Extract2DStep', 'SourceTypeStep', 'WaveCorrStep',
                 'FlatFieldStep', 'OneOverFStep_int', 'BackgroundStep', 'BadPixStep',
                 'PCAReconstructStep']
        stage2_skip = []
        for step in steps:
            if config[step] == 'skip':
                if step == 'OneOverFStep_int':
                    stage2_skip.append('OneOverFStep')
                else:
                    stage2_skip.append(step)
        # Run stage 2.
        stage2_results = run_stage2(stage1_results,
                                    mode=config['observing_mode'],
                                    soss_background_model=config['soss_background_file'],
                                    baseline_ints=config['baseline_ints'],
                                    save_results=config['save_results'],
                                    force_redo=config['force_redo'],
                                    space_thresh=config['space_outlier_threshold'],
                                    time_thresh=config['time_outlier_threshold'],
                                    remove_components=config['remove_components'],
                                    pca_components=config['pca_components'],
                                    soss_timeseries=config['soss_timeseries'],
                                    soss_timeseries_o2=config['soss_timeseries_o2'],
                                    oof_method=config['oof_method'],
                                    output_tag=output_root + config['output_tag_stage2'],
                                    skip_steps=stage2_skip,
                                    generate_lc=config['generate_lc'],
                                    soss_inner_mask_width=config['soss_inner_mask_width'],
                                    soss_outer_mask_width=config['soss_outer_mask_width'],
                                    nirspec_mask_width=config['nirspec_mask_width'],
                                    pixel_masks=config['outlier_maps'],
                                    f277w=config['f277w'],
                                    do_plot=config['do_plots'],
                                    centroids=config['centroids'],
                                    miri_trace_width=config['miri_trace_width'],
                                    miri_background_width=config['miri_background_width'],
                                    miri_background_method=config['miri_background_method'],
                                    **config['stage2_kwargs'])
        stage2_results, deepframe = stage2_results
    else:
        stage2_results = input_files
        deepframe = config['deepframe']
        
    if 3 in config['run_stages']:
        from exotedrf.stage1 import run_stage3
        # If a deepframe is passed in the config file, it takes precedence over anything
        # calculated in Stage 2.
        if config['deepframe'] is None:
            this_deepframe = deepframe
        else:
            this_deepframe = config['deepframe']
        stage3_results = run_stage3(stage2_results,
                                    save_results=config['save_results'],
                                    force_redo=config['force_redo'],
                                    extract_method=config['extract_method'],
                                    soss_specprofile=config['soss_specprofile'],
                                    centroids=config['centroids'],
                                    extract_width=config['extract_width'],
                                    extract_width_soss2=config['extract_width_soss2'],
                                    st_teff=config['st_teff'],
                                    st_logg=config['st_logg'],
                                    st_met=config['st_met'],
                                    planet_letter=config['planet_letter'],
                                    output_tag=output_root + config['output_tag_stage3'],
                                    do_plot=config['do_plots'],
                                    deepframe=this_deepframe,
                                    **config['stage3_kwargs'])

    return
        
    
        


def main():
    parser = argparse.ArgumentParser(description="Render and optionally run Eureka! preprocessing stages.")
    parser.add_argument("--config", required=True, help="Path to the preprocessing YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Only render ECFs; do not execute Eureka!.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    preprocessing = config.get("preprocessing", {})
    backend = str(preprocessing.get("backend", "")).strip().lower() 
    print('backend: ', backend)
    if backend != "eureka" and backend != "exotedrf":
        raise NotImplementedError(f"Preprocessing backend '{backend}' is not implemented yet.")
    
    planet_name = get_planet_name(config)
    output_root = Path(config["paths"]["output_root"])
    eclipse_number = config["eclipse_number"]
    step1_root = build_step1_root(output_root, planet_name, eclipse_number)
    ensure_dir(step1_root)
    
    copy_yaml(args.config, step1_root, "config_input.yaml")
    resolved = resolve_stage_paths(config, planet_name)
    dump_yaml({"resolved": resolved}, step1_root / "resolved_paths.yaml")

    if args.dry_run:
        return
        
    if backend == "eureka":
        print('Running the Eureka! extraction...')
        eventlabel_base = preprocessing.get("eventlabel_base") or planet_name
        eventlabel = f"{eventlabel_base}_eclipse{eclipse_number}"

        for stage_name in ("stage1", "stage2", "stage3"):
            stage_cfg = resolved[stage_name]
            if not stage_cfg["enabled"]:
                continue
            run_eureka_stage(stage_name, eventlabel, Path(stage_cfg["stage_dir"]))
            
    if backend == "exotedrf":
        from pipeline.utils.run_DMS import save_config, unpack_files
        print('Running the exoTEDrf extraction...')
        config_dms_dir = preprocessing['exotedrf']['exotedrf_yaml_dir']
        config_dms = load_yaml(config_dms_dir)
        output_root = config["paths"]["output_root"] + planet_name + '/'
        print(output_root)
        run_exotedrf_stage(config_dms, output_root)


if __name__ == "__main__":
    main()


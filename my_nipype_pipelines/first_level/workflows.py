from my_nipype_pipelines.first_level.utils \
    import (create_contrasts, create_subject_info_from_bids, smooth_imgs)


def create_first_level_spm_wf(bold_img,
                              confounders_file,
                              events_file,
                              output_dir,
                              repetition_time,
                              robuts_wls=True,
                              mask_file=None,
                              confounders=None,
                              contrasts=None,
                              high_pass=None,
                              fwhm=None):
    """
    Parameters
    ----------
    bold_img : nifti file
        Bold image.
    confounders_file : tabular .tsv file
        Tabular file from fmriprep with regressors.
    events_file : tabular .tsv file
        Tabular file with events conditions, durations and onsets as columns.
    output_dir : path like string
        Directory where to store output first-level estimations.
    repetition_time : float
        Repetition time of the acquisition.
    robust_wls: bool
        Model estimation using RobustWLS toolbox or not. The default is True.
    mask_file : nifti file, optional
        Image for explicitly masking the analysis. The default is None.
    confounders : {List of strings or None}, optional
        List with the columns from the confounders file to use as
        confounders. If None, the 6 motion parameters are used.
        The default is None.
    contrasts : List, optional
        List with contrasts. If None, all possible single estimations and
        pairwise contrasts are computed. The default is None.
    high_pass : float, optional
        Rate (in secs) to high-pass filter the data. The default is None.
    fwhm : float, optional
        Smoothing applied to first-level estimations. The default is None.

    Returns
    -------
    Nipype worflow with the pipeline.

    """

    from nipype.pipeline import engine as pe
    import nipype.algorithms.modelgen as model   # model generation
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces import spm
    from nipype.interfaces import io as nio
    from nipype.interfaces.utility import Function, Merge
    from my_nipype_pipelines.first_level.interfaces import Level1DesignWLS

    # TODO: Maybe it's more convenient to ask to pass directly a data frame.
    # as confounders (with column names). This way would be more general,
    # not restricted to only fmriprep like confounders files.
    if confounders is None:
        motion = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        confounders = motion

    subject_info = create_subject_info_from_bids(events=events_file,
                                                 confounds=confounders_file,
                                                 regressor_names=confounders)

    if contrasts is None:
        contrasts = create_contrasts(events=events_file)

    if fwhm:
        try:
            fwhm = float(fwhm)
        except ValueError:
            raise print(f"error convert {fwhm} to float")

    if robuts_wls:
        wf = pe.Workflow(name="first_level_spm_wls")
    else:
        wf = pe.Workflow(name="first_level_spm")

    gun_zip = pe.Node(Gunzip(in_file=bold_img),
                      name="unzip_nifti")

    # TODO: pass units as input argument in function?
    specify_model = pe.Node(
        model.SpecifySPMModel(input_units="secs",
                              output_units="secs",
                              time_repetition=repetition_time,
                              high_pass_filter_cutoff=high_pass,
                              concatenate_runs=False,
                              subject_info=subject_info),
        name="specify_model")

    wf.connect(gun_zip, 'out_file', specify_model, 'functional_runs')

    # TODO: pass bases as input argument in function
    if robuts_wls:
        design = pe.Node(Level1DesignWLS(interscan_interval=repetition_time,
                                         bases={'hrf': {'derivs': [0, 0]}},
                                         model_serial_correlations='wls',
                                         timing_units='secs'),
                         name="level1_design")
    else:
        design = pe.Node(spm.Level1Design(interscan_interval=repetition_time,
                                          bases={'hrf': {'derivs': [0, 0]}},
                                          model_serial_correlations='AR(1)',
                                          timing_units='secs'),
                         name="level1_design")

    wf.connect(specify_model, 'session_info',  design, 'session_info')

    if mask_file:
        gun_zip_mask = pe.Node(Gunzip(in_file=mask_file),
                               name="unzip_mask")
        wf.connect(gun_zip_mask, 'out_file',  design, 'mask_image')

    # TODO: The same as above regarding inputs
    estimate_model = pe.Node(
        spm.model.EstimateModel(estimation_method={'Classical': 1},
                                write_residuals=False),
        name="estimate_model")

    wf.connect(design, 'spm_mat_file', estimate_model, 'spm_mat_file')

    estimate_contrasts = pe.Node(
        spm.model.EstimateContrast(contrasts=contrasts),
        name="estimate_contrasts")

    wf.connect(estimate_model, "spm_mat_file",
               estimate_contrasts, "spm_mat_file")
    wf.connect(estimate_model, "beta_images",
               estimate_contrasts, "beta_images")
    wf.connect(estimate_model, "residual_image",
               estimate_contrasts, "residual_image")

    merge_estimations = pe.Node(Merge(numinputs=3),
                                name="merge_estimations")

    wf.connect(estimate_model, "beta_images", merge_estimations, "in1")
    wf.connect([(estimate_contrasts, merge_estimations,
                 [("con_images", "in2"),
                  ("spmT_images", "in3")])
                ])

    smoothing_estimations = pe.Node(
        Function(input_names=["imgs", "fwhm"],
                 output_names=["smoothed_estimations"],
                 function=smooth_imgs),
        name="smooth_estimations")
    smoothing_estimations.inputs.fwhm = fwhm
    wf.connect(merge_estimations, "out", smoothing_estimations, "imgs")

    # Save data
    datasink = pe.Node(nio.DataSink(base_directory=output_dir),
                       name='data_sink')

    wf.connect(estimate_model, "beta_images", datasink, "@foo")
    wf.connect([(estimate_contrasts,
                 datasink, [("con_images", "@foo.@con_images"),
                            ("spmT_images", "@foo.@spmT_images"),
                            ("spm_mat_file", "@foo.@spm_mat_file")])
                ])

    wf.connect(smoothing_estimations, "smoothed_estimations",
               datasink, "@foo.@smoothed_estimations")

    return wf


def create_first_level_fsl_wf(bold_img,
                              confounders_file,
                              events_file,
                              output_dir,
                              repetition_time,
                              mask_file=None,
                              confounders=None,
                              contrasts=None,
                              high_pass=None,
                              fwhm=None):

    from nipype.pipeline import engine as pe
    import nipype.algorithms.modelgen as model

    from nipype.interfaces import fsl
    from nipype.interfaces import io as nio
    from nipype.interfaces.utility import Function, Merge

    # TODO: Maybe it's more convenient to ask to pass directly a data frame.
    # as confounders (with column names). This way would be more general,
    # not restricted to only fmriprep like confounders files.
    if confounders is None:
        motion = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        confounders = motion

    subject_info = create_subject_info_from_bids(events=events_file,
                                                 confounds=confounders_file,
                                                 regressor_names=confounders)

    if contrasts is None:
        contrasts = create_contrasts(events=events_file)

    if fwhm:
        try:
            fwhm = float(fwhm)
        except ValueError:
            raise print(f"error convert {fwhm} to float")

    wf = pe.Workflow(name="first_level_fsl")

    # If mask file, apply mask to input bold data
    if mask_file:
        mask_bold = pe.Node(fsl.maths.ApplyMask(in_file=bold_img,
                                                mask_file=mask_file),
                            name="mask_bold")
    else:
        mask_bold = pe.Node(pe.utils.IdentityInterface(fields=["out_file"]),
                            name="mask_bold_nomask")
        mask_bold.inputs.out_file = bold_img

    # Node to specify the FSL Model
    specify_model = pe.Node(
        model.SpecifyModel(parameter_source='FSL',
                           input_units='secs',
                           high_pass_filter_cutoff=high_pass,
                           time_repetition=repetition_time,
                           subject_info=subject_info),
        name="specify_model")

    wf.connect(mask_bold, 'out_file', specify_model, "functional_runs")

    # Node to generate the FEAT specific files
    design = pe.Node(fsl.Level1Design(bases={'dgamma': {'derivs': False}},
                                      model_serial_correlations=True,
                                      contrasts=contrasts,
                                      interscan_interval=repetition_time),
                     name="design")

    wf.connect(specify_model, "session_info", design, "session_info")

    feat = pe.Node(fsl.FEAT(), name='FEAT')
    wf.connect(design, "fsf_files", feat, "fsf_file")

    feat_select = pe.Node(nio.SelectFiles({
        'param_estimates': 'stats/pe*.nii.gz',
        'copes': 'stats/cope*.nii.gz',
        'varcopes': 'stats/varcope*.nii.gz',
        'tstats': 'stats/tstat*.nii.gz',
        'zstats': 'stats/zstat*.nii.gz'}),
        name='feat_select')

    wf.connect(feat, "feat_dir", feat_select, "base_directory")

    merge_estimations = pe.Node(Merge(numinputs=5),
                                name="merge_estimations")

    wf.connect([(feat_select, merge_estimations,
                 [("param_estimates", "in1"),
                  ("copes", "in2"),
                  ("varcopes", "in3"),
                  ("tstats", "in4"),
                  ("zstats", "in5")])
                ])

    smoothing_estimations = pe.Node(
        Function(input_names=["imgs", "fwhm"],
                 output_names=["smoothed_estimations"],
                 function=smooth_imgs),
        name="smooth_estimations")
    smoothing_estimations.inputs.fwhm = fwhm
    wf.connect(merge_estimations, "out", smoothing_estimations, "imgs")

    # Save data
    datasink = pe.Node(nio.DataSink(base_directory=output_dir),
                       name='data_sink')

    wf.connect(feat_select, "param_estimates", datasink, "@foo")
    wf.connect([(feat_select,
                 datasink, [("copes", "@foo.@copes"),
                            ("varcopes", "@foo.@varcopes"),
                            ("tstats", "@foo.@tstats"),
                            ("zstats", "@foo.@zstats")])
                ])
    wf.connect(smoothing_estimations, "smoothed_estimations",
               datasink, "@foo.@smoothed_estimations")

    return wf


def create_first_level_nilearn_wf(bold_img,
                                  confounders_file,
                                  events_file,
                                  output_dir,
                                  repetition_time,
                                  mask_file=None,
                                  confounders=None,
                                  contrasts=None,
                                  high_pass=None,
                                  fwhm=None):
    raise NotImplementedError


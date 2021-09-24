from my_nipype_pipelines.first_level.utils \
    import (create_contrasts, create_subject_info_from_bids, smooth_imgs)


def create_first_level_spm_wls_wf(bold_img,
                                  confounders_file,
                                  events_file,
                                  output_dir,
                                  repetition_time,
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
    from nipype.interfaces.utility import Function
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

    wf = pe.Workflow(name="first_level_spm_wls")
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
    design = pe.Node(Level1DesignWLS(interscan_interval=repetition_time,
                                     bases={'hrf': {'derivs': [0, 0]}},
                                     model_serial_correlations='wls',
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

    smoothing_betas = pe.Node(Function(input_names=["imgs", "fwhm"],
                                       output_names=["smoothed_betas"],
                                       function=smooth_imgs),
                              name="smooth_betas")
    smoothing_betas.inputs.fwhm = fwhm
    wf.connect(estimate_model, "beta_images", smoothing_betas, "imgs")

    smoothing_contrasts = pe.Node(Function(input_names=["imgs", "fwhm"],
                                           output_names=["smoothed_cons"],
                                           function=smooth_imgs),
                                  name="smooth_contrasts")
    smoothing_contrasts.inputs.fwhm = fwhm
    wf.connect(estimate_contrasts, "con_images", smoothing_contrasts, "imgs")

    smoothing_stats = pe.Node(Function(input_names=["imgs", "fwhm"],
                                       output_names=["smoothed_stats"],
                                       function=smooth_imgs),
                              name="smooth_stats")
    smoothing_stats.inputs.fwhm = fwhm
    wf.connect(estimate_contrasts, "spmT_images", smoothing_stats, "imgs")

    # Save data
    datasink = pe.Node(nio.DataSink(base_directory=output_dir),
                       name='data_sink')

    wf.connect(estimate_model, "beta_images", datasink, "@foo")
    wf.connect([(estimate_contrasts,
                 datasink, [("con_images", "@foo.@con_images"),
                            ("spmT_images", "@foo.@spmT_images"),
                            ("spm_mat_file", "@foo.@spm_mat_file")])
                ])
    wf.connect(smoothing_betas, "smoothed_betas",
               datasink, "@foo.@smoothed_betas")
    wf.connect(smoothing_contrasts, "smoothed_cons",
               datasink, "@foo.@smoothed_cons")
    wf.connect(smoothing_stats, "smoothed_stats",
               datasink, "@foo.@smoothed_stats")

    return wf

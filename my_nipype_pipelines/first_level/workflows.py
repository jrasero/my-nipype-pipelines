from my_nipype_pipelines.first_level.utils \
    import (create_contrasts, create_subject_info_from_bids, smooth_imgs)


def create_first_level_spm_wls_wf(bold_img,
                                  confounders_file,
                                  events_file,
                                  output_dir,
                                  repetition_time,
                                  confounders=None,
                                  contrasts=None,
                                  high_pass=None,
                                  fwhm=None):
    """

    Parameters
    ----------
    bold_img : TYPE
        DESCRIPTION.
    confounders_file : TYPE
        DESCRIPTION.
    events_file : TYPE
        DESCRIPTION.
    output_dir : TYPE
        DESCRIPTION.
    repetition_time : TYPE
        DESCRIPTION.
    confounders : TYPE, optional
        DESCRIPTION. The default is None.
    contrasts : TYPE, optional
        DESCRIPTION. The default is None.
    high_pass : TYPE, optional
        DESCRIPTION. The default is None.
    fwhm : TYPE, optional
        DESCRIPTION. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    wf : TYPE
        DESCRIPTION.

    """

    from nipype.pipeline import engine as pe
    import nipype.algorithms.modelgen as model   # model generation
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces import spm
    from nipype.interfaces import io as nio
    from nipype.interfaces.utility import Function
    from my_nipype_pipelines.first_level.interfaces import Level1DesignWLS

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

    wf.connect(estimate_model, "beta_images", datasink, "first_level")
    wf.connect([(estimate_contrasts,
                 datasink, [("con_images", "first_level.@con_images"),
                            ("spmT_images", "first_level.@spmT_images"),
                            ("spm_mat_file", "first_level.@spm_mat_file")])
                ])
    wf.connect(smoothing_betas, "smoothed_betas",
               datasink, "first_level.@smoothed_betas")
    wf.connect(smoothing_contrasts, "smoothed_cons",
               datasink, "first_level.@smoothed_cons")
    wf.connect(smoothing_stats, "smoothed_stats",
               datasink, "first_level.@smoothed_stats")

    return wf

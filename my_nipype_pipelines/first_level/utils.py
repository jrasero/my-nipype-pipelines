import pandas as pd


def create_subject_info_from_bids(events,
                                  confounds,
                                  regressor_names):

    from nipype.interfaces.base import Bunch

    if isinstance(events, pd.DataFrame) is False:
        events = pd.read_csv(events, sep="\t")

    if isinstance(confounds, pd.DataFrame) is False:
        confounds = pd.read_csv(confounds, sep="\t")

    if all(item in confounds.columns for item in regressor_names) is False:
        raise ValueError(f"{regressor_names} not in confounders")

    # conditions
    conditions = list(events.trial_type.unique())
    # onsets
    onsets = [list(events[events.trial_type == cond].onset)
              for cond in conditions]
    # durations
    durations = [list(events[events.trial_type == cond].duration)
                 for cond in conditions]
    # regressors
    regressors = [list(confounds[reg].fillna(0)) for reg in regressor_names]

    info = [Bunch(conditions=conditions,
                  onsets=onsets,
                  durations=durations,
                  regressors=regressors,
                  regressor_names=regressor_names,
                  amplitudes=None,
                  tmod=None,
                  pmod=None)
            ]
    return info


def create_contrasts(events):
    if isinstance(events, pd.DataFrame) is False:
        events = pd.read_csv(events, sep="\t")

    # conditions
    conditions = list(events.trial_type.unique())

    # t stats for each condition
    contrasts_uni = [[cond, "T", [cond], [1]] for cond in conditions]

    # t-stat por  pairwise contrasts
    constrasts_pair = []
    for ii, cond1 in enumerate(conditions):
        for jj, cond2 in enumerate(conditions):
            if jj <= ii:
                continue
            constrasts_pair += [[f"{cond1} vs {cond2}",
                                 "T",
                                 [cond1, cond2],
                                 [1, -1]]]

    contrasts = contrasts_uni + constrasts_pair

    return contrasts


def smooth_imgs(imgs, fwhm):
    from nilearn.image import smooth_img
    from pathlib import Path
    import os

    smoothed_imgs = smooth_img(imgs=imgs, fwhm=fwhm)
    pth = os.path.dirname(os.getcwd())

    filenames = [pth + "/" + "smoothed_" + Path(img).name
                 for img in imgs]
    for img, filename in zip(smoothed_imgs, filenames):
        img.to_filename(filename)

    return filenames

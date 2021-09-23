from nipype.interfaces.base import traits
from nipype.interfaces.spm.model import (Level1Design,
                                         Level1DesignInputSpec)


class WLSLevel1DesignInputSpec(Level1DesignInputSpec):
    model_serial_correlations = traits.Enum(
        'AR(1)',
        'FAST',
        'wls',
        'none',
        field='cvi',
        desc=('Model serial correlations '
              'for WLS toolbox: '
              'AR(1), FAST, wls or none. ')
    )


class Level1DesignWLS(Level1Design):
    input_spec = WLSLevel1DesignInputSpec

    _jobtype = "tools.rwls"
    _jobname = "fmri_rwls_spec"

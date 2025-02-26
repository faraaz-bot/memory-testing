
import distutils.core as dc
import numpy as np
import os

if os.environ.get('CC') is None:
    os.environ['CC'] = 'hipcc'

hipfft = dc.Extension('hipfft',
                      sources=['pyhipfft.cpp'],
                      include_dirs = ['/opt/rocm/hipfft/include', np.get_include()],
                      library_dirs = ['/opt/rocm/hipfft/lib'],
                      libraries = ['hipfft'])

dc.setup(name='hipfft',
         version='0.8',
         description="hipFFT wrapper.",
         ext_modules=[hipfft]
)

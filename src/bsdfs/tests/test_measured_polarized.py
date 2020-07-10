import mitsuba
import pytest
import enoki as ek
import numpy as np

from mitsuba.python.test.util import fresolver_append_path

@fresolver_append_path
def test01_evaluation(variant_scalar_spectral_polarized):
    from mitsuba.core import Vector3f, Frame3f
    from mitsuba.core.xml import load_dict
    from mitsuba.render import BSDFContext, SurfaceInteraction3f

    # Here we load a small example pBRDF file and evaluate the BSDF for a fixed
    # incident and outgoing position. Any future changes to polarization frames
    # or table interpolation should keep the values below unchanged.
    #
    # For convenience, we use a pBRDF where the usual resolution of parameters
    # (phi_d x theta_d x theta_h x wavlengths x Mueller mat. ) was significantly
    # downsampled from (361 x 91 x 91 x 5 x 4 x 4) to (22 x 9 x 9 x 5 x 4 x 4).

    bsdf = load_dict({'type': 'measured_polarized',
                      'filename': 'resources/data/tests/pbrdf/pbrdf_spectralon_lowres.tensor'})

    phi_i   = 30 * ek.pi/180
    theta_i = 10 * ek.pi/180
    wi = Vector3f([ek.sin(theta_i)*ek.cos(phi_i),
                   ek.sin(theta_i)*ek.sin(phi_i),
                   ek.cos(theta_i)])

    ctx = BSDFContext()
    si = SurfaceInteraction3f()
    si.p = [0, 0, 0]
    si.wi = wi
    si.sh_frame = Frame3f([0, 0, 1])
    si.wavelengths = [500, 500, 500, 500]

    phi_o   = 180 * ek.pi/180
    theta_o =  40 * ek.pi/180
    wi = Vector3f([ek.sin(theta_o)*ek.cos(phi_o),
                   ek.sin(theta_o)*ek.sin(phi_o),
                   ek.cos(theta_o)])

    value = bsdf.eval(ctx, si, wi)
    value = np.array(value)[0,:,:]  # Extract Mueller matrix for one wavelength

    ref = [[ 0.10709418,  0.00101667,  0.00030897, -0.00033723],
           [-0.00213606,  0.00369295, -0.00124276,  0.00031914],
           [ 0.00045378,  0.00065644, -0.00414510,  0.00068390],
           [ 0.00049230, -0.00086907, -0.00068085, -0.00219314]]

    assert ek.allclose(ref, value)
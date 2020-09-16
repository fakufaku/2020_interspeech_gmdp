# Main implementation of the projection back and generalized minimum distortion
# principle algorithms
#
# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from .surrogate import lp_norm, lpq_norm


def projection_back(Y, X, ref_mic=0, **kwargs):
    """
    Solves the scale ambiguity according to Murata et al., 2001.
    This technique uses the steering vector value corresponding
    to the demixing matrix obtained during separation.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal

    Returns
    -------
    Y: array_like (n_frames, n_bins, n_channels)
        The projected data
    """
    n_frames, n_freq, n_chan = Y.shape

    # find a bunch of non-zero frames
    I_nz = np.argsort(np.linalg.norm(Y, axis=(1, 2)))[-n_chan:]

    # Now we only need to solve a linear system of size n_chan x n_chan
    # per frequency band
    A = Y[I_nz, :, :].transpose([1, 0, 2])
    b = X[I_nz, :].T
    c = np.linalg.solve(A, b)

    return c[None, :, :] * Y


def minimum_distortion_l2(Y, ref):
    """
    This function computes the frequency-domain filter that minimizes
    the squared error to a reference signal. This is commonly used
    to solve the scale ambiguity in BSS.

    Derivation of the projection
    ----------------------------

    The optimal filter `z` minimizes the squared error.

    .. math::

        \min E[|z^* y - x|^2]

    It should thus satsify the orthogonality condition
    and can be derived as follows

    .. math::

        0 & = E[y^*\\, (z^* y - x)]

        0 & = z^*\\, E[|y|^2] - E[y^* x]

        z^* & = \\frac{E[y^* x]}{E[|y|^2]}

        z & = \\frac{E[y x^*]}{E[|y|^2]}

    In practice, the expectations are replaced by the sample
    mean.

    Parameters
    ----------
    Y: array_like (n_frames, n_bins, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_bins)
        The reference signal
    """

    num = np.sum(np.conj(ref[:, :, None]) * Y, axis=0)
    denom = np.sum(np.abs(Y) ** 2, axis=0)
    c = num / np.maximum(1e-15, denom)

    return np.conj(c[None, :, :]) * Y


def minimum_distortion(
    Y, ref, p=None, q=None, rtol=1e-2, max_iter=100,
):
    """
    This function computes the frequency-domain filter that minimizes the sum
    of errors to a reference signal. This is a sparse version of the projection
    back that is commonly used to solve the scale ambiguity in BSS.

    Derivation of the projection
    ----------------------------

    The optimal filter `z` minimizes the expected absolute deviation.

    .. math::

        \min E[|z^* y - x|]

    The optimization is done via the MM algorithm (i.e. IRLS).

    Parameters
    ----------
    Y: array_like (n_frames, n_freq, n_channels)
        The STFT data to project back on the reference signal
    ref: array_like (n_frames, n_freq)
        The reference signal
    p: float (0 < p <= 2)
        The norm to use to measure distortion
    q: float (0 < p <= q <= 2)
        The other exponent when using a mixed norm to measure distortion
    max_iter: int, optional
        Maximum number of iterations
    rtol: float, optional
        Stop the optimization when the algorithm makes less than rtol relative progress

    Returns
    -------
    The projected signal

    The number of iterations
    """

    # by default we do the regular minimum distortion
    if p is None or (p == 2.0 and (q is None or p == q)):
        return minimum_distortion_l2(Y, ref), 1

    n_frames, n_freq, n_channels = Y.shape

    c = np.ones(Y.shape, dtype=Y.dtype)

    eps = 1e-15

    prev_c = None

    epoch = 0
    while epoch < max_iter:

        epoch += 1

        # the current error
        error = ref[:, :, None] - c * Y
        if q is None or p == q:
            weights = lp_norm(error, p=p)
        else:
            weights = lpq_norm(error, p=p, q=q, axis=1)

        # minimize
        num = np.sum(ref[:, :, None] * np.conj(Y) * weights, axis=0)
        denom = np.sum(np.abs(Y) ** 2 * weights, axis=0)
        c = num / np.maximum(eps, denom)

        # condition for termination
        if prev_c is None:
            prev_c = c
            continue

        # relative step length
        delta = np.linalg.norm(c - prev_c) / np.linalg.norm(prev_c)
        prev_c = c
        if delta < rtol:
            break

    return c[None, :, :] * Y, epoch

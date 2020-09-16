# This file contains code to run a sample separation and listen to the output
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
import argparse
import json
import time
from pathlib import Path

import numpy as np
from mir_eval.separation import bss_eval_sources
from scipy.io import wavfile

import bss_scale
import pyroomacoustics as pra
from dereverb_separation import ilrma_t, kagami
from metrics import si_bss_eval
from pyroomacoustics.transform import stft


def auxiva_ilrma_t(X, n_iter=20, proj_back=True, auxiva_n_iter=30, **kwargs):

    Y, W = pra.bss.auxiva(X, n_iter=auxiva_n_iter, return_filters=True, proj_back=False)

    Y = ilrma_t(Y, n_iter=n_iter, proj_back=proj_back, **kwargs)

    if proj_back:
        A = np.linalg.inv(W)
        Y = A[None, :, 0, :] * Y

    return Y


algorithms = {
    "auxiva": pra.bss.auxiva,
    "ilrma": pra.bss.ilrma,
    "sparseauxiva": pra.bss.sparseauxiva,
    "fastmnmf": pra.bss.fastmnmf,
    "ilrma_t": ilrma_t,
    "kagami": kagami,
    "auxiva_ilrma_t": auxiva_ilrma_t,
}

dereverb_algos = ["ilrma_t", "kagami", "auxiva_ilrma_t"]

DATA_DIR = Path("bss_speech_dataset/data")
DATA_META = DATA_DIR / "metadata.json"
REF_MIC = 0
RTOL = 1e-5

if __name__ == "__main__":

    np.random.seed(0)

    with open(DATA_META, "r") as f:
        metadata = json.load(f)

    mics_choices = [int(key[0]) for key in metadata]
    algo_choices = list(algorithms.keys())

    parser = argparse.ArgumentParser(description="Separation example")
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        choices=algo_choices,
        default=algo_choices[0],
        help="BSS algorithm",
    )
    parser.add_argument(
        "-m",
        "--mics",
        type=int,
        choices=mics_choices,
        default=mics_choices[0],
        help="Number of channels",
    )
    parser.add_argument(
        "-p", type=float, help="Outer norm",
    )
    parser.add_argument(
        "-q", type=float, help="Inner norm",
    )
    parser.add_argument("-r", "--room", default=0, type=int, help="Room number")
    parser.add_argument("-b", "--block", default=4096, type=int, help="STFT frame size")
    parser.add_argument("--snr", default=40, type=float, help="Signal-to-Noise Ratio")
    args = parser.parse_args()

    rooms = metadata[f"{args.mics}_channels"]

    assert args.room >= 0 or args.room < len(
        rooms
    ), f"Room must be between 0 and {len(rooms) - 1}"

    t60 = rooms[args.room]["room_params"]["t60"]
    print(f"Using room {args.room} with T60={t60:.3f}")

    # choose and read the audio files

    # the mixtures
    fn_mix = DATA_DIR / rooms[args.room]["mix_filename"]
    fs, mix = wavfile.read(fn_mix)
    mix = mix.astype(np.float64) / 2 ** 15

    # add some noise
    sigma_src = np.std(mix)
    sigma_n = sigma_src * 10 ** (-args.snr / 20)
    mix += np.random.randn(*mix.shape) * sigma_n
    print("SNR:", 10 * np.log10(sigma_src ** 2 / sigma_n ** 2))

    # the reference
    if args.algo in dereverb_algos:
        # for dereverberation algorithms we use the anechoic reference signal
        fn_ref = DATA_DIR / rooms[args.room]["anechoic_filenames"][REF_MIC]
    else:
        fn_ref = DATA_DIR / rooms[args.room]["src_filenames"][REF_MIC]

    fs, ref = wavfile.read(fn_ref)
    ref = ref.astype(np.float64) / 2 ** 15

    # STFT parameters
    hop = args.block // 2
    win_a = pra.hamming(args.block)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

    # STFT
    X = stft.analysis(mix, args.block, hop, win=win_a)

    t1 = time.perf_counter()

    # Separation
    if args.algo == "fastmnmf":
        Y = algorithms[args.algo](X, n_iter=30)
    elif args.algo in dereverb_algos:
        if args.p is None:
            Y = algorithms[args.algo](
                X,
                n_iter=15 * args.mics,
                n_taps=3,
                n_delays=2,
                n_components=1,
                proj_back=True,
            )
        else:
            Y = algorithms[args.algo](
                X,
                n_iter=15 * args.mics,
                n_taps=3,
                n_delays=2,
                n_components=1,
                proj_back=False,
            )
    else:
        Y = algorithms[args.algo](X, n_iter=30, proj_back=False)

    t2 = time.perf_counter()

    print(f"Separation time: {t2 - t1:.3f} s")

    # Projection back

    if args.p is not None:
        Y, n_iter = bss_scale.minimum_distortion(
            Y, X[:, :, REF_MIC], p=args.p, q=args.q
        )
        print("minimum distortion iterations:", n_iter)
    elif args.algo not in dereverb_algos:
        Y = bss_scale.projection_back(Y, X[:, :, REF_MIC])

    t3 = time.perf_counter()

    print(f"Proj. back time: {t3 - t2:.3f} s")

    # iSTFT
    y = stft.synthesis(Y, args.block, hop, win=win_s)
    y = y[args.block - hop :]
    if y.ndim == 1:
        y = y[:, None]

    # Evaluate
    m = np.minimum(ref.shape[0], y.shape[0])

    t4 = time.perf_counter()

    if args.algo in dereverb_algos:
        # conventional metric
        sdr, sir, sar, perm = bss_eval_sources(ref[:m, :].T, y[:m, :].T)
    else:
        # scale invaliant metric
        sdr, sir, sar, perm = si_bss_eval(ref[:m, :], y[:m, :])

    t5 = time.perf_counter()

    print(f"Eval. back time: {t5 - t4:.3f} s")

    wavfile.write("example_mix.wav", fs, mix)
    wavfile.write("example_ref.wav", fs, ref[:m, :])
    wavfile.write("example_output.wav", fs, y[:m, :])

    # Reorder the signals
    print("SDR:", sdr)
    print("SIR:", sir)

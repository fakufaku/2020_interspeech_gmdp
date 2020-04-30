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

algorithms = {
    "auxiva": pra.bss.auxiva,
    "ilrma": pra.bss.ilrma,
    "sparseauxiva": pra.bss.sparseauxiva,
    "fastmnmf": pra.bss.fastmnmf,
    "ilrma_t": ilrma_t,
    "kagami": kagami,
}

dereverb_algos = ["ilrma_t", "kagami"]

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
    parser.add_argument("--snr", default=30, type=float, help="Signal-to-Noise Ratio")
    args = parser.parse_args()

    rooms = metadata[f"{args.mics}_channels"]

    assert args.room >= 0 or args.room < len(
        rooms
    ), f"Room must be between 0 and {len(rooms) - 1}"

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
                X, n_iter=30, n_taps=6, n_delays=2, proj_back=True
            )
        else:
            Y = algorithms[args.algo](
                X, n_iter=30, n_taps=6, n_delays=2, proj_back=False
            )
    else:
        Y = algorithms[args.algo](X, n_iter=30, proj_back=False)

    t2 = time.perf_counter()

    # Projection back

    if args.p is not None:
        Y, n_iter = bss_scale.minimum_distortion(
            Y, X[:, :, REF_MIC], p=args.p, q=args.q
        )
    elif args.algo not in dereverb_algos:
        Y = bss_scale.projection_back(Y, X[:, :, REF_MIC])

    t3 = time.perf_counter()

    # iSTFT
    y = stft.synthesis(Y, args.block, hop, win=win_s)
    y = y[args.block - hop :]
    if y.ndim == 1:
        y = y[:, None]

    # Evaluate
    m = np.minimum(ref.shape[0], y.shape[0])

    # scale invaliant metric
    # sdr, sir, sar, perm = si_bss_eval(ref[:m, :], y[:m, :])

    # conventional metric
    sdr, sir, sar, perm = bss_eval_sources(ref[:m, :].T, y[:m, :].T)

    wavfile.write("example_mix.wav", fs, mix)
    wavfile.write("example_ref.wav", fs, ref[:m, :])
    wavfile.write("example_output.wav", fs, y[:m, :])

    # Reorder the signals
    print("SDR:", sdr)
    print("SIR:", sir)
    print(f"Separation time: {t2 - t1:.3f} s")
    print(f"Proj. back time: {t3 - t2:.3f} s")

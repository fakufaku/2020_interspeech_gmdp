import argparse
import json
import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra
from pyroomacoustics.transform import stft

from metrics import si_bss_eval
import bss_scale

algorithms = {
    "auxiva": pra.bss.auxiva,
    "ilrma": pra.bss.ilrma,
    "sparseauxiva": pra.bss.sparseauxiva,
    "fastmnmf": pra.bss.fastmnmf,
}

DATA_META = "data/metadata.json"
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
    args = parser.parse_args()

    rooms = metadata[f"{args.mics}_channels"]

    assert args.room >= 0 or args.room < len(
        rooms
    ), f"Room must be between 0 and {len(rooms) - 1}"

    # choose and read the audio files

    # the mixtures
    fn_mix = rooms[args.room]["mix_filename"]
    fs, mix = wavfile.read(fn_mix)
    mix = mix.astype(np.float64) / 2 ** 15

    # the reference
    fn_ref = rooms[args.room]["src_filenames"][REF_MIC]
    fs, ref = wavfile.read(fn_ref)
    ref = ref.astype(np.float64) / 2 ** 15

    # STFT parameters
    hop = args.block // 4
    win_a = pra.hamming(args.block)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)

    # STFT
    X = stft.analysis(mix, args.block, hop, win=win_a)

    # Separation
    if args.algo != "fastmnmf":
        Y = algorithms[args.algo](X, n_iter=30, proj_back=False)
    else:
        Y = algorithms[args.algo](X, n_iter=30)

    # Projection back

    if args.p is None:
        Y = bss_scale.projection_back(Y, X[:, :, REF_MIC])
    else:
        Y = bss_scale.minimum_distortion(Y, X[:, :, REF_MIC], p=args.p, q=args.q)

    # iSTFT
    y = stft.synthesis(Y, args.block, hop, win=win_s)
    y = y[args.block - hop :]
    if y.ndim == 1:
        y = y[:, None]

    # Evaluate
    m = np.minimum(ref.shape[0], y.shape[0])
    sdr, sir, sar, perm = si_bss_eval(ref[:m, :], y[:m, :])

    # Reorder the signals
    print(sdr)

import json
from pathlib import Path

import numpy as np

import bss_scale
import pyroomacoustics as pra
from helper import load_audio, save_audio
from metrics import si_bss_eval
from pyroomacoustics.transform import stft

try:
    import mkl

    mkl_available = True
except ImportError:
    mkl_available = False

bss_algorithms = {
    "auxiva": pra.bss.auxiva,
    "ilrma": pra.bss.ilrma,
    # "fastmnmf": pra.bss.fastmnmf,
}


def reconstruct_evaluate(ref, Y, nfft, hop, win=None):
    """
    Apply iSTFT and then evaluate the SI-BSS_EVAL metrics
    """
    # iSTFT
    y = stft.synthesis(Y, nfft, hop, win=win)
    y = y[nfft - hop :]
    if y.ndim == 1:
        y = y[:, None]

    # SI-BSS_EVAL
    m = np.minimum(ref.shape[0], y.shape[0])
    sdr, sir, sar, perm = si_bss_eval(ref[:m, :], y[:m, :])

    return y[:, perm], sdr, sir, sar


def process(args, config):

    n_channels, room_id, bss_algo = args

    if mkl_available:
        mkl.set_num_threads_local(1)

    ref_mic = config["ref_mic"]
    metadata_fn = Path(config["metadata_fn"])
    dataset_dir = Path(*metadata_fn.absolute().parts[:-2])

    with open(config["metadata_fn"], "r") as f:
        metadata = json.load(f)

    rooms = metadata[f"{n_channels}_channels"]

    # the mixtures
    fn_mix = dataset_dir / rooms[room_id]["mix_filename"]
    fs, mix = load_audio(fn_mix)

    # the reference
    fn_ref = dataset_dir / rooms[room_id]["src_filenames"][ref_mic]
    fs, ref = load_audio(fn_ref)

    # STFT parameters
    nfft = config["stft"]["nfft"]
    hop = config["stft"]["hop"]
    if config["stft"]["window"] == "hamming":
        win_a = pra.hamming(nfft)
        win_s = pra.transform.stft.compute_synthesis_window(win_a, hop)
    else:
        raise ValueError("Window not implemented")

    # STFT
    X = stft.analysis(mix, nfft, hop, win=win_a)

    # Separation
    if bss_algo != "fastmnmf":
        Y = bss_algorithms[bss_algo](X, n_iter=10 * n_channels, proj_back=False)
    else:
        Y = bss_algorithms[bss_algo](X, n_iter=10 * n_channels)

    results = []
    t = {
        "room_id": room_id,
        "n_channels": n_channels,
        "bss_algo": bss_algo,
        "proj_algo": None,
        "sdr": None,
        "sir": None,
        "p": None,
        "q": None,
        "n_iter": 1,
    }

    # Evaluation of raw signal
    t["proj_algo"] = "None"
    y, sdr, sir, _ = reconstruct_evaluate(ref, Y, nfft, hop, win=win_s)
    t["sdr"], t["sir"] = sdr.tolist(), sir.tolist()
    results.append(t.copy())

    # projection back
    t["proj_algo"] = "projection_back"
    Z = bss_scale.projection_back(Y, X[:, :, ref_mic])
    y, sdr, sir, _ = reconstruct_evaluate(ref, Z, nfft, hop, win=win_s)
    t["sdr"], t["sir"] = sdr.tolist(), sir.tolist()
    results.append(t.copy())

    # minimum distortion
    lo, hi, step = config["minimum_distortion"]["p_list"]
    kwargs = config["minimum_distortion"]["kwargs"]
    for p in np.arange(lo, hi + step, step):
        for q in np.arange(p, hi + step, step):
            t["p"], t["q"], t["proj_algo"] = (
                f"{p:.1f}",
                f"{q:.1f}",
                "minimum_distortion",
            )

            Z, t["n_iter"] = bss_scale.minimum_distortion(
                Y, X[:, :, ref_mic], p=p, q=q, **kwargs
            )

            y, sdr, sir, _ = reconstruct_evaluate(
                ref, Z, nfft, hop, win=win_s
            )
            t["sdr"] = sdr.tolist()
            t["sir"] = sir.tolist()
            results.append(t.copy())

    return results

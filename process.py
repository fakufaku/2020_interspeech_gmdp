import json
import time
from pathlib import Path

import numpy as np
from mir_eval.separation import bss_eval_sources

import bss_scale
import pyroomacoustics as pra
from dereverb_separation import ilrma_t, kagami
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
    "ilrma_t": ilrma_t,
}

dereverb_algos = ["ilrma_t", "kagami"]


def reconstruct_evaluate(ref, Y, nfft, hop, win=None, si_metric=True):
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
    if si_metric:
        sdr, sir, sar, perm = si_bss_eval(ref[:m, :], y[:m, :])
    else:
        sdr, sir, sar, perm = bss_eval_sources(ref[:m, :].T, y[:m, :].T)

    return y[:, perm], sdr, sir, sar


def process(args, config):

    n_channels, room_id, bss_algo = args

    # the name of the algorithm we'll use for bss
    bss_algo_name = config["bss_algorithms"][bss_algo]["name"]

    if mkl_available:
        mkl.set_num_threads_local(1)

    ref_mic = config["ref_mic"]
    metadata_fn = Path(config["metadata_fn"])
    dataset_dir = metadata_fn.parent

    with open(config["metadata_fn"], "r") as f:
        metadata = json.load(f)

    rooms = metadata[f"{n_channels}_channels"]

    # the mixtures
    fn_mix = dataset_dir / rooms[room_id]["mix_filename"]
    fs, mix = load_audio(fn_mix)

    # add the noise
    sigma_src = np.std(mix)
    sigma_n = sigma_src * 10 ** (-config["snr"] / 20)
    mix += np.random.randn(*mix.shape) * sigma_n

    # the reference
    if bss_algo_name in dereverb_algos:
        # for dereverberation algorithms we use the anechoic reference signal
        fn_ref = dataset_dir / rooms[room_id]["anechoic_filenames"][config["ref_mic"]]
    else:
        fn_ref = dataset_dir / rooms[room_id]["src_filenames"][config["ref_mic"]]
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
    bss_kwargs = config["bss_algorithms"][bss_algo]["kwargs"]
    n_iter_p_ch = config["bss_algorithms"][bss_algo]["n_iter_per_channel"]

    runtime_bss = time.perf_counter()
    if bss_algo_name == "fastmnmf":
        Y = bss_algorithms[bss_algo_name](
            X, n_iter=n_iter_p_ch * n_channels, **bss_kwargs
        )
    elif bss_algo_name in dereverb_algos:
        Y, Y_pb, runtime_pb = bss_algorithms[bss_algo_name](
            X, n_iter=n_iter_p_ch * n_channels, proj_back_both=True, **bss_kwargs
        )
        # adjust start time to remove the projection back
        runtime_bss += runtime_pb
    else:
        Y = bss_algorithms[bss_algo_name](
            X, n_iter=n_iter_p_ch * n_channels, proj_back=False, **bss_kwargs
        )
    runtime_bss = time.perf_counter() - runtime_bss

    results = [{"bss_runtime": {"bss_algo": bss_algo, "runtime": runtime_bss,}}]
    t = {
        "room_id": room_id,
        "n_channels": n_channels,
        "bss_algo": bss_algo,
        "proj_algo": None,
        "runtime": 0.0,
        "sdr": None,
        "sir": None,
        "p": None,
        "q": None,
        "n_iter": 1,
    }

    # Evaluation of raw signal
    t["proj_algo"] = "None"
    y, sdr, sir, _ = reconstruct_evaluate(
        ref, Y, nfft, hop, win=win_s, si_metric=config["si_metric"]
    )
    t["sdr"], t["sir"] = sdr.tolist(), sir.tolist()
    results.append(t.copy())

    # projection back
    t["proj_algo"] = "projection_back"
    if bss_algo in dereverb_algos:
        Z = Y_pb
    else:
        runtime_pb = time.perf_counter()
        Z = bss_scale.projection_back(Y, X[:, :, ref_mic])
        runtime_pb = time.perf_counter() - runtime_pb
    y, sdr, sir, _ = reconstruct_evaluate(
        ref, Z, nfft, hop, win=win_s, si_metric=config["si_metric"]
    )
    t["sdr"], t["sir"] = sdr.tolist(), sir.tolist()
    t["runtime"] = runtime_pb
    results.append(t.copy())

    # minimum distortion
    lo, hi, n_steps = config["minimum_distortion"]["p_list"]
    p_vals = np.linspace(lo, hi, n_steps)
    kwargs = config["minimum_distortion"]["kwargs"]
    for ip, p in enumerate(p_vals):
        for q in p_vals[ip:]:
            t["p"], t["q"], t["proj_algo"] = (
                f"{p:.1f}",
                f"{q:.1f}",
                "minimum_distortion",
            )

            runtime_md = time.perf_counter()
            Z, t["n_iter"] = bss_scale.minimum_distortion(
                Y, X[:, :, ref_mic], p=p, q=q, **kwargs
            )
            runtime_md = time.perf_counter() - runtime_md

            y, sdr, sir, _ = reconstruct_evaluate(
                ref, Z, nfft, hop, win=win_s, si_metric=config["si_metric"]
            )
            t["sdr"] = sdr.tolist()
            t["sir"] = sir.tolist()
            t["runtime"] = runtime_md
            results.append(t.copy())

    return results

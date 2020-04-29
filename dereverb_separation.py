import numpy as np
import scipy as sp


def condition_covariance(x, gamma):
    """see https://stt.msu.edu/users/mauryaas/Ashwini_JPEN.pdf (2.3)"""
    scale = gamma * np.trace(x) / x.shape[-1]
    scaled_eye = np.eye(x.shape[-1]) * scale
    return (x + scaled_eye) / (1 + gamma)


# IP法でWの推定を実施する
# x: fftMax,frameNum,micNum
# d: fftMax,frameNum,micNum
# W: fftMax,micNum,micNum
def kagami_IP_iteration(x, W, d, eps=1.0e-18):
    fftMax = np.shape(x)[0]
    frameNum = np.shape(x)[1]
    micNum = np.shape(x)[2]

    V = np.einsum("ftd,fti,ftj->fdij", 1.0 / np.maximum(d, eps), x, np.conjugate(x))
    V = V / np.float(frameNum)

    W_temp = W.copy()

    for t in range(micNum):
        WV = np.einsum("fmn,fnk->fmk", W_temp, V[:, t, :, :])
        # WV_eps=condition_covariance(WV,eps)
        invWV = np.linalg.pinv(WV)
        W_temp[:, t, :] = np.conjugate(invWV[:, :, t])
        wVw = np.einsum(
            "fm,fmn,fn->f",
            W_temp[:, t, :],
            V[:, t, :, :],
            np.conjugate(W_temp[:, t, :]),
        )
        wVw = np.sqrt(np.abs(wVw))
        wVw = np.expand_dims(wVw, axis=1)
        W_temp[:, t, :] = W_temp[:, t, :] / np.maximum(wVw, eps)

    return W_temp


# x: f,t,extendedmic
# W: f,extendedmic,extendedmic,
# d: f,t,extendedmic
def log_likelihood_kagami_by_frequency(x, W, d, eps=1.0e-18):

    # print("likelihood")
    y = np.einsum("fmn,ftn->ftm", W, x)
    likelihood = np.maximum(np.square(np.abs(y)), eps) / np.maximum(
        np.abs(d), eps
    ) + np.log(np.maximum(np.abs(d), eps))
    likelihood = np.sum(likelihood, axis=(2))
    likelihood = np.average(likelihood, axis=(1))

    W_eps = condition_covariance(W, eps)
    logDetW = -2.0 * np.log(np.abs(np.linalg.det(W_eps)))
    likelihood = likelihood + logDetW
    return likelihood


# y: f,t,sourceNum
# W: f,mic,mic,
# d: f,t,sourceNum
def log_likelihood_ilmra_t_by_frequency(y, W, d, eps=1.0e-18):

    likelihood = np.maximum(np.square(np.abs(y)), eps) / np.maximum(
        np.abs(d), eps
    ) + np.log(np.maximum(np.abs(d), eps))
    likelihood = np.sum(likelihood, axis=(2))
    likelihood = np.average(likelihood, axis=(1))

    W_eps = condition_covariance(W, eps)
    logDetW = -2.0 * np.log(np.abs(np.linalg.det(W_eps)))
    likelihood = likelihood + logDetW
    return likelihood


# IRLMAでbの更新を行う
# y: fftMax,frameNum,extendedMicNum
# b: fftMax,sourceNum,basis
# v: basis,sourceNum,frameNum
# v_delay: basis,sourceNum,frameNum,xTapNum
def ilrma_t_b_iteration(y, b, v, eps=1.0e-18):
    y_power = np.square(np.abs(y))

    hat_y_power = np.einsum("bst,fsb->fts", v, b)
    y_power = np.reshape(y_power, newshape=np.shape(hat_y_power))

    bunnsi = np.einsum(
        "fts,bst->fsb",
        np.maximum(y_power, eps) / np.square(np.maximum(hat_y_power, eps)),
        np.maximum(v, eps),
    )
    bunnbo = np.einsum(
        "fts,bst->fsb", 1.0 / np.maximum(hat_y_power, eps), np.maximum(v, eps)
    )
    ratio = np.sqrt(np.maximum(bunnsi, eps) / np.maximum(bunnbo, eps))
    b = b * ratio

    return b


# IRLMAでvの更新を行う
# y: fftMax,frameNum,micNum
# b: fftMax,sourceNum,basis
# v: basis,sourceNum,frameNum
def ilrma_t_v_iteration(y, b, v, eps=1.0e-18):
    y_power = np.square(np.abs(y))
    fftMax = np.shape(y_power)[0]
    frameNum = np.shape(y_power)[1]
    sourceNum = np.shape(v)[1]
    micNum = sourceNum

    hat_y_power = np.einsum("bst,fsb->fts", v, b)
    hat_y_power = np.reshape(hat_y_power, newshape=np.shape(y_power))
    sourceNum = np.shape(v)[1]

    bunnsi = np.einsum(
        "fts,fsb->bst",
        np.maximum(y_power, eps) / np.square(np.maximum(hat_y_power, eps)),
        np.maximum(b, eps),
    )
    bunnbo = np.einsum(
        "fsb,fts->bst", np.maximum(b, eps), 1.0 / np.maximum(hat_y_power, eps)
    )
    # ratio=np.sqrt(bunnsi/(bunnbo+eps))
    ratio = np.sqrt(np.maximum(bunnsi, eps) / np.maximum(bunnbo, eps))

    v = v * ratio

    return v


# 池下さんのILRMA-Tに基づく音源分離法の1iteration
# x: fftMax,frameNum,xTapNum,micNum
# b: fftMax,sourceNum,basis
# v: basis,sourceNum,frameNum
# P: fftMax,xTapNum,micNum,micNum
# G: fftMax,micNum,xTapNum-1,micNum
# W: fftMax,micNum,micNum
def ilrma_t_iteration(
    x,
    b,
    v,
    P,
    fixB=False,
    fixV=False,
    fixP=False,
    use_increase_constraint=False,
    eps=1.0e-18,
):
    # 各音源の共分散行列を再現する。
    fftMax = np.shape(x)[0]
    frameNum = np.shape(x)[1]
    xTapNum = np.shape(x)[2]
    micNum = np.shape(x)[3]
    sourceNum = np.shape(b)[1]
    basisNum = np.shape(b)[2]

    # 残響除去と分離を同時に実行する。
    y = np.einsum("fdnm,ftdn->ftm", np.conjugate(P), x)

    # 時間周波数分散
    d = np.einsum("bst,fsb->fts", v, b)
    # |Delta f|=
    num_deltaf = xTapNum

    x_hat = np.reshape(x, (fftMax, frameNum, xTapNum * micNum))

    costOrgByFreq = log_likelihood_ilmra_t_by_frequency(y, P[:, 0, :, :], d, eps)
    costOrg = np.average(costOrgByFreq)

    if fixB == False:
        b_temp = ilrma_t_b_iteration(y, b, v, eps)
        # dを更新する

        d_temp = np.einsum("bst,fsb->fts", v, b_temp)

        costTempByFreq = log_likelihood_ilmra_t_by_frequency(
            y, P[:, 0, :, :], d_temp, eps
        )

        if use_increase_constraint == True:
            for freq in range(fftMax):
                if costOrgByFreq[freq] > costTempByFreq[freq]:
                    b[freq, ...] = b_temp[freq, ...]
        else:
            b = b_temp

    # 時間周波数分散
    d = np.einsum("bst,fsb->fts", v, b)

    costBByFreq = log_likelihood_ilmra_t_by_frequency(y, P[:, 0, :, :], d, eps)
    costB = np.average(costBByFreq)

    if fixV == False:
        # y: fftMax,frameNum,micNum
        # b: fftMax,sourceNum,basis
        # v: basis,sourceNum,frameNum
        # v_delay: basis,sourceNum,frameNum,1
        v_temp = ilrma_t_v_iteration(y, b, v, eps=eps)

        d_temp = np.einsum("bst,fsb->fts", v_temp, b)

        costTempByFreq = log_likelihood_ilmra_t_by_frequency(
            y, P[:, 0, :, :], d_temp, eps
        )

        if use_increase_constraint == True:
            for freq in range(fftMax):
                if costBByFreq[freq] > costTempByFreq[freq]:
                    v[freq, ...] = v_temp[freq, ...]
        else:
            v = v_temp

    d = np.einsum("bst,fsb->fts", v, b)

    costVByFreq = log_likelihood_ilmra_t_by_frequency(y, P[:, 0, :, :], d, eps)
    costV = np.average(costVByFreq)

    # フィルタを求める。
    IP1 = True
    IP2 = False
    if fixP == False and IP1 == True:

        G_hat = np.einsum(
            "fts,ftm,ftn->ftsmn", 1.0 / np.maximum(d, eps), x_hat, np.conjugate(x_hat)
        )
        G_hat = np.average(G_hat, axis=1)  # fsmn
        # G_hat_eps=condition_covariance(G_hat,eps) #fsmn
        inv_G_hat = np.linalg.pinv(G_hat)
        # sss
        for n in range(sourceNum):
            # 分離フィルタの逆行列がPo,o^H-1
            P00_H = np.conjugate(P[:, 0, :, :])
            P00_H = np.transpose(P00_H, axes=[0, 2, 1])
            # P00_H_eps=condition_covariance(P00_H,eps)
            A = np.linalg.pinv(P00_H)
            # ステアリングベクトル
            a = A[:, :, n]  # fm
            # fsmn
            a_h_Ga = np.einsum(
                "fm,fmn,fn->f", np.conjugate(a), inv_G_hat[:, n, :micNum, :micNum], a
            )
            power = np.maximum(np.abs(a_h_Ga), eps)
            coef = np.sqrt(power)
            Ga = np.einsum("fmn,fn->fm", inv_G_hat[:, n, :, :micNum], a)
            p = np.einsum("fm,f->fm", Ga, 1.0 / np.maximum(coef, eps))

            detP = np.conjugate(np.linalg.det(P[:, 0, :, :]))  # f
            theta = -np.angle(detP)  # f
            coef = np.cos(theta) + 1.0j * np.sin(theta)
            # p=p*coef[:,np.newaxis]
            p = np.reshape(p, [fftMax, xTapNum, micNum])
            P[:, :, :, n] = p

    if fixP == False and IP2 == True:

        G_hat = np.einsum(
            "fts,ftm,ftn->ftsmn", 1.0 / np.maximum(d, eps), x_hat, np.conjugate(x_hat)
        )
        G_hat = np.average(G_hat, axis=1)  # fsmn
        # G_hat_eps=condition_covariance(G_hat,eps) #fsmn
        inv_G_hat = np.linalg.pinv(G_hat)
        V1 = inv_G_hat[:, 0, :micNum, :micNum]
        V2 = inv_G_hat[:, 1, :micNum, :micNum]
        for k in range(fftMax):
            w, vr = sp.linalg.eig(V1[k, ...], V2[k, ...])  # fm

            if np.real(w[0]) > np.real(w[1]):
                # srew
                temp1 = vr[:, 0]
                temp2 = vr[:, 1]
            else:
                #
                temp1 = vr[:, 1]
                temp2 = vr[:, 0]
            # vr
            if k == 0:
                u1 = temp1[np.newaxis, :]
                u2 = temp2[np.newaxis, :]
            else:
                u1 = np.concatenate((u1, temp1[np.newaxis, :]), axis=0)
                u2 = np.concatenate((u2, temp2[np.newaxis, :]), axis=0)
        u = np.concatenate((u1[:, np.newaxis, :], u2[:, np.newaxis, :]), axis=1)
        # sss
        print(np.shape(u))
        for n in range(sourceNum):
            # fsmn
            V = inv_G_hat[:, n, :micNum, :micNum]
            a = u[:, n, :]
            a_h_Ga = np.einsum("fm,fmn,fn->f", np.conjugate(a), V, a)
            power = np.maximum(np.abs(a_h_Ga), eps)
            coef = np.sqrt(power)
            Ga = np.einsum("fmn,fn->fm", inv_G_hat[:, n, :, :micNum], a)
            p = np.einsum("fm,f->fm", Ga, 1.0 / np.maximum(coef, eps))

            detP = np.conjugate(np.linalg.det(P[:, 0, :, :]))  # f
            theta = -np.angle(detP)  # f
            # coef=np.cos(theta)+1.0j*np.sin(theta)
            # p=p*coef[:,np.newaxis]
            p = np.reshape(p, [fftMax, xTapNum, micNum])
            P[:, :, :, n] = p

    # 残響除去と分離を同時に実行する。

    y = np.einsum("fdnm,ftdn->ftm", np.conjugate(P), x)

    # Projection Back
    P00_H = np.conjugate(P[:, 0, :, :])
    P00_H = np.transpose(P00_H, axes=[0, 2, 1])
    # P00_H_eps=condition_covariance(P00_H,eps)
    A = np.linalg.pinv(P00_H)
    y_pb = np.einsum("fts,fms->fstm", y, A)

    costPByFreq = log_likelihood_ilmra_t_by_frequency(y, P[:, 0, :, :], d, eps)
    costP = np.average(costPByFreq)

    return (y, y_pb, b, v, P, costOrg, costB, costV, costP)


# KagamiICASSP2018を実装する
# x: fftMax,frameNum,xTapNum,micNum
# W: fftMax,micNum,micNum
# G: fftMax,micNum,xTapNum-1,micNum
# b: fftMax,sourceNum,basis
# v: basis,sourceNum,frameNum
def kagami_iteration(
    x,
    W,
    G,
    b,
    v,
    eps=1.0e-18,
    fixG=False,
    fixB=False,
    fixV=False,
    use_increase_constraint=True,
):
    # 各音源の共分散行列を再現する。
    fftMax = np.shape(x)[0]
    frameNum = np.shape(x)[1]
    xTapNum = np.shape(x)[2]
    micNum = np.shape(x)[3]
    sourceNum = np.shape(b)[1]
    basisNum = np.shape(b)[2]

    # 時間周波数分散
    d = np.einsum("bst,fsb->fts", v, b)
    # Dereverb信号の抽出
    # x=x-Gx: Gは転置しない
    x_dereverb = x[..., 0, :] - np.einsum("fmdn,ftdn->ftm", G, x[..., 1:, :])

    y = np.einsum("fij,ftj->fti", W, x_dereverb)

    costOrgByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W, d, eps)
    # IRLMAによる音源モデルの推定
    costOrg = np.average(costOrgByFreq)

    if fixB == False:

        b_temp = ilrma_t_b_iteration(y, b, v, eps)

        # dを更新する

        d_temp = np.einsum("bst,fsb->fts", v, b_temp)

        costTempByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W, d_temp, eps)

        if use_increase_constraint == True:
            for freq in range(fftMax):
                if costOrgByFreq[freq] > costTempByFreq[freq]:
                    b[freq, ...] = b_temp[freq, ...]
        else:
            b = b_temp

    # 時間周波数分散
    d = np.einsum("bst,fsb->fts", v, b)

    costBByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W, d, eps)
    costB = np.average(costBByFreq)

    if fixV == False:
        v_temp = ilrma_t_v_iteration(y, b, v, eps=eps)

        d_temp = np.einsum("bst,fsb->fts", v_temp, b)

        costTempByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W, d_temp, eps)

        if use_increase_constraint == True:
            for freq in range(fftMax):
                if costBByFreq[freq] > costTempByFreq[freq]:
                    v[freq, ...] = v_temp[freq, ...]
        else:
            v = v_temp

    d = np.einsum("bst,fsb->fts", v, b)

    costVByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W, d, eps)

    costV = np.average(costVByFreq)

    # 分離フィルタの更新
    # IP法によるWの更新
    W_temp = kagami_IP_iteration(x_dereverb, W, d, eps)

    costTempByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W_temp, d, eps)
    if use_increase_constraint == True:
        for freq in range(fftMax):
            if costVByFreq[freq] > costTempByFreq[freq]:
                W[freq, ...] = W_temp[freq, ...]
    else:
        W = W_temp

    costWByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W, d, eps)
    costW = np.average(costWByFreq)

    # 残響除去フィルタを更新する
    if fixG == False:
        # invRx: ftmn
        invRx = np.einsum(
            "fnm,ftn,fnk->ftmk", np.conjugate(W), 1.0 / np.maximum(np.abs(d), eps), W
        )
        # ftdn
        hTapNum = (xTapNum - 1) * micNum
        x_delay = x[..., 1:, :]
        x_delay = np.reshape(x_delay, [fftMax, frameNum, hTapNum])

        # Gを更新する
        XX = np.transpose(
            np.einsum("ftd,ftk->ftdk", x_delay, np.conjugate(x_delay)), [0, 1, 3, 2]
        )

        P = np.zeros(
            shape=[fftMax, hTapNum * micNum, hTapNum * micNum], dtype=np.complex128
        )

        for k in range(fftMax):
            for f in range(frameNum):
                P[k, :, :] = P[k, :, :] + np.kron(XX[k, f, :, :], invRx[k, f, :, :])
        # P_eps = condition_covariance(P, eps)
        # P_eps=P
        invP_eps = np.linalg.pinv(P)

        U = np.einsum("ftmn,ftn,ftk->fmk", invRx, x[..., 0, :], np.conjugate(x_delay))
        U = np.reshape(np.transpose(U, [0, 2, 1]), [-1, micNum * hTapNum])
        vec_G = np.einsum("fmk,fk->fm", invP_eps, U)
        G_temp = np.reshape(vec_G, [-1, hTapNum, micNum])
        G_temp = np.transpose(G_temp, [0, 2, 1])
        G_temp = np.reshape(G_temp, [fftMax, micNum, xTapNum - 1, micNum])

        x_dereverb_temp = x[..., 0, :] - np.einsum(
            "fmdn,ftdn->ftm", G_temp, x[..., 1:, :]
        )

        costTempByFreq = log_likelihood_kagami_by_frequency(x_dereverb_temp, W, d, eps)
        if use_increase_constraint == True:
            for freq in range(fftMax):
                if costWByFreq[freq] > costTempByFreq[freq]:
                    G[freq, ...] = G_temp[freq, ...]
        else:
            G = G_temp

    x_dereverb = x[..., 0, :] - np.einsum("fmdn,ftdn->ftm", G, x[..., 1:, :])

    y = np.einsum("fij,ftj->fti", W, x_dereverb)
    invW = np.linalg.pinv(W)
    y_pb = np.einsum("fms,fts->fstm", invW, y)
    costGByFreq = log_likelihood_kagami_by_frequency(x_dereverb, W, d, eps)
    # IRLMAによる音源モデルの推定
    costG = np.average(costGByFreq)

    return (y, y_pb, W, G, b, v, costB, costV, costW, costG)


# delayLineを取得する
# x: freq,time,mic
# x_delay:freq,time,tapNum=L*mic
def get_delay_line(x, D, L):
    freq_num = np.shape(x)[0]
    mic_num = np.shape(x)[2]
    frameNum = np.shape(x)[1]
    x_delay = np.zeros(
        shape=[freq_num, frameNum, (L + 1), mic_num], dtype=np.complex128
    )
    # freq,time,mic
    # print(np.shape(x_delay))
    # print(np.shape(x))
    # freq,time,mic
    x_delay[:, :, 0, :] = x
    for t in range(frameNum):
        for d in range(L):
            t2 = t - d - D
            if t2 >= 0:
                x_delay[:, t, d + 1, :] = x[:, t2, :]
    x_delay = np.reshape(x_delay, [freq_num, frameNum, (L + 1) * mic_num])
    return x_delay


# x: freq,frame,mic
def ilrma_t_dereverb_separation(
    x, iter_num=20, nmf_basis_num=2, tap_num=3, delay_num=1, eps=1.0e-18
):
    x_abs = np.abs(x)

    frameNum = np.shape(x)[1]
    fftMax = np.shape(x)[0]
    channels = np.shape(x)[2]
    source_num = channels

    # ここは入力だから必要ない。
    x_delay = get_delay_line(x, delay_num, tap_num)
    # x_delay: freq_num,frameNum,(L+1)*mic_num
    x_delay = np.reshape(x_delay, [fftMax, frameNum, tap_num + 1, channels])

    weight = np.random.uniform(size=fftMax * source_num * channels * frameNum)
    weight = np.reshape(weight, [fftMax, source_num, channels, frameNum])

    x_abs = np.abs(x)

    # v=np.average(np.reshape(v,[fftMax,L,channels,frameNum]),axis=2)
    # print(np.shape(v))
    v = np.einsum("ftm,fsmt->fst", np.square(x_abs), weight)

    v = np.reshape(v, [fftMax, source_num, frameNum])

    weight = np.random.uniform(size=fftMax * source_num * nmf_basis_num)
    weight = np.reshape(weight, [fftMax, source_num, nmf_basis_num])

    v = np.einsum("fst,fsb->bst", v, weight)
    v_ave = np.mean(v, axis=2, keepdims=True)
    v = v / (v_ave + 1.0e-14)
    v = np.abs(v)

    b = np.ones(shape=(fftMax, source_num, nmf_basis_num))

    W = np.zeros(shape=(fftMax, channels, (tap_num + 1) * channels), dtype=np.complex)
    W[:, :, :channels] = (
        W[:, :, :channels] + np.eye(channels, dtype=np.complex)[None, ...]
    )
    W = np.reshape(W, (fftMax, channels, (tap_num + 1), channels))

    W_org = W.copy()
    # Kagami用
    # W=W_org[...,0,:]
    # G=W_org[...,1:,:]

    # ILRMA-T用
    P = W_org
    P = np.conjugate(P)
    P = np.transpose(P, axes=[0, 2, 1, 3])
    P = np.transpose(P, axes=[0, 1, 3, 2])

    for iter in range(iter_num):
        y, y_pb, b, v, P, costOrg, costB, costV, costP = ilrma_t_iteration(
            x_delay,
            b,
            v,
            P,
            fixB=False,
            fixV=False,
            fixP=False,
            use_increase_constraint=False,
            eps=eps,
        )

        # print(iter, costB, costV, costP)
    return (y, y_pb)


# x: freq,frame,mic
def kagami_dereverb_separation(
    x, iter_num=20, nmf_basis_num=2, tap_num=3, delay_num=1, eps=1.0e-18
):
    x_abs = np.abs(x)

    frameNum = np.shape(x)[1]
    fftMax = np.shape(x)[0]
    channels = np.shape(x)[2]
    source_num = channels

    # ここは入力だから必要ない。
    x_delay = get_delay_line(x, delay_num, tap_num)
    # x_delay: freq_num,frameNum,(L+1)*mic_num
    x_delay = np.reshape(x_delay, [fftMax, frameNum, tap_num + 1, channels])

    weight = np.random.uniform(size=fftMax * source_num * channels * frameNum)
    weight = np.reshape(weight, [fftMax, source_num, channels, frameNum])

    x_abs = np.abs(x)

    # v=np.average(np.reshape(v,[fftMax,L,channels,frameNum]),axis=2)
    # print(np.shape(v))
    v = np.einsum("ftm,fsmt->fst", np.square(x_abs), weight)

    v = np.reshape(v, [fftMax, source_num, frameNum])

    weight = np.random.uniform(size=fftMax * source_num * nmf_basis_num)
    weight = np.reshape(weight, [fftMax, source_num, nmf_basis_num])

    v = np.einsum("fst,fsb->bst", v, weight)
    v_ave = np.mean(v, axis=2, keepdims=True)
    v = v / (v_ave + 1.0e-14)
    v = np.abs(v)

    b = np.ones(shape=(fftMax, source_num, nmf_basis_num))

    W = np.zeros(shape=(fftMax, channels, (tap_num + 1) * channels), dtype=np.complex)
    W[:, :, :channels] = (
        W[:, :, :channels] + np.eye(channels, dtype=np.complex)[None, ...]
    )
    W = np.reshape(W, (fftMax, channels, (tap_num + 1), channels))

    W_org = W.copy()
    # Kagami用
    W = W_org[..., 0, :]
    G = W_org[..., 1:, :]

    for iter in range(iter_num):
        y, y_pb, W, G, b, v, costB, costV, costW, costG = kagami_iteration(
            x_delay,
            W,
            G,
            b,
            v,
            eps=eps,
            fixG=False,
            fixB=False,
            fixV=False,
            use_increase_constraint=False,
        )

        # print(iter, costB, costV, costW, costG)
    return (y, y_pb)


def dereverb_separate(
    X,
    n_iter=20,
    proj_back=True,
    n_components=2,
    n_taps=3,
    n_delays=1,
    algorithm="ilrma_t",
):
    """
    Performs joint dereverberation and separation of the input signal.

    Parameters
    ----------
    X: array_like, shape: (n_frames, n_frequencies, n_channels)
        The input spectrogram
    n_iter: int
        The number of iterations to run the algorithm
    proj_back: bool
        If set to True, performs projection back to adjust the scale
    n_components: int
        The number of basis functions to use in the non-negative matrix factorization
    n_taps: int
        The number of taps to use for dereverberation
    n_delays: int
        ??? (but don't set to 0!)
    algorithm: str
        "ilrma_t" or "kagami"
    """

    # reshape input
    X = X.transpose([1, 0, 2]).copy()

    if algorithm == "ilrma_t":
        Y, Y_pb = ilrma_t_dereverb_separation(
            X,
            iter_num=n_iter,
            nmf_basis_num=n_components,
            tap_num=n_taps,
            delay_num=n_delays,
            eps=1.0e-18,
        )
    elif algorithm == "kagami":
        Y, Y_pb = kagami_dereverb_separation(
            X,
            iter_num=n_iter,
            nmf_basis_num=n_components,
            tap_num=n_taps,
            delay_num=n_delays,
            eps=1.0e-18,
        )
    else:
        raise ValueError(f"Invalide algorithm {algorithm}")

    # Y_PB: shape (n_freq, n_src, n_frames, n_chan)
    # Y: shape (n_freq, n_frames, n_src)

    if proj_back:
        return Y_pb[:, :, :, 0].transpose([2, 0, 1]).copy()
    else:
        return Y.transpose([1, 0, 2]).copy()


def ilrma_t(X, **kwargs):
    return dereverb_separate(X, algorithm="ilrma_t", **kwargs)


def kagami(X, **kwargs):
    return dereverb_separate(X, algorithm="kagami", **kwargs)

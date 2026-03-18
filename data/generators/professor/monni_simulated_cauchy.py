import numpy as np
import torch


def simulate_data_cauchy(
    n=100,        # number of samples (both classes)
    p1=2,         # unique features to class -1
    p2=2,         # unique features to class +1
    pc=3,         # common predictors
    pn=5,         # pure noise predictors
    snr=2.0,      # target location-separation scale (Gaussian-style SNR analogue)
    sigma=1.0,    # scale factor for Cauchy noise
    seed=123141,  # seed
    filex=None,   # file to write X (optional)
    filey=None,   # file to write y (optional)
    filem=None,   # file to write means (optional)
    cauchy_clip=None,  # optional winsorization bound, e.g. 25.0
):
    """
    Cauchy variant of professor data generator with same feature layout logic.

    Important:
    - Cauchy has infinite variance, so `snr` is not a strict variance-based SNR.
    - Here `snr` still controls class-mean separation scale exactly like the Gaussian generator.
    """
    np.random.seed(seed)

    p = p1 + p2 + pc + pn

    d_shared = np.ones(pc)  # opposite sign
    d_p1 = np.ones(p1)      # only class -1
    d_p2 = np.ones(p2)      # only class +1

    mu_minus = np.concatenate(
        [
            d_shared,      # shared
            d_p1,          # unique to -1
            np.zeros(p2),  # unique to +1
            np.zeros(pn),  # noise
        ]
    )

    mu_plus = np.concatenate(
        [
            -d_shared,     # shared opposite
            np.zeros(p1),
            d_p2,
            np.zeros(pn),
        ]
    )

    # Keep the same mean-separation scaling rule as the Gaussian generator.
    diff = mu_plus - mu_minus
    current_norm2 = np.dot(diff, diff)
    scale = np.sqrt(snr * sigma**2 / current_norm2)
    mu_minus *= scale
    mu_plus *= scale

    n1 = n // 2
    n2 = n - n1

    # Heavy-tailed draws (location + scale * standard Cauchy).
    X_minus = mu_minus + sigma * np.random.standard_cauchy(size=(n1, p))
    X_plus = mu_plus + sigma * np.random.standard_cauchy(size=(n2, p))

    if cauchy_clip is not None:
        clip_val = float(cauchy_clip)
        X_minus = np.clip(X_minus, -clip_val, clip_val)
        X_plus = np.clip(X_plus, -clip_val, clip_val)

    # Defensive cleanup in case of extreme numeric overflow.
    X_minus = np.nan_to_num(X_minus, nan=0.0, posinf=1e6, neginf=-1e6)
    X_plus = np.nan_to_num(X_plus, nan=0.0, posinf=1e6, neginf=-1e6)

    X = np.vstack([X_minus, X_plus])
    y = np.concatenate([-np.ones(n1), np.ones(n2)])

    if filey is not None:
        np.savetxt(filey, y, delimiter="\t")
    if filex is not None:
        np.savetxt(filex, X, delimiter="\t")
    if filem is not None:
        mu_matrix = np.vstack([mu_minus, mu_plus])
        np.savetxt(filem, mu_matrix, delimiter="\t")

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    mu_minus_t = torch.from_numpy(mu_minus).float()
    mu_plus_t = torch.from_numpy(mu_plus).float()
    return X_t, y_t, mu_minus_t, mu_plus_t


def generate_professor_split_cauchy(
    iteration=0,
    iter_count=10,
    n_train=400,
    n_test=400,
    n_val=400,
    p1=4,
    p2=4,
    pc=2,
    pn=10,
    snr=2.0,
    sigma=1.0,
    startseed=57474,
    include_validation=False,
    output_prefix=None,
    cauchy_clip=None,
):
    """
    Single split with professor seed logic, but Cauchy-distributed samples.

    train seed = startseed + 2*i + 1
    test seed  = startseed + 2*i + 2
    val seed   = startseed + 2*iter + 1 + i
    """
    seed_train = startseed + 2 * iteration + 1
    seed_test = startseed + 2 * iteration + 2
    seed_val = startseed + 2 * iter_count + 1 + iteration

    filex = filey = filem = None
    filext = fileyt = filemt = None
    filexv = fileyv = filemv = None

    if output_prefix is not None:
        filex = f"{output_prefix}X_{iteration + 1}.txt"
        filey = f"{output_prefix}Y_{iteration + 1}.txt"
        filem = f"{output_prefix}means_{iteration + 1}.txt"
        filext = f"{output_prefix}XT_{iteration + 1}.txt"
        fileyt = f"{output_prefix}YT_{iteration + 1}.txt"
        filemt = f"{output_prefix}meansT_{iteration + 1}.txt"
        if include_validation:
            filexv = f"{output_prefix}XV_{iteration + 1}.txt"
            fileyv = f"{output_prefix}YV_{iteration + 1}.txt"
            filemv = f"{output_prefix}meansV_{iteration + 1}.txt"

    X_train, y_train, mu_minus, mu_plus = simulate_data_cauchy(
        n=n_train,
        p1=p1,
        p2=p2,
        pc=pc,
        pn=pn,
        snr=snr,
        sigma=sigma,
        seed=seed_train,
        filex=filex,
        filey=filey,
        filem=filem,
        cauchy_clip=cauchy_clip,
    )

    X_test, y_test, _, _ = simulate_data_cauchy(
        n=n_test,
        p1=p1,
        p2=p2,
        pc=pc,
        pn=pn,
        snr=snr,
        sigma=sigma,
        seed=seed_test,
        filex=filext,
        filey=fileyt,
        filem=filemt,
        cauchy_clip=cauchy_clip,
    )

    out = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "mu_minus": mu_minus,
        "mu_plus": mu_plus,
        "seed_train": seed_train,
        "seed_test": seed_test,
        # Keep this field name for compatibility with existing tests/plots.
        "snr_true": torch.sum((mu_plus - mu_minus) ** 2).item() / (sigma**2),
    }

    if include_validation:
        X_val, y_val, _, _ = simulate_data_cauchy(
            n=n_val,
            p1=p1,
            p2=p2,
            pc=pc,
            pn=pn,
            snr=snr,
            sigma=sigma,
            seed=seed_val,
            filex=filexv,
            filey=fileyv,
            filem=filemv,
            cauchy_clip=cauchy_clip,
        )
        out["X_val"] = X_val
        out["y_val"] = y_val
        out["seed_val"] = seed_val

    return out


def generate_professor_splits_cauchy(
    iter_count=10,
    **kwargs,
):
    """
    Convenience helper: return a list of professor-style Cauchy splits.
    """
    splits = []
    for i in range(iter_count):
        splits.append(generate_professor_split_cauchy(iteration=i, **kwargs))
    return splits

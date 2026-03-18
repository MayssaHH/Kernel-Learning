import numpy as np
import torch


def simulate_data(
    n=100,        # number of samples (both classes)
    p1=2,         # unique features to class -1
    p2=2,         # unique features to class +1
    pc=3,         # common predictors
    pn=5,         # pure noise predictors
    snr=2.0,      # SNR
    sigma=1.0,    # noise std
    seed=123141,  # seed
    filex=None,   # file to write X (optional)
    filey=None,   # file to write y (optional)
    filem=None,   # file to write means (optional)
):
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

    # --- mean difference ---
    diff = mu_plus - mu_minus
    current_norm2 = np.dot(diff, diff)
    scale = np.sqrt(snr * sigma**2 / current_norm2)
    mu_minus *= scale
    mu_plus *= scale

    # equal size
    n1 = n // 2
    n2 = n - n1

    X_minus = np.random.normal(
        loc=mu_minus,
        scale=sigma,
        size=(n1, p),
    )
    X_plus = np.random.normal(
        loc=mu_plus,
        scale=sigma,
        size=(n2, p),
    )

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


def generate_professor_split(
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
):
    """
    Single split following professor seed logic:
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

    X_train, y_train, mu_minus, mu_plus = simulate_data(
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
    )

    X_test, y_test, _, _ = simulate_data(
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
        "snr_true": torch.sum((mu_plus - mu_minus) ** 2).item() / (sigma**2),
    }

    if include_validation:
        X_val, y_val, _, _ = simulate_data(
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
        )
        out["X_val"] = X_val
        out["y_val"] = y_val
        out["seed_val"] = seed_val

    return out


def generate_professor_splits(
    iter_count=10,
    **kwargs,
):
    """
    Convenience helper: return a list of professor-style splits.
    """
    splits = []
    for i in range(iter_count):
        splits.append(generate_professor_split(iteration=i, **kwargs))
    return splits

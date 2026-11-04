# ============================================================================
# steane_noisy_trained_prune.py  
# [[7,1,3]] Steane code, QAE retrained WITH internal gate noise.
# ============================================================================

import time
import csv
import gc

import numpy as np

import jax
import jax.numpy as jnp

import optax

from qae7_noisy_common_prune import *


# ============================================================================
# Configuration
# ============================================================================

N_TRAIN_STATES = 100
N_TEST_STATES = 1000

BATCH_SIZE = 3
CHUNK = 25

BASE_SEED = 42
OUT_TAG = "noisy_trained"

LAYERS_PATH = [1, 2, 3]
L_FINAL = 3

N_INIT_ROUNDS = 12

INIT_EPOCHS = 20
MAIN_EPOCHS = 180
TOTAL_EPOCHS = (
    INIT_EPOCHS + MAIN_EPOCHS
)

LR = 0.05
WEIGHT_UPDATE_FREQ = 5

SWEEP_ONLY_FINAL_L = True

TRAIN_PARAMETERIZATION = "cr_native"
DEPLOY_PARAMETERIZATION = "fused_zyz"

PRUNE_DURING_TRAINING = False
PRUNE_AT_DEPLOYMENT = True

PR_TOL_ROT = PRUNE_TOL_ROT
PR_TOL_BLK = PRUNE_TOL_BLOCK

# Reject the complete post-training pruning operation if the accumulated
# spectral norm deviation exceeds this value.
PR_DEPLOY_MAX_DEV = 5e-3

# Keep the positive-noise sweep from the supplied script.
P2Q_TRAIN = [
    float(p)
    for p in P2Q_SWEEP
    if p > 0
]

assert TRAIN_PARAMETERIZATION == "cr_native"
assert DEPLOY_PARAMETERIZATION == "fused_zyz"
assert not PRUNE_DURING_TRAINING

assert L_FINAL == L_TARGET, (
    f"L_FINAL={L_FINAL}, but common "
    f"module has L_TARGET={L_TARGET}"
)


# ============================================================================
# History and output fields
# ============================================================================

_H_KEYS = [
    "epoch",
    "phase",
    "loss",
    "loss_std",
    "P",
    "P_std",
    "fid",
    "fid_std",
    "stab",
    "dzl",
    "dxl",
    "dyl",
    "gnorm",
]

_PROBE_KEYS = [
    "loss",
    "loss_std",
    "P",
    "P_std",
    "fid",
    "fid_std",
    "stab",
    "dzl",
    "dxl",
    "dyl",
]

_SW_KEYS = (
    "P",
    "P_std",
    "f_raw",
    "stab",
    "dzl",
    "dxl",
    "dyl",
)


def _key(p):
    return (
        f"p{p:.0e}"
        .replace("-", "m")
        .replace("+", "")
    )


# ============================================================================
# Metrics
# ============================================================================

def baseline_loss_from_metrics(
    stab,
    dzl,
    dxl,
    dyl,
):

    stab = np.asarray(stab)
    dzl = np.asarray(dzl)
    dxl = np.asarray(dxl)
    dyl = np.asarray(dyl)

    return (
        W_STAB_BASE
        * (-N_STAB * stab)
        + W_ZL_BASE * dzl ** 2
        + W_XL_BASE * dxl ** 2
        + W_YL_BASE * dyl ** 2
    )


def probe(
    components,
    params,
    p2q,
):
    p1q = P1Q_RATIO * p2q

    metrics = eval_protocol_chunked(
        components["protocol_metrics"],
        params,
        TR,
        p1q=p1q,
        p2q=p2q,
        chunk=CHUNK,
    )

    baseline_loss = (
        baseline_loss_from_metrics(
            metrics["stab"],
            metrics["dzl"],
            metrics["dxl"],
            metrics["dyl"],
        )
    )

    return {
        "loss": float(
            baseline_loss.mean()
        ),
        "loss_std": float(
            baseline_loss.std()
        ),
        "P": float(
            metrics["P"].mean()
        ),
        "P_std": float(
            metrics["P"].std()
        ),
        "fid": float(
            metrics["f_raw"].mean()
        ),
        "fid_std": float(
            metrics["f_raw"].std()
        ),
        "stab": float(
            metrics["stab"].mean()
        ),
        "dzl": float(
            metrics["dzl"].mean()
        ),
        "dxl": float(
            metrics["dxl"].mean()
        ),
        "dyl": float(
            metrics["dyl"].mean()
        ),
    }


def protocol_sweep(
    components,
    params,
    p2q,
):
    p1q = P1Q_RATIO * p2q

    output = {
        key: np.zeros(
            len(PI_SWEEP),
            dtype=float,
        )
        for key in _SW_KEYS
    }

    print(
        f"      {'p_i':>7} "
        f"{'P_corr':>10} "
        f"{'+-std':>9} "
        f"{'P_uncorr':>10} "
        f"{'P_bare':>10} "
        f"{'F_raw':>10}"
    )

    for index, p_i in enumerate(PI_SWEEP):
        metrics = eval_protocol_chunked(
            components["protocol_metrics"],
            params,
            DATA[p_i],
            p1q=p1q,
            p2q=p2q,
            chunk=CHUNK,
        )

        output["P"][index] = (
            metrics["P"].mean()
        )

        output["P_std"][index] = (
            metrics["P"].std()
        )

        output["f_raw"][index] = (
            metrics["f_raw"].mean()
        )

        output["stab"][index] = (
            metrics["stab"].mean()
        )

        output["dzl"][index] = (
            metrics["dzl"].mean()
        )

        output["dxl"][index] = (
            metrics["dxl"].mean()
        )

        output["dyl"][index] = (
            metrics["dyl"].mean()
        )

        print(
            f"      {p_i:>7.4f} "
            f"{output['P'][index]:>10.6f} "
            f"{output['P_std'][index]:>9.6f} "
            f"{P_UNCORR[index]:>10.6f} "
            f"{P_BARE[index]:>10.6f} "
            f"{output['f_raw'][index]:>10.6f}"
        )

    return output


def protocol_single(
    components,
    params,
    p2q,
    index,
):
    p1q = P1Q_RATIO * p2q

    output = {
        key: np.full(
            len(PI_SWEEP),
            np.nan,
            dtype=float,
        )
        for key in _SW_KEYS
    }

    p_i = PI_SWEEP[index]

    metrics = eval_protocol_chunked(
        components["protocol_metrics"],
        params,
        DATA[p_i],
        p1q=p1q,
        p2q=p2q,
        chunk=CHUNK,
    )

    output["P"][index] = (
        metrics["P"].mean()
    )

    output["P_std"][index] = (
        metrics["P"].std()
    )

    output["f_raw"][index] = (
        metrics["f_raw"].mean()
    )

    output["stab"][index] = (
        metrics["stab"].mean()
    )

    output["dzl"][index] = (
        metrics["dzl"].mean()
    )

    output["dxl"][index] = (
        metrics["dxl"].mean()
    )

    output["dyl"][index] = (
        metrics["dyl"].mean()
    )

    print(
        "      stage-only deployment evaluation: "
        f"p_i={p_i:.6f}, "
        f"P={output['P'][index]:.6f}, "
        f"F_raw={output['f_raw'][index]:.6f}"
    )

    return output


# ============================================================================
# Optimizer and native-coordinate initialization
# ============================================================================

def make_optimizer():
    steps_per_epoch = int(
        np.ceil(
            N_TRAIN_STATES
            / BATCH_SIZE
        )
    )

    learning_rate_schedule = (
        optax.cosine_decay_schedule(
            init_value=LR,
            decay_steps=(
                TOTAL_EPOCHS
                * steps_per_epoch
            ),
        )
    )

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(
            learning_rate_schedule
        ),
    )


def clean_style_random_initialization(
    key,
    n_layers,
    round_index,
):
    """Return original CRX/CRY/CRZ coordinates without ZYZ conversion."""

    n_params = count_params(n_layers)
    mode = round_index % 4

    if mode == 0:
        params = jax.random.uniform(
            key,
            (n_params,),
            minval=-np.pi,
            maxval=np.pi,
        )

    elif mode == 1:
        params = (
            0.15
            * jax.random.normal(
                key,
                (n_params,),
            )
        )

    elif mode == 2:
        params = jax.random.normal(
            key,
            (n_params,),
        )

    else:
        params = jax.random.uniform(
            key,
            (n_params,),
            minval=-0.1,
            maxval=0.1,
        )

    return jnp.asarray(
        params,
        dtype=jnp.float64,
    )


# ============================================================================
# Training loop
# ============================================================================

def run_epochs(
    components,
    params,
    opt_state,
    optimizer,
    rng_key,
    weights,
    p2q,
    start_epoch,
    n_epochs,
    history,
    phase,
    label,
):
    p_i_j = jnp.float64(
        PEXT_TRAIN
    )

    p1q_j = jnp.float64(
        P1Q_RATIO * p2q
    )

    p2q_j = jnp.float64(p2q)

    steps_per_epoch = int(
        np.ceil(
            N_TRAIN_STATES
            / BATCH_SIZE
        )
    )

    ws, wz, wx, wy = weights
    last_metrics = None

    print(
        f"    [{label}] "
        f"{'Ep':>4} "
        f"{'BaseLoss':>11} "
        f"{'P_tr':>9} "
        f"{'F_raw':>9} "
        f"{'<S>':>7} "
        f"{'dZL':>8} "
        f"{'dXL':>8} "
        f"{'dYL':>8} "
        f"{'|g|':>10} "
        f"{'t':>7}"
    )

    for local_epoch in range(n_epochs):
        global_epoch = (
            start_epoch + local_epoch
        )

        epoch_start = time.time()

        if (
            global_epoch
            % WEIGHT_UPDATE_FREQ
            == 0
        ):
            current_metrics = probe(
                components,
                params,
                p2q,
            )

            (
                target_ws,
                target_wz,
                target_wx,
                target_wy,
            ) = compute_target_weights(
                current_metrics["stab"]
            )

            ws = step_weight(
                ws,
                target_ws,
            )

            wz = step_weight(
                wz,
                target_wz,
            )

            wx = step_weight(
                wx,
                target_wx,
            )

            wy = step_weight(
                wy,
                target_wy,
            )

        ws_j = jnp.float64(ws)
        wz_j = jnp.float64(wz)
        wx_j = jnp.float64(wx)
        wy_j = jnp.float64(wy)

        rng_key, permutation_key = (
            jax.random.split(rng_key)
        )

        permutation = np.asarray(
            jax.random.permutation(
                permutation_key,
                N_TRAIN_STATES,
            )
        )

        gradient_norm_sum = 0.0
        batches_seen = 0

        for step in range(
            steps_per_epoch
        ):
            begin = step * BATCH_SIZE

            end = min(
                (step + 1) * BATCH_SIZE,
                N_TRAIN_STATES,
            )

            if begin >= end:
                continue

            batch_indices = jnp.asarray(
                permutation[begin:end],
                dtype=jnp.int32,
            )

            loss_value, gradients = (
                components["loss_and_grad"](
                    params,
                    TR["psi"][batch_indices],
                    TR["tgt"][batch_indices],
                    p_i_j,
                    p1q_j,
                    p2q_j,
                    ws_j,
                    wz_j,
                    wx_j,
                    wy_j,
                )
            )

            loss_value = jnp.where(
                jnp.isfinite(loss_value),
                loss_value,
                0.0,
            )

            gradients = jnp.where(
                jnp.isfinite(gradients),
                gradients,
                0.0,
            )

            gradient_norm_sum += float(
                jnp.linalg.norm(gradients)
            )

            batches_seen += 1

            updates, opt_state = (
                optimizer.update(
                    gradients,
                    opt_state,
                    params,
                )
            )

            params = optax.apply_updates(
                params,
                updates,
            )

        last_metrics = probe(
            components,
            params,
            p2q,
        )

        history["epoch"].append(
            global_epoch
        )

        history["phase"].append(
            int(phase)
        )

        history["gnorm"].append(
            gradient_norm_sum
            / max(1, batches_seen)
        )

        for key in _PROBE_KEYS:
            history[key].append(
                last_metrics[key]
            )

        if (
            local_epoch % 10 == 0
            or local_epoch
            == n_epochs - 1
        ):
            print(
                f"    {global_epoch:>4} "
                f"{last_metrics['loss']:>11.4f} "
                f"{last_metrics['P']:>9.6f} "
                f"{last_metrics['fid']:>9.6f} "
                f"{last_metrics['stab']:>7.4f} "
                f"{last_metrics['dzl']:>8.5f} "
                f"{last_metrics['dxl']:>8.5f} "
                f"{last_metrics['dyl']:>8.5f} "
                f"{history['gnorm'][-1]:>10.3e} "
                f"{time.time() - epoch_start:>6.1f}s"
            )

        if global_epoch % 20 == 0:
            gc.collect()

    return (
        params,
        opt_state,
        rng_key,
        (ws, wz, wx, wy),
        last_metrics,
    )


# ============================================================================
# Post-training deployment compiler and pruning
# ============================================================================

def build_post_training_deployment(
    n_layers,
    trained_native_params,
):
    """Convert native CR parameters to fused ZYZ and optionally prune.

    The supplied native parameters are copied and are never modified.
    """

    trained_native_params = np.asarray(
        trained_native_params,
        dtype=float,
    ).ravel()

    expected_count = count_params(
        n_layers
    )

    if (
        trained_native_params.size
        != expected_count
    ):
        raise ValueError(
            f"L={n_layers} requires "
            f"{expected_count} parameters, got "
            f"{trained_native_params.size}."
        )

    try:
        deployed_params = convert_params_to_zyz(
            trained_native_params,
            n_layers,
        )
    except AssertionError as exc:
        print(
            "    WARNING: ZYZ extraction failed at default "
            f"tolerance ({exc}); retrying with relaxed max_err."
        )
        deployed_params = convert_params_to_zyz(
            trained_native_params,
            n_layers,
            max_err=1e-2,
        )

    use_pruning = False
    deployed_prune_stats = None

    prune_pattern_params = np.array(
        [],
        dtype=float,
    )

    candidate_prune_dev = 0.0

    if PRUNE_AT_DEPLOYMENT:
        print(
            "\n    post-training fused-ZYZ "
            "deployment pruning ..."
        )

        (
            candidate_prune_dev,
            _candidate_ops,
            candidate_stats,
            sparse_params,
        ) = validate_pruning(
            deployed_params,
            n_layers,
            tol_rot=PR_TOL_ROT,
            tol_block=PR_TOL_BLK,
            require_all_angles=(
                PRUNE_REQUIRE_ALL_ANGLES
            ),
            verbose=True,
        )

        if prune_is_trivial(
            candidate_stats
        ):
            print(
                "    no deployment gates are "
                "prunable; using full fused circuit"
            )

        elif (
            candidate_prune_dev
            > PR_DEPLOY_MAX_DEV
        ):
            print(
                "    pruning rejected: "
                f"||U_pruned-U_full||_2="
                f"{candidate_prune_dev:.3e} > "
                f"{PR_DEPLOY_MAX_DEV:.1e}"
            )

        else:
            use_pruning = True

            deployed_prune_stats = (
                candidate_stats
            )

            prune_pattern_params = (
                np.asarray(
                    sparse_params,
                    dtype=float,
                ).copy()
            )

            deployed_params = (
                prune_pattern_params.copy()
            )

            print(
                "    pruning accepted: "
                f"{deployed_prune_stats['n_blocks_kept']}/"
                f"{deployed_prune_stats['n_blocks']} "
                "blocks retained, "
                f"||dU||_2="
                f"{candidate_prune_dev:.3e}"
            )

    deployment_model = make_noisy_qae(
        n_layers=n_layers,
        with_noise=True,
        noisy_rz=NOISY_RZ,
        noisy_reset=NOISY_RESET,
        p_reset=P_RESET,
        remat_groups=REMAT_GROUPS,
        prune_params=(
            prune_pattern_params
            if use_pruning
            else None
        ),
        prune_tol_rot=PR_TOL_ROT,
        prune_tol_block=PR_TOL_BLK,
        prune_require_all_angles=(
            PRUNE_REQUIRE_ALL_ANGLES
        ),
        parameterization=(
            DEPLOY_PARAMETERIZATION
        ),
        verbose=True,
    )

    return {
        "model": deployment_model,
        "params": jnp.asarray(
            deployed_params,
            dtype=jnp.float64,
        ),
        "pruned": int(use_pruning),
        "prune_stats": (
            deployed_prune_stats
        ),
        "prune_pattern_params": (
            prune_pattern_params
        ),
        "candidate_prune_dev": float(
            candidate_prune_dev
        ),
    }


# ============================================================================
# Layerwise training
# ============================================================================

def train_stage(
    n_layers,
    p2q,
    init_params=None,
    full_sweep=True,
):
    training_components = make_noisy_qae(
        n_layers=n_layers,
        with_noise=True,
        noisy_rz=NOISY_RZ,
        noisy_reset=NOISY_RESET,
        p_reset=P_RESET,
        remat_groups=REMAT_GROUPS,
        prune_params=None,
        parameterization=(
            TRAIN_PARAMETERIZATION
        ),
        verbose=True,
    )

    optimizer = make_optimizer()

    layer_seed = (
        BASE_SEED + n_layers
    )

    master_rng = jax.random.PRNGKey(
        layer_seed
    )

    n_rounds = (
        N_INIT_ROUNDS
        if n_layers == 1
        else 1
    )

    warmup_candidates = []

    print(
        f"    seed={layer_seed}, "
        f"batch={BATCH_SIZE}, "
        f"LR={LR}, "
        f"epochs={INIT_EPOCHS}+"
        f"{MAIN_EPOCHS}, "
        "training_parameterization="
        f"{TRAIN_PARAMETERIZATION}, "
        "pruning_during_training=False"
    )

    for round_index in range(
        n_rounds
    ):
        if n_layers == 1:
            master_rng, init_key = (
                jax.random.split(master_rng)
            )

            params0 = (
                clean_style_random_initialization(
                    init_key,
                    n_layers,
                    round_index,
                )
            )

            init_label = (
                f"multistart {round_index}"
            )

        else:
            if init_params is None:
                raise ValueError(
                    f"L={n_layers} requires "
                    "init_params."
                )

            params0 = jnp.asarray(
                init_params,
                dtype=jnp.float64,
            )

            init_label = (
                "native-CR layerwise growth"
            )

        master_rng, train_rng = (
            jax.random.split(master_rng)
        )

        opt_state0 = optimizer.init(
            params0
        )

        weights0 = (
            W_STAB_HIGH,
            W_ZL_LOW,
            W_XL_LOW,
            W_YL_LOW,
        )

        history0 = {
            key: []
            for key in _H_KEYS
        }

        print(
            f"\n    warm-up: {init_label}"
        )

        (
            params_warm,
            opt_state_warm,
            rng_warm,
            weights_warm,
            metrics_warm,
        ) = run_epochs(
            components=training_components,
            params=params0,
            opt_state=opt_state0,
            optimizer=optimizer,
            rng_key=train_rng,
            weights=weights0,
            p2q=p2q,
            start_epoch=0,
            n_epochs=INIT_EPOCHS,
            history=history0,
            phase=1,
            label=f"warm-{round_index}",
        )

        warmup_candidates.append(
            {
                "score": float(
                    metrics_warm["fid"]
                ),
                "params": params_warm,
                "opt_state": opt_state_warm,
                "rng_key": rng_warm,
                "weights": weights_warm,
                "history": history0,
                "fid_warm": float(
                    metrics_warm["fid"]
                ),
                "P_warm": float(
                    metrics_warm["P"]
                ),
            }
        )

        print(
            f"    round {round_index:>2}: "
            f"F_raw="
            f"{metrics_warm['fid']:.6f}, "
            f"P_two_interval="
            f"{metrics_warm['P']:.6f}"
        )

        gc.collect()

    selected_warmup = max(
        warmup_candidates,
        key=lambda candidate: (
            candidate["score"]
        ),
    )

    params = selected_warmup["params"]
    opt_state = selected_warmup[
        "opt_state"
    ]
    rng_key = selected_warmup["rng_key"]
    weights = selected_warmup["weights"]
    history = selected_warmup["history"]

    print(
        "\n    selected warm-up: "
        f"F_raw="
        f"{selected_warmup['fid_warm']:.6f}, "
        f"P_two_interval="
        f"{selected_warmup['P_warm']:.6f}"
    )

    print(
        "\n    continuation on full, "
        "unpruned native-CR circuit ..."
    )

    (
        params,
        opt_state,
        rng_key,
        weights,
        final_training_metrics,
    ) = run_epochs(
        components=training_components,
        params=params,
        opt_state=opt_state,
        optimizer=optimizer,
        rng_key=rng_key,
        weights=weights,
        p2q=p2q,
        start_epoch=INIT_EPOCHS,
        n_epochs=MAIN_EPOCHS,
        history=history,
        phase=2,
        label="main",
    )

    native_params = np.asarray(
        params,
        dtype=float,
    ).copy()

    deployment = (
        build_post_training_deployment(
            n_layers=n_layers,
            trained_native_params=(
                native_params
            ),
        )
    )

    deployment_components = (
        deployment["model"]
    )

    deployment_params = (
        deployment["params"]
    )

    prune_stats = deployment[
        "prune_stats"
    ]

    if full_sweep:
        sweep = protocol_sweep(
            deployment_components,
            deployment_params,
            p2q,
        )

    else:
        sweep = protocol_single(
            deployment_components,
            deployment_params,
            p2q,
            I_TRAIN,
        )

    result = {
        # Full native parameters. These are used for layerwise growth.
        "params": native_params,

        # Fused and possibly sparse deployment parameters.
        "params_deployed": np.asarray(
            deployment_params,
            dtype=float,
        ),

        "prune_pattern_params": (
            np.asarray(
                deployment[
                    "prune_pattern_params"
                ],
                dtype=float,
            )
        ),

        "hist": history,

        "P_train_best": float(
            final_training_metrics["P"]
        ),

        "F_train_best": float(
            final_training_metrics["fid"]
        ),

        "P_warmstart": float(
            selected_warmup["P_warm"]
        ),

        "F_warmstart": float(
            selected_warmup["fid_warm"]
        ),

        "n_cnot_train_full": int(
            2
            * training_components[
                "n_cnot_half"
            ]
        ),

        "n_rot_train_full": int(
            2
            * training_components[
                "n_rot_half"
            ]
        ),

        "c_eff_train": float(
            training_components["C_eff"]
        ),

        # Existing output names refer to the reported deployment circuit.
        "n_cnot_full": int(
            2
            * deployment_components[
                "n_cnot_half"
            ]
        ),

        "n_rot_full": int(
            2
            * deployment_components[
                "n_rot_half"
            ]
        ),

        "c_eff": float(
            deployment_components["C_eff"]
        ),

        "swept_full": bool(full_sweep),

        "pruned": deployment["pruned"],

        "n_blocks": (
            -1
            if prune_stats is None
            else int(
                prune_stats["n_blocks"]
            )
        ),

        "n_blocks_kept": (
            -1
            if prune_stats is None
            else int(
                prune_stats[
                    "n_blocks_kept"
                ]
            )
        ),

        "n_blocks_cancel": (
            -1
            if prune_stats is None
            else int(
                prune_stats[
                    "n_blocks_cancel"
                ]
            )
        ),

        "prune_dev": (
            0.0
            if prune_stats is None
            else float(
                prune_stats[
                    "unitary_dev"
                ]
            )
        ),

        "prune_candidate_dev": float(
            deployment[
                "candidate_prune_dev"
            ]
        ),

        **sweep,
    }

    del training_components
    del deployment_components
    gc.collect()

    return result


# ============================================================================
# Header, reference, validation, and datasets
# ============================================================================

print("=" * 78)
print(
    "  STEANE [[7,1,3]] - "
    "NATIVE-CR NOISE-AWARE QAE TRAINING"
)
print("=" * 78)

print(
    f"  JAX backend : "
    f"{jax.default_backend().upper()}"
)

print_config()

print(
    f"  layers                 : "
    f"{LAYERS_PATH}"
)

print(
    f"  training states        : "
    f"{N_TRAIN_STATES}"
)

print(
    f"  test states            : "
    f"{N_TEST_STATES}"
)

print(
    f"  batch size             : "
    f"{BATCH_SIZE}"
)

print(
    f"  LR                     : "
    f"{LR}"
)

print(
    f"  epochs/layer           : "
    f"{INIT_EPOCHS} + {MAIN_EPOCHS}"
)

print(
    f"  L=1 multistarts        : "
    f"{N_INIT_ROUNDS}"
)

print(
    "  p2q values             : "
    f"{[f'{p:.0e}' for p in P2Q_TRAIN]}"
)

print(
    "  training coordinates   : "
    f"{TRAIN_PARAMETERIZATION}"
)

print(
    "  deployment coordinates : "
    f"{DEPLOY_PARAMETERIZATION}"
)

print(
    "  pruning during training: "
    f"{PRUNE_DURING_TRAINING}"
)

print(
    "  pruning at deployment  : "
    f"{PRUNE_AT_DEPLOYMENT}"
)

print(
    "  stabilizer loss        : "
    "-sum_k <S_k>"
)

reference = load_noiseless_reference(
    l_override=L_FINAL,
    verbose=True,
)

validate_logical_code()
validate_depol_channels()
validate_external_depol()
validate_recovery()
validate_uncorr_formula()

PI_SWEEP = [
    float(value)
    for value in np.asarray(
        reference["p_ext"]
    ).ravel()
]

if not PI_SWEEP:
    PI_SWEEP = list(
        PEXT_SWEEP_DEFAULT
    )

if not any(
    abs(p_i - PEXT_TRAIN) < 1e-15
    for p_i in PI_SWEEP
):
    PI_SWEEP.append(
        float(PEXT_TRAIN)
    )

    PI_SWEEP = sorted(PI_SWEEP)

I_TRAIN = int(
    np.argmin(
        np.abs(
            np.asarray(PI_SWEEP)
            - PEXT_TRAIN
        )
    )
)

print(
    f"\n  p_i sweep : {PI_SWEEP}"
)

print(
    f"  train p_i : {PEXT_TRAIN} "
    f"(index {I_TRAIN})"
)

print(
    "\n  Generating training data ..."
)

TR = generate_protocol_dataset(
    n_states=N_TRAIN_STATES,
    p_i=PEXT_TRAIN,
    chunk=CHUNK,
)

print(
    "  Generating test data: "
    f"{N_TEST_STATES} states x "
    f"{len(PI_SWEEP)} p_i values ..."
)

DATA = {
    p_i: generate_protocol_dataset(
        n_states=N_TEST_STATES,
        p_i=p_i,
        chunk=CHUNK,
    )
    for p_i in PI_SWEEP
}

print(
    "\n  Checking logical baseline against "
    "the exact analytic formula:"
)

for p_i in PI_SWEEP:
    check_uncorr_against_analytic(
        DATA[p_i]
    )

P_UNCORR = np.array(
    [
        DATA[p_i]["p_uncorr"].mean()
        for p_i in PI_SWEEP
    ],
    dtype=float,
)

P_BARE = np.array(
    [
        P_bare_physical(p_i)
        for p_i in PI_SWEEP
    ],
    dtype=float,
)


# ============================================================================
# Train each internal-noise model
# ============================================================================

RES = {}
training_start = time.time()

for p2q in P2Q_TRAIN:
    print(
        "\n" + "=" * 78
    )

    print(
        "  INTERNAL GATE-NOISE TRAINING: "
        f"p2q={p2q:.1e}, "
        f"p1q={P1Q_RATIO * p2q:.1e}"
    )

    print("=" * 78)

    RES[p2q] = {}

    previous_params = None
    previous_layers = None

    for n_layers in LAYERS_PATH:
        print(
            f"\n  --- LAYERWISE STAGE "
            f"L={n_layers} "
            + "-" * 48
        )

        if n_layers == 1:
            initial_params = None

            print(
                "      init: independent "
                "native-coordinate multistart"
            )

        else:
            initial_params = grow_params(
                previous_params,
                previous_layers,
                n_layers,
            )

            print(
                "      init: identity growth "
                f"from full native L="
                f"{previous_layers} parameters"
            )

        stage_result = train_stage(
            n_layers=n_layers,
            p2q=p2q,
            init_params=initial_params,
            full_sweep=(
                n_layers == L_FINAL
                or not SWEEP_ONLY_FINAL_L
            ),
        )

        RES[p2q][n_layers] = (
            stage_result
        )

        # Always grow from the full unpruned native parameters.
        previous_params = (
            stage_result["params"]
        )

        previous_layers = n_layers

    final_train_probability = (
        RES[p2q][L_FINAL]["P"][
            I_TRAIN
        ]
    )

    first_train_probability = (
        RES[p2q][
            LAYERS_PATH[0]
        ]["P"][I_TRAIN]
    )

    print(
        f"\n  >>> SUMMARY p2q={p2q:.1e}, "
        f"p_i={PI_SWEEP[I_TRAIN]:.6f}"
    )

    print(
        "      L=3 deployed P="
        f"{final_train_probability:.6f} | "
        "L=1 deployed P="
        f"{first_train_probability:.6f} | "
        "depth gain="
        f"{final_train_probability - first_train_probability:+.6f}"
    )

    print(
        f"      P_uncorr="
        f"{P_UNCORR[I_TRAIN]:.6f} | "
        f"P_bare={P_BARE[I_TRAIN]:.6f}"
    )

    print(
        "      training C_eff="
        f"{RES[p2q][L_FINAL]['c_eff_train']:.1f} | "
        "deployment C_eff="
        f"{RES[p2q][L_FINAL]['c_eff']:.1f}"
    )

elapsed = time.time() - training_start

print(
    "\n  Total training time: "
    f"{elapsed:.0f}s "
    f"({elapsed / 60.0:.1f} min)"
)


# ============================================================================
# Save NPZ
# ============================================================================

P_AWARE = np.stack(
    [
        RES[p2q][L_FINAL]["P"]
        for p2q in P2Q_TRAIN
    ]
)

P_AWARE_STD = np.stack(
    [
        RES[p2q][L_FINAL]["P_std"]
        for p2q in P2Q_TRAIN
    ]
)

assert np.isfinite(P_AWARE).all(), (
    "Reported final L=3 protocol curve "
    "contains NaN or Inf."
)

save_data = {
    "code_n": np.array(7),
    "code_k": np.array(1),
    "code_d": np.array(3),

    "n_layers": np.array(L_FINAL),
    "l_target": np.array(L_TARGET),
    "layers_path": np.array(
        LAYERS_PATH
    ),

    "training_parameterization": np.array(
        TRAIN_PARAMETERIZATION
    ),

    "deployment_parameterization": np.array(
        DEPLOY_PARAMETERIZATION
    ),

    "p_ext": np.array(PI_SWEEP),
    "p2q": np.array(P2Q_TRAIN),
    "p1q_ratio": np.array(
        P1Q_RATIO
    ),
    "pext_train": np.array(
        PEXT_TRAIN
    ),

    "noisy_rz": np.array(
        int(NOISY_RZ)
    ),

    "noisy_reset": np.array(
        int(NOISY_RESET)
    ),

    "p_reset": np.array(P_RESET),

    "protocol_loss": np.array(0),

    "normalized_stabilizer_loss": (
        np.array(0)
    ),

    "unnormalized_stabilizer_loss": (
        np.array(1)
    ),

    "prune_during_training": np.array(
        int(PRUNE_DURING_TRAINING)
    ),

    "prune_at_deployment": np.array(
        int(PRUNE_AT_DEPLOYMENT)
    ),

    "prune_tol_rot": np.array(
        PR_TOL_ROT
    ),

    "prune_tol_block": np.array(
        PR_TOL_BLK
    ),

    "prune_deploy_max_dev": np.array(
        PR_DEPLOY_MAX_DEV
    ),

    "prune_require_all_angles": np.array(
        int(PRUNE_REQUIRE_ALL_ANGLES)
    ),

    "group_l1": np.array(0.0),

    "n_train": np.array(
        N_TRAIN_STATES
    ),

    "n_test": np.array(
        N_TEST_STATES
    ),

    "batch_size": np.array(
        BATCH_SIZE
    ),

    "seed": np.array(BASE_SEED),
    "base_seed": np.array(BASE_SEED),

    "n_init_rounds": np.array(
        N_INIT_ROUNDS
    ),

    "init_epochs": np.array(
        INIT_EPOCHS
    ),

    "main_epochs": np.array(
        MAIN_EPOCHS
    ),

    "total_epochs": np.array(
        TOTAL_EPOCHS
    ),

    "learning_rate": np.array(LR),

    "weight_update_freq": np.array(
        WEIGHT_UPDATE_FREQ
    ),

    "p_aware": P_AWARE,
    "p_aware_std": P_AWARE_STD,

    "p_uncorr": P_UNCORR,
    "p_bare": P_BARE,

    "p_uncorr_analytic": np.array(
        [
            uncorr_analytic_7q(p_i)
            for p_i in PI_SWEEP
        ]
    ),

    "break_even_p_i": np.array(
        break_even_p_i()
    ),

    "i_pext_train": np.array(
        I_TRAIN
    ),

    "f_raw": np.stack(
        [
            RES[p2q][L_FINAL]["f_raw"]
            for p2q in P2Q_TRAIN
        ]
    ),

    "stab": np.stack(
        [
            RES[p2q][L_FINAL]["stab"]
            for p2q in P2Q_TRAIN
        ]
    ),

    "dzl": np.stack(
        [
            RES[p2q][L_FINAL]["dzl"]
            for p2q in P2Q_TRAIN
        ]
    ),

    "dxl": np.stack(
        [
            RES[p2q][L_FINAL]["dxl"]
            for p2q in P2Q_TRAIN
        ]
    ),

    "dyl": np.stack(
        [
            RES[p2q][L_FINAL]["dyl"]
            for p2q in P2Q_TRAIN
        ]
    ),

    "p_warmstart": np.array(
        [
            RES[p2q][L_FINAL][
                "P_warmstart"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "f_warmstart": np.array(
        [
            RES[p2q][L_FINAL][
                "F_warmstart"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "n_cnot_train_full": np.array(
        [
            RES[p2q][L_FINAL][
                "n_cnot_train_full"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "n_rot_train_full": np.array(
        [
            RES[p2q][L_FINAL][
                "n_rot_train_full"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "c_eff_train": np.array(
        [
            RES[p2q][L_FINAL][
                "c_eff_train"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "n_cnot_full": np.array(
        [
            RES[p2q][L_FINAL][
                "n_cnot_full"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "n_rot_full": np.array(
        [
            RES[p2q][L_FINAL][
                "n_rot_full"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "c_eff": np.array(
        [
            RES[p2q][L_FINAL][
                "c_eff"
            ]
            for p2q in P2Q_TRAIN
        ]
    ),

    "sweep_only_final": np.array(
        int(SWEEP_ONLY_FINAL_L)
    ),

    "stage_sweep_full": np.array(
        [
            int(
                RES[P2Q_TRAIN[0]][
                    n_layers
                ]["swept_full"]
            )
            for n_layers
            in LAYERS_PATH
        ]
    ),

    "reference_results_file": np.array(
        str(reference["results_file"])
    ),

    "reference_pext_file": np.array(
        ""
        if reference["pext_file"] is None
        else str(
            reference["pext_file"]
        )
    ),

    "logical_basis_source": np.array(
        LOGICAL_BASIS_SOURCE
    ),
}

for n_layers in LAYERS_PATH:
    save_data[
        f"p_stage_L{n_layers}"
    ] = np.stack(
        [
            RES[p2q][n_layers]["P"]
            for p2q in P2Q_TRAIN
        ]
    )

    save_data[
        f"p_stage_L{n_layers}_std"
    ] = np.stack(
        [
            RES[p2q][n_layers]["P_std"]
            for p2q in P2Q_TRAIN
        ]
    )

    save_data[
        f"p_train_best_L{n_layers}"
    ] = np.array(
        [
            RES[p2q][n_layers][
                "P_train_best"
            ]
            for p2q in P2Q_TRAIN
        ]
    )

    save_data[
        f"f_train_best_L{n_layers}"
    ] = np.array(
        [
            RES[p2q][n_layers][
                "F_train_best"
            ]
            for p2q in P2Q_TRAIN
        ]
    )

    for field in (
        "n_cnot_train_full",
        "n_rot_train_full",
        "c_eff_train",
        "n_cnot_full",
        "n_rot_full",
        "c_eff",
        "pruned",
        "n_blocks",
        "n_blocks_kept",
        "n_blocks_cancel",
        "prune_dev",
        "prune_candidate_dev",
    ):
        save_data[
            f"{field}_L{n_layers}"
        ] = np.array(
            [
                RES[p2q][n_layers][field]
                for p2q in P2Q_TRAIN
            ]
        )

for p2q in P2Q_TRAIN:
    pkey = _key(p2q)

    for n_layers in LAYERS_PATH:
        stage = RES[p2q][n_layers]

        save_data[
            f"{pkey}_L{n_layers}_params_native"
        ] = np.asarray(
            stage["params"]
        )

        # Backward-compatible alias: this is still the full native vector.
        save_data[
            f"{pkey}_L{n_layers}_params"
        ] = np.asarray(
            stage["params"]
        )

        save_data[
            f"{pkey}_L{n_layers}_params_deployed"
        ] = np.asarray(
            stage["params_deployed"]
        )

        save_data[
            f"{pkey}_L{n_layers}_prune_pattern_params"
        ] = np.asarray(
            stage["prune_pattern_params"]
        )

        for history_key, values in (
            stage["hist"].items()
        ):
            save_data[
                f"{pkey}_L{n_layers}_hist_"
                f"{history_key}"
            ] = np.asarray(values)

save_data["params_init"] = np.array(
    [],
    dtype=float,
)

npz_path = (
    f"internal7_{OUT_TAG}.npz"
)

np.savez(
    npz_path,
    **save_data,
)


# ============================================================================
# Save protocol CSV
# ============================================================================

csv_path = (
    f"internal7_{OUT_TAG}.csv"
)

with open(
    csv_path,
    "w",
    newline="",
) as file:
    writer = csv.writer(file)

    header = [
        "p2q",
        "p1q",
        "p_i",
        "P_corr",
        "P_corr_std",
        "P_uncorr_logical",
        "P_bare_physical",
        "gain_vs_logical",
        "gain_vs_physical",
        "F_raw_1round",
        "stab",
        "dzl",
        "dxl",
        "dyl",
        "training_parameterization",
        "deployment_parameterization",
        "pruned",
        "n_cnot_train_full",
        "n_rot_train_full",
        "c_eff_train",
        "n_cnot_deploy_full",
        "n_rot_deploy_full",
        "c_eff_deploy",
        "n_blocks_kept",
        "n_blocks",
        "n_blocks_cancel",
        "prune_unitary_dev",
        "prune_candidate_dev",
    ]

    header += [
        f"P_L{n_layers}"
        for n_layers in LAYERS_PATH
    ]

    writer.writerow(header)

    for p_index, p2q in enumerate(
        P2Q_TRAIN
    ):
        final_result = RES[p2q][L_FINAL]

        for index, p_i in enumerate(
            PI_SWEEP
        ):
            row = [
                f"{p2q:.6e}",
                f"{P1Q_RATIO * p2q:.6e}",
                f"{p_i:.6f}",

                f"{P_AWARE[p_index, index]:.8f}",
                f"{P_AWARE_STD[p_index, index]:.8f}",
                f"{P_UNCORR[index]:.8f}",
                f"{P_BARE[index]:.8f}",

                f"{P_AWARE[p_index, index] - P_UNCORR[index]:.8f}",
                f"{P_AWARE[p_index, index] - P_BARE[index]:.8f}",

                f"{final_result['f_raw'][index]:.8f}",
                f"{final_result['stab'][index]:.8f}",
                f"{final_result['dzl'][index]:.8f}",
                f"{final_result['dxl'][index]:.8f}",
                f"{final_result['dyl'][index]:.8f}",

                TRAIN_PARAMETERIZATION,
                DEPLOY_PARAMETERIZATION,

                final_result["pruned"],

                final_result[
                    "n_cnot_train_full"
                ],

                final_result[
                    "n_rot_train_full"
                ],

                f"{final_result['c_eff_train']:.6f}",

                final_result[
                    "n_cnot_full"
                ],

                final_result[
                    "n_rot_full"
                ],

                f"{final_result['c_eff']:.6f}",

                final_result[
                    "n_blocks_kept"
                ],

                final_result[
                    "n_blocks"
                ],

                final_result[
                    "n_blocks_cancel"
                ],

                f"{final_result['prune_dev']:.6e}",

                f"{final_result['prune_candidate_dev']:.6e}",
            ]

            for n_layers in LAYERS_PATH:
                value = RES[p2q][
                    n_layers
                ]["P"][index]

                row.append(
                    ""
                    if not np.isfinite(value)
                    else f"{value:.8f}"
                )

            writer.writerow(row)


# ============================================================================
# Save training-history CSV
# ============================================================================

history_path = (
    f"internal7_{OUT_TAG}_history.csv"
)

with open(
    history_path,
    "w",
    newline="",
) as file:
    writer = csv.writer(file)

    writer.writerow(
        [
            "p2q",
            "p1q",
            "n_layers",
            "epoch",
            "phase",
            "baseline_loss_unnormalized",
            "baseline_loss_std",
            "P_protocol",
            "P_protocol_std",
            "F_raw",
            "F_raw_std",
            "stab_mean",
            "dzl",
            "dxl",
            "dyl",
            "grad_norm",
        ]
    )

    for p2q in P2Q_TRAIN:
        for n_layers in LAYERS_PATH:
            history = RES[p2q][
                n_layers
            ]["hist"]

            for index in range(
                len(history["epoch"])
            ):
                writer.writerow(
                    [
                        f"{p2q:.6e}",
                        f"{P1Q_RATIO * p2q:.6e}",
                        n_layers,

                        history["epoch"][index],
                        history["phase"][index],

                        f"{history['loss'][index]:.8f}",
                        f"{history['loss_std'][index]:.8f}",

                        f"{history['P'][index]:.8f}",
                        f"{history['P_std'][index]:.8f}",

                        f"{history['fid'][index]:.8f}",
                        f"{history['fid_std'][index]:.8f}",

                        f"{history['stab'][index]:.8f}",
                        f"{history['dzl'][index]:.8f}",
                        f"{history['dxl'][index]:.8f}",
                        f"{history['dyl'][index]:.8f}",

                        f"{history['gnorm'][index]:.8e}",
                    ]
                )

print(f"\n  Saved: {npz_path}")
print(f"  Saved: {csv_path}")
print(f"  Saved: {history_path}")
print("\n  Done")
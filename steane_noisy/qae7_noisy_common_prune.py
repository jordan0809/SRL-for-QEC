# ============================================================================
# qae7_noisy_common_prune.py
#
# Steane [[7,1,3]] noisy quantum-memory QAE utilities.
# ============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import re

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import jax
jax.config.update("jax_enable_x64", True)

import logging
logging.getLogger("jax").setLevel(logging.WARNING)

try:
    jax.config.update("jax_platform_name", "gpu")
    jax.devices()
except Exception:
    jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np

from functools import reduce


# ============================================================================
# Dimensions and configuration
# ============================================================================

N_CODE = 7
N_ANC = 1
N_TOTAL = N_CODE + N_ANC

SYS_DIM = 2 ** N_CODE
ANC_DIM = 2 ** N_ANC
TOT_DIM = 2 ** N_TOTAL

PEXT_TRAIN = 0.05

PEXT_SWEEP_DEFAULT = [
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
    0.15,
    0.20,
]

P2Q_SWEEP = [
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]

P1Q_RATIO = 0.1
NOISY_RZ = True
NOISY_RESET = False
P_RESET = 0.0

L_TARGET = 3
LAYERWISE_PATH = list(range(1, L_TARGET + 1))
REMAT_GROUPS = 8

PRUNE_TOL_ROT = 1e-4
PRUNE_TOL_BLOCK = 1e-3
PRUNE_REQUIRE_ALL_ANGLES = False


# ============================================================================
# Pauli and Steane-code operators
# ============================================================================

I2 = np.eye(2, dtype=complex)

X_ = np.array(
    [[0, 1], [1, 0]],
    dtype=complex,
)

Y_ = np.array(
    [[0, -1j], [1j, 0]],
    dtype=complex,
)

Z_ = np.array(
    [[1, 0], [0, -1]],
    dtype=complex,
)


def kron_list(ops):
    return reduce(np.kron, ops)


def op_on(qubits, mat, n=N_CODE):
    ops = [I2] * n

    for q in qubits:
        ops[q] = mat

    return kron_list(ops)


STAB_SUPPORTS = [
    [0, 2, 4, 6],
    [1, 2, 5, 6],
    [3, 4, 5, 6],
]

S_mats = (
    [op_on(s, X_) for s in STAB_SUPPORTS]
    + [op_on(s, Z_) for s in STAB_SUPPORTS]
)

X_STAB_IDX = [0, 1, 2]
Z_STAB_IDX = [3, 4, 5]
N_STAB = len(S_mats)

LOGICAL_SUPPORT = [0, 1, 2]

XL_mat = op_on(LOGICAL_SUPPORT, X_)
ZL_mat = op_on(LOGICAL_SUPPORT, Z_)
YL_mat = 1j * XL_mat @ ZL_mat

S_STACK = jnp.array(np.stack(S_mats))
L_STACK = jnp.array(np.stack([ZL_mat, XL_mat, YL_mat]))

ANC_0_J = (
    jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex)
    .at[0, 0]
    .set(1.0)
)

FRESH_0_J = (
    jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex)
    .at[0, 0]
    .set(1.0)
)


def _compute_logical_basis():
    projector = np.eye(SYS_DIM, dtype=complex)

    for stabilizer in S_mats:
        projector = projector @ (
            (np.eye(SYS_DIM) + stabilizer) / 2
        )

    eigenvalues, eigenvectors = np.linalg.eigh(projector)

    code = eigenvectors[
        :,
        np.isclose(eigenvalues, 1.0),
    ]

    assert code.shape[1] == 2, (
        "Code space is not two-dimensional."
    )

    logical_z_in_code = (
        code.conj().T @ ZL_mat @ code
    )

    z_eigenvalues, z_vectors = np.linalg.eigh(
        logical_z_in_code
    )

    ket0 = code @ z_vectors[:, 0]
    ket1 = code @ z_vectors[:, 1]

    if z_eigenvalues[0] < z_eigenvalues[1]:
        ket0, ket1 = ket1, ket0

    ket0 = ket0 / np.linalg.norm(ket0)
    ket1 = ket1 / np.linalg.norm(ket1)

    return ket0, ket1


ket_0L, ket_1L = _compute_logical_basis()
LOGICAL_BASIS_SOURCE = "recomputed (eigh)"


def set_logical_basis(k0, k1, source="loaded from npz"):
    global ket_0L
    global ket_1L
    global LOGICAL_BASIS_SOURCE

    k0 = np.asarray(k0, complex).ravel()
    k1 = np.asarray(k1, complex).ravel()

    assert k0.size == SYS_DIM
    assert k1.size == SYS_DIM

    ket_0L = k0 / np.linalg.norm(k0)
    ket_1L = k1 / np.linalg.norm(k1)
    LOGICAL_BASIS_SOURCE = source


def validate_logical_code(verbose=True):
    worst = 0.0

    for ket, logical_z_sign in (
        (ket_0L, +1.0),
        (ket_1L, -1.0),
    ):
        for stabilizer in S_mats:
            worst = max(
                worst,
                float(
                    np.max(
                        np.abs(stabilizer @ ket - ket)
                    )
                ),
            )

        worst = max(
            worst,
            float(
                np.max(
                    np.abs(
                        ZL_mat @ ket
                        - logical_z_sign * ket
                    )
                )
            ),
        )

    overlap = complex(
        ket_1L.conj() @ (XL_mat @ ket_0L)
    )

    hermitian_y_error = float(
        np.max(
            np.abs(YL_mat - YL_mat.conj().T)
        )
    )

    y_squared_error = float(
        np.max(
            np.abs(
                YL_mat @ YL_mat
                - np.eye(SYS_DIM)
            )
        )
    )

    if verbose:
        print(
            f"  [validate] logical basis "
            f"({LOGICAL_BASIS_SOURCE}): "
            f"stab/ZL dev {worst:.2e} | "
            f"YL Hermitian {hermitian_y_error:.1e}, "
            f"YL^2=I {y_squared_error:.1e}"
        )

        print(
            f"  [validate] <1L|XL|0L> = "
            f"{abs(overlap):.6f} * exp(i*"
            f"{np.angle(overlap):+.6f})"
        )

    assert worst < 1e-9
    assert hermitian_y_error < 1e-12
    assert y_squared_error < 1e-12

    return worst


def logical_state(theta, phi):
    return (
        np.cos(theta / 2) * ket_0L
        + np.exp(1j * phi)
        * np.sin(theta / 2)
        * ket_1L
    )


def state_to_dm(psi):
    psi = np.asarray(psi).reshape(-1, 1)
    return psi @ psi.conj().T


def fibonacci_sphere(n):
    golden = (1 + np.sqrt(5)) / 2

    return [
        (
            np.arccos(
                1 - 2 * (i + 0.5) / n
            ),
            (
                2 * np.pi * i / golden
            ) % (2 * np.pi),
        )
        for i in range(n)
    ]


# ============================================================================
# External depolarizing noise
# ============================================================================

def depolarizing_channel(rho, p):
    result = rho.copy()

    for q in range(N_CODE):
        next_result = (1 - p) * result

        for pauli in (X_, Y_, Z_):
            error = op_on([q], pauli)

            next_result = (
                next_result
                + (p / 3)
                * (error @ result @ error)
            )

        result = next_result

    return result


_SPLIT1_CODE = {
    q: (
        2 ** q,
        2 ** (N_CODE - q - 1),
    )
    for q in range(N_CODE)
}

I2J = jnp.eye(2, dtype=complex)


def _mix1_code(rho, q):
    batch = rho.shape[0]
    left, right = _SPLIT1_CODE[q]

    reshaped = rho.reshape(
        batch,
        left,
        2,
        right,
        left,
        2,
        right,
    )

    traced = jnp.trace(
        reshaped,
        axis1=2,
        axis2=5,
    )

    return jnp.einsum(
        "bimjn,xy->bixmjyn",
        traced,
        I2J,
    ).reshape(
        batch,
        SYS_DIM,
        SYS_DIM,
    )


def depol_code_j(rho, p):
    for q in range(N_CODE):
        rho = (
            (1.0 - 4.0 * p / 3.0) * rho
            + (2.0 * p / 3.0)
            * _mix1_code(rho, q)
        )

    return rho


def p_eff_two_rounds(p):
    p = np.asarray(p, float)

    return 0.75 * (
        1.0
        - (
            1.0
            - 4.0 * p / 3.0
        ) ** 2
    )


def validate_external_depol(
    p=0.137,
    seed=1,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    matrix = (
        rng.normal(size=(SYS_DIM, SYS_DIM))
        + 1j
        * rng.normal(size=(SYS_DIM, SYS_DIM))
    )

    rho = matrix @ matrix.conj().T
    rho /= np.trace(rho).real

    reference = depolarizing_channel(rho, p)

    result = np.asarray(
        depol_code_j(
            jnp.array(rho[None]),
            p,
        )
    )[0]

    error = float(
        np.max(
            np.abs(result - reference)
        )
    )

    if verbose:
        print(
            "  [validate] closed-form external "
            f"depol vs Kraus : {error:.3e}"
        )

    assert error < 1e-12, (
        "External depolarizing implementation is wrong."
    )

    return error


# ============================================================================
# Perfect final QEC
# ============================================================================

_XC = (
    [np.eye(SYS_DIM, dtype=complex)]
    + [
        op_on([q], X_)
        for q in range(N_CODE)
    ]
)

_ZC = (
    [np.eye(SYS_DIM, dtype=complex)]
    + [
        op_on([q], Z_)
        for q in range(N_CODE)
    ]
)

_XCL = ["I"] + [
    f"X{q}"
    for q in range(N_CODE)
]

_ZCL = ["I"] + [
    f"Z{q}"
    for q in range(N_CODE)
]

CORR_OPS = np.stack(
    [
        x_correction @ z_correction
        for x_correction in _XC
        for z_correction in _ZC
    ]
)

CORRECTABLE_LABELS = [
    f"{a}{b}"
    for a in _XCL
    for b in _ZCL
]

N_CORR = CORR_OPS.shape[0]
CORR_OPS_J = jnp.array(CORR_OPS)

P_MAXMIX = N_CORR / SYS_DIM


def recovery_vectors(psi_batch):
    return np.einsum(
        "sij,bj->bsi",
        CORR_OPS,
        np.asarray(psi_batch),
    )


def final_qec_fidelity_np(rho, vectors):
    return np.real(
        np.einsum(
            "bsi,bij,bsj->b",
            vectors.conj(),
            rho,
            vectors,
        )
    )


def validate_recovery(n_states=8, verbose=True):
    worst_projector = 0.0
    worst_no_error = 0.0
    worst_weight_one = 0.0
    worst_weight_two = 0.0

    for theta, phi in fibonacci_sphere(n_states):
        psi = logical_state(theta, phi)
        vectors = recovery_vectors(psi[None])[0]

        projector = np.einsum(
            "si,sj->ij",
            vectors,
            vectors.conj(),
        )

        worst_projector = max(
            worst_projector,
            float(
                np.max(
                    np.abs(
                        projector @ projector
                        - projector
                    )
                )
            ),
            float(
                abs(
                    np.trace(projector).real
                    - N_CORR
                )
            ),
            float(
                np.max(
                    np.abs(
                        projector
                        - projector.conj().T
                    )
                )
            ),
        )

        rho0 = state_to_dm(psi)

        worst_no_error = max(
            worst_no_error,
            abs(
                np.real(
                    np.trace(projector @ rho0)
                )
                - 1.0
            ),
        )

        for q in range(N_CODE):
            for pauli in (X_, Y_, Z_):
                error = op_on([q], pauli)

                recovered = np.real(
                    np.trace(
                        projector
                        @ error
                        @ rho0
                        @ error
                    )
                )

                worst_weight_one = max(
                    worst_weight_one,
                    abs(recovered - 1.0),
                )

        for i in range(N_CODE):
            for j in range(N_CODE):
                if i == j:
                    continue

                error = (
                    op_on([i], X_)
                    @ op_on([j], Z_)
                )

                recovered = np.real(
                    np.trace(
                        projector
                        @ error
                        @ rho0
                        @ error.conj().T
                    )
                )

                worst_weight_two = max(
                    worst_weight_two,
                    abs(recovered - 1.0),
                )

    if verbose:
        print(
            "  [validate] Pi projector / rank 64      : "
            f"{worst_projector:.2e}"
        )
        print(
            "  [validate] P(no error)             = 1 : "
            f"{worst_no_error:.2e}"
        )
        print(
            "  [validate] P(any weight-1 error)   = 1 : "
            f"{worst_weight_one:.2e}"
        )
        print(
            "  [validate] P(X_i Z_j, i != j)      = 1 : "
            f"{worst_weight_two:.2e}"
        )
        print(
            "  [info] maximally mixed floor <P>       : "
            f"{P_MAXMIX:.3f}"
        )

    assert max(
        worst_projector,
        worst_no_error,
        worst_weight_one,
        worst_weight_two,
    ) < 1e-10, (
        "Final perfect-QEC recovery is wrong."
    )

    return worst_projector


# ============================================================================
# Analytic baselines
# ============================================================================

def P_bare_physical(p_i):
    return (
        1.0
        - (2.0 / 3.0)
        * p_eff_two_rounds(p_i)
    )


def _rowspace(rows):
    result = {0}

    for row in rows:
        result |= {
            x ^ row
            for x in result
        }

    return sorted(result)


_H_ROWS = [
    sum(
        1 << (N_CODE - 1 - q)
        for q in support
    )
    for support in STAB_SUPPORTS
]

_CPERP = _rowspace(_H_ROWS)

_UNITS = [0] + [
    1 << (N_CODE - 1 - q)
    for q in range(N_CODE)
]

_ALLOWED = sorted(
    {
        vector ^ error
        for vector in _CPERP
        for error in _UNITS
    }
)

assert len(_ALLOWED) == 64

_A = np.asarray(
    _ALLOWED,
    dtype=np.int64,
)

_UN = (
    _A[:, None]
    | _A[None, :]
).ravel()

CORR_WEIGHT_ENUM = np.bincount(
    np.array(
        [
            bin(int(value)).count("1")
            for value in _UN
        ]
    ),
    minlength=N_CODE + 1,
)


def uncorr_analytic_7q(p_i):
    q = p_eff_two_rounds(p_i)

    no_error_probability = 1.0 - q
    pauli_probability = q / 3.0

    weights = np.arange(N_CODE + 1)

    identity_logical_probability = np.sum(
        CORR_WEIGHT_ENUM
        * no_error_probability ** (
            N_CODE - weights
        )
        * pauli_probability ** weights
    )

    return (
        1.0
        + 2.0 * identity_logical_probability
    ) / 3.0


P_uncorr_analytic = uncorr_analytic_7q


def break_even_p_i(
    lo=1e-7,
    hi=0.5,
    tol=1e-12,
):
    def difference(p):
        return float(
            uncorr_analytic_7q(p)
            - P_bare_physical(p)
        )

    lower = lo
    upper = hi

    if (
        difference(lower) <= 0
        or difference(upper) >= 0
    ):
        return float("nan")

    while upper - lower > tol:
        middle = 0.5 * (
            lower + upper
        )

        if difference(middle) > 0:
            lower = middle
        else:
            upper = middle

    return 0.5 * (
        lower + upper
    )


def validate_uncorr_formula(verbose=True):
    total = int(CORR_WEIGHT_ENUM.sum())

    assert total == 4096
    assert CORR_WEIGHT_ENUM[0] == 1
    assert CORR_WEIGHT_ENUM[1] == 21
    assert CORR_WEIGHT_ENUM[2] == 42

    maximally_mixed = float(
        uncorr_analytic_7q(0.75)
    )

    deviation = abs(
        maximally_mixed - 0.5
    )

    break_even = break_even_p_i()

    if verbose:
        print(
            "  [validate] coset weight enumerator     : "
            f"{list(map(int, CORR_WEIGHT_ENUM))} "
            f"(sum {total})"
        )
        print(
            "  [validate] maximally mixed limit <P>   : "
            f"{maximally_mixed:.12f} "
            f"(dev {deviation:.1e})"
        )
        print(
            "  [info] pseudothreshold p_i*            : "
            f"{break_even:.4f} "
            f"(q*={p_eff_two_rounds(break_even):.4f})"
        )

    assert deviation < 1e-12

    return deviation


# ============================================================================
# Protocol datasets
# ============================================================================

@jax.jit
def _baseline_batch(psi_b, p_i):
    rho_code = jnp.einsum(
        "bi,bj->bij",
        psi_b,
        jnp.conj(psi_b),
    )

    first = depol_code_j(
        rho_code,
        p_i,
    )

    second = depol_code_j(
        first,
        p_i,
    )

    vectors = jnp.einsum(
        "sij,bj->bsi",
        CORR_OPS_J,
        psi_b,
    )

    probability = jnp.real(
        jnp.einsum(
            "bsi,bij,bsj->b",
            jnp.conj(vectors),
            second,
            vectors,
        )
    )

    raw_fidelity = jnp.real(
        jnp.einsum(
            "bi,bij,bj->b",
            jnp.conj(psi_b),
            first,
            psi_b,
        )
    )

    return probability, raw_fidelity


def generate_protocol_dataset(
    n_states,
    p_i,
    chunk=50,
):
    psi = np.stack(
        [
            logical_state(theta, phi)
            for theta, phi
            in fibonacci_sphere(n_states)
        ]
    )

    targets = np.stack(
        [
            np.real(
                np.einsum(
                    "bi,ij,bj->b",
                    psi.conj(),
                    operator,
                    psi,
                )
            )
            for operator in (
                ZL_mat,
                XL_mat,
                YL_mat,
            )
        ],
        axis=1,
    )

    psi_j = jnp.array(psi)

    probability = np.empty(n_states)
    raw_fidelity = np.empty(n_states)

    p_j = jnp.float64(p_i)

    n_chunks = int(
        np.ceil(n_states / chunk)
    )

    for index in range(n_chunks):
        sl = slice(
            index * chunk,
            min(
                (index + 1) * chunk,
                n_states,
            ),
        )

        probability_chunk, fidelity_chunk = (
            _baseline_batch(
                psi_j[sl],
                p_j,
            )
        )

        probability[sl] = np.asarray(
            probability_chunk
        )

        raw_fidelity[sl] = np.asarray(
            fidelity_chunk
        )

    return {
        "psi": psi_j,
        "psi_np": psi,
        "tgt": jnp.array(targets),
        "n": int(n_states),
        "p_i": float(p_i),
        "p_uncorr": probability,
        "f_raw_nocorr": raw_fidelity,
    }


def check_uncorr_against_analytic(
    dataset,
    tol_warn=2e-3,
    tol_fail=1e-2,
):
    numeric = float(
        dataset["p_uncorr"].mean()
    )

    analytic = float(
        uncorr_analytic_7q(
            dataset["p_i"]
        )
    )

    deviation = abs(
        numeric - analytic
    )

    status = (
        "OK"
        if deviation < tol_warn
        else "CHECK"
    )

    print(
        f"    p_i={dataset['p_i']:.3f}: "
        f"P_uncorr numeric {numeric:.6f} "
        f"vs analytic {analytic:.6f} "
        f"(dev {deviation:.2e}) [{status}]"
    )

    assert deviation < tol_fail

    return deviation


# ============================================================================
# Parameter layout and schedules
# ============================================================================

def count_params(
    n_layers,
    n_total=N_TOTAL,
):
    return (
        (n_layers + 1)
        * n_total
        * 2
        + n_layers
        * n_total
        * (n_total - 1)
        * 3
    )


def _combine_terms(*scaled_term_groups):
    accumulator = {}

    for scale, terms in scaled_term_groups:
        for index, coefficient in terms:
            accumulator[index] = (
                accumulator.get(index, 0.0)
                + scale * coefficient
            )

    result = []

    if (
        None in accumulator
        and abs(accumulator[None]) > 1e-15
    ):
        result.append(
            (
                None,
                accumulator[None],
            )
        )

    indices = sorted(
        index
        for index in accumulator
        if index is not None
    )

    for index in indices:
        if abs(accumulator[index]) > 1e-15:
            result.append(
                (
                    index,
                    accumulator[index],
                )
            )

    return tuple(result)


def _eval_terms(params, terms):
    value = 0.0

    for index, coefficient in terms:
        if index is None:
            value = value + coefficient
        else:
            value = (
                value
                + coefficient * params[index]
            )

    return value


def _append_rotation_expression(
    ops,
    axis,
    qubit,
    terms,
):
    terms = tuple(terms)

    if terms:
        ops.append(
            (
                "ROT",
                axis,
                qubit,
                terms,
            )
        )


def _append_compiled_cu(
    ops,
    control,
    target,
    beta_terms,
    gamma_terms,
    delta_terms,
):
    """Append a two-CNOT decomposition of controlled
       RZ(beta) RY(gamma) RZ(delta).
    """

    first_z = _combine_terms(
        (+0.5, delta_terms),
        (-0.5, beta_terms),
    )

    middle_z = _combine_terms(
        (-0.5, delta_terms),
        (-0.5, beta_terms),
    )

    minus_half_y = _combine_terms(
        (-0.5, gamma_terms),
    )

    plus_half_y = _combine_terms(
        (+0.5, gamma_terms),
    )

    _append_rotation_expression(
        ops,
        "Z",
        target,
        first_z,
    )

    ops.append(
        (
            "CNOT",
            control,
            target,
        )
    )

    _append_rotation_expression(
        ops,
        "Z",
        target,
        middle_z,
    )

    _append_rotation_expression(
        ops,
        "Y",
        target,
        minus_half_y,
    )

    ops.append(
        (
            "CNOT",
            control,
            target,
        )
    )

    _append_rotation_expression(
        ops,
        "Y",
        target,
        plus_half_y,
    )

    _append_rotation_expression(
        ops,
        "Z",
        target,
        beta_terms,
    )


def build_schedule(
    n_layers,
    n_total=N_TOTAL,
):
    """Fused-ZYZ schedule: two CNOTs per CRX/CRY/CRZ triplet."""

    ops = []
    index = 0

    for q in range(n_total):
        ops.append(
            (
                "ROT",
                "Y",
                q,
                ((index, 1.0),),
            )
        )
        index += 1

        ops.append(
            (
                "ROT",
                "Z",
                q,
                ((index, 1.0),),
            )
        )
        index += 1

    for _ in range(n_layers):
        for control in range(n_total):
            for target in range(n_total):
                if control == target:
                    continue

                beta = index
                gamma = index + 1
                delta = index + 2
                index += 3

                _append_compiled_cu(
                    ops,
                    control,
                    target,
                    beta_terms=((beta, 1.0),),
                    gamma_terms=((gamma, 1.0),),
                    delta_terms=((delta, 1.0),),
                )

        for q in range(n_total):
            ops.append(
                (
                    "ROT",
                    "Y",
                    q,
                    ((index, 1.0),),
                )
            )
            index += 1

            ops.append(
                (
                    "ROT",
                    "Z",
                    q,
                    ((index, 1.0),),
                )
            )
            index += 1

    assert index == count_params(
        n_layers,
        n_total,
    )

    return ops


def build_cr_native_schedule(
    n_layers,
    n_total=N_TOTAL,
):
    """Original Model-B coordinates.

    Every triplet is interpreted as:
        CRX(alpha_x)
        CRY(alpha_y)
        CRZ(alpha_z)

    Each controlled rotation is compiled separately. Therefore each triplet
    uses six CNOTs during native-coordinate training.
    """

    ops = []
    index = 0

    for q in range(n_total):
        ops.append(
            (
                "ROT",
                "Y",
                q,
                ((index, 1.0),),
            )
        )
        index += 1

        ops.append(
            (
                "ROT",
                "Z",
                q,
                ((index, 1.0),),
            )
        )
        index += 1

    for _ in range(n_layers):
        for control in range(n_total):
            for target in range(n_total):
                if control == target:
                    continue

                alpha_x = index
                alpha_y = index + 1
                alpha_z = index + 2
                index += 3

                # RX(a) = RZ(-pi/2) RY(a) RZ(+pi/2)
                _append_compiled_cu(
                    ops,
                    control,
                    target,
                    beta_terms=(
                        (None, -np.pi / 2),
                    ),
                    gamma_terms=(
                        (alpha_x, 1.0),
                    ),
                    delta_terms=(
                        (None, +np.pi / 2),
                    ),
                )

                # RY(a) = RZ(0) RY(a) RZ(0)
                _append_compiled_cu(
                    ops,
                    control,
                    target,
                    beta_terms=(),
                    gamma_terms=(
                        (alpha_y, 1.0),
                    ),
                    delta_terms=(),
                )

                # RZ(a) = RZ(a) RY(0) RZ(0)
                _append_compiled_cu(
                    ops,
                    control,
                    target,
                    beta_terms=(
                        (alpha_z, 1.0),
                    ),
                    gamma_terms=(),
                    delta_terms=(),
                )

        for q in range(n_total):
            ops.append(
                (
                    "ROT",
                    "Y",
                    q,
                    ((index, 1.0),),
                )
            )
            index += 1

            ops.append(
                (
                    "ROT",
                    "Z",
                    q,
                    ((index, 1.0),),
                )
            )
            index += 1

    assert index == count_params(
        n_layers,
        n_total,
    )

    return ops


def adjoint_schedule(ops):
    result = []

    for op in reversed(ops):
        if op[0] == "ROT":
            _, axis, qubit, terms = op

            result.append(
                (
                    "ROT",
                    axis,
                    qubit,
                    tuple(
                        (
                            index,
                            -coefficient,
                        )
                        for index, coefficient
                        in terms
                    ),
                )
            )
        else:
            result.append(op)

    return result


def gate_counts(ops):
    n_rot = sum(
        1
        for op in ops
        if op[0] == "ROT"
    )

    n_cnot = sum(
        1
        for op in ops
        if op[0] == "CNOT"
    )

    n_rz = sum(
        1
        for op in ops
        if (
            op[0] == "ROT"
            and op[1] == "Z"
        )
    )

    return n_rot, n_cnot, n_rz


def effective_C(
    n_cnot_full,
    n_rot_full,
    n_rz_full,
    p1q_ratio=P1Q_RATIO,
    noisy_rz=NOISY_RZ,
):
    n_noisy_rotations = (
        n_rot_full
        if noisy_rz
        else n_rot_full - n_rz_full
    )

    return (
        (16.0 / 15.0)
        * n_cnot_full
        + (4.0 / 3.0)
        * p1q_ratio
        * n_noisy_rotations
    )


# ============================================================================
# Batched gate application
# ============================================================================

def _rot2(axis, theta):
    pauli = (
        jnp.array(Y_)
        if axis == "Y"
        else jnp.array(Z_)
    )

    return (
        jnp.cos(theta / 2) * I2J
        - 1j
        * jnp.sin(theta / 2)
        * pauli
    )


_SPLIT1 = {
    q: (
        2 ** q,
        2 ** (N_TOTAL - q - 1),
    )
    for q in range(N_TOTAL)
}


def _apply_1q(rho, unitary, q):
    batch = rho.shape[0]
    left, right = _SPLIT1[q]

    tensor = rho.reshape(
        batch,
        left,
        2,
        right,
        TOT_DIM,
    )

    tensor = jnp.einsum(
        "yx,bixjc->biyjc",
        unitary,
        tensor,
    ).reshape(
        batch,
        TOT_DIM,
        TOT_DIM,
    )

    tensor = tensor.reshape(
        batch,
        TOT_DIM,
        left,
        2,
        right,
    )

    adjoint = jnp.conj(unitary).T

    return jnp.einsum(
        "bcixj,xy->bciyj",
        tensor,
        adjoint,
    ).reshape(
        batch,
        TOT_DIM,
        TOT_DIM,
    )


def _cnot_perm_np(
    control,
    target,
    n=N_TOTAL,
):
    indices = np.arange(2 ** n)

    control_bit = (
        indices
        >> (n - 1 - control)
    ) & 1

    return (
        indices
        ^ (
            control_bit
            << (n - 1 - target)
        )
    )


CNOT_PERM_NP = {
    (control, target): _cnot_perm_np(
        control,
        target,
    )
    for control in range(N_TOTAL)
    for target in range(N_TOTAL)
    if control != target
}

CNOT_PERM = {
    key: jnp.array(value)
    for key, value
    in CNOT_PERM_NP.items()
}


def _apply_cnot(
    rho,
    control,
    target,
):
    permutation = CNOT_PERM[
        (control, target)
    ]

    return jnp.take(
        jnp.take(
            rho,
            permutation,
            axis=1,
        ),
        permutation,
        axis=2,
    )


def _mix1(rho, q):
    batch = rho.shape[0]
    left, right = _SPLIT1[q]

    tensor = rho.reshape(
        batch,
        left,
        2,
        right,
        left,
        2,
        right,
    )

    traced = jnp.trace(
        tensor,
        axis1=2,
        axis2=5,
    )

    return jnp.einsum(
        "bimjn,xy->bixmjyn",
        traced,
        I2J,
    ).reshape(
        batch,
        TOT_DIM,
        TOT_DIM,
    )


_SPLIT2 = {}

for _control in range(N_TOTAL):
    for _target in range(N_TOTAL):
        if _control == _target:
            continue

        _q1, _q2 = sorted(
            (_control, _target)
        )

        _SPLIT2[(_control, _target)] = (
            2 ** _q1,
            2 ** (_q2 - _q1 - 1),
            2 ** (N_TOTAL - _q2 - 1),
        )


def _mix2(
    rho,
    control,
    target,
):
    batch = rho.shape[0]

    left, middle, right = _SPLIT2[
        (control, target)
    ]

    tensor = rho.reshape(
        batch,
        left,
        2,
        middle,
        2,
        right,
        left,
        2,
        middle,
        2,
        right,
    )

    tensor = jnp.trace(
        tensor,
        axis1=2,
        axis2=7,
    )

    tensor = jnp.trace(
        tensor,
        axis1=3,
        axis2=7,
    )

    return jnp.einsum(
        "bikpjlq,xy,zw->bixkzpjylwq",
        tensor,
        I2J,
        I2J,
    ).reshape(
        batch,
        TOT_DIM,
        TOT_DIM,
    )


def depol1(rho, q, p):
    return (
        (1.0 - 4.0 * p / 3.0) * rho
        + (2.0 * p / 3.0)
        * _mix1(rho, q)
    )


def depol2(
    rho,
    control,
    target,
    p,
):
    return (
        (1.0 - 16.0 * p / 15.0) * rho
        + (4.0 * p / 15.0)
        * _mix2(
            rho,
            control,
            target,
        )
    )


def _embed(
    pairs,
    n_total=N_TOTAL,
):
    return kron_list(
        [
            pairs.get(q, I2)
            for q in range(n_total)
        ]
    )


def validate_depol_channels(
    p=0.137,
    seed=0,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    matrix = (
        rng.normal(size=(TOT_DIM, TOT_DIM))
        + 1j
        * rng.normal(size=(TOT_DIM, TOT_DIM))
    )

    rho = matrix @ matrix.conj().T
    rho /= np.trace(rho).real

    rho_j = jnp.array(rho[None])
    worst = 0.0

    for q in (0, 3, 7):
        kraus = [
            np.sqrt(1 - p)
            * np.eye(TOT_DIM)
        ]

        kraus += [
            np.sqrt(p / 3)
            * _embed({q: pauli})
            for pauli in (X_, Y_, Z_)
        ]

        reference = sum(
            operator
            @ rho
            @ operator.conj().T
            for operator in kraus
        )

        result = np.asarray(
            depol1(
                rho_j,
                q,
                p,
            )
        )[0]

        worst = max(
            worst,
            float(
                np.max(
                    np.abs(
                        result - reference
                    )
                )
            ),
        )

    paulis = [I2, X_, Y_, Z_]

    for control, target in (
        (0, 1),
        (2, 7),
        (6, 1),
    ):
        kraus = [
            np.sqrt(1 - p)
            * np.eye(TOT_DIM)
        ]

        for i, first_pauli in enumerate(paulis):
            for j, second_pauli in enumerate(paulis):
                if i == 0 and j == 0:
                    continue

                kraus.append(
                    np.sqrt(p / 15)
                    * _embed(
                        {
                            control: first_pauli,
                            target: second_pauli,
                        }
                    )
                )

        reference = sum(
            operator
            @ rho
            @ operator.conj().T
            for operator in kraus
        )

        result = np.asarray(
            depol2(
                rho_j,
                control,
                target,
                p,
            )
        )[0]

        worst = max(
            worst,
            float(
                np.max(
                    np.abs(
                        result - reference
                    )
                )
            ),
        )

    trace_error_1 = float(
        abs(
            np.trace(
                np.asarray(
                    depol1(
                        rho_j,
                        3,
                        p,
                    )
                )[0]
            ).real
            - 1.0
        )
    )

    trace_error_2 = float(
        abs(
            np.trace(
                np.asarray(
                    depol2(
                        rho_j,
                        0,
                        5,
                        p,
                    )
                )[0]
            ).real
            - 1.0
        )
    )

    zero_error_1 = float(
        np.max(
            np.abs(
                np.asarray(
                    depol1(
                        rho_j,
                        3,
                        0.0,
                    )
                )[0]
                - rho
            )
        )
    )

    zero_error_2 = float(
        np.max(
            np.abs(
                np.asarray(
                    depol2(
                        rho_j,
                        0,
                        5,
                        0.0,
                    )
                )[0]
                - rho
            )
        )
    )

    if verbose:
        print(
            "  [validate] internal depol vs Kraus     : "
            f"{worst:.3e}"
        )
        print(
            "  [validate] trace preservation          : "
            f"{max(trace_error_1, trace_error_2):.3e}"
        )
        print(
            "  [validate] depol(p=0) == identity      : "
            f"{max(zero_error_1, zero_error_2):.3e}"
        )

    assert worst < 1e-12
    assert max(
        trace_error_1,
        trace_error_2,
    ) < 1e-12
    assert max(
        zero_error_1,
        zero_error_2,
    ) < 1e-15

    return worst


def _ptrace_system(rho):
    reshaped = rho.reshape(
        -1,
        SYS_DIM,
        ANC_DIM,
        SYS_DIM,
        ANC_DIM,
    )

    return jnp.trace(
        reshaped,
        axis1=1,
        axis2=3,
    )


def _ptrace_ancilla(rho):
    reshaped = rho.reshape(
        -1,
        SYS_DIM,
        ANC_DIM,
        SYS_DIM,
        ANC_DIM,
    )

    return jnp.trace(
        reshaped,
        axis1=2,
        axis2=4,
    )


def _kron_batch_sys_anc(
    system_matrix,
    ancilla_batch,
):
    result = jnp.einsum(
        "ij,bkl->bikjl",
        system_matrix,
        ancilla_batch,
    )

    return result.reshape(
        -1,
        TOT_DIM,
        TOT_DIM,
    )


def _kron_sysbatch_anc(
    system_batch,
    ancilla_matrix,
):
    result = jnp.einsum(
        "bij,kl->bikjl",
        system_batch,
        ancilla_matrix,
    )

    return result.reshape(
        -1,
        TOT_DIM,
        TOT_DIM,
    )


# ============================================================================
# Fused-ZYZ post-training pruning
# ============================================================================

def _dev_from_trace(cosine_value):
    cosine_value = float(
        np.clip(
            cosine_value,
            -1.0,
            1.0,
        )
    )

    return float(
        np.sqrt(
            max(
                0.0,
                2.0 - 2.0 * cosine_value,
            )
        )
    )


def rot_dev(theta):
    return _dev_from_trace(
        np.cos(float(theta) / 2.0)
    )


def block_dev(
    beta,
    gamma,
    delta,
):
    return _dev_from_trace(
        np.cos(float(gamma) / 2.0)
        * np.cos(
            (
                float(beta)
                + float(delta)
            ) / 2.0
        )
    )


def block_prunable(
    beta,
    gamma,
    delta,
    tol_block=PRUNE_TOL_BLOCK,
    require_all=None,
):
    if (
        block_dev(
            beta,
            gamma,
            delta,
        )
        >= tol_block
    ):
        return False

    require_all = (
        PRUNE_REQUIRE_ALL_ANGLES
        if require_all is None
        else require_all
    )

    if require_all:
        return max(
            rot_dev(beta),
            rot_dev(gamma),
            rot_dev(delta),
        ) < tol_block

    return True


def block_param_idx(
    n_layers,
    n_total=N_TOTAL,
):
    index = 2 * n_total
    rows = []

    for _ in range(n_layers):
        for control in range(n_total):
            for target in range(n_total):
                if control == target:
                    continue

                rows.append(
                    (
                        index,
                        index + 1,
                        index + 2,
                    )
                )
                index += 3

        index += 2 * n_total

    assert index == count_params(
        n_layers,
        n_total,
    )

    return np.asarray(
        rows,
        dtype=int,
    )


def _theta_of(params, terms):
    return float(
        _eval_terms(
            params,
            terms,
        )
    )


def _simplify(
    ops,
    params,
    tol_rot,
):
    changed = True

    while changed:
        changed = False
        result = []
        index = 0
        n_ops = len(ops)

        while index < n_ops:
            first = ops[index]

            second = (
                ops[index + 1]
                if index + 1 < n_ops
                else None
            )

            if (
                second is not None
                and first[0] == "CNOT"
                and second[0] == "CNOT"
                and first[1] == second[1]
                and first[2] == second[2]
            ):
                index += 2
                changed = True
                continue

            if (
                second is not None
                and first[0] == "ROT"
                and second[0] == "ROT"
                and first[1] == second[1]
                and first[2] == second[2]
            ):
                result.append(
                    (
                        "ROT",
                        first[1],
                        first[2],
                        tuple(first[3])
                        + tuple(second[3]),
                    )
                )

                index += 2
                changed = True
                continue

            if (
                first[0] == "ROT"
                and rot_dev(
                    _theta_of(
                        params,
                        first[3],
                    )
                ) < tol_rot
            ):
                index += 1
                changed = True
                continue

            result.append(first)
            index += 1

        ops = result

    return ops


def build_pruned_schedule(
    params,
    n_layers,
    n_total=N_TOTAL,
    tol_rot=PRUNE_TOL_ROT,
    tol_block=PRUNE_TOL_BLOCK,
    simplify=True,
    require_all_angles=None,
):
    """Build a pruned fused-ZYZ deployment schedule."""

    params = np.asarray(
        params,
        float,
    ).ravel()

    assert params.size == count_params(
        n_layers,
        n_total,
    )

    ops = []
    index = 0

    n_blocks = 0
    n_blocks_kept = 0
    n_blocks_cancel = 0

    for q in range(n_total):
        ops.append(
            (
                "ROT",
                "Y",
                q,
                ((index, 1.0),),
            )
        )
        index += 1

        ops.append(
            (
                "ROT",
                "Z",
                q,
                ((index, 1.0),),
            )
        )
        index += 1

    for _ in range(n_layers):
        for control in range(n_total):
            for target in range(n_total):
                if control == target:
                    continue

                beta = index
                gamma = index + 1
                delta = index + 2
                index += 3

                n_blocks += 1

                if block_prunable(
                    params[beta],
                    params[gamma],
                    params[delta],
                    tol_block,
                    require_all_angles,
                ):
                    if max(
                        rot_dev(params[beta]),
                        rot_dev(params[gamma]),
                        rot_dev(params[delta]),
                    ) >= tol_block:
                        n_blocks_cancel += 1

                    continue

                n_blocks_kept += 1

                _append_compiled_cu(
                    ops,
                    control,
                    target,
                    beta_terms=(
                        (beta, 1.0),
                    ),
                    gamma_terms=(
                        (gamma, 1.0),
                    ),
                    delta_terms=(
                        (delta, 1.0),
                    ),
                )

        for q in range(n_total):
            ops.append(
                (
                    "ROT",
                    "Y",
                    q,
                    ((index, 1.0),),
                )
            )
            index += 1

            ops.append(
                (
                    "ROT",
                    "Z",
                    q,
                    ((index, 1.0),),
                )
            )
            index += 1

    assert index == count_params(
        n_layers,
        n_total,
    )

    if simplify:
        ops = _simplify(
            ops,
            params,
            tol_rot,
        )

    n_rot, n_cnot, n_rz = gate_counts(ops)

    full_ops = build_schedule(
        n_layers,
        n_total,
    )

    full_rot, full_cnot, full_rz = gate_counts(
        full_ops
    )

    used_parameters = sorted(
        {
            parameter_index
            for op in ops
            if op[0] == "ROT"
            for parameter_index, _
            in op[3]
            if parameter_index is not None
        }
    )

    statistics = {
        "n_layers": int(n_layers),
        "tol_rot": float(tol_rot),
        "tol_block": float(tol_block),
        "n_blocks": int(n_blocks),
        "n_blocks_kept": int(n_blocks_kept),
        "n_blocks_cancel": int(n_blocks_cancel),
        "n_rot_half": int(n_rot),
        "n_cnot_half": int(n_cnot),
        "n_rz_half": int(n_rz),
        "n_rot_half_full": int(full_rot),
        "n_cnot_half_full": int(full_cnot),
        "n_rz_half_full": int(full_rz),
        "n_params_active": len(used_parameters),
        "n_params_total": int(
            count_params(
                n_layers,
                n_total,
            )
        ),
    }

    return ops, statistics


def prune_is_trivial(statistics):
    return (
        statistics["n_cnot_half"]
        == statistics["n_cnot_half_full"]
        and statistics["n_rot_half"]
        == statistics["n_rot_half_full"]
    )


def sparsify_params(params, ops):
    sparse = np.asarray(
        params,
        float,
    ).ravel().copy()

    used = {
        parameter_index
        for op in ops
        if op[0] == "ROT"
        for parameter_index, _
        in op[3]
        if parameter_index is not None
    }

    mask = np.ones(
        sparse.size,
        dtype=bool,
    )

    if used:
        mask[sorted(used)] = False

    sparse[mask] = 0.0

    return sparse, int(mask.sum())


def schedule_unitary(params, ops):
    params = np.asarray(
        params,
        float,
    )

    unitary = np.eye(
        TOT_DIM,
        dtype=complex,
    )

    for op in ops:
        if op[0] == "ROT":
            _, axis, qubit, terms = op

            theta = float(
                _eval_terms(
                    params,
                    terms,
                )
            )

            gate = (
                np.cos(theta / 2)
                * np.eye(2)
                - 1j
                * np.sin(theta / 2)
                * (
                    Y_
                    if axis == "Y"
                    else Z_
                )
            )

            left = 2 ** qubit
            right = 2 ** (
                N_TOTAL - qubit - 1
            )

            unitary = np.einsum(
                "yx,ixjc->iyjc",
                gate,
                unitary.reshape(
                    left,
                    2,
                    right,
                    TOT_DIM,
                ),
            ).reshape(
                TOT_DIM,
                TOT_DIM,
            )

        else:
            unitary = unitary[
                CNOT_PERM_NP[
                    (op[1], op[2])
                ],
                :,
            ]

    return unitary


def validate_pruning(
    params,
    n_layers,
    tol_rot=PRUNE_TOL_ROT,
    tol_block=PRUNE_TOL_BLOCK,
    verbose=True,
    require_all_angles=None,
):
    params = np.asarray(
        params,
        float,
    ).ravel()

    pruned_ops, statistics = (
        build_pruned_schedule(
            params,
            n_layers,
            tol_rot=tol_rot,
            tol_block=tol_block,
            require_all_angles=(
                require_all_angles
            ),
        )
    )

    sparse_params, n_zeroed = sparsify_params(
        params,
        pruned_ops,
    )

    full_unitary = schedule_unitary(
        params,
        build_schedule(n_layers),
    )

    pruned_unitary = schedule_unitary(
        sparse_params,
        pruned_ops,
    )

    deviation = float(
        np.linalg.norm(
            full_unitary - pruned_unitary,
            2,
        )
    )

    second_ops, _ = build_pruned_schedule(
        sparse_params,
        n_layers,
        tol_rot=tol_rot,
        tol_block=tol_block,
        require_all_angles=(
            require_all_angles
        ),
    )

    if gate_counts(second_ops) != gate_counts(
        pruned_ops
    ):
        print(
            "  [prune] WARNING: pruning is not "
            "idempotent"
        )

    statistics["unitary_dev"] = deviation
    statistics["n_params_zeroed"] = n_zeroed

    if verbose:
        print(
            f"  [prune] L={n_layers}: blocks "
            f"{statistics['n_blocks_kept']}/"
            f"{statistics['n_blocks']} "
            f"(cancel-only "
            f"{statistics['n_blocks_cancel']}) | "
            f"CNOT/half "
            f"{statistics['n_cnot_half']}/"
            f"{statistics['n_cnot_half_full']} | "
            f"ROT/half "
            f"{statistics['n_rot_half']}/"
            f"{statistics['n_rot_half_full']} | "
            f"active params "
            f"{statistics['n_params_active']}/"
            f"{statistics['n_params_total']} | "
            f"||U_pruned-U_full||_2="
            f"{deviation:.3e}"
        )

    return (
        deviation,
        pruned_ops,
        statistics,
        sparse_params,
    )


def prune_forward_tol(
    statistics=None,
    safety=20.0,
    floor=1e-11,
):
    if statistics is None:
        return 1e-6

    deviation = statistics.get(
        "unitary_dev"
    )

    if deviation is None:
        n_removed_blocks = (
            statistics["n_blocks"]
            - statistics["n_blocks_kept"]
        )

        n_removed_rotations = (
            statistics["n_rot_half_full"]
            - statistics["n_rot_half"]
        )

        deviation = (
            n_removed_blocks
            * statistics["tol_block"]
            + n_removed_rotations
            * statistics["tol_rot"]
        )

    return max(
        floor,
        safety * float(deviation),
    )


def param_sparsity_report(
    params,
    n_layers,
    n_total=N_TOTAL,
    tols=(
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
    ),
    verbose=True,
):
    params = np.asarray(
        params,
        float,
    ).ravel()

    single_indices = []
    block_indices = []

    index = 0

    for _q in range(n_total):
        single_indices += [
            index,
            index + 1,
        ]
        index += 2

    for _ in range(n_layers):
        for control in range(n_total):
            for target in range(n_total):
                if control == target:
                    continue

                block_indices.append(
                    (
                        index,
                        index + 1,
                        index + 2,
                    )
                )
                index += 3

        for _q in range(n_total):
            single_indices += [
                index,
                index + 1,
            ]
            index += 2

    single_deviations = np.array(
        [
            rot_dev(params[i])
            for i in single_indices
        ]
    )

    block_deviations = np.array(
        [
            block_dev(
                params[beta],
                params[gamma],
                params[delta],
            )
            for beta, gamma, delta
            in block_indices
        ]
    )

    rows = [
        (
            tolerance,
            int(
                (
                    single_deviations
                    < tolerance
                ).sum()
            ),
            int(
                (
                    block_deviations
                    < tolerance
                ).sum()
            ),
        )
        for tolerance in tols
    ]

    if verbose:
        print(
            f"    [sparsity] L={n_layers}: "
            "spectral distance to identity"
        )
        print(
            f"      {'tol':>8} "
            f"{'1q rots ~ I':>17} "
            f"{'C-U blocks ~ I':>19}"
        )

        for tolerance, n_single, n_blocks in rows:
            print(
                f"      {tolerance:>8.0e} "
                f"{n_single:>10}/"
                f"{len(single_deviations):<6} "
                f"{n_blocks:>10}/"
                f"{len(block_deviations):<8}"
            )

    return {
        "dev_single": single_deviations,
        "dev_block": block_deviations,
        "rows": rows,
    }


def prune_tol_scan(
    params,
    n_layers,
    tols=(
        1e-5,
        1e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
    ),
    verbose=True,
):
    params = np.asarray(
        params,
        float,
    ).ravel()

    full_unitary = schedule_unitary(
        params,
        build_schedule(n_layers),
    )

    results = []

    for tolerance in tols:
        ops, statistics = build_pruned_schedule(
            params,
            n_layers,
            tol_rot=tolerance / 10.0,
            tol_block=tolerance,
        )

        sparse_params, _ = sparsify_params(
            params,
            ops,
        )

        deviation = float(
            np.linalg.norm(
                full_unitary
                - schedule_unitary(
                    sparse_params,
                    ops,
                ),
                2,
            )
        )

        c_eff = effective_C(
            2 * statistics["n_cnot_half"],
            2 * statistics["n_rot_half"],
            2 * statistics["n_rz_half"],
        )

        results.append(
            {
                "tol": tolerance,
                "cnot": 2
                * statistics["n_cnot_half"],
                "rot": 2
                * statistics["n_rot_half"],
                "C_eff": c_eff,
                "dev": deviation,
                "blocks": statistics[
                    "n_blocks_kept"
                ],
                "blocks_all": statistics[
                    "n_blocks"
                ],
            }
        )

    if verbose:
        print(
            f"    [prune scan] {'tol':>8} "
            f"{'blocks':>12} {'CNOT':>6} "
            f"{'ROT':>6} {'C_eff':>7} "
            f"{'||dU||_2':>10}"
        )

        for result in results:
            print(
                f"                 "
                f"{result['tol']:>8.0e} "
                f"{result['blocks']:>5}/"
                f"{result['blocks_all']:<6} "
                f"{result['cnot']:>6} "
                f"{result['rot']:>6} "
                f"{result['C_eff']:>7.0f} "
                f"{result['dev']:>10.2e}"
            )

    return results


# ============================================================================
# Noisy QAE
# ============================================================================

def _run_ops(
    rho,
    params,
    ops,
    p1q,
    p2q,
    noisy_rz,
    with_noise,
):
    for op in ops:
        if op[0] == "ROT":
            _, axis, qubit, terms = op

            theta = _eval_terms(
                params,
                terms,
            )

            rho = _apply_1q(
                rho,
                _rot2(axis, theta),
                qubit,
            )

            if (
                with_noise
                and (
                    axis != "Z"
                    or noisy_rz
                )
            ):
                rho = depol1(
                    rho,
                    qubit,
                    p1q,
                )

        else:
            _, control, target = op

            rho = _apply_cnot(
                rho,
                control,
                target,
            )

            if with_noise:
                rho = depol2(
                    rho,
                    control,
                    target,
                    p2q,
                )

    return rho


def _make_runner(
    ops,
    noisy_rz,
    with_noise,
    n_groups,
):
    if len(ops) == 0:
        return (
            lambda rho, params, p1q, p2q: rho
        )

    def make_subfunction(sub_ops):
        def run_subgroup(
            rho,
            params,
            p1q,
            p2q,
        ):
            return _run_ops(
                rho,
                params,
                sub_ops,
                p1q,
                p2q,
                noisy_rz,
                with_noise,
            )

        return run_subgroup

    if (
        n_groups is None
        or n_groups <= 1
    ):
        return make_subfunction(ops)

    n_groups = int(
        min(
            n_groups,
            len(ops),
        )
    )

    groups = np.array_split(
        np.arange(len(ops)),
        n_groups,
    )

    functions = [
        jax.checkpoint(
            make_subfunction(
                [
                    ops[i]
                    for i in group
                ]
            )
        )
        for group in groups
        if len(group)
    ]

    def run(
        rho,
        params,
        p1q,
        p2q,
    ):
        for function in functions:
            rho = function(
                rho,
                params,
                p1q,
                p2q,
            )

        return rho

    return run


def make_noisy_qae(
    n_layers,
    with_noise=True,
    noisy_rz=NOISY_RZ,
    noisy_reset=NOISY_RESET,
    p_reset=P_RESET,
    remat_groups=None,
    prune_params=None,
    prune_tol_rot=PRUNE_TOL_ROT,
    prune_tol_block=PRUNE_TOL_BLOCK,
    prune_require_all_angles=None,
    parameterization="fused_zyz",
    verbose=False,
):
    if parameterization == "cr_native":
        if prune_params is not None:
            raise ValueError(
                "Native-CR training does not support "
                "in-training pruning. Convert the trained "
                "parameters to fused ZYZ and prune only the "
                "deployment copy."
            )

        forward_ops = build_cr_native_schedule(
            n_layers
        )

        prune_statistics = None

    elif parameterization == "fused_zyz":
        if prune_params is None:
            forward_ops = build_schedule(
                n_layers
            )

            prune_statistics = None

        else:
            (
                forward_ops,
                prune_statistics,
            ) = build_pruned_schedule(
                prune_params,
                n_layers,
                tol_rot=prune_tol_rot,
                tol_block=prune_tol_block,
                require_all_angles=(
                    prune_require_all_angles
                ),
            )

    else:
        raise ValueError(
            "Unknown parameterization: "
            f"{parameterization}"
        )

    backward_ops = adjoint_schedule(
        forward_ops
    )

    run_forward = _make_runner(
        forward_ops,
        noisy_rz,
        with_noise,
        remat_groups,
    )

    run_backward = _make_runner(
        backward_ops,
        noisy_rz,
        with_noise,
        remat_groups,
    )

    def _forward(
        params,
        rho_code,
        p1q,
        p2q,
    ):
        rho = _kron_sysbatch_anc(
            rho_code.astype(complex),
            ANC_0_J,
        )

        rho = run_forward(
            rho,
            params,
            p1q,
            p2q,
        )

        rho_ancilla = _ptrace_system(rho)

        rho = _kron_batch_sys_anc(
            FRESH_0_J,
            rho_ancilla,
        )

        if (
            with_noise
            and noisy_reset
            and p_reset > 0
        ):
            for q in range(N_CODE):
                rho = depol1(
                    rho,
                    q,
                    p_reset,
                )

        rho = run_backward(
            rho,
            params,
            p1q,
            p2q,
        )

        return _ptrace_ancilla(rho)

    qae_batch = jax.jit(_forward)

    @jax.jit
    def protocol_metrics(
        params,
        psi_b,
        target_b,
        p_i,
        p1q,
        p2q,
    ):
        rho_code = jnp.einsum(
            "bi,bj->bij",
            psi_b,
            jnp.conj(psi_b),
        )

        noisy_input = depol_code_j(
            rho_code,
            p_i,
        )

        output = _forward(
            params,
            noisy_input,
            p1q,
            p2q,
        )

        output_after_second_interval = (
            depol_code_j(
                output,
                p_i,
            )
        )

        recovery = jnp.einsum(
            "sij,bj->bsi",
            CORR_OPS_J,
            psi_b,
        )

        probability = jnp.real(
            jnp.einsum(
                "bsi,bij,bsj->b",
                jnp.conj(recovery),
                output_after_second_interval,
                recovery,
            )
        )

        raw_fidelity = jnp.real(
            jnp.einsum(
                "bi,bij,bj->b",
                jnp.conj(psi_b),
                output,
                psi_b,
            )
        )

        stabilizers = jnp.real(
            jnp.einsum(
                "kij,bji->bk",
                S_STACK,
                output,
            )
        )

        logicals = jnp.real(
            jnp.einsum(
                "kij,bji->bk",
                L_STACK,
                output,
            )
        )

        trace = jnp.real(
            jnp.einsum(
                "bii->b",
                output,
            )
        )

        return (
            probability,
            raw_fidelity,
            jnp.mean(
                stabilizers,
                axis=1,
            ),
            jnp.abs(
                target_b[:, 0]
                - logicals[:, 0]
            ),
            jnp.abs(
                target_b[:, 1]
                - logicals[:, 1]
            ),
            jnp.abs(
                target_b[:, 2]
                - logicals[:, 2]
            ),
            trace,
        )

    @jax.jit
    def loss_and_grad(
        params,
        psi_b,
        target_b,
        p_i,
        p1q,
        p2q,
        ws,
        wz,
        wx,
        wy,
    ):

        def loss_function(current_params):
            rho_code = jnp.einsum(
                "bi,bj->bij",
                psi_b,
                jnp.conj(psi_b),
            )

            output = _forward(
                current_params,
                depol_code_j(
                    rho_code,
                    p_i,
                ),
                p1q,
                p2q,
            )

            stabilizers = jnp.real(
                jnp.einsum(
                    "kij,bji->bk",
                    S_STACK,
                    output,
                )
            )

            logicals = jnp.real(
                jnp.einsum(
                    "kij,bji->bk",
                    L_STACK,
                    output,
                )
            )

            stabilizer_loss = -jnp.sum(
                stabilizers,
                axis=1,
            )

            logical_loss = (
                wz
                * (
                    target_b[:, 0]
                    - logicals[:, 0]
                ) ** 2
                + wx
                * (
                    target_b[:, 1]
                    - logicals[:, 1]
                ) ** 2
                + wy
                * (
                    target_b[:, 2]
                    - logicals[:, 2]
                ) ** 2
            )

            per_state_loss = (
                ws * stabilizer_loss
                + logical_loss
            )

            return jnp.mean(
                per_state_loss
            )

        return jax.value_and_grad(
            loss_function
        )(params)

    n_rot, n_cnot, n_rz = gate_counts(
        forward_ops
    )

    c_eff = effective_C(
        2 * n_cnot,
        2 * n_rot,
        2 * n_rz,
        P1Q_RATIO,
        noisy_rz,
    )

    if verbose:
        prune_tag = (
            ""
            if prune_statistics is None
            else (
                " | PRUNED blocks "
                f"{prune_statistics['n_blocks_kept']}/"
                f"{prune_statistics['n_blocks']}"
            )
        )

        noise_tag = (
            ""
            if with_noise
            else " [NOISELESS GRAPH]"
        )

        print(
            f"    L={n_layers}: "
            f"{count_params(n_layers)} params | "
            f"parameterization={parameterization} | "
            f"full QAE {2 * n_cnot} CNOT + "
            f"{2 * n_rot} rot "
            f"({2 * n_rz} RZ) | "
            f"C_eff={c_eff:.0f} -> "
            f"lambda(1e-4)="
            f"{np.exp(-c_eff * 1e-4):.4f}"
            f"{prune_tag}{noise_tag}"
        )

    build = {
        "n_layers": int(n_layers),
        "with_noise": bool(with_noise),
        "noisy_rz": bool(noisy_rz),
        "noisy_reset": bool(noisy_reset),
        "p_reset": float(p_reset),
        "remat_groups": remat_groups,
        "prune_params": (
            None
            if prune_params is None
            else np.asarray(
                prune_params,
                float,
            ).ravel().copy()
        ),
        "prune_tol_rot": float(
            prune_tol_rot
        ),
        "prune_tol_block": float(
            prune_tol_block
        ),
        "prune_require_all_angles": (
            prune_require_all_angles
        ),
        "parameterization": parameterization,
    }

    return {
        "qae_batch": qae_batch,
        "protocol_metrics": protocol_metrics,
        "loss_and_grad": loss_and_grad,
        "n_params": count_params(n_layers),
        "n_layers": n_layers,
        "n_rot_half": n_rot,
        "n_cnot_half": n_cnot,
        "n_rz_half": n_rz,
        "C_eff": c_eff,
        "prune": prune_statistics,
        "with_noise": with_noise,
        "ops_half": forward_ops,
        "build": build,
        "parameterization": parameterization,
    }


# ============================================================================
# Noise-off and pruning validation
# ============================================================================

def schedules_match(model_a, model_b):
    return (
        model_a["ops_half"]
        == model_b["ops_half"]
    )


def make_noise_off_twin(
    model,
    verbose=False,
):
    arguments = dict(model["build"])

    arguments["with_noise"] = False
    arguments["remat_groups"] = None

    return make_noisy_qae(
        verbose=verbose,
        **arguments,
    )


def forward_dev(
    model_a,
    model_b,
    params_a,
    params_b,
    psi_batch,
    p1q=0.0,
    p2q=0.0,
    chunk=8,
):
    psi = np.asarray(
        psi_batch,
        complex,
    ).reshape(
        -1,
        SYS_DIM,
    )

    psi_j = jnp.array(psi)

    params_a = jnp.array(
        np.asarray(
            params_a,
            float,
        ).ravel()
    )

    params_b = jnp.array(
        np.asarray(
            params_b,
            float,
        ).ravel()
    )

    p1q_j = jnp.float64(p1q)
    p2q_j = jnp.float64(p2q)

    worst = 0.0

    n_chunks = int(
        np.ceil(
            psi.shape[0] / chunk
        )
    )

    for index in range(n_chunks):
        sl = slice(
            index * chunk,
            min(
                (index + 1) * chunk,
                psi.shape[0],
            ),
        )

        rho = jnp.einsum(
            "bi,bj->bij",
            psi_j[sl],
            jnp.conj(psi_j[sl]),
        )

        output_a = model_a["qae_batch"](
            params_a,
            rho,
            p1q_j,
            p2q_j,
        )

        output_b = model_b["qae_batch"](
            params_b,
            rho,
            p1q_j,
            p2q_j,
        )

        worst = max(
            worst,
            float(
                jnp.max(
                    jnp.abs(
                        output_a - output_b
                    )
                )
            ),
        )

    return worst


def validate_noise_off(
    model_noisy,
    model_exact,
    params,
    psi_batch,
    params_exact=None,
    tol=1e-11,
    chunk=8,
    verbose=True,
    prune_st=None,
    label=(
        "noisy @p1q=p2q=0 vs coherent"
    ),
):
    same_schedule = schedules_match(
        model_noisy,
        model_exact,
    )

    used_tolerance = float(tol)

    if not same_schedule:
        used_tolerance = max(
            used_tolerance,
            prune_forward_tol(prune_st),
        )

        if verbose:
            print(
                "  [validate] WARNING: graphs have "
                "different schedules; treating this as "
                "a pruning comparison."
            )

    deviation = forward_dev(
        model_noisy,
        model_exact,
        params,
        (
            params
            if params_exact is None
            else params_exact
        ),
        psi_batch,
        0.0,
        0.0,
        chunk,
    )

    if verbose:
        print(
            f"  [validate] {label}: "
            f"{deviation:.2e} "
            f"(tol {used_tolerance:.1e}, "
            f"schedules "
            f"{'match' if same_schedule else 'differ'})"
        )

    assert deviation < used_tolerance, (
        "Noise-off reproduction failed: "
        f"{deviation:.3e} >= "
        f"{used_tolerance:.3e}"
    )

    return deviation


def validate_prune_forward(
    model_pruned,
    model_full,
    params_pruned,
    params_full,
    psi_batch,
    prune_st,
    p1q=0.0,
    p2q=0.0,
    safety=20.0,
    chunk=8,
    verbose=True,
):
    deviation = forward_dev(
        model_pruned,
        model_full,
        params_pruned,
        params_full,
        psi_batch,
        p1q,
        p2q,
        chunk,
    )

    tolerance = prune_forward_tol(
        prune_st,
        safety=safety,
    )

    if verbose:
        print(
            "  [validate] pruned vs full forward "
            f"map (p2q={p2q:g}): "
            f"{deviation:.2e} "
            f"(tol {tolerance:.1e})"
        )

    assert deviation < tolerance

    return deviation


# ============================================================================
# Chunked protocol evaluation
# ============================================================================

_PM_KEYS = [
    "P",
    "f_raw",
    "stab",
    "dzl",
    "dxl",
    "dyl",
    "trace",
]


def eval_protocol_chunked(
    protocol_metrics_function,
    params,
    dataset,
    p1q,
    p2q,
    chunk=20,
):
    n_states = dataset["n"]

    output = {
        key: np.empty(n_states)
        for key in _PM_KEYS
    }

    p_i_j = jnp.float64(
        dataset["p_i"]
    )

    p1q_j = jnp.float64(p1q)
    p2q_j = jnp.float64(p2q)

    n_chunks = int(
        np.ceil(n_states / chunk)
    )

    for index in range(n_chunks):
        sl = slice(
            index * chunk,
            min(
                (index + 1) * chunk,
                n_states,
            ),
        )

        result = protocol_metrics_function(
            params,
            dataset["psi"][sl],
            dataset["tgt"][sl],
            p_i_j,
            p1q_j,
            p2q_j,
        )

        for name, value in zip(
            _PM_KEYS,
            result,
        ):
            output[name][sl] = np.asarray(
                value
            )

    return output


# ============================================================================
# Native CR to fused ZYZ conversion
# ============================================================================

def _rx(theta):
    return np.array(
        [
            [
                np.cos(theta / 2),
                -1j * np.sin(theta / 2),
            ],
            [
                -1j * np.sin(theta / 2),
                np.cos(theta / 2),
            ],
        ],
        dtype=complex,
    )


def _ry(theta):
    return np.array(
        [
            [
                np.cos(theta / 2),
                -np.sin(theta / 2),
            ],
            [
                np.sin(theta / 2),
                np.cos(theta / 2),
            ],
        ],
        dtype=complex,
    )


def _rz(theta):
    return np.array(
        [
            [
                np.exp(-1j * theta / 2),
                0,
            ],
            [
                0,
                np.exp(1j * theta / 2),
            ],
        ],
        dtype=complex,
    )


def zyz_from_su2(
    unitary,
    branch_tol=1e-9,   # when to use the exact 2-angle formula
    max_err=1e-6,       # acceptance tolerance (now tight, error is ~1e-12 normally)
):
    unitary = np.asarray(unitary, dtype=complex)

    cosine = abs(unitary[0, 0])
    sine = abs(unitary[1, 0])

    gamma = 2 * np.arctan2(sine, cosine)

    if cosine > branch_tol and sine > branch_tol:
        angle_plus = np.angle(unitary[1, 1])
        angle_minus = np.angle(unitary[1, 0])
        beta = angle_plus + angle_minus
        delta = angle_plus - angle_minus
    elif cosine <= branch_tol:
        beta = 2 * np.angle(unitary[1, 0])
        delta = 0.0
    else:
        beta = 2 * np.angle(unitary[1, 1])
        delta = 0.0

    reconstructed = _rz(beta) @ _ry(gamma) @ _rz(delta)
    error = np.max(np.abs(reconstructed - unitary))

    assert error < max_err, (
        f"ZYZ extraction failed: {error:.2e}"
    )

    return float(beta), float(gamma), float(delta)


def convert_params_to_zyz(
    params_old,
    n_layers,
    n_total=N_TOTAL,
):
    params_old = np.asarray(
        params_old,
        dtype=float,
    ).ravel()

    assert params_old.size == count_params(
        n_layers,
        n_total,
    )

    result = params_old.copy()

    index = 2 * n_total

    for _ in range(n_layers):
        for control in range(n_total):
            for target in range(n_total):
                if control == target:
                    continue

                alpha_x = params_old[index]
                alpha_y = params_old[index + 1]
                alpha_z = params_old[index + 2]

                combined = (
                    _rz(alpha_z)
                    @ _ry(alpha_y)
                    @ _rx(alpha_x)
                )

                result[index:index + 3] = (
                    zyz_from_su2(combined)
                )

                index += 3

        index += 2 * n_total

    assert index == params_old.size

    return result


def convert_params_to_zyz(
    params_old,
    n_layers,
    n_total=N_TOTAL,
):
    params_old = np.asarray(
        params_old,
        dtype=float,
    ).ravel()

    assert params_old.size == count_params(
        n_layers,
        n_total,
    )

    result = params_old.copy()

    index = 2 * n_total

    for _ in range(n_layers):
        for control in range(n_total):
            for target in range(n_total):
                if control == target:
                    continue

                alpha_x = params_old[index]
                alpha_y = params_old[index + 1]
                alpha_z = params_old[index + 2]

                combined = (
                    _rz(alpha_z)
                    @ _ry(alpha_y)
                    @ _rx(alpha_x)
                )

                result[index:index + 3] = (
                    zyz_from_su2(combined)
                )

                index += 3

        index += 2 * n_total

    assert index == params_old.size

    return result


def validate_decomposition(
    params_old,
    n_layers,
    verbose=True,
    skip=False,
):
    params_old = np.asarray(
        params_old,
        float,
    ).ravel()

    params_zyz = convert_params_to_zyz(
        params_old,
        n_layers,
    )

    if skip:
        if verbose:
            print(
                "  [validate] decomposition check "
                "SKIPPED"
            )

        return 0.0, params_zyz

    native_unitary = schedule_unitary(
        params_old,
        build_cr_native_schedule(n_layers),
    )

    fused_unitary = schedule_unitary(
        params_zyz,
        build_schedule(n_layers),
    )

    error = float(
        np.max(
            np.abs(
                native_unitary
                - fused_unitary
            )
        )
    )

    if verbose:
        print(
            f"  [validate] L={n_layers}: "
            "native CR vs fused ZYZ max abs "
            f"deviation = {error:.3e}"
        )

    assert error < 1e-9, (
        "Native-to-fused decomposition failed."
    )

    return error, params_zyz


# ============================================================================
# Adaptive spectral-loss weights
# ============================================================================

W_STAB_HIGH = 5.0
W_STAB_LOW = 0.5

W_ZL_LOW = 5.0
W_ZL_HIGH = 20.0

W_XL_LOW = 5.0
W_XL_HIGH = 20.0

W_YL_LOW = 5.0
W_YL_HIGH = 20.0

STAB_SAT_LOW = 0.70
STAB_SAT_HIGH = 0.99

MAX_W_DELTA = 1.0

W_STAB_BASE = 1.0
W_ZL_BASE = 10.0
W_XL_BASE = 10.0
W_YL_BASE = 10.0


def compute_target_weights(avg_stab):
    saturation = float(
        np.clip(
            (
                avg_stab
                - STAB_SAT_LOW
            )
            / (
                STAB_SAT_HIGH
                - STAB_SAT_LOW
            ),
            0,
            1,
        )
    )

    return (
        float(
            W_STAB_HIGH
            - (
                W_STAB_HIGH
                - W_STAB_LOW
            )
            * saturation
        ),
        float(
            W_ZL_LOW
            + (
                W_ZL_HIGH
                - W_ZL_LOW
            )
            * saturation
        ),
        float(
            W_XL_LOW
            + (
                W_XL_HIGH
                - W_XL_LOW
            )
            * saturation
        ),
        float(
            W_YL_LOW
            + (
                W_YL_HIGH
                - W_YL_LOW
            )
            * saturation
        ),
    )


def step_weight(current, target):
    return (
        current
        + float(
            np.clip(
                target - current,
                -MAX_W_DELTA,
                MAX_W_DELTA,
            )
        )
    )


# ============================================================================
# Noiseless reference loading
# ============================================================================

REF_NPZ = "../steane_code/data/steane_depolar_results_layerwise.npz"
REF_VSP = "../steane_code/data/steane_depolar_fidelity_vs_p_layerwise.npz"

NOISELESS_RESULTS_NPZ = None
NOISELESS_PEXT_NPZ = None

SEARCH_DIRS = (
    ".",
    "..",
    "../steane_code/data",
    "steane_code/data",
    "../results",
)

_RES_NAMES = [
    "steane_depolar_results_layerwise.npz",
]

_VSP_NAMES = [
    "steane_depolar_fidelity_vs_p_layerwise.npz",
]

LAYERWISE_ONLY = True


def _keys(npz_file):
    return set(
        getattr(
            npz_file,
            "files",
            [],
        )
    )


def _pick(
    npz_file,
    names,
    default=None,
):
    keys = _keys(npz_file)

    for name in names:
        if name in keys:
            return npz_file[name]

    return default


def _candidate_paths(
    explicit,
    default_path,
    names,
):
    candidates = [
        path
        for path in (
            explicit,
            default_path,
        )
        if path
    ]

    if default_path:
        directory = os.path.dirname(
            default_path
        )

        candidates += [
            os.path.join(
                directory,
                name,
            )
            for name in names
        ]

    for directory in SEARCH_DIRS:
        candidates += [
            os.path.join(
                directory,
                name,
            )
            for name in names
        ]

    seen = set()
    result = []

    for path in candidates:
        normalized = os.path.normpath(path)

        if (
            LAYERWISE_ONLY
            and "layerwise"
            not in os.path.basename(normalized)
        ):
            continue

        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def _first_existing(paths):
    for path in paths:
        if os.path.isfile(path):
            return path

    return None


def autodiscover_reference(
    search_dirs=SEARCH_DIRS,
    verbose=True,
):
    results_file = None
    curve_file = None
    files = []

    for directory in search_dirs:
        files += sorted(
            glob.glob(
                os.path.join(
                    directory,
                    "*layerwise*.npz",
                )
            )
        )

    for filename in files:
        try:
            npz_file = np.load(
                filename,
                allow_pickle=True,
            )
        except Exception:
            continue

        keys = _keys(npz_file)

        has_parameters = any(
            re.fullmatch(
                r"L\d+_params",
                key,
            )
            for key in keys
        )

        has_curve = (
            "f_qae" in keys
            and bool(
                keys
                & {
                    "p_eval",
                    "p",
                    "p_noise",
                    "p_sweep",
                }
            )
        )

        if (
            has_parameters
            and results_file is None
            and "meta_layers" in keys
        ):
            results_file = filename

        if (
            has_curve
            and curve_file is None
        ):
            curve_file = filename

    if verbose:
        print(
            f"  [ref] results npz : {results_file}"
        )
        print(
            f"  [ref] vs-p npz    : {curve_file}"
        )

    return results_file, curve_file


def load_noiseless_reference(
    verbose=True,
    l_override=None,
    results_npz=None,
    pext_npz=None,
    search_dirs=None,
    strict=True,
    **_unused,
):
    del strict

    global SEARCH_DIRS

    if search_dirs is not None:
        SEARCH_DIRS = tuple(search_dirs)

    n_layers = int(
        L_TARGET
        if l_override is None
        else l_override
    )

    tried = _candidate_paths(
        results_npz
        or NOISELESS_RESULTS_NPZ,
        REF_NPZ,
        _RES_NAMES,
    )

    results_path = _first_existing(tried)

    if results_path is None:
        raise SystemExit(
            "ABORT: layerwise results npz not found.\n"
            f"  current folder = {os.getcwd()}\n"
            "  tried:\n"
            + "\n".join(
                "    " + path
                for path in tried
            )
        )

    npz_file = np.load(
        results_path,
        allow_pickle=True,
    )

    if "meta_layerwise" in npz_file.files:
        layerwise = bool(
            int(
                np.asarray(
                    npz_file["meta_layerwise"]
                ).ravel()[0]
            )
        )

        if (
            LAYERWISE_ONLY
            and not layerwise
        ):
            raise SystemExit(
                f"ABORT: {results_path} has "
                "meta_layerwise=0."
            )

    else:
        layerwise = False

        if verbose:
            print(
                "  [ref] WARNING: no "
                "'meta_layerwise' key."
            )

    parameter_key = (
        f"L{n_layers}_params"
    )

    if parameter_key not in npz_file.files:
        available = sorted(
            int(
                key[
                    1:-len("_params")
                ]
            )
            for key in npz_file.files
            if re.fullmatch(
                r"L\d+_params",
                key,
            )
        )

        raise SystemExit(
            f"ABORT: '{parameter_key}' is not "
            f"in {results_path}. Available: "
            f"{available}"
        )

    params_old = np.asarray(
        npz_file[parameter_key],
        dtype=float,
    ).ravel()

    required_count = count_params(
        n_layers
    )

    if params_old.size != required_count:
        raise SystemExit(
            f"ABORT: '{parameter_key}' holds "
            f"{params_old.size} parameters, but "
            f"L={n_layers} requires "
            f"{required_count}."
        )

    params_all = {}

    for key in npz_file.files:
        if re.fullmatch(
            r"L\d+_params",
            key,
        ):
            layer = int(
                key[
                    1:-len("_params")
                ]
            )

            value = np.asarray(
                npz_file[key],
                dtype=float,
            ).ravel()

            if value.size == count_params(layer):
                params_all[layer] = value

    f_test = {
        layer: float(
            np.asarray(
                npz_file[
                    f"L{layer}_f_qae_test"
                ]
            ).mean()
        )
        for layer in sorted(params_all)
        if (
            f"L{layer}_f_qae_test"
            in npz_file.files
        )
    }

    best_layer = (
        int(
            np.asarray(
                npz_file["best_n_layers"]
            ).ravel()[0]
        )
        if "best_n_layers" in npz_file.files
        else n_layers
    )

    p_train = (
        float(
            np.asarray(
                npz_file["meta_p_noise"]
            ).ravel()[0]
        )
        if "meta_p_noise" in npz_file.files
        else float(PEXT_TRAIN)
    )

    ket0 = _pick(
        npz_file,
        [
            "ket_0L",
            "ket0L",
            "logical_ket_0L",
        ],
    )

    ket1 = _pick(
        npz_file,
        [
            "ket_1L",
            "ket1L",
            "logical_ket_1L",
        ],
    )

    if (
        ket0 is not None
        and ket1 is not None
    ):
        set_logical_basis(
            ket0,
            ket1,
            source=(
                "loaded from "
                + os.path.basename(
                    results_path
                )
            ),
        )

    curve_path = _first_existing(
        _candidate_paths(
            pext_npz
            or NOISELESS_PEXT_NPZ,
            REF_VSP,
            _VSP_NAMES,
        )
    )

    p_ext = None
    f_curve = None
    has_curve = False
    curve_layer = None

    if curve_path is not None:
        curve_file = np.load(
            curve_path,
            allow_pickle=True,
        )

        p_values = _pick(
            curve_file,
            [
                "p_eval",
                "p",
                "p_noise",
                "p_sweep",
            ],
        )

        f_values = _pick(
            curve_file,
            ["f_qae"],
        )

        if (
            p_values is not None
            and f_values is not None
        ):
            p_ext = np.asarray(
                p_values,
                dtype=float,
            ).ravel()

            f_curve = np.asarray(
                f_values,
                dtype=float,
            ).ravel()

            has_curve = (
                p_ext.size == f_curve.size
                and p_ext.size > 1
            )

            if (
                "best_n_layers"
                in curve_file.files
            ):
                curve_layer = int(
                    np.asarray(
                        curve_file[
                            "best_n_layers"
                        ]
                    ).ravel()[0]
                )

    if p_ext is None:
        p_ext = np.array(
            [
                0.01,
                0.02,
                0.05,
                0.08,
                0.12,
                0.16,
                0.20,
                0.25,
            ]
        )

    reference = {
        "results_file": results_path,
        "pext_file": (
            curve_path
            if has_curve
            else None
        ),
        "n_layers": n_layers,
        "params_old": params_old,
        "params_old_all": params_all,
        "layer_sweep": sorted(params_all),
        "best_n_layers": best_layer,
        "layerwise": layerwise,
        "p_train": p_train,
        "p_ext": p_ext,
        "sigma": p_ext,
        "has_curve": has_curve,
        "f_noiseless_qae": (
            f_curve
            if has_curve
            else None
        ),
        "f_qae_is_target_L": bool(
            curve_layer == n_layers
        ) if curve_layer is not None else False,
        "f_test_at_train_pext": f_test,
        "f_std_test_ideal": (
            np.asarray(
                npz_file["f_std_test"],
                dtype=float,
            )
            if "f_std_test" in npz_file.files
            else None
        ),
        "f_noisy_test_ideal": (
            np.asarray(
                npz_file["f_noisy_test"],
                dtype=float,
            )
            if "f_noisy_test" in npz_file.files
            else None
        ),
    }

    if verbose:
        print(
            f"  [ref] file          : "
            f"{results_path} [LAYERWISE ONLY]"
        )
        print(
            f"  [ref] using L       : "
            f"{n_layers} "
            f"({params_old.size} parameters)"
        )
        print(
            f"  [ref] layers stored : "
            f"{reference['layer_sweep']} | "
            f"best={best_layer}"
        )
        print(
            f"  [ref] trained at p  : "
            f"{p_train}"
        )

        if f_test:
            print(
                "  [ref] ideal F(test) : "
                f"{ {k: round(v, 5) for k, v in f_test.items()} }"
            )

        print(
            "  [ref] vs-p curve    : "
            + (
                f"yes, {curve_path}"
                if has_curve
                else "no"
            )
        )

        print(
            "  [ref] logical basis : "
            f"{LOGICAL_BASIS_SOURCE}"
        )

    return reference


def grow_params(
    previous_params,
    previous_layers,
    new_layers,
):
    """Append an identity layer while preserving the original layout."""

    assert new_layers >= previous_layers

    extra = (
        count_params(new_layers)
        - count_params(previous_layers)
    )

    return np.concatenate(
        [
            np.asarray(
                previous_params,
                float,
            ),
            np.zeros(
                extra,
                dtype=float,
            ),
        ]
    )


def print_config():
    print(
        "  Steane [[7,1,3]] | "
        f"N_TOTAL={N_TOTAL} "
        "(7 code + 1 latent ancilla)"
    )

    print(
        "  external : depolarizing p_i per "
        f"half interval, train p={PEXT_TRAIN}"
    )

    print(
        "  internal : "
        f"p1q={P1Q_RATIO}*p2q, "
        f"NOISY_RZ={NOISY_RZ}, "
        f"NOISY_RESET={NOISY_RESET}, "
        f"p_reset={P_RESET}"
    )

    print(
        f"  depth    : L_TARGET={L_TARGET} -> "
        f"{count_params(L_TARGET)} parameters"
    )

    native_rot, native_cnot, native_rz = (
        gate_counts(
            build_cr_native_schedule(
                L_TARGET
            )
        )
    )

    fused_rot, fused_cnot, fused_rz = (
        gate_counts(
            build_schedule(
                L_TARGET
            )
        )
    )

    print(
        "  native training circuit : "
        f"{2 * native_cnot} CNOT + "
        f"{2 * native_rot} rotations "
        f"({2 * native_rz} RZ) | "
        f"C_eff="
        f"{effective_C(2 * native_cnot, 2 * native_rot, 2 * native_rz):.0f}"
    )

    print(
        "  fused deployment circuit: "
        f"{2 * fused_cnot} CNOT + "
        f"{2 * fused_rot} rotations "
        f"({2 * fused_rz} RZ) | "
        f"C_eff="
        f"{effective_C(2 * fused_cnot, 2 * fused_rot, 2 * fused_rz):.0f}"
    )

    print(
        "  deployment pruning: "
        f"||R-I||_2 < {PRUNE_TOL_ROT:.0e}, "
        f"||C-U-I||_2 < "
        f"{PRUNE_TOL_BLOCK:.0e}, "
        f"require_all_angles="
        f"{PRUNE_REQUIRE_ALL_ANGLES}"
    )

    print(
        f"  reference: {REF_NPZ} "
        "[LAYERWISE ONLY]"
    )
#!/usr/bin/env python3

# Quantum_amplitude_faze V4


"""
Loto 7/39 — Quantum Amplitude NEXT V4.

LOKALNA SIMULACIJA:
    Sedam pozicija, po šest kubita.
    Registri se računaju zasebno.
    Amplitudno kodiranje, RZ/RY, CNOT, Bornovo pravilo.
    Nema materijalizacije stanja od 42 kubita.

MODEL:
    Zajednička raspodela nad rastućim sedmorkama.
    Učenje raspodele analitičkim gradijentima.
    Klasični L-BFGS-B optimizator.
    Tačna normalizacija, marginale i maksimum.
    Bez slučajnog uzorkovanja, dopunjavanja ili popravljanja NEXT.

V4 IZBOR PARAMETARA:
    Završnih 15% prelaza isključeno je iz izbora.
    Prethodnih 85% deli se na:
        početni trening;
        tri uzastopna validaciona perioda.
    Trening raste kroz periode.
    Standardizacija svakog treninga uči se zasebno.
    Konfiguracija i trajanje biraju se najmanjim ponderisanim
    prosečnim NLL-om kroz sva tri validaciona perioda.
    Ako optimizator ranije konvergira, njegova završna tačka
    koristi se za veće dozvoljene budžete.
    Test i završno učenje počinju iznova.

Zavisnosti:
    numpy, scipy, qiskit==1.4.4

CSV:
    Bez zaglavlja; sedam brojeva po redu.
    Od najstarijeg do najnovijeg kola.
"""


import csv
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


PATHS = (
    "/Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv",
)

CONFIGS = ((1, 1e-4), (2, 1e-3), (2, 1e-2))

# Jednaki kandidati za trajanje u svim validacionim periodima.
BUDGETS = (25, 50, 75, 100, 150, 200,
           250, 300, 400, 500, 650, 800)

LOG_EVERY = 25
FTOL = 1e-10
GTOL = 1e-6

QUBITS = 6
DIM = 1 << QUBITS
NUMBERS = 39
POSITIONS = 7

CNOTS = (
    (0, 1), (1, 2), (2, 3),
    (3, 4), (4, 5), (5, 0),
)

BASIS = np.arange(DIM)
PERMUTATIONS = {
    pair: BASIS ^ (((BASIS >> pair[0]) & 1) << pair[1])
    for pair in CNOTS
}

SUPPORT = np.zeros((POSITIONS, NUMBERS), dtype=np.float64)
for position in range(POSITIONS):
    # Svaka pozicija zadržava svih 33 dozvoljenih vrednosti.
    SUPPORT[
        position,
        position:NUMBERS - POSITIONS + position + 1,
    ] = 1.0


def read_history(path):
    rows = []
    with Path(path).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for line, row in enumerate(csv.reader(handle), 1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != POSITIONS:
                raise ValueError(
                    f"{path}, red {line}: očekivano je 7 kolona."
                )
            try:
                values = [int(value.strip()) for value in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: očekivani su celi "
                    "brojevi bez zaglavlja."
                ) from exc
            if len(set(values)) != POSITIONS:
                raise ValueError(f"{path}, red {line}: duplikati.")
            if min(values) < 1 or max(values) > NUMBERS:
                raise ValueError(
                    f"{path}, red {line}: broj van 1–39."
                )
            rows.append(sorted(values))

    if len(rows) < 100:
        raise ValueError(f"{path}: potrebno je najmanje 100 kola.")
    return np.asarray(rows, dtype=np.int64)


def features(history):
    """Kontekst reda t koristi isključivo kola 0..t."""
    values = history.astype(np.float64)
    count = np.arange(
        1, len(history) + 1, dtype=np.float64
    )[:, None]
    running_mean = np.cumsum(values, axis=0) / count
    running_variance = np.maximum(
        np.cumsum(values * values, axis=0) / count
        - running_mean ** 2,
        0.0,
    )

    delta = np.zeros_like(values)
    delta[1:] = values[1:] - values[:-1]

    present = np.zeros((len(history), NUMBERS))
    present[
        np.arange(len(history))[:, None], history - 1
    ] = 1.0

    # Iste karakteristike kao u prethodnim verzijama.
    # 1 + 7 + 7 + 7 + 39 + 3 = 64.
    raw = np.column_stack((
        np.ones(len(history)),
        values,
        running_mean,
        delta,
        present,
        running_variance[:, [0, 3, 6]],
    ))
    if raw.shape != (len(history), DIM):
        raise ArithmeticError("Neispravan broj karakteristika.")
    if not np.isfinite(raw).all():
        raise ArithmeticError("Neispravne karakteristike.")
    return raw


class Encoder:
    def fit(self, raw):
        self.mean = raw.mean(axis=0)
        self.scale = raw.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        return self

    def transform(self, raw):
        values = (raw - self.mean) / self.scale
        values[:, 0] = 1.0
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ArithmeticError(
                "Neispravna amplitudna normalizacija."
            )
        return (values / norms).astype(np.complex128)


def rotation(kind, angle, derivative=False):
    if kind == "ry":
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        if derivative:
            return 0.5 * np.array(
                [[-s, -c], [c, -s]], dtype=np.complex128
            )
        return np.array(
            [[c, -s], [s, c]], dtype=np.complex128
        )

    if kind == "rz":
        a, b = np.exp(-0.5j * angle), np.exp(0.5j * angle)
        return np.diag(
            [-0.5j * a, 0.5j * b] if derivative else [a, b]
        )
    raise ValueError(kind)


def gate(state, matrix, qubit):
    """Qiskit little-endian konvencija."""
    stride = 1 << qubit
    source = state.reshape(len(state), -1, 2, stride)
    result = np.empty_like(source)
    result[:, :, 0, :] = (
        matrix[0, 0] * source[:, :, 0, :]
        + matrix[0, 1] * source[:, :, 1, :]
    )
    result[:, :, 1, :] = (
        matrix[1, 0] * source[:, :, 0, :]
        + matrix[1, 1] * source[:, :, 1, :]
    )
    return result.reshape(state.shape)


def evolve(encoded, angles, layers, record=False):
    state, tape, index = encoded, [], 0
    for _ in range(layers):
        for qubit in range(QUBITS):
            for kind in ("rz", "ry"):
                if record:
                    tape.append((kind, qubit, index, state))
                state = gate(
                    state, rotation(kind, angles[index]), qubit
                )
                index += 1

        for control, target in CNOTS:
            state = state[:, PERMUTATIONS[(control, target)]]
            if record:
                tape.append(("cx", control, target, None))
    return state, tape


def reverse_gradient(adjoint, tape, angles):
    gradient = np.zeros_like(angles)
    for kind, qubit, index, before in reversed(tape):
        if kind == "cx":
            adjoint = adjoint[
                :, PERMUTATIONS[(qubit, index)]
            ]
            continue

        derivative = gate(
            before,
            rotation(kind, angles[index], derivative=True),
            qubit,
        )
        gradient[index] = (
            2.0 * np.vdot(adjoint, derivative).real
        )
        adjoint = gate(
            adjoint,
            rotation(kind, angles[index]).conj().T,
            qubit,
        )
    return gradient


def initial_angles(layers):
    indices = np.arange(
        1, layers * QUBITS * 2 + 1, dtype=np.float64
    )
    one_register = 0.2 * np.sin(indices * math.sqrt(2.0))
    return np.tile(one_register, (POSITIONS, 1)).ravel()


def predict_probabilities(encoded, angles, layers):
    parameters = angles.reshape(POSITIONS, -1)
    output = []
    for position in range(POSITIONS):
        state, _ = evolve(
            encoded, parameters[position], layers
        )
        weights = (
            np.abs(state[:, :NUMBERS]) ** 2
            * SUPPORT[position]
        )
        mass = weights.sum(axis=1, keepdims=True)
        if np.any(mass <= 0) or not np.isfinite(mass).all():
            raise ArithmeticError(
                "Nulta ili neispravna masa podrške."
            )
        output.append(weights / mass)
    return np.stack(output, axis=1)


def log_probabilities(probabilities):
    if probabilities.shape[-2:] != (POSITIONS, NUMBERS):
        raise ValueError("Neispravan oblik raspodele.")
    if (
        np.any(probabilities < 0)
        or not np.isfinite(probabilities).all()
    ):
        raise ArithmeticError("Neispravne verovatnoće.")
    if not np.allclose(
        probabilities.sum(axis=-1),
        1.0, atol=2e-10, rtol=0,
    ):
        raise ArithmeticError(
            "Pozicione raspodele nisu normalizovane."
        )

    result = np.full_like(probabilities, -np.inf)
    np.log(
        probabilities, out=result, where=probabilities > 0
    )
    return result


def prefix_exclusive(values):
    result = np.full_like(values, -np.inf)
    result[..., 1:] = np.logaddexp.accumulate(
        values, axis=-1
    )[..., :-1]
    return result


def suffix_exclusive(values):
    return prefix_exclusive(values[..., ::-1])[..., ::-1]


def joint_distribution(logp, need_marginals=False):
    """
    P(c | x) = proizvod_j p_j(c_j | x) / Z(x),
    za c1 < c2 < ... < c7.
    """
    forward = np.full_like(logp, -np.inf)
    forward[..., 0, :] = logp[..., 0, :]
    for position in range(1, POSITIONS):
        forward[..., position, :] = (
            logp[..., position, :]
            + prefix_exclusive(
                forward[..., position - 1, :]
            )
        )

    log_z = np.logaddexp.reduce(
        forward[..., -1, :], axis=-1
    )
    if (
        not np.isfinite(log_z).all()
        or np.any(log_z > 2e-10)
    ):
        raise ArithmeticError(
            "Neispravna zajednička normalizacija."
        )
    if not need_marginals:
        return log_z, None

    backward = np.full_like(logp, -np.inf)
    backward[..., -1, :] = logp[..., -1, :]
    for position in range(POSITIONS - 2, -1, -1):
        backward[..., position, :] = (
            logp[..., position, :]
            + suffix_exclusive(
                backward[..., position + 1, :]
            )
        )

    reverse_z = np.logaddexp.reduce(
        backward[..., 0, :], axis=-1
    )
    if not np.allclose(
        log_z, reverse_z, atol=2e-9, rtol=0
    ):
        raise ArithmeticError(
            "Prefiks i sufiks daju različitu masu."
        )

    valid = (
        np.isfinite(forward)
        & np.isfinite(backward)
        & np.isfinite(logp)
    )
    log_marginals = np.full_like(logp, -np.inf)
    log_marginals[valid] = (
        forward[valid] + backward[valid] - logp[valid]
    )
    log_marginals -= np.asarray(log_z)[..., None, None]
    marginals = np.exp(log_marginals)

    if not np.allclose(
        marginals.sum(axis=-1),
        1.0, atol=2e-8, rtol=0,
    ):
        raise ArithmeticError(
            "Zajedničke marginale nisu normalizovane."
        )
    if np.any(
        marginals.sum(axis=-2) > 1.0 + 2e-8
    ):
        raise ArithmeticError(
            "Zajednička raspodela dopušta duplikate."
        )
    return log_z, marginals


def observed_log_weights(logp, targets):
    values = logp[
        np.arange(len(targets))[:, None],
        np.arange(POSITIONS)[None, :],
        targets - 1,
    ].sum(axis=1)
    if not np.isfinite(values).all():
        raise ArithmeticError(
            "Opažena kombinacija ima nultu masu."
        )
    return values


def mean_nll(probabilities, targets):
    logp = log_probabilities(probabilities)
    log_z, _ = joint_distribution(logp)
    return float(np.mean(
        log_z - observed_log_weights(logp, targets)
    ))


def objective(angles, encoded, targets, layers, penalty):
    probabilities = predict_probabilities(
        encoded, angles, layers
    )
    logp = log_probabilities(probabilities)
    log_z, marginals = joint_distribution(
        logp, need_marginals=True
    )

    nll = float(np.mean(
        log_z - observed_log_weights(logp, targets)
    ))
    loss = nll + penalty * float(np.mean(angles ** 2))

    residual = marginals.copy()
    residual[
        np.arange(len(targets))[:, None],
        np.arange(POSITIONS)[None, :],
        targets - 1,
    ] -= 1.0

    parameters = angles.reshape(POSITIONS, -1)
    gradient = np.empty_like(parameters)

    # U memoriji je diferencijaciona traka jednog registra.
    for position in range(POSITIONS):
        state, tape = evolve(
            encoded, parameters[position],
            layers, record=True,
        )
        weights = np.abs(state[:, :NUMBERS]) ** 2
        coefficient = np.zeros(
            state.shape, dtype=np.float64
        )
        np.divide(
            residual[:, position, :],
            weights,
            out=coefficient[:, :NUMBERS],
            where=weights > 0,
        )
        adjoint = coefficient * state / len(targets)
        gradient[position] = reverse_gradient(
            adjoint, tape, parameters[position]
        )
        del state, tape, coefficient, adjoint

    gradient = gradient.ravel()
    gradient += 2.0 * penalty * angles / angles.size

    if (
        not np.isfinite(loss)
        or not np.isfinite(gradient).all()
    ):
        raise ArithmeticError(
            "Neispravan cilj ili gradijent."
        )
    return loss, gradient


def check_gradient(encoded, targets, layers, penalty):
    encoded, targets = encoded[:5], targets[:5]
    angles = initial_angles(layers)
    _, analytic = objective(
        angles, encoded, targets, layers, penalty
    )
    width = layers * QUBITS * 2
    indices = {
        position * width + offset
        for position in range(POSITIONS)
        for offset in (0, 1, width - 1)
    }

    step = 1e-6
    for index in sorted(indices):
        plus, minus = angles.copy(), angles.copy()
        plus[index] += step
        minus[index] -= step
        numeric = (
            objective(
                plus, encoded, targets, layers, penalty
            )[0]
            - objective(
                minus, encoded, targets, layers, penalty
            )[0]
        ) / (2.0 * step)

        if not np.isclose(
            numeric, analytic[index],
            atol=3e-6, rtol=3e-4,
        ):
            raise ArithmeticError(
                f"Gradijent nije potvrđen: {index}; "
                f"analitički={analytic[index]}, "
                f"numerički={numeric}."
            )


class AmplitudeRegressor:
    def __init__(self, layers, penalty):
        self.layers = layers
        self.penalty = penalty

    def fit(self, raw, targets, budget, validation=None):
        self.encoder = Encoder().fit(raw)
        encoded = self.encoder.transform(raw)
        check_gradient(
            encoded, targets, self.layers, self.penalty
        )

        if validation is None:
            validation_encoded = None
            validation_targets = None
            requested = ()
        else:
            validation_raw, validation_targets = validation
            validation_encoded = self.encoder.transform(
                validation_raw
            )
            requested = tuple(
                value for value in BUDGETS if value <= budget
            )

        self.curve = {}
        progress = {
            "iteration": 0,
            "last_loss": None,
            "last_logged": -1,
        }
        cache = {}

        def fun(theta):
            if (
                "theta" in cache
                and np.array_equal(theta, cache["theta"])
            ):
                return (
                    cache["loss"],
                    cache["gradient"].copy(),
                )

            value, gradient = objective(
                theta, encoded, targets,
                self.layers, self.penalty,
            )
            cache["theta"] = theta.copy()
            cache["loss"] = value
            cache["gradient"] = gradient.copy()
            return value, gradient

        def validation_score(theta):
            probabilities = predict_probabilities(
                validation_encoded, theta, self.layers
            )
            return mean_nll(
                probabilities, validation_targets
            )

        def observe(theta, iteration, final=False):
            if (
                progress["last_logged"] == iteration
                and not final
            ):
                return

            value, gradient = fun(theta)
            gradient_norm = float(
                np.linalg.norm(gradient, ord=np.inf)
            )
            previous = progress["last_loss"]
            change = (
                None if previous is None else previous - value
            )
            change_text = (
                "—" if change is None else f"{change:.3e}"
            )
            text = (
                f"    Iteracija {iteration}: "
                f"cilj={value:.8f}; "
                f"|grad|inf={gradient_norm:.3e}; "
                f"pad od prethodnog ispisa={change_text}"
            )

            if validation_encoded is not None:
                if iteration in requested:
                    if iteration not in self.curve:
                        self.curve[iteration] = {
                            "nll": validation_score(theta),
                            "actual": iteration,
                        }
                    text += (
                        "; validacioni NLL="
                        f"{self.curve[iteration]['nll']:.8f}"
                    )

            print(text, flush=True)
            progress["last_loss"] = value
            progress["last_logged"] = iteration

        def callback(theta):
            progress["iteration"] += 1
            iteration = progress["iteration"]
            if (
                iteration == 1
                or iteration % LOG_EVERY == 0
                or iteration in requested
            ):
                observe(theta, iteration)

        initial = initial_angles(self.layers)
        start_loss = fun(initial)[0]
        print(
            f"    Početni cilj={start_loss:.8f}; "
            f"dozvoljeno iteracija={budget}",
            flush=True,
        )

        result = minimize(
            fun,
            initial,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": int(budget),
                "ftol": FTOL,
                "gtol": GTOL,
                "maxls": 40,
            },
        )

        self.angles = result.x.copy()
        self.actual_iterations = int(result.nit)
        self.optimizer_success = bool(result.success)

        final_loss, final_gradient = fun(self.angles)
        if (
            self.actual_iterations < 1
            or not np.isfinite(final_loss)
            or not np.isfinite(self.angles).all()
            or final_loss > start_loss + 1e-8
        ):
            raise ArithmeticError(
                "Optimizacija nije dala validan rezultat učenja."
            )

        if (
            not result.success
            and self.actual_iterations < budget
        ):
            raise RuntimeError(
                "Optimizator je prekinut pre dozvoljenog trajanja: "
                + str(result.message)
            )

        if progress["last_logged"] != self.actual_iterations:
            observe(
                self.angles, self.actual_iterations, final=True
            )

        # Ranija konvergencija predstavlja isto pravilo
        # za sve veće dozvoljene budžete.
        if validation_encoded is not None:
            missing = [
                value for value in requested
                if value not in self.curve
            ]
            if missing:
                if (
                    not result.success
                    or any(
                        value < self.actual_iterations
                        for value in missing
                    )
                ):
                    raise ArithmeticError(
                        "Nedostaju validacione tačke putanje."
                    )

                score = validation_score(self.angles)
                for value in missing:
                    self.curve[value] = {
                        "nll": score,
                        "actual": self.actual_iterations,
                    }
                print(
                    "    Ranija konvergencija: završna tačka "
                    "važi za budžete "
                    + ", ".join(map(str, missing)),
                    flush=True,
                )

        self.gradient_norm = float(
            np.linalg.norm(final_gradient, ord=np.inf)
        )
        self.training_nll = (
            final_loss
            - self.penalty
            * float(np.mean(self.angles ** 2))
        )

        if result.success:
            self.stop_reason = (
                "Ispunjen kriterijum zaustavljanja optimizatora: "
                + str(result.message)
            )
        else:
            self.stop_reason = (
                "Dostignut dozvoljeni broj iteracija; "
                "kriterijum konvergencije nije ispunjen."
            )

        print(
            f"    Zaustavljanje: {self.stop_reason}\n"
            f"    Izvršeno iteracija: {self.actual_iterations}\n"
            f"    Trening NLL: {self.training_nll:.8f}\n"
            f"    Završni |grad|inf: {self.gradient_norm:.3e}",
            flush=True,
        )
        return self

    def predict(self, raw):
        return predict_probabilities(
            self.encoder.transform(raw),
            self.angles,
            self.layers,
        )

    def check_qiskit(self, raw):
        encoded = self.encoder.transform(raw[:1])
        parameters = self.angles.reshape(POSITIONS, -1)

        for position in range(POSITIONS):
            circuit = QuantumCircuit(QUBITS)
            circuit.initialize(encoded[0], range(QUBITS))
            index = 0

            for _ in range(self.layers):
                for qubit in range(QUBITS):
                    circuit.rz(
                        float(parameters[position, index]),
                        qubit,
                    )
                    index += 1
                    circuit.ry(
                        float(parameters[position, index]),
                        qubit,
                    )
                    index += 1
                for control, target in CNOTS:
                    circuit.cx(control, target)

            reference = Statevector.from_instruction(
                circuit
            ).data
            calculated, _ = evolve(
                encoded, parameters[position], self.layers
            )

            if not np.allclose(
                reference, calculated[0],
                atol=2e-10, rtol=0,
            ):
                raise ArithmeticError(
                    f"Qiskit neslaganje: pozicija {position + 1}."
                )
            if not np.isclose(
                np.sum(np.abs(calculated[0]) ** 2),
                1.0, atol=2e-10, rtol=0,
            ):
                raise ArithmeticError(
                    "Kvantno kolo ne čuva normu."
                )


def exact_next(probabilities):
    logp = log_probabilities(probabilities)
    log_z, _ = joint_distribution(
        logp, need_marginals=True
    )
    log_z = float(log_z)

    values = logp[0].copy()
    parents = np.full(
        (POSITIONS, NUMBERS), -1, dtype=np.int64
    )

    for position in range(1, POSITIONS):
        updated = np.full(NUMBERS, -np.inf)
        best_value, best_index = -np.inf, -1

        for number in range(NUMBERS):
            previous = number - 1
            if (
                previous >= 0
                and values[previous] > best_value
            ):
                best_value = values[previous]
                best_index = previous

            if (
                best_index >= 0
                and np.isfinite(logp[position, number])
            ):
                updated[number] = (
                    best_value + logp[position, number]
                )
                parents[position, number] = best_index

        values = updated

    last = int(np.argmax(values))
    log_maximum = float(values[last])
    if not np.isfinite(log_maximum):
        raise ArithmeticError(
            "Nema maksimuma pozitivne mase."
        )

    answer = [last]
    for position in range(POSITIONS - 1, 0, -1):
        last = int(parents[position, last])
        if last < 0:
            raise ArithmeticError(
                "Neispravna putanja maksimuma."
            )
        answer.append(last)

    answer = tuple(value + 1 for value in reversed(answer))
    if (
        len(set(answer)) != POSITIONS
        or answer[0] < 1
        or answer[-1] > NUMBERS
        or any(a >= b for a, b in zip(answer, answer[1:]))
    ):
        raise ArithmeticError("Neispravna NEXT kombinacija.")

    return answer, log_maximum - log_z


def check_exact_search(probabilities):
    small = probabilities.copy()
    small[:, 9:] = 0.0
    mass = small.sum(axis=1, keepdims=True)
    if np.any(mass <= 0):
        raise ArithmeticError(
            "Nema mase za proveru male podrške."
        )
    small /= mass

    logp = log_probabilities(small)
    cases = list(
        itertools.combinations(range(9), POSITIONS)
    )
    scores = np.array([
        sum(logp[j, case[j]] for j in range(POSITIONS))
        for case in cases
    ])

    brute_z = float(np.logaddexp.reduce(scores))
    dp_z, _ = joint_distribution(
        logp, need_marginals=True
    )
    answer, log_mass = exact_next(small)
    selected = sum(
        logp[j, answer[j] - 1] for j in range(POSITIONS)
    )

    if not (
        np.isclose(
            brute_z, float(dp_z), atol=2e-9, rtol=0
        )
        and np.isclose(
            selected, scores.max(), atol=2e-9, rtol=0
        )
        and np.isclose(
            log_mass,
            scores.max() - brute_z,
            atol=2e-9,
            rtol=0,
        )
    ):
        raise ArithmeticError(
            "Provera tačnog maksimuma nije prošla."
        )


def check_boundaries():
    for answer in (
        tuple(range(1, 8)),
        tuple(range(33, 40)),
    ):
        probabilities = np.zeros((POSITIONS, NUMBERS))
        probabilities[
            np.arange(POSITIONS), np.array(answer) - 1
        ] = 1.0

        predicted, log_mass = exact_next(probabilities)
        if (
            predicted != answer
            or not np.isclose(
                log_mass, 0.0, atol=1e-12, rtol=0
            )
        ):
            raise ArithmeticError(
                "Provera graničnih kombinacija nije prošla."
            )


def joint_marginals(probabilities):
    _, marginals = joint_distribution(
        log_probabilities(probabilities),
        need_marginals=True,
    )
    return marginals


def chronological_folds(pretest_end):
    """
    Prva polovina pretest istorije je početni trening.
    Preostala polovina daje tri nepreklopljena perioda.
    """
    first = pretest_end // 2
    remaining = pretest_end - first
    edges = [
        first + remaining * index // 3
        for index in range(4)
    ]

    folds = [
        (edges[index], edges[index + 1])
        for index in range(3)
    ]
    if any(
        train_end < 20 or validation_end - train_end < 10
        for train_end, validation_end in folds
    ):
        raise ValueError(
            "Premalo podataka za tri validaciona perioda."
        )
    if folds[-1][1] != pretest_end:
        raise ArithmeticError("Neispravan završetak validacije.")
    return folds


def choose_parameters(raw, targets, pretest_end):
    folds = chronological_folds(pretest_end)
    sizes = np.array([
        validation_end - train_end
        for train_end, validation_end in folds
    ], dtype=np.float64)
    weights = sizes / sizes.sum()

    print("\nHRONOLOŠKI VALIDACIONI PERIODI", flush=True)
    for index, (train_end, validation_end) in enumerate(folds, 1):
        print(
            f"  Period {index}: trening prelazi 1–{train_end}; "
            f"validacija {train_end + 1}–{validation_end}",
            flush=True,
        )

    aggregate = []

    for layers, penalty in CONFIGS:
        curves = []

        for index, (train_end, validation_end) in enumerate(
            folds, 1
        ):
            print(
                f"\nPretraga: slojevi={layers}; "
                f"regularizacija={penalty:g}; period={index}/3",
                flush=True,
            )
            model = AmplitudeRegressor(
                layers, penalty
            ).fit(
                raw[:train_end],
                targets[:train_end],
                budget=max(BUDGETS),
                validation=(
                    raw[train_end:validation_end],
                    targets[train_end:validation_end],
                ),
            )
            curves.append(model.curve)
            del model

        print(
            f"\nZBIRNO: slojevi={layers}; "
            f"regularizacija={penalty:g}",
            flush=True,
        )
        for budget in BUDGETS:
            scores = np.array([
                curve[budget]["nll"] for curve in curves
            ])
            if not np.isfinite(scores).all():
                raise ArithmeticError(
                    "Neispravni validacioni rezultati."
                )

            mean = float(np.dot(weights, scores))
            deviation = float(np.sqrt(
                np.dot(weights, (scores - mean) ** 2)
            ))
            actual = tuple(
                curve[budget]["actual"] for curve in curves
            )

            item = {
                "layers": layers,
                "penalty": penalty,
                "budget": budget,
                "mean": mean,
                "deviation": deviation,
                "scores": tuple(float(value) for value in scores),
                "actual": actual,
            }
            aggregate.append(item)

            print(
                f"  Budžet={budget:3d}; "
                f"NLL perioda="
                + "/".join(f"{value:.6f}" for value in scores)
                + f"; prosek={mean:.8f}; "
                + f"rasipanje={deviation:.6f}; "
                + f"stvarne iteracije={actual}",
                flush=True,
            )

    # Glavni kriterijum je prosečan NLL.
    # Ostali elementi razrešavaju samo potpuno jednaku vrednost.
    best = min(
        aggregate,
        key=lambda item: (
            item["mean"],
            item["deviation"],
            item["budget"],
            item["layers"],
            item["penalty"],
        ),
    )

    print(
        "\nIZABRANO PREKO TRI VALIDACIONA PERIODA:\n"
        f"  Slojevi: {best['layers']}\n"
        f"  Regularizacija: {best['penalty']:g}\n"
        f"  Dozvoljeno trajanje: {best['budget']} iteracija\n"
        f"  Prosečni validacioni NLL: {best['mean']:.8f}\n"
        f"  Rasipanje NLL-a između perioda: "
        f"{best['deviation']:.8f}\n"
        "  Pojedinačni NLL: "
        + ", ".join(f"{value:.8f}" for value in best["scores"]),
        flush=True,
    )
    return best


def run_file(path):
    history = read_history(path)
    raw = features(history)

    # Svako kolo osim poslednjeg ima svoj istorijski prelaz.
    x, targets = raw[:-1], history[1:]
    transitions = len(targets)
    pretest_end = int(0.85 * transitions)

    print(f"\nV4 — FAJL: {path}", flush=True)
    print(
        f"Kola: {len(history)}; prelazi: {transitions}; "
        f"pretest: {pretest_end}; "
        f"odvojeni test: {transitions - pretest_end}",
        flush=True,
    )

    choice = choose_parameters(
        x, targets, pretest_end
    )
    layers = choice["layers"]
    penalty = choice["penalty"]
    budget = choice["budget"]

    # Izbor je završen pre računanja rezultata testa.
    print(
        "\nUČENJE ZA ODVOJENI TEST "
        "NA SVIM PRETEST PRELAZIMA",
        flush=True,
    )
    test_model = AmplitudeRegressor(
        layers, penalty
    ).fit(
        x[:pretest_end],
        targets[:pretest_end],
        budget=budget,
    )
    test_probabilities = test_model.predict(
        x[pretest_end:]
    )
    test_nll = mean_nll(
        test_probabilities, targets[pretest_end:]
    )
    test_iterations = test_model.actual_iterations

    print(
        f"Odvojeni test — NLL: {test_nll:.8f}",
        flush=True,
    )
    del test_model, test_probabilities

    # Završni model dobija sve istorijske prelaze.
    # Počinje iznova sa standardizacijom cele trening istorije.
    print(
        "\nZAVRŠNO UČENJE NA CELOJ ISTORIJI",
        flush=True,
    )
    final_model = AmplitudeRegressor(
        layers, penalty
    ).fit(x, targets, budget=budget)

    final_model.check_qiskit(raw[-1:])
    probabilities = final_model.predict(raw[-1:])[0]

    check_exact_search(probabilities)
    answer, log_mass = exact_next(probabilities)
    marginals = joint_marginals(probabilities)

    previous = final_model.predict(raw[-2:-1])[0]
    previous_marginals = joint_marginals(previous)
    context_change = float(np.max(np.abs(
        marginals - previous_marginals
    )))

    if context_change < 1e-12:
        print(
            "PROVERA: gotovo jednake zajedničke marginale "
            "za poslednja dva istorijska konteksta.",
            flush=True,
        )

    if any(
        all(answer[start + k] == answer[start] + k for k in range(5))
        for start in range(POSITIONS - 4)
    ):
        print(
            "PROVERA: najmanje pet uzastopnih brojeva u NEXT. "
            "Rezultat nije filtriran; pregledati ponašanje modela.",
            flush=True,
        )

    if answer[-1] - answer[0] <= 10:
        print(
            "PROVERA: svih sedam NEXT brojeva je u rasponu <=10. "
            "Ovo je dijagnostička oznaka; rezultat nije menjan.",
            flush=True,
        )

    repeats = int(np.sum(np.all(
        history == np.asarray(answer), axis=1
    )))
    if repeats:
        print(
            f"PROVERA: NEXT postoji u istoriji {repeats} puta. "
            "Rezultat nije menjan.",
            flush=True,
        )

    print(
        "\nNEXT V4: " + ", ".join(map(str, answer))
        + f"\nLogaritamska masa NEXT u modelu: {log_mass:.8f}"
        + "\nPromena zajedničkih marginala prema prethodnom "
        + f"kontekstu: {context_change:.10g}"
        + "\nQiskit, gradijenti, normalizacija, granice i maksimum: "
        + "ugrađene provere prošle."
        + "\nMasa modela nije izmerena uspešnost predikcije.",
        flush=True,
    )

    return {
        "path": path,
        "answer": answer,
        "marginals": marginals,
        "context_change": context_change,
        "history_hash": hashlib.sha256(
            history.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "cv_nll": choice["mean"],
        "cv_deviation": choice["deviation"],
        "cv_scores": choice["scores"],
        "test_nll": test_nll,
        "layers": layers,
        "penalty": penalty,
        "budget": budget,
        "test_iterations": test_iterations,
        "final_iterations": final_model.actual_iterations,
        "final_gradient": final_model.gradient_norm,
        "final_stop": final_model.stop_reason,
    }


def main():
    np.seterr(
        over="raise", invalid="raise", divide="raise"
    )
    check_boundaries()
    results = [run_file(path) for path in PATHS]

    print("\nKONAČNI NEXT REZULTATI — V4", flush=True)
    for item in results:
        print(
            f"{Path(item['path']).name}: "
            + ", ".join(map(str, item["answer"])),
            flush=True,
        )

    print(
        "\nVALIDACIJA I ODVOJENI TEST — manji NLL je bolji",
        flush=True,
    )
    for item in results:
        print(
            f"{Path(item['path']).name}:\n"
            f"  prosečni validacioni NLL={item['cv_nll']:.8f}\n"
            f"  rasipanje između perioda="
            f"{item['cv_deviation']:.8f}\n"
            f"  pojedinačni NLL={item['cv_scores']}\n"
            f"  test NLL={item['test_nll']:.8f}\n"
            f"  slojevi={item['layers']}; "
            f"regularizacija={item['penalty']:g}\n"
            f"  izabrani budžet={item['budget']}; "
            f"test izvršio={item['test_iterations']}; "
            f"završni izvršio={item['final_iterations']}\n"
            f"  završni |grad|inf={item['final_gradient']:.3e}\n"
            f"  završno zaustavljanje: {item['final_stop']}",
            flush=True,
        )

    for index, first in enumerate(results):
        for second in results[index + 1:]:
            if first["answer"] != second["answer"]:
                continue

            difference = float(np.max(np.abs(
                first["marginals"] - second["marginals"]
            )))
            same_history = (
                first["history_hash"] == second["history_hash"]
            )

            print(
                "\nPROVERA ISTIH NEXT:\n"
                f"  {Path(first['path']).name}\n"
                f"  {Path(second['path']).name}\n"
                f"  Iste istorije: {same_history}\n"
                f"  Najveća razlika zajedničkih marginala: "
                f"{difference:.10g}\n"
                f"  Promene konteksta: "
                f"{first['context_change']:.10g}, "
                f"{second['context_change']:.10g}\n"
                "  Modeli su učeni zasebno. Rezultati nisu menjani "
                "radi različitosti. Ove provere same ne isključuju "
                "degeneraciju modela.",
                flush=True,
            )


if __name__ == "__main__":
    main()



"""
V4 — FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv
Kola: 4708; prelazi: 4707; pretest: 4000; odvojeni test: 707

HRONOLOŠKI VALIDACIONI PERIODI
  Period 1: trening prelazi 1–2000; validacija 2001–2666
  Period 2: trening prelazi 1–2666; validacija 2667–3333
  Period 3: trening prelazi 1–3333; validacija 3334–4000

Pretraga: slojevi=1; regularizacija=0.0001; period=1/3
    Početni cilj=23.52709407; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.63987024; |grad|inf=3.483e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.62777987; |grad|inf=2.795e-01; pad od prethodnog ispisa=1.012e+00; validacioni NLL=21.92627897
    Iteracija 50: cilj=21.33812210; |grad|inf=6.540e-02; pad od prethodnog ispisa=2.897e-01; validacioni NLL=21.65377613
    Iteracija 75: cilj=20.89045918; |grad|inf=1.015e-01; pad od prethodnog ispisa=4.477e-01; validacioni NLL=21.08458642
    Iteracija 100: cilj=20.61851442; |grad|inf=3.275e-02; pad od prethodnog ispisa=2.719e-01; validacioni NLL=20.95296480
    Iteracija 125: cilj=20.10711460; |grad|inf=9.604e-02; pad od prethodnog ispisa=5.114e-01
    Iteracija 150: cilj=19.75963146; |grad|inf=3.473e-02; pad od prethodnog ispisa=3.475e-01; validacioni NLL=20.20609363
    Iteracija 175: cilj=19.63556195; |grad|inf=6.029e-02; pad od prethodnog ispisa=1.241e-01
    Iteracija 200: cilj=19.58774106; |grad|inf=5.111e-03; pad od prethodnog ispisa=4.782e-02; validacioni NLL=20.13895172
    Iteracija 225: cilj=19.56677941; |grad|inf=6.614e-03; pad od prethodnog ispisa=2.096e-02
    Iteracija 250: cilj=19.54810111; |grad|inf=3.416e-03; pad od prethodnog ispisa=1.868e-02; validacioni NLL=20.17869794
    Iteracija 275: cilj=19.54579136; |grad|inf=2.139e-03; pad od prethodnog ispisa=2.310e-03
    Iteracija 300: cilj=19.54567077; |grad|inf=3.441e-04; pad od prethodnog ispisa=1.206e-04; validacioni NLL=20.17189437
    Iteracija 325: cilj=19.53240885; |grad|inf=6.067e-03; pad od prethodnog ispisa=1.326e-02
    Iteracija 350: cilj=19.52826158; |grad|inf=3.508e-03; pad od prethodnog ispisa=4.147e-03
    Iteracija 375: cilj=19.52259578; |grad|inf=2.662e-03; pad od prethodnog ispisa=5.666e-03
    Iteracija 400: cilj=19.52226953; |grad|inf=1.386e-04; pad od prethodnog ispisa=3.263e-04; validacioni NLL=20.16799179
    Iteracija 425: cilj=19.52226644; |grad|inf=9.660e-05; pad od prethodnog ispisa=3.091e-06
    Iteracija 450: cilj=19.52225883; |grad|inf=4.057e-05; pad od prethodnog ispisa=7.606e-06
    Iteracija 469: cilj=19.52225858; |grad|inf=1.153e-05; pad od prethodnog ispisa=2.468e-07
    Ranija konvergencija: završna tačka važi za budžete 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 469
    Trening NLL: 19.52211894
    Završni |grad|inf: 1.153e-05

Pretraga: slojevi=1; regularizacija=0.0001; period=2/3
    Početni cilj=23.50121620; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.69077398; |grad|inf=3.571e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.54951838; |grad|inf=1.654e-01; pad od prethodnog ispisa=1.141e+00; validacioni NLL=21.79115070
    Iteracija 50: cilj=20.78360709; |grad|inf=3.449e-02; pad od prethodnog ispisa=7.659e-01; validacioni NLL=20.91825965
    Iteracija 75: cilj=20.66107405; |grad|inf=2.145e-02; pad od prethodnog ispisa=1.225e-01; validacioni NLL=20.82394130
    Iteracija 100: cilj=20.53224649; |grad|inf=4.485e-02; pad od prethodnog ispisa=1.288e-01; validacioni NLL=20.74905234
    Iteracija 125: cilj=20.32636679; |grad|inf=1.027e-01; pad od prethodnog ispisa=2.059e-01
    Iteracija 150: cilj=19.91941566; |grad|inf=4.431e-02; pad od prethodnog ispisa=4.070e-01; validacioni NLL=20.16746952
    Iteracija 175: cilj=19.83979774; |grad|inf=1.747e-02; pad od prethodnog ispisa=7.962e-02
    Iteracija 200: cilj=19.61044822; |grad|inf=2.436e-02; pad od prethodnog ispisa=2.293e-01; validacioni NLL=19.95114641
    Iteracija 225: cilj=19.51104160; |grad|inf=3.445e-02; pad od prethodnog ispisa=9.941e-02
    Iteracija 250: cilj=19.49438911; |grad|inf=2.075e-02; pad od prethodnog ispisa=1.665e-02; validacioni NLL=19.83661577
    Iteracija 275: cilj=19.46984793; |grad|inf=4.532e-03; pad od prethodnog ispisa=2.454e-02
    Iteracija 300: cilj=19.46948388; |grad|inf=3.887e-04; pad od prethodnog ispisa=3.640e-04; validacioni NLL=19.86484758
    Iteracija 325: cilj=19.46943449; |grad|inf=2.261e-04; pad od prethodnog ispisa=4.939e-05
    Iteracija 350: cilj=19.46940671; |grad|inf=3.287e-04; pad od prethodnog ispisa=2.778e-05
    Iteracija 375: cilj=19.46921269; |grad|inf=3.181e-04; pad od prethodnog ispisa=1.940e-04
    Iteracija 400: cilj=19.46912067; |grad|inf=3.457e-04; pad od prethodnog ispisa=9.202e-05; validacioni NLL=19.86553116
    Iteracija 425: cilj=19.46911046; |grad|inf=6.687e-05; pad od prethodnog ispisa=1.022e-05
    Iteracija 450: cilj=19.46910623; |grad|inf=1.154e-04; pad od prethodnog ispisa=4.226e-06
    Iteracija 475: cilj=19.46910567; |grad|inf=3.743e-05; pad od prethodnog ispisa=5.576e-07
    Iteracija 491: cilj=19.46910559; |grad|inf=1.378e-05; pad od prethodnog ispisa=8.021e-08
    Ranija konvergencija: završna tačka važi za budžete 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 491
    Trening NLL: 19.46894908
    Završni |grad|inf: 1.378e-05

Pretraga: slojevi=1; regularizacija=0.0001; period=3/3
    Početni cilj=23.53881449; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.01574747; |grad|inf=9.153e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.24109571; |grad|inf=2.877e-01; pad od prethodnog ispisa=1.775e+00; validacioni NLL=21.04045300
    Iteracija 50: cilj=21.01064881; |grad|inf=2.456e-02; pad od prethodnog ispisa=2.304e-01; validacioni NLL=20.88648456
    Iteracija 75: cilj=20.63207643; |grad|inf=1.424e-02; pad od prethodnog ispisa=3.786e-01; validacioni NLL=20.57212904
    Iteracija 100: cilj=20.15164329; |grad|inf=5.716e-02; pad od prethodnog ispisa=4.804e-01; validacioni NLL=20.30951623
    Iteracija 125: cilj=20.02671129; |grad|inf=5.800e-02; pad od prethodnog ispisa=1.249e-01
    Iteracija 150: cilj=19.98739891; |grad|inf=4.830e-03; pad od prethodnog ispisa=3.931e-02; validacioni NLL=20.07410866
    Iteracija 175: cilj=19.96438288; |grad|inf=4.807e-02; pad od prethodnog ispisa=2.302e-02
    Iteracija 200: cilj=19.71789497; |grad|inf=2.591e-02; pad od prethodnog ispisa=2.465e-01; validacioni NLL=20.04600449
    Iteracija 225: cilj=19.63144183; |grad|inf=5.794e-03; pad od prethodnog ispisa=8.645e-02
    Iteracija 250: cilj=19.57695378; |grad|inf=8.055e-03; pad od prethodnog ispisa=5.449e-02; validacioni NLL=19.86256278
    Iteracija 275: cilj=19.56747956; |grad|inf=6.402e-03; pad od prethodnog ispisa=9.474e-03
    Iteracija 300: cilj=19.56213666; |grad|inf=1.884e-03; pad od prethodnog ispisa=5.343e-03; validacioni NLL=19.84566354
    Iteracija 325: cilj=19.55961034; |grad|inf=2.751e-03; pad od prethodnog ispisa=2.526e-03
    Iteracija 350: cilj=19.55886677; |grad|inf=4.852e-04; pad od prethodnog ispisa=7.436e-04
    Iteracija 375: cilj=19.55875438; |grad|inf=4.195e-04; pad od prethodnog ispisa=1.124e-04
    Iteracija 400: cilj=19.55870900; |grad|inf=2.490e-04; pad od prethodnog ispisa=4.538e-05; validacioni NLL=19.87325187
    Iteracija 425: cilj=19.55870491; |grad|inf=6.786e-05; pad od prethodnog ispisa=4.090e-06
    Iteracija 450: cilj=19.55870328; |grad|inf=8.977e-05; pad od prethodnog ispisa=1.634e-06
    Iteracija 475: cilj=19.55870289; |grad|inf=1.051e-05; pad od prethodnog ispisa=3.849e-07
    Iteracija 482: cilj=19.55870287; |grad|inf=3.378e-05; pad od prethodnog ispisa=1.799e-08
    Ranija konvergencija: završna tačka važi za budžete 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 482
    Trening NLL: 19.55853041
    Završni |grad|inf: 3.378e-05

ZBIRNO: slojevi=1; regularizacija=0.0001
  Budžet= 25; NLL perioda=21.926279/21.791151/21.040453; prosek=21.58579073; rasipanje=0.389680; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=21.653776/20.918260/20.886485; prosek=21.15258964; rasipanje=0.354364; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=21.084586/20.823941/20.572129; prosek=20.82675673; rasipanje=0.209193; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.952965/20.749052/20.309516; prosek=20.67036990; rasipanje=0.268486; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.206094/20.167470/20.074109; prosek=20.14919550; rasipanje=0.055405; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.138952/19.951146/20.046004; prosek=20.04532075; rasipanje=0.076663; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.178698/19.836616/19.862563; prosek=19.95918246; rasipanje=0.155466; stvarne iteracije=(250, 250, 250)
  Budžet=300; NLL perioda=20.171894/19.864848/19.845664; prosek=19.96069628; rasipanje=0.149433; stvarne iteracije=(300, 300, 300)
  Budžet=400; NLL perioda=20.167992/19.865531/19.873252; prosek=19.96882541; rasipanje=0.140762; stvarne iteracije=(400, 400, 400)
  Budžet=500; NLL perioda=20.167178/19.865606/19.873118; prosek=19.96853460; rasipanje=0.140390; stvarne iteracije=(469, 491, 482)
  Budžet=650; NLL perioda=20.167178/19.865606/19.873118; prosek=19.96853460; rasipanje=0.140390; stvarne iteracije=(469, 491, 482)
  Budžet=800; NLL perioda=20.167178/19.865606/19.873118; prosek=19.96853460; rasipanje=0.140390; stvarne iteracije=(469, 491, 482)

Pretraga: slojevi=2; regularizacija=0.001; period=1/3
    Početni cilj=23.81491146; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.87941794; |grad|inf=6.600e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.89860646; |grad|inf=2.079e-01; pad od prethodnog ispisa=1.981e+00; validacioni NLL=21.33173425
    Iteracija 50: cilj=19.27653688; |grad|inf=2.741e-02; pad od prethodnog ispisa=1.622e+00; validacioni NLL=20.21368026
    Iteracija 75: cilj=19.16577979; |grad|inf=3.507e-02; pad od prethodnog ispisa=1.108e-01; validacioni NLL=20.24056244
    Iteracija 100: cilj=19.06884865; |grad|inf=2.027e-02; pad od prethodnog ispisa=9.693e-02; validacioni NLL=20.30646922
    Iteracija 125: cilj=19.02226801; |grad|inf=1.487e-02; pad od prethodnog ispisa=4.658e-02
    Iteracija 150: cilj=18.99751085; |grad|inf=7.699e-03; pad od prethodnog ispisa=2.476e-02; validacioni NLL=20.40348215
    Iteracija 175: cilj=18.98709557; |grad|inf=1.982e-02; pad od prethodnog ispisa=1.042e-02
    Iteracija 200: cilj=18.98410215; |grad|inf=6.654e-05; pad od prethodnog ispisa=2.993e-03; validacioni NLL=20.34676570
    Iteracija 214: cilj=18.98410191; |grad|inf=1.871e-05; pad od prethodnog ispisa=2.323e-07
    Ranija konvergencija: završna tačka važi za budžete 250, 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 214
    Trening NLL: 18.98272022
    Završni |grad|inf: 1.871e-05

Pretraga: slojevi=2; regularizacija=0.001; period=2/3
    Početni cilj=23.88428746; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.90741011; |grad|inf=4.949e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.09102413; |grad|inf=2.157e-01; pad od prethodnog ispisa=1.816e+00; validacioni NLL=21.27979996
    Iteracija 50: cilj=19.55499612; |grad|inf=3.606e-02; pad od prethodnog ispisa=1.536e+00; validacioni NLL=20.34434121
    Iteracija 75: cilj=19.41098647; |grad|inf=1.551e-02; pad od prethodnog ispisa=1.440e-01; validacioni NLL=20.35755881
    Iteracija 100: cilj=19.34725932; |grad|inf=1.633e-02; pad od prethodnog ispisa=6.373e-02; validacioni NLL=20.44148725
    Iteracija 125: cilj=19.26393307; |grad|inf=1.420e-02; pad od prethodnog ispisa=8.333e-02
    Iteracija 150: cilj=19.22835442; |grad|inf=1.257e-02; pad od prethodnog ispisa=3.558e-02; validacioni NLL=20.33859558
    Iteracija 175: cilj=19.20711269; |grad|inf=1.086e-03; pad od prethodnog ispisa=2.124e-02
    Iteracija 200: cilj=19.19701339; |grad|inf=1.665e-02; pad od prethodnog ispisa=1.010e-02; validacioni NLL=20.37484289
    Iteracija 225: cilj=19.16061451; |grad|inf=8.892e-03; pad od prethodnog ispisa=3.640e-02
    Iteracija 250: cilj=19.12054305; |grad|inf=1.408e-02; pad od prethodnog ispisa=4.007e-02; validacioni NLL=20.41023123
    Iteracija 275: cilj=19.11028187; |grad|inf=9.239e-03; pad od prethodnog ispisa=1.026e-02
    Iteracija 300: cilj=19.09989751; |grad|inf=1.367e-02; pad od prethodnog ispisa=1.038e-02; validacioni NLL=20.39484319
    Iteracija 325: cilj=19.09834169; |grad|inf=3.120e-05; pad od prethodnog ispisa=1.556e-03
    Iteracija 331: cilj=19.09834163; |grad|inf=1.828e-05; pad od prethodnog ispisa=5.692e-08
    Ranija konvergencija: završna tačka važi za budžete 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 331
    Trening NLL: 19.09709491
    Završni |grad|inf: 1.828e-05

Pretraga: slojevi=2; regularizacija=0.001; period=3/3
    Početni cilj=23.95154872; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.37386634; |grad|inf=4.108e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.48240458; |grad|inf=1.507e-01; pad od prethodnog ispisa=2.891e+00; validacioni NLL=20.75763016
    Iteracija 50: cilj=19.55998495; |grad|inf=3.001e-02; pad od prethodnog ispisa=9.224e-01; validacioni NLL=20.09955714
    Iteracija 75: cilj=19.40528379; |grad|inf=2.222e-02; pad od prethodnog ispisa=1.547e-01; validacioni NLL=20.07479782
    Iteracija 100: cilj=19.35461987; |grad|inf=2.002e-02; pad od prethodnog ispisa=5.066e-02; validacioni NLL=20.12184306
    Iteracija 125: cilj=19.31503869; |grad|inf=2.360e-02; pad od prethodnog ispisa=3.958e-02
    Iteracija 150: cilj=19.29877929; |grad|inf=5.251e-03; pad od prethodnog ispisa=1.626e-02; validacioni NLL=20.18119182
    Iteracija 175: cilj=19.26564897; |grad|inf=3.349e-03; pad od prethodnog ispisa=3.313e-02
    Iteracija 200: cilj=19.26496956; |grad|inf=6.865e-04; pad od prethodnog ispisa=6.794e-04; validacioni NLL=20.24936192
    Iteracija 225: cilj=19.26343986; |grad|inf=6.517e-03; pad od prethodnog ispisa=1.530e-03
    Iteracija 250: cilj=19.25845146; |grad|inf=8.688e-03; pad od prethodnog ispisa=4.988e-03; validacioni NLL=20.23699259
    Iteracija 275: cilj=19.24258737; |grad|inf=1.048e-03; pad od prethodnog ispisa=1.586e-02
    Iteracija 300: cilj=19.23285093; |grad|inf=1.115e-02; pad od prethodnog ispisa=9.736e-03; validacioni NLL=20.30008665
    Iteracija 325: cilj=19.22661422; |grad|inf=3.202e-03; pad od prethodnog ispisa=6.237e-03
    Iteracija 350: cilj=19.22598531; |grad|inf=1.607e-03; pad od prethodnog ispisa=6.289e-04
    Iteracija 375: cilj=19.22561543; |grad|inf=1.047e-04; pad od prethodnog ispisa=3.699e-04
    Iteracija 400: cilj=19.22561419; |grad|inf=1.605e-05; pad od prethodnog ispisa=1.241e-06; validacioni NLL=20.32278417
    Iteracija 405: cilj=19.22561416; |grad|inf=9.681e-06; pad od prethodnog ispisa=2.665e-08
    Ranija konvergencija: završna tačka važi za budžete 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 405
    Trening NLL: 19.22482391
    Završni |grad|inf: 9.681e-06

ZBIRNO: slojevi=2; regularizacija=0.001
  Budžet= 25; NLL perioda=21.331734/21.279800/20.757630; prosek=21.12295045; rasipanje=0.259285; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=20.213680/20.344341/20.099557; prosek=20.21919563; rasipanje=0.100034; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=20.240562/20.357559/20.074798; prosek=20.22429823; rasipanje=0.116036; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.306469/20.441487/20.121843; prosek=20.28992491; rasipanje=0.131049; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.403482/20.338596/20.181192; prosek=20.30770866; rasipanje=0.093332; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.346766/20.374843/20.249362; prosek=20.32364528; rasipanje=0.053781; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.346742/20.410231/20.236993; prosek=20.33131413; rasipanje=0.071577; stvarne iteracije=(214, 250, 250)
  Budžet=300; NLL perioda=20.346742/20.394843/20.300087; prosek=20.34722408; rasipanje=0.038695; stvarne iteracije=(214, 300, 300)
  Budžet=400; NLL perioda=20.346742/20.405716/20.322784; prosek=20.35841980; rasipanje=0.034856; stvarne iteracije=(214, 331, 400)
  Budžet=500; NLL perioda=20.346742/20.405716/20.322775; prosek=20.35841689; rasipanje=0.034859; stvarne iteracije=(214, 331, 405)
  Budžet=650; NLL perioda=20.346742/20.405716/20.322775; prosek=20.35841689; rasipanje=0.034859; stvarne iteracije=(214, 331, 405)
  Budžet=800; NLL perioda=20.346742/20.405716/20.322775; prosek=20.35841689; rasipanje=0.034859; stvarne iteracije=(214, 331, 405)

Pretraga: slojevi=2; regularizacija=0.01; period=1/3
    Početni cilj=23.81509452; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.87955299; |grad|inf=6.601e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.79757079; |grad|inf=1.417e-01; pad od prethodnog ispisa=2.082e+00; validacioni NLL=21.08016451
    Iteracija 50: cilj=19.31444820; |grad|inf=7.351e-02; pad od prethodnog ispisa=1.483e+00; validacioni NLL=19.98604301
    Iteracija 75: cilj=19.18251892; |grad|inf=8.458e-03; pad od prethodnog ispisa=1.319e-01; validacioni NLL=20.01302379
    Iteracija 100: cilj=19.11457226; |grad|inf=2.061e-02; pad od prethodnog ispisa=6.795e-02; validacioni NLL=20.12010326
    Iteracija 125: cilj=19.05143621; |grad|inf=2.905e-02; pad od prethodnog ispisa=6.314e-02
    Iteracija 150: cilj=19.02658110; |grad|inf=7.175e-03; pad od prethodnog ispisa=2.486e-02; validacioni NLL=20.09011759
    Iteracija 175: cilj=19.00673903; |grad|inf=1.085e-02; pad od prethodnog ispisa=1.984e-02
    Iteracija 200: cilj=18.99921012; |grad|inf=7.495e-04; pad od prethodnog ispisa=7.529e-03; validacioni NLL=20.07251341
    Iteracija 225: cilj=18.99755585; |grad|inf=3.262e-03; pad od prethodnog ispisa=1.654e-03
    Iteracija 250: cilj=18.99731570; |grad|inf=7.587e-05; pad od prethodnog ispisa=2.402e-04; validacioni NLL=20.05873570
    Iteracija 275: cilj=18.99731448; |grad|inf=1.367e-05; pad od prethodnog ispisa=1.215e-06
    Iteracija 280: cilj=18.99731446; |grad|inf=1.766e-05; pad od prethodnog ispisa=1.655e-08
    Ranija konvergencija: završna tačka važi za budžete 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 280
    Trening NLL: 18.98856332
    Završni |grad|inf: 1.766e-05

Pretraga: slojevi=2; regularizacija=0.01; period=2/3
    Početni cilj=23.88447053; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.90753631; |grad|inf=4.949e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.73346644; |grad|inf=3.072e-01; pad od prethodnog ispisa=2.174e+00; validacioni NLL=21.08518083
    Iteracija 50: cilj=19.55583911; |grad|inf=1.450e-01; pad od prethodnog ispisa=1.178e+00; validacioni NLL=20.40305535
    Iteracija 75: cilj=19.39391562; |grad|inf=1.632e-02; pad od prethodnog ispisa=1.619e-01; validacioni NLL=20.38463596
    Iteracija 100: cilj=19.24734121; |grad|inf=3.846e-02; pad od prethodnog ispisa=1.466e-01; validacioni NLL=20.48638724
    Iteracija 125: cilj=19.18194493; |grad|inf=2.445e-02; pad od prethodnog ispisa=6.540e-02
    Iteracija 150: cilj=19.15125498; |grad|inf=6.317e-03; pad od prethodnog ispisa=3.069e-02; validacioni NLL=20.54761721
    Iteracija 175: cilj=19.13323997; |grad|inf=2.252e-02; pad od prethodnog ispisa=1.802e-02
    Iteracija 200: cilj=19.11573177; |grad|inf=3.674e-03; pad od prethodnog ispisa=1.751e-02; validacioni NLL=20.51961899
    Iteracija 225: cilj=19.11436582; |grad|inf=3.411e-04; pad od prethodnog ispisa=1.366e-03
    Iteracija 250: cilj=19.11427079; |grad|inf=1.141e-04; pad od prethodnog ispisa=9.503e-05; validacioni NLL=20.53134185
    Iteracija 275: cilj=19.11426484; |grad|inf=2.313e-05; pad od prethodnog ispisa=5.944e-06
    Iteracija 281: cilj=19.11426483; |grad|inf=4.882e-06; pad od prethodnog ispisa=1.615e-08
    Ranija konvergencija: završna tačka važi za budžete 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 281
    Trening NLL: 19.10651939
    Završni |grad|inf: 4.882e-06

Pretraga: slojevi=2; regularizacija=0.01; period=3/3
    Početni cilj=23.95173178; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.37406037; |grad|inf=4.109e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.16585990; |grad|inf=9.192e-02; pad od prethodnog ispisa=3.208e+00; validacioni NLL=20.60432631
    Iteracija 50: cilj=19.55904779; |grad|inf=5.065e-02; pad od prethodnog ispisa=6.068e-01; validacioni NLL=20.26332349
    Iteracija 75: cilj=19.37601610; |grad|inf=2.075e-02; pad od prethodnog ispisa=1.830e-01; validacioni NLL=20.15286757
    Iteracija 100: cilj=19.32979297; |grad|inf=6.278e-03; pad od prethodnog ispisa=4.622e-02; validacioni NLL=20.20222082
    Iteracija 125: cilj=19.29318395; |grad|inf=1.321e-02; pad od prethodnog ispisa=3.661e-02
    Iteracija 150: cilj=19.25911527; |grad|inf=7.046e-03; pad od prethodnog ispisa=3.407e-02; validacioni NLL=20.25941109
    Iteracija 175: cilj=19.23835559; |grad|inf=3.698e-03; pad od prethodnog ispisa=2.076e-02
    Iteracija 200: cilj=19.21347537; |grad|inf=1.187e-02; pad od prethodnog ispisa=2.488e-02; validacioni NLL=20.22701967
    Iteracija 225: cilj=19.20355644; |grad|inf=2.662e-03; pad od prethodnog ispisa=9.919e-03
    Iteracija 250: cilj=19.19947152; |grad|inf=1.224e-03; pad od prethodnog ispisa=4.085e-03; validacioni NLL=20.20251387
    Iteracija 275: cilj=19.18986410; |grad|inf=8.825e-03; pad od prethodnog ispisa=9.607e-03
    Iteracija 300: cilj=19.18658037; |grad|inf=2.926e-04; pad od prethodnog ispisa=3.284e-03; validacioni NLL=20.16262788
    Iteracija 325: cilj=19.18100010; |grad|inf=5.600e-03; pad od prethodnog ispisa=5.580e-03
    Iteracija 350: cilj=19.17387048; |grad|inf=8.138e-04; pad od prethodnog ispisa=7.130e-03
    Iteracija 375: cilj=19.17332053; |grad|inf=5.132e-05; pad od prethodnog ispisa=5.500e-04
    Iteracija 400: cilj=19.17331941; |grad|inf=1.149e-05; pad od prethodnog ispisa=1.115e-06; validacioni NLL=20.13555125
    Iteracija 402: cilj=19.17331941; |grad|inf=6.713e-06; pad od prethodnog ispisa=4.977e-09
    Ranija konvergencija: završna tačka važi za budžete 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 402
    Trening NLL: 19.16356164
    Završni |grad|inf: 6.713e-06

ZBIRNO: slojevi=2; regularizacija=0.01
  Budžet= 25; NLL perioda=21.080165/21.085181/20.604326; prosek=20.92314541; rasipanje=0.225533; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=19.986043/20.403055/20.263323; prosek=20.21758966; rasipanje=0.173270; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=20.013024/20.384636/20.152868; prosek=20.18359435; rasipanje=0.153240; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.120103/20.486387/20.202221; prosek=20.26964518; rasipanje=0.156939; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.090118/20.547617/20.259411; prosek=20.29915310; rasipanje=0.188854; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.072513/20.519619/20.227020; prosek=20.27315096; rasipanje=0.185402; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.058736/20.531342/20.202514; prosek=20.26429987; rasipanje=0.197805; stvarne iteracije=(250, 250, 250)
  Budžet=300; NLL perioda=20.058881/20.531953/20.162628; prosek=20.25125005; rasipanje=0.203027; stvarne iteracije=(280, 281, 300)
  Budžet=400; NLL perioda=20.058881/20.531953/20.135551; prosek=20.24221999; rasipanje=0.207325; stvarne iteracije=(280, 281, 400)
  Budžet=500; NLL perioda=20.058881/20.531953/20.135572; prosek=20.24222705; rasipanje=0.207321; stvarne iteracije=(280, 281, 402)
  Budžet=650; NLL perioda=20.058881/20.531953/20.135572; prosek=20.24222705; rasipanje=0.207321; stvarne iteracije=(280, 281, 402)
  Budžet=800; NLL perioda=20.058881/20.531953/20.135572; prosek=20.24222705; rasipanje=0.207321; stvarne iteracije=(280, 281, 402)

IZABRANO PREKO TRI VALIDACIONA PERIODA:
  Slojevi: 1
  Regularizacija: 0.0001
  Dozvoljeno trajanje: 250 iteracija
  Prosečni validacioni NLL: 19.95918246
  Rasipanje NLL-a između perioda: 0.15546596
  Pojedinačni NLL: 20.17869794, 19.83661577, 19.86256278

UČENJE ZA ODVOJENI TEST NA SVIM PRETEST PRELAZIMA
    Početni cilj=23.50849465; dozvoljeno iteracija=250
    Iteracija 1: cilj=22.69888206; |grad|inf=5.794e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.24999042; |grad|inf=2.014e-01; pad od prethodnog ispisa=1.449e+00
    Iteracija 50: cilj=20.92187735; |grad|inf=1.085e-01; pad od prethodnog ispisa=3.281e-01
    Iteracija 75: cilj=20.65851013; |grad|inf=6.343e-02; pad od prethodnog ispisa=2.634e-01
    Iteracija 100: cilj=20.36397492; |grad|inf=1.362e-02; pad od prethodnog ispisa=2.945e-01
    Iteracija 125: cilj=20.14609042; |grad|inf=5.402e-02; pad od prethodnog ispisa=2.179e-01
    Iteracija 150: cilj=20.11130085; |grad|inf=1.324e-02; pad od prethodnog ispisa=3.479e-02
    Iteracija 175: cilj=20.07545907; |grad|inf=1.111e-02; pad od prethodnog ispisa=3.584e-02
    Iteracija 200: cilj=20.05544217; |grad|inf=1.291e-02; pad od prethodnog ispisa=2.002e-02
    Iteracija 225: cilj=20.02113934; |grad|inf=8.634e-03; pad od prethodnog ispisa=3.430e-02
    Iteracija 250: cilj=20.01107452; |grad|inf=1.053e-02; pad od prethodnog ispisa=1.006e-02
    Zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 250
    Trening NLL: 20.01096752
    Završni |grad|inf: 1.053e-02
Odvojeni test — NLL: 20.45179954

ZAVRŠNO UČENJE NA CELOJ ISTORIJI
    Početni cilj=23.53425883; dozvoljeno iteracija=250
    Iteracija 1: cilj=22.73438862; |grad|inf=8.772e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.43157776; |grad|inf=2.552e-01; pad od prethodnog ispisa=1.303e+00
    Iteracija 50: cilj=20.88606983; |grad|inf=1.713e-01; pad od prethodnog ispisa=5.455e-01
    Iteracija 75: cilj=20.58580422; |grad|inf=1.807e-02; pad od prethodnog ispisa=3.003e-01
    Iteracija 100: cilj=20.49210299; |grad|inf=1.522e-02; pad od prethodnog ispisa=9.370e-02
    Iteracija 125: cilj=20.40826175; |grad|inf=4.161e-02; pad od prethodnog ispisa=8.384e-02
    Iteracija 150: cilj=20.20062753; |grad|inf=4.776e-03; pad od prethodnog ispisa=2.076e-01
    Iteracija 175: cilj=20.18948582; |grad|inf=6.002e-02; pad od prethodnog ispisa=1.114e-02
    Iteracija 200: cilj=20.14632990; |grad|inf=7.136e-03; pad od prethodnog ispisa=4.316e-02
    Iteracija 225: cilj=20.03652519; |grad|inf=1.139e-02; pad od prethodnog ispisa=1.098e-01
    Iteracija 250: cilj=20.00147099; |grad|inf=1.920e-02; pad od prethodnog ispisa=3.505e-02
    Zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 250
    Trening NLL: 20.00134200
    Završni |grad|inf: 1.920e-02

NEXT V4: 6, 12, 14, 17, 22, 33, 37
Logaritamska masa NEXT u modelu: -7.90340738
Promena zajedničkih marginala prema prethodnom kontekstu: 0.5822252432
Qiskit, gradijenti, normalizacija, granice i maksimum: ugrađene provere prošle.
Masa modela nije izmerena uspešnost predikcije.



V4 — FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv
Kola: 2976; prelazi: 2975; pretest: 2528; odvojeni test: 447

HRONOLOŠKI VALIDACIONI PERIODI
  Period 1: trening prelazi 1–1264; validacija 1265–1685
  Period 2: trening prelazi 1–1685; validacija 1686–2106
  Period 3: trening prelazi 1–2106; validacija 2107–2528

Pretraga: slojevi=1; regularizacija=0.0001; period=1/3
    Početni cilj=23.30118642; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.45684010; |grad|inf=4.758e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.08573539; |grad|inf=2.156e-01; pad od prethodnog ispisa=1.371e+00; validacioni NLL=21.19749845
    Iteracija 50: cilj=20.05604081; |grad|inf=3.498e-02; pad od prethodnog ispisa=1.030e+00; validacioni NLL=20.55606629
    Iteracija 75: cilj=19.89090780; |grad|inf=5.747e-02; pad od prethodnog ispisa=1.651e-01; validacioni NLL=20.51996426
    Iteracija 100: cilj=19.83499678; |grad|inf=2.395e-02; pad od prethodnog ispisa=5.591e-02; validacioni NLL=20.48053707
    Iteracija 125: cilj=19.55229233; |grad|inf=1.677e-02; pad od prethodnog ispisa=2.827e-01
    Iteracija 150: cilj=19.33213249; |grad|inf=2.128e-02; pad od prethodnog ispisa=2.202e-01; validacioni NLL=20.21445173
    Iteracija 175: cilj=19.24971878; |grad|inf=1.054e-02; pad od prethodnog ispisa=8.241e-02
    Iteracija 200: cilj=19.22636124; |grad|inf=2.145e-02; pad od prethodnog ispisa=2.336e-02; validacioni NLL=20.23163217
    Iteracija 225: cilj=19.21853112; |grad|inf=1.011e-03; pad od prethodnog ispisa=7.830e-03
    Iteracija 250: cilj=19.21818404; |grad|inf=8.078e-04; pad od prethodnog ispisa=3.471e-04; validacioni NLL=20.21119510
    Iteracija 275: cilj=19.21811901; |grad|inf=8.147e-05; pad od prethodnog ispisa=6.502e-05
    Iteracija 300: cilj=19.21811720; |grad|inf=2.365e-05; pad od prethodnog ispisa=1.817e-06; validacioni NLL=20.21098715
    Iteracija 310: cilj=19.21811713; |grad|inf=9.255e-06; pad od prethodnog ispisa=6.446e-08
    Ranija konvergencija: završna tačka važi za budžete 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 310
    Trening NLL: 19.21796175
    Završni |grad|inf: 9.255e-06

Pretraga: slojevi=1; regularizacija=0.0001; period=2/3
    Početni cilj=23.40422305; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.43595632; |grad|inf=4.104e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.81341484; |grad|inf=3.677e-01; pad od prethodnog ispisa=6.225e-01; validacioni NLL=22.25279730
    Iteracija 50: cilj=21.24834271; |grad|inf=2.073e-02; pad od prethodnog ispisa=5.651e-01; validacioni NLL=21.59841023
    Iteracija 75: cilj=20.98588345; |grad|inf=2.889e-02; pad od prethodnog ispisa=2.625e-01; validacioni NLL=21.41939605
    Iteracija 100: cilj=20.70307076; |grad|inf=4.386e-02; pad od prethodnog ispisa=2.828e-01; validacioni NLL=21.09844832
    Iteracija 125: cilj=20.47645782; |grad|inf=1.753e-02; pad od prethodnog ispisa=2.266e-01
    Iteracija 150: cilj=20.19947300; |grad|inf=4.232e-02; pad od prethodnog ispisa=2.770e-01; validacioni NLL=20.51841188
    Iteracija 175: cilj=19.99805455; |grad|inf=3.939e-02; pad od prethodnog ispisa=2.014e-01
    Iteracija 200: cilj=19.77905303; |grad|inf=1.400e-02; pad od prethodnog ispisa=2.190e-01; validacioni NLL=20.11840674
    Iteracija 225: cilj=19.73087249; |grad|inf=4.726e-03; pad od prethodnog ispisa=4.818e-02
    Iteracija 250: cilj=19.72601114; |grad|inf=2.514e-03; pad od prethodnog ispisa=4.861e-03; validacioni NLL=20.12925374
    Iteracija 275: cilj=19.61034501; |grad|inf=1.914e-03; pad od prethodnog ispisa=1.157e-01
    Iteracija 300: cilj=19.57385452; |grad|inf=2.539e-02; pad od prethodnog ispisa=3.649e-02; validacioni NLL=20.02685048
    Iteracija 325: cilj=19.52165143; |grad|inf=2.075e-02; pad od prethodnog ispisa=5.220e-02
    Iteracija 350: cilj=19.49133251; |grad|inf=2.054e-02; pad od prethodnog ispisa=3.032e-02
    Iteracija 375: cilj=19.45822333; |grad|inf=1.127e-02; pad od prethodnog ispisa=3.311e-02
    Iteracija 400: cilj=19.45137866; |grad|inf=3.327e-02; pad od prethodnog ispisa=6.845e-03; validacioni NLL=19.93635326
    Iteracija 425: cilj=19.40823859; |grad|inf=1.371e-03; pad od prethodnog ispisa=4.314e-02
    Iteracija 450: cilj=19.40810166; |grad|inf=3.000e-04; pad od prethodnog ispisa=1.369e-04
    Iteracija 475: cilj=19.40807550; |grad|inf=1.939e-04; pad od prethodnog ispisa=2.616e-05
    Iteracija 500: cilj=19.40804464; |grad|inf=4.778e-04; pad od prethodnog ispisa=3.086e-05; validacioni NLL=19.87015958
    Iteracija 525: cilj=19.40802637; |grad|inf=4.643e-05; pad od prethodnog ispisa=1.827e-05
    Iteracija 550: cilj=19.40802567; |grad|inf=1.574e-05; pad od prethodnog ispisa=6.986e-07
    Iteracija 564: cilj=19.40802559; |grad|inf=1.529e-05; pad od prethodnog ispisa=7.298e-08
    Ranija konvergencija: završna tačka važi za budžete 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 564
    Trening NLL: 19.40787920
    Završni |grad|inf: 1.529e-05

Pretraga: slojevi=1; regularizacija=0.0001; period=3/3
    Početni cilj=23.47316131; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.64708216; |grad|inf=9.040e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.95386057; |grad|inf=2.813e-01; pad od prethodnog ispisa=6.932e-01; validacioni NLL=21.78045062
    Iteracija 50: cilj=21.15901390; |grad|inf=6.503e-02; pad od prethodnog ispisa=7.948e-01; validacioni NLL=21.14722762
    Iteracija 75: cilj=21.09904120; |grad|inf=6.674e-02; pad od prethodnog ispisa=5.997e-02; validacioni NLL=21.20440958
    Iteracija 100: cilj=20.95165148; |grad|inf=1.277e-01; pad od prethodnog ispisa=1.474e-01; validacioni NLL=21.14339473
    Iteracija 125: cilj=20.73831221; |grad|inf=6.503e-02; pad od prethodnog ispisa=2.133e-01
    Iteracija 150: cilj=20.22741962; |grad|inf=1.185e-01; pad od prethodnog ispisa=5.109e-01; validacioni NLL=20.57994822
    Iteracija 175: cilj=19.99388650; |grad|inf=3.322e-02; pad od prethodnog ispisa=2.335e-01
    Iteracija 200: cilj=19.91532037; |grad|inf=5.080e-02; pad od prethodnog ispisa=7.857e-02; validacioni NLL=20.47713206
    Iteracija 225: cilj=19.70011910; |grad|inf=1.606e-01; pad od prethodnog ispisa=2.152e-01
    Iteracija 250: cilj=19.52866718; |grad|inf=2.004e-03; pad od prethodnog ispisa=1.715e-01; validacioni NLL=20.38913066
    Iteracija 275: cilj=19.52037472; |grad|inf=1.864e-02; pad od prethodnog ispisa=8.292e-03
    Iteracija 300: cilj=19.51393054; |grad|inf=6.825e-04; pad od prethodnog ispisa=6.444e-03; validacioni NLL=20.46697291
    Iteracija 325: cilj=19.51333831; |grad|inf=1.193e-03; pad od prethodnog ispisa=5.922e-04
    Iteracija 350: cilj=19.37430721; |grad|inf=2.741e-02; pad od prethodnog ispisa=1.390e-01
    Iteracija 375: cilj=19.36247019; |grad|inf=6.900e-03; pad od prethodnog ispisa=1.184e-02
    Iteracija 400: cilj=19.35791690; |grad|inf=1.371e-03; pad od prethodnog ispisa=4.553e-03; validacioni NLL=20.37419799
    Iteracija 425: cilj=19.35767775; |grad|inf=3.042e-04; pad od prethodnog ispisa=2.392e-04
    Iteracija 450: cilj=19.35767391; |grad|inf=4.593e-05; pad od prethodnog ispisa=3.833e-06
    Iteracija 468: cilj=19.35767382; |grad|inf=5.208e-06; pad od prethodnog ispisa=8.941e-08
    Ranija konvergencija: završna tačka važi za budžete 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 468
    Trening NLL: 19.35751424
    Završni |grad|inf: 5.208e-06

ZBIRNO: slojevi=1; regularizacija=0.0001
  Budžet= 25; NLL perioda=21.197498/22.252797/21.780451; prosek=21.74361129; rasipanje=0.431442; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=20.556066/21.598410/21.147228; prosek=21.10060496; rasipanje=0.426645; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=20.519964/21.419396/21.204410; prosek=21.04804710; rasipanje=0.383375; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.480537/21.098448/21.143395; prosek=20.90764670; rasipanje=0.302390; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.214452/20.518412/20.579948; prosek=20.43771656; rasipanje=0.159767; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.231632/20.118407/20.477132; prosek=20.27588300; rasipanje=0.149779; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.211195/20.129254/20.389131; prosek=20.24330862; rasipanje=0.108515; stvarne iteracije=(250, 250, 250)
  Budžet=300; NLL perioda=20.210987/20.026850/20.466973; prosek=20.23512042; rasipanje=0.180522; stvarne iteracije=(300, 300, 300)
  Budžet=400; NLL perioda=20.210942/19.936353/20.374198; prosek=20.17398945; rasipanje=0.180682; stvarne iteracije=(310, 400, 400)
  Budžet=500; NLL perioda=20.210942/19.870160/20.391480; prosek=20.15771223; rasipanje=0.216168; stvarne iteracije=(310, 500, 468)
  Budžet=650; NLL perioda=20.210942/19.870820/20.391480; prosek=20.15793211; rasipanje=0.215876; stvarne iteracije=(310, 564, 468)
  Budžet=800; NLL perioda=20.210942/19.870820/20.391480; prosek=20.15793211; rasipanje=0.215876; stvarne iteracije=(310, 564, 468)

Pretraga: slojevi=2; regularizacija=0.001; period=1/3
    Početni cilj=23.86956972; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.96724109; |grad|inf=6.310e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.68000024; |grad|inf=9.727e-02; pad od prethodnog ispisa=3.287e+00; validacioni NLL=20.62763332
    Iteracija 50: cilj=19.02576038; |grad|inf=2.086e-02; pad od prethodnog ispisa=6.542e-01; validacioni NLL=20.31917125
    Iteracija 75: cilj=18.81007728; |grad|inf=2.214e-02; pad od prethodnog ispisa=2.157e-01; validacioni NLL=20.17017812
    Iteracija 100: cilj=18.72409722; |grad|inf=1.918e-02; pad od prethodnog ispisa=8.598e-02; validacioni NLL=20.13712112
    Iteracija 125: cilj=18.63114536; |grad|inf=1.285e-02; pad od prethodnog ispisa=9.295e-02
    Iteracija 150: cilj=18.59984185; |grad|inf=5.713e-03; pad od prethodnog ispisa=3.130e-02; validacioni NLL=20.05620583
    Iteracija 175: cilj=18.56554626; |grad|inf=1.970e-02; pad od prethodnog ispisa=3.430e-02
    Iteracija 200: cilj=18.52543449; |grad|inf=4.885e-03; pad od prethodnog ispisa=4.011e-02; validacioni NLL=20.17720734
    Iteracija 225: cilj=18.50644549; |grad|inf=1.664e-03; pad od prethodnog ispisa=1.899e-02
    Iteracija 250: cilj=18.49479872; |grad|inf=7.021e-03; pad od prethodnog ispisa=1.165e-02; validacioni NLL=20.14439101
    Iteracija 275: cilj=18.48924389; |grad|inf=1.060e-02; pad od prethodnog ispisa=5.555e-03
    Iteracija 300: cilj=18.48673670; |grad|inf=5.589e-05; pad od prethodnog ispisa=2.507e-03; validacioni NLL=20.17616758
    Iteracija 304: cilj=18.48673666; |grad|inf=7.344e-06; pad od prethodnog ispisa=4.144e-08
    Ranija konvergencija: završna tačka važi za budžete 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 304
    Trening NLL: 18.48579436
    Završni |grad|inf: 7.344e-06

Pretraga: slojevi=2; regularizacija=0.001; period=2/3
    Početni cilj=23.94252452; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.99102925; |grad|inf=3.814e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.06948925; |grad|inf=1.338e-01; pad od prethodnog ispisa=2.922e+00; validacioni NLL=20.40900422
    Iteracija 50: cilj=19.20430675; |grad|inf=3.724e-02; pad od prethodnog ispisa=8.652e-01; validacioni NLL=20.21992784
    Iteracija 75: cilj=19.05382546; |grad|inf=1.518e-02; pad od prethodnog ispisa=1.505e-01; validacioni NLL=20.26670890
    Iteracija 100: cilj=18.98241933; |grad|inf=1.094e-02; pad od prethodnog ispisa=7.141e-02; validacioni NLL=20.33815954
    Iteracija 125: cilj=18.93349562; |grad|inf=1.052e-02; pad od prethodnog ispisa=4.892e-02
    Iteracija 150: cilj=18.91197059; |grad|inf=9.514e-03; pad od prethodnog ispisa=2.153e-02; validacioni NLL=20.23880330
    Iteracija 175: cilj=18.90126229; |grad|inf=8.968e-03; pad od prethodnog ispisa=1.071e-02
    Iteracija 200: cilj=18.88522965; |grad|inf=2.581e-02; pad od prethodnog ispisa=1.603e-02; validacioni NLL=20.30465609
    Iteracija 225: cilj=18.85986787; |grad|inf=1.746e-03; pad od prethodnog ispisa=2.536e-02
    Iteracija 246: cilj=18.85971084; |grad|inf=5.855e-06; pad od prethodnog ispisa=1.570e-04
    Ranija konvergencija: završna tačka važi za budžete 250, 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 246
    Trening NLL: 18.85886403
    Završni |grad|inf: 5.855e-06

Pretraga: slojevi=2; regularizacija=0.001; period=3/3
    Početni cilj=24.09205380; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.16776491; |grad|inf=4.999e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.65985292; |grad|inf=1.671e-01; pad od prethodnog ispisa=3.508e+00; validacioni NLL=20.10094609
    Iteracija 50: cilj=19.25364050; |grad|inf=2.049e-02; pad od prethodnog ispisa=4.062e-01; validacioni NLL=19.90354075
    Iteracija 75: cilj=19.16881363; |grad|inf=2.425e-02; pad od prethodnog ispisa=8.483e-02; validacioni NLL=20.06314791
    Iteracija 100: cilj=19.07966631; |grad|inf=1.934e-02; pad od prethodnog ispisa=8.915e-02; validacioni NLL=20.02241764
    Iteracija 125: cilj=19.04335682; |grad|inf=1.414e-02; pad od prethodnog ispisa=3.631e-02
    Iteracija 150: cilj=19.01627950; |grad|inf=5.475e-03; pad od prethodnog ispisa=2.708e-02; validacioni NLL=19.91870337
    Iteracija 175: cilj=19.00413328; |grad|inf=1.863e-02; pad od prethodnog ispisa=1.215e-02
    Iteracija 200: cilj=18.98230590; |grad|inf=9.566e-03; pad od prethodnog ispisa=2.183e-02; validacioni NLL=19.97343711
    Iteracija 225: cilj=18.97794375; |grad|inf=7.260e-04; pad od prethodnog ispisa=4.362e-03
    Iteracija 250: cilj=18.97787922; |grad|inf=1.485e-04; pad od prethodnog ispisa=6.454e-05; validacioni NLL=20.03051926
    Iteracija 275: cilj=18.97787389; |grad|inf=9.943e-05; pad od prethodnog ispisa=5.326e-06
    Iteracija 300: cilj=18.97786403; |grad|inf=6.083e-05; pad od prethodnog ispisa=9.856e-06; validacioni NLL=20.02942482
    Iteracija 323: cilj=18.97786344; |grad|inf=6.216e-06; pad od prethodnog ispisa=5.919e-07
    Ranija konvergencija: završna tačka važi za budžete 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 323
    Trening NLL: 18.97690437
    Završni |grad|inf: 6.216e-06

ZBIRNO: slojevi=2; regularizacija=0.001
  Budžet= 25; NLL perioda=20.627633/20.409004/20.100946; prosek=20.37897441; rasipanje=0.216106; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=20.319171/20.219928/19.903541; prosek=20.14735357; rasipanje=0.177294; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=20.170178/20.266709/20.063148; prosek=20.16659640; rasipanje=0.083158; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.137121/20.338160/20.022418; prosek=20.16578592; rasipanje=0.130508; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.056206/20.238803/19.918703; prosek=20.07111683; rasipanje=0.131130; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.177207/20.304656/19.973437; prosek=20.15162576; rasipanje=0.136449; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.144391/20.322736/20.030519; prosek=20.16577503; rasipanje=0.120274; stvarne iteracije=(250, 246, 250)
  Budžet=300; NLL perioda=20.176168/20.322736/20.029425; prosek=20.17599345; rasipanje=0.119767; stvarne iteracije=(300, 246, 300)
  Budžet=400; NLL perioda=20.176059/20.322736/20.029451; prosek=20.17596601; rasipanje=0.119757; stvarne iteracije=(304, 246, 323)
  Budžet=500; NLL perioda=20.176059/20.322736/20.029451; prosek=20.17596601; rasipanje=0.119757; stvarne iteracije=(304, 246, 323)
  Budžet=650; NLL perioda=20.176059/20.322736/20.029451; prosek=20.17596601; rasipanje=0.119757; stvarne iteracije=(304, 246, 323)
  Budžet=800; NLL perioda=20.176059/20.322736/20.029451; prosek=20.17596601; rasipanje=0.119757; stvarne iteracije=(304, 246, 323)

Pretraga: slojevi=2; regularizacija=0.01; period=1/3
    Početni cilj=23.86975279; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.96740909; |grad|inf=6.309e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.59881789; |grad|inf=1.010e-01; pad od prethodnog ispisa=3.369e+00; validacioni NLL=20.17466201
    Iteracija 50: cilj=19.01369239; |grad|inf=2.623e-02; pad od prethodnog ispisa=5.851e-01; validacioni NLL=20.00757752
    Iteracija 75: cilj=18.91300693; |grad|inf=1.878e-02; pad od prethodnog ispisa=1.007e-01; validacioni NLL=20.07301069
    Iteracija 100: cilj=18.86040939; |grad|inf=1.795e-02; pad od prethodnog ispisa=5.260e-02; validacioni NLL=20.16267443
    Iteracija 125: cilj=18.80662604; |grad|inf=1.871e-02; pad od prethodnog ispisa=5.378e-02
    Iteracija 150: cilj=18.76530183; |grad|inf=7.899e-03; pad od prethodnog ispisa=4.132e-02; validacioni NLL=20.15183235
    Iteracija 175: cilj=18.75971606; |grad|inf=4.558e-03; pad od prethodnog ispisa=5.586e-03
    Iteracija 200: cilj=18.75571979; |grad|inf=4.187e-03; pad od prethodnog ispisa=3.996e-03; validacioni NLL=20.07071253
    Iteracija 225: cilj=18.75222394; |grad|inf=1.001e-04; pad od prethodnog ispisa=3.496e-03
    Iteracija 250: cilj=18.75221480; |grad|inf=7.894e-04; pad od prethodnog ispisa=9.137e-06; validacioni NLL=20.10459729
    Iteracija 275: cilj=18.75209425; |grad|inf=1.435e-05; pad od prethodnog ispisa=1.206e-04
    Iteracija 277: cilj=18.75209424; |grad|inf=1.399e-05; pad od prethodnog ispisa=5.610e-09
    Ranija konvergencija: završna tačka važi za budžete 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 277
    Trening NLL: 18.74387574
    Završni |grad|inf: 1.399e-05

Pretraga: slojevi=2; regularizacija=0.01; period=2/3
    Početni cilj=23.94270759; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.99119477; |grad|inf=3.816e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.67346081; |grad|inf=2.618e-01; pad od prethodnog ispisa=3.318e+00; validacioni NLL=20.39493119
    Iteracija 50: cilj=19.28451489; |grad|inf=2.122e-02; pad od prethodnog ispisa=3.889e-01; validacioni NLL=20.29751444
    Iteracija 75: cilj=19.09331470; |grad|inf=2.063e-02; pad od prethodnog ispisa=1.912e-01; validacioni NLL=20.18005531
    Iteracija 100: cilj=19.03133939; |grad|inf=5.466e-02; pad od prethodnog ispisa=6.198e-02; validacioni NLL=20.08808406
    Iteracija 125: cilj=18.97290654; |grad|inf=1.401e-02; pad od prethodnog ispisa=5.843e-02
    Iteracija 150: cilj=18.95363414; |grad|inf=1.324e-02; pad od prethodnog ispisa=1.927e-02; validacioni NLL=20.01227141
    Iteracija 175: cilj=18.93950278; |grad|inf=9.714e-03; pad od prethodnog ispisa=1.413e-02
    Iteracija 200: cilj=18.92390348; |grad|inf=9.447e-03; pad od prethodnog ispisa=1.560e-02; validacioni NLL=20.03483760
    Iteracija 225: cilj=18.91686123; |grad|inf=3.676e-03; pad od prethodnog ispisa=7.042e-03
    Iteracija 250: cilj=18.91631369; |grad|inf=1.055e-04; pad od prethodnog ispisa=5.475e-04; validacioni NLL=20.07439424
    Iteracija 271: cilj=18.91631238; |grad|inf=7.246e-06; pad od prethodnog ispisa=1.309e-06
    Ranija konvergencija: završna tačka važi za budžete 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 271
    Trening NLL: 18.90698452
    Završni |grad|inf: 7.246e-06

Pretraga: slojevi=2; regularizacija=0.01; period=3/3
    Početni cilj=24.09223686; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.16795670; |grad|inf=5.047e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.68150545; |grad|inf=6.194e-02; pad od prethodnog ispisa=3.486e+00; validacioni NLL=20.59653123
    Iteracija 50: cilj=19.25262262; |grad|inf=3.131e-02; pad od prethodnog ispisa=4.289e-01; validacioni NLL=20.40222057
    Iteracija 75: cilj=19.12138734; |grad|inf=5.459e-02; pad od prethodnog ispisa=1.312e-01; validacioni NLL=20.51671859
    Iteracija 100: cilj=19.08383256; |grad|inf=1.659e-02; pad od prethodnog ispisa=3.755e-02; validacioni NLL=20.51658124
    Iteracija 125: cilj=19.05419419; |grad|inf=3.557e-02; pad od prethodnog ispisa=2.964e-02
    Iteracija 150: cilj=19.00422141; |grad|inf=2.021e-02; pad od prethodnog ispisa=4.997e-02; validacioni NLL=20.41336058
    Iteracija 175: cilj=18.99336674; |grad|inf=2.727e-04; pad od prethodnog ispisa=1.085e-02
    Iteracija 200: cilj=18.99334182; |grad|inf=5.289e-05; pad od prethodnog ispisa=2.492e-05; validacioni NLL=20.40697444
    Iteracija 223: cilj=18.99334138; |grad|inf=4.568e-05; pad od prethodnog ispisa=4.448e-07
    Ranija konvergencija: završna tačka važi za budžete 250, 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 223
    Trening NLL: 18.98508143
    Završni |grad|inf: 4.568e-05

ZBIRNO: slojevi=2; regularizacija=0.01
  Budžet= 25; NLL perioda=20.174662/20.394931/20.596531; prosek=20.38887256; rasipanje=0.172315; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=20.007578/20.297514/20.402221; prosek=20.23590253; rasipanje=0.166923; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=20.073011/20.180055/20.516719; prosek=20.25680066; rasipanje=0.189122; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.162674/20.088084/20.516581; prosek=20.25598624; rasipanje=0.186982; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.151832/20.012271/20.413361; prosek=20.19266286; rasipanje=0.166299; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.070713/20.034838/20.406974; prosek=20.17102834; rasipanje=0.167677; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.104597/20.074394/20.407117; prosek=20.19553717; rasipanje=0.150294; stvarne iteracije=(250, 250, 223)
  Budžet=300; NLL perioda=20.113990/20.074480/20.407117; prosek=20.19869424; rasipanje=0.148431; stvarne iteracije=(277, 271, 223)
  Budžet=400; NLL perioda=20.113990/20.074480/20.407117; prosek=20.19869424; rasipanje=0.148431; stvarne iteracije=(277, 271, 223)
  Budžet=500; NLL perioda=20.113990/20.074480/20.407117; prosek=20.19869424; rasipanje=0.148431; stvarne iteracije=(277, 271, 223)
  Budžet=650; NLL perioda=20.113990/20.074480/20.407117; prosek=20.19869424; rasipanje=0.148431; stvarne iteracije=(277, 271, 223)
  Budžet=800; NLL perioda=20.113990/20.074480/20.407117; prosek=20.19869424; rasipanje=0.148431; stvarne iteracije=(277, 271, 223)

IZABRANO PREKO TRI VALIDACIONA PERIODA:
  Slojevi: 2
  Regularizacija: 0.001
  Dozvoljeno trajanje: 150 iteracija
  Prosečni validacioni NLL: 20.07111683
  Rasipanje NLL-a između perioda: 0.13113007
  Pojedinačni NLL: 20.05620583, 20.23880330, 19.91870337

UČENJE ZA ODVOJENI TEST NA SVIM PRETEST PRELAZIMA
    Početni cilj=24.11631489; dozvoljeno iteracija=150
    Iteracija 1: cilj=23.23566470; |grad|inf=3.422e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.39061486; |grad|inf=1.792e-01; pad od prethodnog ispisa=2.845e+00
    Iteracija 50: cilj=19.46663757; |grad|inf=2.484e-02; pad od prethodnog ispisa=9.240e-01
    Iteracija 75: cilj=19.33349583; |grad|inf=2.206e-02; pad od prethodnog ispisa=1.331e-01
    Iteracija 100: cilj=19.19846164; |grad|inf=9.579e-03; pad od prethodnog ispisa=1.350e-01
    Iteracija 125: cilj=19.16818697; |grad|inf=1.322e-02; pad od prethodnog ispisa=3.027e-02
    Iteracija 150: cilj=19.10642407; |grad|inf=1.795e-02; pad od prethodnog ispisa=6.176e-02
    Zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 150
    Trening NLL: 19.10569392
    Završni |grad|inf: 1.795e-02
Odvojeni test — NLL: 20.29079213

ZAVRŠNO UČENJE NA CELOJ ISTORIJI
    Početni cilj=24.05945083; dozvoljeno iteracija=150
    Iteracija 1: cilj=23.22578644; |grad|inf=3.525e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.79258422; |grad|inf=7.658e-02; pad od prethodnog ispisa=3.433e+00
    Iteracija 50: cilj=19.44288942; |grad|inf=3.123e-02; pad od prethodnog ispisa=3.497e-01
    Iteracija 75: cilj=19.32486263; |grad|inf=2.476e-02; pad od prethodnog ispisa=1.180e-01
    Iteracija 100: cilj=19.22638454; |grad|inf=3.444e-02; pad od prethodnog ispisa=9.848e-02
    Iteracija 125: cilj=19.19321574; |grad|inf=4.991e-03; pad od prethodnog ispisa=3.317e-02
    Iteracija 150: cilj=19.18599577; |grad|inf=1.255e-02; pad od prethodnog ispisa=7.220e-03
    Zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 150
    Trening NLL: 19.18504596
    Završni |grad|inf: 1.255e-02

NEXT V4: 5, 15, 18, 22, 28, 30, 38
Logaritamska masa NEXT u modelu: -9.62119489
Promena zajedničkih marginala prema prethodnom kontekstu: 0.2754412567
Qiskit, gradijenti, normalizacija, granice i maksimum: ugrađene provere prošle.
Masa modela nije izmerena uspešnost predikcije.



V4 — FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv
Kola: 1732; prelazi: 1731; pretest: 1471; odvojeni test: 260

HRONOLOŠKI VALIDACIONI PERIODI
  Period 1: trening prelazi 1–735; validacija 736–980
  Period 2: trening prelazi 1–980; validacija 981–1225
  Period 3: trening prelazi 1–1225; validacija 1226–1471

Pretraga: slojevi=1; regularizacija=0.0001; period=1/3
    Početni cilj=23.48984355; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.52338220; |grad|inf=3.831e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.45889201; |grad|inf=9.600e-02; pad od prethodnog ispisa=1.064e+00; validacioni NLL=21.40726907
    Iteracija 50: cilj=20.50406554; |grad|inf=2.734e-01; pad od prethodnog ispisa=9.548e-01; validacioni NLL=20.93077446
    Iteracija 75: cilj=20.24609587; |grad|inf=1.701e-01; pad od prethodnog ispisa=2.580e-01; validacioni NLL=20.90888465
    Iteracija 100: cilj=19.76259505; |grad|inf=2.563e-02; pad od prethodnog ispisa=4.835e-01; validacioni NLL=20.50834134
    Iteracija 125: cilj=19.68783138; |grad|inf=3.098e-02; pad od prethodnog ispisa=7.476e-02
    Iteracija 150: cilj=19.40980044; |grad|inf=3.290e-02; pad od prethodnog ispisa=2.780e-01; validacioni NLL=20.12316193
    Iteracija 175: cilj=19.17994680; |grad|inf=9.001e-02; pad od prethodnog ispisa=2.299e-01
    Iteracija 200: cilj=18.74517854; |grad|inf=1.444e-02; pad od prethodnog ispisa=4.348e-01; validacioni NLL=20.07058121
    Iteracija 225: cilj=18.72397857; |grad|inf=6.461e-03; pad od prethodnog ispisa=2.120e-02
    Iteracija 250: cilj=18.72067701; |grad|inf=1.803e-03; pad od prethodnog ispisa=3.302e-03; validacioni NLL=20.05524586
    Iteracija 275: cilj=18.72010745; |grad|inf=1.323e-04; pad od prethodnog ispisa=5.696e-04
    Iteracija 298: cilj=18.72010531; |grad|inf=1.071e-05; pad od prethodnog ispisa=2.143e-06
    Ranija konvergencija: završna tačka važi za budžete 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 298
    Trening NLL: 18.71987096
    Završni |grad|inf: 1.071e-05

Pretraga: slojevi=1; regularizacija=0.0001; period=2/3
    Početni cilj=23.39375709; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.60579308; |grad|inf=4.523e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.27615156; |grad|inf=1.054e-01; pad od prethodnog ispisa=1.330e+00; validacioni NLL=21.11182184
    Iteracija 50: cilj=20.67766295; |grad|inf=5.712e-02; pad od prethodnog ispisa=5.985e-01; validacioni NLL=20.68829926
    Iteracija 75: cilj=20.59417296; |grad|inf=7.668e-03; pad od prethodnog ispisa=8.349e-02; validacioni NLL=20.59792926
    Iteracija 100: cilj=20.35709485; |grad|inf=1.707e-01; pad od prethodnog ispisa=2.371e-01; validacioni NLL=20.39739753
    Iteracija 125: cilj=20.22085189; |grad|inf=2.055e-01; pad od prethodnog ispisa=1.362e-01
    Iteracija 150: cilj=19.68626310; |grad|inf=6.147e-02; pad od prethodnog ispisa=5.346e-01; validacioni NLL=20.16339007
    Iteracija 175: cilj=19.38027750; |grad|inf=7.173e-03; pad od prethodnog ispisa=3.060e-01
    Iteracija 200: cilj=19.27867884; |grad|inf=3.019e-02; pad od prethodnog ispisa=1.016e-01; validacioni NLL=20.22063648
    Iteracija 225: cilj=19.17138674; |grad|inf=3.926e-02; pad od prethodnog ispisa=1.073e-01
    Iteracija 250: cilj=19.11235949; |grad|inf=2.160e-03; pad od prethodnog ispisa=5.903e-02; validacioni NLL=20.02870263
    Iteracija 275: cilj=19.10559917; |grad|inf=1.760e-03; pad od prethodnog ispisa=6.760e-03
    Iteracija 300: cilj=19.03412093; |grad|inf=3.873e-02; pad od prethodnog ispisa=7.148e-02; validacioni NLL=20.04033942
    Iteracija 325: cilj=18.95329786; |grad|inf=2.463e-02; pad od prethodnog ispisa=8.082e-02
    Iteracija 350: cilj=18.94557630; |grad|inf=5.140e-05; pad od prethodnog ispisa=7.722e-03
    Iteracija 361: cilj=18.94557615; |grad|inf=2.824e-06; pad od prethodnog ispisa=1.523e-07
    Ranija konvergencija: završna tačka važi za budžete 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 361
    Trening NLL: 18.94539789
    Završni |grad|inf: 2.824e-06

Pretraga: slojevi=1; regularizacija=0.0001; period=3/3
    Početni cilj=23.51903742; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.59444031; |grad|inf=5.375e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=21.54463553; |grad|inf=1.481e-01; pad od prethodnog ispisa=1.050e+00; validacioni NLL=22.35637558
    Iteracija 50: cilj=21.21787603; |grad|inf=6.843e-02; pad od prethodnog ispisa=3.268e-01; validacioni NLL=22.05326488
    Iteracija 75: cilj=20.86066706; |grad|inf=4.911e-02; pad od prethodnog ispisa=3.572e-01; validacioni NLL=21.46747233
    Iteracija 100: cilj=20.63739100; |grad|inf=3.292e-02; pad od prethodnog ispisa=2.233e-01; validacioni NLL=21.20774876
    Iteracija 125: cilj=20.44550672; |grad|inf=1.404e-02; pad od prethodnog ispisa=1.919e-01
    Iteracija 150: cilj=20.43149527; |grad|inf=9.804e-03; pad od prethodnog ispisa=1.401e-02; validacioni NLL=21.13593906
    Iteracija 175: cilj=20.18016230; |grad|inf=5.182e-02; pad od prethodnog ispisa=2.513e-01
    Iteracija 200: cilj=20.11508738; |grad|inf=1.033e-02; pad od prethodnog ispisa=6.507e-02; validacioni NLL=20.75392167
    Iteracija 225: cilj=19.75241357; |grad|inf=4.861e-01; pad od prethodnog ispisa=3.627e-01
    Iteracija 250: cilj=19.44666217; |grad|inf=2.863e-02; pad od prethodnog ispisa=3.058e-01; validacioni NLL=20.24846127
    Iteracija 275: cilj=19.32813327; |grad|inf=2.108e-02; pad od prethodnog ispisa=1.185e-01
    Iteracija 300: cilj=19.27259179; |grad|inf=1.673e-02; pad od prethodnog ispisa=5.554e-02; validacioni NLL=20.25967525
    Iteracija 325: cilj=19.23520124; |grad|inf=3.171e-03; pad od prethodnog ispisa=3.739e-02
    Iteracija 350: cilj=19.22495638; |grad|inf=5.638e-03; pad od prethodnog ispisa=1.024e-02
    Iteracija 375: cilj=19.22336784; |grad|inf=9.866e-05; pad od prethodnog ispisa=1.589e-03
    Iteracija 382: cilj=19.22336756; |grad|inf=1.269e-05; pad od prethodnog ispisa=2.703e-07
    Ranija konvergencija: završna tačka važi za budžete 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 382
    Trening NLL: 19.22321385
    Završni |grad|inf: 1.269e-05

ZBIRNO: slojevi=1; regularizacija=0.0001
  Budžet= 25; NLL perioda=21.407269/21.111822/22.356376; prosek=21.62614900; rasipanje=0.531255; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=20.930774/20.688299/22.053265; prosek=21.22523943; rasipanje=0.594977; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=20.908885/20.597929/21.467472; prosek=20.99207555; rasipanje=0.359939; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.508341/20.397398/21.207749; prosek=20.70517964; rasipanje=0.358960; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.123162/20.163390/21.135939; prosek=20.47506284; rasipanje=0.468550; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.070581/20.220636/20.753922; prosek=20.34893079; rasipanje=0.293413; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.055246/20.028703/20.248461; prosek=20.11099029; rasipanje=0.098005; stvarne iteracije=(250, 250, 250)
  Budžet=300; NLL perioda=20.050320/20.040339/20.259675; prosek=20.11697230; rasipanje=0.101194; stvarne iteracije=(298, 300, 300)
  Budžet=400; NLL perioda=20.050320/19.994857/20.375228; prosek=20.14045447; rasipanje=0.167880; stvarne iteracije=(298, 361, 382)
  Budžet=500; NLL perioda=20.050320/19.994857/20.375228; prosek=20.14045447; rasipanje=0.167880; stvarne iteracije=(298, 361, 382)
  Budžet=650; NLL perioda=20.050320/19.994857/20.375228; prosek=20.14045447; rasipanje=0.167880; stvarne iteracije=(298, 361, 382)
  Budžet=800; NLL perioda=20.050320/19.994857/20.375228; prosek=20.14045447; rasipanje=0.167880; stvarne iteracije=(298, 361, 382)

Pretraga: slojevi=2; regularizacija=0.001; period=1/3
    Početni cilj=23.89503929; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.22679526; |grad|inf=8.772e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.60237994; |grad|inf=1.192e-01; pad od prethodnog ispisa=3.624e+00; validacioni NLL=20.55684042
    Iteracija 50: cilj=18.46460913; |grad|inf=2.288e-02; pad od prethodnog ispisa=1.138e+00; validacioni NLL=20.42473936
    Iteracija 75: cilj=18.28948570; |grad|inf=2.494e-02; pad od prethodnog ispisa=1.751e-01; validacioni NLL=20.53760575
    Iteracija 100: cilj=18.20386843; |grad|inf=2.894e-02; pad od prethodnog ispisa=8.562e-02; validacioni NLL=20.50907427
    Iteracija 125: cilj=18.17390255; |grad|inf=1.400e-02; pad od prethodnog ispisa=2.997e-02
    Iteracija 150: cilj=18.17097523; |grad|inf=5.732e-03; pad od prethodnog ispisa=2.927e-03; validacioni NLL=20.49915284
    Iteracija 175: cilj=18.16401102; |grad|inf=2.942e-04; pad od prethodnog ispisa=6.964e-03
    Iteracija 194: cilj=18.16400883; |grad|inf=1.081e-05; pad od prethodnog ispisa=2.184e-06
    Ranija konvergencija: završna tačka važi za budžete 200, 250, 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 194
    Trening NLL: 18.16306029
    Završni |grad|inf: 1.081e-05

Pretraga: slojevi=2; regularizacija=0.001; period=2/3
    Početni cilj=23.80163839; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.10468410; |grad|inf=7.604e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.18208522; |grad|inf=1.550e-01; pad od prethodnog ispisa=3.923e+00; validacioni NLL=20.46167779
    Iteracija 50: cilj=18.57181852; |grad|inf=2.153e-02; pad od prethodnog ispisa=6.103e-01; validacioni NLL=20.80290023
    Iteracija 75: cilj=18.43976491; |grad|inf=1.985e-02; pad od prethodnog ispisa=1.321e-01; validacioni NLL=20.28444706
    Iteracija 100: cilj=18.38743616; |grad|inf=2.490e-02; pad od prethodnog ispisa=5.233e-02; validacioni NLL=20.43165179
    Iteracija 125: cilj=18.34293188; |grad|inf=6.055e-03; pad od prethodnog ispisa=4.450e-02
    Iteracija 150: cilj=18.32878405; |grad|inf=4.072e-03; pad od prethodnog ispisa=1.415e-02; validacioni NLL=20.58577323
    Iteracija 175: cilj=18.31342372; |grad|inf=1.725e-03; pad od prethodnog ispisa=1.536e-02
    Iteracija 200: cilj=18.31053476; |grad|inf=5.353e-03; pad od prethodnog ispisa=2.889e-03; validacioni NLL=20.51382024
    Iteracija 225: cilj=18.30472812; |grad|inf=8.118e-04; pad od prethodnog ispisa=5.807e-03
    Iteracija 250: cilj=18.30462667; |grad|inf=7.607e-05; pad od prethodnog ispisa=1.015e-04; validacioni NLL=20.49845146
    Iteracija 275: cilj=18.30453652; |grad|inf=2.907e-05; pad od prethodnog ispisa=9.014e-05
    Iteracija 300: cilj=18.30453627; |grad|inf=8.939e-06; pad od prethodnog ispisa=2.509e-07; validacioni NLL=20.48018627
    Iteracija 301: cilj=18.30453627; |grad|inf=8.873e-06; pad od prethodnog ispisa=9.301e-10
    Ranija konvergencija: završna tačka važi za budžete 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 301
    Trening NLL: 18.30366086
    Završni |grad|inf: 8.873e-06

Pretraga: slojevi=2; regularizacija=0.001; period=3/3
    Početni cilj=23.82528784; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.05094641; |grad|inf=8.802e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.86715442; |grad|inf=9.702e-02; pad od prethodnog ispisa=3.184e+00; validacioni NLL=20.92971407
    Iteracija 50: cilj=19.32895246; |grad|inf=7.942e-02; pad od prethodnog ispisa=5.382e-01; validacioni NLL=20.44929853
    Iteracija 75: cilj=18.91364176; |grad|inf=3.188e-02; pad od prethodnog ispisa=4.153e-01; validacioni NLL=20.36515351
    Iteracija 100: cilj=18.82984325; |grad|inf=2.045e-02; pad od prethodnog ispisa=8.380e-02; validacioni NLL=20.26118200
    Iteracija 125: cilj=18.75272023; |grad|inf=3.844e-03; pad od prethodnog ispisa=7.712e-02
    Iteracija 150: cilj=18.72806464; |grad|inf=5.265e-03; pad od prethodnog ispisa=2.466e-02; validacioni NLL=20.49521975
    Iteracija 175: cilj=18.70139531; |grad|inf=1.107e-02; pad od prethodnog ispisa=2.667e-02
    Iteracija 200: cilj=18.69037229; |grad|inf=6.174e-03; pad od prethodnog ispisa=1.102e-02; validacioni NLL=20.41333631
    Iteracija 225: cilj=18.68800366; |grad|inf=3.552e-04; pad od prethodnog ispisa=2.369e-03
    Iteracija 242: cilj=18.68800150; |grad|inf=6.962e-06; pad od prethodnog ispisa=2.167e-06
    Ranija konvergencija: završna tačka važi za budžete 250, 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 242
    Trening NLL: 18.68718151
    Završni |grad|inf: 6.962e-06

ZBIRNO: slojevi=2; regularizacija=0.001
  Budžet= 25; NLL perioda=20.556840/20.461678/20.929714; prosek=20.64979161; rasipanje=0.202103; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=20.424739/20.802900/20.449299; prosek=20.55883035; rasipanje=0.172699; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=20.537606/20.284447/20.365154; prosek=20.39569389; rasipanje=0.105524; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.509074/20.431652/20.261182; prosek=20.40044655; rasipanje=0.103608; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.499153/20.585773/20.495220; prosek=20.52667248; rasipanje=0.041779; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.451851/20.513820/20.413336; prosek=20.45960609; rasipanje=0.041400; stvarne iteracije=(194, 200, 200)
  Budžet=250; NLL perioda=20.451851/20.498451/20.459945; prosek=20.47006844; rasipanje=0.020320; stvarne iteracije=(194, 250, 242)
  Budžet=300; NLL perioda=20.451851/20.480186/20.459945; prosek=20.46398831; rasipanje=0.011910; stvarne iteracije=(194, 300, 242)
  Budžet=400; NLL perioda=20.451851/20.480187/20.459945; prosek=20.46398858; rasipanje=0.011910; stvarne iteracije=(194, 301, 242)
  Budžet=500; NLL perioda=20.451851/20.480187/20.459945; prosek=20.46398858; rasipanje=0.011910; stvarne iteracije=(194, 301, 242)
  Budžet=650; NLL perioda=20.451851/20.480187/20.459945; prosek=20.46398858; rasipanje=0.011910; stvarne iteracije=(194, 301, 242)
  Budžet=800; NLL perioda=20.451851/20.480187/20.459945; prosek=20.46398858; rasipanje=0.011910; stvarne iteracije=(194, 301, 242)

Pretraga: slojevi=2; regularizacija=0.01; period=1/3
    Početni cilj=23.89522236; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.22696414; |grad|inf=8.772e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.07883678; |grad|inf=5.054e-02; pad od prethodnog ispisa=4.148e+00; validacioni NLL=20.47549876
    Iteracija 50: cilj=18.38518493; |grad|inf=4.819e-02; pad od prethodnog ispisa=6.937e-01; validacioni NLL=20.03753532
    Iteracija 75: cilj=18.23490693; |grad|inf=2.150e-02; pad od prethodnog ispisa=1.503e-01; validacioni NLL=19.99079451
    Iteracija 100: cilj=18.17436529; |grad|inf=1.833e-02; pad od prethodnog ispisa=6.054e-02; validacioni NLL=20.25787416
    Iteracija 125: cilj=18.10544181; |grad|inf=5.911e-03; pad od prethodnog ispisa=6.892e-02
    Iteracija 150: cilj=18.06680335; |grad|inf=3.336e-03; pad od prethodnog ispisa=3.864e-02; validacioni NLL=20.33297482
    Iteracija 175: cilj=18.06528053; |grad|inf=1.064e-03; pad od prethodnog ispisa=1.523e-03
    Iteracija 200: cilj=18.06503953; |grad|inf=1.409e-03; pad od prethodnog ispisa=2.410e-04; validacioni NLL=20.34956549
    Iteracija 225: cilj=18.06493152; |grad|inf=5.796e-05; pad od prethodnog ispisa=1.080e-04
    Iteracija 241: cilj=18.06493075; |grad|inf=8.682e-06; pad od prethodnog ispisa=7.779e-07
    Ranija konvergencija: završna tačka važi za budžete 250, 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 241
    Trening NLL: 18.05521357
    Završni |grad|inf: 8.682e-06

Pretraga: slojevi=2; regularizacija=0.01; period=2/3
    Početni cilj=23.80182145; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.10487192; |grad|inf=7.604e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=19.08393551; |grad|inf=8.787e-02; pad od prethodnog ispisa=4.021e+00; validacioni NLL=19.99344292
    Iteracija 50: cilj=18.64471775; |grad|inf=2.415e-02; pad od prethodnog ispisa=4.392e-01; validacioni NLL=19.96299162
    Iteracija 75: cilj=18.53405577; |grad|inf=1.130e-02; pad od prethodnog ispisa=1.107e-01; validacioni NLL=20.14656974
    Iteracija 100: cilj=18.48210814; |grad|inf=2.988e-02; pad od prethodnog ispisa=5.195e-02; validacioni NLL=20.10266304
    Iteracija 125: cilj=18.46180965; |grad|inf=4.789e-03; pad od prethodnog ispisa=2.030e-02
    Iteracija 150: cilj=18.46131708; |grad|inf=3.053e-04; pad od prethodnog ispisa=4.926e-04; validacioni NLL=20.04727302
    Iteracija 175: cilj=18.46101097; |grad|inf=1.790e-03; pad od prethodnog ispisa=3.061e-04
    Iteracija 200: cilj=18.46063352; |grad|inf=4.565e-04; pad od prethodnog ispisa=3.775e-04; validacioni NLL=20.06722873
    Iteracija 225: cilj=18.46039920; |grad|inf=2.290e-04; pad od prethodnog ispisa=2.343e-04
    Iteracija 250: cilj=18.46038390; |grad|inf=5.173e-05; pad od prethodnog ispisa=1.530e-05; validacioni NLL=20.05266099
    Iteracija 275: cilj=18.46038327; |grad|inf=2.105e-05; pad od prethodnog ispisa=6.276e-07
    Iteracija 298: cilj=18.46038315; |grad|inf=8.287e-06; pad od prethodnog ispisa=1.205e-07
    Ranija konvergencija: završna tačka važi za budžete 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 298
    Trening NLL: 18.45053469
    Završni |grad|inf: 8.287e-06

Pretraga: slojevi=2; regularizacija=0.01; period=3/3
    Početni cilj=23.82547091; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.05113468; |grad|inf=8.930e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.15756679; |grad|inf=1.245e-01; pad od prethodnog ispisa=2.894e+00; validacioni NLL=20.82280970
    Iteracija 50: cilj=19.63303413; |grad|inf=7.811e-02; pad od prethodnog ispisa=5.245e-01; validacioni NLL=20.71714240
    Iteracija 75: cilj=18.85103126; |grad|inf=2.858e-02; pad od prethodnog ispisa=7.820e-01; validacioni NLL=20.17488553
    Iteracija 100: cilj=18.73958826; |grad|inf=2.632e-02; pad od prethodnog ispisa=1.114e-01; validacioni NLL=20.09846408
    Iteracija 125: cilj=18.63475664; |grad|inf=1.756e-02; pad od prethodnog ispisa=1.048e-01
    Iteracija 150: cilj=18.57091062; |grad|inf=1.760e-02; pad od prethodnog ispisa=6.385e-02; validacioni NLL=20.35896350
    Iteracija 175: cilj=18.55375710; |grad|inf=3.505e-03; pad od prethodnog ispisa=1.715e-02
    Iteracija 200: cilj=18.51315068; |grad|inf=1.771e-02; pad od prethodnog ispisa=4.061e-02; validacioni NLL=20.33588315
    Iteracija 225: cilj=18.48423461; |grad|inf=3.859e-05; pad od prethodnog ispisa=2.892e-02
    Iteracija 231: cilj=18.48423457; |grad|inf=1.039e-05; pad od prethodnog ispisa=4.449e-08
    Ranija konvergencija: završna tačka važi za budžete 250, 300, 400, 500, 650, 800
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 231
    Trening NLL: 18.47521492
    Završni |grad|inf: 1.039e-05

ZBIRNO: slojevi=2; regularizacija=0.01
  Budžet= 25; NLL perioda=20.475499/19.993443/20.822810; prosek=20.43111671; rasipanje=0.340150; stvarne iteracije=(25, 25, 25)
  Budžet= 50; NLL perioda=20.037535/19.962992/20.717142; prosek=20.23987246; rasipanje=0.339534; stvarne iteracije=(50, 50, 50)
  Budžet= 75; NLL perioda=19.990795/20.146570/20.174886; prosek=20.10417946; rasipanje=0.080924; stvarne iteracije=(75, 75, 75)
  Budžet=100; NLL perioda=20.257874/20.102663/20.098464; prosek=20.15292633; rasipanje=0.074154; stvarne iteracije=(100, 100, 100)
  Budžet=150; NLL perioda=20.332975/20.047273/20.358964; prosek=20.24655672; rasipanje=0.141171; stvarne iteracije=(150, 150, 150)
  Budžet=200; NLL perioda=20.349565/20.067229/20.335883; prosek=20.25100793; rasipanje=0.129939; stvarne iteracije=(200, 200, 200)
  Budžet=250; NLL perioda=20.341643/20.052661/20.252105; prosek=20.21551932; rasipanje=0.120712; stvarne iteracije=(241, 250, 231)
  Budžet=300; NLL perioda=20.341643/20.052195/20.252105; prosek=20.21536417; rasipanje=0.120922; stvarne iteracije=(241, 298, 231)
  Budžet=400; NLL perioda=20.341643/20.052195/20.252105; prosek=20.21536417; rasipanje=0.120922; stvarne iteracije=(241, 298, 231)
  Budžet=500; NLL perioda=20.341643/20.052195/20.252105; prosek=20.21536417; rasipanje=0.120922; stvarne iteracije=(241, 298, 231)
  Budžet=650; NLL perioda=20.341643/20.052195/20.252105; prosek=20.21536417; rasipanje=0.120922; stvarne iteracije=(241, 298, 231)
  Budžet=800; NLL perioda=20.341643/20.052195/20.252105; prosek=20.21536417; rasipanje=0.120922; stvarne iteracije=(241, 298, 231)

IZABRANO PREKO TRI VALIDACIONA PERIODA:
  Slojevi: 2
  Regularizacija: 0.01
  Dozvoljeno trajanje: 75 iteracija
  Prosečni validacioni NLL: 20.10417946
  Rasipanje NLL-a između perioda: 0.08092405
  Pojedinačni NLL: 19.99079451, 20.14656974, 20.17488553

UČENJE ZA ODVOJENI TEST NA SVIM PRETEST PRELAZIMA
    Početni cilj=23.93073736; dozvoljeno iteracija=75
    Iteracija 1: cilj=23.04803379; |grad|inf=4.976e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.06174401; |grad|inf=1.179e-01; pad od prethodnog ispisa=2.986e+00
    Iteracija 50: cilj=19.20420605; |grad|inf=1.047e-01; pad od prethodnog ispisa=8.575e-01
    Iteracija 75: cilj=18.98670597; |grad|inf=2.367e-02; pad od prethodnog ispisa=2.175e-01
    Zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 75
    Trening NLL: 18.98060847
    Završni |grad|inf: 2.367e-02
Odvojeni test — NLL: 20.26694961

ZAVRŠNO UČENJE NA CELOJ ISTORIJI
    Početni cilj=23.90231479; dozvoljeno iteracija=75
    Iteracija 1: cilj=23.20286841; |grad|inf=5.236e-01; pad od prethodnog ispisa=—
    Iteracija 25: cilj=20.32046030; |grad|inf=1.713e-01; pad od prethodnog ispisa=2.882e+00
    Iteracija 50: cilj=19.28415865; |grad|inf=6.960e-02; pad od prethodnog ispisa=1.036e+00
    Iteracija 75: cilj=19.04003591; |grad|inf=2.730e-02; pad od prethodnog ispisa=2.441e-01
    Zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 75
    Trening NLL: 19.03395619
    Završni |grad|inf: 2.730e-02

NEXT V4: 1, 14, 15, 18, 20, 22, 29
Logaritamska masa NEXT u modelu: -9.24030977
Promena zajedničkih marginala prema prethodnom kontekstu: 0.2970231799
Qiskit, gradijenti, normalizacija, granice i maksimum: ugrađene provere prošle.
Masa modela nije izmerena uspešnost predikcije.



KONAČNI NEXT REZULTATI — V4
loto7_4708_k85.csv: 6, 12, 14, 17, 22, 33, 37
loto7_4708_k85_loto_2976.csv: 5, 15, 18, 22, 28, 30, 38
loto7_4708_k85_loto_plus_1732.csv: 1, 14, 15, 18, 20, 22, 29



VALIDACIJA I ODVOJENI TEST — manji NLL je bolji
loto7_4708_k85.csv:
  prosečni validacioni NLL=19.95918246
  rasipanje između perioda=0.15546596
  pojedinačni NLL=(20.17869793821576, 19.836615767674743, 19.862562783156584)
  test NLL=20.45179954
  slojevi=1; regularizacija=0.0001
  izabrani budžet=250; test izvršio=250; završni izvršio=250
  završni |grad|inf=1.920e-02
  završno zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
loto7_4708_k85_loto_2976.csv:
  prosečni validacioni NLL=20.07111683
  rasipanje između perioda=0.13113007
  pojedinačni NLL=(20.05620583323649, 20.23880330482682, 19.9187033678565)
  test NLL=20.29079213
  slojevi=2; regularizacija=0.001
  izabrani budžet=150; test izvršio=150; završni izvršio=150
  završni |grad|inf=1.255e-02
  završno zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
loto7_4708_k85_loto_plus_1732.csv:
  prosečni validacioni NLL=20.10417946
  rasipanje između perioda=0.08092405
  pojedinačni NLL=(19.990794508505413, 20.146569739521947, 20.174885530916523)
  test NLL=20.26694961
  slojevi=2; regularizacija=0.01
  izabrani budžet=75; test izvršio=75; završni izvršio=75
  završni |grad|inf=2.730e-02
  završno zaustavljanje: Dostignut dozvoljeni broj iteracija; kriterijum konvergencije nije ispunjen.
"""



"""
V4 rezultate uporediti sa prethodnim rezultatima, posebno validaciju, odvojeni test i završetak optimizacije.

V4 je završio sva tri CSV-a i prema logu prošao ugrađene provere. Rezultat je mešovit — nije najbolji na sva tri testa.
Poređenje test NLL-a, gde je manje bolje:

CSV	                    V2	        V3	        V4
Spojeni — 4.708 kola	20,51741	20,12409	20,45180
Loto — 2.976 kola	    20,00979	20,52109	20,29079
Loto Plus — 1.732 kola	20,40965	20,40364	20,26695


Tri validaciona perioda dala su širu osnovu za izbor parametara, ali nisu donela dosledno poboljšanje na testu. Zato ne mogu opravdano proglasiti V4 najboljom prediktivnom verzijom.
Završni modeli stali su na validacijom izabranim budžetima 250, 150 i 75 iteracija. To je predviđeno ponašanje V4; sama poruka da konvergencija nije dostignuta nije razlog za novu doradu.

V4 ostaje završena implementacija ove metode, sa ovim rezultatima:
- Spojeni: 6, 12, 14, 17, 22, 33, 37
- Loto: 5, 15, 18, 22, 28, 30, 38
- Loto Plus: 1, 14, 15, 18, 20, 22, 29

Dodatno potrošeno računanje nije donelo opšte poboljšanje. 
"""



"""
V4 ima smisla zbog stabilnosti izbora konfiguracije i trajanja učenja.

Poređenje odvojenog testa — manji NLL je bolji:
CSV	                    V2	        V3	        Promena
Spojeni — 4.708 kola	20,51741	20,12409	Poboljšanje
Loto — 2.976 kola	    20,00979	20,52109	Pogoršanje
Loto Plus — 1.732 kola	20,40965	20,40364	Malo poboljšanje

Kod zasebnog Lota validacija je izabrala 75 iteracija, ali rezultat na testu se pogoršao. To pokazuje da bolji rezultat na jednom validacionom periodu nije doneo dosledno poboljšanje. Prenošenje istog broja iteracija na veći trening skup takođe može dati drugačiji stepen naučenosti.
Za V4 nekoliko hronoloških validacionih perioda sa rastućim trening skupom. Konfiguraciju i trajanje birao bih prema njihovom zajedničkom rezultatu, umesto prema jednom periodu. Završni model ponovo bi učio na celoj istoriji.
Šest kubita po poziciji, amplitude, kvantne operacije i tačan maksimum ostaju. Prema logu, matematičke provere prolaze; opravdana sledeća dorada jeste pouzdaniji izbor parametara, bez unapred obećanog poboljšanja.

Tri hronološka validaciona perioda sa rastućim trening skupom biraće konfiguraciju i trajanje učenja prema prosečnom NLL-u. Završnih 15% kola ostaje izvan tog izbora. Kvantni model i šest kubita po poziciji ostaju isti.

V4 bira konfiguraciju i trajanje preko tri hronološka validaciona perioda. Test ne učestvuje u izboru, a završno učenje koristi celu istoriju svakog CSV-a.
Šest kubita po poziciji i kvantni model iz V3 ostaju. Pretraga je duža: devet putanja učenja po CSV-u. 

- Svaki validacioni period koristi samo prethodnu istoriju, sa zasebnom standardizacijom.
- Konfiguracija i trajanje biraju se zajedničkim rezultatom tri perioda; test ne učestvuje.
- Ranija konvergencija pravilno se prenosi na veće dozvoljene budžete.
- Završno učenje koristi sve istorijske prelaze.
- Normalizacija, gradijenti i tačan izbor sedmorke zadržani su iz prethodne konstrukcije.

Glavni trošak V4 je računanje: devet putanja učenja po CSV-u, svaka do 800 iteracija ili ranije konvergencije. To je posledica uvedene validacije kroz tri perioda.
"""

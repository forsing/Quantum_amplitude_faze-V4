#!/usr/bin/env python3

# Quantum_amplitude_faze V3


"""
Loto 7/39 — Quantum Amplitude NEXT V3.

LOKALNI SIMULATOR:
    Sedam pozicija, po šest kubita.
    Registri se računaju zasebno; nema stanja od 42 kubita.
    Amplitudno kodiranje, RZ/RY, CNOT i Bornovo pravilo.

ZADRŽANO IZ V2:
    Zajednička raspodela nad rastućim sedmorkama.
    Analitički gradijenti.
    Tačna normalizacija, marginale i maksimum.
    Bez slučajnog generisanja, popravljanja ili filtriranja NEXT.

NOVO U V3:
    Hronološka validacija bira konfiguraciju i broj iteracija.
    Čuvaju se parametri sa najmanjim validacionim NLL.
    Pretraga može trajati do 800 iteracija.
    Zaustavljanje pretrage pri dužem izostanku poboljšanja validacije.
    Ispis norme gradijenta, promene cilja i razloga zaustavljanja.
    Test i završno učenje kreću od determinističkog početka.
    Standardizacija se ponovo uči na pripadajućem trening skupu.

Zavisnosti:
    numpy, scipy, qiskit==1.4.4

Ulaz:
    CSV bez zaglavlja, sedam brojeva po redu.
    Redovi od najstarijeg do najnovijeg kola.
"""


import csv
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

MAX_SEARCH_ITER = 800
CHECK_EVERY = 25
MIN_SEARCH_ITER = 400
PATIENCE_ITER = 200
PATIENCE_DELTA = 1e-5

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
for j in range(POSITIONS):
    # Svaka pozicija ima svih 33 dozvoljenih vrednosti.
    SUPPORT[j, j:NUMBERS - POSITIONS + j + 1] = 1.0


class ValidationStop(Exception):
    """Kontrolisano zaustavljanje pretrage zbog validacije."""


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
                    f"{path}, red {line}: očekivani su celi brojevi "
                    "bez zaglavlja."
                ) from exc

            if len(set(values)) != POSITIONS:
                raise ValueError(f"{path}, red {line}: duplikati.")
            if min(values) < 1 or max(values) > NUMBERS:
                raise ValueError(f"{path}, red {line}: broj van 1–39.")

            # Sortiraju se brojevi unutar kola, nikada redovi.
            rows.append(sorted(values))

    if len(rows) < 100:
        raise ValueError(f"{path}: potrebno je najmanje 100 kola.")

    return np.asarray(rows, dtype=np.int64)


def features(history):
    """Svaki kontekst koristi samo podatke do svog kola."""
    h = history.astype(np.float64)
    count = np.arange(
        1, len(h) + 1, dtype=np.float64
    )[:, None]

    running_mean = np.cumsum(h, axis=0) / count
    running_variance = np.maximum(
        np.cumsum(h * h, axis=0) / count - running_mean ** 2,
        0.0,
    )

    delta = np.zeros_like(h)
    delta[1:] = h[1:] - h[:-1]

    present = np.zeros((len(h), NUMBERS), dtype=np.float64)
    present[
        np.arange(len(h))[:, None], history - 1
    ] = 1.0

    # Iste karakteristike kao V1 i V2:
    # 1 + 7 + 7 + 7 + 39 + 3 = 64.
    raw = np.column_stack((
        np.ones(len(h)),
        h,
        running_mean,
        delta,
        present,
        running_variance[:, [0, 3, 6]],
    ))

    if raw.shape != (len(h), DIM) or not np.isfinite(raw).all():
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
        c = math.cos(angle / 2)
        s = math.sin(angle / 2)

        if derivative:
            return 0.5 * np.array(
                [[-s, -c], [c, -s]], dtype=np.complex128
            )
        return np.array(
            [[c, -s], [s, c]], dtype=np.complex128
        )

    if kind == "rz":
        a = np.exp(-0.5j * angle)
        b = np.exp(0.5j * angle)
        return np.diag(
            [-0.5j * a, 0.5j * b] if derivative else [a, b]
        )

    raise ValueError(kind)


def gate(state, matrix, qubit):
    """Qiskit little-endian redosled."""
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
    state = encoded
    tape = []
    index = 0

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
            state = state[
                :, PERMUTATIONS[(control, target)]
            ]
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
    count = layers * QUBITS * 2
    indices = np.arange(1, count + 1, dtype=np.float64)
    one_register = 0.2 * np.sin(indices * math.sqrt(2.0))

    # Fiksna inicijalizacija; nema slučajnosti ili dodatog šuma.
    return np.tile(one_register, (POSITIONS, 1)).ravel()


def predict_probabilities(encoded, angles, layers):
    parameters = angles.reshape(POSITIONS, -1)
    output = []

    for j in range(POSITIONS):
        state, _ = evolve(encoded, parameters[j], layers)
        weights = (
            np.abs(state[:, :NUMBERS]) ** 2 * SUPPORT[j]
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
        1.0,
        atol=2e-10,
        rtol=0,
    ):
        raise ArithmeticError(
            "Pozicione raspodele nisu normalizovane."
        )

    result = np.full_like(probabilities, -np.inf)
    np.log(
        probabilities,
        out=result,
        where=probabilities > 0,
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
    Za rastuću kombinaciju c:

        P(c | x) = proizvod_j p_j(c_j | x) / Z(x).

    Normalizacija i marginale računaju se tačno.
    """
    forward = np.full_like(logp, -np.inf)
    forward[..., 0, :] = logp[..., 0, :]

    for j in range(1, POSITIONS):
        forward[..., j, :] = (
            logp[..., j, :]
            + prefix_exclusive(forward[..., j - 1, :])
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

    for j in range(POSITIONS - 2, -1, -1):
        backward[..., j, :] = (
            logp[..., j, :]
            + suffix_exclusive(backward[..., j + 1, :])
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
        1.0,
        atol=2e-8,
        rtol=0,
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

    # Čuva se diferencijaciona traka samo jednog registra.
    for j in range(POSITIONS):
        state, tape = evolve(
            encoded, parameters[j], layers, record=True
        )

        weights = np.abs(state[:, :NUMBERS]) ** 2
        coefficient = np.zeros(
            state.shape, dtype=np.float64
        )
        np.divide(
            residual[:, j, :],
            weights,
            out=coefficient[:, :NUMBERS],
            where=weights > 0,
        )

        adjoint = coefficient * state / len(targets)
        gradient[j] = reverse_gradient(
            adjoint, tape, parameters[j]
        )

        del state, tape, coefficient, adjoint

    gradient = gradient.ravel()
    gradient += 2.0 * penalty * angles / angles.size

    if (
        not np.isfinite(loss)
        or not np.isfinite(gradient).all()
    ):
        raise ArithmeticError(
            "Neispravan zajednički cilj ili gradijent."
        )

    return loss, gradient


def check_gradient(encoded, targets, layers, penalty):
    x, y = encoded[:5], targets[:5]
    angles = initial_angles(layers)

    _, analytic = objective(
        angles, x, y, layers, penalty
    )

    width = layers * QUBITS * 2
    indices = {
        j * width + offset
        for j in range(POSITIONS)
        for offset in (0, 1, width - 1)
    }

    step = 1e-6
    for index in sorted(indices):
        plus, minus = angles.copy(), angles.copy()
        plus[index] += step
        minus[index] -= step

        numeric = (
            objective(plus, x, y, layers, penalty)[0]
            - objective(minus, x, y, layers, penalty)[0]
        ) / (2.0 * step)

        if not np.isclose(
            numeric,
            analytic[index],
            atol=3e-6,
            rtol=3e-4,
        ):
            raise ArithmeticError(
                f"Gradijent nije potvrđen: parametar {index}; "
                f"analitički={analytic[index]}, "
                f"numerički={numeric}."
            )


class AmplitudeRegressor:
    def __init__(self, layers, penalty):
        self.layers = layers
        self.penalty = penalty

    def fit(self, raw, targets, budget, validation=None):
        """
        validation:
            Ako postoji, bira se najbolja proverena iteracija.
            Ako ne postoji, koristi se rezultat fiksnog trajanja.

        Validacija nikada ne ulazi u gradijent ili standardizaciju.
        """
        if budget < 1:
            raise ValueError("Trajanje učenja mora biti pozitivno.")

        self.encoder = Encoder().fit(raw)
        encoded = self.encoder.transform(raw)

        if validation is None:
            validation_encoded = None
            validation_targets = None
        else:
            validation_raw, validation_targets = validation
            validation_encoded = self.encoder.transform(
                validation_raw
            )

        check_gradient(
            encoded, targets, self.layers, self.penalty
        )

        initial = initial_angles(self.layers)
        cache = {}
        progress = {
            "iteration": 0,
            "last_observed": None,
            "last_logged_loss": None,
            "best_score": math.inf,
            "best_angles": None,
            "best_iteration": None,
            "patience_reference": math.inf,
            "last_significant_improvement": 0,
        }

        def fun(theta):
            cached_theta = cache.get("theta")
            if (
                cached_theta is not None
                and np.array_equal(theta, cached_theta)
            ):
                return cache["loss"], cache["gradient"]

            value, gradient = objective(
                theta,
                encoded,
                targets,
                self.layers,
                self.penalty,
            )
            cache["theta"] = theta.copy()
            cache["loss"] = value
            cache["gradient"] = gradient.copy()
            return value, gradient

        start_loss, _ = fun(initial)
        print(
            f"    Početni cilj={start_loss:.8f}; "
            f"dozvoljeno iteracija={budget}",
            flush=True,
        )

        def observe(theta, iteration):
            previous_observation = progress["last_observed"]
            if (
                previous_observation is not None
                and previous_observation[0] == iteration
                and np.array_equal(
                    previous_observation[1], theta
                )
            ):
                return

            value, gradient = fun(theta)
            gradient_norm = float(
                np.linalg.norm(gradient, ord=np.inf)
            )

            previous_loss = progress["last_logged_loss"]
            change = (
                math.nan
                if previous_loss is None
                else previous_loss - value
            )

            line = (
                f"    Iteracija {iteration}: "
                f"cilj={value:.8f}; "
                f"|grad|inf={gradient_norm:.3e}; "
                f"pad cilja={change:.3e}"
            )

            if validation_encoded is not None:
                predicted = predict_probabilities(
                    validation_encoded, theta, self.layers
                )
                score = mean_nll(
                    predicted, validation_targets
                )
                line += f"; validacioni NLL={score:.8f}"

                # Čuvaju se stvarno najbolji provereni parametri.
                if score < progress["best_score"]:
                    progress["best_score"] = score
                    progress["best_angles"] = theta.copy()
                    progress["best_iteration"] = iteration

                # Patience se resetuje pri poboljšanju >= delta.
                if (
                    score
                    < progress["patience_reference"]
                    - PATIENCE_DELTA
                ):
                    progress["patience_reference"] = score
                    progress[
                        "last_significant_improvement"
                    ] = iteration

            print(line, flush=True)
            progress["last_logged_loss"] = value
            progress["last_observed"] = (
                iteration, theta.copy()
            )

        def callback(theta):
            progress["iteration"] += 1
            iteration = progress["iteration"]

            if iteration == 1 or iteration % CHECK_EVERY == 0:
                observe(theta, iteration)

                if validation_encoded is not None:
                    waiting = (
                        iteration
                        - progress[
                            "last_significant_improvement"
                        ]
                    )
                    if (
                        iteration >= MIN_SEARCH_ITER
                        and waiting >= PATIENCE_ITER
                    ):
                        raise ValidationStop()

        result = None
        try:
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
        except ValidationStop:
            self.stop_reason = (
                "Pretraga zaustavljena zbog izostanka "
                "značajnog poboljšanja validacije."
            )
            self.optimizer_success = False
            self.actual_iterations = progress["iteration"]
        else:
            self.actual_iterations = int(result.nit)
            self.optimizer_success = bool(result.success)
            observe(result.x, self.actual_iterations)

            if result.success:
                self.stop_reason = (
                    "Ispunjen kriterijum zaustavljanja optimizatora: "
                    + str(result.message)
                )
            elif self.actual_iterations >= budget:
                if validation_encoded is None:
                    self.stop_reason = (
                        "Dostignuto trajanje izabrano validacijom; "
                        "kriterijum konvergencije nije ispunjen."
                    )
                else:
                    self.stop_reason = (
                        "Dostignuta gornja granica pretrage; "
                        "kriterijum konvergencije nije ispunjen."
                    )
            else:
                self.stop_reason = (
                    "Optimizator zaustavljen bez prijavljene "
                    "konvergencije: " + str(result.message)
                )

        if validation_encoded is not None:
            if (
                progress["best_angles"] is None
                or progress["best_iteration"] < 1
            ):
                raise ArithmeticError(
                    "Nije izabrana validna učena tačka."
                )

            self.angles = progress["best_angles"].copy()
            self.selected_iterations = int(
                progress["best_iteration"]
            )
            self.validation_nll = float(
                progress["best_score"]
            )
        else:
            if result is None or result.nit < 1:
                raise ArithmeticError(
                    "Optimizator nije izvršio korak učenja."
                )
            self.angles = result.x.copy()
            self.selected_iterations = int(result.nit)
            self.validation_nll = None

        final_loss, final_gradient = fun(self.angles)
        if (
            not np.isfinite(final_loss)
            or not np.isfinite(self.angles).all()
            or final_loss > start_loss + 1e-8
        ):
            raise ArithmeticError(
                "Odabrana tačka nije validan rezultat učenja."
            )

        self.training_nll = (
            final_loss
            - self.penalty * float(np.mean(self.angles ** 2))
        )
        self.gradient_norm = float(
            np.linalg.norm(final_gradient, ord=np.inf)
        )

        print(
            f"    Zaustavljanje: {self.stop_reason}\n"
            f"    Izvršeno iteracija: {self.actual_iterations}\n"
            f"    Korišćena iteracija: {self.selected_iterations}\n"
            f"    NLL korišćene tačke: {self.training_nll:.8f}\n"
            f"    |grad|inf korišćene tačke: "
            f"{self.gradient_norm:.3e}",
            flush=True,
        )

        if self.validation_nll is not None:
            print(
                f"    Najbolji provereni validacioni NLL: "
                f"{self.validation_nll:.8f}",
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

        for j in range(POSITIONS):
            circuit = QuantumCircuit(QUBITS)
            circuit.initialize(encoded[0], range(QUBITS))
            index = 0

            for _ in range(self.layers):
                for qubit in range(QUBITS):
                    circuit.rz(
                        float(parameters[j, index]), qubit
                    )
                    index += 1
                    circuit.ry(
                        float(parameters[j, index]), qubit
                    )
                    index += 1

                for control, target in CNOTS:
                    circuit.cx(control, target)

            reference = Statevector.from_instruction(
                circuit
            ).data
            calculated, _ = evolve(
                encoded, parameters[j], self.layers
            )

            if not np.allclose(
                reference,
                calculated[0],
                atol=2e-10,
                rtol=0,
            ):
                raise ArithmeticError(
                    f"Qiskit neslaganje: pozicija {j + 1}."
                )

            if not np.isclose(
                np.sum(np.abs(calculated[0]) ** 2),
                1.0,
                atol=2e-10,
                rtol=0,
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

    for j in range(1, POSITIONS):
        updated = np.full(NUMBERS, -np.inf)
        best_value, best_index = -np.inf, -1

        for k in range(NUMBERS):
            previous = k - 1

            if previous >= 0 and values[previous] > best_value:
                best_value = values[previous]
                best_index = previous

            if best_index >= 0 and np.isfinite(logp[j, k]):
                updated[k] = best_value + logp[j, k]
                parents[j, k] = best_index

        values = updated

    last = int(np.argmax(values))
    log_maximum = float(values[last])

    if not np.isfinite(log_maximum):
        raise ArithmeticError(
            "Nije pronađen maksimum pozitivne mase."
        )

    answer = [last]
    for j in range(POSITIONS - 1, 0, -1):
        last = int(parents[j, last])
        if last < 0:
            raise ArithmeticError(
                "Neispravna putanja maksimuma."
            )
        answer.append(last)

    answer = tuple(k + 1 for k in reversed(answer))

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
    """Donje i gornje granične sedmorke."""
    for answer in (
        tuple(range(1, 8)),
        tuple(range(33, 40)),
    ):
        p = np.zeros((POSITIONS, NUMBERS))
        p[np.arange(POSITIONS), np.array(answer) - 1] = 1.0

        predicted, log_mass = exact_next(p)
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


def run_file(path):
    history = read_history(path)
    raw = features(history)

    # Svi istorijski prelazi ostaju raspoloživi.
    x, y = raw[:-1], history[1:]
    n = len(y)
    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)

    if min(
        train_end,
        validation_end - train_end,
        n - validation_end,
    ) < 10:
        raise ValueError(
            "Premalo uzoraka za hronološku podelu."
        )

    print(f"\nV3 — FAJL: {path}", flush=True)
    print(
        f"Kola: {len(history)}; prelazi: {n}; "
        f"trening/validacija/test: "
        f"{train_end}/"
        f"{validation_end - train_end}/"
        f"{n - validation_end}",
        flush=True,
    )

    best = None

    for layers, penalty in CONFIGS:
        print(
            f"\nValidaciona pretraga: slojevi={layers}, "
            f"regularizacija={penalty:g}",
            flush=True,
        )

        candidate = AmplitudeRegressor(
            layers, penalty
        ).fit(
            x[:train_end],
            y[:train_end],
            budget=MAX_SEARCH_ITER,
            validation=(
                x[train_end:validation_end],
                y[train_end:validation_end],
            ),
        )

        choice = {
            "layers": layers,
            "penalty": penalty,
            "iterations": candidate.selected_iterations,
            "validation_nll": candidate.validation_nll,
        }

        if (
            best is None
            or choice["validation_nll"] < best["validation_nll"]
        ):
            best = choice

        del candidate

    if best is None:
        raise ArithmeticError(
            "Nije izabrana validna konfiguracija."
        )

    layers = best["layers"]
    penalty = best["penalty"]
    budget = best["iterations"]

    print(
        "\nIZABRANO ISKLJUČIVO VALIDACIJOM:\n"
        f"  Slojevi: {layers}\n"
        f"  Regularizacija: {penalty:g}\n"
        f"  Trajanje: {budget} iteracija\n"
        f"  Validacioni NLL: {best['validation_nll']:.8f}",
        flush=True,
    )

    # Nova standardizacija i deterministički početak.
    # Test nije korišćen za konfiguraciju ili trajanje.
    print(
        "\nUčenje na treningu i validaciji za odvojeni test:",
        flush=True,
    )
    test_model = AmplitudeRegressor(
        layers, penalty
    ).fit(
        x[:validation_end],
        y[:validation_end],
        budget=budget,
    )

    test_probabilities = test_model.predict(
        x[validation_end:]
    )
    test_nll = mean_nll(
        test_probabilities, y[validation_end:]
    )

    print(
        f"Odvojeni test — NLL: {test_nll:.8f}",
        flush=True,
    )

    test_iterations = test_model.actual_iterations
    del test_model, test_probabilities

    # Ponovo novi početak, sada na CELOJ istoriji.
    print(
        "\nZavršno učenje na SVIM istorijskim prelazima:",
        flush=True,
    )
    final_model = AmplitudeRegressor(
        layers, penalty
    ).fit(x, y, budget=budget)

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
            "PROVERA: zajedničke marginale se gotovo ne menjaju "
            "između poslednja dva istorijska konteksta.",
            flush=True,
        )

    if any(
        all(answer[j + k] == answer[j] + k for k in range(5))
        for j in range(POSITIONS - 4)
    ):
        print(
            "PROVERA: najmanje pet uzastopnih brojeva u NEXT. "
            "Rezultat nije filtriran; pregledati ponašanje modela.",
            flush=True,
        )

    repeats = int(np.sum(np.all(
        history == np.asarray(answer), axis=1
    )))
    if repeats:
        print(
            f"PROVERA: NEXT se pojavljuje u istoriji {repeats} puta. "
            "Rezultat nije menjan.",
            flush=True,
        )

    print(
        "\nNEXT V3: " + ", ".join(map(str, answer))
        + f"\nLogaritamska masa NEXT u modelu: {log_mass:.8f}"
        + "\nPromena zajedničkih marginala prema prethodnom "
        + f"kontekstu: {context_change:.10g}"
        + "\nQiskit, gradijenti, normalizacija, granice i "
        + "maksimum: ugrađene provere prošle."
        + "\nMasa modela nije izmerena uspešnost predikcije.",
        flush=True,
    )

    return {
        "path": path,
        "answer": answer,
        "marginals": marginals,
        "context_change": context_change,
        "validation_nll": best["validation_nll"],
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
        over="raise",
        invalid="raise",
        divide="raise",
    )

    check_boundaries()
    results = [run_file(path) for path in PATHS]

    print("\nKONAČNI NEXT REZULTATI — V3", flush=True)
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
            f"  validacija={item['validation_nll']:.8f}; "
            f"test={item['test_nll']:.8f}\n"
            f"  slojevi={item['layers']}; "
            f"regularizacija={item['penalty']:g}\n"
            f"  izabrano trajanje={item['budget']}; "
            f"test izvršio={item['test_iterations']}; "
            f"završni izvršio={item['final_iterations']}\n"
            f"  završni |grad|inf={item['final_gradient']:.3e}\n"
            f"  završno zaustavljanje: {item['final_stop']}",
            flush=True,
        )

    for i, first in enumerate(results):
        for second in results[i + 1:]:
            if first["answer"] != second["answer"]:
                continue

            difference = float(np.max(np.abs(
                first["marginals"] - second["marginals"]
            )))

            print(
                "\nPROVERA ISTIH NEXT:\n"
                f"  {Path(first['path']).name}\n"
                f"  {Path(second['path']).name}\n"
                f"  Najveća razlika zajedničkih marginala: "
                f"{difference:.10g}\n"
                f"  Promene konteksta: "
                f"{first['context_change']:.10g}, "
                f"{second['context_change']:.10g}\n"
                "  Ulazi i modeli obrađeni su zasebno. "
                "Rezultati nisu menjani radi različitosti. "
                "Ove provere same ne isključuju degeneraciju.",
                flush=True,
            )


if __name__ == "__main__":
    main()



"""
V3 — FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv
Kola: 4708; prelazi: 4707; trening/validacija/test: 3294/706/707

Validaciona pretraga: slojevi=1, regularizacija=0.0001
    Početni cilj=23.53874470; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.83206034; |grad|inf=3.569e-01; pad cilja=nan; validacioni NLL=22.28702390
    Iteracija 25: cilj=20.93122869; |grad|inf=1.115e-01; pad cilja=1.901e+00; validacioni NLL=20.91995474
    Iteracija 50: cilj=20.66672472; |grad|inf=2.576e-01; pad cilja=2.645e-01; validacioni NLL=20.83488979
    Iteracija 75: cilj=20.63808738; |grad|inf=1.899e-02; pad cilja=2.864e-02; validacioni NLL=20.80121987
    Iteracija 100: cilj=20.15220293; |grad|inf=5.195e-02; pad cilja=4.859e-01; validacioni NLL=20.42651034
    Iteracija 125: cilj=19.98577595; |grad|inf=9.935e-03; pad cilja=1.664e-01; validacioni NLL=20.21420946
    Iteracija 150: cilj=19.97154744; |grad|inf=7.300e-03; pad cilja=1.423e-02; validacioni NLL=20.22485984
    Iteracija 175: cilj=19.96792866; |grad|inf=3.155e-03; pad cilja=3.619e-03; validacioni NLL=20.22315393
    Iteracija 200: cilj=19.85688555; |grad|inf=6.020e-02; pad cilja=1.110e-01; validacioni NLL=20.15971081
    Iteracija 225: cilj=19.78743230; |grad|inf=6.782e-03; pad cilja=6.945e-02; validacioni NLL=20.13259172
    Iteracija 250: cilj=19.78338186; |grad|inf=7.436e-03; pad cilja=4.050e-03; validacioni NLL=20.11529256
    Iteracija 275: cilj=19.77707526; |grad|inf=4.949e-03; pad cilja=6.307e-03; validacioni NLL=20.08088010
    Iteracija 300: cilj=19.76116411; |grad|inf=7.031e-03; pad cilja=1.591e-02; validacioni NLL=20.07689721
    Iteracija 325: cilj=19.75817006; |grad|inf=1.914e-03; pad cilja=2.994e-03; validacioni NLL=20.10312463
    Iteracija 350: cilj=19.75723941; |grad|inf=2.254e-03; pad cilja=9.306e-04; validacioni NLL=20.10006873
    Iteracija 375: cilj=19.75695402; |grad|inf=9.747e-04; pad cilja=2.854e-04; validacioni NLL=20.09851963
    Iteracija 400: cilj=19.75690110; |grad|inf=7.439e-05; pad cilja=5.291e-05; validacioni NLL=20.09676272
    Iteracija 425: cilj=19.75689961; |grad|inf=5.243e-05; pad cilja=1.492e-06; validacioni NLL=20.09695638
    Iteracija 450: cilj=19.75689753; |grad|inf=3.145e-04; pad cilja=2.085e-06; validacioni NLL=20.09686740
    Iteracija 475: cilj=19.75689468; |grad|inf=2.811e-04; pad cilja=2.843e-06; validacioni NLL=20.09708990
    Iteracija 500: cilj=19.75689237; |grad|inf=3.537e-05; pad cilja=2.317e-06; validacioni NLL=20.09700687
    Zaustavljanje: Pretraga zaustavljena zbog izostanka značajnog poboljšanja validacije.
    Izvršeno iteracija: 500
    Korišćena iteracija: 300
    NLL korišćene tačke: 19.76102611
    |grad|inf korišćene tačke: 7.031e-03
    Najbolji provereni validacioni NLL: 20.07689721

Validaciona pretraga: slojevi=2, regularizacija=0.001
    Početni cilj=23.93891982; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.09650707; |grad|inf=5.380e-01; pad cilja=nan; validacioni NLL=23.30448995
    Iteracija 25: cilj=20.59336753; |grad|inf=1.579e-01; pad cilja=2.503e+00; validacioni NLL=20.82441681
    Iteracija 50: cilj=19.64995461; |grad|inf=1.911e-02; pad cilja=9.434e-01; validacioni NLL=20.27220979
    Iteracija 75: cilj=19.54907385; |grad|inf=1.894e-02; pad cilja=1.009e-01; validacioni NLL=20.29100701
    Iteracija 100: cilj=19.50167764; |grad|inf=6.116e-03; pad cilja=4.740e-02; validacioni NLL=20.28887579
    Iteracija 125: cilj=19.48143555; |grad|inf=1.485e-02; pad cilja=2.024e-02; validacioni NLL=20.24078171
    Iteracija 150: cilj=19.47746177; |grad|inf=3.423e-03; pad cilja=3.974e-03; validacioni NLL=20.22323163
    Iteracija 175: cilj=19.47358496; |grad|inf=1.296e-03; pad cilja=3.877e-03; validacioni NLL=20.24146786
    Iteracija 200: cilj=19.47321775; |grad|inf=7.635e-04; pad cilja=3.672e-04; validacioni NLL=20.25073781
    Iteracija 225: cilj=19.47127293; |grad|inf=7.443e-04; pad cilja=1.945e-03; validacioni NLL=20.23338938
    Iteracija 250: cilj=19.47021756; |grad|inf=7.211e-04; pad cilja=1.055e-03; validacioni NLL=20.24675743
    Iteracija 275: cilj=19.46346511; |grad|inf=1.595e-02; pad cilja=6.752e-03; validacioni NLL=20.26250591
    Iteracija 300: cilj=19.46038056; |grad|inf=4.054e-04; pad cilja=3.085e-03; validacioni NLL=20.26670868
    Iteracija 325: cilj=19.46034940; |grad|inf=1.404e-04; pad cilja=3.116e-05; validacioni NLL=20.26892157
    Iteracija 350: cilj=19.46034365; |grad|inf=8.383e-05; pad cilja=5.748e-06; validacioni NLL=20.26813372
    Iteracija 375: cilj=19.46031774; |grad|inf=1.485e-04; pad cilja=2.590e-05; validacioni NLL=20.27051653
    Iteracija 400: cilj=19.46030806; |grad|inf=3.723e-05; pad cilja=9.686e-06; validacioni NLL=20.26949103
    Zaustavljanje: Pretraga zaustavljena zbog izostanka značajnog poboljšanja validacije.
    Izvršeno iteracija: 400
    Korišćena iteracija: 150
    NLL korišćene tačke: 19.47686098
    |grad|inf korišćene tačke: 3.423e-03
    Najbolji provereni validacioni NLL: 20.22323163

Validaciona pretraga: slojevi=2, regularizacija=0.01
    Početni cilj=23.93910288; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.09666180; |grad|inf=5.371e-01; pad cilja=nan; validacioni NLL=23.30445946
    Iteracija 25: cilj=20.80245302; |grad|inf=1.936e-01; pad cilja=2.294e+00; validacioni NLL=21.07058571
    Iteracija 50: cilj=19.55699524; |grad|inf=3.811e-02; pad cilja=1.245e+00; validacioni NLL=20.30629374
    Iteracija 75: cilj=19.36450133; |grad|inf=3.266e-02; pad cilja=1.925e-01; validacioni NLL=20.33639011
    Iteracija 100: cilj=19.27558977; |grad|inf=8.968e-03; pad cilja=8.891e-02; validacioni NLL=20.37230407
    Iteracija 125: cilj=19.25740144; |grad|inf=3.679e-03; pad cilja=1.819e-02; validacioni NLL=20.36089512
    Iteracija 150: cilj=19.24959536; |grad|inf=1.335e-02; pad cilja=7.806e-03; validacioni NLL=20.35833722
    Iteracija 175: cilj=19.24802718; |grad|inf=2.461e-04; pad cilja=1.568e-03; validacioni NLL=20.36175289
    Iteracija 200: cilj=19.24802416; |grad|inf=1.204e-05; pad cilja=3.021e-06; validacioni NLL=20.36238464
    Iteracija 205: cilj=19.24802415; |grad|inf=7.303e-06; pad cilja=1.061e-08; validacioni NLL=20.36234187
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 205
    Korišćena iteracija: 50
    NLL korišćene tačke: 19.54989745
    |grad|inf korišćene tačke: 3.811e-02
    Najbolji provereni validacioni NLL: 20.30629374

IZABRANO ISKLJUČIVO VALIDACIJOM:
  Slojevi: 1
  Regularizacija: 0.0001
  Trajanje: 300 iteracija
  Validacioni NLL: 20.07689721

Učenje na treningu i validaciji za odvojeni test:
    Početni cilj=23.50849465; dozvoljeno iteracija=300
    Iteracija 1: cilj=22.69888206; |grad|inf=5.794e-01; pad cilja=nan
    Iteracija 25: cilj=21.24998720; |grad|inf=2.014e-01; pad cilja=1.449e+00
    Iteracija 50: cilj=20.95765242; |grad|inf=1.664e-01; pad cilja=2.923e-01
    Iteracija 75: cilj=20.68254818; |grad|inf=3.102e-02; pad cilja=2.751e-01
    Iteracija 100: cilj=20.35557656; |grad|inf=1.593e-02; pad cilja=3.270e-01
    Iteracija 125: cilj=20.08434398; |grad|inf=2.986e-02; pad cilja=2.712e-01
    Iteracija 150: cilj=20.04306053; |grad|inf=2.360e-02; pad cilja=4.128e-02
    Iteracija 175: cilj=20.02640766; |grad|inf=1.176e-02; pad cilja=1.665e-02
    Iteracija 200: cilj=20.02219999; |grad|inf=5.962e-03; pad cilja=4.208e-03
    Iteracija 225: cilj=19.97612086; |grad|inf=5.477e-02; pad cilja=4.608e-02
    Iteracija 250: cilj=19.69944473; |grad|inf=7.181e-02; pad cilja=2.767e-01
    Iteracija 275: cilj=19.63656736; |grad|inf=1.972e-02; pad cilja=6.288e-02
    Iteracija 300: cilj=19.62640131; |grad|inf=3.406e-03; pad cilja=1.017e-02
    Zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 300
    Korišćena iteracija: 300
    NLL korišćene tačke: 19.62624283
    |grad|inf korišćene tačke: 3.406e-03
Odvojeni test — NLL: 20.12409468

Završno učenje na SVIM istorijskim prelazima:
    Početni cilj=23.53425883; dozvoljeno iteracija=300
    Iteracija 1: cilj=22.73438862; |grad|inf=8.772e-01; pad cilja=nan
    Iteracija 25: cilj=21.43157776; |grad|inf=2.552e-01; pad cilja=1.303e+00
    Iteracija 50: cilj=20.88606983; |grad|inf=1.713e-01; pad cilja=5.455e-01
    Iteracija 75: cilj=20.58580422; |grad|inf=1.807e-02; pad cilja=3.003e-01
    Iteracija 100: cilj=20.49210299; |grad|inf=1.522e-02; pad cilja=9.370e-02
    Iteracija 125: cilj=20.40826175; |grad|inf=4.161e-02; pad cilja=8.384e-02
    Iteracija 150: cilj=20.20062753; |grad|inf=4.776e-03; pad cilja=2.076e-01
    Iteracija 175: cilj=20.18948582; |grad|inf=6.002e-02; pad cilja=1.114e-02
    Iteracija 200: cilj=20.14632990; |grad|inf=7.136e-03; pad cilja=4.316e-02
    Iteracija 225: cilj=20.03652519; |grad|inf=1.139e-02; pad cilja=1.098e-01
    Iteracija 250: cilj=20.00147099; |grad|inf=1.920e-02; pad cilja=3.505e-02
    Iteracija 275: cilj=19.98215318; |grad|inf=1.774e-02; pad cilja=1.932e-02
    Iteracija 300: cilj=19.97358957; |grad|inf=7.663e-04; pad cilja=8.564e-03
    Zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 300
    Korišćena iteracija: 300
    NLL korišćene tačke: 19.97345854
    |grad|inf korišćene tačke: 7.663e-04

NEXT V3: 6, 12, 14, 17, 22, 33, 37
Logaritamska masa NEXT u modelu: -7.93797889
Promena zajedničkih marginala prema prethodnom kontekstu: 0.582638383
Qiskit, gradijenti, normalizacija, granice i maksimum: ugrađene provere prošle.
Masa modela nije izmerena uspešnost predikcije.



V3 — FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv
Kola: 2976; prelazi: 2975; trening/validacija/test: 2082/446/447

Validaciona pretraga: slojevi=1, regularizacija=0.0001
    Početni cilj=23.46817453; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.60919406; |grad|inf=7.510e-01; pad cilja=nan; validacioni NLL=22.53701093
    Iteracija 25: cilj=21.66921764; |grad|inf=2.291e-01; pad cilja=9.400e-01; validacioni NLL=21.62119139
    Iteracija 50: cilj=21.20158489; |grad|inf=1.170e-01; pad cilja=4.676e-01; validacioni NLL=21.44762279
    Iteracija 75: cilj=20.97088848; |grad|inf=6.836e-02; pad cilja=2.307e-01; validacioni NLL=21.30229468
    Iteracija 100: cilj=20.80944218; |grad|inf=3.612e-02; pad cilja=1.614e-01; validacioni NLL=21.29044342
    Iteracija 125: cilj=20.43982888; |grad|inf=6.861e-02; pad cilja=3.696e-01; validacioni NLL=20.93122678
    Iteracija 150: cilj=20.35775652; |grad|inf=1.413e-01; pad cilja=8.207e-02; validacioni NLL=20.79865362
    Iteracija 175: cilj=19.81155449; |grad|inf=2.989e-02; pad cilja=5.462e-01; validacioni NLL=20.47650267
    Iteracija 200: cilj=19.75556710; |grad|inf=1.273e-02; pad cilja=5.599e-02; validacioni NLL=20.46228363
    Iteracija 225: cilj=19.63862689; |grad|inf=4.184e-02; pad cilja=1.169e-01; validacioni NLL=20.39292486
    Iteracija 250: cilj=19.45223524; |grad|inf=2.270e-02; pad cilja=1.864e-01; validacioni NLL=20.38871243
    Iteracija 275: cilj=19.43355668; |grad|inf=2.154e-03; pad cilja=1.868e-02; validacioni NLL=20.34174505
    Iteracija 300: cilj=19.42980747; |grad|inf=1.370e-02; pad cilja=3.749e-03; validacioni NLL=20.40167454
    Iteracija 325: cilj=19.42703953; |grad|inf=2.006e-03; pad cilja=2.768e-03; validacioni NLL=20.40246416
    Iteracija 350: cilj=19.41978209; |grad|inf=2.479e-03; pad cilja=7.257e-03; validacioni NLL=20.38359824
    Iteracija 375: cilj=19.41874408; |grad|inf=1.784e-03; pad cilja=1.038e-03; validacioni NLL=20.41516610
    Iteracija 400: cilj=19.41846380; |grad|inf=6.211e-04; pad cilja=2.803e-04; validacioni NLL=20.40896946
    Iteracija 425: cilj=19.41833692; |grad|inf=6.096e-04; pad cilja=1.269e-04; validacioni NLL=20.40763826
    Iteracija 450: cilj=19.41832167; |grad|inf=2.022e-04; pad cilja=1.525e-05; validacioni NLL=20.41012002
    Iteracija 475: cilj=19.41832067; |grad|inf=1.659e-05; pad cilja=1.004e-06; validacioni NLL=20.41029230
    Zaustavljanje: Pretraga zaustavljena zbog izostanka značajnog poboljšanja validacije.
    Izvršeno iteracija: 475
    Korišćena iteracija: 275
    NLL korišćene tačke: 19.43339811
    |grad|inf korišćene tačke: 2.154e-03
    Najbolji provereni validacioni NLL: 20.34174505

Validaciona pretraga: slojevi=2, regularizacija=0.001
    Početni cilj=24.09695052; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.17426527; |grad|inf=4.337e-01; pad cilja=nan; validacioni NLL=23.50974782
    Iteracija 25: cilj=19.91031170; |grad|inf=2.197e-01; pad cilja=3.264e+00; validacioni NLL=20.54519585
    Iteracija 50: cilj=19.26039102; |grad|inf=6.127e-02; pad cilja=6.499e-01; validacioni NLL=20.22347247
    Iteracija 75: cilj=19.14713001; |grad|inf=4.520e-02; pad cilja=1.133e-01; validacioni NLL=20.09679790
    Iteracija 100: cilj=19.10185240; |grad|inf=8.019e-03; pad cilja=4.528e-02; validacioni NLL=20.02531681
    Iteracija 125: cilj=19.09532935; |grad|inf=5.893e-03; pad cilja=6.523e-03; validacioni NLL=19.99661632
    Iteracija 150: cilj=19.09260078; |grad|inf=3.270e-04; pad cilja=2.729e-03; validacioni NLL=20.00094378
    Iteracija 175: cilj=19.09256490; |grad|inf=2.112e-04; pad cilja=3.589e-05; validacioni NLL=20.00048288
    Iteracija 200: cilj=19.09253752; |grad|inf=5.562e-05; pad cilja=2.738e-05; validacioni NLL=20.00334899
    Iteracija 217: cilj=19.09253667; |grad|inf=4.293e-06; pad cilja=8.430e-07; validacioni NLL=20.00348144
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 217
    Korišćena iteracija: 125
    NLL korišćene tačke: 19.09454491
    |grad|inf korišćene tačke: 5.893e-03
    Najbolji provereni validacioni NLL: 19.99661632

Validaciona pretraga: slojevi=2, regularizacija=0.01
    Početni cilj=24.09713358; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.17444739; |grad|inf=4.336e-01; pad cilja=nan; validacioni NLL=23.50973524
    Iteracija 25: cilj=19.91907720; |grad|inf=9.529e-02; pad cilja=3.255e+00; validacioni NLL=20.80767769
    Iteracija 50: cilj=19.13075875; |grad|inf=1.887e-02; pad cilja=7.883e-01; validacioni NLL=20.18208398
    Iteracija 75: cilj=19.02185401; |grad|inf=3.911e-02; pad cilja=1.089e-01; validacioni NLL=19.96554908
    Iteracija 100: cilj=18.98440381; |grad|inf=4.207e-03; pad cilja=3.745e-02; validacioni NLL=20.04406916
    Iteracija 125: cilj=18.98079614; |grad|inf=2.242e-03; pad cilja=3.608e-03; validacioni NLL=20.06192682
    Iteracija 150: cilj=18.97515508; |grad|inf=9.994e-03; pad cilja=5.641e-03; validacioni NLL=20.11808949
    Iteracija 175: cilj=18.95891767; |grad|inf=4.727e-03; pad cilja=1.624e-02; validacioni NLL=20.08619730
    Iteracija 200: cilj=18.95844983; |grad|inf=1.926e-04; pad cilja=4.678e-04; validacioni NLL=20.08869216
    Iteracija 225: cilj=18.95838940; |grad|inf=8.946e-05; pad cilja=6.043e-05; validacioni NLL=20.08722447
    Iteracija 240: cilj=18.95838865; |grad|inf=1.326e-05; pad cilja=7.514e-07; validacioni NLL=20.08699015
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 240
    Korišćena iteracija: 75
    NLL korišćene tačke: 19.01455704
    |grad|inf korišćene tačke: 3.911e-02
    Najbolji provereni validacioni NLL: 19.96554908

IZABRANO ISKLJUČIVO VALIDACIJOM:
  Slojevi: 2
  Regularizacija: 0.01
  Trajanje: 75 iteracija
  Validacioni NLL: 19.96554908

Učenje na treningu i validaciji za odvojeni test:
    Početni cilj=24.11649796; dozvoljeno iteracija=75
    Iteracija 1: cilj=23.23585241; |grad|inf=3.432e-01; pad cilja=nan
    Iteracija 25: cilj=20.40138293; |grad|inf=2.678e-01; pad cilja=2.834e+00
    Iteracija 50: cilj=19.43554516; |grad|inf=1.908e-02; pad cilja=9.658e-01
    Iteracija 75: cilj=19.29841691; |grad|inf=1.992e-02; pad cilja=1.371e-01
    Zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 75
    Korišćena iteracija: 75
    NLL korišćene tačke: 19.29148207
    |grad|inf korišćene tačke: 1.992e-02
Odvojeni test — NLL: 20.52108775

Završno učenje na SVIM istorijskim prelazima:
    Početni cilj=24.05963390; dozvoljeno iteracija=75
    Iteracija 1: cilj=23.22596961; |grad|inf=3.522e-01; pad cilja=nan
    Iteracija 25: cilj=19.97261970; |grad|inf=1.220e-01; pad cilja=3.253e+00
    Iteracija 50: cilj=19.53255805; |grad|inf=4.325e-02; pad cilja=4.401e-01
    Iteracija 75: cilj=19.35584533; |grad|inf=1.747e-02; pad cilja=1.767e-01
    Zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 75
    Korišćena iteracija: 75
    NLL korišćene tačke: 19.34905175
    |grad|inf korišćene tačke: 1.747e-02

NEXT V3: 15, 17, 31, 34, 37, 38, 39
Logaritamska masa NEXT u modelu: -9.17720816
Promena zajedničkih marginala prema prethodnom kontekstu: 0.3257422332
Qiskit, gradijenti, normalizacija, granice i maksimum: ugrađene provere prošle.
Masa modela nije izmerena uspešnost predikcije.



V3 — FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv
Kola: 1732; prelazi: 1731; trening/validacija/test: 1211/260/260

Validaciona pretraga: slojevi=1, regularizacija=0.0001
    Početni cilj=23.50951353; dozvoljeno iteracija=800
    Iteracija 1: cilj=22.57999837; |grad|inf=5.743e-01; pad cilja=nan; validacioni NLL=23.05290969
    Iteracija 25: cilj=21.52356458; |grad|inf=1.879e-01; pad cilja=1.056e+00; validacioni NLL=22.21814665
    Iteracija 50: cilj=21.11125072; |grad|inf=1.483e-01; pad cilja=4.123e-01; validacioni NLL=21.45800794
    Iteracija 75: cilj=20.61887524; |grad|inf=2.892e-02; pad cilja=4.924e-01; validacioni NLL=21.08467457
    Iteracija 100: cilj=20.50112301; |grad|inf=2.740e-02; pad cilja=1.178e-01; validacioni NLL=21.14014410
    Iteracija 125: cilj=20.30298255; |grad|inf=1.924e-02; pad cilja=1.981e-01; validacioni NLL=20.97532621
    Iteracija 150: cilj=20.01785120; |grad|inf=8.311e-02; pad cilja=2.851e-01; validacioni NLL=20.78578941
    Iteracija 175: cilj=19.97904024; |grad|inf=2.906e-02; pad cilja=3.881e-02; validacioni NLL=20.73235829
    Iteracija 200: cilj=19.86443978; |grad|inf=1.266e-01; pad cilja=1.146e-01; validacioni NLL=20.45146038
    Iteracija 225: cilj=19.55760054; |grad|inf=1.724e-01; pad cilja=3.068e-01; validacioni NLL=20.39591479
    Iteracija 250: cilj=19.46747354; |grad|inf=1.418e-02; pad cilja=9.013e-02; validacioni NLL=20.26938021
    Iteracija 275: cilj=19.40237594; |grad|inf=7.538e-03; pad cilja=6.510e-02; validacioni NLL=20.18593303
    Iteracija 300: cilj=19.39717288; |grad|inf=1.313e-02; pad cilja=5.203e-03; validacioni NLL=20.22367003
    Iteracija 325: cilj=19.36750706; |grad|inf=1.517e-02; pad cilja=2.967e-02; validacioni NLL=20.30737798
    Iteracija 350: cilj=19.36538965; |grad|inf=3.149e-04; pad cilja=2.117e-03; validacioni NLL=20.27219008
    Iteracija 375: cilj=19.36538225; |grad|inf=1.358e-04; pad cilja=7.399e-06; validacioni NLL=20.27303225
    Iteracija 389: cilj=19.36538167; |grad|inf=3.322e-05; pad cilja=5.779e-07; validacioni NLL=20.27263263
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 389
    Korišćena iteracija: 275
    NLL korišćene tačke: 19.40223789
    |grad|inf korišćene tačke: 7.538e-03
    Najbolji provereni validacioni NLL: 20.18593303

Validaciona pretraga: slojevi=2, regularizacija=0.001
    Početni cilj=23.81618092; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.12161056; |grad|inf=6.624e-01; pad cilja=nan; validacioni NLL=22.85124307
    Iteracija 25: cilj=19.49102503; |grad|inf=1.135e-01; pad cilja=3.631e+00; validacioni NLL=20.48494872
    Iteracija 50: cilj=18.90657650; |grad|inf=2.717e-02; pad cilja=5.844e-01; validacioni NLL=20.36335364
    Iteracija 75: cilj=18.73992578; |grad|inf=1.003e-02; pad cilja=1.667e-01; validacioni NLL=20.34384795
    Iteracija 100: cilj=18.66738534; |grad|inf=6.285e-03; pad cilja=7.254e-02; validacioni NLL=20.29843883
    Iteracija 125: cilj=18.65240506; |grad|inf=3.730e-03; pad cilja=1.498e-02; validacioni NLL=20.36382012
    Iteracija 150: cilj=18.64789492; |grad|inf=5.816e-04; pad cilja=4.510e-03; validacioni NLL=20.34427952
    Iteracija 175: cilj=18.64226499; |grad|inf=9.548e-03; pad cilja=5.630e-03; validacioni NLL=20.33448045
    Iteracija 200: cilj=18.63538485; |grad|inf=5.941e-03; pad cilja=6.880e-03; validacioni NLL=20.34908408
    Iteracija 225: cilj=18.62093337; |grad|inf=9.255e-04; pad cilja=1.445e-02; validacioni NLL=20.36468527
    Iteracija 250: cilj=18.61720652; |grad|inf=1.818e-03; pad cilja=3.727e-03; validacioni NLL=20.37697737
    Iteracija 275: cilj=18.61711393; |grad|inf=1.176e-04; pad cilja=9.259e-05; validacioni NLL=20.39824568
    Iteracija 286: cilj=18.61711298; |grad|inf=8.862e-06; pad cilja=9.407e-07; validacioni NLL=20.39876655
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 286
    Korišćena iteracija: 100
    NLL korišćene tačke: 18.63550669
    |grad|inf korišćene tačke: 6.285e-03
    Najbolji provereni validacioni NLL: 20.29843883

Validaciona pretraga: slojevi=2, regularizacija=0.01
    Početni cilj=23.81636398; dozvoljeno iteracija=800
    Iteracija 1: cilj=23.12175735; |grad|inf=6.605e-01; pad cilja=nan; validacioni NLL=22.85119205
    Iteracija 25: cilj=20.34086163; |grad|inf=4.342e-01; pad cilja=2.781e+00; validacioni NLL=21.28182594
    Iteracija 50: cilj=19.06725594; |grad|inf=5.420e-02; pad cilja=1.274e+00; validacioni NLL=20.32619418
    Iteracija 75: cilj=18.79534090; |grad|inf=2.158e-02; pad cilja=2.719e-01; validacioni NLL=19.96603563
    Iteracija 100: cilj=18.72055213; |grad|inf=1.162e-02; pad cilja=7.479e-02; validacioni NLL=19.88613226
    Iteracija 125: cilj=18.64193257; |grad|inf=1.640e-02; pad cilja=7.862e-02; validacioni NLL=19.95118908
    Iteracija 150: cilj=18.59066218; |grad|inf=1.205e-02; pad cilja=5.127e-02; validacioni NLL=19.87984100
    Iteracija 175: cilj=18.55997935; |grad|inf=6.412e-03; pad cilja=3.068e-02; validacioni NLL=19.83280917
    Iteracija 200: cilj=18.55590135; |grad|inf=2.505e-03; pad cilja=4.078e-03; validacioni NLL=19.83022510
    Iteracija 225: cilj=18.54879993; |grad|inf=1.089e-02; pad cilja=7.101e-03; validacioni NLL=19.79657516
    Iteracija 250: cilj=18.53515912; |grad|inf=2.464e-03; pad cilja=1.364e-02; validacioni NLL=19.82703744
    Iteracija 275: cilj=18.53486970; |grad|inf=1.531e-04; pad cilja=2.894e-04; validacioni NLL=19.81696702
    Iteracija 300: cilj=18.53484877; |grad|inf=1.646e-04; pad cilja=2.093e-05; validacioni NLL=19.81806461
    Iteracija 325: cilj=18.53483139; |grad|inf=7.892e-05; pad cilja=1.738e-05; validacioni NLL=19.81610889
    Iteracija 350: cilj=18.53483048; |grad|inf=1.505e-05; pad cilja=9.164e-07; validacioni NLL=19.81631354
    Iteracija 375: cilj=18.53483024; |grad|inf=9.828e-06; pad cilja=2.341e-07; validacioni NLL=19.81652224
    Iteracija 377: cilj=18.53483024; |grad|inf=1.170e-05; pad cilja=3.511e-09; validacioni NLL=19.81651986
    Zaustavljanje: Ispunjen kriterijum zaustavljanja optimizatora: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
    Izvršeno iteracija: 377
    Korišćena iteracija: 225
    NLL korišćene tačke: 18.53895011
    |grad|inf korišćene tačke: 1.089e-02
    Najbolji provereni validacioni NLL: 19.79657516

IZABRANO ISKLJUČIVO VALIDACIJOM:
  Slojevi: 2
  Regularizacija: 0.01
  Trajanje: 225 iteracija
  Validacioni NLL: 19.79657516

Učenje na treningu i validaciji za odvojeni test:
    Početni cilj=23.93073736; dozvoljeno iteracija=225
    Iteracija 1: cilj=23.04803379; |grad|inf=4.976e-01; pad cilja=nan
    Iteracija 25: cilj=20.06174401; |grad|inf=1.179e-01; pad cilja=2.986e+00
    Iteracija 50: cilj=19.20420605; |grad|inf=1.047e-01; pad cilja=8.575e-01
    Iteracija 75: cilj=18.98670597; |grad|inf=2.367e-02; pad cilja=2.175e-01
    Iteracija 100: cilj=18.88992010; |grad|inf=1.651e-02; pad cilja=9.679e-02
    Iteracija 125: cilj=18.85459349; |grad|inf=7.150e-03; pad cilja=3.533e-02
    Iteracija 150: cilj=18.82889679; |grad|inf=6.205e-03; pad cilja=2.570e-02
    Iteracija 175: cilj=18.76206629; |grad|inf=2.749e-02; pad cilja=6.683e-02
    Iteracija 200: cilj=18.71460592; |grad|inf=6.089e-03; pad cilja=4.746e-02
    Iteracija 225: cilj=18.70731634; |grad|inf=1.719e-03; pad cilja=7.290e-03
    Zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 225
    Korišćena iteracija: 225
    NLL korišćene tačke: 18.69898614
    |grad|inf korišćene tačke: 1.719e-03
Odvojeni test — NLL: 20.40364257

Završno učenje na SVIM istorijskim prelazima:
    Početni cilj=23.90231479; dozvoljeno iteracija=225
    Iteracija 1: cilj=23.20286841; |grad|inf=5.236e-01; pad cilja=nan
    Iteracija 25: cilj=20.32046030; |grad|inf=1.713e-01; pad cilja=2.882e+00
    Iteracija 50: cilj=19.28415865; |grad|inf=6.960e-02; pad cilja=1.036e+00
    Iteracija 75: cilj=19.04003591; |grad|inf=2.730e-02; pad cilja=2.441e-01
    Iteracija 100: cilj=18.96374641; |grad|inf=1.086e-02; pad cilja=7.629e-02
    Iteracija 125: cilj=18.90846321; |grad|inf=1.288e-02; pad cilja=5.528e-02
    Iteracija 150: cilj=18.87378784; |grad|inf=1.154e-02; pad cilja=3.468e-02
    Iteracija 175: cilj=18.85076393; |grad|inf=1.817e-02; pad cilja=2.302e-02
    Iteracija 200: cilj=18.84827966; |grad|inf=1.748e-04; pad cilja=2.484e-03
    Iteracija 225: cilj=18.84826406; |grad|inf=5.059e-05; pad cilja=1.559e-05
    Zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
    Izvršeno iteracija: 225
    Korišćena iteracija: 225
    NLL korišćene tačke: 18.84083413
    |grad|inf korišćene tačke: 5.059e-05

NEXT V3: 3, 14, 28, 29, 31, 32, 34
Logaritamska masa NEXT u modelu: -8.86196005
Promena zajedničkih marginala prema prethodnom kontekstu: 0.2291430627
Qiskit, gradijenti, normalizacija, granice i maksimum: ugrađene provere prošle.
Masa modela nije izmerena uspešnost predikcije.



KONAČNI NEXT REZULTATI — V3
loto7_4708_k85.csv: 6, 12, 14, 17, 22, 33, 37
loto7_4708_k85_loto_2976.csv: 15, 17, 31, 34, 37, 38, 39
loto7_4708_k85_loto_plus_1732.csv: 3, 14, 28, 29, 31, 32, 34



VALIDACIJA I ODVOJENI TEST — manji NLL je bolji
loto7_4708_k85.csv:
  validacija=20.07689721; test=20.12409468
  slojevi=1; regularizacija=0.0001
  izabrano trajanje=300; test izvršio=300; završni izvršio=300
  završni |grad|inf=7.663e-04
  završno zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
loto7_4708_k85_loto_2976.csv:
  validacija=19.96554908; test=20.52108775
  slojevi=2; regularizacija=0.01
  izabrano trajanje=75; test izvršio=75; završni izvršio=75
  završni |grad|inf=1.747e-02
  završno zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
loto7_4708_k85_loto_plus_1732.csv:
  validacija=19.79657516; test=20.40364257
  slojevi=2; regularizacija=0.01
  izabrano trajanje=225; test izvršio=225; završni izvršio=225
  završni |grad|inf=5.059e-05
  završno zaustavljanje: Dostignuto trajanje izabrano validacijom; kriterijum konvergencije nije ispunjen.
"""



"""
V3 ima konkretno opravdanje: završiti i kontrolisati optimizaciju. Kvantnu konstrukciju i šest kubita po poziciji zadržavam.
Poređenje odvojenog testa, gde je manji NLL bolji:

CSV	                    V1	        V2	        Ishod V2
Spojeni — 4.708 kola	20,42956	20,51741	Malo lošiji
Loto — 2.976 kola	    20,26286	20,00979	Bolji
Loto Plus — 1.732 kola	20,89222	20,40965	Bolji

V2 ima bolju validaciju na sva tri CSV-a, ali svih 15 optimizacija prekinuto je zbog granice od 200 iteracija. Kod nekih cilj još primetno opada pred prekid. To nije dokaz greške u raspodeli, ali ostavlja učenje nedovršenim prema zadatim kriterijumima.
Za V3 promenio samo:
- Kontrolu završetka optimizacije: veći dozvoljeni broj iteracija i ispis norme gradijenta, promene cilja i razloga zaustavljanja.
- Izbor trajanja učenja kroz hronološku validaciju: proveravati sačuvane tačke tokom učenja i izabrati trajanje zajedno sa konfiguracijom. Test ostaje odvojen; završno učenje koristi celu istoriju i prethodno izabrano trajanje.
Provere matematike i simulatora jesu prošle prema logu; sledeća opravdana dorada odnosi se na optimizaciju. Duže učenje samo po sebi ne garantuje bolji rezultat, zato trajanje treba birati validacijom.

V3 - zadržavam šest kubita po poziciji i zajedničku raspodelu iz V2. Dodajem izbor trajanja učenja kroz hronološku validaciju, praćenje gradijenta i jasnu razliku između dostignutog limita i konvergencije.

V3 će tokom pretrage čuvati parametre sa najmanjim validacionim NLL-om i izabrati njihov broj iteracija. Za odvojeni test i završno učenje kreće od istog determinističkog početka, sa novom standardizacijom prilagođenom njihovim trening podacima. U svakom slučaju koristi prethodno izabran broj iteracija, osim ako optimizator ranije ispuni kriterijum zaustavljanja.

V3 zadržava kvantni model iz V2. Dodaje izbor trajanja učenja na validaciji, najviše 800 iteracija tokom pretrage, čuvanje najboljih parametara i praćenje gradijenta. Test i završno učenje koriste prethodno izabrano trajanje.
"""

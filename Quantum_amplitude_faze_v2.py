#!/usr/bin/env python3

# Quantum_amplitude_faze V2


"""
Loto 7/39 — Quantum Amplitude NEXT V2.

Lokalna, deterministička simulacija:
    7 pozicija, po 6 kubita; registri se računaju zasebno.
    Nema materijalizacije stanja od 42 kubita.
    Amplitudno kodiranje, RZ/RY rotacije, CNOT, Bornovo pravilo.

V2 promena:
    Zajedničko učenje raspodele nad rastućim sedmorkama.
    Cilj je negativna logaritamska gustina opažene kombinacije,
    uz tačnu normalizaciju preko svih dozvoljenih kombinacija.
    Nema zasebnog klasifikatora ili klasičnog regresora.

Klasični deo:
    Standardizacija karakteristika.
    L-BFGS-B optimizacija sa analitičkim gradijentima.
    Dinamičko programiranje za normalizaciju, marginale i MAP.

Zavisnosti:
    numpy, scipy, qiskit==1.4.4

CSV redosled:
    Od najstarijeg do najnovijeg kola.
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
    "/data/loto7_4708_k85.csv",
    "/data/loto7_4708_k85_loto_2976.csv",
    "/data/loto7_4708_k85_loto_plus_1732.csv",
)

# Zadržani kandidati iz V1 radi uporedivosti.
CONFIGS = ((1, 1e-4), (2, 1e-3), (2, 1e-2))

MAX_ITER = 200
QUBITS = 6
DIM = 1 << QUBITS
NUMBERS = 39
POSITIONS = 7
CNOTS = ((0, 1), (1, 2), (2, 3),
         (3, 4), (4, 5), (5, 0))

BASIS = np.arange(DIM)
PERMUTATIONS = {
    pair: BASIS ^ (((BASIS >> pair[0]) & 1) << pair[1])
    for pair in CNOTS
}

SUPPORT = np.zeros((POSITIONS, NUMBERS))
for j in range(POSITIONS):
    SUPPORT[j, j:NUMBERS - POSITIONS + j + 1] = 1.0


def read_history(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) != POSITIONS:
                raise ValueError(
                    f"{path}, red {line}: očekivano je 7 kolona."
                )
            try:
                values = [int(item.strip()) for item in row]
            except ValueError as exc:
                raise ValueError(
                    f"{path}, red {line}: očekivani su celi brojevi "
                    "bez zaglavlja."
                ) from exc
            if len(set(values)) != POSITIONS:
                raise ValueError(f"{path}, red {line}: duplikati.")
            if min(values) < 1 or max(values) > NUMBERS:
                raise ValueError(f"{path}, red {line}: broj van 1–39.")
            rows.append(sorted(values))

    if len(rows) < 100:
        raise ValueError(f"{path}: potrebno je najmanje 100 kola.")
    return np.asarray(rows, dtype=np.int64)


def features(history):
    """Iste karakteristike kao V1; bez pristupa budućim kolima."""
    h = history.astype(np.float64)
    count = np.arange(1, len(h) + 1, dtype=np.float64)[:, None]
    mean = np.cumsum(h, axis=0) / count
    variance = np.maximum(
        np.cumsum(h * h, axis=0) / count - mean ** 2, 0.0
    )

    delta = np.zeros_like(h)
    delta[1:] = h[1:] - h[:-1]

    present = np.zeros((len(h), NUMBERS))
    present[np.arange(len(h))[:, None], history - 1] = 1.0

    # 1 + 7 + 7 + 7 + 39 + 3 = 64 amplitude.
    raw = np.column_stack((
        np.ones(len(h)), h, mean, delta, present,
        variance[:, [0, 3, 6]],
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
            raise ArithmeticError("Neispravna amplitudna normalizacija.")
        return (values / norms).astype(np.complex128)


def rotation(kind, angle, derivative=False):
    if kind == "ry":
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        if derivative:
            return 0.5 * np.array(
                [[-s, -c], [c, -s]], dtype=np.complex128
            )
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if kind == "rz":
        a, b = np.exp(-0.5j * angle), np.exp(0.5j * angle)
        return np.diag(
            [-0.5j * a, 0.5j * b] if derivative else [a, b]
        )
    raise ValueError(kind)


def gate(state, matrix, qubit):
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
            adjoint = adjoint[:, PERMUTATIONS[(qubit, index)]]
            continue
        derivative = gate(
            before,
            rotation(kind, angles[index], derivative=True),
            qubit,
        )
        gradient[index] = 2.0 * np.vdot(adjoint, derivative).real
        adjoint = gate(
            adjoint, rotation(kind, angles[index]).conj().T, qubit
        )
    return gradient


def initial_angles(layers):
    count = layers * QUBITS * 2
    indices = np.arange(1, count + 1, dtype=np.float64)
    one = 0.2 * np.sin(indices * math.sqrt(2.0))
    return np.tile(one, (POSITIONS, 1)).ravel()


def predict_probabilities(encoded, angles, layers):
    parameters = angles.reshape(POSITIONS, -1)
    output = []
    for j in range(POSITIONS):
        state, _ = evolve(encoded, parameters[j], layers)
        weights = np.abs(state[:, :NUMBERS]) ** 2 * SUPPORT[j]
        mass = weights.sum(axis=1, keepdims=True)
        if np.any(mass <= 0) or not np.isfinite(mass).all():
            raise ArithmeticError("Nulta ili neispravna masa podrške.")
        output.append(weights / mass)
    return np.stack(output, axis=1)


def log_probabilities(p):
    if p.shape[-2:] != (POSITIONS, NUMBERS):
        raise ValueError("Neispravan oblik raspodele.")
    if np.any(p < 0) or not np.isfinite(p).all():
        raise ArithmeticError("Neispravne verovatnoće.")
    if not np.allclose(p.sum(axis=-1), 1.0, atol=2e-10, rtol=0):
        raise ArithmeticError("Pozicione raspodele nisu normalizovane.")
    result = np.full_like(p, -np.inf)
    np.log(p, out=result, where=p > 0)
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
    P(c|x) = proizvod_j p_j(c_j|x) / Z(x), za c1 < ... < c7.
    Z i marginale računaju se tačno, bez nabrajanja svih sedmorki.
    """
    forward = np.full_like(logp, -np.inf)
    forward[..., 0, :] = logp[..., 0, :]
    for j in range(1, POSITIONS):
        forward[..., j, :] = (
            logp[..., j, :]
            + prefix_exclusive(forward[..., j - 1, :])
        )

    z = np.logaddexp.reduce(forward[..., -1, :], axis=-1)
    if not np.isfinite(z).all() or np.any(z > 2e-10):
        raise ArithmeticError("Neispravna zajednička normalizacija.")

    if not need_marginals:
        return z, None

    backward = np.full_like(logp, -np.inf)
    backward[..., -1, :] = logp[..., -1, :]
    for j in range(POSITIONS - 2, -1, -1):
        backward[..., j, :] = (
            logp[..., j, :]
            + suffix_exclusive(backward[..., j + 1, :])
        )

    reverse_z = np.logaddexp.reduce(backward[..., 0, :], axis=-1)
    if not np.allclose(z, reverse_z, atol=2e-9, rtol=0):
        raise ArithmeticError("Prefiks i sufiks daju različitu masu.")

    valid = (
        np.isfinite(forward)
        & np.isfinite(backward)
        & np.isfinite(logp)
    )
    log_marginals = np.full_like(logp, -np.inf)
    log_marginals[valid] = (
        forward[valid] + backward[valid] - logp[valid]
    )
    log_marginals -= np.asarray(z)[..., None, None]
    marginals = np.exp(log_marginals)

    if not np.allclose(
        marginals.sum(axis=-1), 1.0, atol=2e-8, rtol=0
    ):
        raise ArithmeticError("Zajedničke marginale nisu normalizovane.")
    if np.any(marginals.sum(axis=-2) > 1.0 + 2e-8):
        raise ArithmeticError("Zajednička raspodela dopušta duplikate.")
    return z, marginals


def observed_log_weights(logp, targets):
    values = logp[
        np.arange(len(targets))[:, None],
        np.arange(POSITIONS)[None, :],
        targets - 1,
    ].sum(axis=1)
    if not np.isfinite(values).all():
        raise ArithmeticError("Opažena kombinacija ima nultu masu.")
    return values


def mean_nll(p, targets):
    logp = log_probabilities(p)
    z, _ = joint_distribution(logp)
    return float(np.mean(z - observed_log_weights(logp, targets)))


def objective(angles, encoded, targets, layers, penalty):
    p = predict_probabilities(encoded, angles, layers)
    logp = log_probabilities(p)
    z, marginals = joint_distribution(logp, need_marginals=True)
    nll = float(np.mean(z - observed_log_weights(logp, targets)))
    loss = nll + penalty * float(np.mean(angles ** 2))

    # dL / d log(w_jk) = marginala_jk - opaženi indikator_jk.
    residual = marginals.copy()
    residual[
        np.arange(len(targets))[:, None],
        np.arange(POSITIONS)[None, :],
        targets - 1,
    ] -= 1.0

    parameters = angles.reshape(POSITIONS, -1)
    gradient = np.empty_like(parameters)

    # Tapa za samo jedan registar istovremeno: ograničena memorija.
    for j in range(POSITIONS):
        state, tape = evolve(
            encoded, parameters[j], layers, record=True
        )
        weights = np.abs(state[:, :NUMBERS]) ** 2
        coefficient = np.zeros(state.shape, dtype=np.float64)
        np.divide(
            residual[:, j, :],
            weights,
            out=coefficient[:, :NUMBERS],
            where=weights > 0,
        )
        adjoint = coefficient * state / len(targets)
        gradient[j] = reverse_gradient(adjoint, tape, parameters[j])
        del state, tape, coefficient, adjoint

    gradient = gradient.ravel()
    gradient += 2.0 * penalty * angles / angles.size
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ArithmeticError("Neispravan zajednički cilj ili gradijent.")
    return loss, gradient


def check_gradient(encoded, targets, layers, penalty):
    x, y = encoded[:5], targets[:5]
    angles = initial_angles(layers)
    _, analytic = objective(angles, x, y, layers, penalty)
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
            numeric, analytic[index], atol=3e-6, rtol=3e-4
        ):
            raise ArithmeticError(
                f"Zajednički gradijent nije potvrđen: {index}; "
                f"analitički={analytic[index]}, numerički={numeric}."
            )


class AmplitudeRegressor:
    def __init__(self, layers, penalty):
        self.layers = layers
        self.penalty = penalty

    def fit(self, raw, targets):
        self.encoder = Encoder().fit(raw)
        encoded = self.encoder.transform(raw)
        check_gradient(encoded, targets, self.layers, self.penalty)
        initial = initial_angles(self.layers)
        progress = {"iteration": 0, "loss": math.nan}

        def fun(theta):
            value, gradient = objective(
                theta, encoded, targets, self.layers, self.penalty
            )
            progress["loss"] = value
            return value, gradient

        start_loss = fun(initial)[0]

        def callback(_):
            progress["iteration"] += 1
            if progress["iteration"] % 10 == 0:
                print(
                    f"    Iteracija {progress['iteration']}: "
                    f"zajednički cilj={progress['loss']:.8f}",
                    flush=True,
                )

        result = minimize(
            fun, initial, method="L-BFGS-B", jac=True,
            callback=callback,
            options={
                "maxiter": MAX_ITER,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxls": 40,
            },
        )
        if (
            not np.isfinite(result.fun)
            or not np.isfinite(result.x).all()
            or result.fun > start_loss + 1e-9
        ):
            raise ArithmeticError("Optimizacija nije dala validan model.")

        self.angles = result.x.copy()
        self.converged = bool(result.success)
        self.message = str(result.message)
        self.training_nll = mean_nll(
            predict_probabilities(encoded, self.angles, self.layers),
            targets,
        )
        print(
            f"    Početni cilj={start_loss:.8f}; "
            f"završni cilj={result.fun:.8f}; "
            f"trening NLL={self.training_nll:.8f}; "
            f"iteracije={result.nit}",
            flush=True,
        )
        if not self.converged:
            print(
                f"    OPTIMIZATOR: {self.message}. "
                "Zadržan je konačni validan rezultat; "
                "konvergencija nije potvrđena.",
                flush=True,
            )
        return self

    def predict(self, raw):
        return predict_probabilities(
            self.encoder.transform(raw), self.angles, self.layers
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
                    circuit.rz(float(parameters[j, index]), qubit)
                    index += 1
                    circuit.ry(float(parameters[j, index]), qubit)
                    index += 1
                for control, target in CNOTS:
                    circuit.cx(control, target)

            reference = Statevector.from_instruction(circuit).data
            calculated, _ = evolve(encoded, parameters[j], self.layers)
            if not np.allclose(
                reference, calculated[0], atol=2e-10, rtol=0
            ):
                raise ArithmeticError(
                    f"Qiskit neslaganje: pozicija {j + 1}."
                )
            if not np.isclose(
                np.sum(np.abs(calculated[0]) ** 2),
                1.0, atol=2e-10, rtol=0,
            ):
                raise ArithmeticError("Kolo ne čuva normu stanja.")


def exact_next(p):
    logp = log_probabilities(p)
    z, _ = joint_distribution(logp, need_marginals=True)
    z = float(z)
    values = logp[0].copy()
    parents = np.full((POSITIONS, NUMBERS), -1, dtype=np.int64)

    for j in range(1, POSITIONS):
        updated = np.full(NUMBERS, -np.inf)
        best_value, best_index = -np.inf, -1
        for k in range(NUMBERS):
            previous = k - 1
            if previous >= 0 and values[previous] > best_value:
                best_value, best_index = values[previous], previous
            if best_index >= 0 and np.isfinite(logp[j, k]):
                updated[k] = best_value + logp[j, k]
                parents[j, k] = best_index
        values = updated

    last = int(np.argmax(values))
    log_max = float(values[last])
    if not np.isfinite(log_max):
        raise ArithmeticError("Nema maksimuma pozitivne mase.")

    answer = [last]
    for j in range(POSITIONS - 1, 0, -1):
        last = int(parents[j, last])
        if last < 0:
            raise ArithmeticError("Neispravna putanja maksimuma.")
        answer.append(last)

    answer = tuple(k + 1 for k in reversed(answer))
    if (
        len(set(answer)) != POSITIONS
        or answer[0] < 1
        or answer[-1] > NUMBERS
        or any(a >= b for a, b in zip(answer, answer[1:]))
    ):
        raise ArithmeticError("Neispravna NEXT kombinacija.")
    return answer, log_max - z


def check_exact_search(p):
    small = p.copy()
    small[:, 9:] = 0.0
    mass = small.sum(axis=1, keepdims=True)
    if np.any(mass <= 0):
        raise ArithmeticError("Nema mase za proveru male podrške.")
    small /= mass
    logp = log_probabilities(small)

    cases = list(itertools.combinations(range(9), POSITIONS))
    scores = np.array([
        sum(logp[j, c[j]] for j in range(POSITIONS))
        for c in cases
    ])
    brute_z = float(np.logaddexp.reduce(scores))
    dp_z, _ = joint_distribution(logp, need_marginals=True)
    answer, log_mass = exact_next(small)
    selected = sum(logp[j, answer[j] - 1] for j in range(POSITIONS))

    if not (
        np.isclose(brute_z, float(dp_z), atol=2e-9, rtol=0)
        and np.isclose(selected, scores.max(), atol=2e-9, rtol=0)
        and np.isclose(
            log_mass, scores.max() - brute_z, atol=2e-9, rtol=0
        )
    ):
        raise ArithmeticError("Provera tačnog maksimuma nije prošla.")


def run_file(path):
    history = read_history(path)
    raw = features(history)
    x, y = raw[:-1], history[1:]
    n = len(y)
    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)

    print(f"\nV2 — FAJL: {path}", flush=True)
    print(
        f"Kola: {len(history)}; prelazi: {n}; "
        f"trening/validacija/test: "
        f"{train_end}/{validation_end - train_end}/{n - validation_end}",
        flush=True,
    )

    best_config, best_score = None, math.inf
    for layers, penalty in CONFIGS:
        print(
            f"\nValidacioni kandidat: slojevi={layers}, "
            f"regularizacija={penalty:g}",
            flush=True,
        )
        model = AmplitudeRegressor(layers, penalty).fit(
            x[:train_end], y[:train_end]
        )
        validation_p = model.predict(x[train_end:validation_end])
        score = mean_nll(validation_p, y[train_end:validation_end])
        print(f"  Validacioni NLL: {score:.8f}", flush=True)
        if score < best_score:
            best_config, best_score = (layers, penalty), score

    if best_config is None:
        raise ArithmeticError("Nije izabran validan kandidat.")

    layers, penalty = best_config
    print(
        f"\nIzabrano: slojevi={layers}, regularizacija={penalty:g}",
        flush=True,
    )

    # Konfiguracija je zaključana pre pristupa test rezultatima.
    print("\nUčenje na treningu i validaciji za odvojeni test:", flush=True)
    test_model = AmplitudeRegressor(layers, penalty).fit(
        x[:validation_end], y[:validation_end]
    )
    test_p = test_model.predict(x[validation_end:])
    test_nll = mean_nll(test_p, y[validation_end:])
    print(f"Odvojeni test — NLL: {test_nll:.8f}", flush=True)

    print("\nZavršno učenje na SVIM istorijskim prelazima:", flush=True)
    final_model = AmplitudeRegressor(layers, penalty).fit(x, y)
    final_model.check_qiskit(raw[-1:])

    p = final_model.predict(raw[-1:])[0]
    check_exact_search(p)
    answer, log_mass = exact_next(p)

    previous_p = final_model.predict(raw[-2:-1])[0]
    context_change = float(np.max(np.abs(p - previous_p)))
    if context_change < 1e-12:
        print(
            "PROVERA: raspodela se gotovo ne menja između "
            "poslednja dva istorijska konteksta.",
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

    print("\nNEXT V2: " + ", ".join(map(str, answer)), flush=True)
    print(
        f"Logaritamska masa NEXT u modelu: {log_mass:.8f}\n"
        f"Promena raspodele prema prethodnom kontekstu: "
        f"{context_change:.10g}\n"
        "Qiskit, zajednički gradijenti, normalizacija i maksimum: "
        "provere prošle.",
        flush=True,
    )
    if not final_model.converged:
        print(
            "Završna optimizacija nije prijavila konvergenciju: "
            + final_model.message,
            flush=True,
        )

    return {
        "path": path,
        "answer": answer,
        "p": p,
        "context_change": context_change,
        "validation_nll": best_score,
        "test_nll": test_nll,
    }


def main():
    np.seterr(over="raise", invalid="raise", divide="raise")
    results = [run_file(path) for path in PATHS]

    print("\nKONAČNI NEXT REZULTATI — V2", flush=True)
    for item in results:
        print(
            f"{Path(item['path']).name}: "
            + ", ".join(map(str, item["answer"])),
            flush=True,
        )

    print("\nVALIDACIJA I ODVOJENI TEST — manji NLL je bolji", flush=True)
    for item in results:
        print(
            f"{Path(item['path']).name}: "
            f"validacija={item['validation_nll']:.8f}; "
            f"test={item['test_nll']:.8f}",
            flush=True,
        )

    for i, first in enumerate(results):
        for second in results[i + 1:]:
            if first["answer"] != second["answer"]:
                continue
            difference = float(np.max(np.abs(first["p"] - second["p"])))
            print(
                "\nPROVERA ISTIH NEXT:\n"
                f"  {Path(first['path']).name}\n"
                f"  {Path(second['path']).name}\n"
                f"  Najveća razlika raspodela: {difference:.10g}\n"
                f"  Promene konteksta: "
                f"{first['context_change']:.10g}, "
                f"{second['context_change']:.10g}\n"
                "  Ulazi i modeli obrađeni su zasebno. "
                "Rezultati se ne menjaju radi različitosti. "
                "Ove provere same ne isključuju degeneraciju modela.",
                flush=True,
            )


if __name__ == "__main__":
    main()



"""
V2 — FAJL: /data/loto7_4708_k85.csv
Kola: 4708; prelazi: 4707; trening/validacija/test: 3294/706/707

Validacioni kandidat: slojevi=1, regularizacija=0.0001
    Iteracija 10: zajednički cilj=21.91956765
    Iteracija 20: zajednički cilj=21.01557557
    Iteracija 30: zajednički cilj=20.90079645
    Iteracija 40: zajednički cilj=20.86174909
    Iteracija 50: zajednički cilj=20.66672472
    Iteracija 60: zajednički cilj=20.65332311
    Iteracija 70: zajednički cilj=20.64009855
    Iteracija 80: zajednički cilj=20.58171342
    Iteracija 90: zajednički cilj=20.34471581
    Iteracija 100: zajednički cilj=20.15220293
    Iteracija 110: zajednički cilj=20.12796748
    Iteracija 120: zajednički cilj=19.98731351
    Iteracija 130: zajednički cilj=19.98303900
    Iteracija 140: zajednički cilj=19.97541060
    Iteracija 150: zajednički cilj=19.97154744
    Iteracija 160: zajednički cilj=19.96868908
    Iteracija 170: zajednički cilj=19.96814110
    Iteracija 180: zajednički cilj=19.96649249
    Iteracija 190: zajednički cilj=19.95760456
    Iteracija 200: zajednički cilj=19.85688555
    Početni cilj=23.53874470; završni cilj=19.85688555; trening NLL=19.85675476; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.15971081

Validacioni kandidat: slojevi=2, regularizacija=0.001
    Iteracija 10: zajednički cilj=22.04876384
    Iteracija 20: zajednički cilj=21.31817592
    Iteracija 30: zajednički cilj=20.24169866
    Iteracija 40: zajednički cilj=19.75527127
    Iteracija 50: zajednički cilj=19.64995461
    Iteracija 60: zajednički cilj=19.59634413
    Iteracija 70: zajednički cilj=19.56590215
    Iteracija 80: zajednički cilj=19.53608438
    Iteracija 90: zajednički cilj=19.51462336
    Iteracija 100: zajednički cilj=19.50167764
    Iteracija 110: zajednički cilj=19.48728124
    Iteracija 120: zajednički cilj=19.48323726
    Iteracija 130: zajednički cilj=19.47980609
    Iteracija 140: zajednički cilj=19.47910399
    Iteracija 150: zajednički cilj=19.47746177
    Iteracija 160: zajednički cilj=19.47539778
    Iteracija 170: zajednički cilj=19.47394595
    Iteracija 180: zajednički cilj=19.47349274
    Iteracija 190: zajednički cilj=19.47336331
    Iteracija 200: zajednički cilj=19.47321775
    Početni cilj=23.93891982; završni cilj=19.47321775; trening NLL=19.47257475; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.25073781

Validacioni kandidat: slojevi=2, regularizacija=0.01
    Iteracija 10: zajednički cilj=22.16655326
    Iteracija 20: zajednički cilj=21.35099654
    Iteracija 30: zajednički cilj=20.57451044
    Iteracija 40: zajednički cilj=20.00145684
    Iteracija 50: zajednički cilj=19.55699524
    Iteracija 60: zajednički cilj=19.45836775
    Iteracija 70: zajednički cilj=19.39211536
    Iteracija 80: zajednički cilj=19.33321394
    Iteracija 90: zajednički cilj=19.30914403
    Iteracija 100: zajednički cilj=19.27558977
    Iteracija 110: zajednički cilj=19.26747375
    Iteracija 120: zajednički cilj=19.25980914
    Iteracija 130: zajednički cilj=19.25558608
    Iteracija 140: zajednički cilj=19.25450597
    Iteracija 150: zajednički cilj=19.24959536
    Iteracija 160: zajednički cilj=19.24827728
    Iteracija 170: zajednički cilj=19.24803788
    Iteracija 180: zajednički cilj=19.24802497
    Iteracija 190: zajednički cilj=19.24802433
    Iteracija 200: zajednički cilj=19.24802416
    Početni cilj=23.93910288; završni cilj=19.24802416; trening NLL=19.23919721; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.36238464

Izabrano: slojevi=1, regularizacija=0.0001

Učenje na treningu i validaciji za odvojeni test:
    Iteracija 10: zajednički cilj=21.82005680
    Iteracija 20: zajednički cilj=21.35648635
    Iteracija 30: zajednički cilj=21.09879812
    Iteracija 40: zajednički cilj=21.04039025
    Iteracija 50: zajednički cilj=20.98665306
    Iteracija 60: zajednički cilj=20.72093435
    Iteracija 70: zajednički cilj=20.57608126
    Iteracija 80: zajednički cilj=20.56361444
    Iteracija 90: zajednički cilj=20.50426404
    Iteracija 100: zajednički cilj=20.31752847
    Iteracija 110: zajednički cilj=20.28629356
    Iteracija 120: zajednički cilj=20.14916389
    Iteracija 130: zajednički cilj=20.08821237
    Iteracija 140: zajednički cilj=20.08363668
    Iteracija 150: zajednički cilj=20.07053634
    Iteracija 160: zajednički cilj=20.05493273
    Iteracija 170: zajednički cilj=20.04831597
    Iteracija 180: zajednički cilj=20.03880195
    Iteracija 190: zajednički cilj=20.03374915
    Iteracija 200: zajednički cilj=20.02950938
    Početni cilj=23.50849465; završni cilj=20.02950938; trening NLL=20.02940488; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
Odvojeni test — NLL: 20.51740773

Završno učenje na SVIM istorijskim prelazima:
    Iteracija 10: zajednički cilj=22.28380973
    Iteracija 20: zajednički cilj=21.70020806
    Iteracija 30: zajednički cilj=21.27268533
    Iteracija 40: zajednički cilj=21.07991595
    Iteracija 50: zajednički cilj=20.88606983
    Iteracija 60: zajednički cilj=20.82095503
    Iteracija 70: zajednički cilj=20.59172029
    Iteracija 80: zajednički cilj=20.58370682
    Iteracija 90: zajednički cilj=20.53218527
    Iteracija 100: zajednički cilj=20.49210299
    Iteracija 110: zajednički cilj=20.48244759
    Iteracija 120: zajednički cilj=20.43022046
    Iteracija 130: zajednički cilj=20.39393221
    Iteracija 140: zajednički cilj=20.20620288
    Iteracija 150: zajednički cilj=20.20062753
    Iteracija 160: zajednički cilj=20.19520014
    Iteracija 170: zajednički cilj=20.19181079
    Iteracija 180: zajednički cilj=20.17395254
    Iteracija 190: zajednički cilj=20.15651274
    Iteracija 200: zajednički cilj=20.14632990
    Početni cilj=23.53425883; završni cilj=20.14632990; trening NLL=20.14622099; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.

NEXT V2: 6, x, 14, y, 19, z, 37
Logaritamska masa NEXT u modelu: -8.15133137
Promena raspodele prema prethodnom kontekstu: 0.1494392374
Qiskit, zajednički gradijenti, normalizacija i maksimum: provere prošle.
Završna optimizacija nije prijavila konvergenciju: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT



V2 — FAJL: /data/loto7_4708_k85_loto_2976.csv
Kola: 2976; prelazi: 2975; trening/validacija/test: 2082/446/447

Validacioni kandidat: slojevi=1, regularizacija=0.0001
    Iteracija 10: zajednički cilj=22.32403077
    Iteracija 20: zajednički cilj=21.75956733
    Iteracija 30: zajednički cilj=21.47502705
    Iteracija 40: zajednički cilj=21.41236816
    Iteracija 50: zajednički cilj=21.20158489
    Iteracija 60: zajednički cilj=21.14967512
    Iteracija 70: zajednički cilj=21.02777587
    Iteracija 80: zajednički cilj=20.95661459
    Iteracija 90: zajednički cilj=20.86948741
    Iteracija 100: zajednički cilj=20.80944218
    Iteracija 110: zajednički cilj=20.75271313
    Iteracija 120: zajednički cilj=20.45293404
    Iteracija 130: zajednički cilj=20.43533865
    Iteracija 140: zajednički cilj=20.42876368
    Iteracija 150: zajednički cilj=20.35775652
    Iteracija 160: zajednički cilj=19.87928347
    Iteracija 170: zajednički cilj=19.83098381
    Iteracija 180: zajednički cilj=19.79095628
    Iteracija 190: zajednički cilj=19.77547511
    Iteracija 200: zajednički cilj=19.75556710
    Početni cilj=23.46817453; završni cilj=19.75556710; trening NLL=19.75544707; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.46228363

Validacioni kandidat: slojevi=2, regularizacija=0.001
    Iteracija 10: zajednički cilj=21.26220996
    Iteracija 20: zajednički cilj=20.03321521
    Iteracija 30: zajednički cilj=19.73170228
    Iteracija 40: zajednički cilj=19.35121401
    Iteracija 50: zajednički cilj=19.26039102
    Iteracija 60: zajednički cilj=19.21021141
    Iteracija 70: zajednički cilj=19.17084781
    Iteracija 80: zajednički cilj=19.12498922
    Iteracija 90: zajednički cilj=19.10854001
    Iteracija 100: zajednički cilj=19.10185240
    Iteracija 110: zajednički cilj=19.09743070
    Iteracija 120: zajednički cilj=19.09641122
    Iteracija 130: zajednički cilj=19.09425810
    Iteracija 140: zajednički cilj=19.09277267
    Iteracija 150: zajednički cilj=19.09260078
    Iteracija 160: zajednički cilj=19.09257482
    Iteracija 170: zajednički cilj=19.09256875
    Iteracija 180: zajednički cilj=19.09255778
    Iteracija 190: zajednički cilj=19.09254250
    Iteracija 200: zajednički cilj=19.09253752
    Početni cilj=24.09695052; završni cilj=19.09253752; trening NLL=19.09173726; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.00334899

Validacioni kandidat: slojevi=2, regularizacija=0.01
    Iteracija 10: zajednički cilj=21.65989631
    Iteracija 20: zajednički cilj=20.15742170
    Iteracija 30: zajednički cilj=19.51603249
    Iteracija 40: zajednički cilj=19.18947216
    Iteracija 50: zajednički cilj=19.13075875
    Iteracija 60: zajednički cilj=19.08653312
    Iteracija 70: zajednički cilj=19.05146685
    Iteracija 80: zajednički cilj=19.00733143
    Iteracija 90: zajednički cilj=18.99859880
    Iteracija 100: zajednički cilj=18.98440381
    Iteracija 110: zajednički cilj=18.98175385
    Iteracija 120: zajednički cilj=18.98103225
    Iteracija 130: zajednički cilj=18.98034917
    Iteracija 140: zajednički cilj=18.97987846
    Iteracija 150: zajednički cilj=18.97515508
    Iteracija 160: zajednički cilj=18.96954720
    Iteracija 170: zajednički cilj=18.96212979
    Iteracija 180: zajednički cilj=18.95849276
    Iteracija 190: zajednički cilj=18.95846919
    Iteracija 200: zajednički cilj=18.95844983
    Početni cilj=24.09713358; završni cilj=18.95844983; trening NLL=18.95044007; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.08869216

Izabrano: slojevi=2, regularizacija=0.001

Učenje na treningu i validaciji za odvojeni test:
    Iteracija 10: zajednički cilj=22.20268197
    Iteracija 20: zajednički cilj=20.69082189
    Iteracija 30: zajednički cilj=19.70012270
    Iteracija 40: zajednički cilj=19.37161343
    Iteracija 50: zajednički cilj=19.28145059
    Iteracija 60: zajednički cilj=19.22998470
    Iteracija 70: zajednički cilj=19.19565054
    Iteracija 80: zajednički cilj=19.18883476
    Iteracija 90: zajednički cilj=19.17566286
    Iteracija 100: zajednički cilj=19.16423613
    Iteracija 110: zajednički cilj=19.16253646
    Iteracija 120: zajednički cilj=19.15681411
    Iteracija 130: zajednički cilj=19.14706283
    Iteracija 140: zajednički cilj=19.14416788
    Iteracija 150: zajednički cilj=19.13782364
    Iteracija 160: zajednički cilj=19.13654317
    Iteracija 170: zajednički cilj=19.13372946
    Iteracija 180: zajednički cilj=19.12513047
    Iteracija 190: zajednički cilj=19.12214883
    Iteracija 200: zajednički cilj=19.11610058
    Početni cilj=24.11631489; završni cilj=19.11610058; trening NLL=19.11539093; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
Odvojeni test — NLL: 20.00978793

Završno učenje na SVIM istorijskim prelazima:
    Iteracija 10: zajednički cilj=21.47720154
    Iteracija 20: zajednički cilj=20.05460517
    Iteracija 30: zajednički cilj=19.64169130
    Iteracija 40: zajednički cilj=19.52026902
    Iteracija 50: zajednički cilj=19.44288942
    Iteracija 60: zajednički cilj=19.39164738
    Iteracija 70: zajednički cilj=19.34959747
    Iteracija 80: zajednički cilj=19.30748803
    Iteracija 90: zajednički cilj=19.25838030
    Iteracija 100: zajednički cilj=19.22638454
    Iteracija 110: zajednički cilj=19.20047122
    Iteracija 120: zajednički cilj=19.19379050
    Iteracija 130: zajednički cilj=19.19078652
    Iteracija 140: zajednički cilj=19.18819698
    Iteracija 150: zajednički cilj=19.18599577
    Iteracija 160: zajednički cilj=19.18227123
    Iteracija 170: zajednički cilj=19.18036561
    Iteracija 180: zajednički cilj=19.18000691
    Iteracija 190: zajednički cilj=19.17974946
    Iteracija 200: zajednički cilj=19.17966021
    Početni cilj=24.05945083; završni cilj=19.17966021; trening NLL=19.17869087; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.

NEXT V2: 5, x, 14, y, 28, z, 38
Logaritamska masa NEXT u modelu: -9.83083503
Promena raspodele prema prethodnom kontekstu: 0.2008381922
Qiskit, zajednički gradijenti, normalizacija i maksimum: provere prošle.
Završna optimizacija nije prijavila konvergenciju: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT



V2 — FAJL: /data/loto7_4708_k85_loto_plus_1732.csv
Kola: 1732; prelazi: 1731; trening/validacija/test: 1211/260/260

Validacioni kandidat: slojevi=1, regularizacija=0.0001
    Iteracija 10: zajednički cilj=22.15613454
    Iteracija 20: zajednički cilj=21.63052535
    Iteracija 30: zajednički cilj=21.38313524
    Iteracija 40: zajednički cilj=21.31599727
    Iteracija 50: zajednički cilj=21.11125072
    Iteracija 60: zajednički cilj=20.82681498
    Iteracija 70: zajednički cilj=20.64259042
    Iteracija 80: zajednički cilj=20.58778489
    Iteracija 90: zajednički cilj=20.54059479
    Iteracija 100: zajednički cilj=20.50112301
    Iteracija 110: zajednički cilj=20.41947896
    Iteracija 120: zajednički cilj=20.31405669
    Iteracija 130: zajednički cilj=20.30098718
    Iteracija 140: zajednički cilj=20.28921658
    Iteracija 150: zajednički cilj=20.01785120
    Iteracija 160: zajednički cilj=19.99328086
    Iteracija 170: zajednički cilj=19.98259385
    Iteracija 180: zajednički cilj=19.96653867
    Iteracija 190: zajednički cilj=19.95335925
    Iteracija 200: zajednički cilj=19.86443978
    Početni cilj=23.50951353; završni cilj=19.86443978; trening NLL=19.86431384; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.45146038

Validacioni kandidat: slojevi=2, regularizacija=0.001
    Iteracija 10: zajednički cilj=22.32839444
    Iteracija 20: zajednički cilj=19.86636126
    Iteracija 30: zajednički cilj=19.20354690
    Iteracija 40: zajednički cilj=19.02982769
    Iteracija 50: zajednički cilj=18.90657650
    Iteracija 60: zajednički cilj=18.82502886
    Iteracija 70: zajednički cilj=18.76300188
    Iteracija 80: zajednički cilj=18.72555494
    Iteracija 90: zajednički cilj=18.69115562
    Iteracija 100: zajednički cilj=18.66738534
    Iteracija 110: zajednički cilj=18.65848242
    Iteracija 120: zajednički cilj=18.65564464
    Iteracija 130: zajednički cilj=18.65181447
    Iteracija 140: zajednički cilj=18.64887069
    Iteracija 150: zajednički cilj=18.64789492
    Iteracija 160: zajednički cilj=18.64771325
    Iteracija 170: zajednički cilj=18.64554202
    Iteracija 180: zajednički cilj=18.63910040
    Iteracija 190: zajednički cilj=18.63592817
    Iteracija 200: zajednički cilj=18.63543657
    Početni cilj=23.81618092; završni cilj=18.63543657; trening NLL=18.60360222; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.35158119

Validacioni kandidat: slojevi=2, regularizacija=0.01
    Iteracija 10: zajednički cilj=22.33058130
    Iteracija 20: zajednički cilj=20.72075837
    Iteracija 30: zajednički cilj=19.52532437
    Iteracija 40: zajednički cilj=19.08649778
    Iteracija 50: zajednički cilj=18.94365139
    Iteracija 60: zajednički cilj=18.86303307
    Iteracija 70: zajednički cilj=18.81577585
    Iteracija 80: zajednički cilj=18.77395189
    Iteracija 90: zajednički cilj=18.75608941
    Iteracija 100: zajednički cilj=18.72650887
    Iteracija 110: zajednički cilj=18.71596880
    Iteracija 120: zajednički cilj=18.70132014
    Iteracija 130: zajednički cilj=18.69527825
    Iteracija 140: zajednički cilj=18.68858119
    Iteracija 150: zajednički cilj=18.68064366
    Iteracija 160: zajednički cilj=18.67540170
    Iteracija 170: zajednički cilj=18.64791673
    Iteracija 180: zajednički cilj=18.62268461
    Iteracija 190: zajednički cilj=18.58928738
    Iteracija 200: zajednički cilj=18.57888166
    Početni cilj=23.81636398; završni cilj=18.57888166; trening NLL=18.56896230; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
  Validacioni NLL: 20.33634501

Izabrano: slojevi=2, regularizacija=0.01

Učenje na treningu i validaciji za odvojeni test:
    Iteracija 10: zajednički cilj=22.32690534
    Iteracija 20: zajednički cilj=20.37656160
    Iteracija 30: zajednički cilj=19.94085099
    Iteracija 40: zajednički cilj=19.72809298
    Iteracija 50: zajednički cilj=19.20420605
    Iteracija 60: zajednički cilj=19.08704311
    Iteracija 70: zajednički cilj=19.01243993
    Iteracija 80: zajednički cilj=18.95662094
    Iteracija 90: zajednički cilj=18.92514569
    Iteracija 100: zajednički cilj=18.88992010
    Iteracija 110: zajednički cilj=18.87268881
    Iteracija 120: zajednički cilj=18.86068338
    Iteracija 130: zajednički cilj=18.85183729
    Iteracija 140: zajednički cilj=18.83880179
    Iteracija 150: zajednički cilj=18.82889679
    Iteracija 160: zajednički cilj=18.81252737
    Iteracija 170: zajednički cilj=18.78014704
    Iteracija 180: zajednički cilj=18.74660939
    Iteracija 190: zajednički cilj=18.72120842
    Iteracija 200: zajednički cilj=18.71460592
    Početni cilj=23.93073736; završni cilj=18.71460592; trening NLL=18.70647649; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.
Odvojeni test — NLL: 20.40964841

Završno učenje na SVIM istorijskim prelazima:
    Iteracija 10: zajednički cilj=22.05378611
    Iteracija 20: zajednički cilj=20.90365462
    Iteracija 30: zajednički cilj=19.88179071
    Iteracija 40: zajednički cilj=19.43756238
    Iteracija 50: zajednički cilj=19.28415865
    Iteracija 60: zajednički cilj=19.18080080
    Iteracija 70: zajednički cilj=19.06795268
    Iteracija 80: zajednički cilj=19.01480266
    Iteracija 90: zajednički cilj=18.98333863
    Iteracija 100: zajednički cilj=18.96374641
    Iteracija 110: zajednički cilj=18.92838106
    Iteracija 120: zajednički cilj=18.91320705
    Iteracija 130: zajednički cilj=18.90318141
    Iteracija 140: zajednički cilj=18.89085943
    Iteracija 150: zajednički cilj=18.87378784
    Iteracija 160: zajednički cilj=18.86776720
    Iteracija 170: zajednički cilj=18.85541257
    Iteracija 180: zajednički cilj=18.84857988
    Iteracija 190: zajednički cilj=18.84835981
    Iteracija 200: zajednički cilj=18.84827966
    Početni cilj=23.90231479; završni cilj=18.84827966; trening NLL=18.84084006; iteracije=200
    OPTIMIZATOR: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. Zadržan je konačni validan rezultat; konvergencija nije potvrđena.

NEXT V2: 3, x, 28, y, 31, z, 34
Logaritamska masa NEXT u modelu: -8.86244820
Promena raspodele prema prethodnom kontekstu: 0.2018989109
Qiskit, zajednički gradijenti, normalizacija i maksimum: provere prošle.
Završna optimizacija nije prijavila konvergenciju: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT



KONAČNI NEXT REZULTATI — V2
loto7_4708_k85.csv: 6, x, 14, y, 19, z, 37
loto7_4708_k85_loto_2976.csv: 5, x, 14, y, 28, z, 38
loto7_4708_k85_loto_plus_1732.csv: 3, x, 28, y, 31, z, 34



VALIDACIJA I ODVOJENI TEST — manji NLL je bolji
loto7_4708_k85.csv: validacija=20.15971081; test=20.51740773
loto7_4708_k85_loto_2976.csv: validacija=20.00334899; test=20.00978793
loto7_4708_k85_loto_plus_1732.csv: validacija=20.33634501; test=20.40964841
"""



"""
Ima opravdanja za V2. U V1 rezultatu sve provere su prošle, a jedan sloj dao je najbolji validacioni rezultat za sva tri CSV-a.
V1 kod uči pozicije odvojeno, pa njihove raspodele naknadno povezuje uslovom rastućeg redosleda. 
V2 direktno uči tu zajedničku raspodelu iz koje bira NEXT. 
Šest kubita po poziciji, amplitude, faze, lokalni simulator i tačan maksimum bez popravljanja brojeva.
V2 popravka usklađenosti učenja i izlaza. 
Ne tvrdim da daje bolje rezultate. 
"""

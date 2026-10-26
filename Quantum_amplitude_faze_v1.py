#!/usr/bin/env python3

# Quantum_amplitude_faze V1


"""
Loto 7/39 — učenje kvantnih amplituda i faza.

Zavisnosti:
    numpy, scipy, qiskit==1.4.4

Svaki CSV obrađuje se zasebno.
Redosled redova: od najstarijeg do najnovijeg kola.

Kvantni deo:
    Sedam registara od po šest kubita.
    Amplitudno kodiranje istorijskog konteksta.
    Naučene RZ/RY rotacije i CNOT operacije.
    Bornovo pravilo za verovatnoće.

Klasični deo:
    Standardizacija ulaza, L-BFGS-B optimizator.
    Tačno uslovljavanje na rastuće sedmorke.
    Dinamičko programiranje za normalizaciju i tačan maksimum.

Nema slučajnog uzorkovanja, dopunjavanja ili popravljanja NEXT.
Nema unapred zadate normalne ili uniformne izlazne raspodele.
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

# Parametri se biraju isključivo na hronološkoj validaciji.
CONFIGS = (
    (1, 1e-4),
    (2, 1e-3),
    (2, 1e-2),
)

MAX_ITER = 100
QUBITS = 6
DIM = 1 << QUBITS
NUMBERS = 39
POSITIONS = 7
THRESHOLDS = np.arange(1, NUMBERS)
BASIS = np.arange(DIM)
CNOTS = ((0, 1), (1, 2), (2, 3),
         (3, 4), (4, 5), (5, 0))


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
            # Menja se samo redosled brojeva unutar jednog kola.
            rows.append(sorted(values))

    if len(rows) < 100:
        raise ValueError(f"{path}: potrebno je najmanje 100 kola.")
    return np.asarray(rows, dtype=np.int64)


def features(history):
    """Svaki red koristi samo podatke do tog kola, uključujući njega."""
    h = history.astype(np.float64)
    count = np.arange(1, len(h) + 1, dtype=np.float64)[:, None]
    running_mean = np.cumsum(h, axis=0) / count
    running_var = np.maximum(
        np.cumsum(h * h, axis=0) / count - running_mean ** 2,
        0.0,
    )

    delta = np.zeros_like(h)
    delta[1:] = h[1:] - h[:-1]

    present = np.zeros((len(h), NUMBERS), dtype=np.float64)
    present[np.arange(len(h))[:, None], history - 1] = 1.0

    # 1 + 7 + 7 + 7 + 39 + 3 = 64 ulazne amplitude.
    x = np.column_stack((
        np.ones(len(h)),
        h,
        running_mean,
        delta,
        present,
        running_var[:, [0, 3, 6]],
    ))
    if x.shape != (len(h), DIM) or not np.isfinite(x).all():
        raise ArithmeticError("Neispravne istorijske karakteristike.")
    return x


class Encoder:
    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.0
        return self

    def transform(self, x):
        z = (x - self.mean) / self.scale
        z[:, 0] = 1.0
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ArithmeticError("Neispravna amplitudna normalizacija.")
        return (z / norms).astype(np.complex128)


def rotation(kind, angle, derivative=False):
    if kind == "ry":
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        if derivative:
            return 0.5 * np.array(
                [[-s, -c], [c, -s]], dtype=np.complex128
            )
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    if kind == "rz":
        a = np.exp(-0.5j * angle)
        b = np.exp(0.5j * angle)
        if derivative:
            return np.diag([-0.5j * a, 0.5j * b])
        return np.diag([a, b])

    raise ValueError(kind)


def single_gate(state, matrix, qubit):
    """Qiskit little-endian konvencija."""
    stride = 1 << qubit
    src = state.reshape(len(state), -1, 2, stride)
    out = np.empty_like(src)
    out[:, :, 0, :] = (
        matrix[0, 0] * src[:, :, 0, :]
        + matrix[0, 1] * src[:, :, 1, :]
    )
    out[:, :, 1, :] = (
        matrix[1, 0] * src[:, :, 0, :]
        + matrix[1, 1] * src[:, :, 1, :]
    )
    return out.reshape(state.shape)


def cnot(state, control, target):
    permutation = BASIS ^ (((BASIS >> control) & 1) << target)
    return state[:, permutation]


def initial_angles(layers):
    """Fiksan početak optimizacije; nema slučajnosti ili dodatog šuma."""
    i = np.arange(1, layers * QUBITS * 2 + 1, dtype=np.float64)
    return 0.2 * np.sin(i * math.sqrt(2.0))


def evolve(encoded, angles, layers, record=False):
    state = encoded
    tape = []
    index = 0
    for _ in range(layers):
        for qubit in range(QUBITS):
            for kind in ("rz", "ry"):
                if record:
                    tape.append((kind, qubit, index, state))
                state = single_gate(
                    state, rotation(kind, angles[index]), qubit
                )
                index += 1
        for control, target in CNOTS:
            state = cnot(state, control, target)
            if record:
                tape.append(("cx", control, target, None))
    return state, tape


def support(position):
    mask = np.zeros(DIM, dtype=np.float64)
    # Pozicija 0: brojevi 1–33; pozicija 6: brojevi 7–39.
    mask[position:NUMBERS - POSITIONS + position + 1] = 1.0
    return mask


def born(state, mask):
    weights = np.abs(state) ** 2 * mask[None, :]
    mass = weights.sum(axis=1, keepdims=True)
    if np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ArithmeticError("Nulta ili neispravna masa podrške.")
    return weights / mass, mass


def objective(angles, encoded, target_cdf, layers, penalty, mask):
    state, tape = evolve(encoded, angles, layers, record=True)
    probabilities, mass = born(state, mask)
    cdf = np.cumsum(probabilities[:, :NUMBERS], axis=1)[:, :-1]
    residual = cdf - target_cdf

    # Regresija kumulativne raspodele kvadratnom greškom.
    loss = float(np.mean(residual ** 2))
    loss += penalty * float(np.mean(angles ** 2))

    g_cdf = 2.0 * residual / residual.size
    g_prob = np.zeros_like(probabilities)
    g_prob[:, :NUMBERS - 1] = np.cumsum(
        g_cdf[:, ::-1], axis=1
    )[:, ::-1]

    g_weight = (
        g_prob - np.sum(g_prob * probabilities, axis=1, keepdims=True)
    ) / mass
    adjoint = g_weight * mask[None, :] * state

    gradient = np.zeros_like(angles)
    for kind, qubit, index, before in reversed(tape):
        if kind == "cx":
            adjoint = cnot(adjoint, qubit, index)
            continue
        derivative = single_gate(
            before,
            rotation(kind, angles[index], derivative=True),
            qubit,
        )
        gradient[index] = 2.0 * np.vdot(adjoint, derivative).real
        adjoint = single_gate(
            adjoint,
            rotation(kind, angles[index]).conj().T,
            qubit,
        )

    gradient += 2.0 * penalty * angles / angles.size
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ArithmeticError("Neispravan cilj ili gradijent.")
    return loss, gradient


def check_gradient(encoded, targets, layers, penalty):
    x = encoded[:5]
    y = (targets[:5, 3, None] <= THRESHOLDS).astype(np.float64)
    angles = initial_angles(layers)
    mask = support(3)
    _, analytic = objective(angles, x, y, layers, penalty, mask)
    step = 1e-6
    for index in sorted(set((0, 1, len(angles) // 2, len(angles) - 1))):
        plus, minus = angles.copy(), angles.copy()
        plus[index] += step
        minus[index] -= step
        numeric = (
            objective(plus, x, y, layers, penalty, mask)[0]
            - objective(minus, x, y, layers, penalty, mask)[0]
        ) / (2 * step)
        if not np.isclose(numeric, analytic[index], atol=2e-7, rtol=2e-4):
            raise ArithmeticError(
                f"Gradijent nije potvrđen: parametar {index}."
            )


class QuantumModel:
    def __init__(self, layers, penalty):
        self.layers = layers
        self.penalty = penalty
        self.angles = []
        self.unconverged = []

    def fit(self, raw, targets):
        self.encoder = Encoder().fit(raw)
        encoded = self.encoder.transform(raw)
        check_gradient(encoded, targets, self.layers, self.penalty)

        self.angles = []
        self.unconverged = []
        for position in range(POSITIONS):
            target_cdf = (
                targets[:, position, None] <= THRESHOLDS
            ).astype(np.float64)
            mask = support(position)

            def fun(theta):
                return objective(
                    theta, encoded, target_cdf,
                    self.layers, self.penalty, mask,
                )

            start = initial_angles(self.layers)
            start_loss = fun(start)[0]
            result = minimize(
                fun,
                start,
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": MAX_ITER,
                    "ftol": 1e-10,
                    "gtol": 1e-7,
                    "maxls": 30,
                },
            )
            if (
                not np.isfinite(result.fun)
                or not np.isfinite(result.x).all()
                or result.fun > start_loss + 1e-10
            ):
                raise ArithmeticError("Optimizacija nije dala validan model.")

            if not result.success:
                self.unconverged.append(position + 1)
                print(
                    f"    Pozicija {position + 1}: {result.message}",
                    flush=True,
                )

            self.angles.append(result.x.copy())
            print(
                f"    Pozicija {position + 1}/7: "
                f"cilj={result.fun:.7f}, iteracije={result.nit}",
                flush=True,
            )
        return self

    def predict(self, raw):
        encoded = self.encoder.transform(raw)
        output = []
        for position, angles in enumerate(self.angles):
            state, _ = evolve(encoded, angles, self.layers)
            norms = np.sum(np.abs(state) ** 2, axis=1)
            if not np.allclose(norms, 1.0, atol=2e-10, rtol=0):
                raise ArithmeticError("Kolo ne čuva normu stanja.")
            p, _ = born(state, support(position))
            output.append(p[:, :NUMBERS])
        return np.stack(output, axis=1)

    def check_qiskit(self, raw):
        encoded = self.encoder.transform(raw[:1])
        for position, angles in enumerate(self.angles):
            circuit = QuantumCircuit(QUBITS)
            circuit.initialize(encoded[0], range(QUBITS))
            index = 0
            for _ in range(self.layers):
                for qubit in range(QUBITS):
                    circuit.rz(float(angles[index]), qubit)
                    index += 1
                    circuit.ry(float(angles[index]), qubit)
                    index += 1
                for control, target in CNOTS:
                    circuit.cx(control, target)
            reference = Statevector.from_instruction(circuit).data
            calculated, _ = evolve(encoded, angles, self.layers)
            if not np.allclose(
                reference, calculated[0], atol=2e-10, rtol=0
            ):
                raise ArithmeticError(
                    f"Qiskit neslaganje za poziciju {position + 1}."
                )


def log_probabilities(p):
    if np.any(p < 0) or not np.isfinite(p).all():
        raise ArithmeticError("Neispravne verovatnoće.")
    if not np.allclose(p.sum(axis=-1), 1.0, atol=2e-10, rtol=0):
        raise ArithmeticError("Raspodele pozicija nisu normalizovane.")
    logp = np.full_like(p, -np.inf)
    np.log(p, out=logp, where=p > 0)
    return logp


def prefix_exclusive(values):
    out = np.full_like(values, -np.inf)
    out[..., 1:] = np.logaddexp.accumulate(values, axis=-1)[..., :-1]
    return out


def suffix_exclusive(values):
    return prefix_exclusive(values[..., ::-1])[..., ::-1]


def log_normalizer(logp):
    forward = logp[..., 0, :]
    for position in range(1, POSITIONS):
        forward = logp[..., position, :] + prefix_exclusive(forward)
    z = np.logaddexp.reduce(forward, axis=-1)
    if not np.isfinite(z).all() or np.any(z > 2e-10):
        raise ArithmeticError("Neispravna zajednička normalizacija.")
    return z


def mean_nll(p, targets):
    logp = log_probabilities(p)
    z = log_normalizer(logp)
    selected = logp[
        np.arange(len(targets))[:, None],
        np.arange(POSITIONS)[None, :],
        targets - 1,
    ].sum(axis=1)
    values = z - selected
    if not np.isfinite(values).all():
        raise ArithmeticError("Nulta masa za opaženu kombinaciju.")
    return float(values.mean())


def exact_next(p):
    """Tačan MAP za c1 < c2 < ... < c7, bez heurističke pretrage."""
    logp = log_probabilities(p)
    z = float(log_normalizer(logp))
    values = logp[0].copy()
    parent = np.full((POSITIONS, NUMBERS), -1, dtype=np.int64)

    for position in range(1, POSITIONS):
        updated = np.full(NUMBERS, -np.inf)
        best_value, best_index = -np.inf, -1
        for number in range(NUMBERS):
            previous = number - 1
            if previous >= 0 and values[previous] > best_value:
                best_value = values[previous]
                best_index = previous
            if best_index >= 0 and np.isfinite(logp[position, number]):
                updated[number] = best_value + logp[position, number]
                parent[position, number] = best_index
        values = updated

    last = int(np.argmax(values))
    log_max = float(values[last])
    if not np.isfinite(log_max):
        raise ArithmeticError("Nije pronađena kombinacija pozitivne mase.")

    answer = [last]
    for position in range(POSITIONS - 1, 0, -1):
        last = int(parent[position, last])
        if last < 0:
            raise ArithmeticError("Neispravna putanja maksimuma.")
        answer.append(last)
    answer = np.asarray(answer[::-1], dtype=np.int64) + 1

    if (
        len(set(answer.tolist())) != POSITIONS
        or answer[0] < 1
        or answer[-1] > NUMBERS
        or not np.all(np.diff(answer) > 0)
    ):
        raise ArithmeticError("Neispravna NEXT kombinacija.")

    check_distribution(logp, z)
    return tuple(int(n) for n in answer), log_max - z


def check_distribution(logp, z):
    """Nezavisna provera normalizacije preko prefiksa i sufiksa."""
    forward = np.full_like(logp, -np.inf)
    backward = np.full_like(logp, -np.inf)
    forward[0] = logp[0]
    backward[-1] = logp[-1]

    for position in range(1, POSITIONS):
        forward[position] = (
            logp[position] + prefix_exclusive(forward[position - 1])
        )
    for position in range(POSITIONS - 2, -1, -1):
        backward[position] = (
            logp[position] + suffix_exclusive(backward[position + 1])
        )

    reverse_z = float(np.logaddexp.reduce(backward[0]))
    if not np.isclose(z, reverse_z, atol=2e-10, rtol=0):
        raise ArithmeticError("Prefiks i sufiks daju različitu normalizaciju.")

    marginals = np.zeros_like(logp)
    valid = (
        np.isfinite(forward)
        & np.isfinite(backward)
        & np.isfinite(logp)
    )
    marginals[valid] = np.exp(
        forward[valid] + backward[valid] - logp[valid] - z
    )
    if not np.allclose(
        marginals.sum(axis=1), 1.0, atol=2e-9, rtol=0
    ):
        raise ArithmeticError("Marginale nisu normalizovane.")
    if np.any(marginals.sum(axis=0) > 1.0 + 2e-9):
        raise ArithmeticError("Zajednička raspodela dopušta duplikate.")


def check_exact_search(p):
    """Provera algoritma na maloj podršci iscrpnim nabrajanjem."""
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
    dp_z = float(log_normalizer(logp))
    answer, log_mass = exact_next(small)
    selected_score = sum(
        logp[j, answer[j] - 1] for j in range(POSITIONS)
    )
    if (
        not np.isclose(brute_z, dp_z, atol=2e-10, rtol=0)
        or not np.isclose(
            selected_score, np.max(scores), atol=2e-10, rtol=0
        )
        or not np.isclose(
            log_mass, np.max(scores) - brute_z, atol=2e-10, rtol=0
        )
    ):
        raise ArithmeticError("Tačna pretraga nije prošla proveru.")


def run_file(path):
    history = read_history(path)
    raw = features(history)

    # Svaki istorijski prelaz koristi se kao uzorak.
    x, y = raw[:-1], history[1:]
    n = len(y)
    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)
    if min(train_end, validation_end - train_end, n - validation_end) < 10:
        raise ValueError("Premalo uzoraka za hronološku podelu.")

    print(f"\nFAJL: {path}", flush=True)
    print(
        f"Kola: {len(history)}; prelazi: {n}; "
        f"trening/validacija/test: "
        f"{train_end}/{validation_end - train_end}/{n - validation_end}",
        flush=True,
    )

    best_config = None
    best_score = math.inf

    for layers, penalty in CONFIGS:
        print(
            f"\nValidacioni kandidat: slojevi={layers}, "
            f"regularizacija={penalty:g}",
            flush=True,
        )
        model = QuantumModel(layers, penalty).fit(
            x[:train_end], y[:train_end]
        )
        predicted = model.predict(x[train_end:validation_end])
        score = mean_nll(predicted, y[train_end:validation_end])
        print(f"  Validacioni NLL: {score:.8f}", flush=True)
        if score < best_score:
            best_score = score
            best_config = (layers, penalty)

    if best_config is None:
        raise ArithmeticError("Nije izabran validan model.")

    layers, penalty = best_config
    print(
        f"\nIzabrano: slojevi={layers}, regularizacija={penalty:g}",
        flush=True,
    )

    # Test se ne koristi za izbor parametara ili promenu konfiguracije.
    print("\nUčenje na treningu i validaciji za odvojeni test:", flush=True)
    test_model = QuantumModel(layers, penalty).fit(
        x[:validation_end], y[:validation_end]
    )
    test_p = test_model.predict(x[validation_end:])
    test_nll = mean_nll(test_p, y[validation_end:])
    print(f"Odvojeni test — NLL: {test_nll:.8f}", flush=True)

    # Završno učenje koristi SVE istorijske prelaze.
    print("\nZavršno učenje na celoj istoriji:", flush=True)
    final_model = QuantumModel(layers, penalty).fit(x, y)

    final_context = raw[-1:]
    final_model.check_qiskit(final_context)
    p = final_model.predict(final_context)[0]
    check_exact_search(p)
    answer, log_mass = exact_next(p)

    # Koristi se samo za proveru mogućeg ignorisanja ulaza.
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
            "PROVERA: NEXT sadrži najmanje pet uzastopnih brojeva. "
            "Rezultat nije filtriran; pregledati ponašanje modela.",
            flush=True,
        )
    if final_model.unconverged:
        print(
            "NAPOMENA: završna optimizacija nije prijavila konvergenciju "
            f"za pozicije {final_model.unconverged}. "
            "Prikazan je njen konačni validan rezultat.",
            flush=True,
        )

    print("\nNEXT: " + ", ".join(map(str, answer)), flush=True)
    print(
        f"Logaritamska masa NEXT u naučenom modelu: {log_mass:.8f}",
        flush=True,
    )
    print(
        "Masa modela nije izmerena uspešnost predikcije. "
        "Qiskit, gradijenti, normalizacija i maksimum: provere prošle.",
        flush=True,
    )
    return {
        "path": path,
        "answer": answer,
        "probabilities": p,
        "context_change": context_change,
    }


def main():
    np.seterr(over="raise", invalid="raise", divide="raise")
    results = [run_file(path) for path in PATHS]

    print("\nKONAČNI NEXT REZULTATI", flush=True)
    for result in results:
        print(
            f"{Path(result['path']).name}: "
            + ", ".join(map(str, result["answer"])),
            flush=True,
        )

    for i, first in enumerate(results):
        for second in results[i + 1:]:
            if first["answer"] != second["answer"]:
                continue
            difference = float(np.max(np.abs(
                first["probabilities"] - second["probabilities"]
            )))
            print(
                "\nPROVERA ISTIH NEXT KOMBINACIJA:\n"
                f"  {Path(first['path']).name}\n"
                f"  {Path(second['path']).name}\n"
                f"  Najveća razlika raspodela: {difference:.10g}\n"
                f"  Promena konteksta prvog modela: "
                f"{first['context_change']:.10g}\n"
                f"  Promena konteksta drugog modela: "
                f"{second['context_change']:.10g}\n"
                "  Ulazi i modeli obrađeni su zasebno. "
                "Jednak maksimum se ne menja veštački. "
                "Ove brojke same ne isključuju degeneraciju modela.",
                flush=True,
            )


if __name__ == "__main__":
    main()



"""
FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv
Kola: 4708; prelazi: 4707; trening/validacija/test: 3294/706/707

Validacioni kandidat: slojevi=1, regularizacija=0.0001
    Pozicija 1/7: cilj=0.1677737, iteracije=50
    Pozicija 2/7: cilj=0.1272731, iteracije=39
    Pozicija 3/7: cilj=0.1050399, iteracije=41
    Pozicija 4/7: cilj=0.1021450, iteracije=28
    Pozicija 5/7: cilj=0.1058471, iteracije=32
    Pozicija 6/7: cilj=0.1304003, iteracije=40
    Pozicija 7/7: cilj=0.1793650, iteracije=41
  Validacioni NLL: 20.93436692

Validacioni kandidat: slojevi=2, regularizacija=0.001
    Pozicija 1/7: cilj=0.1476796, iteracije=60
    Pozicija 2/7: cilj=0.1111770, iteracije=73
    Pozicija 3/7: cilj=0.1029394, iteracije=62
    Pozicija 4/7: cilj=0.1002150, iteracije=54
    Pozicija 5/7: cilj=0.1031599, iteracije=53
    Pozicija 6/7: cilj=0.1197966, iteracije=55
    Pozicija 7/7: cilj=0.1475793, iteracije=60
  Validacioni NLL: 21.48328643

Validacioni kandidat: slojevi=2, regularizacija=0.01
    Pozicija 1/7: cilj=0.1559112, iteracije=50
    Pozicija 2/7: cilj=0.1230566, iteracije=51
    Pozicija 3/7: cilj=0.1074090, iteracije=45
    Pozicija 4/7: cilj=0.1044558, iteracije=44
    Pozicija 5/7: cilj=0.1067535, iteracije=60
    Pozicija 6/7: cilj=0.1261934, iteracije=40
    Pozicija 7/7: cilj=0.1696036, iteracije=40
  Validacioni NLL: 21.69923788

Izabrano: slojevi=1, regularizacija=0.0001

Učenje na treningu i validaciji za odvojeni test:
    Pozicija 1/7: cilj=0.1669776, iteracije=48
    Pozicija 2/7: cilj=0.1277828, iteracije=36
    Pozicija 3/7: cilj=0.1042406, iteracije=26
    Pozicija 4/7: cilj=0.1006082, iteracije=42
    Pozicija 5/7: cilj=0.1063045, iteracije=39
    Pozicija 6/7: cilj=0.1304735, iteracije=38
    Pozicija 7/7: cilj=0.1790131, iteracije=34
Odvojeni test — NLL: 20.42956067

Završno učenje na celoj istoriji:
    Pozicija 1/7: cilj=0.1683906, iteracije=39
    Pozicija 2/7: cilj=0.1262385, iteracije=48
    Pozicija 3/7: cilj=0.1046756, iteracije=30
    Pozicija 4/7: cilj=0.1025633, iteracije=30
    Pozicija 5/7: cilj=0.1062465, iteracije=38
    Pozicija 6/7: cilj=0.1302664, iteracije=35
    Pozicija 7/7: cilj=0.1788776, iteracije=35

NEXT: 8, 9, 10, 16, 17, 29, 35
Logaritamska masa NEXT u naučenom modelu: -8.53966015
Masa modela nije izmerena uspešnost predikcije. Qiskit, gradijenti, normalizacija i maksimum: provere prošle.



FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv
Kola: 2976; prelazi: 2975; trening/validacija/test: 2082/446/447

Validacioni kandidat: slojevi=1, regularizacija=0.0001
    Pozicija 1/7: cilj=0.1720553, iteracije=42
    Pozicija 2/7: cilj=0.1290088, iteracije=33
    Pozicija 3/7: cilj=0.1049661, iteracije=48
    Pozicija 4/7: cilj=0.1000480, iteracije=36
    Pozicija 5/7: cilj=0.1067812, iteracije=33
    Pozicija 6/7: cilj=0.1326754, iteracije=42
    Pozicija 7/7: cilj=0.1806390, iteracije=42
  Validacioni NLL: 20.50848835

Validacioni kandidat: slojevi=2, regularizacija=0.001
    Pozicija 1/7: cilj=0.1456950, iteracije=67
    Pozicija 2/7: cilj=0.1099727, iteracije=71
    Pozicija 3/7: cilj=0.0997767, iteracije=48
    Pozicija 4/7: cilj=0.0986792, iteracije=50
    Pozicija 5/7: cilj=0.1040986, iteracije=41
    Pozicija 6/7: cilj=0.1218006, iteracije=60
    Pozicija 7/7: cilj=0.1661405, iteracije=69
  Validacioni NLL: 21.96595879

Validacioni kandidat: slojevi=2, regularizacija=0.01
    Pozicija 1/7: cilj=0.1541169, iteracije=45
    Pozicija 2/7: cilj=0.1213413, iteracije=53
    Pozicija 3/7: cilj=0.1053369, iteracije=42
    Pozicija 4/7: cilj=0.1035016, iteracije=26
    Pozicija 5/7: cilj=0.1067041, iteracije=38
    Pozicija 6/7: cilj=0.1247391, iteracije=43
    Pozicija 7/7: cilj=0.1588695, iteracije=58
  Validacioni NLL: 21.65381742

Izabrano: slojevi=1, regularizacija=0.0001

Učenje na treningu i validaciji za odvojeni test:
    Pozicija 1/7: cilj=0.1710524, iteracije=48
    Pozicija 2/7: cilj=0.1257469, iteracije=48
    Pozicija 3/7: cilj=0.1072878, iteracije=40
    Pozicija 4/7: cilj=0.1005062, iteracije=58
    Pozicija 5/7: cilj=0.1064232, iteracije=29
    Pozicija 6/7: cilj=0.1320291, iteracije=30
    Pozicija 7/7: cilj=0.1799570, iteracije=45
Odvojeni test — NLL: 20.26286128

Završno učenje na celoj istoriji:
    Pozicija 1/7: cilj=0.1675482, iteracije=50
    Pozicija 2/7: cilj=0.1277574, iteracije=43
    Pozicija 3/7: cilj=0.1070030, iteracije=25
    Pozicija 4/7: cilj=0.1054821, iteracije=35
    Pozicija 5/7: cilj=0.1066034, iteracije=29
    Pozicija 6/7: cilj=0.1317729, iteracije=32
    Pozicija 7/7: cilj=0.1801047, iteracije=39

NEXT: 14, 15, 19, 22, 23, 26, 27
Logaritamska masa NEXT u naučenom modelu: -8.69819789
Masa modela nije izmerena uspešnost predikcije. Qiskit, gradijenti, normalizacija i maksimum: provere prošle.



FAJL: /Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv
Kola: 1732; prelazi: 1731; trening/validacija/test: 1211/260/260

Validacioni kandidat: slojevi=1, regularizacija=0.0001
    Pozicija 1/7: cilj=0.1747474, iteracije=36
    Pozicija 2/7: cilj=0.1322765, iteracije=49
    Pozicija 3/7: cilj=0.1062274, iteracije=39
    Pozicija 4/7: cilj=0.1018846, iteracije=29
    Pozicija 5/7: cilj=0.1060046, iteracije=31
    Pozicija 6/7: cilj=0.1304289, iteracije=31
    Pozicija 7/7: cilj=0.1772391, iteracije=31
  Validacioni NLL: 20.79333113

Validacioni kandidat: slojevi=2, regularizacija=0.001
    Pozicija 1/7: cilj=0.1526549, iteracije=69
    Pozicija 2/7: cilj=0.1165654, iteracije=60
    Pozicija 3/7: cilj=0.1048303, iteracije=44
    Pozicija 4/7: cilj=0.1009842, iteracije=68
    Pozicija 5/7: cilj=0.1037704, iteracije=52
    Pozicija 6/7: cilj=0.1194011, iteracije=65
    Pozicija 7/7: cilj=0.1583056, iteracije=64
  Validacioni NLL: 20.98296516

Validacioni kandidat: slojevi=2, regularizacija=0.01
    Pozicija 1/7: cilj=0.1607689, iteracije=65
    Pozicija 2/7: cilj=0.1338279, iteracije=66
    Pozicija 3/7: cilj=0.1097259, iteracije=57
    Pozicija 4/7: cilj=0.1043047, iteracije=34
    Pozicija 5/7: cilj=0.1102635, iteracije=65
    Pozicija 6/7: cilj=0.1309300, iteracije=46
    Pozicija 7/7: cilj=0.1619738, iteracije=54
  Validacioni NLL: 20.99547979

Izabrano: slojevi=1, regularizacija=0.0001

Učenje na treningu i validaciji za odvojeni test:
    Pozicija 1/7: cilj=0.1708769, iteracije=30
    Pozicija 2/7: cilj=0.1262146, iteracije=42
    Pozicija 3/7: cilj=0.1086391, iteracije=50
    Pozicija 4/7: cilj=0.1015799, iteracije=31
    Pozicija 5/7: cilj=0.1059806, iteracije=35
    Pozicija 6/7: cilj=0.1289246, iteracije=34
    Pozicija 7/7: cilj=0.1768596, iteracije=30
Odvojeni test — NLL: 20.89222093

Završno učenje na celoj istoriji:
    Pozicija 1/7: cilj=0.1684904, iteracije=70
    Pozicija 2/7: cilj=0.1259004, iteracije=39
    Pozicija 3/7: cilj=0.1065211, iteracije=37
    Pozicija 4/7: cilj=0.1032578, iteracije=26
    Pozicija 5/7: cilj=0.1060512, iteracije=32
    Pozicija 6/7: cilj=0.1290084, iteracije=33
    Pozicija 7/7: cilj=0.1766335, iteracije=32

NEXT: 10, 12, 14, 23, 27, 29, 31
Logaritamska masa NEXT u naučenom modelu: -8.56484661
Masa modela nije izmerena uspešnost predikcije. Qiskit, gradijenti, normalizacija i maksimum: provere prošle.



KONAČNI NEXT REZULTATI
loto7_4708_k85.csv: 8, 9, 10, 16, 17, 29, 35
loto7_4708_k85_loto_2976.csv: 14, 15, 19, 22, 23, 26, 27
loto7_4708_k85_loto_plus_1732.csv: 10, 12, 14, 23, 27, 29, 31
"""



"""
- Amplitude i faze: model može predstavljati kandidate amplitudama, a fazama omogućiti interferenciju.
- Bornovo pravilo: iz amplituda dobijam normalizovane verovatnoće, \(P(c)=|\alpha_c|^2\), za kombinaciju \(c\).
- Interferencija: konstrukcijom modela mogu pojačavati neke kandidate i potiskivati druge, prema strukturi naučenoj iz istorije.
- Amplitudno pojačavanje: Groverov postupak može pojačati kandidate koje označi zadati kriterijum. Za adaptaciju ključno je kako istorijski podaci određuju taj kriterijum; Grover ga sam ne uči.

istorija → naučene amplitude i faze → interferencija → raspodela kombinacija → NEXT 


Model uči rotacije RZ i RY u sedam registara od po šest kubita. RZ menja faze, RY omogućava interferenciju, a Bornovo pravilo daje raspodelu. Konačna NEXT kombinacija dobija se tačnim maksimumom zajedničke raspodele nad dozvoljenim sedmorkama.
Učenje koristi klasični optimizator i tačnu simulaciju kvantnih amplituda. Završni model koristi celu istoriju svakog CSV-a. Provere gradijenata, raspodele i slaganja sa Qiskitom ugrađene su u kod.

Za simulator sačuvati kompleksne amplitude i faze kroz sve operacije, pa tek na izlazu izračunati verovatnoće. RZ/RY/CNOT deluju nad amplitudama, a zatim born() računa njihove kvadrate modula. Ovo je standardan kvantni postupak

Evolucija stanja u simulatoru već je deterministička za zadati ulaz i parametre. Bornova raspodela izračunava se bez slučajnog uzorkovanja, a NEXT se bira tačnim maksimumom.
"""

"""Música de fondo seria, con tensión que sigue las escenas del reel. Generada con código: sin derechos de autor.

Capas: nota grave sostenida, acordes de cuerdas en menor, latido grave, trémolo agudo, crepitar de fuego,
un golpe antes del incendio más grande y un arroyo suave al inicio y al cierre.

    python musica_tension.py 50 prueba.wav      # escenas de igual duración, para escucharla sola
"""
import sys

import numpy as np
import soundfile as sf

SR = 48000

# Por escena del guion: acorde (MIDI), tensión de 0 a 1 y si hay fuego
ESCENAS = [
    ([50, 53, 57, 62], 0.25, False),   # 1. zonas que dan agua (Re menor)
    ([46, 50, 53, 58], 0.40, True),    # 2. 135,171 focos (Si bemol)
    ([43, 50, 55, 58], 0.50, True),    # 3. segunda más alta desde 2012 (Sol menor)
    ([45, 52, 55, 61], 0.70, True),    # 4. Puno (La 7, pide resolver)
    ([50, 51, 57, 62], 0.95, True),    # 5. Sicuani, el más grande (Re menor con Mi bemol: disonancia)
    ([46, 52, 53, 58], 0.80, True),    # 6. Huayllay, a 1 km del agua (Si bemol con Mi: tritono)
    ([41, 48, 53, 57], 0.10, False),   # 7. cierre (Fa mayor, se calma)
]


def _hz(midi):
    return 440 * 2 ** ((midi - 69) / 12)


def _filtrar(x, bajo, alto):
    """Pasa banda con FFT y bordes suaves."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.ones_like(f)
    if bajo:
        m *= 1 / (1 + (bajo / np.maximum(f, 1e-3)) ** 4)
    if alto:
        m *= 1 / (1 + (f / alto) ** 4)
    return np.fft.irfft(X * m, len(x))


def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def _tramos(dur, escenas):
    """Cada escena de la música va de la mitad de la pausa anterior a la mitad de la siguiente."""
    inicios = [0.0] + [(escenas[i - 1][1] + escenas[i][0]) / 2 for i in range(1, len(escenas))]
    return list(zip(inicios, inicios[1:] + [dur]))


def generar(dur, escenas=None, semilla=11):
    """dur en segundos; escenas = [(inicio, fin), ...] de la narración. Devuelve (n, 2) float32 a 48 kHz."""
    rng = np.random.default_rng(semilla)
    n = int(dur * SR)
    t = np.arange(n) / SR
    if escenas is None:
        paso = dur / len(ESCENAS)
        escenas = [(i * paso, (i + 1) * paso) for i in range(len(ESCENAS))]
    tramos = _tramos(dur, escenas)
    datos = [ESCENAS[min(i, len(ESCENAS) - 2)] if i < len(tramos) - 1 else ESCENAS[-1] for i in range(len(tramos))]

    # curvas de tensión y de fuego, suaves entre escenas
    centros = [(a + b) / 2 for a, b in tramos]
    tension = np.interp(t, centros, [d[1] for d in datos])
    fuego = np.interp(t, centros, [d[1] if d[2] else 0.0 for d in datos])
    izq, der = np.zeros(n), np.zeros(n)

    # 1. Nota grave sostenida (Re), crece con la tensión
    dron = sum(np.sin(2 * np.pi * _hz(26) * k * t + 0.3 * np.sin(2 * np.pi * 0.07 * t)) / k ** 1.5 for k in range(1, 7))
    dron += 0.6 * np.sin(2 * np.pi * _hz(38) * 1.002 * t)
    dron = _norm(dron) * (0.35 + 0.5 * tension)
    izq += 0.30 * dron
    der += 0.30 * dron

    # 2. Acordes de cuerdas: más brillantes cuanto más tensa la escena; se funden 0.8 s entre escenas
    for (a, b), (acorde, ten, _) in zip(tramos, datos):
        i0, i1 = max(0, int((a - 0.8) * SR)), min(n, int((b + 0.8) * SR))
        tt = t[i0:i1]
        env = np.clip(np.minimum(tt - (a - 0.8), (b + 0.8) - tt) / 1.6, 0, 1) ** 2
        caida = 2.2 - 1.2 * ten
        for lado, det in ((izq, -0.003), (der, 0.003)):
            s = np.zeros(len(tt))
            for nota in acorde:
                f = _hz(nota) * (1 + det)
                ph = rng.uniform(0, 2 * np.pi)
                vib = 0.002 * np.sin(2 * np.pi * 5 * tt + ph)
                s += sum(np.sin(2 * np.pi * f * k * (tt + vib / k) + ph * k) / k ** caida for k in range(1, 7))
            lado[i0:i1] += 0.10 * s * env

    # 3. Latido grave a 66 por minuto, solo cuando hay tensión
    golpe_t = np.arange(int(0.35 * SR)) / SR
    golpe = np.sin(2 * np.pi * (45 * golpe_t + 40 * (1 - np.exp(-golpe_t * 25)) / 25)) * np.exp(-golpe_t * 12)
    for inicio in np.arange(0.5, dur - 1, 60 / 66):
        for desfase, fuerza in ((0, 1.0), (0.24, 0.6)):
            i = int((inicio + desfase) * SR)
            nivel = max(0.0, tension[min(i, n - 1)] - 0.3) / 0.7
            if nivel > 0 and i + len(golpe) < n:
                izq[i:i + len(golpe)] += 0.45 * fuerza * nivel * golpe
                der[i:i + len(golpe)] += 0.45 * fuerza * nivel * golpe

    # 4. Trémolo agudo (Re y La; Mi bemol en el clímax) cuando la tensión pasa de 0.55
    trem = np.clip((tension - 0.55) / 0.4, 0, 1) * (0.55 + 0.45 * np.sin(2 * np.pi * 9 * t))
    agudo = sum(np.sin(2 * np.pi * _hz(m) * t) + 0.3 * np.sin(4 * np.pi * _hz(m) * t) for m in (74, 81))
    agudo += np.clip((tension - 0.85) / 0.1, 0, 1) * np.sin(2 * np.pi * _hz(75) * t)
    izq += 0.05 * trem * agudo
    der += 0.05 * trem * np.roll(agudo, 300)

    # 5. Fuego: retumbo grave y chasquidos al azar, más densos cuanto más fuego
    retumbo = _norm(_filtrar(rng.standard_normal(n), 30, 160))
    izq += 0.25 * fuego * retumbo
    der += 0.25 * fuego * np.roll(retumbo, 1200)
    chasquido = _filtrar(rng.standard_normal(int(0.012 * SR)), 1500, 7000) * np.hanning(int(0.012 * SR))
    chasquido = _norm(chasquido)
    for i in np.flatnonzero(rng.random(n) < fuego * 28 / SR):
        if i + len(chasquido) < n:
            pan, v = rng.random(), rng.uniform(0.2, 1)
            izq[i:i + len(chasquido)] += 0.10 * v * (1 - pan) * chasquido
            der[i:i + len(chasquido)] += 0.10 * v * pan * chasquido

    # 6. Antes del incendio más grande (escena 5): un siseo que sube y un golpe grave
    if len(tramos) >= 5:
        clim = tramos[4][0]
        i0, i1 = int(max(0, clim - 2.5) * SR), int(clim * SR)
        sube = _filtrar(rng.standard_normal(i1 - i0), 2000, 9000) * np.linspace(0, 1, i1 - i0) ** 3
        izq[i0:i1] += 0.12 * _norm(sube)
        der[i0:i1] += 0.12 * _norm(sube)
        bt = np.arange(int(1.8 * SR)) / SR
        bum = np.sin(2 * np.pi * (38 * bt + 60 * (1 - np.exp(-bt * 8)) / 8)) * np.exp(-bt * 2.5)
        j = min(n - len(bt), i1)
        izq[j:j + len(bt)] += 0.6 * bum
        der[j:j + len(bt)] += 0.6 * bum

    # 7. Arroyo suave en la primera y la última escena (el agua)
    agua_env = np.zeros(n)
    for k in (0, len(tramos) - 1):
        a, b = tramos[k]
        agua_env += np.clip(np.minimum(t - a + 0.8, b + 0.8 - t) / 1.6, 0, 1)
    agua = _norm(_filtrar(rng.standard_normal(n), 400, 3500) * (0.5 + _norm(_filtrar(rng.standard_normal(n), 0, 9)) ** 2))
    izq += 0.06 * agua_env * agua
    der += 0.06 * agua_env * np.roll(agua, 2400)

    x = np.stack([izq, der], axis=1)
    x *= (np.minimum(1, t / 1.5) * np.minimum(1, (dur - t) / 2.5))[:, None]
    return (0.9 * x / np.max(np.abs(x))).astype(np.float32)


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 50
    salida = sys.argv[2] if len(sys.argv) > 2 else "musica_tension.wav"
    sf.write(salida, generar(dur), SR)
    print("listo:", salida)

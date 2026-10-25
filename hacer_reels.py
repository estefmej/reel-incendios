"""Dos reels a partir de salidas/guion.json, con el código de la sesión 15 (reel.py):

    salidas/reel.mp4                voz clonada + música de fondo (baja sola cuando hablas)
    salidas/reel_solo_musica.mp4    las mismas escenas y subtítulos, solo con música

    .venv/Scripts/python hacer_reels.py
"""
import json
import subprocess
import time
from pathlib import Path

AQUI = Path(__file__).parent

import truststore  # usa los certificados de Windows (Kaspersky intercepta HTTPS)

truststore.inject_into_ssl()

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import reel  # noqa: E402
from musica_tension import SR as SR_MUSICA, generar  # noqa: E402

AUTOR = "Estefanny"
FUENTE = "NASA FIRMS, INPE, Sentinel-2, ESA WorldCover"
VOZ = AQUI / "mi_voz.m4a"
MUESTRA = AQUI / "muestra_escena1.wav"   # escena 1 con la entonación elegida; no se sube (privacidad)
SEMILLA = 1234
CREDITO = "Música: generada con código (musica_tension.py), sin derechos de autor"
SALIDAS = AQUI / "salidas"


# Zonas seguras de Instagram: abajo va la descripción del post y a la derecha los botones (me gusta, comentar...).
PIE_Y = 1420            # autora y fuente: antes en y = 1490, más cerca del borde inferior
SUB_ANCHO_MAX = 640     # ancho máximo del texto del subtítulo: la caja queda entre x ≈ 184 y x ≈ 896


def base(titulo, fuente_datos, autor, aviso=True):
    """Como reel._base, con el pie más arriba; sin aviso de voz con IA para el reel de solo música."""
    img = Image.new("RGB", (reel.ANCHO, reel.ALTO), reel.FONDO)
    d = ImageDraw.Draw(img)
    d.text((reel.MARGEN, 250), titulo.upper(), font=reel._fuente("PublicSans", 40, 800), fill=reel.ACENTO)
    if aviso:
        texto = "VOZ GENERADA CON IA"
        f = reel._fuente("PublicSans", 28, 700)
        x1 = reel.MARGEN + d.textlength(texto, font=f) + 36
        d.rounded_rectangle((reel.MARGEN, 315, x1, 365), radius=25, outline=reel.SUAVE, width=2)
        d.text((reel.MARGEN + 18, 340), texto, font=f, fill=reel.SUAVE, anchor="lm")
    pie = reel._fuente("PublicSans", 32, 500)
    ancho = reel.ANCHO - 2 * reel.MARGEN
    lineas = [f"{autor} · Fuente: {fuente_datos}"]
    if d.textlength(lineas[0], font=pie) > ancho:   # si no entra, se corta después de «Fuente:», no en medio de una fuente
        lineas = [f"{autor} · Fuente:"] + reel._envolver(d, fuente_datos, pie, ancho)
    for i, linea in enumerate(lineas):
        d.text((reel.MARGEN, PIE_Y + i * 44), linea, font=pie, fill=reel.SUAVE)
    return img


def base_sin_voz(titulo, fuente_datos, autor):
    return base(titulo, fuente_datos, autor, aviso=False)


_bloques_original = reel._bloques


def bloques_angostos(palabras, max_palabras=4, max_letras=24):
    """Como reel._bloques, pero además corta un bloque si su texto pasaría de SUB_ANCHO_MAX píxeles."""
    fuente = reel._fuente("PublicSans", 62, 800)
    medir = ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength
    salida = []
    for bloque in _bloques_original(palabras, max_palabras, max_letras):
        actual = []
        for p in bloque:
            if actual and medir(" ".join(x[2] for x in actual + [p]), font=fuente) > SUB_ANCHO_MAX:
                salida.append(actual)
                actual = []
            actual.append(p)
        salida.append(actual)
    return salida


reel._base = base
reel._bloques = bloques_angostos


def ffmpeg(*args):
    subprocess.run([reel.FFMPEG, "-y", "-v", "error", *map(str, args)], check=True)


t0 = time.time()
SALIDAS.mkdir(exist_ok=True)
trabajo = SALIDAS / "trabajo"
trabajo.mkdir(exist_ok=True)
guion = json.loads((SALIDAS / "guion.json").read_text(encoding="utf-8"))   # no se modifica

# 1. Voz clonada pareja en todas las escenas (opción D de la prueba de consistencia):
#    la escena 1 ya generada (muestra_escena1.wav) es la muestra de voz para las demás, con semilla fija,
#    menos variación y un tope de largo por escena
frase_1 = guion["escenas"][0]["frase"]
voz = reel.Voz(VOZ, modelo="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
if not MUESTRA.exists():   # primera vez: la escena 1 sale de tu grabación; escúchala y, si no te gusta, borra el archivo
    w, sr = voz.tts.generate_voice_clone(text=[frase_1], language=["Spanish"], voice_clone_prompt=voz.prompt,
                                         max_new_tokens=int(12 * len(frase_1.split())) + 48)
    sf.write(MUESTRA, w[0], sr)
    reel._reloj(f"muestra nueva: {MUESTRA.name}", t0)
muestra, sr_muestra = sf.read(MUESTRA, dtype="float32")
voz.prompt = voz.tts.create_voice_clone_prompt(ref_audio=(reel._a16k(muestra, sr_muestra), 16000), ref_text=frase_1)
voz.intento = 0


def decir_parejo(frases, estilos=None):
    """La escena 1 se usa tal cual; las demás se generan con la muestra. Cada reintento cambia la semilla."""
    salida = []
    for f in frases:
        if f == frase_1:
            salida.append(muestra)
            continue
        torch.manual_seed(SEMILLA + voz.intento)
        w, voz.sr = voz.tts.generate_voice_clone(
            text=[f], language=["Spanish"], voice_clone_prompt=voz.prompt,
            max_new_tokens=int(12 * len(f.split())) + 48,   # el doble de lo que toma leer a 2 palabras/s
            temperature=0.6, top_k=20, subtalker_temperature=0.6, subtalker_top_k=20)
        salida.append(w[0])
    voz.intento += 1
    return salida


voz.decir = decir_parejo
voz.sr = sr_muestra
reel._reloj("voz cargada; muestra: escena 1 del reel anterior", t0)
narracion = reel.narrar(voz, guion)
reel._reloj(f"narración: {narracion.duracion:.1f} s; WER por escena: "
            + ", ".join(f"{e['wer']:.2f}" for e in guion["escenas"]), t0)
reel.subtitulos(narracion, guion)
reel.a_srt(narracion, SALIDAS / "reel.srt")
reel.portada(guion, narracion, SALIDAS / "portada.png", FUENTE, AUTOR)
solo_voz = reel.video(guion, narracion, trabajo / "reel_solo_voz.mp4", FUENTE, AUTOR)
dur = narracion.duracion
reel._reloj("video con voz listo", t0)

# 2. Música seria que sigue las escenas: más tensión en los incendios, calma en el cierre
musica = generar(dur, narracion.escenas)
sf.write(SALIDAS / "musica_tension.wav", musica, SR_MUSICA)
reel._reloj("música generada", t0)

# 3. Voz + música: la música baja sola cuando hablas (sidechain)
ffmpeg("-i", solo_voz, "-i", SALIDAS / "musica_tension.wav",
       "-filter_complex",
       "[1:a]volume=0.6[m];"
       "[0:a]asplit[v][sc];"
       "[m][sc]sidechaincompress=threshold=0.02:ratio=8:attack=30:release=500[md];"
       "[v][md]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[a]",
       "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
       "-movflags", "+faststart", SALIDAS / "reel.mp4")
reel._reloj("reel con voz y música listo", t0)

# 4. Solo música: mismas escenas y subtítulos (como texto en pantalla), sin aviso de voz con IA
sin_voz = reel.Narracion(musica[:int(dur * SR_MUSICA)], SR_MUSICA, narracion.escenas, narracion.palabras)
reel._base = base_sin_voz
reel.video(guion, sin_voz, SALIDAS / "reel_solo_musica.mp4", FUENTE, AUTOR)

(SALIDAS / "creditos.txt").write_text(
    f"Autora: {AUTOR}\nFuente de los datos: {FUENTE}\n{CREDITO}\n"
    "Voz generada con IA (Qwen3-TTS) a partir de la voz de la autora.\n", encoding="utf-8")
reel._reloj(f"listo: {SALIDAS}", t0)

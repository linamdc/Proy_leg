"""
Genera archivos de movimiento (.mot) de marcha para ver los modelos OpenSim de
6 años (data/opensim_6y/) caminando en la interfaz gráfica de OpenSim.

Usa las mismas curvas que old_marcha.py (data/marcha/):
  marcha_sano.mot   cadera, rodilla y tobillo de Senden (3 a 6 años, lenta),
                    zancada de 0,95 s, avanzando a 0,84 m/s.
  marcha_P4.mot     rodilla de P4 (Lerner 2017), zancada de 1,14 s, avanzando a
                    0,40 m/s (caminadora de Lerner). SUPUESTO: cadera y tobillo
                    de Senden, porque de P4 solo hay rodilla.
La pierna izquierda repite la derecha con medio ciclo de desfase (supuesto de
simetría). La pelvis no rota: solo avanza y queda a ALTURA_PELVIS.

Ver la marcha (OpenSim 4.x, interfaz gráfica, descarga gratis en simtk.org):
  1. File > Open Model...   data/opensim_6y/Male_06years_marcha.osim
  2. File > Load Motion...  data/opensim_6y/marcha_sano.mot  (o marcha_P4.mot)
  3. Play en la barra de movimiento (abajo). Se puede cargar más de un
     movimiento y cambiar entre ellos en el panel Navigator > Motions.

Generar los .mot, copiar y pegar en la terminal desde la carpeta del repo (Proy_leg):
  python scripts/old_exportar_mot.py
"""

from pathlib import Path

import numpy as np

from old_marcha import ZANCADA_P4, ZANCADA_SANO, curvas

SALIDA = Path(__file__).resolve().parents[1] / "data" / "opensim_6y"

CICLOS = 4
FRECUENCIA = 100           # Hz
VELOCIDAD_SANO = 0.835     # m/s, Senden 3–6 años, lenta (gait_slow, Walking.Speed)
VELOCIDAD_P4 = 0.40        # m/s, caminadora de Lerner
# Altura de la pelvis sobre el piso con las piernas extendidas, sumando los
# desplazamientos de las juntas del modelo Male de 6 años: cadera 0,044 m,
# fémur 0,277 m (spline de knee_r en 0°), tibia 0,275 m, talus 0,034 m, y unos
# 0,01 m del talón. Aproximado: si el pie se hunde o flota, se ajusta aquí.
ALTURA_PELVIS = 0.64       # m

# Columnas del CSV -> coordenadas del modelo (Gait2354). La rodilla es
# NEGATIVA en flexión en este modelo, al revés que en MyoLeg y Plug-in Gait.
COORDENADAS = {
    "cadera_flexion": ("hip_flexion", 1),
    "cadera_aduccion": ("hip_adduction", 1),
    "cadera_rotacion": ("hip_rotation", 1),
    "rodilla": ("knee_angle", -1),
    "tobillo": ("ankle_angle", 1),
}


def escribir_mot(nombre, caso, zancada, velocidad):
    c, _, _, _ = curvas(caso)
    t = np.arange(0, CICLOS * zancada, 1 / FRECUENCIA)
    pct = (t / zancada % 1) * 100
    columnas = {"pelvis_tx": velocidad * t, "pelvis_ty": np.full_like(t, ALTURA_PELVIS)}
    for lado, desfase in (("_r", 0.0), ("_l", 50.0)):
        for col, (coord, signo) in COORDENADAS.items():
            columnas[coord + lado] = signo * np.interp((pct + desfase) % 100, c["pct"], c[col], period=100)

    datos = np.column_stack([t] + list(columnas.values()))
    archivo = SALIDA / nombre
    with open(archivo, "w", encoding="utf-8") as f:
        f.write(f"{nombre[:-4]}\nversion=1\nnRows={len(t)}\nnColumns={datos.shape[1]}\n"
                "inDegrees=yes\nendheader\n")
        f.write("\t".join(["time"] + list(columnas)) + "\n")
        np.savetxt(f, datos, delimiter="\t", fmt="%.6f")
    print(f"{archivo}: {CICLOS} ciclos, {len(t)} filas")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    escribir_mot("marcha_sano.mot", "sano", ZANCADA_SANO, VELOCIDAD_SANO)
    escribir_mot("marcha_P4.mot", "pc_moderado", ZANCADA_P4, VELOCIDAD_P4)


if __name__ == "__main__":
    main()

"""
Curvas de marcha de las dos piernas y de la pelvis para el modelo de 6 años.

Fuente: Senden et al., grupo de 3 a 6 años, velocidad lenta
(07_AgeGroups_slow.xlsx del archivo de OSF 3xqew), media cada 1 % del ciclo.
Cada pierna tiene su propio ciclo: el 0 % de la derecha es su contacto
inicial y el 0 % de la izquierda es el contacto inicial izquierdo. Las hojas de
la pelvis van en el ciclo de la derecha.

Convención Plug-in Gait (positivo): flexión de cadera, aducción, rotación
interna, flexión de rodilla, dorsiflexión; pelvis con inclinación anterior,
oblicuidad hacia arriba y rotación interna del lado derecho.

Salida (data/marcha/):
  senden_3a6_lenta_ambas.csv
    pct, cadera_flexion_r, cadera_aduccion_r, cadera_rotacion_r, rodilla_r, tobillo_r,
         cadera_flexion_l, cadera_aduccion_l, cadera_rotacion_l, rodilla_l, tobillo_l,
         pelvis_inclinacion, pelvis_oblicuidad, pelvis_rotacion   (°)
  senden_3a6_lenta_parametros.csv
    zancada (s), apoyo_r y apoyo_l (% del ciclo), desfase_l (% del ciclo), velocidad (m/s)

Uso (desde la carpeta del repo):
  python scripts/preparar_datos_marcha_6y.py
  python scripts/preparar_datos_marcha_6y.py --senden RUTA/07_AgeGroups_slow.xlsx
"""

import argparse
from pathlib import Path

import numpy as np
import openpyxl

DESCARGAS = Path.home() / "Downloads"
SENDEN = DESCARGAS / "3xqew-osfstorage-archive" / "07_AgeGroups_slow.xlsx"
SALIDA = Path(__file__).resolve().parents[1] / "data" / "marcha"

HOJAS = {
    "cadera_flexion_r": "Rotation RHipFlex_slow",
    "cadera_aduccion_r": "Rotation RHipAbAd_slow",
    "cadera_rotacion_r": "Rotation RHipRot_slow",
    "rodilla_r": "Rotation RKneeFlex_slow",
    "tobillo_r": "Rotation RAnkleFlex_slow",
    "cadera_flexion_l": "Rotation LHipFlex_slow",
    "cadera_aduccion_l": "Rotation LHipAbAd_slow",
    "cadera_rotacion_l": "Rotation LHipRot_slow",
    "rodilla_l": "Rotation LKneeFlex_slow",
    "tobillo_l": "Rotation LAnkleFlex_slow",
    "pelvis_inclinacion": "Rotation PelvicTil_slow",
    "pelvis_oblicuidad": "Rotation PelvicObl_slow",
    "pelvis_rotacion": "Rotation PelvicRot_slow",
}
PCT = np.arange(0, 101)


def periodica(pct, valores):
    return np.interp(PCT, np.asarray(pct, float), np.asarray(valores, float), period=100)


def leer(ruta):
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    curvas = {}
    for nombre, hoja in HOJAS.items():
        filas = [f for f in wb[hoja].iter_rows(min_row=3, values_only=True) if f[0] is not None]
        curvas[nombre] = periodica([f[0] for f in filas], [f[1] for f in filas])  # col. 1: 3-6 años
    g = {f[0]: f[1] for f in wb["gait_slow"].iter_rows(min_row=3, values_only=True) if f[0]}
    zancada = g["Stride.Time [s] mean"]
    param = {
        "zancada": zancada,
        "apoyo_r": 100 * g["R.Stance.Time [s] mean"] / g["R.Stride.Time [s] mean"],
        "apoyo_l": 100 * g["L.Stance.Time [s] mean"] / g["L.Stride.Time [s] mean"],
        # paso izquierdo: del contacto derecho al contacto izquierdo
        "desfase_l": 100 * g["L.Step.Time [s] mean"] / zancada,
        "velocidad": g["Walking.Speed [m/s] mean"],
    }
    return curvas, param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--senden", type=Path, default=SENDEN)
    args = parser.parse_args()
    curvas, param = leer(args.senden)

    SALIDA.mkdir(parents=True, exist_ok=True)
    np.savetxt(SALIDA / "senden_3a6_lenta_ambas.csv",
               np.column_stack([PCT] + list(curvas.values())), delimiter=",", fmt="%.4f",
               header="Senden et al., 3-6 anios, velocidad lenta, ambas piernas y pelvis, media (grados). "
                      "Generado por preparar_datos_marcha_6y.py\n" + ",".join(["pct"] + list(curvas)),
               comments="# ")
    np.savetxt(SALIDA / "senden_3a6_lenta_parametros.csv",
               np.array([list(param.values())]), delimiter=",", fmt="%.4f",
               header="Senden et al., 3-6 anios, velocidad lenta (gait_slow). "
                      "Generado por preparar_datos_marcha_6y.py\n" + ",".join(param), comments="# ")
    for k, v in param.items():
        print(f"  {k:10s} {v:.3f}")
    print(f"Listo: {SALIDA}")


if __name__ == "__main__":
    main()

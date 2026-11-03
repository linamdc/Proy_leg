"""
Pasa las curvas de marcha de los Excel originales a CSV dentro del repo.

Fuentes:
  - Senden et al., grupo de 3 a 6 años a velocidad lenta (07_AgeGroups_slow.xlsx
    del archivo de OSF 3xqew): cadera (flexión, aducción, rotación), rodilla y
    tobillo de la pierna derecha, media cada 1 % del ciclo.
  - Lerner et al. 2017, P4 (datos_lerner2017_P4.xlsx, digitalizado por el
    equipo): ángulo de rodilla sin exoesqueleto y torque del exoesqueleto.

Salida (data/marcha/):
  senden_3a6_lenta.csv   pct, cadera_flexion, cadera_aduccion, cadera_rotacion, rodilla, tobillo  (°)
  lerner_P4.csv          pct, rodilla (°), torque_exo (N·m)

Los huecos de P4 (0–1 % y 99–100 %) se completan interpolando a través del
final del ciclo, que es periódico.

Uso (desde la carpeta del repo; las rutas de los Excel se pueden cambiar):
  python scripts/preparar_datos_marcha.py
  python scripts/preparar_datos_marcha.py --senden RUTA.xlsx --lerner RUTA.xlsx
"""

import argparse
from pathlib import Path

import numpy as np
import openpyxl

DESCARGAS = Path.home() / "Downloads"
SENDEN = DESCARGAS / "3xqew-osfstorage-archive" / "07_AgeGroups_slow.xlsx"
LERNER = DESCARGAS / "datos_lerner2017_P4.xlsx"
SALIDA = Path(__file__).resolve().parents[1] / "data" / "marcha"

# Hojas de Senden: convención Plug-in Gait, la misma de MyoLeg (flexión,
# aducción, rotación interna y dorsiflexión positivas).
HOJAS_SENDEN = {
    "cadera_flexion": "Rotation RHipFlex_slow",
    "cadera_aduccion": "Rotation RHipAbAd_slow",
    "cadera_rotacion": "Rotation RHipRot_slow",
    "rodilla": "Rotation RKneeFlex_slow",
    "tobillo": "Rotation RAnkleFlex_slow",
}
PCT = np.arange(0, 101)


def periodica(pct, valores):
    """Interpola en 0..100 % tratando el ciclo como periódico."""
    pct, valores = np.asarray(pct, float), np.asarray(valores, float)
    return np.interp(PCT, pct, valores, period=100)


def leer_senden(ruta):
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    columnas = {}
    for nombre, hoja in HOJAS_SENDEN.items():
        filas = [f for f in wb[hoja].iter_rows(min_row=3, values_only=True) if f[0] is not None]
        # Columna 1: media del grupo de 3 a 6 años.
        columnas[nombre] = periodica([f[0] for f in filas], [f[1] for f in filas])
    return columnas


def leer_lerner(ruta):
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    filas = [f for f in wb["Datos P4"].iter_rows(min_row=2, values_only=True) if f[0] is not None]
    con_rodilla = [f for f in filas if f[1] is not None]
    rodilla = periodica([f[0] for f in con_rodilla], [f[1] for f in con_rodilla])
    torque = np.array([next(f[9] for f in filas if f[0] == p) for p in PCT], float)
    return {"rodilla": rodilla, "torque_exo": torque}


def guardar(nombre, columnas, comentario):
    SALIDA.mkdir(parents=True, exist_ok=True)
    datos = np.column_stack([PCT] + list(columnas.values()))
    np.savetxt(SALIDA / nombre, datos, delimiter=",", fmt="%.4f",
               header=comentario + "\n" + ",".join(["pct"] + list(columnas)), comments="# ")
    print(f"{SALIDA / nombre}: {len(PCT)} filas")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--senden", type=Path, default=SENDEN)
    parser.add_argument("--lerner", type=Path, default=LERNER)
    args = parser.parse_args()

    guardar("senden_3a6_lenta.csv", leer_senden(args.senden),
            "Senden et al., 3-6 anios, velocidad lenta, pierna derecha, media (grados). Generado por preparar_datos_marcha.py")
    guardar("lerner_P4.csv", leer_lerner(args.lerner),
            "Lerner et al. 2017, P4: rodilla sin exo (grados) y torque del exo (N*m). Generado por preparar_datos_marcha.py")


if __name__ == "__main__":
    main()

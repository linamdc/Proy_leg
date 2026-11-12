"""
Fuerzas del suelo y momentos de rodilla medidos, para la dinámica inversa.

Fuentes:
  - Senden et al., 3 a 6 años, velocidad lenta (07_AgeGroups_slow.xlsx del
    archivo de OSF 3xqew), media cada 1 % del ciclo de cada pierna:
    fuerza vertical y horizontal (anteroposterior) del suelo, en pesos
    corporales, y momento de flexo-extensión de rodilla derecha (N·m/kg,
    + = extensor). Cada pierna en su propio ciclo (0 % = su contacto inicial).
  - Lerner et al. 2017, P4 (datos_lerner2017_P4.xlsx, digitalizado por el
    equipo): momento de rodilla sin exoesqueleto (N·m/kg, + = extensor). Los
    huecos (0–5 % y 99–100 %) se completan interpolando a través del final
    del ciclo.

Salida (data/marcha/):
  senden_3a6_lenta_fuerzas.csv   pct, fvert_r, fhor_r, fvert_l, fhor_l (pesos corporales),
                                 momento_rodilla_r (N·m/kg)
  lerner_P4_momento.csv          pct, momento_rodilla (N·m/kg)

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):
  python scripts/preparar_fuerzas_marcha.py
"""

import argparse
from pathlib import Path

import numpy as np
import openpyxl

from preparar_datos_marcha import LERNER, PCT, SENDEN, guardar, periodica

HOJAS = {
    "fvert_r": "FvertR_slow",
    "fhor_r": "FhorR_slow",
    "fvert_l": "FvertL_slow",
    "fhor_l": "FhorL_slow",
    "momento_rodilla_r": "Moment RKneeFlex_slow",
}


def leer_senden(ruta):
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    columnas = {}
    for nombre, hoja in HOJAS.items():
        filas = [f for f in wb[hoja].iter_rows(min_row=3, values_only=True) if f[0] is not None]
        columnas[nombre] = periodica([f[0] for f in filas], [f[1] for f in filas])   # media 3–6 años
    return columnas


def leer_lerner(ruta):
    wb = openpyxl.load_workbook(ruta, read_only=True, data_only=True)
    filas = [f for f in wb["Datos P4"].iter_rows(min_row=2, values_only=True)
             if f[0] is not None and f[3] is not None]
    return {"momento_rodilla": periodica([f[0] for f in filas], [f[3] for f in filas])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--senden", type=Path, default=SENDEN)
    parser.add_argument("--lerner", type=Path, default=LERNER)
    args = parser.parse_args()
    guardar("senden_3a6_lenta_fuerzas.csv", leer_senden(args.senden),
            "Senden et al., 3-6 anios, lenta: fuerzas del suelo (pesos corporales) y momento de rodilla der "
            "(N*m/kg, + extensor). Generado por preparar_fuerzas_marcha.py")
    guardar("lerner_P4_momento.csv", leer_lerner(args.lerner),
            "Lerner et al. 2017, P4 sin exo: momento de rodilla (N*m/kg, + extensor). "
            "Generado por preparar_fuerzas_marcha.py")


if __name__ == "__main__":
    main()

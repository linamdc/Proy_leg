"""
Reproduce un ciclo de marcha en la pierna del niño, en el visor de MuJoCo.

La pierna sigue los ángulos medidos en cada % del ciclo (cinemática impuesta):
no cae ni rebota, se mueve como al caminar. La pelvis está fija, así que la
pierna "camina en el aire", como sobre un banco.

  - sano: cadera, rodilla y tobillo de Senden (3 a 6 años, lenta), zancada de 0,95 s.
  - pc_*: rodilla de P4 (Lerner 2017, sin exoesqueleto), zancada de 1,14 s.
    SUPUESTO: cadera y tobillo de Senden, porque de P4 solo hay rodilla.
  - pc_*_exo: lo mismo, y el motor aplica el perfil de torque del exoesqueleto
    de Lerner (apoyo 0,19 y 0,17 N·m/kg, oscilación temprana 0, tardía 0,07).
    Como la cinemática está impuesta, el torque del motor no cambia el
    movimiento: se muestra en la terminal y en el sensor exo_torque.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):

  python scripts/old_marcha.py --caso sano
  python scripts/old_marcha.py --caso pc_moderado
  python scripts/old_marcha.py --caso pc_moderado_exo
  python scripts/old_marcha.py --caso pc_severo_exo --lento 3      # 3 veces más lento

Los datos salen de data/marcha/ (ver preparar_datos_marcha.py).
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "modelos" / "old_pierna_nino"
DATOS = RAIZ / "data" / "marcha"

ZANCADA_SANO = 0.952   # s, Senden 3–6 años, lenta (gait_slow, Stride.Time)
ZANCADA_P4 = 1.14      # s, 2 × 60 / 105 pasos/min (Lerner Tabla 2)
DESPEGUE_SANO = 58.6   # % del ciclo: Stance.Time / Stride.Time = 0,556 / 0,952
DESPEGUE_P4 = 67.0     # % del ciclo (banda gris de la Figura 2 de Lerner)

ARTICULACIONES = {
    "cadera_flexion": "hip_flexion_r",
    "cadera_aduccion": "hip_adduction_r",
    "cadera_rotacion": "hip_rotation_r",
    "rodilla": "knee_angle_r",
    "tobillo": "ankle_angle_r",
}


def leer_csv(nombre):
    with open(DATOS / nombre, encoding="utf-8") as f:
        encabezado = [linea for linea in f if linea.startswith("#")][-1]
    columnas = encabezado.lstrip("# ").strip().split(",")
    datos = np.loadtxt(DATOS / nombre, delimiter=",", comments="#")
    return {c: datos[:, i] for i, c in enumerate(columnas)}


def curvas(caso):
    """Curvas (° por % del ciclo), torque del exo (N·m o None), zancada y despegue."""
    senden = leer_csv("senden_3a6_lenta.csv")
    if caso == "sano":
        return senden, None, ZANCADA_SANO, DESPEGUE_SANO
    p4 = leer_csv("lerner_P4.csv")
    c = dict(senden)
    c["rodilla"] = p4["rodilla"]
    torque = p4["torque_exo"] if caso.endswith("_exo") else None
    return c, torque, ZANCADA_P4, DESPEGUE_P4


def fase(pct, despegue):
    if pct < despegue:
        return "apoyo"
    return "oscilación temprana" if pct < 80 else "oscilación tardía"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caso", default="pc_moderado_exo")
    parser.add_argument("--lento", type=float, default=1.0, help="factor de cámara lenta")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODELOS / f"pierna_nino_{args.caso}.xml"))
    data = mujoco.MjData(model)
    c, torque, zancada, despegue = curvas(args.caso)
    pct = c["pct"]

    juntas = {k: model.joint(j).id for k, j in ARTICULACIONES.items()}
    rodilla = juntas["rodilla"]
    # Grados de libertad acoplados a la rodilla (traslaciones de tibia y rótula).
    acoples = [(model.jnt_qposadr[model.eq_obj1id[i]], model.eq_data[i][:5].copy())
               for i in range(model.neq)
               if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[i] == rodilla]
    exo = model.actuator("exo_rodilla").id if torque is not None else None

    def poner(p):
        data.qpos[:] = model.qpos0
        for k, j in juntas.items():
            lo, hi = model.jnt_range[j]
            data.qpos[model.jnt_qposadr[j]] = np.clip(np.radians(np.interp(p, pct, c[k], period=100)), lo, hi)
        q = data.qpos[model.jnt_qposadr[rodilla]]
        for adr, coef in acoples:
            data.qpos[adr] = np.polyval(coef[::-1], q)
        if exo is not None:
            data.ctrl[exo] = np.interp(p, pct, torque, period=100)
        mujoco.mj_forward(model, data)

    print(f"{args.caso}: zancada de {zancada:.2f} s" + (f" (x{args.lento:g} más lento)" if args.lento != 1 else "")
          + ". Cierra la ventana para salir.")
    if args.caso != "sano":
        print("  Cadera y tobillo: Senden (supuesto); rodilla: P4 de Lerner.")
    with mujoco.viewer.launch_passive(model, data) as v:
        inicio = time.time()
        while v.is_running():
            t = (time.time() - inicio) / args.lento
            p = (t / zancada % 1) * 100
            data.time = t
            poner(p)
            v.sync()
            texto = (f"  ciclo {p:5.1f} %  {fase(p, despegue):<20} rodilla "
                     f"{np.degrees(data.qpos[model.jnt_qposadr[rodilla]]):5.1f}°")
            if exo is not None:
                texto += f"  exo {data.sensor('exo_torque').data[0]:4.2f} N·m"
            print(texto, end="\r")
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()

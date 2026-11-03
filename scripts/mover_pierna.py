"""
Mueve la pierna del niño en el visor de MuJoCo.

Sentado: la cadera se sostiene a 90° de flexión (sin aducción ni rotación) y
la rodilla cuelga. Cada PERIODO / 2
segundos la asistencia se prende y se apaga:
  - con exoesqueleto (casos _exo): el motor da TORQUE_EXO de extensión;
  - sin exoesqueleto: se activa el cuádriceps al nivel ACTIVACION_CUADRICEPS.
Así se ve cómo extiende la rodilla cada caso y cómo cae al soltarla.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):

  python scripts/mover_pierna.py --caso pc_moderado_exo
  python scripts/mover_pierna.py --caso pc_severo_exo
  python scripts/mover_pierna.py --caso pc_moderado          # sin exo: activa el cuádriceps
  python scripts/mover_pierna.py --caso sano

Casos: sano, pc_sin_acortamiento, pc_moderado, pc_severo, y los de PC con _exo.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

CARPETA = Path(__file__).resolve().parents[1] / "modelos" / "pierna_nino"

# Postura de la cadera, sostenida: sentado con el muslo apoyado. Se sostienen
# los tres grados de libertad, no solo la flexión: MyoLeg no tiene ligamentos
# ni tono muscular, así que con los músculos relajados la aducción y la
# rotación quedan casi libres y el muslo se tuerce y golpea los topes.
CADERA = {"hip_flexion_r": 90.0, "hip_adduction_r": 0.0, "hip_rotation_r": 0.0}   # °
PERIODO = 4.0                  # s: mitad con asistencia, mitad sin
TORQUE_EXO = 4.0               # N·m extensor (torque pico nominal del requerimiento)
ACTIVACION_CUADRICEPS = 0.3    # 0 a 1
CUADRICEPS = ["recfem_r", "vasint_r", "vaslat_r", "vasmed_r"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caso", default="pc_moderado_exo")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(CARPETA / f"pierna_nino_{args.caso}.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("popliteo").id)

    cadera = [(model.jnt_qposadr[model.joint(j).id], model.jnt_dofadr[model.joint(j).id], np.radians(a))
              for j, a in CADERA.items()]
    rodilla_q = model.jnt_qposadr[model.joint("knee_angle_r").id]
    con_exo = args.caso.endswith("_exo")
    if con_exo:
        exo = model.actuator("exo_rodilla").id
    else:
        cuadriceps = [model.actuator(m).id for m in CUADRICEPS]

    print(f"{args.caso}: asistencia {'exo ' + str(TORQUE_EXO) + ' N·m' if con_exo else f'cuádriceps al {ACTIVACION_CUADRICEPS:.0%}'}, "
          f"prendida y apagada cada {PERIODO / 2:.0f} s. Cierra la ventana para salir.")
    with mujoco.viewer.launch_passive(model, data) as v:
        ultimo_print = 0.0
        while v.is_running():
            inicio = time.time()
            prendida = (data.time % PERIODO) < PERIODO / 2
            if con_exo:
                data.ctrl[exo] = TORQUE_EXO if prendida else 0.0
            else:
                data.ctrl[cuadriceps] = ACTIVACION_CUADRICEPS if prendida else 0.0
            # Cadera sostenida: como si el niño estuviera sentado.
            for q, dof, angulo in cadera:
                data.qpos[q] = angulo
                data.qvel[dof] = 0.0
            mujoco.mj_step(model, data)
            v.sync()
            if data.time - ultimo_print > 0.5:
                print(f"  t = {data.time:5.1f} s  asistencia {'ON ' if prendida else 'OFF'}  "
                      f"rodilla {np.degrees(data.qpos[rodilla_q]):5.1f}°", end="\r")
                ultimo_print = data.time
            # Tiempo real.
            espera = model.opt.timestep - (time.time() - inicio)
            if espera > 0:
                time.sleep(espera)


if __name__ == "__main__":
    main()

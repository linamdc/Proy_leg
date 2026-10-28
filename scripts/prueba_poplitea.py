"""
Prueba del ángulo poplíteo simulada en MyoLeg (myo_sim + MuJoCo).

Qué hace:
  1. Carga el modelo completo de piernas (myolegs, 80 músculos).
  2. Simula tres condiciones de los isquiotibiales: normal, acortamiento
     moderado y acortamiento severo.
  3. Para cadera a 0° (de pie) y a 90° (posición de la prueba) barre la rodilla
     desde 120° de flexión hasta extensión completa y calcula el torque pasivo
     que hacen los isquiotibiales sobre la rodilla.
  4. El ángulo poplíteo es el ángulo de flexión en el que ese torque alcanza
     el torque que aplica el examinador (T_EXAMINADOR).

Uso:
  python prueba_poplitea.py              # tabla + gráfica
  python prueba_poplitea.py --ver severo # además abre el visor en la pose final

Convención del modelo: knee_angle_r = 0 es extensión completa y los valores
positivos son flexión (0° a 120°). hip_flexion_r positivo es flexión de cadera.
"""

import argparse

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import myo_sim

# ----------------------------------------------------------------------------
# Parámetros que puedes cambiar
# ----------------------------------------------------------------------------

# Torque con el que el examinador intenta extender la rodilla (N·m).
# Está calibrado para que el modelo sin alterar dé un ángulo poplíteo normal.
T_EXAMINADOR = 25.0

# Acortamiento expresado como fracción de la longitud óptima de fibra (L0).
# 0.10 = el músculo empieza a generar tensión pasiva con un 10% de L0 antes.
CONDICIONES = {
    "normal": 0.00,
    "moderado": 0.10,
    "severo": 0.20,
}

# Músculos que se acortan: los tres isquiotibiales biarticulares.
# La cabeza corta del bíceps (bfsh_r) solo cruza la rodilla; si la agregas
# aquí, la rodilla también quedará limitada con la cadera recta.
ACORTADOS = ["semimem_r", "semiten_r", "bflh_r"]

# Músculos cuyo torque pasivo se suma en la rodilla.
ISQUIOTIBIALES = ["semimem_r", "semiten_r", "bflh_r", "bfsh_r"]

ANGULOS_CADERA = [0, 90]
BARRIDO_RODILLA = np.arange(120.0, -0.01, -0.5)  # de flexión a extensión

# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------

model, data = myo_sim.load("myolegs")

JNT_CADERA = model.joint("hip_flexion_r").id
JNT_RODILLA = model.joint("knee_angle_r").id
DOF_RODILLA = model.jnt_dofadr[JNT_RODILLA]

# La rodilla de MyoLeg tiene grados de libertad acoplados (traslaciones de la
# tibia y la rótula) que siguen al ángulo de rodilla mediante polinomios.
# Como aquí ponemos la pose a mano, hay que aplicarlos nosotros.
ACOPLES = [
    (model.jnt_qposadr[model.eq_obj1id[i]], model.eq_data[i][:5].copy())
    for i in range(model.neq)
    if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
    and model.eq_obj2id[i] == JNT_RODILLA
]

LENGTHRANGE_ORIGINAL = model.actuator_lengthrange.copy()


def aplicar_acortamiento(fraccion):
    """Desplaza hacia abajo el rango de longitudes de los músculos acortados.

    En MuJoCo la longitud normalizada del músculo es
        L = range0 + (longitud - lengthrange0) / L0
    así que restar fraccion * L0 a lengthrange hace que, a la misma longitud
    real, el músculo quede más estirado y genere tensión pasiva antes.
    """
    model.actuator_lengthrange[:] = LENGTHRANGE_ORIGINAL
    for nombre in ACORTADOS:
        a = model.actuator(nombre).id
        r0, r1 = model.actuator_gainprm[a][:2]
        lr0, lr1 = LENGTHRANGE_ORIGINAL[a]
        L0 = (lr1 - lr0) / (r1 - r0)
        model.actuator_lengthrange[a] -= fraccion * L0


def poner_pose(cadera_deg, rodilla_deg):
    """Coloca la pierna derecha en la pose pedida, con músculos relajados."""
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0
    data.qpos[model.jnt_qposadr[JNT_CADERA]] = np.radians(cadera_deg)
    q = np.radians(rodilla_deg)
    data.qpos[model.jnt_qposadr[JNT_RODILLA]] = q
    for adr, c in ACOPLES:
        data.qpos[adr] = sum(c[k] * q**k for k in range(5))
    mujoco.mj_forward(model, data)


def brazo_momento_rodilla(actuador_id):
    """Brazo de momento del músculo sobre la rodilla (matriz dispersa)."""
    inicio = data.moment_rowadr[actuador_id]
    for k in range(inicio, inicio + data.moment_rownnz[actuador_id]):
        if data.moment_colind[k] == DOF_RODILLA:
            return data.actuator_moment[k]
    return 0.0


def torque_pasivo_rodilla():
    """Torque de flexión (N·m) que hacen los isquiotibiales relajados.

    Con activación 0, la fuerza del músculo es solo su componente pasiva.
    Positivo = tiende a flexionar la rodilla, es decir, se opone a extenderla.
    """
    total = 0.0
    for nombre in ISQUIOTIBIALES:
        a = model.actuator(nombre).id
        total += brazo_momento_rodilla(a) * data.actuator_force[a]
    return total


def curva_torque(cadera_deg):
    torques = []
    for rodilla in BARRIDO_RODILLA:
        poner_pose(cadera_deg, rodilla)
        torques.append(torque_pasivo_rodilla())
    return np.array(torques)


def angulo_popliteo(torques):
    """Ángulo de flexión más extendido al que se llega con T_EXAMINADOR."""
    # El barrido va de flexión a extensión; el torque crece al extender.
    # Nos quedamos con el tramo continuo desde 120° hasta que se supera T.
    ultimo = BARRIDO_RODILLA[0]
    for ang, t in zip(BARRIDO_RODILLA, torques):
        if t > T_EXAMINADOR:
            return ultimo
        ultimo = ang
    return 0.0


def clasificar(angulo):
    if angulo < 25:
        return "normal"
    if angulo <= 45:
        return "acortamiento moderado"
    if angulo > 50:
        return "acortamiento severo"
    return "límite moderado/severo"


# ----------------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", choices=list(CONDICIONES), default=None,
                        help="abre el visor en la pose final con cadera a 90°")
    args = parser.parse_args()

    resultados = {}
    for cond, fraccion in CONDICIONES.items():
        aplicar_acortamiento(fraccion)
        for cadera in ANGULOS_CADERA:
            torques = curva_torque(cadera)
            resultados[(cond, cadera)] = (torques, angulo_popliteo(torques))

    print(f"\nTorque del examinador: {T_EXAMINADOR:.0f} N·m\n")
    print(f"{'Condición':<10} {'Acort.':>7} {'Cadera 0°':>11} {'Cadera 90°':>12}   Clasificación (cadera 90°)")
    print("-" * 80)
    for cond, fraccion in CONDICIONES.items():
        a0 = resultados[(cond, 0)][1]
        a90 = resultados[(cond, 90)][1]
        print(f"{cond:<10} {fraccion*100:>6.0f}% {a0:>10.1f}° {a90:>11.1f}°   {clasificar(a90)}")
    print("\nCon la cadera a 0° la rodilla llega a extensión en los tres casos:")
    print("la limitación depende de la cadera, así que es muscular y no articular.\n")

    graficar(resultados)

    if args.ver:
        aplicar_acortamiento(CONDICIONES[args.ver])
        poner_pose(90, resultados[(args.ver, 90)][1])
        ver_pose()


def graficar(resultados):
    # Rampa ordinal de un solo tono: más oscuro = más acortamiento.
    colores = {"normal": "#86b6ef", "moderado": "#2a78d6", "severo": "#104281"}
    tinta = "#3d3d3a"
    tinta_suave = "#73726c"

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, cadera in zip(ejes, ANGULOS_CADERA):
        for cond in CONDICIONES:
            torques, ang = resultados[(cond, cadera)]
            ax.plot(BARRIDO_RODILLA, torques, color=colores[cond], lw=2, label=cond)
            if cadera == 90:
                ax.plot(ang, T_EXAMINADOR, "o", ms=8, color=colores[cond],
                        mec="white", mew=2, zorder=3)
                ax.annotate(f"{ang:.1f}°", (ang, T_EXAMINADOR),
                            xytext=(0, -18), textcoords="offset points",
                            ha="center", color=tinta, fontsize=9)
        ax.axhline(T_EXAMINADOR, color=tinta_suave, lw=1, ls="--")
        ax.text(118, T_EXAMINADOR + 1.5, "torque del examinador",
                color=tinta_suave, fontsize=8)
        ax.set_xlim(120, 0)  # de flexión (izq.) a extensión (der.)
        ax.set_title(f"Cadera a {cadera}°", color=tinta, fontsize=11, loc="left")
        ax.set_xlabel("Flexión de rodilla (°)  →  extensión", color=tinta_suave)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#c3c2b7")
        ax.tick_params(colors=tinta_suave)
    ejes[0].set_ylabel("Torque pasivo de isquiotibiales (N·m)", color=tinta_suave)
    ejes[0].legend(frameon=False, title="Isquiotibiales", loc="upper left")
    fig.suptitle("Prueba del ángulo poplíteo simulada en MyoLeg",
                 color=tinta, fontsize=13, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig("prueba_poplitea.png", dpi=150)
    print("Gráfica guardada en prueba_poplitea.png")
    plt.show()


def ver_pose():
    import mujoco.viewer
    # Pose estática: no se simula, solo se muestra.
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)
            v.sync()


if __name__ == "__main__":
    main()
"""
Simulación DINÁMICA de la marcha del niño de 6 años con torques que siguen la
marcha medida (opción 1: el exoesqueleto sí cambia las fuerzas del niño).

ESTADO: la simulación es estable y sigue la marcha (error de rodilla ~1°), pero
sus resultados del exoesqueleto NO son confiables todavía. Con la pelvis
llevada por el arnés y el pie fijo en el suelo, cada pierna es una cadena
cerrada sobrerrestringida: la ayuda del exo y la falta de fuerza de la rodilla
se reparten entre cadera, tobillo, suelo y arnés como fuerzas internas, en vez
de cambiar el torque o el ángulo de la rodilla. Medido (pc vs pc_exo, el exo
aporta 0,16 N·m/kg en el apoyo): el torque biológico de rodilla baja entre 0,01
y 0,29 N·m/kg según la rigidez del arnés y del PD; y con el extensor limitado
a 0,69 N·m/kg la rodilla no se flexiona más (8,5° en sano y en débil).

  - Pelvis: un "arnés" la lleva por la trayectoria de marcha_6y.py. Es firme en
    avance, lateral y rotaciones, y BLANDO en vertical: el peso lo cargan las
    piernas por el contacto de los pies con el suelo (se reporta cuánto peso
    toma el arnés, debería ser poco).
  - Piernas y tronco: dinámica real (masa, inercia, contacto con el suelo).
  - Niño: un controlador PD en cada articulación (lumbar, cadera, rodilla,
    tobillo) sigue la marcha de referencia. Su torque es el TORQUE BIOLÓGICO.
    Los músculos van sin fuerza: todo el torque del niño pasa por el PD (en la
    opción 2 lo reemplazarán los músculos).
  - Exoesqueleto (exo.py): en cada rodilla, el perfil de torque de Lerner según
    la fase de esa pierna, saturado a ±6 N·m.

Experimento 1, misma marcha de P4 con y sin exo: el niño pone menos torque
cuando el exo ayuda. Corridas: sin dispositivo, exo apagado (solo su masa) y
exo encendido.
Experimento 2, niño débil: el torque extensor biológico de la rodilla se
limita a la fuerza de P4 (FUERZA_EXTENSORA_P4) y la referencia es la marcha del
niño sano. Si el límite no alcanza, la rodilla se queda más flexionada; con el
exo, se extiende más.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):

  python scripts/old_dinamica_6y.py                        # los dos experimentos: tabla + gráficas
  python scripts/old_dinamica_6y.py --ver pc_exo           # ver una corrida en el visor
  python scripts/old_dinamica_6y.py --ver pc               #   (pc, pc_exo_apagado, pc_exo,
  python scripts/old_dinamica_6y.py --ver debil_exo        #    sano, debil, debil_exo)

Las gráficas se guardan en resultados/.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import marcha_6y as m6

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"

PASO = 0.001          # s; el PD es rígido y con 5 ms se vuelve inestable
CICLOS = 5            # ciclos simulados por corrida
DESCARTE = 1          # ciclos iniciales que no se promedian (transitorio)

# Fuerza extensora isométrica de rodilla de P4 (Lerner Tabla 1), por kg.
FUERZA_EXTENSORA_P4 = 0.69   # N·m/kg

# Ganancias (Kp en N·m/rad o N/m; Kd en N·m·s/rad o N·s/m).
ARNES = {
    "pelvis_tx": (20000, 800), "pelvis_tz": (20000, 800),
    "pelvis_ty": (2000, 200),          # blando: el peso va a las piernas
    "pelvis_rotation": (800, 30), "pelvis_tilt": (800, 30), "pelvis_list": (800, 30),
}
BIOLOGICO = {   # error de rodilla ~1,2° RMS; más rígido sigue mejor pero el torque se vuelve ruidoso
    "lumbar": (1200, 30),
    "hip": (1500, 30),
    "knee_angle": (900, 16),
    "ankle_angle": (600, 10),
}

CORRIDAS = {
    #  nombre: (caso de marcha, exo en el modelo, exo encendido, débil)
    "pc": ("pc", False, False, False),
    "pc_exo_apagado": ("pc", True, False, False),
    "pc_exo": ("pc_exo", True, True, False),
    "sano": ("sano", False, False, False),
    "debil": ("sano", False, False, True),
    "debil_exo": ("sano", True, True, True),
}
TITULOS = {
    "pc": "P4 sin dispositivo", "pc_exo_apagado": "P4 con exo apagado", "pc_exo": "P4 con exo",
    "sano": "sano", "debil": "débil sin exo", "debil_exo": "débil con exo",
}


class Simulacion:
    def __init__(self, corrida):
        caso, con_exo, encendido, debil = CORRIDAS[corrida]
        self.m = m6.caso_marcha(caso)
        if encendido and self.m["torque"] is None:
            # Débil con exo: perfil de Lerner sobre la marcha del sano. El
            # perfil cambia de fase en el despegue de P4 (67 %); se ajusta al
            # despegue de esta marcha para no empujar la extensión en la
            # oscilación temprana.
            perfil = m6.caso_marcha("pc_exo")["torque"]
            pct = self.m["curvas"]["pct"]
            origen = np.interp(pct, [0, self.m["despegue"]["r"], 100], [0, m6.DESPEGUE_P4, 100])
            self.m["torque"] = np.interp(origen, pct, perfil)
        self.model, self.data = m6.cargar_modelo(con_exo)
        model = self.model
        model.opt.timestep = PASO
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # músculos sin fuerza (activa ni pasiva): el torque biológico es el PD
        for i in range(model.nu):
            if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE:
                model.actuator_gainprm[i][7] = model.actuator_biasprm[i][7] = 0.0

        # referencia: tabla de poses de marcha_6y (con su propio MjData)
        ref = m6.Marcha(model, mujoco.MjData(model), self.m)
        ref.precalcular()
        # el eje de avance de la pelvis no es del todo horizontal en el modelo
        # convertido: al avanzar un ciclo, la altura se corrige con esta pendiente
        self.tabla, self.por_ciclo = ref.tabla, ref.por_ciclo
        self.pendiente = ref.eje_x[2] / ref.eje_y[2]
        self.tx, self.ty = ref.tx, ref.ty
        self.ref = ref

        self.masa = m6.cargar_modelo(False)[0].body_mass.sum()   # masa del niño, sin exo
        self.pd = []
        for j in range(model.njnt):
            nombre = model.joint(j).name
            g = ARNES.get(nombre) or next((v for k, v in BIOLOGICO.items() if nombre.startswith(k)), None)
            if g is not None:
                self.pd.append((nombre, model.jnt_qposadr[j], model.jnt_dofadr[j], *g))
        self.bio = [p for p in self.pd if p[0] not in ARNES]
        # La parte de amortiguamiento del PD (-Kd·v) va como amortiguamiento de
        # la junta, que implicitfast integra de forma implícita: aplicada como
        # fuerza explícita, el PD se vuelve inestable con paso de 1 ms. En
        # paso() se aplica el resto, Kp·e + Kd·v_ref; el torque total es el
        # mismo PD.
        self.amort0 = model.dof_damping.copy()
        for _, _, va, _, kd in self.pd:
            model.dof_damping[va] += kd
        self.rodilla = {l: (model.jnt_qposadr[model.joint(f"knee_angle_{l}").id],
                            model.jnt_dofadr[model.joint(f"knee_angle_{l}").id]) for l in m6.LADOS}
        # Contacto del pie solo en su fase de apoyo. La referencia mezcla curvas
        # de distintos niños y el pie que oscila pasa a ~1,5 cm del suelo: con
        # unos grados de error de seguimiento roza el piso, carga peso y el PD
        # pelea contra el suelo. Los niños medidos no arrastran el pie.
        self.pies = {l: [model.geom(f"contacto_{n}_{l}").id for n in m6.CONTACTOS] for l in m6.LADOS}
        self.tope_ext = FUERZA_EXTENSORA_P4 * self.masa if debil else None
        self.exo = ({l: model.actuator(f"exo_rodilla_{l}").id for l in m6.LADOS}
                    if encendido else None)

    def q_ref(self, t):
        z = self.m["zancada"]
        c = t / z
        k = int(np.floor(c))
        f = (c - k) * m6.MUESTRAS
        i0 = int(f) % m6.MUESTRAS
        i1 = (i0 + 1) % m6.MUESTRAS
        w = f - int(f)
        a, b = self.tabla[i0].copy(), self.tabla[i1].copy()
        if i1 == 0:                       # el ciclo siguiente empieza un ciclo más adelante
            b[self.tx] += self.por_ciclo
            b[self.ty] -= self.por_ciclo * self.pendiente
        q = (1 - w) * a + w * b
        q[self.tx] += k * self.por_ciclo
        q[self.ty] -= k * self.por_ciclo * self.pendiente
        return q

    def v_ref(self, t, h=1e-3):
        return (self.q_ref(t + h) - self.q_ref(t - h)) / (2 * h)

    def fase(self, t, lado):
        p = (t / self.m["zancada"] % 1) * 100
        return self.ref.fase_pierna(p, lado)

    def iniciar(self):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = self.q_ref(0.0)
        d.qvel[:] = self.v_ref(0.0)
        mujoco.mj_forward(self.model, d)

    def paso(self):
        """Un paso de simulación. Devuelve el torque biológico de cada junta."""
        d, t = self.data, self.data.time
        qr, vr = self.q_ref(t), self.v_ref(t)
        d.qfrc_applied[:] = 0.0
        tau = {}
        for nombre, qa, va, kp, kd in self.pd:
            u = kp * (qr[qa] - d.qpos[qa]) + kd * (vr[va] - d.qvel[va])
            if self.tope_ext is not None and nombre.startswith("knee_angle"):
                u = min(u, self.tope_ext)          # + = extensión en este modelo
            # MuJoCo resta Kd·v por el amortiguamiento de la junta
            d.qfrc_applied[va] = u + kd * d.qvel[va]
            tau[nombre] = u
        for l, geoms in self.pies.items():
            en_apoyo = self.fase(t, l) < self.m["despegue"][l]
            self.model.geom_contype[geoms] = self.model.geom_conaffinity[geoms] = int(en_apoyo)
        if self.exo is not None:
            for l, a in self.exo.items():
                d.ctrl[a] = np.interp(self.fase(t, l), self.ref.pct, self.m["torque"], period=100)
        mujoco.mj_step(self.model, d)
        return tau, qr


def correr(corrida):
    s = Simulacion(corrida)
    s.iniciar()
    z = s.m["zancada"]
    reg = {k: [] for k in ("pct", "rod", "rod_ref", "tau", "vel", "exo", "arnes_y")}
    while s.data.time < CICLOS * z:
        tau, qr = s.paso()
        t = s.data.time
        if t < DESCARTE * z:
            continue
        qa, va = s.rodilla["r"]
        reg["pct"].append(s.fase(t, "r"))
        reg["rod"].append(-np.degrees(s.data.qpos[qa]))          # flexión positiva
        reg["rod_ref"].append(-np.degrees(qr[qa]))
        reg["tau"].append(tau["knee_angle_r"])                    # + = extensor
        reg["vel"].append(s.data.qvel[va])
        reg["exo"].append(s.data.sensor("exo_torque_r").data[0] if s.exo else 0.0)
        reg["arnes_y"].append(tau["pelvis_ty"])
    r = {k: np.array(v) for k, v in reg.items()}
    r["masa"] = s.masa
    r["despegue"] = s.m["despegue"]["r"]
    r["zancada"] = z
    r["ok"] = bool(np.all(np.isfinite(s.data.qpos)))
    return r


def por_ciclo(r, clave, bins=101):
    """Promedio por % del ciclo (0..100)."""
    idx = np.clip(np.round(r["pct"]).astype(int), 0, bins - 1)
    suma = np.bincount(idx, r[clave], minlength=bins)
    n = np.bincount(idx, minlength=bins)
    return suma / np.maximum(n, 1)


def metricas(r):
    M = r["masa"]
    apoyo = r["pct"] < r["despegue"]
    potencia = r["tau"] * r["vel"] / M                    # W/kg, + = genera
    dt = PASO
    ciclos = (len(r["pct"]) * dt) / r["zancada"]
    return {
        "error_rms": np.sqrt(np.mean((r["rod"] - r["rod_ref"]) ** 2)),
        "pico_ext": r["tau"][apoyo].max() / M,
        "media_ext_apoyo": r["tau"][apoyo].mean() / M,
        "pico_absorcion": -potencia[apoyo].min(),
        "trabajo_pos": potencia[potencia > 0].sum() * dt / ciclos,
        "trabajo_neg": potencia[potencia < 0].sum() * dt / ciclos,
        "arnes": r["arnes_y"].mean() / (M * 9.81) * 100,
        "rodilla_min_apoyo": r["rod"][apoyo].min(),
    }


def tabla(nombres, res):
    print(f"\n{'Corrida':<22}{'error rod.':>11}{'pico ext.':>11}{'media ext.':>12}{'absorción':>11}"
          f"{'W+':>8}{'W-':>8}{'rod. mín':>10}{'arnés':>8}")
    print(f"{'':<22}{'(°)':>11}{'(N·m/kg)':>11}{'apoyo':>12}{'(W/kg)':>11}{'(J/kg)':>8}{'(J/kg)':>8}"
          f"{'apoyo(°)':>10}{'(% peso)':>8}")
    print("-" * 101)
    for n in nombres:
        k = metricas(res[n])
        print(f"{TITULOS[n]:<22}{k['error_rms']:>11.2f}{k['pico_ext']:>11.3f}{k['media_ext_apoyo']:>12.3f}"
              f"{k['pico_absorcion']:>11.2f}{k['trabajo_pos']:>8.3f}{k['trabajo_neg']:>8.3f}"
              f"{k['rodilla_min_apoyo']:>10.1f}{k['arnes']:>8.1f}" + ("" if res[n]["ok"] else "  ¡NaN!"))


def graficar(nombres, res, titulo, archivo):
    colores = ["#86b6ef", "#2a78d6", "#104281"]
    tinta, suave = "#3d3d3a", "#73726c"
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4))
    pct = np.arange(101)
    for n, c in zip(nombres, colores):
        r = res[n]
        M = r["masa"]
        ejes[0].plot(pct, por_ciclo(r, "rod"), color=c, lw=2, label=TITULOS[n])
        ejes[1].plot(pct, por_ciclo(r, "tau") / M, color=c, lw=2, label=TITULOS[n])
        r["pot"] = r["tau"] * r["vel"] / M
        ejes[2].plot(pct, por_ciclo(r, "pot"), color=c, lw=2, label=TITULOS[n])
    ejes[0].plot(pct, por_ciclo(res[nombres[0]], "rod_ref"), color=suave, lw=1.2, ls="--", label="referencia")
    exo = next((res[n] for n in nombres if np.any(res[n]["exo"])), None)
    if exo is not None:
        ejes[1].plot(pct, por_ciclo(exo, "exo") / exo["masa"], color="#eb6834", lw=1.5, ls="--", label="torque del exo")
    for ax, t, y in zip(ejes, ("Ángulo de rodilla", "Torque biológico de rodilla", "Potencia biológica de rodilla"),
                        ("Flexión (°)", "Torque (N·m/kg), + = extensor", "Potencia (W/kg), + = genera")):
        ax.axvline(res[nombres[0]]["despegue"], color="#c3c2b7", lw=1)
        ax.axhline(0, color="#e6e5e0", lw=0.8)
        ax.set_title(t, loc="left", color=tinta, fontsize=11)
        ax.set_xlabel("Ciclo de marcha (%)", color=suave)
        ax.set_ylabel(y, color=suave)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=suave)
    ejes[1].legend(frameon=False, fontsize=8)
    ejes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(titulo, color=tinta, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    RESULTADOS.mkdir(exist_ok=True)
    fig.savefig(RESULTADOS / archivo, dpi=150)
    print(f"Gráfica guardada en {RESULTADOS / archivo}")


def ver(corrida):
    import mujoco.viewer
    s = Simulacion(corrida)
    s.iniciar()
    print(f"{TITULOS[corrida]}: simulación dinámica en tiempo real. Cierra la ventana para salir.")
    with mujoco.viewer.launch_passive(s.model, s.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = s.model.body("pelvis").id
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, 90.0, -10.0
        inicio = time.time()
        while v.is_running():
            while s.data.time < time.time() - inicio:
                tau, _ = s.paso()
            v.sync()
            qa, _ = s.rodilla["r"]
            print(f"  t {s.data.time:5.2f} s  rodilla der {-np.degrees(s.data.qpos[qa]):5.1f}°  "
                  f"torque biológico {tau['knee_angle_r']:6.2f} N·m"
                  + (f"  exo {s.data.sensor('exo_torque_r').data[0]:4.2f} N·m" if s.exo else ""), end="\r")
            time.sleep(1 / 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", choices=list(CORRIDAS), default=None)
    args = parser.parse_args()
    if args.ver:
        ver(args.ver)
        return

    res = {}
    for n in CORRIDAS:
        print(f"simulando {TITULOS[n]}...", flush=True)
        res[n] = correr(n)

    print("\nExperimento 1: misma marcha de P4 con y sin exoesqueleto (pierna derecha)")
    tabla(["pc", "pc_exo_apagado", "pc_exo"], res)
    print("\nExperimento 2: niño débil (extensor de rodilla limitado a "
          f"{FUERZA_EXTENSORA_P4} N·m/kg) siguiendo la marcha del sano")
    tabla(["sano", "debil", "debil_exo"], res)
    graficar(["pc", "pc_exo_apagado", "pc_exo"], res, "Experimento 1: marcha de P4 con y sin exoesqueleto",
             "old_dinamica_6y_exp1.png")
    graficar(["sano", "debil", "debil_exo"], res, "Experimento 2: niño débil siguiendo la marcha del sano",
             "old_dinamica_6y_exp2.png")
    plt.show()


if __name__ == "__main__":
    main()

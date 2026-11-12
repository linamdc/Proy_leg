"""
Paso 2 de la simulación con músculos: SIMULACIÓN HACIA ADELANTE. Los músculos
mueven el modelo y el exoesqueleto SÍ cambia la marcha.

  1. Orden del niño: las activaciones de la optimización estática
     (musculos_6y.py) para el niño C8 con el cuádriceps débil de P4, llevando
     el exo apagado (con su masa). Es su marcha habitual con el dispositivo.
  2. Simulación: los 68 músculos reciben esa orden (adelantada lo que tarda la
     activación). Además:
       - arnés: sostiene la pelvis en avance, de lado y en rotación (para que
         no se caiga), pero deja LIBRE su altura: el peso lo cargan las piernas;
       - corrección suave en cadera, rodilla y tobillo (como un reflejo), para
         que la marcha no se desvíe. Es blanda a propósito: con ella la postura
         del apoyo (cuánto se agacha) la deciden los músculos y el exo, no un
         controlador rígido;
       - contacto de cada pie con el suelo solo en su fase de apoyo.
  3. Experimento: la MISMA orden muscular con el exo apagado y con el exo
     encendido, con dos perfiles: el de Lerner (casi constante en todo el
     apoyo) y el propuesto (50 % de la demanda extensora del apoyo). La
     diferencia en la rodilla la predice la simulación, no se impone.

Resultado (ver la tabla): el perfil de Lerner extiende la rodilla en el apoyo
en todas las rigideces estables, más cuanto más blando es el control del
niño (de ~1° con control rígido a ~11° a mitad del apoyo con control blando,
cerca de los +12° medidos por Lerner); el propuesto baja el esfuerzo pero casi
no cambia la postura. Por debajo de reflejo x0,8 la marcha se vuelve
inestable. La orden muscular de la optimización estática no basta sola para
caminar estable (sin reflejo de rodilla la marcha se desvía 14–22°).

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):
  python scripts/marcha_musculos_6y.py                      # experimento + gráfica
  python scripts/marcha_musculos_6y.py --ver encendido      # ver la simulación (apagado | encendido)
  python scripts/comparar_marcha.py --musculos --gif        # medido | simulado sin exo | simulado con exo
"""

import argparse
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import dinamica_inversa_6y as di
import marcha_6y as m6
import musculos_6y as mu

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"

PASO = 0.001            # s
CICLOS = 4
DESCARTE = 1            # ciclos de transitorio que no se miden
ADELANTO = 0.015        # s: la orden se adelanta lo que tarda la activación del músculo

ARNES = {"pelvis_tx": (20000, 800), "pelvis_tz": (20000, 800),
         "pelvis_rotation": (800, 30), "pelvis_tilt": (800, 30), "pelvis_list": (800, 30),
         "lumbar_extension": (300, 10), "lumbar_bending": (300, 10), "lumbar_rotation": (300, 10)}
# corrección suave (N·m/rad, N·m·s/rad) por articulación, a multiplicar por la rigidez
REFLEJO = {"hip_flexion": (60, 3), "hip_adduction": (120, 4), "hip_rotation": (120, 4),
           "knee_angle": (40, 2), "ankle_angle": (40, 2)}
RIGIDECES = (0.8, 1.25, 2.0)    # blando (estable más bajo), el que mejor sigue la marcha medida, rígido
# Contacto de cada pie con el suelo solo en su fase de apoyo. Con el contacto
# siempre activo, la pierna que oscila arrastra el pie y tropieza (la marcha se
# desvía 10–34° y los resultados dejan de ser estables).
CONTACTO_SIEMPRE = False
# El contacto se activa también desde CONTACTO_DESDE (% del ciclo de esa
# pierna), al final de la oscilación: el pie baja para pisar y, si llega antes
# que en la marcha medida, pisa en vez de atravesar el suelo (sin esto se
# hundía hasta 16 mm entre el 75 % y el 100 %). El roce que hace tropezar está
# en la oscilación temprana y media, que sigue sin contacto.
CONTACTO_DESDE = 100.0   # 100 = desactivado: con 75 % el pie aterriza antes y la marcha se cae
# LIMITACIÓN conocida: al final de la oscilación (75–100 %) el pie que oscila
# baja 1–2 cm por debajo del suelo, porque en la simulación la pelvis queda
# ~15 mm más baja que en la marcha medida y ese pie no tiene contacto (si lo
# tiene, roza y tropieza). No afecta el apoyo, que es lo que se mide. Se
# probó y se descartó: contacto siempre activo, contacto desde el 75 %, sostén
# vertical del arnés (6–22 % del peso) y doblar más rodilla y tobillo en la
# oscilación: o no lo arreglaban o desestabilizaban la marcha.
# En la oscilación la pierna no carga peso y el niño guía el pie con
# precisión: su corrección se multiplica por FACTOR_OSCILACION, para que el pie
# pase por encima del suelo en vez de atravesarlo. La postura del apoyo, que es
# lo que se mide, sigue con el control blando.
FACTOR_OSCILACION = 4.0
# Contacto del pie más rígido que el de MuJoCo por defecto (0,02 s): con el
# peso del niño sobre las esferas de los pies, el pie se hundía hasta 29 mm.
SOLREF_PIE = (0.004, 1.0)
RIGIDEZ_VER = 0.8
PERFIL_VER = "exo correctivo"


@lru_cache(maxsize=None)
def preparar(dispositivo=True):
    """Orden muscular (activaciones) y perfiles del exo del niño débil: con el
    exo puesto (apagado, con su masa) o sin el dispositivo."""
    inv = di.Inversa("crouch_exo" if dispositivo else "crouch")
    factor = mu.FUERZA_EXTENSORA_PC * inv.masa_nino / mu.torque_extensor_max(di.Inversa("crouch").model)
    mu.debilitar(inv.model, factor)
    datos = mu.torques_necesarios(inv)
    act, _ = mu.optimizar(datos, np.zeros((len(datos), 2)))
    perfiles = mu.perfiles_exo(datos, inv)                          # N·m por rodilla (r, l)
    ids = mu.musculos(inv.model)
    nombres = [inv.model.actuator(i).name for i in ids]
    # referencia continua: tabla de un ciclo + avance por ciclo
    ref = inv.ref
    tendencia = np.zeros(inv.model.nq)
    tendencia[ref.tx] = ref.por_ciclo
    tendencia[ref.ty] = -ref.por_ciclo * ref.eje_x[2] / ref.eje_y[2]
    return dict(act=act, perfiles=perfiles, nombres=nombres, factor=factor, q=inv.q.copy(), v=inv.v.copy(),
                tendencia=tendencia, zancada=inv.m["zancada"], despegue=dict(inv.m["despegue"]),
                desfase=inv.m["desfase"], masa=inv.masa_nino)


class Simulacion:
    def __init__(self, exo_encendido, rigidez=1.0, perfil="exo propuesto", dispositivo=True):
        """dispositivo=False: el niño sin el exo (ni su masa), con su propia
        orden muscular; exo_encendido no aplica."""
        P = preparar(dispositivo)
        self.P = P
        self.perfil = P["perfiles"][perfil]
        self.model, self.data = m6.cargar_modelo(dispositivo)
        exo_encendido = exo_encendido and dispositivo
        model = self.model
        mu.debilitar(model, P["factor"])
        model.opt.timestep = PASO
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.encendido = exo_encendido
        self.musc = np.array([model.actuator(n).id for n in P["nombres"]])
        self.actadr = model.actuator_actadr[self.musc]          # posición de su activación en data.act
        self.exo = [model.actuator(f"exo_rodilla_{l}").id for l in m6.LADOS] if dispositivo else None
        self.pd = []
        for j in range(model.njnt):
            n = model.joint(j).name
            g = ARNES.get(n)
            if g is None:
                base = next((v for k, v in REFLEJO.items() if n.startswith(k)), None)
                g = None if base is None else (base[0] * rigidez, base[1] * rigidez)
            if g is not None:
                lado = n[-1] if n.endswith(("_r", "_l")) and n not in ARNES else None
                self.pd.append((n, model.jnt_qposadr[j], model.jnt_dofadr[j], *g, lado))
                model.dof_damping[model.jnt_dofadr[j]] += g[1]       # amortiguamiento implícito (estable)
        self.amort = model.dof_damping.copy()
        self.ty = model.jnt_qposadr[model.joint("pelvis_ty").id]
        self.pies = {l: [model.geom(f"contacto_{n}_{l}").id for n in m6.CONTACTOS] for l in m6.LADOS}
        for l in m6.LADOS:
            model.geom_solref[self.pies[l]] = SOLREF_PIE
        self.n = m6.MUESTRAS

    def fase(self, t):
        return (t / self.P["zancada"] % 1) * 100

    def interp(self, tabla, t):
        """Valor de una tabla de un ciclo (MUESTRAS x ...) en el tiempo t."""
        f = (t / self.P["zancada"] % 1) * self.n
        i0 = int(f) % self.n
        w = f - int(f)
        return (1 - w) * tabla[i0] + w * tabla[(i0 + 1) % self.n]

    def q_ref(self, t):
        k = int(t / self.P["zancada"])
        f = (t / self.P["zancada"] % 1) * self.n
        i0, w = int(f) % self.n, f - int(f)
        a, b = self.P["q"][i0], self.P["q"][(i0 + 1) % self.n].copy()
        if i0 + 1 == self.n:
            b = b + self.P["tendencia"]
        return (1 - w) * a + w * b + k * self.P["tendencia"]

    def iniciar(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.q_ref(0.0)
        self.data.qvel[:] = self.interp(self.P["v"], 0.0)
        self.data.act[:] = 0
        self.data.act[self.actadr] = self.interp(self.P["act"], 0.0)
        mujoco.mj_forward(self.model, self.data)

    def paso(self):
        d, t = self.data, self.data.time
        qr, vr = self.q_ref(t), self.interp(self.P["v"], t)
        d.qfrc_applied[:] = 0
        p = self.fase(t)
        oscila = {l: (p if l == "r" else (p - self.P["desfase"]) % 100) >= self.P["despegue"][l]
                  for l in m6.LADOS}
        for n, qa, va, kp, kd, lado in self.pd:
            f = FACTOR_OSCILACION if lado is not None and oscila[lado] else 1.0
            # -kd·v lo pone el amortiguamiento de la junta (implícito)
            self.model.dof_damping[va] = self.amort[va] + (f - 1) * kd
            d.qfrc_applied[va] = f * (kp * (qr[qa] - d.qpos[qa]) + kd * vr[va])
        d.ctrl[self.musc] = np.clip(self.interp(self.P["act"], t + ADELANTO), 0, 1)
        for k, l in enumerate(m6.LADOS):
            fase_l = p if l == "r" else (p - self.P["desfase"]) % 100
            apoyo = fase_l < self.P["despegue"][l]
            if not CONTACTO_SIEMPRE:
                activo = apoyo or fase_l >= CONTACTO_DESDE
                self.model.geom_contype[self.pies[l]] = self.model.geom_conaffinity[self.pies[l]] = int(activo)
            if self.exo is not None:
                d.ctrl[self.exo[k]] = self.interp(self.perfil[:, k], t) if self.encendido else 0.0
        mujoco.mj_step(self.model, d)
        return qr


def correr(exo_encendido, rigidez, perfil="exo propuesto", dispositivo=True):
    s = Simulacion(exo_encendido, rigidez, perfil, dispositivo)
    s.iniciar()
    z = s.P["zancada"]
    m = s.model
    rq = m.jnt_qposadr[m.joint("knee_angle_r").id]
    hq = m.jnt_qposadr[m.joint("hip_flexion_r").id]
    aq = m.jnt_qposadr[m.joint("ankle_angle_r").id]
    vasto = m.actuator("vas_int_r").id
    reg = {k: [] for k in ("pct", "rod", "rod_ref", "cadera", "tobillo", "altura", "arnes", "exo", "act_vasto")}
    ok = True
    while s.data.time < CICLOS * z:
        qr = s.paso()
        if not np.all(np.isfinite(s.data.qpos)) or np.abs(s.data.qvel).max() > 200:
            ok = False
            break
        if s.data.time < DESCARTE * z:
            continue
        reg["pct"].append(s.fase(s.data.time))
        reg["rod"].append(-np.degrees(s.data.qpos[rq]))
        reg["rod_ref"].append(-np.degrees(qr[rq]))
        reg["cadera"].append(np.degrees(s.data.qpos[hq]))
        reg["tobillo"].append(np.degrees(s.data.qpos[aq]))
        reg["altura"].append(s.data.qpos[s.ty] - qr[s.ty])
        reg["exo"].append(s.data.sensor("exo_torque_r").data[0] if s.exo is not None else 0.0)
        reg["act_vasto"].append(s.data.act[m.actuator_actadr[vasto]])
        # fuerza vertical del arnés: no hay (altura libre); se mide la del suelo
        fz = 0.0
        for i in range(s.data.ncon):
            f = np.zeros(6)
            mujoco.mj_contactForce(m, s.data, i, f)
            fz += f[0]
        reg["arnes"].append(fz / (m.body_subtreemass[0] * 9.81))
    r = {k: np.array(v) for k, v in reg.items()}
    r["ok"] = ok
    r["despegue"] = s.P["despegue"]["r"]
    return r


def trayectoria(exo_encendido, rigidez=RIGIDEZ_VER, perfil=PERFIL_VER, ciclos=4, fps=25, dispositivo=True):
    """qpos de la simulación muestreado a fps (para mostrarla), sin el primer ciclo."""
    s = Simulacion(exo_encendido, rigidez, perfil, dispositivo)
    s.iniciar()
    z = s.P["zancada"]
    t_muestras = np.arange(z, ciclos * z, 1 / fps)
    qs, k = [], 0
    while k < len(t_muestras):
        s.paso()
        if not np.all(np.isfinite(s.data.qpos)):
            raise RuntimeError("simulación inestable")
        while k < len(t_muestras) and s.data.time >= t_muestras[k]:
            qs.append(s.data.qpos.copy())
            k += 1
    return t_muestras - z, np.array(qs), s.model


def por_ciclo(r, clave):
    idx = np.clip(np.round(r["pct"]).astype(int), 0, 100)
    return np.bincount(idx, r[clave], minlength=101) / np.maximum(np.bincount(idx, minlength=101), 1)


def resumen(r):
    ap = r["pct"] < r["despegue"]
    medio = (r["pct"] > 15) & (r["pct"] < 50)
    return dict(rod_apoyo=r["rod"][ap].mean(), rod_medio=r["rod"][medio].mean(), rod_ref_medio=r["rod_ref"][medio].mean(),
                error=np.sqrt(np.mean((r["rod"] - r["rod_ref"]) ** 2)), suelo=r["arnes"].mean(),
                altura=r["altura"][ap].mean() * 1000, vasto=r["act_vasto"][ap].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", choices=["apagado", "encendido"], default=None)
    args = parser.parse_args()
    if args.ver:
        ver(args.ver == "encendido")
        return

    P = preparar()
    print(f"Orden muscular lista (niño débil, cuádriceps x {P['factor']:.2f}, exo puesto).")
    res = {}
    casos = (("sin dispositivo", False, "exo propuesto"), ("apagado", False, "exo propuesto"), ("exo Lerner", True, "exo Lerner"),
             ("exo propuesto", True, "exo propuesto"), ("exo correctivo", True, "exo correctivo"))
    for rig in RIGIDECES:
        for nombre, enc, perfil in casos:
            res[(rig, nombre)] = correr(enc, rig, perfil, dispositivo=(nombre != "sin dispositivo"))
            print(f"  reflejo x{rig}, {nombre}: {'ok' if res[(rig, nombre)]['ok'] else 'INESTABLE'}")

    print(f"\nRODILLA DERECHA, SIMULACIÓN CON MÚSCULOS ({m6.NINO_CROUCH}, cuádriceps débil, exo puesto)")
    print(f"{'reflejo':<10}{'exo':<15}{'flexión media':>14}{'flexión a mitad':>16}{'extensión por':>15}"
          f"{'error vs':>10}{'peso en':>9}{'vasto':>8}")
    print(f"{'':<10}{'':<15}{'apoyo (°)':>14}{'del apoyo (°)':>16}{'el exo (°)':>15}{'medida (°)':>10}{'suelo':>9}{'(act.)':>8}")
    print("-" * 97)
    for rig in RIGIDECES:
        base = res[(rig, "apagado")]
        kb = resumen(base) if base["ok"] and len(base["pct"]) else None
        for nombre, _, _ in casos:
            r = res[(rig, nombre)]
            if not r["ok"] or len(r["pct"]) == 0 or kb is None:
                print(f"x{rig:<9}{nombre:<15}  inestable")
                continue
            k = resumen(r)
            ext = "" if nombre in ("apagado", "sin dispositivo") else f"{kb['rod_medio'] - k['rod_medio']:+.1f}"
            print(f"x{rig:<9}{nombre:<15}{k['rod_apoyo']:>14.1f}{k['rod_medio']:>16.1f}{ext:>15}"
                  f"{k['error']:>10.1f}{k['suelo']:>9.2f}{k['vasto']:>8.2f}")
    print("  extensión por el exo: a mitad del apoyo (15–50 %), + = la rodilla se extiende. Lerner midió +12°.")
    graficar(res)


def graficar(res):
    tinta, suave = "#3d3d3a", "#73726c"
    fig, ejes = plt.subplots(1, len(RIGIDECES), figsize=(5.5 * len(RIGIDECES), 4.4), sharey=True)
    titulos = {RIGIDECES[0]: "control blando", RIGIDECES[1]: "control intermedio", RIGIDECES[-1]: "control rígido"}
    for ax, rig in zip(np.atleast_1d(ejes), RIGIDECES):
        a = res[(rig, "apagado")]
        ax.axvspan(0, a["despegue"], color="#f1efe8", zorder=0)
        if len(a["pct"]):
            ax.plot(np.arange(101), por_ciclo(a, "rod_ref"), color=suave, ls=":", lw=1.5, label="marcha medida (C8)")
        for nombre, col in (("apagado", "#2a78d6"), ("exo propuesto", "#1a9e5c"), ("exo Lerner", "#eb6834"),
                            ("exo correctivo", "#7b3fbf")):
            r = res[(rig, nombre)]
            if r["ok"] and len(r["pct"]):
                ax.plot(np.arange(101), por_ciclo(r, "rod"), color=col, lw=2.2,
                        label="simulada, exo apagado" if nombre == "apagado" else f"simulada, {nombre}")
        ax.set_title(f"Reflejo x{rig} ({titulos.get(rig, '')})", loc="left", color=tinta)
        ax.set_xlabel("Ciclo de marcha (%)", color=suave)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    np.atleast_1d(ejes)[0].set_ylabel("Flexión de rodilla derecha (°)", color=suave)
    np.atleast_1d(ejes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Paso 2: con los mismos músculos, el exo cambia la rodilla (simulación hacia adelante)",
                 color=tinta, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    RESULTADOS.mkdir(exist_ok=True)
    fig.savefig(RESULTADOS / "marcha_musculos_6y.png", dpi=150)
    print(f"\nGráfica guardada en {RESULTADOS / 'marcha_musculos_6y.png'}")


def ver(encendido):
    import mujoco.viewer
    s = Simulacion(encendido, RIGIDEZ_VER, PERFIL_VER)
    s.iniciar()
    rq = s.model.jnt_qposadr[s.model.joint("knee_angle_r").id]
    with mujoco.viewer.launch_passive(s.model, s.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = s.model.body("pelvis").id
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.0, 90.0, -10.0
        inicio = time.time()
        while v.is_running():
            while s.data.time < time.time() - inicio:
                qr = s.paso()
            v.sync()
            print(f"  t {s.data.time:5.2f} s  rodilla der {-np.degrees(s.data.qpos[rq]):5.1f}° "
                  f"(medida {-np.degrees(qr[rq]):5.1f}°)  exo {s.data.sensor('exo_torque_r').data[0]:4.2f} N·m", end="\r")
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()

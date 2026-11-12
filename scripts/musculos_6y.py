"""
Paso 1 de la simulación con músculos: OPTIMIZACIÓN ESTÁTICA sobre la marcha
medida del niño C8 (Steele 2010) en el modelo de 6 años.

En cada instante del ciclo se buscan las activaciones de los 68 músculos que
producen los torques de la dinámica inversa (dinamica_inversa_6y.py) en las 10
coordenadas de las piernas (cadera: flexión, aducción, rotación; rodilla;
tobillo; en las dos piernas), con el menor esfuerzo: mínima suma de
activaciones al cuadrado (el criterio clásico de OpenSim). Lo que los músculos
no alcanzan lo cubre un torque de RESERVA por coordenada, penalizado: es el
torque que le falta al niño.

  - Fuerza de cada músculo: el modelo de MuJoCo (fuerza-longitud-velocidad y
    fuerza pasiva) en la pose y velocidad de ese instante: F = ganancia·a + pasiva.
  - Brazo de momento sobre cada coordenada: por trabajo virtual, sumando las
    coordenadas acopladas (traslaciones de la tibia, puntos móviles de los
    músculos), igual que el torque neto.
  - Exoesqueleto: su torque se resta del que necesita la rodilla (biológico =
    neto - exo), con la masa del exo incluida en la dinámica inversa.

Niños:
  sano de fuerza: los músculos del modelo tal como vienen (Schmid, 6 años).
  PC (débil): el cuádriceps (rect_fem, vas_int) reducido para que el torque
    extensor isométrico máximo de rodilla sea 0,69 N·m/kg, la fuerza de P4
    (Lerner Tabla 1). SUPUESTO: C8 tuviera la fuerza de P4 (no se midió).

Corridas: sin exo, exo con el perfil de Lerner, exo con el perfil propuesto
(50 % de la demanda extensora del apoyo) y exo con el perfil CORRECTIVO (el
propuesto con un piso de NIVEL_APOYO en todo el apoyo y la rampa de la
oscilación tardía; es el que se recomienda).

Salidas (resultados/):
  musculos_6y.png         activación de los grupos musculares en el ciclo
  activaciones_6y.npz     activaciones de cada músculo (para el paso 2)

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):
  python scripts/musculos_6y.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

import dinamica_inversa_6y as di
import marcha_6y as m6

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"

LADOS = ("r", "l")
COORDENADAS = [f"{c}_{l}" for l in LADOS for c in
               ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle")]
CUADRICEPS = ["rect_fem", "vas_int"]
GRUPOS = {   # para las gráficas (pierna derecha)
    "cuádriceps (vasto)": ["vas_int"],
    "recto femoral": ["rect_fem"],
    "isquiotibiales": ["bifemlh", "bifemsh"],
    "gastrocnemio": ["med_gas"],
    "sóleo": ["soleus"],
    "tibial anterior": ["tib_ant"],
    "glúteo mayor": ["glut_max1", "glut_max2", "glut_max3"],
}
FUERZA_EXTENSORA_PC = 0.69     # N·m/kg (P4, Lerner Tabla 1)
FRACCION_ASISTENCIA = 0.5      # perfil propuesto: fracción de la demanda extensora del apoyo
TORQUE_MAX = 6.0               # N·m
# Perfil CORRECTIVO (el que se recomienda): en el apoyo sigue la demanda como
# el propuesto, pero nunca baja de NIVEL_APOYO, para seguir extendiendo la
# rodilla en todo el apoyo (como Lerner, 0,15–0,18 N·m/kg); libre en la
# oscilación temprana para que la rodilla se doble; en la oscilación tardía
# (desde OSC_TARDIA) una rampa hasta NIVEL_FINAL para llegar al contacto más
# extendido (sección 1 del documento: rampa hasta 0,05 N·m/kg). Tope: el torque
# pico nominal del requerimiento.
NIVEL_APOYO = 0.15             # N·m/kg
NIVEL_FINAL = 0.05             # N·m/kg
OSC_TARDIA = 80.0              # % del ciclo
TORQUE_NOMINAL = 4.0           # N·m
PESO_IGUALDAD = 1e4            # los torques deben cumplirse (casi exacto)
PESO_RESERVA = 1.0             # costo por (N·m)² de reserva frente a 1 por activación²


# ----------------------------------------------------------------------------
# Modelo: músculos, acoples y fuerza del cuádriceps
# ----------------------------------------------------------------------------

def musculos(model):
    return [i for i in range(model.nu) if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE]


def matriz_acoples(model, qpos):
    """C (nv x coordenadas): ∂q_j/∂q_k de cada grado de libertad respecto a
    cada coordenada independiente de las piernas (1 en la propia coordenada,
    la derivada del polinomio en las acopladas)."""
    C = np.zeros((model.nv, len(COORDENADAS)))
    col = {}
    for k, nombre in enumerate(COORDENADAS):
        j = model.joint(nombre).id
        C[model.jnt_dofadr[j], k] = 1.0
        col[j] = k
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        k = col.get(model.eq_obj2id[i])
        if k is None:
            continue
        j1, j2 = model.eq_obj1id[i], model.eq_obj2id[i]
        x = qpos[model.jnt_qposadr[j2]] - model.qpos0[model.jnt_qposadr[j2]]
        C[model.jnt_dofadr[j1], k] = np.polyval(np.polyder(model.eq_data[i][:5][::-1]), x)
    return C


def momentos(model, data):
    """Matriz densa de brazos de momento (nu x nv): dL/dq."""
    R = np.zeros((model.nu, model.nv))
    for a in range(model.nu):
        ini, n = data.moment_rowadr[a], data.moment_rownnz[a]
        R[a, data.moment_colind[ini:ini + n]] = data.actuator_moment[ini:ini + n]
    return R


def ganancia_y_pasiva(model, data, ids):
    """Fuerza activa por unidad de activación y fuerza pasiva de cada músculo
    (negativas: el músculo tira)."""
    g, b = np.zeros(len(ids)), np.zeros(len(ids))
    for n, a in enumerate(ids):
        L, V = data.actuator_length[a], data.actuator_velocity[a]
        lr, acc0 = model.actuator_lengthrange[a], model.actuator_acc0[a]
        g[n] = mujoco.mju_muscleGain(L, V, lr, acc0, model.actuator_gainprm[a][:9])
        b[n] = mujoco.mju_muscleBias(L, lr, acc0, model.actuator_biasprm[a][:9])
    return g, b


def torque_extensor_max(model, lado="r"):
    """Torque extensor isométrico máximo de rodilla del cuádriceps (N·m),
    sentado (cadera a 90°), barrido de 0 a 90° de flexión."""
    data = mujoco.MjData(model)
    ids = [model.actuator(f"{m}_{lado}").id for m in CUADRICEPS]
    rod = model.joint(f"knee_angle_{lado}").id
    maximo = 0.0
    for flex in np.arange(0, 91, 2.0):
        data.qpos[:] = model.qpos0
        data.qpos[model.jnt_qposadr[model.joint(f"hip_flexion_{lado}").id]] = np.radians(90)
        data.qpos[model.jnt_qposadr[rod]] = -np.radians(flex)          # flexión negativa en el modelo
        for _ in range(2):
            for i in range(model.neq):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                    a_, b_ = model.jnt_qposadr[model.eq_obj1id[i]], model.jnt_qposadr[model.eq_obj2id[i]]
                    data.qpos[a_] = model.qpos0[a_] + np.polyval(model.eq_data[i][:5][::-1],
                                                                  data.qpos[b_] - model.qpos0[b_])
        mujoco.mj_forward(model, data)
        C = matriz_acoples(model, data.qpos)
        k = COORDENADAS.index(f"knee_angle_{lado}")
        brazo = momentos(model, data)[ids] @ C[:, k]
        g, _ = ganancia_y_pasiva(model, data, ids)
        maximo = max(maximo, float(brazo @ g))          # + = extensión en el modelo
    return maximo


def debilitar(model, factor):
    for m in CUADRICEPS:
        for l in LADOS:
            a = model.actuator(f"{m}_{l}").id
            model.actuator_gainprm[a][2] *= factor
            model.actuator_biasprm[a][2] *= factor


# ----------------------------------------------------------------------------
# Optimización estática
# ----------------------------------------------------------------------------

def torques_necesarios(inv):
    """Para cada instante: torque neto en las coordenadas de las piernas,
    brazos de momento efectivos de los músculos y su fuerza (ganancia, pasiva)."""
    model, d, ref = inv.model, inv.data, inv.ref
    ids = musculos(model)
    salida = []
    for i in range(m6.MUESTRAS):
        d.qpos[:], d.qvel[:], d.qacc[:] = inv.q[i], inv.v[i], inv.a[i]
        mujoco.mj_inverse(model, d)
        externa = np.zeros(model.nv)
        for l in m6.LADOS:
            medida = ref.fuerza_suelo(inv.p[i], l)
            if medida is not None:
                cop, F = medida
                mujoco.mj_applyFT(model, d, F, np.zeros(3), cop, inv.calcn[l], externa)
        C = matriz_acoples(model, d.qpos)
        tau = (d.qfrc_inverse - externa) @ C
        brazos = momentos(model, d)[ids] @ C               # músculos x coordenadas
        g, b = ganancia_y_pasiva(model, d, ids)
        salida.append((tau, brazos, g, b))
    return salida


def optimizar(datos, exo):
    """Activaciones (instantes x músculos) y reservas (instantes x coordenadas).
    exo: torque extensor del exo en cada rodilla (instantes x 2: r, l)."""
    n_m = len(datos[0][2])
    n_c = len(COORDENADAS)
    k_rod = [COORDENADAS.index(f"knee_angle_{l}") for l in LADOS]
    act, res = [], []
    for i, (tau, brazos, g, b) in enumerate(datos):
        objetivo = tau.copy()
        objetivo[k_rod] -= exo[i]                        # el exo pone su parte
        objetivo -= brazos.T @ b                          # la fuerza pasiva ya está
        A = brazos.T * g                                  # coordenadas x músculos
        sw, sr = np.sqrt(PESO_IGUALDAD), np.sqrt(PESO_RESERVA)
        M = np.block([[sw * A, sw * np.eye(n_c)],
                      [np.eye(n_m), np.zeros((n_m, n_c))],
                      [np.zeros((n_c, n_m)), sr * np.eye(n_c)]])
        y = np.concatenate([sw * objetivo, np.zeros(n_m), np.zeros(n_c)])
        lo = np.concatenate([np.zeros(n_m), np.full(n_c, -np.inf)])
        hi = np.concatenate([np.ones(n_m), np.full(n_c, np.inf)])
        sol = lsq_linear(M, y, bounds=(lo, hi), method="bvls", lsmr_tol="auto")
        act.append(sol.x[:n_m])
        res.append(sol.x[n_m:])
    return np.array(act), np.array(res)


def perfiles_exo(datos, inv_apagado):
    """Torque del exo en cada rodilla (instantes x 2) para cada perfil."""
    ref = inv_apagado.ref
    p = inv_apagado.p
    k_rod = [COORDENADAS.index(f"knee_angle_{l}") for l in LADOS]
    neto = np.array([[t[0][k] for k in k_rod] for t in datos])     # necesita la rodilla, con la masa del exo
    fase = np.array([[ref.fase_pierna(x, l) for l in LADOS] for x in p])
    despegue = np.array([inv_apagado.m["despegue"][l] for l in LADOS])
    lerner = (np.zeros_like(neto) if ref.torque is None else         # sin dispositivo no hay perfil
              np.array([[np.interp(f, ref.pct, ref.torque, period=100) for f in fila] for fila in fase]))
    M = inv_apagado.masa_nino
    return {"sin exo": np.zeros_like(neto), "exo Lerner": lerner,
            "exo propuesto": perfil_propuesto(neto, fase, despegue),
            "exo correctivo": perfil_correctivo(neto, fase, despegue, M)}


def perfil_propuesto(neto, fase, despegue):
    """Torque del exo (N·m): FRACCION_ASISTENCIA del momento extensor que
    necesita la rodilla en el apoyo; cero en la oscilación."""
    return np.clip(FRACCION_ASISTENCIA * np.maximum(neto, 0) * (fase < despegue), 0, TORQUE_MAX)


def perfil_correctivo(neto, fase, despegue, M):
    """Torque del exo (N·m) recomendado: en el apoyo, el propuesto con un piso
    de NIVEL_APOYO (sigue extendiendo la rodilla aunque la demanda baje), con
    tope TORQUE_NOMINAL; libre en la oscilación temprana; rampa hasta
    NIVEL_FINAL en la oscilación tardía."""
    apoyo = fase < despegue
    rampa = np.clip((fase - OSC_TARDIA) / (100 - OSC_TARDIA), 0, 1) * NIVEL_FINAL * M
    piso = np.maximum(perfil_propuesto(neto, fase, despegue), NIVEL_APOYO * M)
    return np.where(apoyo, np.clip(piso, 0, TORQUE_NOMINAL), rampa)


# ----------------------------------------------------------------------------
# Corridas y resultados
# ----------------------------------------------------------------------------

def correr():
    resultados = {}
    inv_sin = di.Inversa("crouch")
    inv_exo = di.Inversa("crouch_exo")         # modelo con la masa del exo (y el perfil de Lerner)
    M = inv_sin.masa_nino
    T_sano = torque_extensor_max(inv_sin.model)
    factor = FUERZA_EXTENSORA_PC * M / T_sano
    print(f"Cuádriceps del modelo de 6 años: extensor isométrico máximo {T_sano:.1f} N·m ({T_sano / M:.2f} N·m/kg)")
    print(f"PC: cuádriceps x {factor:.2f} para llegar a {FUERZA_EXTENSORA_PC} N·m/kg (P4)")

    for nino in ("sano de fuerza", "PC (débil)"):
        if nino.startswith("PC"):
            debilitar(inv_sin.model, factor)
            debilitar(inv_exo.model, factor)
        datos_sin = torques_necesarios(inv_sin)
        datos_exo = torques_necesarios(inv_exo)
        perfiles = perfiles_exo(datos_exo, inv_exo)
        for perfil, tau_exo in perfiles.items():
            datos = datos_sin if perfil == "sin exo" else datos_exo
            act, res = optimizar(datos, tau_exo)
            resultados[(nino, perfil)] = dict(act=act, res=res, exo=tau_exo)
            print(f"  {nino:<15} {perfil:<14} listo")
    ids = musculos(inv_sin.model)
    nombres = [inv_sin.model.actuator(i).name for i in ids]
    return resultados, nombres, inv_sin, M


def grupo(r, nombres, musc, lado="r"):
    idx = [nombres.index(f"{m}_{lado}") for m in musc]
    return r["act"][:, idx].mean(axis=1)


def tabla(resultados, nombres, inv, M):
    desp = inv.m["despegue"]["r"]
    p = inv.p
    ap = p < desp
    k_rod = COORDENADAS.index("knee_angle_r")
    print(f"\nACTIVACIÓN MUSCULAR, PIERNA DERECHA DE {m6.NINO_CROUCH} EN EL APOYO (0 a 1)")
    print(f"{'':<34}{'vasto':>14}{'recto fem.':>14}{'isquios':>10}{'sóleo':>8}{'reserva rodilla':>18}")
    print(f"{'':<34}{'media / pico':>14}{'media / pico':>14}{'media':>10}{'media':>8}{'pico (N·m/kg)':>18}")
    print("-" * 98)
    for (nino, perfil), r in resultados.items():
        v, rf = grupo(r, nombres, ["vas_int"]), grupo(r, nombres, ["rect_fem"])
        hq, so = grupo(r, nombres, ["bifemlh", "bifemsh"]), grupo(r, nombres, ["soleus"])
        reserva = np.abs(r["res"][ap, k_rod]).max() / M
        print(f"{nino + ', ' + perfil:<34}{v[ap].mean():>7.2f} /{v[ap].max():>5.2f}{rf[ap].mean():>8.2f} /{rf[ap].max():>5.2f}"
              f"{hq[ap].mean():>10.2f}{so[ap].mean():>8.2f}{reserva:>18.3f}")
    print("  reserva = torque que los músculos no alcanzan a producir (lo que le falta al niño)")


def graficar(resultados, nombres, inv):
    p, desp = inv.p, inv.m["despegue"]["r"]
    tinta, suave = "#3d3d3a", "#73726c"
    colores = {"sin exo": "#2a78d6", "exo Lerner": "#eb6834", "exo propuesto": "#1a9e5c",
               "exo correctivo": "#7b3fbf"}
    fig, ejes = plt.subplots(2, 3, figsize=(16, 8.4))
    paneles = [("PC (débil)", "cuádriceps (vasto)"), ("PC (débil)", "recto femoral"), ("PC (débil)", "isquiotibiales"),
               ("sano de fuerza", "cuádriceps (vasto)"), ("PC (débil)", "sóleo"), ("PC (débil)", "reserva")]
    for ax, (nino, g) in zip(ejes.flat, paneles):
        ax.axvspan(0, desp, color="#f1efe8", zorder=0)
        for perfil, col in colores.items():
            r = resultados[(nino, perfil)]
            if g == "reserva":
                ax.plot(p, r["res"][:, COORDENADAS.index("knee_angle_r")], color=col, lw=2, label=perfil)
            else:
                ax.plot(p, grupo(r, nombres, GRUPOS[g]), color=col, lw=2, label=perfil)
        titulo = (f"{g}, niño {nino}" if g != "reserva" else "torque de rodilla que falta (reserva), PC")
        ax.set_title(titulo, loc="left", color=tinta, fontsize=11)
        ax.set_xlabel("Ciclo de marcha (%)", color=suave)
        ax.set_ylabel("Activación (0–1)" if g != "reserva" else "N·m, + = extensor", color=suave)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=suave)
    ejes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Paso 1, optimización estática: activación muscular con y sin exo ({m6.NINO_CROUCH}, pierna derecha)",
                 color=tinta, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "musculos_6y.png", dpi=150)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    resultados, nombres, inv, M = correr()
    tabla(resultados, nombres, inv, M)
    graficar(resultados, nombres, inv)
    np.savez(RESULTADOS / "activaciones_6y.npz", nombres=np.array(nombres), pct=inv.p,
             **{f"{n}__{p}".replace(" ", "_").replace("(", "").replace(")", "").replace("é", "e"): r["act"]
                for (n, p), r in resultados.items()})
    print(f"\nGuardado: {RESULTADOS / 'musculos_6y.png'} y {RESULTADOS / 'activaciones_6y.npz'}")
    plt.show()


if __name__ == "__main__":
    main()

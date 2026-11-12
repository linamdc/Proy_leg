"""
Resultados del proyecto en cuatro figuras (resultados/), una por pregunta.
El perfil que se recomienda es el CORRECTIVO (musculos_6y.perfil_correctivo):
sigue la demanda de la rodilla al comienzo del apoyo, nunca baja de 0,15 N·m/kg
en el resto del apoyo (para extender la rodilla), es cero en la oscilación
temprana y sube a 0,05 N·m/kg en la oscilación tardía.

  1. EL PROBLEMA (fig1_problema.png): ¿en qué se diferencia la rodilla de un
     niño con marcha agachada de la de un niño sano?
       a) ángulo de rodilla en el ciclo: sano (Senden), P4 (Lerner) y C8 (Steele);
       b) momento neto de rodilla: sano y P4 medidos, C8 por dinámica inversa.

  2. EL ESFUERZO (fig2_solucion.png): ¿cuánto le quita el exoesqueleto a la
     rodilla de C8 con cada perfil de torque (con la misma marcha)?
       a) torque del exo: perfil de Lerner (casi constante en el apoyo), el que
          sigue la demanda (50 % del momento extensor) y el CORRECTIVO, sobre
          el momento neto;
       b) torque biológico de rodilla (neto - exo) sin exo y con cada perfil;
       c) potencia biológica de rodilla.
     Con la masa del exo incluida (1,0 kg por pierna).

  3. LOS SENSORES Y EL CONTROL (fig3_sensores.png): ¿detectan los sensores del
     dispositivo las fases a tiempo para aplicar ese perfil? Tres zancadas de
     C8 simuladas a 500 Hz:
       a) FSR de la plantilla (bajo la planta: talón, antepié y dedos; aquí su
          suma, que es la fuerza vertical medida de C8), con ruido, y el umbral
          de apoyo;
       b) encoder de la rodilla (14 bits, dentro del motor AK60-6): ángulo;
       c) velocidad de la rodilla derivada del encoder, con el umbral que
          detecta la oscilación tardía;
       d) fase detectada por la máquina de estados frente a la real, y el
          torque que manda el controlador (perfil correctivo, con 20 ms de
          latencia) frente al perfil ideal.

  4. LA CORRECCIÓN (fig4_postura.png): ¿extiende el exo la rodilla? Simulación
     con músculos (marcha_musculos_6y.py): la misma orden muscular con el exo
     apagado y encendido con cada perfil, para tres niveles de control del niño.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):

  python scripts/resultados_proyecto.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dinamica_inversa_6y as di
import marcha_6y as m6
import musculos_6y as mu

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
G = 9.81

# Los perfiles del exo (propuesto y correctivo) y sus parámetros están en
# musculos_6y.py (perfil_propuesto, perfil_correctivo).

# Sensores y control (sección 4 y 5 del documento)
FRECUENCIA = 500                 # Hz, controlador
LATENCIA = 0.020                 # s, del evento al cambio de torque (requerimiento ≤ 20 ms)
RUIDO_FSR = 0.02                 # pesos corporales (desvío estándar)
UMBRAL_APOYO = (0.10, 0.05)      # pesos corporales: entra / sale (histéresis)
BITS_ENCODER = 14
UMBRAL_EXTENSION = 150.0         # °/s de extensión que marcan la oscilación tardía
ZANCADAS = 3

TINTA, SUAVE, REJILLA = "#3d3d3a", "#73726c", "#e6e5e0"
C_SANO, C_P4, C_C8 = "#8a8983", "#eb6834", "#2a78d6"
C_SIN, C_LERNER, C_PROP, C_CORR = "#2a78d6", "#eb6834", "#1a9e5c", "#7b3fbf"


def estilo(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, loc="left", color=TINTA, fontsize=11)
    ax.set_xlabel(xlabel, color=SUAVE)
    ax.set_ylabel(ylabel, color=SUAVE)
    ax.grid(True, color=REJILLA, lw=0.8)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=SUAVE)


def apoyo(ax, despegue, color="#f1efe8"):
    ax.axvspan(0, despegue, color=color, zorder=0)


def a_101(pct, y):
    return np.interp(np.arange(101), pct, y, period=100)


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------

def datos():
    senden = m6.leer_csv("senden_3a6_lenta_ambas.csv")
    senden_f = m6.leer_csv("senden_3a6_lenta_fuerzas.csv")
    par_s = {k: float(v[0]) for k, v in m6.leer_csv("senden_3a6_lenta_parametros.csv").items()}
    lerner = m6.leer_csv("lerner_P4.csv")
    lerner_m = m6.leer_csv("lerner_P4_momento.csv")
    c8 = m6.caso_marcha("crouch")

    sin = di.Inversa("crouch").correr()
    apagado = di.Inversa("crouch_exo_apagado").correr()
    lern = di.Inversa("crouch_exo").correr()
    M = di.Inversa("crouch").masa_nino
    despegue = c8["despegue"]["r"]

    # perfiles calculados sobre la demanda con la masa del exo (musculos_6y)
    pct = apagado["pct"]

    def con_perfil(exo):
        r = dict(apagado, exo=exo, bio=apagado["neto"] - exo)
        r["bio_kg"] = r["bio"] / M
        r["pot_kg"] = r["bio"] * apagado["vel"] / M
        return r

    propuesto = con_perfil(mu.perfil_propuesto(apagado["neto"], pct, despegue))
    correctivo = con_perfil(mu.perfil_correctivo(apagado["neto"], pct, despegue, M))

    return dict(senden=senden, senden_f=senden_f, par_s=par_s, lerner=lerner, lerner_m=lerner_m,
                c8=c8, sin=sin, apagado=apagado, lern=lern, prop=propuesto, corr=correctivo,
                M=M, despegue=despegue)


# ----------------------------------------------------------------------------
# Figura 1: el problema
# ----------------------------------------------------------------------------

def figura_problema(D):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
    pct = np.arange(101)
    c8 = D["c8"]["curvas"]
    rod_c8 = -c8["knee_angle_r"]                      # modelo: flexión negativa
    apoyo(a, D["despegue"])
    a.plot(D["senden"]["pct"], D["senden"]["rodilla_r"], color=C_SANO, lw=2.2, label="niño sano (Senden, 3–6 años)")
    a.plot(D["lerner"]["pct"], D["lerner"]["rodilla"], color=C_P4, lw=2.2, label="P4, PC (Lerner 2017)")
    a.plot(c8["pct"], rod_c8, color=C_C8, lw=2.2, label=f"{m6.NINO_CROUCH}, PC (Steele 2010)")
    estilo(a, "a) Ángulo de rodilla: la rodilla agachada nunca se extiende", "Ciclo de marcha (%)",
           "Flexión de rodilla (°)")
    from matplotlib.patches import Patch
    a.legend(handles=a.get_legend_handles_labels()[0] + [Patch(color="#f1efe8", label=f"apoyo ({m6.NINO_CROUCH})")],
             frameon=False, fontsize=8, loc="upper left")

    apoyo(b, D["despegue"])
    b.plot(pct, D["senden_f"]["momento_rodilla_r"], color=C_SANO, lw=2.2, label="niño sano (medido)")
    b.plot(pct, D["lerner_m"]["momento_rodilla"], color=C_P4, lw=2.2, label="P4 (medido)")
    b.plot(D["sin"]["pct"], D["sin"]["neto"] / D["M"], color=C_C8, lw=2.2, label=f"{m6.NINO_CROUCH} (dinámica inversa)")
    estilo(b, "b) Momento de rodilla: más extensor y por más tiempo en el apoyo", "Ciclo de marcha (%)",
           "Momento neto (N·m/kg), + = extensor")
    b.legend(frameon=False, fontsize=8)
    fig.suptitle("El problema: la marcha agachada exige al extensor de la rodilla durante el apoyo",
                 color=TINTA, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "fig1_problema.png", dpi=160)


# ----------------------------------------------------------------------------
# Figura 2: la solución
# ----------------------------------------------------------------------------

def metricas(r, despegue, M):
    ap = r["pct"] < despegue
    dt = r["dt"]
    pot = r["bio"] * r["vel"] / M
    bio = r["bio"] / M
    return dict(pico_ext=bio[ap].max(), media_ext=np.clip(bio[ap], 0, None).mean(),
                flexor=-np.clip(bio[ap], None, 0).mean(),   # cuánto tiene que contrarrestar el niño
                abs_pico=-pot[ap].min(), w_pos=pot[pot > 0].sum() * dt, w_neg=pot[pot < 0].sum() * dt,
                exo_pico=(r["exo"] / M).max())


def figura_solucion(D):
    M, desp = D["M"], D["despegue"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.6))
    corridas = (("sin exo", D["sin"], C_SIN), ("exo, perfil de Lerner", D["lern"], C_LERNER),
                (f"exo, sigue la demanda ({mu.FRACCION_ASISTENCIA:.0%})", D["prop"], C_PROP),
                ("exo, perfil CORRECTIVO (recomendado)", D["corr"], C_CORR))
    for ax in (a, b, c):
        apoyo(ax, desp)
    a.plot(D["apagado"]["pct"], D["apagado"]["neto"] / M, color=SUAVE, lw=1.8, label="momento neto que necesita la rodilla")
    a.plot(D["lern"]["pct"], D["lern"]["exo"] / M, color=C_LERNER, lw=2.2, label="exo: perfil de Lerner")
    a.plot(D["prop"]["pct"], D["prop"]["exo"] / M, color=C_PROP, lw=2.2, label="exo: sigue la demanda")
    a.plot(D["corr"]["pct"], D["corr"]["exo"] / M, color=C_CORR, lw=2.8, label="exo: perfil CORRECTIVO (recomendado)")
    estilo(a, "a) Torque del exoesqueleto", "Ciclo de marcha (%)", "N·m/kg, + = extensor")
    a.legend(frameon=False, fontsize=8)
    for nombre, r, col in corridas:
        b.plot(r["pct"], r["bio"] / M, color=col, lw=2.2, label=nombre)
        c.plot(r["pct"], r["bio"] * r["vel"] / M, color=col, lw=2.2, label=nombre)
    estilo(b, "b) Torque biológico de rodilla (neto - exo)", "Ciclo de marcha (%)", "N·m/kg, + = extensor")
    estilo(c, "c) Potencia biológica de rodilla", "Ciclo de marcha (%)", "W/kg, + = genera, - = absorbe")
    b.legend(frameon=False, fontsize=8)
    fig.suptitle(f"La solución: asistencia extensora en el apoyo (niño {m6.NINO_CROUCH}, con la masa del exo incluida)",
                 color=TINTA, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "fig2_solucion.png", dpi=160)

    k = {n: metricas(r, desp, M) for n, r, _ in corridas}
    k["exo apagado (solo masa)"] = metricas(D["apagado"], desp, M)
    print(f"\nRODILLA DERECHA DE {m6.NINO_CROUCH} EN EL APOYO (por kg del niño)")
    print(f"{'':<46}{'pico ext.':>10}{'media ext.':>11}{'flexor':>9}{'absorción':>10}{'W+':>8}{'W-':>8}{'exo pico':>10}")
    print(f"{'':<46}{'N·m/kg':>10}{'N·m/kg':>11}{'N·m/kg':>9}{'W/kg':>10}{'J/kg':>8}{'J/kg':>8}{'N·m/kg':>10}")
    print("-" * 112)
    for n in ("sin exo", "exo apagado (solo masa)", "exo, perfil de Lerner",
              f"exo, sigue la demanda ({mu.FRACCION_ASISTENCIA:.0%})", "exo, perfil CORRECTIVO (recomendado)"):
        x = k[n]
        print(f"{n:<46}{x['pico_ext']:>10.3f}{x['media_ext']:>11.3f}{x['flexor']:>9.3f}{x['abs_pico']:>10.2f}"
              f"{x['w_pos']:>8.3f}{x['w_neg']:>8.3f}{x['exo_pico']:>10.3f}")
    print("  flexor = torque flexor medio que el niño tiene que poner en el apoyo para contrarrestar al exo")
    return k


# ----------------------------------------------------------------------------
# Figura 3: sensores y control
# ----------------------------------------------------------------------------

def sensores(D, semilla=1):
    """Tres zancadas de C8 a 500 Hz: señales de los sensores, máquina de estados
    y torque mandado."""
    rng = np.random.default_rng(semilla)
    c8 = D["c8"]
    z, desp = c8["zancada"], D["despegue"]
    M = D["M"]
    t = np.arange(0, ZANCADAS * z, 1 / FRECUENCIA)
    p = (t / z % 1) * 100
    curvas, f = c8["curvas"], c8["fuerzas"]

    # señales reales
    fsr_real = np.interp(p, f["pct"], np.nan_to_num(f["fuerza_arriba_r"]), period=100) / G
    fsr_real = np.where(p < desp, fsr_real, 0.0)
    ang_real = -np.interp(p, curvas["pct"], curvas["knee_angle_r"], period=100)     # flexión, °
    # sensores
    fsr = np.maximum(fsr_real + rng.normal(0, RUIDO_FSR, t.size), 0)
    paso = 360.0 / 2**BITS_ENCODER
    enc = np.round(ang_real / paso) * paso
    # velocidad: derivada del encoder con un filtro de media móvil de 10 ms. La
    # marcha es periódica: se extiende la señal por los dos lados para que el
    # filtro no invente picos en los bordes.
    n = int(0.010 * FRECUENCIA)
    extendida = np.concatenate([enc[-n:], enc, enc[:n]])
    vel = np.gradient(np.convolve(extendida, np.ones(n) / n, mode="same"), 1 / FRECUENCIA)[n:-n]   # °/s, + = flexiona

    # perfil correctivo en función del avance dentro del apoyo (0..1)
    pr = D["corr"]
    ap = pr["pct"] < desp
    perfil_s = pr["pct"][ap] / desp
    perfil_tau = pr["exo"][ap]

    # máquina de estados: APOYO / OSC_TEMPRANA / OSC_TARDIA
    estado, fase_det, cmd = "OSC_TARDIA", [], []
    t_contacto, apoyo_est = None, desp / 100 * z          # duración estimada del apoyo
    t_tardia, tardia_dur = 0.0, (100 - mu.OSC_TARDIA) / 100 * z   # duración estimada de la oscilación tardía
    pico_visto = False
    for i in range(t.size):
        if estado != "APOYO" and fsr[i] > UMBRAL_APOYO[0]:
            estado, t_contacto = "APOYO", t[i]
        elif estado == "APOYO" and fsr[i] < UMBRAL_APOYO[1]:
            apoyo_est = 0.5 * apoyo_est + 0.5 * (t[i] - t_contacto)   # se adapta a cada niño
            estado, pico_visto = "OSC_TEMPRANA", False
        elif estado == "OSC_TEMPRANA":
            if vel[i] < 0:
                pico_visto = True                      # la rodilla ya empezó a extenderse
            if pico_visto and -vel[i] > UMBRAL_EXTENSION:
                estado, t_tardia = "OSC_TARDIA", t[i]
        fase_det.append(estado)
        if estado == "APOYO":
            s = (t[i] - t_contacto) / apoyo_est
            cmd.append(np.interp(s, perfil_s, perfil_tau, right=0.0))
        elif estado == "OSC_TARDIA" and t_contacto is not None:
            # rampa pequeña para llegar al contacto con la rodilla más extendida
            cmd.append(min(1.0, (t[i] - t_tardia) / tardia_dur) * mu.NIVEL_FINAL * M)
        else:
            cmd.append(0.0)                            # oscilación temprana: modo libre
    cmd = np.array(cmd)
    retraso = int(LATENCIA * FRECUENCIA)
    cmd = np.concatenate([np.zeros(retraso), cmd[:-retraso]])
    ideal = np.interp(p, pr["pct"], pr["exo"], period=100)

    # fases reales y errores de detección
    real = np.where(p < desp, "APOYO", "OSC")
    contactos_reales = t[1:][(p[1:] < p[:-1])]                      # el ciclo vuelve a 0
    despegues_reales = t[1:][(p[1:] >= desp) & (p[:-1] < desp)]
    det = np.array(fase_det)
    contactos_det = t[1:][(det[1:] == "APOYO") & (det[:-1] != "APOYO")]
    despegues_det = t[1:][(det[1:] != "APOYO") & (det[:-1] == "APOYO")]
    err = lambda reales, det_: [1000 * (min(det_, key=lambda x: abs(x - r)) - r) for r in reales] if len(det_) else []
    return dict(t=t, p=p, fsr=fsr, fsr_real=fsr_real, enc=enc, vel=vel, det=det, real=real, cmd=cmd, ideal=ideal,
                err_contacto=err(contactos_reales, contactos_det), err_despegue=err(despegues_reales, despegues_det),
                M=M, z=z)


def figura_sensores(D):
    S = sensores(D)
    t, M = S["t"], S["M"]
    fig, ejes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    colores = {"APOYO": "#e3edf9", "OSC_TEMPRANA": "#fbefe2", "OSC_TARDIA": "#e5f4ea"}
    for ax in ejes:                                    # fondo: fase detectada
        cambio = np.r_[0, np.flatnonzero(S["det"][1:] != S["det"][:-1]) + 1, t.size]
        for i0, i1 in zip(cambio[:-1], cambio[1:]):
            ax.axvspan(t[i0], t[min(i1, t.size - 1)], color=colores[S["det"][i0]], zorder=0, lw=0)
    a, b, c, d = ejes
    a.plot(t, S["fsr"], color=C_SIN, lw=1, label="FSR de la plantilla, suma de talón + antepié + dedos (con ruido)")
    a.plot(t, S["fsr_real"], color=TINTA, lw=1, ls=":", label="fuerza real")
    a.axhline(UMBRAL_APOYO[0], color=C_LERNER, lw=1, ls="--", label=f"umbral de apoyo ({UMBRAL_APOYO[0]:.0%} del peso)")
    estilo(a, "a) Sensor de fuerza del pie: detecta contacto inicial y despegue", "", "Fuerza (× peso)")
    a.legend(frameon=False, fontsize=8, loc="upper right")
    b.plot(t, S["enc"], color=C_SIN, lw=1.5)
    estilo(b, f"b) Encoder de la rodilla ({BITS_ENCODER} bits): ángulo", "", "Flexión (°)")
    c.plot(t, S["vel"], color=C_SIN, lw=1.2)
    c.axhline(-UMBRAL_EXTENSION, color=C_PROP, lw=1, ls="--", label=f"umbral de extensión ({UMBRAL_EXTENSION:.0f} °/s)")
    estilo(c, "c) Velocidad de rodilla (derivada del encoder): detecta la oscilación tardía", "",
           "°/s, - = extiende")
    c.legend(frameon=False, fontsize=8, loc="upper right")
    d.plot(t, S["ideal"] / M, color=TINTA, lw=1.2, ls=":", label="perfil ideal (fase real)")
    d.plot(t, S["cmd"] / M, color=C_CORR, lw=2, label=f"torque mandado (máquina de estados, {LATENCIA*1000:.0f} ms de latencia)")
    estilo(d, "d) Control: torque extensor del exo (perfil correctivo)", "Tiempo (s)", "N·m/kg")
    d.legend(frameon=False, fontsize=8, loc="upper right")
    from matplotlib.patches import Patch
    a.legend(handles=a.get_legend().legend_handles + [Patch(color=v, label=f"fase detectada: {k.replace('_', ' ').lower()}")
                                                      for k, v in colores.items()],
             frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.suptitle(f"Sensores y control: la máquina de estados aplica el perfil a tiempo ({m6.NINO_CROUCH}, 500 Hz)",
                 color=TINTA, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "fig3_sensores.png", dpi=160)

    print(f"\nDETECCIÓN DE FASES ({ZANCADAS} zancadas de {m6.NINO_CROUCH}, ruido del FSR {RUIDO_FSR:.0%} del peso)")
    for nombre, e in (("contacto inicial", S["err_contacto"]), ("despegue", S["err_despegue"])):
        if e:
            print(f"  {nombre:<17} error {np.mean(e):+5.1f} ms (máx |{np.max(np.abs(e)):.1f}| ms)")
    rms = np.sqrt(np.mean((S["cmd"] - S["ideal"]) ** 2)) / M
    print(f"  torque mandado vs ideal: diferencia RMS {rms:.3f} N·m/kg (incluye los {LATENCIA*1000:.0f} ms de latencia)")
    return S


# ----------------------------------------------------------------------------
# Figura 4: corrección de la postura (simulación con músculos)
# ----------------------------------------------------------------------------

def figura_postura():
    import marcha_musculos_6y as mm
    perfiles = (("exo, sigue la demanda", "exo propuesto", C_PROP), ("exo, perfil de Lerner", "exo Lerner", C_LERNER),
                ("exo, perfil CORRECTIVO", "exo correctivo", C_CORR))
    res = {}
    for rig in mm.RIGIDECES:
        res[(rig, None)] = mm.correr(False, rig)
        for _, perfil, _ in perfiles:
            res[(rig, perfil)] = mm.correr(True, rig, perfil)

    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    rig = mm.RIGIDECES[1]                       # control intermedio: el que mejor reproduce a C8
    base = res[(rig, None)]
    apoyo(a, base["despegue"])
    x = np.arange(101)
    a.plot(x, mm.por_ciclo(base, "rod_ref"), color=SUAVE, ls=":", lw=1.6, label=f"{m6.NINO_CROUCH} medido")
    a.plot(x, mm.por_ciclo(base, "rod"), color=C_SIN, lw=2.2, label="simulado, exo apagado")
    for nombre, perfil, col in perfiles:
        a.plot(x, mm.por_ciclo(res[(rig, perfil)], "rod"), color=col, lw=2.8 if "CORRECTIVO" in nombre else 2,
               label=f"simulado, {nombre}")
    estilo(a, f"a) Flexión de rodilla con los mismos músculos (control intermedio, reflejo x{rig})",
           "Ciclo de marcha (%)", "Flexión de rodilla (°)")
    a.legend(frameon=False, fontsize=8)

    ancho = 0.25
    etiquetas = {mm.RIGIDECES[0]: "control blando", mm.RIGIDECES[1]: "intermedio", mm.RIGIDECES[-1]: "control rígido"}
    print("\nCORRECCIÓN DE LA POSTURA (simulación con músculos): extensión de rodilla a mitad del apoyo (15–50 %)")
    for k, (nombre, perfil, col) in enumerate(perfiles):
        valores = []
        for rig in mm.RIGIDECES:
            a0, e = res[(rig, None)], res[(rig, perfil)]
            ok = a0["ok"] and e["ok"] and mm.resumen(e)["error"] < 15      # descarta corridas que se descontrolan
            valores.append(mm.resumen(a0)["rod_medio"] - mm.resumen(e)["rod_medio"] if ok else np.nan)
        pos = np.arange(len(mm.RIGIDECES)) + (k - 1) * ancho
        b.bar(pos, np.nan_to_num(valores), ancho, color=col, label=nombre)
        for p_, v in zip(pos, valores):
            b.text(p_, (0 if np.isnan(v) else v) + 0.2, "no\nestable" if np.isnan(v) else f"{v:+.1f}°",
                   ha="center", va="bottom", fontsize=8, color=TINTA)
        print(f"  {nombre:<24}" + "  ".join(f"{etiquetas[r]}: " + ("no estable" if np.isnan(v) else f"{v:+.1f}°")
                                             for r, v in zip(mm.RIGIDECES, valores)))
    b.axhline(m6.MEJORA_EXTENSION, color=TINTA, ls="--", lw=1, label=f"medido por Lerner con su exo (+{m6.MEJORA_EXTENSION:.0f}°)")
    b.set_xticks(np.arange(len(mm.RIGIDECES)), [f"{etiquetas[r]}\n(reflejo x{r})" for r in mm.RIGIDECES])
    estilo(b, "b) Extensión de rodilla a mitad del apoyo por el exo", "", "Extensión (°), + = más extendida")
    b.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Corrección de la postura: con la misma orden muscular, el exo extiende la rodilla "
                 f"({m6.NINO_CROUCH}, cuádriceps débil)", color=TINTA, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTADOS / "fig4_postura.png", dpi=160)


def main():
    RESULTADOS.mkdir(exist_ok=True)
    D = datos()
    figura_problema(D)
    figura_solucion(D)
    figura_sensores(D)
    figura_postura()
    print(f"\nFiguras guardadas en {RESULTADOS}: fig1_problema.png, fig2_solucion.png, fig3_sensores.png, "
          "fig4_postura.png")
    plt.show()


if __name__ == "__main__":
    main()

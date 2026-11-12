"""
Dinámica inversa de la marcha del niño de 6 años: torque y potencia biológicos
de la rodilla con y sin exoesqueleto (el método de Lerner et al. 2017).

  1. Cinemática: la marcha de marcha_6y.py (un ciclo), suavizada con una serie
     de Fourier para derivar velocidades y aceleraciones sin ruido.
  2. Fuerzas del suelo: las medidas por Senden (3 a 6 años, lenta) en cada pie,
     escaladas por el peso del modelo (con el exo, si lo lleva). Se aplican en
     el centro de presión, que avanza en línea recta del talón a la punta del
     pie durante el apoyo (SUPUESTO: Senden no da el centro de presión).
     En los casos de PC se ajustan al apoyo de P4 (67 %) igual que la cadera y
     el tobillo, y se usan con su forma de niño sano (SUPUESTO: no hay fuerzas
     del suelo de P4).
  3. mj_inverse da el torque NETO que necesita cada articulación. El de la
     rodilla suma, por trabajo virtual, las traslaciones acopladas de la tibia.
  4. Torque BIOLÓGICO = neto - torque del exoesqueleto (perfil de Lerner).

Corridas (pierna derecha):
  sano             cinemática y fuerzas de Senden.
  pc               rodilla de P4, sin dispositivo.
  pc_exo_apagado   con la masa del exo (1,0 kg por pierna), sin torque: el costo
                   de llevar el dispositivo.
  pc_exo           con el exo encendido: biológico = neto - exo.
  crouch           niño C8 de Steele 2010 (marcha agachada leve): cinemática,
                   fuerza del suelo y centro de presión MEDIDOS del mismo niño
                   (Marcha.fuerza_suelo de marcha_6y.py), sin supuestos de mezcla.
  crouch_exo_apagado, crouch_exo   lo mismo con el exo (perfil de Lerner por kg,
                   escalado a la masa del niño del modelo).

Validación: el momento neto del sano se compara con el medido por Senden, y el
de P4 sin dispositivo con el medido por Lerner. Los residuales de la pelvis
(fuerza que faltaría para que fuerzas y movimiento cuadren) miden cuánto
encajan la cinemática y las fuerzas, que vienen de niños distintos.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):

  python scripts/dinamica_inversa_6y.py                         # tabla + gráficas
  python scripts/dinamica_inversa_6y.py --ver crouch_exo        # ver la marcha con los torques
  python scripts/dinamica_inversa_6y.py --ver crouch --lento 3  #   (sano, pc, pc_exo_apagado, pc_exo,
  python scripts/dinamica_inversa_6y.py --ver pc_exo            #    crouch, crouch_exo_apagado, crouch_exo)

La gráfica se guarda en resultados/dinamica_inversa_6y.png.
Datos: data/marcha/ (preparar_fuerzas_marcha.py, preparar_datos_marcha_6y.py).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import marcha_6y as m6

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"

ARMONICOS = 12        # armónicos de Fourier que se conservan al suavizar
G = 9.81

CORRIDAS = {
    #  nombre: (caso de marcha, exo en el modelo, exo encendido)
    "sano": ("sano", False, False),
    "pc": ("pc", False, False),
    "pc_exo_apagado": ("pc", True, False),
    "pc_exo": ("pc_exo", True, True),
    "crouch": ("crouch", False, False),
    "crouch_exo_apagado": ("crouch", True, False),
    "crouch_exo": ("crouch_exo", True, True),
    "crouch_exo_mejora": ("crouch_exo_mejora", True, True),
}
TITULOS = {"sano": "sano", "pc": "P4 sin dispositivo", "pc_exo_apagado": "P4 con exo apagado",
           "pc_exo": "P4 con exo (biológico)", "crouch": f"{m6.NINO_CROUCH} sin dispositivo",
           "crouch_exo_apagado": f"{m6.NINO_CROUCH} con exo apagado",
           "crouch_exo": f"{m6.NINO_CROUCH} con exo (biológico)",
           "crouch_exo_mejora": f"{m6.NINO_CROUCH} con exo, postura +12° (biológico)"}
PELVIS = ["pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_rotation", "pelvis_tilt", "pelvis_list"]


def suavizar(columna, n_arm):
    """Serie de Fourier truncada de una señal periódica, y sus derivadas
    primera y segunda respecto al índice de muestra."""
    n = len(columna)
    c = np.fft.rfft(columna)
    c[n_arm + 1:] = 0
    k = np.arange(len(c))
    w = 2j * np.pi * k / n
    return (np.fft.irfft(c, n), np.fft.irfft(c * w, n), np.fft.irfft(c * w * w, n))


class Inversa:
    def __init__(self, corrida):
        caso, con_exo, encendido = CORRIDAS[corrida]
        self.m = m6.caso_marcha(caso)
        self.model, self.data = m6.cargar_modelo(con_exo)
        model = self.model
        self.masa_nino = m6.cargar_modelo(False)[0].body_mass.sum()
        self.peso = model.body_mass.sum() * G          # el suelo carga también el exo
        self.encendido = encendido

        # La referencia comparte el MjData: sus fuerzas del suelo (caso crouch)
        # se calculan con la pose que se acaba de poner.
        ref = m6.Marcha(model, self.data, self.m)
        ref.precalcular()
        self.ref = ref
        z = self.m["zancada"]
        n = m6.MUESTRAS
        self.dt = z / n
        self.p = ref.pcts

        # Cinemática suavizada. El avance de la pelvis no es periódico: cada
        # ciclo empieza por_ciclo más adelante y, como el eje de avance del
        # modelo no es del todo horizontal, con la altura corregida en
        # proporción (ver Marcha.poner). Esa tendencia se quita, se suaviza lo
        # periódico y se vuelve a sumar.
        q = ref.tabla.copy()
        tendencia = np.zeros(model.nq)
        tendencia[ref.tx] = ref.por_ciclo
        tendencia[ref.ty] = -ref.por_ciclo * ref.eje_x[2] / ref.eje_y[2]
        rampa = np.outer(np.arange(n) / n, tendencia)
        q -= rampa
        self.q, self.v, self.a = np.zeros_like(q), np.zeros_like(q), np.zeros_like(q)
        for j in range(model.nq):
            s, d1, d2 = suavizar(q[:, j], ARMONICOS)
            self.q[:, j], self.v[:, j], self.a[:, j] = s, d1 / self.dt, d2 / self.dt**2
        self.q += rampa
        self.v += tendencia / z

        fz = m6.leer_csv("senden_3a6_lenta_fuerzas.csv")
        self.f = fz
        par = {k: float(v[0]) for k, v in m6.leer_csv("senden_3a6_lenta_parametros.csv").items()}
        self.apoyo_senden = {"r": par["apoyo_r"], "l": par["apoyo_l"]}

        rod = model.joint("knee_angle_r").id
        self.rodilla = (model.jnt_qposadr[rod], model.jnt_dofadr[rod])
        # coordenadas acopladas a la rodilla derecha: (dof, junta, coeficientes)
        self.acoples = [(model.jnt_dofadr[model.eq_obj1id[i]], model.eq_data[i][:5][::-1].copy())
                        for i in range(model.neq)
                        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[i] == rod]
        self.pelvis = [model.jnt_dofadr[model.joint(n_).id] for n_ in PELVIS]
        self.pies = {l: {n_: model.geom(f"contacto_{n_}_{l}").id for n_ in m6.CONTACTOS} for l in m6.LADOS}
        self.calcn = {l: model.body(f"calcn_{l}").id for l in m6.LADOS}

        # Sin restricciones (los acoples se respetan en la cinemática y el
        # contacto se reemplaza por las fuerzas medidas) ni resortes o
        # amortiguadores de las juntas (no son del niño).
        model.opt.disableflags |= (int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT) | int(mujoco.mjtDisableBit.mjDSBL_SPRING)
                                   | int(mujoco.mjtDisableBit.mjDSBL_DAMPER))

    def fuerza_suelo(self, p, lado):
        """Fuerza del suelo sobre el pie (mundo, N) y avance del apoyo (0..1)."""
        fase = self.ref.fase_pierna(p, lado)
        despegue = self.m["despegue"][lado]
        apoyo = self.apoyo_senden[lado]
        if fase >= despegue:
            return np.zeros(3), None
        origen = np.interp(fase, [0, despegue, 100], [0, apoyo, 100])
        fv = np.interp(origen, self.f["pct"], self.f[f"fvert_{lado}"], period=100)
        fh = np.interp(origen, self.f["pct"], self.f[f"fhor_{lado}"], period=100)
        # Senden: horizontal + = frenado (hacia atrás); el modelo avanza en +x
        return np.array([-fh, 0.0, fv]) * self.peso, origen / apoyo

    def correr(self):
        model, d = self.model, self.data
        qa, va = self.rodilla
        out = {k: [] for k in ("pct", "neto", "exo", "vel", "res_f", "res_m")}
        self.flechas = []           # (centro de presión, fuerza) de cada pie, para el visor
        for i in range(m6.MUESTRAS):
            d.qpos[:], d.qvel[:], d.qacc[:] = self.q[i], self.v[i], self.a[i]
            mujoco.mj_inverse(model, d)
            externa = np.zeros(model.nv)
            self.flechas.append([])
            for l in m6.LADOS:
                if self.m["fuerzas"] is not None:
                    # crouch: fuerza y centro de presión medidos del mismo niño
                    medida = self.ref.fuerza_suelo(self.p[i], l)
                    if medida is None:
                        continue
                    cop, F = medida
                else:
                    F, s = self.fuerza_suelo(self.p[i], l)
                    if s is None:
                        continue
                    talon = d.geom_xpos[self.pies[l]["talon"]].copy()
                    punta = d.geom_xpos[self.pies[l]["dedos"]].copy()
                    cop = talon + np.clip(s, 0, 1) * (punta - talon)
                    cop[2] -= self.ref.radio
                mujoco.mj_applyFT(model, d, F, np.zeros(3), cop, self.calcn[l], externa)
                self.flechas[-1].append((l, cop.copy(), F.copy()))
            tau = d.qfrc_inverse - externa
            # rodilla efectiva por trabajo virtual: τ_θ + Σ τ_j · ∂q_j/∂θ
            x = d.qpos[qa] - model.qpos0[qa]
            neto = tau[va] + sum(tau[dof] * np.polyval(np.polyder(c), x) for dof, c in self.acoples)
            exo = 0.0
            if self.encendido:
                # en N·m (en crouch, Marcha ya lo escaló a la masa del niño)
                exo = np.interp(self.ref.fase_pierna(self.p[i], "r"), self.ref.pct, self.ref.torque, period=100)
            out["pct"].append(self.p[i])
            out["neto"].append(neto)
            out["exo"].append(exo)
            out["vel"].append(d.qvel[va])
            out["res_f"].append(np.linalg.norm(tau[self.pelvis[:3]]))
            out["res_m"].append(np.linalg.norm(tau[self.pelvis[3:]]))
        r = {k: np.array(v) for k, v in out.items()}
        M = self.masa_nino
        r["bio"] = r["neto"] - r["exo"]
        r["bio_kg"] = r["bio"] / M
        r["pot_kg"] = r["bio"] * r["vel"] / M            # + = genera (extensión positiva en el modelo)
        r["res_f_peso"] = r["res_f"] / (M * G) * 100
        r["res_m_kg"] = r["res_m"] / M
        r["despegue"] = self.m["despegue"]["r"]
        r["dt"] = self.dt
        return r


def metricas(r):
    apoyo = r["pct"] < r["despegue"]
    pot, dt = r["pot_kg"], r["dt"]
    return {
        "pico_ext": r["bio_kg"][apoyo].max(),
        "media_ext": r["bio_kg"][apoyo].mean(),
        "pico_abs": -pot[apoyo].min(),
        "w_pos": pot[pot > 0].sum() * dt,
        "w_neg": pot[pot < 0].sum() * dt,
        "res_f": r["res_f_peso"].mean(),
        "res_m": r["res_m_kg"].mean(),
    }


def comparar(r, medido):
    """Correlación y diferencia RMS (N·m/kg) del momento neto con uno medido."""
    neto = np.interp(np.arange(101), r["pct"], r["neto"] / CORR["masa"], period=100)
    return np.corrcoef(neto, medido)[0, 1], np.sqrt(np.mean((neto - medido) ** 2))


CORR = {}


def ver(corrida, lento=1.0):
    """Muestra la marcha de una corrida con la fuerza del suelo (flechas) e
    imprime en cada instante el momento neto, el torque del exo y el torque
    biológico de la rodilla derecha (los mismos números de la gráfica)."""
    import time
    import mujoco.viewer
    inv = Inversa(corrida)
    r = inv.correr()
    model, d, ref = inv.model, inv.data, inv.ref
    M, peso = inv.masa_nino, model.body_subtreemass[0] * G
    z, n = inv.m["zancada"], m6.MUESTRAS
    subida = ref.eje_x[2] / ref.eje_y[2]
    print(f"{TITULOS[corrida]}: dinámica inversa. Flechas rojas = fuerza del suelo en su centro de presión. "
          "Cierra la ventana para salir.")
    print("  rodilla derecha en N·m/kg: neto = lo que necesita la articulación; "
          "exo = lo que pone el motor; biológico = neto - exo (lo que pone el niño)")
    with mujoco.viewer.launch_passive(model, d) as visor:
        visor.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        visor.cam.trackbodyid = model.body("pelvis").id
        visor.cam.distance, visor.cam.azimuth, visor.cam.elevation = 2.0, 90.0, -10.0
        inicio = time.time()
        while visor.is_running():
            t = (time.time() - inicio) / lento
            ciclos = t / z
            k = int(ciclos)
            i = int((ciclos - k) * n) % n
            # avanza por la pista TRAMO ciclos y vuelve, como marcha_6y
            avance = (k % m6.TRAMO) * ref.por_ciclo - m6.TRAMO * ref.por_ciclo / 2
            d.qpos[:] = inv.q[i]
            d.qpos[ref.tx] += avance
            d.qpos[ref.ty] -= avance * subida
            mujoco.mj_forward(model, d)
            desplazo = avance * ref.eje_x
            m6.flechas(visor, [(cop + desplazo, F) for _, cop, F in inv.flechas[i]], peso)
            visor.sync()
            p = r["pct"][i]
            fz = sum(F[2] for l, _, F in inv.flechas[i] if l == "r") / peso
            print(f"  ciclo {p:5.1f} %  {m6.fase(p, r['despegue']):<13} rodilla {-np.degrees(d.qpos[inv.rodilla[0]]):5.1f}°  "
                  f"neto {r['neto'][i] / M:+.3f}  exo {r['exo'][i] / M:+.3f}  biológico {r['bio_kg'][i]:+.3f} N·m/kg  "
                  f"suelo der {fz:4.2f}×peso", end="\r")
            time.sleep(1 / 60)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", choices=list(CORRIDAS), default=None, help="ver una corrida en el visor")
    parser.add_argument("--lento", type=float, default=1.0, help="factor de cámara lenta para --ver")
    args = parser.parse_args()
    if args.ver:
        ver(args.ver, args.lento)
        return

    res = {}
    for n in CORRIDAS:
        inv = Inversa(n)
        CORR["masa"] = inv.masa_nino
        res[n] = inv.correr()

    senden = m6.leer_csv("senden_3a6_lenta_fuerzas.csv")["momento_rodilla_r"]
    lerner = m6.leer_csv("lerner_P4_momento.csv")["momento_rodilla"]
    print("\nValidación del momento neto de rodilla (N·m/kg):")
    for n, medido, fuente in (("sano", senden, "Senden"), ("pc", lerner, "Lerner (P4)")):
        rho, rms = comparar(res[n], medido)
        print(f"  {TITULOS[n]:<20} vs {fuente:<12} correlación {rho:.2f}, diferencia RMS {rms:.3f}")

    print(f"\n{'Rodilla derecha':<24}{'pico ext.':>10}{'media ext.':>11}{'absorción':>10}{'W+':>8}{'W-':>8}"
          f"{'residual':>10}{'residual':>10}")
    print(f"{'':<24}{'(N·m/kg)':>10}{'apoyo':>11}{'(W/kg)':>10}{'(J/kg)':>8}{'(J/kg)':>8}{'(% peso)':>10}{'(N·m/kg)':>10}")
    print("-" * 91)
    k = {n: metricas(r) for n, r in res.items()}
    for n in CORRIDAS:
        x = k[n]
        print(f"{TITULOS[n]:<24}{x['pico_ext']:>10.3f}{x['media_ext']:>11.3f}{x['pico_abs']:>10.2f}"
              f"{x['w_pos']:>8.3f}{x['w_neg']:>8.3f}{x['res_f']:>10.1f}{x['res_m']:>10.3f}")
    for base, apagado, encendido in (("pc", "pc_exo_apagado", "pc_exo"),
                                     ("crouch", "crouch_exo_apagado", "crouch_exo")):
        b, a, e = k[base], k[apagado], k[encendido]
        # Diferencias absolutas (el porcentaje no sirve si la media del apoyo es
        # cercana a cero, como en los niños con marcha agachada leve).
        print(f"\n{TITULOS[base]} (diferencias en N·m/kg y W/kg):")
        print(f"  masa del exo (apagado - sin dispositivo): media ext. apoyo {a['media_ext'] - b['media_ext']:+.3f}, "
              f"pico {a['pico_ext'] - b['pico_ext']:+.3f}")
        print(f"  exo encendido - sin dispositivo:           media ext. apoyo {e['media_ext'] - b['media_ext']:+.3f}, "
              f"pico {e['pico_ext'] - b['pico_ext']:+.3f}, absorción pico {e['pico_abs'] - b['pico_abs']:+.2f}")

    graficar(res, ("pc", "pc_exo_apagado", "pc_exo"), "dinamica_inversa_6y.png",
             "Dinámica inversa, niño de 6 años con la rodilla de P4", validacion=(senden, lerner))
    graficar(res, ("crouch", "crouch_exo_apagado", "crouch_exo"), "dinamica_inversa_6y_crouch.png",
             f"Dinámica inversa, niño de 6 años con la marcha agachada de {m6.NINO_CROUCH} (Steele 2010)")
    plt.show()


def graficar(res, nombres, archivo, titulo, validacion=None):
    tinta, suave = "#3d3d3a", "#73726c"
    colores = dict(zip(nombres, ("#86b6ef", "#2a78d6", "#104281")))
    fig, ejes = plt.subplots(1, 3, figsize=(16, 4.6))
    pct = np.arange(101)
    ax = ejes[0]
    if validacion is not None:
        senden, lerner = validacion
        ax.plot(pct, senden, color="#8a8983", ls=":", lw=1.5, label="Senden medido")
        ax.plot(pct, lerner, color=colores[nombres[0]], ls=":", lw=1.5, label="Lerner P4 medido")
        ax.plot(res["sano"]["pct"], res["sano"]["neto"] / CORR["masa"], color="#8a8983", lw=2, label="sano (modelo)")
        ax.set_title("Validación: momento neto de rodilla", loc="left", color=tinta)
    else:
        ax.set_title("Momento neto de rodilla", loc="left", color=tinta)
    ax.plot(res[nombres[0]]["pct"], res[nombres[0]]["neto"] / CORR["masa"], color=colores[nombres[0]], lw=2,
            label=f"{TITULOS[nombres[0]]} (modelo)")
    for n in nombres:
        r = res[n]
        ejes[1].plot(r["pct"], r["bio_kg"], color=colores[n], lw=2, label=TITULOS[n])
        ejes[2].plot(r["pct"], r["pot_kg"], color=colores[n], lw=2, label=TITULOS[n])
    ejes[1].plot(res[nombres[2]]["pct"], res[nombres[2]]["exo"] / CORR["masa"], color="#eb6834", ls="--", lw=1.5,
                 label="torque del exo")
    ejes[1].set_title("Torque biológico de rodilla", loc="left", color=tinta)
    ejes[2].set_title("Potencia biológica de rodilla", loc="left", color=tinta)
    for ax, y in zip(ejes, ("N·m/kg (+ = extensor)", "N·m/kg (+ = extensor)", "W/kg (+ = genera)")):
        ax.axhline(0, color="#e6e5e0", lw=0.8)
        ax.axvline(res[nombres[0]]["despegue"], color="#c3c2b7", lw=1)
        ax.set_xlabel("Ciclo de marcha (%)", color=suave)
        ax.set_ylabel(y, color=suave)
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=suave)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(titulo + ": rodilla derecha", color=tinta, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    RESULTADOS.mkdir(exist_ok=True)
    fig.savefig(RESULTADOS / archivo, dpi=150)
    print(f"Gráfica guardada en {RESULTADOS / archivo}")


if __name__ == "__main__":
    main()

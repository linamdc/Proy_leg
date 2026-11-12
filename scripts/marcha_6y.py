"""
Marcha de las dos piernas en el modelo del niño de 6 años (MuJoCo), con
cadera, rodilla y tobillo en ambas piernas y los pies en el suelo.

Cinemática impuesta: cada articulación sigue el ángulo medido en cada % del
ciclo. La pelvis ya no está colgada:
  - su inclinación, oblicuidad y rotación siguen a Senden (sin la media, que
    depende de cómo Plug-in Gait define la pelvis);
  - su altura se ajusta para que el pie más bajo toque el suelo;
  - avanza lo necesario para que el pie que acaba de pisar no resbale, así
    que la velocidad sale de la cinemática y no se impone (sano: 0,90 m/s;
    Senden midió 0,84 m/s).

Casos (las dos piernas en todos):
  - sano:   cadera, rodilla y tobillo de Senden (3 a 6 años, lenta), cada pierna
            con su curva; zancada de 0,95 s; la izquierda desfasada según el
            tiempo de paso (50,3 % del ciclo).
  - pc:     rodilla de P4 (Lerner 2017, sin exoesqueleto) en las dos piernas,
            desfasada medio ciclo; zancada de 1,14 s; despegue al 67 %.
            SUPUESTOS: cadera, tobillo y pelvis de Senden, registradas en el
            despegue de P4 (su apoyo se estira del 59 % al 67 % del ciclo, la
            forma no cambia); el tobillo suma la flexión extra de la rodilla de
            P4 para que el pie apoye plano (FRACCION_TOBILLO) y el pie que
            oscila se levanta hasta el despeje de Senden (DESPEJE_PIE); y la
            curva de P4 (pierna más afectada) se usa en las dos piernas.
  - pc_exo: lo mismo, con el exoesqueleto en las dos piernas (el de Lerner era
            bilateral): las piezas de exo.py (motor AK60-6, barras, valvas,
            almohadilla y tobillo libre, 1,0 kg por pierna) y un actuador por
            rodilla, saturado a ±6 N·m, que aplica el perfil de torque de P4
            en la fase de cada pierna. Como la cinemática está impuesta, el
            torque no cambia el movimiento: se muestra en la terminal y en los
            sensores.

  - crouch: un niño real con marcha agachada (Steele et al. 2010, SimTK
            crouchgait; por defecto C8, 23,3 kg, rodilla a 33° en el apoyo):
            pelvis, tronco, caderas, rodillas y tobillos de las dos piernas de
            una misma zancada, y la fuerza del suelo con su centro de presión
            dibujada como flecha. Sin supuestos de mezcla ni registro.
  - crouch_exo: lo mismo, con el exoesqueleto de exo.py en las dos piernas y el
            perfil de torque de P4 por kg, escalado a la masa del niño del
            modelo (sin el exo) y ajustado al despegue de este niño.
  - crouch_exo_mejora: lo mismo, con la postura que se espera con el exo según
            lo medido por Lerner (+12° de extensión de rodilla en el apoyo,
            MEJORA_EXTENSION). No es una predicción de la simulación.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):

  python scripts/marcha_6y.py --caso sano
  python scripts/marcha_6y.py --caso pc
  python scripts/marcha_6y.py --caso pc_exo
  python scripts/marcha_6y.py --caso crouch
  python scripts/marcha_6y.py --caso crouch_exo
  python scripts/marcha_6y.py --caso crouch_exo_mejora     # con la mejora de postura de Lerner
  python scripts/marcha_6y.py --caso crouch --nino C9    # otro niño de Steele
  python scripts/marcha_6y.py --caso sano --lento 3      # 3 veces más lento
  python scripts/marcha_6y.py --caso sano --cinta        # sin avanzar (como en cinta)

Datos: data/marcha/ (ver preparar_datos_marcha_6y.py, preparar_datos_marcha.py y
preparar_datos_crouch_steele.py).
Modelo: modelos/nino6y_mujoco/nino6y_marcha.xml.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import exo as ex

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "modelos" / "nino6y_mujoco" / "nino6y_marcha.xml"
DATOS = RAIZ / "data" / "marcha"

ZANCADA_P4 = 1.14      # s, 2 × 60 / 105 pasos/min (Lerner Tabla 2)
DESPEGUE_P4 = 67.0     # % del ciclo (banda gris de la Figura 2 de Lerner)
DESFASE_P4 = 50.0      # % del ciclo, supuesto simétrico
# Compensación de la marcha agachada en el caso pc. De P4 solo hay rodilla: si
# se usan la cadera y el tobillo del niño sano con una rodilla 25–40° más
# flexionada, el pie gira con la pierna y queda en punta (planta a -20° a -58°
# en el apoyo). En el plano sagital la inclinación del pie es cadera - rodilla
# + tobillo, así que la flexión extra de la rodilla (Δ = rodilla de P4 -
# rodilla de Senden, solo si es positiva) se reparte entre tobillo
# (dorsiflexión) y cadera (flexión) y el pie conserva la orientación respecto
# al suelo del niño sano. SUPUESTO: el reparto real de P4 no se conoce; 1.0 =
# todo en el tobillo, 0.0 = todo en la cadera.
FRACCION_TOBILLO = 1.0
# En la oscilación temprana P4 dobla la rodilla menos que el sano (55° contra
# 70°) y la punta del pie que oscila queda más baja que el pie de apoyo. Como
# haría el niño, se sube la punta (más dorsiflexión) hasta este despeje sobre el
# pie de apoyo: el despeje medio medido por Senden a los 3–6 años (gait_slow,
# foot.clear = 1,54 cm). Solo en los casos pc.
DESPEJE_PIE = 0.0154   # m
MUESTRAS = 400         # poses precalculadas por ciclo
# Grosor del muslo y de la pierna para ubicar valvas, barras y motor del exo.
# El modelo de 6 años no tiene cápsulas de colisión en las piernas: se usan las
# de la pierna MyoLeg escalada a P4 (femur2_col_r y tibia1_col_r).
RADIO_MUSLO = 0.0456   # m
RADIO_PIERNA = 0.0381  # m
TRAMO = 4              # ciclos antes de volver al inicio de la pista

LADOS = ("r", "l")
JUNTAS = {  # nombre de la curva -> (junta del modelo, signo)
    "cadera_flexion": ("hip_flexion", 1.0),
    "cadera_aduccion": ("hip_adduction", 1.0),
    "cadera_rotacion": ("hip_rotation", 1.0),
    "rodilla": ("knee_angle", -1.0),     # en el modelo la flexión es negativa
    "tobillo": ("ankle_angle", 1.0),
}
PELVIS = {  # curva -> (junta, signo)   (ejes del modelo: ver la cabecera)
    "pelvis_inclinacion": ("pelvis_tilt", -1.0),   # tilt + = retroversión
    "pelvis_oblicuidad": ("pelvis_list", -1.0),    # list + = baja el lado derecho
    "pelvis_rotacion": ("pelvis_rotation", 1.0),   # rotation + = adelanta el derecho
}
CONTACTOS = ("talon", "antepie_a", "antepie_b", "dedos")
NINO_CROUCH = "C8"     # Steele 2010: 23,3 kg, rodilla a 33° en el apoyo
MASA_P4 = 20.6         # kg (Lerner Tabla 1)
CROUCH_JUNTAS = ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle")
CROUCH_TRONCO = ("pelvis_tilt", "pelvis_list", "pelvis_rotation",
                 "lumbar_extension", "lumbar_bending", "lumbar_rotation")
ESCALA_FLECHA = 0.5    # m de flecha por cada peso corporal
# Mejora esperada de la postura con el exo (caso crouch_exo_mejora). La
# simulación no predice el cambio de la marcha (la cinemática está impuesta): se
# usa el efecto MEDIDO por Lerner et al. 2017 con su exoesqueleto, +12° de
# extensión de rodilla en el apoyo. Se aplica con una campana que es máxima a
# mitad del apoyo y vale cero en el contacto inicial y en el despegue; el
# tobillo compensa lo mismo para que el pie siga apoyando plano.
MEJORA_EXTENSION = 12.0   # °


def leer_csv(nombre):
    with open(DATOS / nombre, encoding="utf-8") as f:
        encabezado = [linea for linea in f if linea.startswith("#")][-1]
    columnas = encabezado.lstrip("# ").strip().split(",")
    datos = np.atleast_2d(np.loadtxt(DATOS / nombre, delimiter=",", comments="#"))
    return {c: datos[:, i] for i, c in enumerate(columnas)}


JUNTAS_SENDEN = [(f"{curva}_{lado}", f"{junta}_{lado}", signo, lado)
                 for lado in LADOS for curva, (junta, signo) in JUNTAS.items()]
TRONCO_SENDEN = [(curva, junta, signo) for curva, (junta, signo) in PELVIS.items()]


def caso_crouch(caso, nino):
    """Un mismo niño con marcha agachada (Steele 2010): todo de la misma zancada."""
    carpeta = f"crouch_steele/{nino}"
    c = leer_csv(f"{carpeta}_cinematica.csv")          # ciclo derecho, convención OpenSim
    par = {k: float(v[0]) for k, v in leer_csv(f"{carpeta}_parametros.csv").items()}
    f = leer_csv(f"{carpeta}_fuerzas.csv")             # ciclo propio de cada pierna
    pct, desfase = c["pct"], par["desfase_l"]
    for j in CROUCH_JUNTAS:                            # izquierda: a su propio ciclo
        c[f"{j}_l"] = np.interp((pct + desfase) % 100, pct, c[f"{j}_l"], period=100)
    if caso.endswith("_mejora"):
        for lado in LADOS:
            desp = par[f"despegue_{lado}"]
            campana = np.where(pct < desp, np.sin(np.pi * pct / desp) ** 2, 0.0)
            # en el modelo la flexión de rodilla es negativa: extender = sumar
            c[f"knee_angle_{lado}"] = c[f"knee_angle_{lado}"] + MEJORA_EXTENSION * campana
            c[f"ankle_angle_{lado}"] = c[f"ankle_angle_{lado}"] - MEJORA_EXTENSION * campana
    torque = None
    if "_exo" in caso:
        # N·m/kg (se escala a la masa del niño del modelo). El perfil cambia de
        # fase en el despegue de P4 (67 %): se ajusta al despegue de este niño.
        perfil = leer_csv("lerner_P4.csv")["torque_exo"] / MASA_P4
        origen = np.interp(pct, [0, par["despegue_r"], 100], [0, DESPEGUE_P4, 100])
        torque = np.interp(origen, pct, perfil)
    return dict(curvas=c, juntas=[(f"{j}_{l}", f"{j}_{l}", 1.0, l) for l in LADOS for j in CROUCH_JUNTAS],
                tronco=[(j, j, 1.0) for j in CROUCH_TRONCO], fuerzas=f,
                torque=torque, torque_por_kg=True, zancada=par["zancada"],
                despegue={"r": par["despegue_r"], "l": par["despegue_l"]}, desfase=desfase,
                velocidad_ref=par["velocidad"], nino=nino, masa_nino=par["masa"], despeje=None)


def caso_marcha(caso, nino=NINO_CROUCH):
    """Curvas (° por % del ciclo de cada pierna), torque del exo y tiempos."""
    if caso.startswith("crouch"):
        return caso_crouch(caso, nino)
    base = dict(juntas=JUNTAS_SENDEN, tronco=TRONCO_SENDEN, fuerzas=None, torque_por_kg=False, despeje=None)
    c = leer_csv("senden_3a6_lenta_ambas.csv")
    par = {k: float(v[0]) for k, v in leer_csv("senden_3a6_lenta_parametros.csv").items()}
    for k in PELVIS:
        c[k] = c[k] - c[k].mean()
    if caso == "sano":
        return dict(base, curvas=c, torque=None, zancada=par["zancada"],
                    despegue={"r": par["apoyo_r"], "l": par["apoyo_l"]},
                    desfase=par["desfase_l"], velocidad_ref=par["velocidad"])
    p4 = leer_csv("lerner_P4.csv")
    c = dict(c)
    # Registro en el despegue: el apoyo de Senden (hasta su despegue) se estira
    # hasta el despegue de P4 y la oscilación se comprime, para que cadera,
    # tobillo y pelvis cambien de fase al mismo tiempo que la rodilla de P4.
    # La forma de las curvas no cambia, solo su reparto en el ciclo.
    pct = c["pct"]
    for nombre in list(c):
        if nombre == "pct":
            continue
        lado = "l" if nombre.endswith("_l") else "r"
        apoyo = par[f"apoyo_{lado}"]
        origen = np.interp(pct, [0, DESPEGUE_P4, 100], [0, apoyo, 100])
        c[nombre] = np.interp(origen, pct, c[nombre], period=100)
    for lado in LADOS:
        # Solo se compensa la flexión EXTRA (Δ > 0). En la oscilación temprana
        # P4 dobla la rodilla menos que el sano (Δ < 0): compensar ahí bajaría
        # la punta del pie por debajo del suelo.
        delta = np.maximum(p4["rodilla"] - c[f"rodilla_{lado}"], 0.0)
        c[f"tobillo_{lado}"] = c[f"tobillo_{lado}"] + FRACCION_TOBILLO * delta
        c[f"cadera_flexion_{lado}"] = c[f"cadera_flexion_{lado}"] + (1 - FRACCION_TOBILLO) * delta
        c[f"rodilla_{lado}"] = p4["rodilla"]
    return dict(base, curvas=c, torque=p4["torque_exo"] if caso.endswith("_exo") else None,
                zancada=ZANCADA_P4, despegue={"r": DESPEGUE_P4, "l": DESPEGUE_P4},
                desfase=DESFASE_P4, velocidad_ref=None, despeje=DESPEJE_PIE)


def cargar_modelo(con_exo):
    spec = spec_modelo(con_exo)
    model = spec.compile()
    return model, mujoco.MjData(model)


def spec_modelo(con_exo):
    """MjSpec del niño de 6 años, con o sin el exoesqueleto (sin compilar)."""
    spec = mujoco.MjSpec.from_file(str(MODELO))
    if con_exo:
        # El mismo exoesqueleto que la pierna MyoLeg (exo.py), en las dos
        # piernas: piezas con su masa, inercia del rotor, tope de 100° y
        # actuador saturado a ±6 N·m. Las piezas se ubican con las piernas
        # extendidas (qpos0, que es una pose válida: ver ajustar_mujoco316.py).
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        for lado in LADOS:
            rodilla, lateral, adelante, _ = ex.direcciones(
                model, data, f"knee_angle_{lado}", f"femur_{lado}", "pelvis",
                f"calcn_{lado}", f"toes_{lado}", f"tibia_{lado}")
            # FSR de la plantilla: uno bajo cada punto de apoyo del pie (el
            # punto más bajo de cada esfera de contacto), en el cuerpo que apoya
            planta = []
            for n in CONTACTOS:
                g = model.geom(f"contacto_{n}_{lado}").id
                punto = data.geom_xpos[g] - np.array([0, 0, model.geom_size[g][0]])
                planta.append((model.body(model.geom_bodyid[g]).name, n, punto))
            ex.agregar_exo(spec, model, data, femur=f"femur_{lado}", tibia=f"tibia_{lado}",
                           talus=f"talus_{lado}", rodilla=rodilla, lateral=lateral, adelante=adelante,
                           radio_muslo=RADIO_MUSLO, radio_pierna=RADIO_PIERNA,
                           junta_rodilla=f"knee_angle_{lado}", signo_flexion=-1,
                           nombre_muslo=f"exo_muslo_{lado}", nombre_pierna=f"exo_pierna_{lado}",
                           nombre_actuador=f"exo_rodilla_{lado}", nombre_sensor=f"exo_torque_{lado}",
                           sufijo=f"_{lado}", planta=planta)
        ex.agregar_materiales(spec)
    return spec


class Marcha:
    def __init__(self, model, data, m):
        self.model, self.data, self.m = model, data, m
        self.pct = m["curvas"]["pct"]
        q = lambda n: model.jnt_qposadr[model.joint(n).id]
        self.juntas = []
        for curva, junta, signo, lado in m["juntas"]:
            j = model.joint(junta).id
            self.juntas.append((curva, lado, model.jnt_qposadr[j], signo, model.jnt_range[j]))
        self.pelvis = [(curva, q(junta), signo) for curva, junta, signo in m["tronco"]]
        self.tobillo = {l: model.joint(f"ankle_angle_{l}").id for l in LADOS}
        self.tx, self.ty, self.tz = q("pelvis_tx"), q("pelvis_ty"), q("pelvis_tz")
        # ejes de traslación de la pelvis en el mundo (en el modelo convertido el
        # de avance no es del todo horizontal: la altura se corrige con ellos)
        mujoco.mj_kinematics(model, data)
        eje = lambda n: data.xaxis[model.joint(n).id].copy()
        self.eje_x, self.eje_y, self.eje_z = eje("pelvis_tx"), eje("pelvis_ty"), eje("pelvis_tz")
        self.plano = np.linalg.inv(np.column_stack([self.eje_x[:2], self.eje_z[:2]]))
        # todas las coordenadas acopladas (rodilla, puntos móviles de los músculos)
        self.acoples = [(model.jnt_qposadr[model.eq_obj1id[i]], model.jnt_qposadr[model.eq_obj2id[i]],
                         model.eq_data[i][:5][::-1].copy())
                        for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT]
        self.contactos = {lado: [model.geom(f"contacto_{n}_{lado}").id for n in CONTACTOS] for lado in LADOS}
        self.radio = model.geom_size[self.contactos["r"][0]][0]
        self.base = (model.key_qpos[0] if model.nkey else model.qpos0).copy()
        self.exo = ({lado: model.actuator(f"exo_rodilla_{lado}").id for lado in LADOS}
                    if m["torque"] is not None else None)
        # masa total (el suelo carga también el exo) y masa del niño sin el exo
        self.masa = model.body_subtreemass[0]
        self.masa_nino = self.masa - sum(model.body_mass[b] for b in range(model.nbody)
                                         if model.body(b).name.startswith("exo_"))
        self.torque = (m["torque"] * self.masa_nino if m["torque"] is not None and m["torque_por_kg"]
                       else m["torque"])

    def fase_pierna(self, p, lado):
        return p if lado == "r" else (p - self.m["desfase"]) % 100

    def poner_angulos(self, p):
        d, q0 = self.data, self.model.qpos0
        d.qpos[:] = self.base
        c = self.m["curvas"]
        for curva, lado, adr, signo, (lo, hi) in self.juntas:
            ang = np.radians(signo * np.interp(self.fase_pierna(p, lado), self.pct, c[curva], period=100))
            d.qpos[adr] = np.clip(ang, lo, hi)
        for curva, adr, signo in self.pelvis:
            d.qpos[adr] = np.radians(signo * np.interp(p, self.pct, c[curva], period=100))
        for _ in range(2):
            for a, b, coef in self.acoples:
                d.qpos[a] = q0[a] + np.polyval(coef, d.qpos[b] - q0[b])
        d.qpos[self.tx] = d.qpos[self.ty] = d.qpos[self.tz] = 0.0
        mujoco.mj_kinematics(self.model, d)

    def pie(self, lado):
        return self.data.geom_xpos[self.contactos[lado]]

    def despejar(self, p):
        """Sube la punta del pie que oscila (más dorsiflexión) hasta quedar
        DESPEJE_PIE por encima del pie de apoyo. Para si el tobillo llega a su
        límite o si subir la punta ya no levanta el pie (el talón queda abajo)."""
        if self.m["despeje"] is None:
            return
        apoyo = [l for l in LADOS if self.fase_pierna(p, l) < self.m["despegue"][l]]
        if not apoyo or len(apoyo) == len(LADOS):
            return
        z_apoyo = min(self.pie(l)[:, 2].min() for l in apoyo)
        for l in LADOS:
            if l in apoyo:
                continue
            j = self.tobillo[l]
            adr, tope = self.model.jnt_qposadr[j], self.model.jnt_range[j][1]
            z = self.pie(l)[:, 2].min()
            while z < z_apoyo + self.m["despeje"] and self.data.qpos[adr] < tope:
                self.data.qpos[adr] = min(self.data.qpos[adr] + np.radians(0.5), tope)
                mujoco.mj_kinematics(self.model, self.data)
                z_nuevo = self.pie(l)[:, 2].min()
                if z_nuevo <= z:
                    break
                z = z_nuevo

    def altura(self, fila, z):
        """pelvis_ty para que el punto más bajo de los pies quede en el suelo."""
        return (-z - fila[self.tx] * self.eje_x[2] - fila[self.tz] * self.eje_z[2]) / self.eje_y[2]

    def precalcular(self):
        """Poses de un ciclo con la pelvis apoyada y desplazándose (adelante y de
        lado) para que el pie que acaba de pisar no resbale."""
        pcts = np.linspace(0, 100, MUESTRAS, endpoint=False)
        tabla, alturas, mover, ancla, meta = [], [], np.zeros(2), None, np.zeros(2)
        for p in pcts:
            self.poner_angulos(p)
            self.despejar(p)
            alturas.append(min(self.pie(l)[:, 2].min() for l in LADOS) - self.radio)
            # el pie que acaba de pisar no resbala; en el doble apoyo el que va a
            # despegar se puede arrastrar un poco
            en_apoyo = [l for l in LADOS if self.fase_pierna(p, l) < self.m["despegue"][l]]
            lado = min(en_apoyo, key=lambda l: self.fase_pierna(p, l)) if en_apoyo else ancla
            punto = self.pie(lado)[:, :2].mean(axis=0)
            if lado != ancla:
                ancla, meta = lado, punto + mover
            mover = meta - punto
            tabla.append(self.data.qpos.copy())
            tabla[-1][[self.tx, self.tz]] = self.plano @ mover
        tabla = np.array(tabla)
        # desplazamiento de un ciclo completo (la misma cuenta una vuelta más)
        self.poner_angulos(0.0)
        por_ciclo = meta - self.pie(ancla)[:, :2].mean(axis=0) - (self.plano_inv() @ tabla[0, [self.tx, self.tz]])
        # lo lateral que no vuelve a cero se reparte en el ciclo (sin deriva)
        lateral = self.plano @ np.array([0.0, por_ciclo[1]])
        tabla[:, [self.tx, self.tz]] -= np.outer(pcts / 100, lateral)
        for fila, z in zip(tabla, alturas):
            fila[self.ty] = self.altura(fila, z)
        self.por_ciclo = (self.plano @ np.array([por_ciclo[0], 0.0]))[0]
        self.tabla, self.pcts = tabla, pcts
        return por_ciclo[0] / self.m["zancada"]

    def plano_inv(self):
        return np.column_stack([self.eje_x[:2], self.eje_z[:2]])

    def poner(self, t, cinta=False):
        ciclos = t / self.m["zancada"]
        p = (ciclos % 1) * 100
        i = int(p / 100 * MUESTRAS) % MUESTRAS
        d = self.data
        d.qpos[:] = self.tabla[i]
        tx0 = d.qpos[self.tx]
        if cinta:
            d.qpos[self.tx] = 0.0
        else:
            d.qpos[self.tx] += (int(ciclos) % TRAMO) * self.por_ciclo - TRAMO * self.por_ciclo / 2
        d.qpos[self.ty] -= (d.qpos[self.tx] - tx0) * self.eje_x[2] / self.eje_y[2]
        if self.exo is not None:
            for lado, a in self.exo.items():
                d.ctrl[a] = np.interp(self.fase_pierna(p, lado), self.pct, self.torque, period=100)
        mujoco.mj_forward(self.model, d)
        return p


    def fuerza_suelo(self, p, lado):
        """Fuerza del suelo (N) y centro de presión (m) en el mundo, o None en oscilación.

        Las fuerzas medidas (N/kg) se escalan a la masa del modelo; el centro de
        presión se ubica respecto al centro del tobillo del modelo (adelante) y
        a la línea media del pie (hacia afuera)."""
        f = self.m["fuerzas"]
        if f is None:
            return None
        pl = self.fase_pierna(p, lado)
        val = lambda k: np.interp(pl, self.pct, np.nan_to_num(f[f"{k}_{lado}"]), period=100)
        if pl >= self.m["despegue"][lado] or val("fuerza_arriba") <= 0:
            return None
        afuera = -1.0 if lado == "r" else 1.0           # en el mundo, la derecha es -y
        d = self.data
        cop = np.array([d.xanchor[self.tobillo[lado]][0] + val("cop_adelante"),
                        self.pie(lado)[:, 1].mean() + afuera * val("cop_afuera"), 0.0])
        # el pie del modelo no es el del niño: el centro de presión se mantiene
        # dentro de la huella (solo pasa al final del apoyo, con poca fuerza)
        huella = self.pie(lado)[:, :2]
        cop[:2] = np.clip(cop[:2], huella.min(0) - self.radio, huella.max(0) + self.radio)
        F = self.masa * np.array([val("fuerza_adelante"), afuera * val("fuerza_afuera"), val("fuerza_arriba")])
        return cop, F


def flechas(visor, fuerzas, peso):
    visor.user_scn.ngeom = 0
    for cop, F in fuerzas:
        g = visor.user_scn.geoms[visor.user_scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([0.9, 0.2, 0.2, 0.9], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, cop, cop + F / peso * ESCALA_FLECHA)
        visor.user_scn.ngeom += 1


def fase(p, despegue):
    if p < despegue:
        return "apoyo"
    return "osc. temprana" if p < (despegue + 100) / 2 else "osc. tardía"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caso", default="sano",
                        choices=["sano", "pc", "pc_exo", "crouch", "crouch_exo", "crouch_exo_mejora"])
    parser.add_argument("--nino", default=NINO_CROUCH,
                        help="niño de Steele 2010 para crouch (C2, C3, C5, C7, C8, C9, C10)")
    parser.add_argument("--lento", type=float, default=1.0, help="factor de cámara lenta")
    parser.add_argument("--cinta", action="store_true", help="sin avanzar, como en cinta")
    args = parser.parse_args()

    m = caso_marcha(args.caso, args.nino)
    model, data = cargar_modelo(m["torque"] is not None)
    marcha = Marcha(model, data, m)
    v = marcha.precalcular()

    print(f"{args.caso}: zancada de {m['zancada']:.2f} s, velocidad resultante {v:.2f} m/s"
          + (f" (medida: {m['velocidad_ref']:.2f} m/s)" if m["velocidad_ref"] else "")
          + (f", x{args.lento:g} más lento" if args.lento != 1 else "") + ". Cierra la ventana para salir.")
    if args.caso.startswith("pc"):
        print("  Cadera y tobillo: Senden (supuesto); rodilla: P4 de Lerner en las dos piernas.")
    if args.caso.startswith("crouch"):
        print(f"  Niño {m['nino']} de Steele 2010 ({m['masa_nino']:.1f} kg): todas las articulaciones de una misma "
              "zancada; flechas rojas = fuerza del suelo en el centro de presión (escalada a la masa del modelo).")

    rodilla = {l: model.jnt_qposadr[model.joint(f"knee_angle_{l}").id] for l in LADOS}
    tobillo = {l: model.jnt_qposadr[model.joint(f"ankle_angle_{l}").id] for l in LADOS}
    with mujoco.viewer.launch_passive(model, data) as visor:
        visor.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        visor.cam.trackbodyid = model.body("pelvis").id
        visor.cam.distance, visor.cam.azimuth, visor.cam.elevation = 2.0, 90.0, -10.0
        inicio = time.time()
        while visor.is_running():
            t = (time.time() - inicio) / args.lento
            p = marcha.poner(t, args.cinta)
            data.time = t
            suelo = {l: marcha.fuerza_suelo(p, l) for l in LADOS}
            flechas(visor, [x for x in suelo.values() if x is not None], marcha.masa * 9.81)
            visor.sync()
            partes = []
            for l in LADOS:
                pl = marcha.fase_pierna(p, l)
                s = (f"{l.upper()} {pl:5.1f}% {fase(pl, m['despegue'][l]):<13} "
                     f"rod {-np.degrees(data.qpos[rodilla[l]]):5.1f}° tob {np.degrees(data.qpos[tobillo[l]]):5.1f}°")
                if marcha.exo is not None:
                    s += f" exo {data.sensor(f'exo_torque_{l}').data[0]:4.2f} N·m"
                if m["fuerzas"] is not None:
                    fz = suelo[l][1][2] / (marcha.masa * 9.81) if suelo[l] is not None else 0.0
                    s += f" suelo {fz:4.2f}×peso"
                partes.append(s)
            print("  " + " | ".join(partes), end="\r")
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()

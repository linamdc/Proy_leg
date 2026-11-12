"""
Exoesqueleto de rodilla (secciones 4 y 5 del documento), común a los modelos:
  - pierna MyoLeg escalada al niño (old_exportar_pierna.py);
  - niño de 6 años convertido de OpenSim, dos piernas (marcha_6y.py).

agregar_exo() pone en un MjSpec las piezas rígidas (motor, barras, valvas,
almohadilla, articulación del tobillo) y el actuador de torque de una pierna.
"""

import mujoco
import numpy as np

# Actuador CubeMars AK60-6 V1.1 KV140 (hoja técnica en cubemars.com): 3 N·m
# continuo, 9 N·m pico, reducción 6:1. Se satura por software a 6 N·m
# (torque pico máximo del requerimiento).
TORQUE_MAX_EXO = 6.0       # N·m
MASA_MOTOR = 0.368         # kg (la tabla técnica dice 368 g; en otra parte de la página, 358 g)
# Plano 2D: Ø79 mm y 39,5 mm de largo total (29 mm de cuerpo + cubo de salida).
DIAMETRO_MOTOR = 0.079     # m
ANCHO_MOTOR = 0.0395       # m
# Inercia del rotor reflejada en la rodilla, J_rotor · N². Va como armature de
# la rodilla, que representa justamente eso; el cuerpo del motor lleva solo su
# masa.
INERCIA_ROTOR = 2.435e-5   # kg·m² (243,5 g·cm²)
REDUCCION = 6
INERCIA_REFLEJADA = INERCIA_ROTOR * REDUCCION**2   # 8,77e-4 kg·m²

# Reparto supuesto del resto de la masa: hasta 0,632 kg para que con el motor
# quede en ≤ 1,0 kg por pierna. El único dato real es el del motor.
MASA_BARRA_MUSLO = 0.12    # kg, barra lateral (aluminio)
MASA_VALVA_MUSLO = 0.15    # kg, valva posterior termoplástica con correa
MASA_BARRA_PIERNA = 0.10
MASA_VALVA_PIERNA = 0.12
MASA_ALMOHADILLA = 0.04    # kg, almohadilla anterior sobre la rótula
MASA_TOBILLO = 0.10        # kg, articulación lateral del tobillo, libre (sin tope), como en Lerner
DIAMETRO_TOBILLO = 0.030   # m
ANCHO_TOBILLO = 0.015      # m
MASA_TOTAL = (MASA_MOTOR + MASA_BARRA_MUSLO + MASA_VALVA_MUSLO + MASA_BARRA_PIERNA
              + MASA_VALVA_PIERNA + MASA_ALMOHADILLA + MASA_TOBILLO)

# Valvas de 8 cm de alto, centradas entre el 55 % y el 70 % de cada segmento
# medido desde la rodilla (recomendación de la caracterización del segmento).
ALTO_VALVA = 0.08          # m
POSICION_VALVA = 0.625     # fracción del segmento desde la rodilla
ESPESOR_VALVA = 0.004      # m
# Fracción de cada segmento que cubre la barra lateral, desde la rodilla. La
# de la pierna llega hasta el tobillo, donde se une a la articulación del tobillo.
LARGO_BARRA = {"muslo": 0.75, "pierna": 1.0}
AREA_ALMOHADILLA = (0.08, 0.05)  # m (mediolateral, proximal-distal): 40 cm²

TOPE_FLEXION_EXO = 100.0   # ° de flexión, tope mecánico (rango 0–100°)

# El color va en rgba y no solo en el material: las clases por defecto de los
# modelos fijan rgba y eso taparía el color del material.
COLORES = {"exo_motor": [0.15, 0.15, 0.17, 1], "exo_aluminio": [0.72, 0.74, 0.78, 1],
           "exo_valva": [0.16, 0.47, 0.84, 0.9],
           "exo_concha": [0.80, 0.88, 0.97, 1], "exo_correa": [0.18, 0.19, 0.22, 1],
           "exo_acento": [0.95, 0.55, 0.10, 1]}

# ----------------------------------------------------------------------------
# Apariencia (solo visual, sin masa)
# ----------------------------------------------------------------------------
# Las piezas con masa son formas simples (barras, cajas, cilindros) y quedan en
# GRUPO_MASA, que el visor oculta por defecto. Lo que se ve son mallas curvas
# sin masa: conchas que envuelven el muslo y la pierna, correas, barras planas
# a los dos lados, carcasa del motor y bisagras. Así la masa y la inercia del
# exo no cambian y se ve como un dispositivo real.
VISUAL = True
GRUPO_MASA = 3
# Concha: cubre atrás y a los lados del segmento (ángulo centrado atrás), desde
# y hasta estas fracciones del segmento medidas desde la rodilla. El radio va
# de distal a proximal multiplicado por estos factores (el muslo es más ancho
# arriba; la pierna, en la pantorrilla).
CONCHA = {"muslo": dict(desde=0.16, hasta=0.86, angulo=230, r_cerca=1.05, r_lejos=1.30),
          "pierna": dict(desde=0.14, hasta=0.80, angulo=220, r_cerca=1.12, r_lejos=0.82)}
ESPESOR_CONCHA = 0.004     # m
HOLGURA_CONCHA = 0.006     # m entre la superficie del segmento y la concha
CORREAS = (0.40, 0.72)     # fracciones del segmento con correa anterior
ALTO_CORREA = 0.028        # m

# ----------------------------------------------------------------------------
# Sensores del dispositivo (sección 5 del documento). Todos van en el
# dispositivo o sobre la piel, nunca dentro del cuerpo:
#   - encoder de 14 bits del AK60-6: dentro del motor, coaxial con la rodilla;
#     mide el giro de la pierna respecto al muslo (ángulo y velocidad);
#   - IMU (acelerómetro + giroscopio): en la cara externa de cada concha,
#     anterolateral, a POSICION_IMU del segmento;
#   - FSR de la plantilla: bajo la planta del pie, uno en cada punto de apoyo
#     (talón, antepié y dedos), si se pasan los puntos de la planta;
#   - torque del motor: lo estima el driver por la corriente (exo_torque).
# ----------------------------------------------------------------------------
POSICION_IMU = {"muslo": 0.55, "pierna": 0.50}
TAMANO_IMU = (0.012, 0.004, 0.009)     # m, medias medidas (ancho, espesor, alto)
TAMANO_FSR = (0.012, 0.012, 0.006)     # m, medias medidas; el alto abarca el punto de contacto
COLOR_SENSOR = [0.10, 0.80, 0.30, 1]


def agregar_materiales(spec):
    spec.add_material(name="exo_motor", rgba=COLORES["exo_motor"], specular=0.4, shininess=0.5)
    spec.add_material(name="exo_aluminio", rgba=COLORES["exo_aluminio"], specular=0.6, shininess=0.7)
    spec.add_material(name="exo_valva", rgba=COLORES["exo_valva"], specular=0.2)
    spec.add_material(name="exo_concha", rgba=COLORES["exo_concha"], specular=0.35, shininess=0.6)
    spec.add_material(name="exo_correa", rgba=COLORES["exo_correa"], specular=0.05)
    spec.add_material(name="exo_acento", rgba=COLORES["exo_acento"], specular=0.4, shininess=0.5)


def _malla_concha(base, eje, adelante, h0, h1, r0, r1, espesor, centro, abertura, n_ang=28, n_alt=10):
    """Vértices (mundo) y caras de una concha cerrada: un trozo de tubo de
    `abertura` grados centrado en `centro` (0 = adelante, 180 = atrás), entre
    las alturas h0 y h1 a lo largo de `eje` desde `base`, con radio interior que
    va de r0 a r1 y el espesor dado. Cerrada (caras interior, exterior, bordes y
    extremos) para que MuJoCo la acepte como malla."""
    perp = np.cross(eje, adelante)
    angs = np.radians(centro + np.linspace(-abertura / 2, abertura / 2, n_ang))
    alts = np.linspace(0, 1, n_alt)
    v = []
    for capa in (0, 1):                      # 0 = interior, 1 = exterior
        for a in alts:
            r = r0 + a * (r1 - r0) + capa * espesor
            c = base + (h0 + a * (h1 - h0)) * eje
            for t in angs:
                v.append(c + r * (np.cos(t) * adelante + np.sin(t) * perp))
    idx = lambda capa, i, j: capa * n_alt * n_ang + i * n_ang + j
    f = []
    for i in range(n_alt - 1):
        for j in range(n_ang - 1):
            a, b, c, d = idx(1, i, j), idx(1, i, j + 1), idx(1, i + 1, j + 1), idx(1, i + 1, j)
            f += [(a, b, c), (a, c, d)]                       # exterior
            a, b, c, d = idx(0, i, j), idx(0, i, j + 1), idx(0, i + 1, j + 1), idx(0, i + 1, j)
            f += [(a, c, b), (a, d, c)]                       # interior
    for i in (0, n_alt - 1):                                  # bordes de abajo y de arriba
        for j in range(n_ang - 1):
            a, b, c, d = idx(0, i, j), idx(0, i, j + 1), idx(1, i, j + 1), idx(1, i, j)
            f += [(a, b, c), (a, c, d)] if i == 0 else [(a, c, b), (a, d, c)]
    for j in (0, n_ang - 1):                                  # extremos laterales
        for i in range(n_alt - 1):
            a, b, c, d = idx(0, i, j), idx(0, i + 1, j), idx(1, i + 1, j), idx(1, i, j)
            f += [(a, b, c), (a, c, d)] if j == n_ang - 1 else [(a, c, b), (a, d, c)]
    return np.array(v), np.array(f)


def _agregar_malla(spec, cuerpo, nombre, verts_locales, caras, material):
    malla = spec.add_mesh(name=f"{nombre}_malla")
    malla.uservert = verts_locales.flatten()
    malla.userface = caras.flatten()
    # inercia de "cáscara": no exige un volumen bien definido (la masa es 0)
    malla.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    cuerpo.add_geom(name=nombre, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=malla.name, mass=0,
                    contype=0, conaffinity=0, group=1, material=material, rgba=COLORES[material])


def agregar_exo(spec, model, data, *, femur, tibia, talus, rodilla, lateral, adelante,
                radio_muslo, radio_pierna, junta_rodilla, signo_flexion,
                nombre_muslo="exo_muslo", nombre_pierna="exo_pierna",
                nombre_actuador="exo_rodilla", nombre_sensor="exo_torque",
                sufijo="", planta=None):
    """Exoesqueleto de rodilla de una pierna: partes rígidas pegadas al muslo y
    a la pierna (correas perfectamente ajustadas) y un actuador de torque.

    Las piezas se ubican con la pierna extendida (pose de model/data): rodilla,
    lateral y adelante están en el mundo, y cada pieza se pasa al marco del
    fémur o de la tibia. Cada grupo va en su propio cuerpo sin articulación,
    hijo del hueso, para que MuJoCo calcule su masa e inercia a partir de las
    geometrías y las sume al segmento.

    signo_flexion: +1 si la flexión de la rodilla es positiva en el modelo
    (MyoLeg), -1 si es negativa (OpenSim Gait2354). El actuador se arma para
    que ctrl POSITIVO sea torque EXTENSOR en los dos casos.

    Sensores (ver arriba): encoder_rodilla{sufijo}_pos/_vel, imu_muslo{sufijo}
    e imu_pierna{sufijo} (_acc, _gyro) y, si se pasa planta = lista de
    (cuerpo, nombre, punto de la planta en el mundo), un FSR fsr_{nombre}{sufijo}
    (sensor de contacto) en cada punto.
    """
    unit = lambda v: v / np.linalg.norm(v)
    cadera = data.xpos[model.body(femur).id]
    tobillo = data.xpos[model.body(talus).id]

    def quat(R_hueso, x, y):
        """Orientación local de una caja con ejes x, y (mundo) y z = x × y, para
        que la matriz sea siempre una rotación (det = +1)."""
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R_hueso.T @ np.column_stack([x, y, np.cross(x, y)])).flatten())
        return q

    # Con VISUAL, las piezas con masa quedan ocultas (GRUPO_MASA) y se ven las
    # mallas de _visual_segmento.
    estilo = lambda material: dict(contype=0, conaffinity=0, group=GRUPO_MASA if VISUAL else 1,
                                   material=material, rgba=COLORES[material])

    def caja(cuerpo, nombre, R, local, p0, p1, ancho, espesor, frente, material):
        """Barra plana sin masa de p0 a p1 (mundo): espesor hacia los lados y
        ancho hacia `frente`."""
        largo_b = np.linalg.norm(p1 - p0)
        z = (p1 - p0) / largo_b
        y = unit(frente - (frente @ z) * z)
        cuerpo.add_geom(name=nombre, type=mujoco.mjtGeom.mjGEOM_BOX, mass=0,
                        size=[espesor / 2, ancho / 2, largo_b / 2], pos=local((p0 + p1) / 2),
                        quat=quat(R, np.cross(y, z), y), contype=0, conaffinity=0, group=1,
                        material=material, rgba=COLORES[material])

    def cilindro(cuerpo, nombre, local, p0, p1, radio_c, material):
        cuerpo.add_geom(name=nombre, type=mujoco.mjtGeom.mjGEOM_CYLINDER, mass=0, size=[radio_c, 0, 0],
                        fromto=np.concatenate([local(p0), local(p1)]), contype=0, conaffinity=0,
                        group=1, material=material, rgba=COLORES[material])

    def _visual_segmento(segmento, cuerpo, nombre, R, local, eje, largo, anterior, radio):
        cfg = CONCHA[segmento]
        h0, h1 = cfg["desde"] * largo, cfg["hasta"] * largo
        rc, rl = radio * cfg["r_cerca"] + HOLGURA_CONCHA, radio * cfg["r_lejos"] + HOLGURA_CONCHA
        r_en = lambda h: rc + (np.clip(h, h0, h1) - h0) / (h1 - h0) * (rl - rc)
        # concha que envuelve atrás y los lados
        v, f = _malla_concha(rodilla, eje, anterior, h0, h1, rc, rl, ESPESOR_CONCHA, 180, cfg["angulo"])
        _agregar_malla(spec, cuerpo, f"{nombre}_concha", np.array([local(p) for p in v]), f, "exo_concha")
        # correas anteriores que cierran la concha
        for k, frac in enumerate(CORREAS):
            h = frac * largo
            r = r_en(h) + ESPESOR_CONCHA + 0.001
            v, f = _malla_concha(rodilla, eje, anterior, h - ALTO_CORREA / 2, h + ALTO_CORREA / 2, r, r,
                                 0.003, 0, 360 - cfg["angulo"] + 24)
            _agregar_malla(spec, cuerpo, f"{nombre}_correa{k + 1}", np.array([local(p) for p in v]), f, "exo_correa")
        # barras planas lateral y medial, por fuera de la concha
        h_fin = max(LARGO_BARRA[segmento], cfg["hasta"]) * largo
        separa = ESPESOR_CONCHA + 0.004
        for lado, s in (("lat", 1.0), ("med", -1.0)):
            p0 = rodilla + s * (r_en(0) + separa) * lateral
            p1 = rodilla + h_fin * eje + s * (r_en(h_fin) + separa) * lateral
            caja(cuerpo, f"{nombre}_barra_{lado}", R, local, p0, p1, 0.014, 0.004, anterior, "exo_aluminio")
        if segmento == "muslo":
            # carcasa del motor (lateral) con anillo y eje de salida, y bisagra medial
            r0 = rodilla + (radio + 0.005) * lateral
            r1 = r0 + ANCHO_MOTOR * lateral
            cilindro(cuerpo, f"{nombre}_carcasa", local, r0, r1, DIAMETRO_MOTOR / 2, "exo_motor")
            cilindro(cuerpo, f"{nombre}_anillo", local, r1 - 0.006 * lateral, r1 + 0.001 * lateral,
                     DIAMETRO_MOTOR / 2 + 0.0015, "exo_acento")
            cilindro(cuerpo, f"{nombre}_eje", local, r1, r1 + 0.007 * lateral, 0.013, "exo_aluminio")
            m0 = rodilla - (r_en(0) + separa) * lateral
            cilindro(cuerpo, f"{nombre}_bisagra_med", local, m0, m0 - 0.009 * lateral, 0.02, "exo_aluminio")
            # rodillera curva por encima de la rótula
            ancho_r, alto_r = AREA_ALMOHADILLA
            v, f = _malla_concha(rodilla, eje, anterior, 0.02, 0.02 + alto_r, radio + HOLGURA_CONCHA,
                                 radio + HOLGURA_CONCHA, 0.008, 0, 110)
            _agregar_malla(spec, cuerpo, f"{nombre}_rodillera", np.array([local(p) for p in v]), f, "exo_valva")
        else:
            # bisagras del tobillo (libres) a los dos lados
            for lado, s in (("lat", 1.0), ("med", -1.0)):
                c0 = rodilla + largo * eje + s * (r_en(largo) + separa) * lateral
                cilindro(cuerpo, f"{nombre}_tobillo_{lado}", local, c0, c0 + s * 0.010 * lateral, 0.016, "exo_motor")

    for segmento, hueso, distal, radio, m_barra, m_valva, nombre in (
            ("muslo", femur, cadera, radio_muslo, MASA_BARRA_MUSLO, MASA_VALVA_MUSLO, nombre_muslo),
            ("pierna", tibia, tobillo, radio_pierna, MASA_BARRA_PIERNA, MASA_VALVA_PIERNA, nombre_pierna)):
        b = model.body(hueso).id
        eje = unit(distal - rodilla)
        largo = np.linalg.norm(distal - rodilla)
        anterior = unit(adelante - (adelante @ eje) * eje)
        R, x = data.xmat[b].reshape(3, 3), data.xpos[b]
        local = lambda p: R.T @ (p - x)
        cuerpo = spec.body(hueso).add_body(name=nombre)

        d_barra = radio + 0.012
        cuerpo.add_geom(name=f"{nombre}_barra", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0],
                        fromto=np.concatenate([local(rodilla + d_barra * lateral),
                                               local(rodilla + d_barra * lateral + LARGO_BARRA[segmento] * largo * eje)]),
                        mass=m_barra, **estilo("exo_aluminio"))
        # Valva posterior: lámina que cubre la mitad de atrás del segmento.
        centro = rodilla + POSICION_VALVA * largo * eje - (radio + ESPESOR_VALVA / 2) * anterior
        cuerpo.add_geom(name=f"{nombre}_valva", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[radio, ESPESOR_VALVA / 2, ALTO_VALVA / 2], pos=local(centro),
                        quat=quat(R, lateral, anterior),
                        mass=m_valva, **estilo("exo_valva"))
        if segmento == "muslo":
            # Motor coaxial con la rodilla por el lado lateral; la carcasa va
            # en la barra del muslo.
            r0 = rodilla + (radio + 0.005) * lateral
            cuerpo.add_geom(name=f"{nombre}_motor", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[DIAMETRO_MOTOR / 2, 0, 0],
                            fromto=np.concatenate([local(r0), local(r0 + ANCHO_MOTOR * lateral)]),
                            mass=MASA_MOTOR, **estilo("exo_motor"))
            # Almohadilla anterior por encima de la rótula: recibe la suma de
            # las fuerzas de las dos valvas (sistema de tres puntos).
            ancho, alto = AREA_ALMOHADILLA
            centro = rodilla + (0.02 + alto / 2) * eje + (radio + 0.006) * anterior
            cuerpo.add_geom(name=f"{nombre}_almohadilla", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[ancho / 2, 0.006, alto / 2], pos=local(centro),
                            quat=quat(R, lateral, anterior),
                            mass=MASA_ALMOHADILLA, **estilo("exo_valva"))
        else:
            # Articulación lateral del tobillo, al final de la barra de la
            # pierna. Es libre, como en el exoesqueleto de Lerner, para que los
            # resultados sean comparables: no limita el tobillo y solo aporta
            # su masa, pegada a la tibia. El estribo del pie no se modela.
            t0 = rodilla + largo * eje + d_barra * lateral
            cuerpo.add_geom(name=f"{nombre}_tobillo", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[DIAMETRO_TOBILLO / 2, 0, 0],
                            fromto=np.concatenate([local(t0), local(t0 + ANCHO_TOBILLO * lateral)]),
                            mass=MASA_TOBILLO, **estilo("exo_motor"))
        if VISUAL:
            _visual_segmento(segmento, cuerpo, nombre, R, local, eje, largo, anterior, radio)

        # IMU sobre la cara externa de la concha, anterolateral (fuera del cuerpo)
        cfg = CONCHA[segmento]
        h = POSICION_IMU[segmento] * largo
        a = np.clip((h - cfg["desde"] * largo) / ((cfg["hasta"] - cfg["desde"]) * largo), 0, 1)
        r_piel = radio * (cfg["r_cerca"] + a * (cfg["r_lejos"] - cfg["r_cerca"])) + HOLGURA_CONCHA
        hacia = unit(lateral + anterior)
        imu = rodilla + h * eje + (r_piel + ESPESOR_CONCHA + TAMANO_IMU[1]) * hacia
        nombre_imu = f"imu_{segmento}{sufijo}"
        cuerpo.add_site(name=nombre_imu, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TAMANO_IMU),
                        pos=local(imu), quat=quat(R, unit(np.cross(hacia, eje)), hacia), group=1, rgba=COLOR_SENSOR)
        for tipo, sub in ((mujoco.mjtSensor.mjSENS_ACCELEROMETER, "acc"), (mujoco.mjtSensor.mjSENS_GYRO, "gyro")):
            spec.add_sensor(name=f"{nombre_imu}_{sub}", type=tipo, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=nombre_imu)
        if segmento == "muslo":
            # encoder dentro del motor, en su eje (coaxial con la rodilla)
            centro_motor = rodilla + (radio + 0.005 + ANCHO_MOTOR / 2) * lateral
            cuerpo.add_site(name=f"encoder_rodilla{sufijo}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[0.012, ANCHO_MOTOR / 2 + 0.002, 0], pos=local(centro_motor),
                            quat=quat(R, anterior, unit(np.cross(lateral, anterior))), group=GRUPO_MASA,
                            rgba=COLOR_SENSOR)

    junta = spec.joint(junta_rodilla)
    tope = np.radians(TOPE_FLEXION_EXO)
    if signo_flexion > 0:
        junta.range = [junta.range[0], min(junta.range[1], tope)]
    else:
        junta.range = [max(junta.range[0], -tope), junta.range[1]]
    junta.armature += INERCIA_REFLEJADA

    # ctrl positivo = torque EXTENSOR: gear opuesto al signo de la flexión.
    spec.add_actuator(name=nombre_actuador, target=junta_rodilla, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gear=[-signo_flexion, 0, 0, 0, 0, 0], ctrllimited=True,
                      ctrlrange=[-TORQUE_MAX_EXO, TORQUE_MAX_EXO], forcelimited=True,
                      forcerange=[-TORQUE_MAX_EXO, TORQUE_MAX_EXO])
    spec.add_sensor(name=nombre_sensor, type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
                    objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname=nombre_actuador)
    # encoder: el giro relativo pierna-muslo es el ángulo de la rodilla
    for tipo, sub in ((mujoco.mjtSensor.mjSENS_JOINTPOS, "pos"), (mujoco.mjtSensor.mjSENS_JOINTVEL, "vel")):
        spec.add_sensor(name=f"encoder_rodilla{sufijo}_{sub}", type=tipo,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=junta_rodilla)

    if planta is not None:
        # FSR de la plantilla: uno bajo cada punto de apoyo de la planta del
        # pie. El sitio es una caja delgada que abarca el punto donde el pie
        # toca el suelo; el sensor de contacto suma las fuerzas normales dentro
        # de ella. Cada FSR va en el cuerpo que apoya (los de los dedos se
        # mueven con los dedos).
        for pie, nombre, p in planta:
            b = model.body(pie).id
            R, x = data.xmat[b].reshape(3, 3), data.xpos[b]
            sitio = f"fsr_{nombre}{sufijo}"
            spec.body(pie).add_site(name=sitio, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(TAMANO_FSR),
                                    pos=R.T @ (np.asarray(p) - x), group=1, rgba=COLOR_SENSOR)
            spec.add_sensor(name=sitio, type=mujoco.mjtSensor.mjSENS_TOUCH,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=sitio)


def direcciones(model, data, junta_rodilla, femur, pelvis, calcn, toes, tibia):
    """Rodilla y direcciones lateral y anterior (mundo) de una pierna."""
    rodilla = data.xpos[model.body(tibia).id].copy()
    arriba = np.array([0, 0, 1.0])
    horizontal = lambda v: (v - v @ arriba * arriba) / np.linalg.norm(v - v @ arriba * arriba)
    # Lateral = eje de flexión de la rodilla, con el signo que apunta hacia
    # afuera (del centro de la pelvis hacia la cadera de ese lado).
    lateral = horizontal(data.xaxis[model.joint(junta_rodilla).id])
    if lateral @ (data.xpos[model.body(femur).id] - data.xpos[model.body(pelvis).id]) < 0:
        lateral = -lateral
    adelante = horizontal(np.cross(lateral, arriba))
    if adelante @ (data.xpos[model.body(toes).id] - data.xpos[model.body(calcn).id]) < 0:
        adelante = -adelante
    return rodilla, lateral, adelante, arriba

"""
Exporta la pierna derecha del niño (modelo_nino.py) como XML de MuJoCo independiente.

Abrir los XML en el visor de MuJoCo. Copiar y pegar en la terminal, desde la
carpeta del repo (Proy_leg). En el visor, el keyframe "popliteo" se carga desde
el panel derecho: Simulation > Key > 1, botón "Load key".

  Niño sano:
    python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_sano.xml

  PC sin acortamiento:
    python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_pc_sin_acortamiento.xml

  PC moderado:
    python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_pc_moderado.xml

  PC severo:
    python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_pc_severo.xml

  Con exoesqueleto (el motor se prueba en el panel derecho: Control > exo_rodilla):
    python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_pc_sin_acortamiento_exo.xml
    python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_pc_moderado_exo.xml
    python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_pc_severo_exo.xml

  Volver a exportar (después de cambiar modelo_nino.py o el exoesqueleto):
    python scripts/exportar_pierna.py

Genera en modelos/pierna_nino/ un XML por caso (sano y los escenarios de PC),
con las mallas de los huesos copiadas en meshes/ y el escenario en escena.xml.
Los XML no necesitan myo_sim para abrirse: basta con mujoco.

Qué queda en el XML:
  - Pelvis fija en el espacio (sin la articulación libre) y pierna derecha
    completa: cadera (3 GdL), rodilla con sus acoples, tobillo, subtalar y
    metatarsofalángica, con los 40 músculos de esa pierna.
  - La pierna cuelga con el pie HOLGURA_PIE por encima del piso.
  - Dos IMU (acelerómetro + giroscopio) a la mitad del muslo y de la pierna,
    sobre el eje del hueso, donde irían las valvas del exoesqueleto.
  - Sensores de ángulo de cadera, rodilla y tobillo, y velocidad de rodilla.
  - Cámaras lateral, frontal e isométrica que siguen a la tibia.
  - Keyframes "home" (pierna extendida) y "popliteo" (cadera a 90° y rodilla
    en el ángulo poplíteo de ese caso).
  - Versiones _exo (solo PC): exoesqueleto de rodilla con motor AK60-6,
    barras, valvas, almohadilla, actuador de torque y tope mecánico (ver
    parámetros del exoesqueleto abajo y _agregar_exo).

NO EDITAR los XML a mano: se sobrescriben. Para cambiar algo de la pierna se
cambia modelo_nino.py o este script y se vuelve a exportar.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import modelo_nino as mn

CARPETA = Path(__file__).resolve().parents[1] / "modelos" / "pierna_nino"
MALLAS_ORIGEN = Path(mn.myo_sim.__file__).parent / "models"

HOLGURA_PIE = 0.05   # m entre el punto más bajo del pie y el piso

# (version, escenario, con exoesqueleto). El exoesqueleto es para el niño con
# PC, así que el sano solo se exporta sin él.
CASOS = [("sano", None, False)] + [("pc", e, exo) for e in mn.ESCENARIOS_PC for exo in (False, True)]

# ----------------------------------------------------------------------------
# Exoesqueleto (sección 4 y 5 del documento)
# ----------------------------------------------------------------------------

# Actuador CubeMars AK60-6 V1.1: 3 N·m continuo, 9 N·m pico, 368 g [6].
# Se satura por software a 6 N·m (torque pico máximo del requerimiento).
TORQUE_MAX_EXO = 6.0       # N·m
MASA_MOTOR = 0.368         # kg
# Medidas aproximadas del motor: verificar con la hoja de datos.
DIAMETRO_MOTOR = 0.080     # m
ANCHO_MOTOR = 0.040        # m
# Inercia reflejada del rotor en la rodilla (J_rotor · N², kg·m²): se suma a
# la armadura de knee_angle_r. Pendiente: J_rotor de la hoja de datos.
INERCIA_REFLEJADA = 0.0

# Reparto supuesto del resto de la masa; con el motor suma 0,97 kg (≤ 1,0 kg).
MASA_BARRA_MUSLO = 0.12    # kg, barra lateral (aluminio)
MASA_VALVA_MUSLO = 0.18    # kg, valva posterior termoplástica con correa
MASA_BARRA_PIERNA = 0.10
MASA_VALVA_PIERNA = 0.15
MASA_ALMOHADILLA = 0.05    # kg, almohadilla anterior sobre la rótula

# Valvas de 8 cm de alto, centradas entre el 55 % y el 70 % de cada segmento
# medido desde la rodilla (recomendación de la caracterización del segmento).
ALTO_VALVA = 0.08          # m
POSICION_VALVA = 0.625     # fracción del segmento desde la rodilla
ESPESOR_VALVA = 0.004      # m
LARGO_BARRA = 0.75         # fracción del segmento que cubre la barra lateral
AREA_ALMOHADILLA = (0.08, 0.05)  # m (mediolateral, proximal-distal): 40 cm²

TOPE_FLEXION_EXO = 100.0   # ° de flexión, tope mecánico (rango 0–100°)

CUERPOS_PIERNA = ["pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "patella_r"]
MALLAS_PIERNA = ["r_pelvis", "r_femur", "r_tibia", "r_fibula", "r_talus", "r_foot", "r_bofoot", "r_patella"]


def nombre_caso(version, escenario, exo=False):
    nombre = "sano" if version == "sano" else f"pc_{escenario}"
    return nombre + "_exo" if exo else nombre


# ----------------------------------------------------------------------------
# Recorte: solo la pierna derecha
# ----------------------------------------------------------------------------

def _recortar(spec):
    for act in list(spec.actuators):
        if act.name.endswith("_l"):
            spec.delete(act)
    for tendon in list(spec.tendons):
        if tendon.name.endswith("_l_tendon"):
            spec.delete(tendon)
    for eq in list(spec.equalities):
        if eq.name1.endswith("_l"):
            spec.delete(eq)
    for sensor in list(spec.sensors):
        if sensor.name.startswith("l_"):
            spec.delete(sensor)
    # Todos los pares de contacto de MyoLeg son entre una pierna y la otra.
    for pair in list(spec.pairs):
        spec.delete(pair)

    # Tronco, cabeza y pierna izquierda.
    spec.delete(spec.body("sacrum"))
    spec.delete(spec.body("femur_l"))
    # Mitad izquierda de la pelvis: malla, colisión, envolventes y sitios.
    pelvis = spec.body("pelvis")
    for geom in list(pelvis.geoms):
        if geom.name.endswith("_l"):
            spec.delete(geom)
    for site in list(pelvis.sites):
        if site.name.endswith("_l"):
            spec.delete(site)

    # Pelvis fija: sin articulación libre.
    spec.delete(spec.joint("root"))

    # Escenario de MyoSuite y cámaras del cuerpo completo: los reemplaza escena.xml.
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    for cam in list(spec.cameras):
        spec.delete(cam)
    for light in list(spec.worldbody.lights):
        spec.delete(light)

    for mesh in list(spec.meshes):
        if mesh.name not in MALLAS_PIERNA:
            spec.delete(mesh)
    usados = {"myolegs_mat", "myolegs_matskin"}
    for mat in list(spec.materials):
        if mat.name not in usados:
            spec.delete(mat)
    for tex in list(spec.textures):
        if tex.name != "myolegs_tex":
            spec.delete(tex)


def _punto_mas_bajo(model, data):
    zmin = np.inf
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            zmin = min(zmin, v[:, 2].min())
    return zmin


# ----------------------------------------------------------------------------
# Lo que se agrega: IMU, sensores, cámaras y keyframes
# ----------------------------------------------------------------------------

def _agregar_instrumentos(spec):
    # IMU a la mitad de cada segmento. La posición de un cuerpo hijo en el
    # marco del padre es el vector de la articulación proximal a la distal.
    for nombre, cuerpo, distal in (("imu_muslo", "femur_r", "tibia_r"),
                                   ("imu_pierna", "tibia_r", "talus_r")):
        spec.body(cuerpo).add_site(name=nombre, pos=0.5 * spec.body(distal).pos,
                                   type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.008, 0.012],
                                   rgba=[0.95, 0.55, 0.10, 1], group=2)
        for tipo, sufijo in ((mujoco.mjtSensor.mjSENS_ACCELEROMETER, "acc"),
                             (mujoco.mjtSensor.mjSENS_GYRO, "gyro")):
            spec.add_sensor(name=f"{nombre}_{sufijo}", type=tipo,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=nombre)

    for junta in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r"):
        spec.add_sensor(name=f"{junta}_pos", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=junta)
    spec.add_sensor(name="knee_angle_r_vel", type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                    objtype=mujoco.mjtObj.mjOBJ_JOINT, objname="knee_angle_r")


def _direcciones(model, data):
    """Rodilla y direcciones lateral, anterior y vertical en el mundo."""
    rodilla = data.xpos[model.body("tibia_r").id].copy()
    arriba = np.array([0, 0, 1.0])
    unit = lambda v: (v - v @ arriba * arriba) / np.linalg.norm(v - v @ arriba * arriba)
    # Lateral = eje de flexión de la rodilla, con el signo que apunta hacia
    # afuera (del centro de la pelvis hacia la cadera derecha).
    lateral = unit(data.xaxis[model.joint("knee_angle_r").id])
    if lateral @ (data.xpos[model.body("femur_r").id] - data.xpos[model.body("pelvis").id]) < 0:
        lateral = -lateral
    adelante = unit(np.cross(lateral, arriba))
    if adelante @ (data.xpos[model.body("toes_r").id] - data.xpos[model.body("calcn_r").id]) < 0:
        adelante = -adelante
    return rodilla, lateral, adelante, arriba


def _agregar_camaras(spec, model, data):
    rodilla, lateral, adelante, arriba = _direcciones(model, data)
    for nombre, pos in (("lateral", rodilla + 1.4 * lateral),
                        ("frontal", rodilla + 1.4 * adelante + 0.1 * arriba),
                        ("isometrica", rodilla + 1.0 * (lateral + adelante) + 0.6 * arriba)):
        spec.worldbody.add_camera(name=nombre, pos=pos, fovy=45,
                                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                                  targetbody="tibia_r")


def _agregar_exo(spec, model, data):
    """Exoesqueleto de rodilla: partes rígidas pegadas al muslo y a la pierna
    (correas perfectamente ajustadas) y un actuador de torque en la rodilla.

    Las piezas se ubican con la pierna extendida (pose inicial): se calculan en
    el mundo y se pasan al marco del fémur o de la tibia. Cada grupo va en su
    propio cuerpo sin articulación, hijo del hueso, para que MuJoCo calcule su
    masa e inercia a partir de las geometrías y las sume al segmento.
    """
    rodilla, lateral, adelante, _ = _direcciones(model, data)
    unit = lambda v: v / np.linalg.norm(v)

    def segmento(hueso, distal, radio_col):
        b = model.body(hueso).id
        p_distal = data.xpos[model.body(distal).id]
        eje = unit(p_distal - rodilla) if hueso == "tibia_r" else unit(data.xpos[b] - rodilla)
        largo = np.linalg.norm((p_distal if hueso == "tibia_r" else data.xpos[b]) - rodilla)
        anterior = unit(adelante - (adelante @ eje) * eje)
        R, x = data.xmat[b].reshape(3, 3), data.xpos[b]
        cuerpo = spec.body(hueso).add_body(name=f"exo_{'muslo' if hueso == 'femur_r' else 'pierna'}")
        radio = model.geom_size[model.geom(radio_col).id][0]
        return cuerpo, eje, largo, anterior, radio, (lambda p: R.T @ (p - x)), R

    def quat(R_hueso, x, y):
        """Orientación local de una caja con ejes x, y (mundo) y z = x × y, para
        que la matriz sea siempre una rotación (det = +1)."""
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R_hueso.T @ np.column_stack([x, y, np.cross(x, y)])).flatten())
        return q

    # El color va en rgba y no solo en el material: la clase por defecto de
    # MyoLeg fija rgba y eso taparía el color del material.
    colores = {"exo_motor": [0.15, 0.15, 0.17, 1], "exo_aluminio": [0.72, 0.74, 0.78, 1],
               "exo_valva": [0.16, 0.47, 0.84, 0.9]}
    estilo = lambda material: dict(contype=0, conaffinity=0, group=1,
                                   material=material, rgba=colores[material])
    for hueso, distal, radio_col, m_barra, m_valva in (
            ("femur_r", "tibia_r", "femur2_col_r", MASA_BARRA_MUSLO, MASA_VALVA_MUSLO),
            ("tibia_r", "talus_r", "tibia1_col_r", MASA_BARRA_PIERNA, MASA_VALVA_PIERNA)):
        cuerpo, eje, largo, anterior, radio, local, R = segmento(hueso, distal, radio_col)
        d_barra = radio + 0.012
        cuerpo.add_geom(name=f"{cuerpo.name}_barra", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.006, 0, 0],
                        fromto=np.concatenate([local(rodilla + d_barra * lateral),
                                               local(rodilla + d_barra * lateral + LARGO_BARRA * largo * eje)]),
                        mass=m_barra, **estilo("exo_aluminio"))
        # Valva posterior: lámina que cubre la mitad de atrás del segmento.
        centro = rodilla + POSICION_VALVA * largo * eje - (radio + ESPESOR_VALVA / 2) * anterior
        cuerpo.add_geom(name=f"{cuerpo.name}_valva", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[radio, ESPESOR_VALVA / 2, ALTO_VALVA / 2], pos=local(centro),
                        quat=quat(R, lateral, anterior),
                        mass=m_valva, **estilo("exo_valva"))
        if hueso == "femur_r":
            # Motor coaxial con la rodilla por el lado lateral; la carcasa va
            # en la barra del muslo.
            r0 = rodilla + (radio + 0.005) * lateral
            cuerpo.add_geom(name="exo_motor", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[DIAMETRO_MOTOR / 2, 0, 0],
                            fromto=np.concatenate([local(r0), local(r0 + ANCHO_MOTOR * lateral)]),
                            mass=MASA_MOTOR, **estilo("exo_motor"))
            # Almohadilla anterior por encima de la rótula: recibe la suma de
            # las fuerzas de las dos valvas (sistema de tres puntos).
            ancho, alto = AREA_ALMOHADILLA
            centro = rodilla + (0.02 + alto / 2) * eje + (radio + 0.006) * anterior
            cuerpo.add_geom(name="exo_almohadilla", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[ancho / 2, 0.006, alto / 2], pos=local(centro),
                            quat=quat(R, lateral, anterior),
                            mass=MASA_ALMOHADILLA, **estilo("exo_valva"))

    spec.add_material(name="exo_motor", rgba=colores["exo_motor"], specular=0.4, shininess=0.5)
    spec.add_material(name="exo_aluminio", rgba=colores["exo_aluminio"], specular=0.6, shininess=0.7)
    spec.add_material(name="exo_valva", rgba=colores["exo_valva"], specular=0.2)

    rodilla_jnt = spec.joint("knee_angle_r")
    rodilla_jnt.range = [rodilla_jnt.range[0], np.radians(TOPE_FLEXION_EXO)]
    rodilla_jnt.armature += INERCIA_REFLEJADA

    # gear = -1: ctrl positivo es torque EXTENSOR (knee_angle_r positivo es flexión).
    spec.add_actuator(name="exo_rodilla", target="knee_angle_r", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gear=[-1, 0, 0, 0, 0, 0], ctrllimited=True,
                      ctrlrange=[-TORQUE_MAX_EXO, TORQUE_MAX_EXO], forcelimited=True,
                      forcerange=[-TORQUE_MAX_EXO, TORQUE_MAX_EXO])
    spec.add_sensor(name="exo_torque", type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
                    objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname="exo_rodilla")


def construir(version, escenario, exo=False):
    spec = mn.spec_nino(version, escenario or mn.ESCENARIO_PC_DEFECTO)
    _recortar(spec)
    _agregar_instrumentos(spec)

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = spec.body("Full Body")
    base.name = "base"
    base.pos = base.pos + [0, 0, HOLGURA_PIE - _punto_mas_bajo(model, data)]

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _agregar_camaras(spec, model, data)
    if exo:
        _agregar_exo(spec, model, data)
        model = spec.compile()
        data = mujoco.MjData(model)

    spec.add_key(name="home", qpos=model.qpos0)
    T_ex = mn.calibracion()["torque_examinador"]
    popliteo = mn.angulo_popliteo(model, T_ex)
    mn._poner_pose(model, data, 90.0, popliteo)
    spec.add_key(name="popliteo", qpos=data.qpos.copy())

    spec.modelname = f"pierna_nino_{nombre_caso(version, escenario, exo)}"
    spec.meshdir = ""
    spec.texturedir = ""
    return spec, popliteo


# ----------------------------------------------------------------------------
# Escritura
# ----------------------------------------------------------------------------

def _texto_exo():
    masa = (MASA_MOTOR + MASA_BARRA_MUSLO + MASA_VALVA_MUSLO + MASA_BARRA_PIERNA
            + MASA_VALVA_PIERNA + MASA_ALMOHADILLA)
    return f"""
     EXOESQUELETO DE RODILLA ({masa:.2f} kg): motor AK60-6 coaxial con la rodilla por el lado
     lateral ({MASA_MOTOR * 1000:.0f} g), barras laterales, valvas posteriores de {ALTO_VALVA * 100:.0f} cm al
     {POSICION_VALVA:.1%} de cada segmento y almohadilla anterior sobre la rótula. Piezas rígidas
     en los cuerpos exo_muslo (hijo de femur_r) y exo_pierna (hijo de tibia_r).
     Actuador "exo_rodilla": torque en knee_angle_r, ctrl en N·m, POSITIVO = EXTENSOR,
     saturado a ±{TORQUE_MAX_EXO:.0f} N·m. Sensor "exo_torque". Tope mecánico de flexión a {TOPE_FLEXION_EXO:.0f}°.
     La curva torque-velocidad, la velocidad de corte y el retraso van en el controlador.
"""


def _encabezado(version, escenario, popliteo, exo=False):
    cal = mn.calibracion()
    if version == "sano":
        patologia = "     Niño sano: sin acortamiento ni contractura."
        extensor = mn.FUERZA_EXTENSORA_SANO
    else:
        esc = mn.ESCENARIOS_PC[escenario]
        patologia = (f"     PC, escenario {escenario}: cuádriceps al {100 * cal['extensores_pc']:.0f} % del sano,\n"
                     f"     isquiotibiales biarticulares acortados {cal['acortamientos'][escenario]:.1%} de L0,\n"
                     f"     contractura en flexión de rodilla de {esc['contractura']:.1f}°.")
        extensor = mn.FUERZA_EXTENSORA_PC
    return f"""<!-- Pierna derecha del niño ({nombre_caso(version, escenario, exo)}), MyoLeg escalado a P4 de Lerner et al. 2017.

     GENERADO por scripts/exportar_pierna.py a partir de scripts/modelo_nino.py.
     NO EDITAR A MANO: se sobrescribe al volver a exportar.

     Tamaño: {mn.ESTATURA_NINO} m y {mn.MASA_NINO} kg (cuerpo entero). Longitudes x {mn.S_L:.3f},
     masas x {mn.S_M:.4f}, inercias x s_M*s_L^2.
     Fuerza: extensor de rodilla {extensor:.2f} N·m/kg, flexor {mn.FUERZA_FLEXORA_SANO:.2f} N·m/kg
     (pico isométrico sentado, cadera a 90°).
{patologia}
     Ángulo poplíteo (cadera a 90°, examinador {cal['torque_examinador']:.2f} N·m): {popliteo:.1f}°.
     La espasticidad no está en el modelo: va en el controlador.

     Pelvis fija y pie a {HOLGURA_PIE * 100:.0f} cm del piso. Músculos con ctrl en [0, 1] (activación).
     Keyframes: "home" (pierna extendida) y "popliteo" (pose final de la prueba).
     Convención: knee_angle_r = 0 es extensión completa y positivo es flexión.
     La rodilla tiene grados de libertad acoplados (traslaciones de tibia y rótula)
     que siguen a knee_angle_r por las igualdades de <equality>; si se pone la
     pose a mano en qpos, hay que calcularlos con esos polinomios.
{_texto_exo() if exo else ""}-->
"""


def _limpiar_xml(xml):
    """Arregla lo que deja spec.to_xml() con el modelo de myo_sim.

    - El modelo se arma uniendo fragmentos, y la clase por defecto de la
      pierna queda envuelta en un <default> sin nombre, que MuJoCo no acepta
      al volver a leer el XML. Se desenvuelve.
    - Las clases del tronco y la cabeza apuntan a materiales que ya no están.
    - <visual> y <statistic> son los del cuerpo completo; los pone escena.xml.
    """
    raiz = ET.fromstring(xml)
    for etiqueta in ("visual", "statistic"):
        for e in raiz.findall(etiqueta):
            raiz.remove(e)
    defaults = raiz.find("default")
    for hijo in list(defaults):
        clase = hijo.get("class")
        if clase is None:
            i = list(defaults).index(hijo)
            defaults.remove(hijo)
            for k, nieto in enumerate(list(hijo)):
                defaults.insert(i + k, nieto)
        elif clase.startswith(("myotorso", "myohead")):
            defaults.remove(hijo)
    ET.SubElement(raiz, "include", file="escena.xml")
    ET.indent(raiz, space="  ")
    return ET.tostring(raiz, encoding="unicode") + "\n"


def exportar(version, escenario, exo=False):
    spec, popliteo = construir(version, escenario, exo)
    spec.compile()
    xml = _limpiar_xml(spec.to_xml())
    archivo = CARPETA / f"{spec.modelname}.xml"
    archivo.write_text(_encabezado(version, escenario, popliteo, exo) + xml, encoding="utf-8")
    return archivo


def main():
    (CARPETA / "meshes").mkdir(parents=True, exist_ok=True)
    for nombre in MALLAS_PIERNA:
        shutil.copy(MALLAS_ORIGEN / "meshes" / f"{nombre}.stl", CARPETA / "meshes")
    for version, escenario, exo in CASOS:
        archivo = exportar(version, escenario, exo)
        model = mujoco.MjModel.from_xml_path(str(archivo))
        musculos = sum(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
        masa_exo = sum(model.body_mass[model.body(b).id] for b in ("exo_muslo", "exo_pierna")) if exo else 0.0
        print(f"{archivo.relative_to(CARPETA.parents[1])}: {model.njnt} articulaciones, "
              f"{musculos} músculos" + (f", exo de {masa_exo:.3f} kg" if exo else ""))


if __name__ == "__main__":
    main()

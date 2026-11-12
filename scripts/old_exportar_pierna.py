"""
Exporta la pierna derecha del niño (old_modelo_nino.py) como XML de MuJoCo independiente.

Abrir los XML en el visor de MuJoCo. Copiar y pegar en la terminal, desde la
carpeta del repo (Proy_leg). En el visor, el keyframe "popliteo" se carga desde
el panel derecho: Simulation > Key > 1, botón "Load key".

  Niño sano:
    python -m mujoco.viewer --mjcf=modelos/old_pierna_nino/pierna_nino_sano.xml

  PC sin acortamiento:
    python -m mujoco.viewer --mjcf=modelos/old_pierna_nino/pierna_nino_pc_sin_acortamiento.xml

  PC moderado:
    python -m mujoco.viewer --mjcf=modelos/old_pierna_nino/pierna_nino_pc_moderado.xml

  PC severo:
    python -m mujoco.viewer --mjcf=modelos/old_pierna_nino/pierna_nino_pc_severo.xml

  Con exoesqueleto (el motor se prueba en el panel derecho: Control > exo_rodilla):
    python -m mujoco.viewer --mjcf=modelos/old_pierna_nino/pierna_nino_pc_sin_acortamiento_exo.xml
    python -m mujoco.viewer --mjcf=modelos/old_pierna_nino/pierna_nino_pc_moderado_exo.xml
    python -m mujoco.viewer --mjcf=modelos/old_pierna_nino/pierna_nino_pc_severo_exo.xml

  Volver a exportar (después de cambiar old_modelo_nino.py o el exoesqueleto):
    python scripts/old_exportar_pierna.py

Genera en modelos/old_pierna_nino/ un XML por caso (sano y los escenarios de PC),
con las mallas de los huesos copiadas en meshes/ y el escenario en escena.xml.
Los XML no necesitan myo_sim para abrirse: basta con mujoco.

Qué queda en el XML:
  - Pelvis fija en el espacio (sin la articulación libre) y pierna derecha
    completa: cadera (3 GdL), rodilla con sus acoples, tobillo, subtalar y
    metatarsofalángica, con los 40 músculos de esa pierna.
  - La pierna cuelga con el pie HOLGURA_PIE por encima del piso.
  - Sensores, siempre sobre la piel o en el dispositivo (nunca dentro del cuerpo):
      sin exo: dos IMU (acelerómetro + giroscopio) apoyados en la piel, en la
      cara anterolateral, a la mitad del muslo y de la pierna;
      con exo: los IMU van sobre las conchas y se agrega el encoder del motor
      (encoder_rodilla_pos / _vel), ver exo.py;
      FSR de la planta: los sensores de contacto r_foot (talón) y r_toes
      (dedos) que trae MyoLeg bajo la planta del pie;
      ángulo de cadera, rodilla y tobillo y velocidad de rodilla del modelo
      (referencia para comparar con los sensores).
  - Cámaras lateral, frontal e isométrica que siguen a la tibia.
  - Keyframes "home" (pierna extendida) y "popliteo" (cadera a 90° y rodilla
    en el ángulo poplíteo de ese caso).
  - Versiones _exo (solo PC): exoesqueleto de rodilla con motor AK60-6,
    barras, valvas, almohadilla, actuador de torque y tope mecánico (ver
    parámetros del exoesqueleto abajo y _agregar_exo).

NO EDITAR los XML a mano: se sobrescriben. Para cambiar algo de la pierna se
cambia old_modelo_nino.py o este script y se vuelve a exportar.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import exo as ex
import old_modelo_nino as mn

CARPETA = Path(__file__).resolve().parents[1] / "modelos" / "old_pierna_nino"
MALLAS_ORIGEN = Path(mn.myo_sim.__file__).parent / "models"

HOLGURA_PIE = 0.05   # m entre el punto más bajo del pie y el piso

# (version, escenario, con exoesqueleto). El exoesqueleto es para el niño con
# PC, así que el sano solo se exporta sin él.
CASOS = [("sano", None, False)] + [("pc", e, exo) for e in mn.ESCENARIOS_PC for exo in (False, True)]

# Los parámetros del exoesqueleto están en exo.py (común con marcha_6y.py).

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

def _agregar_imu_piel(spec, model, data):
    """IMU sobre la PIEL (sin exo): a la mitad de cada segmento, en la cara
    anterolateral, apoyados en la superficie del muslo y de la pierna (radio de
    las cápsulas de colisión de MyoLeg). Con exo, los IMU van sobre las conchas
    del dispositivo (exo.agregar_exo)."""
    rodilla, lateral, adelante, _ = _direcciones(model, data)
    unit = lambda v: v / np.linalg.norm(v)
    for nombre, hueso, proximal, distal, capsula in (
            ("imu_muslo", "femur_r", "femur_r", "tibia_r", "femur2_col_r"),
            ("imu_pierna", "tibia_r", "tibia_r", "talus_r", "tibia1_col_r")):
        b = model.body(hueso).id
        p0, p1 = data.xpos[model.body(proximal).id], data.xpos[model.body(distal).id]
        eje = unit(p1 - p0)
        hacia = unit(lateral + adelante)
        hacia = unit(hacia - (hacia @ eje) * eje)
        espesor = ex.TAMANO_IMU[1]
        punto = (p0 + p1) / 2 + (model.geom_size[model.geom(capsula).id][0] + espesor) * hacia
        R, x = data.xmat[b].reshape(3, 3), data.xpos[b]
        orient = np.column_stack([np.cross(hacia, eje), hacia, eje])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R.T @ orient).flatten())
        spec.body(hueso).add_site(name=nombre, pos=R.T @ (punto - x), quat=q,
                                  type=mujoco.mjtGeom.mjGEOM_BOX, size=list(ex.TAMANO_IMU),
                                  rgba=ex.COLOR_SENSOR, group=1)
        for tipo, sufijo in ((mujoco.mjtSensor.mjSENS_ACCELEROMETER, "acc"),
                             (mujoco.mjtSensor.mjSENS_GYRO, "gyro")):
            spec.add_sensor(name=f"{nombre}_{sufijo}", type=tipo,
                            objtype=mujoco.mjtObj.mjOBJ_SITE, objname=nombre)


def _agregar_instrumentos(spec):
    for junta in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r"):
        spec.add_sensor(name=f"{junta}_pos", type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=junta)
    spec.add_sensor(name="knee_angle_r_vel", type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                    objtype=mujoco.mjtObj.mjOBJ_JOINT, objname="knee_angle_r")


def _direcciones(model, data):
    """Rodilla y direcciones lateral, anterior y vertical en el mundo."""
    return ex.direcciones(model, data, "knee_angle_r", "femur_r", "pelvis", "calcn_r", "toes_r", "tibia_r")


def _agregar_camaras(spec, model, data):
    rodilla, lateral, adelante, arriba = _direcciones(model, data)
    for nombre, pos in (("lateral", rodilla + 1.4 * lateral),
                        ("frontal", rodilla + 1.4 * adelante + 0.1 * arriba),
                        ("isometrica", rodilla + 1.0 * (lateral + adelante) + 0.6 * arriba)):
        spec.worldbody.add_camera(name=nombre, pos=pos, fovy=45,
                                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                                  targetbody="tibia_r")


def _agregar_exo(spec, model, data):
    """Exoesqueleto de rodilla en la pierna derecha (ver exo.agregar_exo). El
    grosor del muslo y de la pierna sale de las cápsulas de colisión de MyoLeg."""
    rodilla, lateral, adelante, _ = _direcciones(model, data)
    ex.agregar_exo(spec, model, data, femur="femur_r", tibia="tibia_r", talus="talus_r",
                   rodilla=rodilla, lateral=lateral, adelante=adelante,
                   radio_muslo=model.geom_size[model.geom("femur2_col_r").id][0],
                   radio_pierna=model.geom_size[model.geom("tibia1_col_r").id][0],
                   junta_rodilla="knee_angle_r", signo_flexion=+1)
    ex.agregar_materiales(spec)


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
        _agregar_exo(spec, model, data)          # trae sus IMU (sobre las conchas) y el encoder
    else:
        _agregar_imu_piel(spec, model, data)
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
    return f"""
     EXOESQUELETO DE RODILLA ({ex.MASA_TOTAL:.3f} kg): motor AK60-6 coaxial con la rodilla por el lado
     lateral ({ex.MASA_MOTOR * 1000:.0f} g, Ø{ex.DIAMETRO_MOTOR * 1000:.0f} x {ex.ANCHO_MOTOR * 1000:.1f} mm), barras laterales, valvas
     posteriores de {ex.ALTO_VALVA * 100:.0f} cm al {ex.POSICION_VALVA:.1%} de cada segmento, almohadilla anterior sobre
     la rótula y articulación lateral del tobillo, libre (sin tope, como en Lerner). Piezas rígidas en los cuerpos
     exo_muslo (hijo de femur_r) y exo_pierna (hijo de tibia_r).
     Inercia del rotor reflejada en la rodilla: {ex.INERCIA_REFLEJADA:.2e} kg·m², sumada al armature
     de knee_angle_r.
     Actuador "exo_rodilla": torque en knee_angle_r, ctrl en N·m, POSITIVO = EXTENSOR,
     saturado a ±{ex.TORQUE_MAX_EXO:.0f} N·m. Sensor "exo_torque". Tope mecánico de flexión a {ex.TOPE_FLEXION_EXO:.0f}°.
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

     GENERADO por scripts/old_exportar_pierna.py a partir de scripts/old_modelo_nino.py.
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

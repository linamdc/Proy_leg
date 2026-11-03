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

  Volver a exportar (después de cambiar modelo_nino.py):
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

CASOS = [("sano", None)] + [("pc", e) for e in mn.ESCENARIOS_PC]

CUERPOS_PIERNA = ["pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "patella_r"]
MALLAS_PIERNA = ["r_pelvis", "r_femur", "r_tibia", "r_fibula", "r_talus", "r_foot", "r_bofoot", "r_patella"]


def nombre_caso(version, escenario):
    return "sano" if version == "sano" else f"pc_{escenario}"


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


def _agregar_camaras(spec, model, data):
    rodilla = data.xpos[model.body("tibia_r").id]
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
    for nombre, pos in (("lateral", rodilla + 1.4 * lateral),
                        ("frontal", rodilla + 1.4 * adelante + 0.1 * arriba),
                        ("isometrica", rodilla + 1.0 * (lateral + adelante) + 0.6 * arriba)):
        spec.worldbody.add_camera(name=nombre, pos=pos, fovy=45,
                                  mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
                                  targetbody="tibia_r")


def construir(version, escenario):
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

    spec.add_key(name="home", qpos=model.qpos0)
    T_ex = mn.calibracion()["torque_examinador"]
    popliteo = mn.angulo_popliteo(model, T_ex)
    mn._poner_pose(model, data, 90.0, popliteo)
    spec.add_key(name="popliteo", qpos=data.qpos.copy())

    spec.modelname = f"pierna_nino_{nombre_caso(version, escenario)}"
    spec.meshdir = ""
    spec.texturedir = ""
    return spec, popliteo


# ----------------------------------------------------------------------------
# Escritura
# ----------------------------------------------------------------------------

def _encabezado(version, escenario, popliteo):
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
    return f"""<!-- Pierna derecha del niño ({nombre_caso(version, escenario)}), MyoLeg escalado a P4 de Lerner et al. 2017.

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
-->
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


def exportar(version, escenario):
    spec, popliteo = construir(version, escenario)
    spec.compile()
    xml = _limpiar_xml(spec.to_xml())
    archivo = CARPETA / f"{spec.modelname}.xml"
    archivo.write_text(_encabezado(version, escenario, popliteo) + xml, encoding="utf-8")
    return archivo


def main():
    (CARPETA / "meshes").mkdir(parents=True, exist_ok=True)
    for nombre in MALLAS_PIERNA:
        shutil.copy(MALLAS_ORIGEN / "meshes" / f"{nombre}.stl", CARPETA / "meshes")
    for version, escenario in CASOS:
        archivo = exportar(version, escenario)
        model = mujoco.MjModel.from_xml_path(str(archivo))
        print(f"{archivo.relative_to(CARPETA.parents[1])}: {model.nbody - 2} cuerpos, "
              f"{model.njnt} articulaciones, {model.nu} músculos, "
              f"{model.body_mass[1:].sum():.2f} kg")


if __name__ == "__main__":
    main()

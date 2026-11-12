"""
Versión de marcha (dos piernas) del modelo pediátrico de 6 años de Schmid et al.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg). Cambiar
RUTA por la carpeta donde está el modelo original descargado de SimTK
(spine-children, OS4 v2.1, Fullbody TLSpine):

  Niño (male), el que corresponde a P4 (masculino, Lerner Tabla 1):
    python scripts/modelo_6y_marcha.py RUTA/Male_06years_FullBody_TLSpine_OS4_v2.1.osim data/opensim_6y/Male_06years_marcha.osim

  Niña (female):
    python scripts/modelo_6y_marcha.py RUTA/Female_06years_FullBody_TLSpine_OS4_v2.1.osim data/opensim_6y/Female_06years_marcha.osim

Los .osim de salida buscan las mallas por nombre en data/opensim_6y/Geometry/,
así que la carpeta Geometry tiene que quedar junto a ellos.

Parte del .osim real de SimTK (spine-children, OS4 v2.1, Fullbody TLSpine) y:
  - deja igual la pelvis, el sacro y las dos piernas (cuerpos, juntas, masas,
    inercias, geometría y los 23 músculos de Gait2354 por pierna);
  - junta columna, costillas, esternón, abdomen, cabeza y brazos en un solo
    cuerpo "torso" con la masa, el centro de masa y la inercia combinados de
    esos cuerpos (calculados en la pose con todas las coordenadas en cero);
  - une el torso al sacro con una junta lumbar de 3 GDL en la posición de la
    junta L5-S1 original (lumbar_extension, lumbar_bending, lumbar_rotation);
  - mantiene los 11 fascículos del psoas por lado; sus puntos en las lumbares
    pasan al torso en la misma posición;
  - quita los músculos del tronco, de los brazos y los actuadores de costillas
    y hombros, y añade 3 actuadores de coordenada para la junta lumbar.

Con la columna en cero el modelo nuevo da la misma masa, el mismo centro de
masa, las mismas longitudes musculares y la misma dinámica inversa en las
coordenadas de la pelvis y las piernas que el modelo completo.

Uso (necesita el paquete opensim de Python, p. ej. `pip install opensim`):
  python scripts/modelo_6y_marcha.py ENTRADA.osim SALIDA.osim
"""

import sys
import copy
import xml.etree.ElementTree as ET

import numpy as np
import opensim as osim

KEEP = {"pelvis", "sacrum",
        "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
        "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l"}
LEG = KEEP - {"pelvis", "sacrum"}
BASE = "lumbar5"           # el marco del torso coincide con el de L5
LUMBAR_COORDS = [("lumbar_extension", "L5_S1_FE"),
                 ("lumbar_bending", "L5_S1_LB"),
                 ("lumbar_rotation", "L5_S1_AR")]
LUMBAR_RANGE = np.deg2rad(90.0)
LUMBAR_ACT_FORCE = 100.0   # N·m por unidad de control


def fmt(v):
    return " ".join(repr(float(x)) for x in v)


def transforms_en_cero(path):
    """Transformaciones de cada cuerpo respecto a L5 con todas las coordenadas en 0."""
    model = osim.Model(path)
    state = model.initSystem()
    for c in model.getCoordinateSet():
        c.setLocked(state, False)
        c.setValue(state, 0.0, False)
    model.assemble(state)
    model.realizePosition(state)
    bs = model.getBodySet()

    def en_suelo(body):
        X = body.getTransformInGround(state)
        R = np.array([[X.R().get(r, c) for c in range(3)] for r in range(3)])
        p = np.array([X.p().get(k) for k in range(3)])
        return R, p

    R0, p0 = en_suelo(bs.get(BASE))
    rel = {}
    for i in range(bs.getSize()):
        b = bs.get(i)
        Rg, pg = en_suelo(b)
        R = R0.T @ Rg
        p = R0.T @ (pg - p0)
        rel[b.getName()] = (R, p, xyz_fijo(R))
    return rel


def xyz_fijo(R):
    """Ángulos X-Y-Z de cuerpo fijo (convención de 'orientation' en OpenSim)."""
    # R = Rx(a) · Ry(b) · Rz(c)
    b = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    a = np.arctan2(-R[1, 2], R[2, 2])
    c = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([a, b, c])


def inercia_combinada(bodies_xml, rel):
    m_tot, mc = 0.0, np.zeros(3)
    datos = []
    for b in bodies_xml:
        m = float(b.findtext("mass"))
        c = np.array([float(x) for x in b.findtext("mass_center").split()])
        I6 = [float(x) for x in b.findtext("inertia").split()]
        I = np.array([[I6[0], I6[3], I6[4]],
                      [I6[3], I6[1], I6[5]],
                      [I6[4], I6[5], I6[2]]])
        R, p, _ = rel[b.get("name")]
        c_t = R @ c + p
        datos.append((m, c_t, R @ I @ R.T))
        m_tot += m
        mc += m * c_t
    com = mc / m_tot
    I_tot = np.zeros((3, 3))
    for m, c_t, I_t in datos:
        d = c_t - com
        I_tot += I_t + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return m_tot, com, I_tot


def construir(entrada, salida):
    rel = transforms_en_cero(entrada)
    tree = ET.parse(entrada)
    root = tree.getroot()
    model = root.find("Model")
    nombre = model.get("name")
    model.set("name", nombre + "_marcha")

    # ---------------- cuerpos ----------------
    bodyset = model.find("BodySet/objects")
    upper = [b for b in bodyset if b.get("name") not in KEEP]
    m_t, com_t, I_t = inercia_combinada(upper, rel)

    torso = ET.Element("Body", name="torso")
    fg = ET.SubElement(torso, "FrameGeometry", name="frame_geometry")
    ET.SubElement(fg, "socket_frame").text = ".."
    ET.SubElement(fg, "scale_factors").text = "0.2 0.2 0.2"
    ET.SubElement(torso, "attached_geometry")
    wos = ET.SubElement(torso, "WrapObjectSet", name="wrapobjectset")
    ET.SubElement(wos, "objects"); ET.SubElement(wos, "groups")
    ET.SubElement(torso, "mass").text = repr(m_t)
    ET.SubElement(torso, "mass_center").text = fmt(com_t)
    ET.SubElement(torso, "inertia").text = fmt([I_t[0, 0], I_t[1, 1], I_t[2, 2],
                                                I_t[0, 1], I_t[0, 2], I_t[1, 2]])

    # marcos con la geometría visual de cada cuerpo juntado
    vis_frames = []
    for b in upper:
        ag = b.find("attached_geometry")
        meshes = [] if ag is None else list(ag)
        if not meshes:
            continue
        _, p, xyz = rel[b.get("name")]
        f = ET.Element("PhysicalOffsetFrame", name=f"torso_geom_{b.get('name')}")
        ET.SubElement(f, "attached_geometry").extend(copy.deepcopy(meshes))
        ET.SubElement(f, "socket_parent").text = "/bodyset/torso"
        ET.SubElement(f, "translation").text = fmt(p)
        ET.SubElement(f, "orientation").text = fmt(xyz)
        vis_frames.append(f)

    for b in upper:
        bodyset.remove(b)
    bodyset.append(torso)

    # ---------------- juntas ----------------
    jointset = model.find("JointSet/objects")
    l5s1 = next(j for j in jointset if j.get("name") == "L5_S1_IVDjnt")
    back = copy.deepcopy(l5s1)
    back.set("name", "back")
    back.find("socket_child_frame").text = "torso_offset"
    for co in back.find("coordinates"):
        nuevo = dict((v, k) for k, v in LUMBAR_COORDS)[co.get("name")]
        co.set("name", nuevo)
        co.find("range").text = fmt([-LUMBAR_RANGE, LUMBAR_RANGE])
    for ta in back.iter("TransformAxis"):
        t = ta.find("coordinates")
        if t is not None and t.text:
            viejo_a_nuevo = {v: k for k, v in LUMBAR_COORDS}
            t.text = " ".join(viejo_a_nuevo.get(x, x) for x in t.text.split())
    for fr in back.find("frames"):
        if fr.get("name") == "lumbar5_offset":
            fr.set("name", "torso_offset")
            fr.find("socket_parent").text = "/bodyset/torso"
    back.find("frames").extend(vis_frames)

    for j in list(jointset):
        child_frame = j.findtext("socket_child_frame")
        hijo = None
        for fr in j.find("frames"):
            if fr.get("name") == child_frame:
                hijo = fr.findtext("socket_parent").split("/")[-1]
        if hijo not in KEEP:
            jointset.remove(j)
    jointset.append(back)

    # ---------------- fuerzas ----------------
    forceset = model.find("ForceSet/objects")
    for f in list(forceset):
        if f.tag.endswith("Muscle"):
            pts = list(f.find("GeometryPath/PathPointSet/objects"))
            cuerpos = {p.findtext("socket_parent_frame").split("/")[-1] for p in pts}
            if not cuerpos & LEG:
                forceset.remove(f)
                continue
            for p in pts:
                b = p.findtext("socket_parent_frame").split("/")[-1]
                if b in KEEP:
                    continue
                if p.tag != "PathPoint":
                    raise RuntimeError(f"{f.get('name')}: {p.tag} en {b}")
                R, t, _ = rel[b]
                loc = np.array([float(x) for x in p.findtext("location").split()])
                p.find("location").text = fmt(R @ loc + t)
                p.find("socket_parent_frame").text = "/bodyset/torso"
        else:
            forceset.remove(f)

    for nuevo, _ in LUMBAR_COORDS:
        a = ET.SubElement(forceset, "CoordinateActuator", name=f"{nuevo}_actuator")
        ET.SubElement(a, "min_control").text = "-inf"
        ET.SubElement(a, "max_control").text = "inf"
        ET.SubElement(a, "coordinate").text = nuevo
        ET.SubElement(a, "optimal_force").text = repr(LUMBAR_ACT_FORCE)

    ET.indent(tree, space="\t")
    tree.write(salida, encoding="UTF-8", xml_declaration=True)
    return m_t


if __name__ == "__main__":
    m = construir(sys.argv[1], sys.argv[2])
    print(f"Listo: {sys.argv[2]}  (masa del torso juntado: {m:.3f} kg)")

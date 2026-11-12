"""
Ajusta el XML que sale de MyoConverter (nino6y_marcha_cvt3.xml, hecho con
MuJoCo 2.3.7) para usarlo en MuJoCo 3.16 y para marcha con los pies en el suelo:

  1. quita <option collision="..."> (no existe en MuJoCo 3);
  2. las mallas de los huesos quedan solo para ver (sin contacto ni masa);
  3. agrega una caja semitransparente en el torso (solo visual; la masa y la
     inercia del torso ya vienen del .osim);
  4. agrega 4 esferas de contacto por pie (talón, antepié medial y lateral,
     dedos) colocadas en la superficie de abajo de las mallas del pie;
  5. sube la pelvis para que, en la pose inicial, los pies queden apoyados;
  6. deja qpos0 como una pose válida. MyoConverter deja cada coordenada
     acoplada (traslaciones de la rodilla y puntos móviles de los músculos)
     con su cero en el origen del cuerpo, y el polinomio de la igualdad con
     un término independiente a0: en qpos0 la tibia queda pegada a la cadera.
     Se mueve cada cuerpo a0 a lo largo del eje de su traslación y se resta
     a0 al polinomio, al rango y al keyframe. El movimiento no cambia (las
     mismas poses dan las mismas posiciones); solo cambia dónde está el cero
     de esas coordenadas.

Uso:
  python ajustar_mujoco316.py ENTRADA_cvt3.xml SALIDA.xml
(la carpeta Geometry/ tiene que estar junto a ENTRADA y a SALIDA)
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

R_CONTACTO = 0.008       # m, radio de las esferas de contacto del pie
FRICCION = "1.0 0.005 0.0001"


def cargar(xml_texto, carpeta):
    actual = os.getcwd()
    os.chdir(carpeta)
    try:
        return mujoco.MjModel.from_xml_string(xml_texto)
    finally:
        os.chdir(actual)


def vertices_en_mundo(m, d, geom_name):
    g = m.geom(geom_name).id
    mesh = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    v = m.mesh_vert[a:a + n]
    return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T


def a_cuerpo(d, body_id, p):
    R = d.xmat[body_id].reshape(3, 3)
    return (p - d.xpos[body_id]) @ R


def hornear_acoples(root, carpeta):
    """Paso 6: pasa el término independiente de cada igualdad de juntas al
    cuerpo de la junta acoplada, para que qpos0 cumpla todos los acoples.

    MuJoCo impone  y - y0 = a0 + a1 (x - x0) + ...  con y0, x0 de qpos0 (0 en
    estas juntas). Una traslación desplaza su cuerpo q·eje en el marco del
    cuerpo, o sea R·eje·q en el marco del padre: moviendo el cuerpo R·eje·a0 y
    usando y' = y - a0, las posiciones son las mismas para cualquier pose y el
    nuevo polinomio tiene a0 = 0.
    """
    m = cargar(ET.tostring(root, encoding="unicode"), carpeta)
    cuerpos = {b.get("name"): b for b in root.iter("body")}
    # Las igualdades también son elementos <joint> (dentro de <equality>): se
    # distinguen porque tienen joint1.
    juntas = {j.get("name"): j for j in root.iter("joint") if j.get("name") and not j.get("joint1")}
    igualdades = {e.get("joint1"): e for e in root.iter("joint") if e.get("joint1")}
    claves = list(root.iter("key"))

    desplazamiento = {}
    for i in range(m.neq):
        if m.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        j = m.eq_obj1id[i]
        a0 = float(m.eq_data[i][0])
        if a0 == 0.0:
            continue
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE:
            raise RuntimeError(f"{m.joint(j).name}: acople de rotación con a0 = {a0}; no está previsto")
        b = m.jnt_bodyid[j]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.body_quat[b])
        desplazamiento[b] = desplazamiento.get(b, np.zeros(3)) + R.reshape(3, 3) @ m.jnt_axis[j] * a0

        nombre = m.joint(j).name
        e = igualdades[nombre]
        coef = [float(x) for x in e.get("polycoef").split()]
        coef[0] = 0.0
        e.set("polycoef", " ".join(repr(x) for x in coef))
        junta = juntas[nombre]
        if junta.get("range"):
            lo, hi = (float(x) for x in junta.get("range").split())
            junta.set("range", f"{lo - a0!r} {hi - a0!r}")
        adr = m.jnt_qposadr[j]
        for k in claves:
            if k.get("qpos"):
                q = [float(x) for x in k.get("qpos").split()]
                q[adr] -= a0
                k.set("qpos", " ".join(repr(float(x)) for x in q))

    for b, d in desplazamiento.items():
        cuerpos[m.body(b).name].set("pos", " ".join(repr(float(x)) for x in m.body_pos[b] + d))
    print(f"  acoples horneados: {sum(1 for i in range(m.neq) if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT)} "
          f"igualdades, {len(desplazamiento)} cuerpos movidos")


def main(entrada, salida):
    carpeta = os.path.dirname(os.path.abspath(entrada))
    texto = open(entrada, encoding="utf-8").read()
    root = ET.fromstring(texto)
    for op in list(root.iter("option")):
        op.attrib.pop("collision", None)
    root.set("model", "nino6y_marcha")

    # 2. sin contactos predefinidos de MyoConverter (suelo contra cada malla);
    #    ninguna geometría choca salvo el suelo y las esferas de los pies
    for c in list(root.findall("contact")):
        root.remove(c)
    for g in root.iter("geom"):
        if g.get("type") != "plane":
            g.set("contype", "0")
            g.set("conaffinity", "0")
        if g.get("type") == "mesh":
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("group", "0")
            g.set("mass", "0")

    # suelo con contacto
    for g in root.iter("geom"):
        if g.get("type") == "plane":
            g.set("contype", "1")
            g.set("conaffinity", "1")
            g.set("friction", FRICCION)
            g.set("rgba", "0.8 0.8 0.8 1")

    m = cargar(ET.tostring(root, encoding="unicode"), carpeta)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    cuerpos = {b.get("name"): b for b in root.iter("body")}

    # 3. caja del torso centrada en su centro de masa
    t = m.body("torso").id
    com = m.body_ipos[t]
    ET.SubElement(cuerpos["torso"], "geom", name="torso_caja", type="box",
                  pos=" ".join(f"{x:.5f}" for x in com),
                  size="0.06 0.14 0.08", rgba="0.91 0.85 0.78 0.35",
                  contype="0", conaffinity="0", group="0", mass="0")

    # 4. esferas de contacto en los pies
    for lado, sufijo in (("r", ""), ("l", "l_")):
        calc = vertices_en_mundo(m, d, f"calcn_{lado}_geom_1")
        dedos = vertices_en_mundo(m, d, f"toes_{lado}_geom_1")
        z0 = min(calc[:, 2].min(), dedos[:, 2].min())
        bajo = calc[calc[:, 2] < z0 + 0.012]
        talon = bajo[np.argmin(bajo[:, 0])]
        frente = bajo[bajo[:, 0] > bajo[:, 0].max() - 0.015]
        medial_lateral = [frente[np.argmin(frente[:, 1])], frente[np.argmax(frente[:, 1])]]
        bajo_d = dedos[dedos[:, 2] < z0 + 0.012]
        punta = bajo_d[np.argmax(bajo_d[:, 0])] if len(bajo_d) else dedos[np.argmin(dedos[:, 2])]
        puntos = [("calcn", "talon", talon),
                  ("calcn", "antepie_a", medial_lateral[0]),
                  ("calcn", "antepie_b", medial_lateral[1]),
                  ("toes", "dedos", punta)]
        for cuerpo, nombre, p in puntos:
            centro = p.copy()
            centro[2] = z0 + R_CONTACTO          # todas apoyan a la misma altura
            b = m.body(f"{cuerpo}_{lado}").id
            loc = a_cuerpo(d, b, centro)
            ET.SubElement(cuerpos[f"{cuerpo}_{lado}"], "geom",
                          name=f"contacto_{nombre}_{lado}", type="sphere",
                          size=f"{R_CONTACTO}", pos=" ".join(f"{x:.5f}" for x in loc),
                          rgba="0.2 0.5 0.9 0.6", contype="1", conaffinity="1",
                          group="1", mass="0", friction=FRICCION, condim="3")

    # 5. subir la pelvis: el punto más bajo de los pies queda en z = 0
    dz = -z0
    pelvis = cuerpos["pelvis"]
    pos = np.array([float(x) for x in pelvis.get("pos").split()])
    pos[2] += dz
    pelvis.set("pos", " ".join(f"{x:.5f}" for x in pos))

    # 6. qpos0 como pose válida: el desfase a0 de cada acople pasa al cuerpo
    hornear_acoples(root, carpeta)

    ET.indent(root, space="  ")
    open(salida, "w", encoding="utf-8").write(ET.tostring(root, encoding="unicode"))
    print(f"Listo: {salida}  (pelvis subida {dz*100:.1f} cm)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

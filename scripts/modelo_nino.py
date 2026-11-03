"""
Modelo MyoLeg escalado a un niño de 5 años: versión sana y con parálisis cerebral.

Ver la pierna en MuJoCo (pose de la prueba del ángulo poplíteo). Copiar y
pegar en la terminal, desde la carpeta del repo (Proy_leg):

  Niño sano:
    python scripts/modelo_nino.py --ver sano

  PC sin acortamiento:
    python scripts/modelo_nino.py --ver pc --escenario sin_acortamiento

  PC moderado:
    python scripts/modelo_nino.py --ver pc --escenario moderado

  PC severo:
    python scripts/modelo_nino.py --ver pc --escenario severo

  Tabla con todas las versiones:
    python scripts/modelo_nino.py

Las dos versiones tienen el tamaño del participante P4 de Lerner et al. 2017
(1,15 m; 20,6 kg), así que la única diferencia entre ellas es la patología.

Escalado uniforme del adulto (MyoLeg, ~1,69 m y ~75 kg):
  longitudes          x  s_L = 1,15 / estatura del modelo
  masas               x  s_M = 20,6 / masa del modelo
  inercias            x  s_M * s_L^2
  fuerza muscular     x  s_M^(2/3)   (fuerza ∝ área de sección ∝ masa^(2/3))
  rango de longitudes de los músculos (lengthrange)  x  s_L

Calibración de la fuerza del niño sano:
  El adulto escalado queda con fuerza de adulto, así que se reduce la fuerza
  máxima del cuádriceps y de los isquiotibiales hasta los torques isométricos
  de McKay et al. 2017 (Tabla e-1, niños de 3 a 9 años): extensores 34,9 N·m
  y flexores 27,0 N·m, normalizados por la masa de P4 (≈ 1,7 y 1,3 N·m/kg).
  Senden no midió fuerza. El grupo es amplio y la fuerza en niños depende de
  la estatura, así que para un niño de 5 años es más un techo que un valor exacto.

Calibración con la prueba del ángulo poplíteo (cadera a 90°):
  El torque del examinador se ajusta para que el niño sano, con los
  isquiotibiales sin alterar, dé el ángulo poplíteo de Senden (14°, grupo de
  3 a 6 años). Con ese mismo torque se busca, para cada escenario de PC, el
  acortamiento de los isquiotibiales que da el ángulo poplíteo del escenario.

Versión "pc" (diplejía espástica, GMFCS II, marcha agachada), en ambas piernas:
  - Debilidad del cuádriceps: su fuerza se reduce hasta que el torque extensor
    isométrico máximo de la rodilla sea 0,69 N·m/kg (P4, Lerner Tabla 1).
  - Acortamiento de isquiotibiales y contractura en flexión de la rodilla:
    Lerner no los reporta para P4, así que son supuestos por escenario
    (ESCENARIOS_PC). Referencia: ángulo poplíteo medio de 31° (0–60°) en
    jóvenes con PC, mayor en GMFCS más altos (Cloodt et al. 2021).
  La espasticidad (MAS 1+ en la pierna más afectada y 1 en la menos afectada,
  Lerner Tabla 1) depende de la velocidad de estiramiento, así que no es una
  propiedad del modelo: va en el controlador como reflejo de estiramiento.

Limitación: la rodilla del modelo va de 0° a 120°, así que no reproduce los
4–5° de hiperextensión ni los 145–148° de flexión pasiva del niño sano
(McKay Tabla e-4, Senden).

Uso como módulo:
  from modelo_nino import cargar_nino, spec_nino
  model, data = cargar_nino("sano")
  model, data = cargar_nino("pc", "moderado")   # sin_acortamiento | moderado | severo
  spec = spec_nino("pc", "severo")              # MjSpec editable, p. ej. para el exoesqueleto

Uso directo:
  ver los comandos al inicio de este archivo
"""

import argparse
from functools import lru_cache

import mujoco
import numpy as np
import myo_sim

# ----------------------------------------------------------------------------
# Parámetros que puedes cambiar
# ----------------------------------------------------------------------------

ESTATURA_NINO = 1.15   # m (P4)
MASA_NINO = 20.6       # kg (P4)

# Fuerza isométrica de rodilla del niño sano (N·m/kg): McKay et al. 2017,
# Tabla e-1, niños de 3 a 9 años, dividida por la masa de P4.
FUERZA_EXTENSORA_SANO = 34.9 / MASA_NINO
FUERZA_FLEXORA_SANO = 27.0 / MASA_NINO
# Fuerza extensora isométrica de rodilla de P4 (N·m/kg), Lerner Tabla 1.
# No hay dato de los flexores de P4: quedan como los del niño sano.
FUERZA_EXTENSORA_PC = 0.69
# Postura de la medición con dinamómetro: sentado, cadera a 90°. Se toma el
# máximo del torque en este barrido de flexión de rodilla. Ninguna fuente da
# un ángulo fijo: Lerner midió "at the mid-point of range of motion" (sin
# postura ni cadera) y McKay no describe la posición en el artículo (está en
# el Apéndice B de su protocolo, McKay et al. 2016). Las dos miden en un rango
# medio, donde está el pico de la curva del cuádriceps, y se usa el mismo
# criterio para el sano y para P4 para que la proporción entre ellos sea coherente.
CADERA_DINAMOMETRO = 90.0
BARRIDO_DINAMOMETRO = np.arange(0.0, 90.01, 1.0)

CUADRICEPS = ["recfem", "vasint", "vaslat", "vasmed"]
# Flexores que se calibran. Los gemelos también flexionan la rodilla, pero se
# dejan igual para no debilitar el tobillo.
ISQUIOTIBIALES = ["semimem", "semiten", "bflh", "bfsh"]

# Ángulo poplíteo del niño sano (°): extensión de rodilla con la cadera a 90°,
# Senden, 3 a 6 años, -14 ± 17°.
POPLITEO_SANO = 14.0

# Escenarios de PC: ángulo poplíteo objetivo (°) y contractura en flexión de
# la rodilla (°). Son supuestos; se toma el centro de cada rango propuesto.
ESCENARIOS_PC = {
    "sin_acortamiento": dict(popliteo=None, contractura=0.0),  # isquios como el sano
    "moderado": dict(popliteo=37.5, contractura=5.0),          # 30–45°, ~5°
    "severo": dict(popliteo=55.0, contractura=12.5),           # 50–60°, 10–15°
}
ESCENARIO_PC_DEFECTO = "moderado"

# Isquiotibiales biarticulares que se acortan, igual que en prueba_poplitea.py.
# La cabeza corta del bíceps solo cruza la rodilla.
ISQUIOS_ACORTADOS = ["semimem", "semiten", "bflh"]

LADOS = ["_r", "_l"]

# Barrido de la prueba del ángulo poplíteo: de flexión a extensión.
BARRIDO_POPLITEO = np.arange(120.0, -0.01, -0.5)

# ----------------------------------------------------------------------------
# Escalado
# ----------------------------------------------------------------------------

_ADULTO = myo_sim.load_model("myolegs")


def estatura(model):
    """Estatura del modelo en su pose inicial: extensión vertical de las mallas."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    zmin, zmax = np.inf, -np.inf
    for g in range(model.ngeom):
        if (model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[g] < 3
                and model.geom_bodyid[g] != 0):      # excluye el escenario (cuerpo world)
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            zmin, zmax = min(zmin, v[:, 2].min()), max(zmax, v[:, 2].max())
    return zmax - zmin


S_L = ESTATURA_NINO / estatura(_ADULTO)
S_M = MASA_NINO / _ADULTO.body_mass.sum()
S_F = S_M ** (2 / 3)


def _escalar(spec):
    adulto = _ADULTO
    for mesh in spec.meshes:
        mesh.scale = mesh.scale * S_L

    # El escenario (cuerpo world) no se escala.
    for body in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY):
        body.pos = body.pos * S_L
        # Masa e inercia se copian del adulto compilado y se escalan, así no
        # dependen de si el cuerpo las calculaba a partir de sus geometrías.
        b = adulto.body(body.name).id
        body.explicitinertial = True
        body.mass = adulto.body_mass[b] * S_M
        body.ipos = adulto.body_ipos[b] * S_L
        body.iquat = adulto.body_iquat[b]
        body.inertia = adulto.body_inertia[b] * S_M * S_L**2
        body.fullinertia = [np.nan] * 6

        for frame in body.frames:
            frame.pos = frame.pos * S_L
        for geom in body.geoms:
            geom.pos = geom.pos * S_L
            geom.size = geom.size * S_L
            if not np.isnan(geom.fromto[0]):
                geom.fromto = geom.fromto * S_L
        for site in body.sites:
            site.pos = site.pos * S_L
            site.size = site.size * S_L
        for joint in body.joints:
            joint.pos = joint.pos * S_L
            if joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                joint.range = joint.range * S_L
                joint.ref = joint.ref * S_L
                joint.damping = joint.damping * S_M
                joint.armature = joint.armature * S_M
            elif joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                # Misma escala que la inercia, para que su efecto relativo no cambie.
                joint.damping = joint.damping * S_M * S_L**2
                joint.armature = joint.armature * S_M * S_L**2

    # Los grados de libertad acoplados a la rodilla: los polinomios de las
    # traslaciones dan metros, así que se escalan; los de las rotaciones no.
    for eq in spec.equalities:
        if eq.type == mujoco.mjtEq.mjEQ_JOINT:
            j = adulto.joint(eq.name1).id
            if adulto.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                eq.data[:5] = eq.data[:5] * S_L

    for act in spec.actuators:
        if act.gaintype == mujoco.mjtGain.mjGAIN_MUSCLE:
            act.lengthrange = act.lengthrange * S_L
            act.gainprm[2] *= S_F
            act.biasprm[2] *= S_F


def _L0(lengthrange, gainprm):
    """Longitud óptima de fibra: L = range0 + (longitud - lengthrange0) / L0."""
    return (lengthrange[1] - lengthrange[0]) / (gainprm[1] - gainprm[0])


def _actuador(spec, nombre):
    return next(a for a in spec.actuators if a.name == nombre)


def _escalar_fuerza(spec, musculos, factor):
    for nombre in musculos:
        a = _actuador(spec, nombre)
        a.gainprm[2] *= factor
        a.biasprm[2] *= factor


def _acortar(spec, musculos, fraccion):
    """Desplaza hacia abajo el lengthrange: a la misma longitud real el músculo
    queda más estirado y genera tensión pasiva antes (ver prueba_poplitea.py)."""
    for nombre in musculos:
        a = _actuador(spec, nombre)
        a.lengthrange = a.lengthrange - fraccion * _L0(a.lengthrange, a.gainprm)


# ----------------------------------------------------------------------------
# Pruebas clínicas simuladas
# ----------------------------------------------------------------------------

def _poner_pose(model, data, cadera_deg, rodilla_deg, lado="_r"):
    """Pose estática, músculos relajados, con los grados de libertad acoplados
    a la rodilla."""
    jr = model.joint("knee_angle" + lado).id
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0
    data.act[:] = 0
    data.ctrl[:] = 0
    data.qpos[model.jnt_qposadr[model.joint("hip_flexion" + lado).id]] = np.radians(cadera_deg)
    q = np.radians(rodilla_deg)
    data.qpos[model.jnt_qposadr[jr]] = q
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[i] == jr:
            c = model.eq_data[i][:5]
            data.qpos[model.jnt_qposadr[model.eq_obj1id[i]]] = np.polyval(c[::-1], q)
    mujoco.mj_forward(model, data)


def _brazos_momento(model, data, cadera_deg, rodilla_deg, lado, ids):
    """dL/dθ de cada músculo respecto a la flexión de rodilla, con todos los
    grados de libertad acoplados. El cuádriceps se inserta en la rótula, que
    se mueve por sus propios acoples, así que no basta con la columna de
    knee_angle en actuator_moment. Deja el modelo en la pose pedida."""
    h = 0.25
    _poner_pose(model, data, cadera_deg, rodilla_deg + h, lado)
    L_mas = data.actuator_length[ids].copy()
    _poner_pose(model, data, cadera_deg, rodilla_deg - h, lado)
    L_menos = data.actuator_length[ids].copy()
    _poner_pose(model, data, cadera_deg, rodilla_deg, lado)
    return (L_mas - L_menos) / np.radians(2 * h)


def _musculos(model, lado):
    return [i for i in range(model.nu) if model.actuator(i).name.endswith(lado)]


def torque_isometrico_max(model, musculos, signo, lado="_r"):
    """Torque activo máximo (N·m) del grupo con activación 1, en la postura
    del dinamómetro. signo = -1 para extensores, +1 para flexores."""
    data = mujoco.MjData(model)
    ids = [model.actuator(m + lado).id for m in musculos]
    maximo = 0.0
    for rodilla in BARRIDO_DINAMOMETRO:
        dL = _brazos_momento(model, data, CADERA_DINAMOMETRO, rodilla, lado, ids)
        pasiva = data.actuator_force[ids].copy()
        data.act[ids] = 1.0
        mujoco.mj_forward(model, data)
        activa = data.actuator_force[ids] - pasiva
        # Torque en flexión = dL/dθ · F (F < 0 porque el músculo tira).
        maximo = max(maximo, signo * np.sum(dL * activa))
    return maximo


def torque_extensor_max(model, lado="_r"):
    return torque_isometrico_max(model, CUADRICEPS, -1, lado)


def torque_flexor_max(model, lado="_r"):
    return torque_isometrico_max(model, ISQUIOTIBIALES, +1, lado)


def torque_pasivo_rodilla(model, data, cadera_deg, rodilla_deg, lado="_r"):
    """Torque pasivo de flexión de rodilla (N·m) de todos los músculos de la
    pierna relajados. Positivo = se opone a extender la rodilla."""
    ids = _musculos(model, lado)
    dL = _brazos_momento(model, data, cadera_deg, rodilla_deg, lado, ids)
    return np.sum(dL * data.actuator_force[ids])


def angulo_popliteo(model, torque_examinador, cadera_deg=90.0, lado="_r"):
    """Ángulo de flexión más extendido al que el examinador lleva la rodilla.

    Barre de flexión a extensión y se detiene donde el torque pasivo supera el
    del examinador. Una contractura (límite inferior de la articulación) lo
    limita también.
    """
    data = mujoco.MjData(model)
    ultimo = BARRIDO_POPLITEO[0]
    for rodilla in BARRIDO_POPLITEO:
        if torque_pasivo_rodilla(model, data, cadera_deg, rodilla, lado) > torque_examinador:
            break
        ultimo = rodilla
    else:
        ultimo = 0.0
    limite = np.degrees(model.jnt_range[model.joint("knee_angle" + lado).id][0])
    return max(ultimo, limite)


# ----------------------------------------------------------------------------
# Calibración
# ----------------------------------------------------------------------------

def _acortar_modelo(model, ids, fraccion, lengthrange_original):
    for a in ids:
        lr = lengthrange_original[a]
        model.actuator_lengthrange[a] = lr - fraccion * _L0(lr, model.actuator_gainprm[a])


def _escalar_fuerza_modelo(model, musculos, factor):
    for m in musculos:
        for lado in LADOS:
            a = model.actuator(m + lado).id
            model.actuator_gainprm[a][2] *= factor
            model.actuator_biasprm[a][2] *= factor


@lru_cache(maxsize=None)
def calibracion():
    """Factores de fuerza (sano y PC), torque del examinador (N·m) y
    acortamiento de isquiotibiales (fracción de L0) de cada escenario.
    Se calibra sobre el modelo compilado y luego se aplica al MjSpec."""
    spec = myo_sim.load_spec("myolegs")
    _escalar(spec)
    model = spec.compile()
    data = mujoco.MjData(model)

    cal = {}
    cal["extensores_sano"] = FUERZA_EXTENSORA_SANO * MASA_NINO / torque_extensor_max(model)
    cal["flexores_sano"] = FUERZA_FLEXORA_SANO * MASA_NINO / torque_flexor_max(model)
    _escalar_fuerza_modelo(model, CUADRICEPS, cal["extensores_sano"])
    _escalar_fuerza_modelo(model, ISQUIOTIBIALES, cal["flexores_sano"])

    # La fuerza pasiva escala con la máxima, así que el examinador se calibra
    # ya con la fuerza del niño sano.
    cal["torque_examinador"] = torque_examinador = torque_pasivo_rodilla(model, data, 90.0, POPLITEO_SANO)

    # Debilidad de PC aplicada al modelo compilado, para calibrar sobre él.
    cal["extensores_pc"] = FUERZA_EXTENSORA_PC / FUERZA_EXTENSORA_SANO
    _escalar_fuerza_modelo(model, CUADRICEPS, cal["extensores_pc"])

    ids = [model.actuator(m + "_r").id for m in ISQUIOS_ACORTADOS]
    lr0 = model.actuator_lengthrange.copy()
    acortamientos = {}
    for nombre, esc in ESCENARIOS_PC.items():
        if esc["popliteo"] is None:
            acortamientos[nombre] = 0.0
            continue
        # Bisección: el torque pasivo en el ángulo objetivo crece con el acortamiento.
        bajo, alto = 0.0, 1.0
        for _ in range(30):
            medio = (bajo + alto) / 2
            _acortar_modelo(model, ids, medio, lr0)
            if torque_pasivo_rodilla(model, data, 90.0, esc["popliteo"]) > torque_examinador:
                alto = medio
            else:
                bajo = medio
        acortamientos[nombre] = (bajo + alto) / 2
        model.actuator_lengthrange[:] = lr0
    cal["acortamientos"] = acortamientos
    return cal


# ----------------------------------------------------------------------------
# Versiones
# ----------------------------------------------------------------------------

def spec_nino(version="sano", escenario=ESCENARIO_PC_DEFECTO):
    """MjSpec del niño sin compilar. version: "sano" o "pc"; escenario (solo
    para "pc"): una clave de ESCENARIOS_PC."""
    if version not in ("sano", "pc"):
        raise ValueError("version debe ser 'sano' o 'pc'")
    spec = myo_sim.load_spec("myolegs")
    _escalar(spec)
    cal = calibracion()
    for lado in LADOS:
        _escalar_fuerza(spec, [m + lado for m in CUADRICEPS], cal["extensores_sano"])
        _escalar_fuerza(spec, [m + lado for m in ISQUIOTIBIALES], cal["flexores_sano"])
    if version == "sano":
        return spec

    contractura = ESCENARIOS_PC[escenario]["contractura"]
    for lado in LADOS:
        _escalar_fuerza(spec, [m + lado for m in CUADRICEPS], cal["extensores_pc"])
        _acortar(spec, [m + lado for m in ISQUIOS_ACORTADOS], cal["acortamientos"][escenario])
        if contractura:
            rodilla = spec.joint("knee_angle" + lado)
            rodilla.range = [np.radians(contractura), rodilla.range[1]]
    return spec


def cargar_nino(version="sano", escenario=ESCENARIO_PC_DEFECTO):
    """Devuelve (MjModel, MjData) del niño. Ver spec_nino."""
    model = spec_nino(version, escenario).compile()
    return model, mujoco.MjData(model)


# ----------------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", choices=["sano", "pc"], default=None,
                        help="abre el visor con esa versión en la pose de la prueba del ángulo poplíteo")
    parser.add_argument("--escenario", choices=list(ESCENARIOS_PC), default=ESCENARIO_PC_DEFECTO,
                        help="escenario de PC para --ver")
    args = parser.parse_args()

    if args.ver:
        ver_popliteo(args.ver, args.escenario)
        return

    cal = calibracion()
    T_ex, acortamientos = cal["torque_examinador"], cal["acortamientos"]
    print(f"Adulto MyoLeg: {estatura(_ADULTO):.3f} m, {_ADULTO.body_mass.sum():.1f} kg")
    print(f"Escala: longitudes x {S_L:.3f}, masas x {S_M:.4f}, fuerzas x {S_F:.4f}")
    print(f"Fuerza del niño sano (McKay): cuádriceps x {cal['extensores_sano']:.3f}, "
          f"isquiotibiales x {cal['flexores_sano']:.3f} sobre el adulto escalado")
    print(f"Fuerza del cuádriceps en PC: {100 * cal['extensores_pc']:.0f} % de la del niño sano")
    print(f"Torque del examinador (calibrado con el sano a {POPLITEO_SANO:.0f}°): {T_ex:.2f} N·m\n")

    versiones = [("sano", None)] + [("pc", e) for e in ESCENARIOS_PC]
    print(f"{'Versión':<22} {'Extensor rodilla':>20} {'Flexor rodilla':>20} {'Acort. isquios':>15} "
          f"{'Contractura':>12} {'Poplíteo (cadera 90°)':>22} {'Cadera 0°':>10}")
    print("-" * 127)
    for version, escenario in versiones:
        model, _ = cargar_nino(version, escenario)
        T = torque_extensor_max(model)
        Tf = torque_flexor_max(model)
        acort = acortamientos[escenario] if escenario else 0.0
        contr = ESCENARIOS_PC[escenario]["contractura"] if escenario else 0.0
        nombre = version if escenario is None else f"pc ({escenario})"
        print(f"{nombre:<22} {T:>7.1f} N·m ({T / MASA_NINO:.2f}/kg) {Tf:>7.1f} N·m ({Tf / MASA_NINO:.2f}/kg) {acort:>14.1%} "
              f"{contr:>11.1f}° {angulo_popliteo(model, T_ex):>21.1f}° "
              f"{angulo_popliteo(model, T_ex, cadera_deg=0):>9.1f}°")


def ver_popliteo(version, escenario=ESCENARIO_PC_DEFECTO):
    """Abre el visor con la pierna derecha en la pose final de la prueba del
    ángulo poplíteo (cadera a 90°). Pose estática: no se simula."""
    import mujoco.viewer
    model, data = cargar_nino(version, escenario)
    angulo = angulo_popliteo(model, calibracion()["torque_examinador"])
    _poner_pose(model, data, 90.0, angulo)
    nombre = version if version == "sano" else f"pc ({escenario})"
    print(f"\n{nombre}: ángulo poplíteo {angulo:.1f}°")
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)
            v.sync()


if __name__ == "__main__":
    main()

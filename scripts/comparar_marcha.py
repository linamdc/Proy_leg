"""
Tres marchas lado a lado en una sola escena de MuJoCo, para mostrar la
diferencia: niño sano, marcha agachada y marcha agachada con el exoesqueleto.

  - Niño sano: Senden et al. (3 a 6 años, lenta).
  - Marcha agachada: niño C8 de Steele et al. 2010 (parálisis cerebral, todo de
    una misma zancada). Con --agachada pc se usa en cambio la rodilla de P4
    (Lerner 2017) con cadera y tobillo de Senden.
  - Agachada con exo: el exoesqueleto de exo.py en las dos piernas, con la
    postura que se espera con el exo según lo MEDIDO por Lerner et al. 2017
    (+12° de extensión de rodilla en el apoyo; ver MEJORA_EXTENSION en
    marcha_6y.py). La simulación no predice ese cambio: la cinemática está
    impuesta. Con --sin-mejora el niño con exo camina igual que sin exo (sirve
    para ver solo el dispositivo y el torque del motor).

Los tres caminan en el lugar, como en una cinta, uno al lado del otro, vistos
de costado. En la terminal y en pantalla se ve el ángulo de rodilla de cada uno
y el torque del exo.

Copiar y pegar en la terminal, desde la carpeta del repo (Proy_leg):

  python scripts/comparar_marcha.py                     # visor
  python scripts/comparar_marcha.py --lento 3           # en cámara lenta
  python scripts/comparar_marcha.py --agachada pc       # con P4 en vez de C8
  python scripts/comparar_marcha.py --sin-mejora        # el niño con exo con la misma marcha que sin exo
  python scripts/comparar_marcha.py --gif               # guarda resultados/comparacion_marcha.gif
  python scripts/comparar_marcha.py --musculos          # sano | con PC simulado sin exo | con PC simulado con exo
  python scripts/comparar_marcha.py --musculos --gif    # guarda resultados/comparacion_musculos.gif
  python scripts/comparar_marcha.py --musculos --reflejo 1.25 --perfil "exo propuesto"
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

import marcha_6y as m6

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
SEPARACION = 0.8       # m entre un niño y el siguiente (a lo largo de la marcha)


def casos(agachada, mejora=True):
    base = "crouch" if agachada == "c8" else "pc"
    nombre = f"Marcha agachada ({m6.NINO_CROUCH})" if agachada == "c8" else "Marcha agachada (P4)"
    con_exo = nombre.replace("Marcha agachada", "Agachada con exo")
    if mejora and base == "crouch":
        # postura esperada con el exo: +12° de extensión en el apoyo (Lerner)
        tercero = (f"{base}_exo_mejora", con_exo + f": +{m6.MEJORA_EXTENSION:.0f}° (Lerner)", True)
    else:
        tercero = (f"{base}_exo", con_exo + " (misma marcha)", True)
    return [("sano", "Niño sano", False), (base, nombre, False), tercero]


class Comparacion:
    def __init__(self, agachada="c8", mejora=True, entradas=None):
        self.casos = entradas or casos(agachada, mejora)
        padre = mujoco.MjSpec()
        padre.option.timestep = 0.005
        padre.visual.global_.offwidth, padre.visual.global_.offheight = 1600, 900
        padre.visual.headlight.ambient = [0.45, 0.45, 0.47]
        padre.visual.headlight.diffuse = [0.55, 0.55, 0.55]
        padre.add_texture(name="cielo", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.25, 0.45, 0.75],
                          rgb2=[0.85, 0.9, 0.96], width=256, height=1536)
        padre.add_texture(name="damero", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.42, 0.44, 0.47],
                          rgb2=[0.70, 0.72, 0.75], width=512, height=512)
        mat = padre.add_material(name="piso", texrepeat=[10, 10], texuniform=True, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "damero"
        padre.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="piso")
        padre.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.6, -1], diffuse=[0.6, 0.6, 0.6])

        self.sub = []
        for k, (caso, titulo, exo) in enumerate(self.casos):
            m = m6.caso_marcha(caso)
            spec = m6.spec_modelo(exo)
            model = spec.compile()
            marcha = m6.Marcha(model, mujoco.MjData(model), m)
            marcha.precalcular()
            prefijo = f"n{k}_"
            marco = padre.worldbody.add_frame(pos=[(k - 1) * SEPARACION, 0, 0])
            marco.attach_body(spec.body("pelvis"), prefijo, "")
            self.sub.append(dict(caso=caso, titulo=titulo, m=m, model=model, marcha=marcha, prefijo=prefijo))
        self.model = padre.compile()
        self.data = mujoco.MjData(self.model)
        M = self.model
        for s in self.sub:
            sm = s["model"]
            # índices de qpos del modelo combinado para cada junta del modelo individual
            s["origen"] = np.array([sm.jnt_qposadr[j] for j in range(sm.njnt)])
            s["destino"] = np.array([M.jnt_qposadr[M.joint(s["prefijo"] + sm.joint(j).name).id]
                                     for j in range(sm.njnt)])
            s["rodilla"] = sm.jnt_qposadr[sm.joint("knee_angle_r").id]

    def poner(self, t):
        """Pose de los tres en el tiempo t (cada uno con su propia zancada)."""
        estado = []
        for s in self.sub:
            mar = s["marcha"]
            p = mar.poner(t, cinta=True)
            self.data.qpos[s["destino"]] = mar.data.qpos[s["origen"]]
            fase = mar.fase_pierna(p, "r")
            exo = mar.data.sensor("exo_torque_r").data[0] if mar.exo is not None else None
            estado.append(dict(titulo=s["titulo"], pct=fase, fase=m6.fase(fase, s["m"]["despegue"]["r"]),
                               rodilla=-np.degrees(mar.data.qpos[s["rodilla"]]), exo=exo))
        mujoco.mj_forward(self.model, self.data)
        return estado


class ComparacionMusculos(Comparacion):
    """Niño sano (marcha medida, Senden) | niño con PC simulado con músculos SIN
    exo (sin el dispositivo) | niño con PC simulado con músculos CON exo
    (marcha_musculos_6y.py). La diferencia entre los dos con PC la produce la
    simulación: el exo cambia la postura."""

    def __init__(self, reflejo, perfil):
        import marcha_musculos_6y as mm
        super().__init__(entradas=[("sano", "Niño sano", False),
                                   ("crouch", f"Con PC ({m6.NINO_CROUCH}): sin exo", False),
                                   ("crouch_exo", f"Con PC ({m6.NINO_CROUCH}): con {perfil}", True)])
        z = mm.preparar()["zancada"]
        for s, con_exo in zip(self.sub[1:], (False, True)):
            t, qs, modelo = mm.trayectoria(con_exo, reflejo, perfil, dispositivo=con_exo)
            ref = s["marcha"]
            # caminar en el lugar: se quita el avance medio, corrigiendo la
            # altura como Marcha.poner (el eje de avance no es horizontal)
            qs = qs.copy()
            avance = np.polyval(np.polyfit(t, qs[:, ref.tx], 1), t)   # avance medio (recta)
            qs[:, ref.tx] -= avance
            qs[:, ref.ty] += avance * ref.eje_x[2] / ref.eje_y[2]
            # las juntas son las mismas (el exo no agrega juntas): se copian por índice
            s.update(t=t, qs=qs, origen=np.array([modelo.jnt_qposadr[j] for j in range(modelo.njnt)]))
        self.zancada = z

    def poner(self, t):
        estado = []
        for k, s in enumerate(self.sub):
            if k == 0:
                mar = s["marcha"]
                p = mar.poner(t, cinta=True)
                self.data.qpos[s["destino"]] = mar.data.qpos[s["origen"]]
                rod = -np.degrees(mar.data.qpos[s["rodilla"]])
            else:
                i = int(t * 25) % len(s["t"])
                self.data.qpos[s["destino"]] = s["qs"][i][s["origen"]]
                p = (s["t"][i] / self.zancada % 1) * 100
                rod = -np.degrees(s["qs"][i][s["rodilla"]])
            estado.append(dict(titulo=s["titulo"], pct=p, fase=m6.fase(p, s["m"]["despegue"]["r"]),
                               rodilla=rod, exo=None))
        mujoco.mj_forward(self.model, self.data)
        return estado


def texto(e):
    linea = f"{e['titulo']}: {e['fase']:<13} rodilla {e['rodilla']:5.1f}°"
    if e["exo"] is not None:
        linea += f"  exo {e['exo']:4.2f} N·m"
    return linea


def camara(cam, model):
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.45]
    cam.distance, cam.azimuth, cam.elevation = 2.1, 90.0, -4.0


def ver(comp, lento):
    import mujoco.viewer
    print("De izquierda a derecha (vistos de costado): " + " | ".join(s["titulo"] for s in comp.sub))
    print("Cierra la ventana para salir.")
    with mujoco.viewer.launch_passive(comp.model, comp.data) as visor:
        camara(visor.cam, comp.model)
        inicio = time.time()
        while visor.is_running():
            estado = comp.poner((time.time() - inicio) / lento)
            try:
                visor.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "\n".join(e["titulo"] for e in estado),
                                  "\n".join(f"{e['fase']}  rodilla {e['rodilla']:4.0f}°"
                                            + (f"  exo {e['exo']:.1f} N·m" if e["exo"] is not None else "")
                                            for e in estado))])
            except Exception:
                pass
            visor.sync()
            print("  " + " | ".join(texto(e) for e in estado), end="\r")
            time.sleep(1 / 60)


def gif(comp, lento, segundos=3.0, fps=25):
    from PIL import Image, ImageDraw, ImageFont
    RESULTADOS.mkdir(exist_ok=True)
    cam = mujoco.MjvCamera()
    camara(cam, comp.model)
    try:
        fuente = ImageFont.truetype("arial.ttf", 22)
        chica = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        fuente = chica = ImageFont.load_default()
    cuadros = []
    with mujoco.Renderer(comp.model, 540, 1200) as r:
        for i in range(int(segundos * fps * lento)):
            estado = comp.poner(i / fps / lento)
            r.update_scene(comp.data, camera=cam)
            img = Image.fromarray(r.render())
            dib = ImageDraw.Draw(img)
            for k, e in enumerate(estado):
                x = 30 + k * 400
                dib.text((x, 18), e["titulo"], fill=(30, 30, 30), font=fuente)
                info = f"{e['fase']}, rodilla {e['rodilla']:.0f}°"
                if e["exo"] is not None:
                    info += f"\nexo {e['exo']:.1f} N·m"
                dib.text((x, 48), info, fill=(60, 60, 60), font=chica)
            cuadros.append(img)
    archivo = RESULTADOS / ("comparacion_musculos.gif" if isinstance(comp, ComparacionMusculos)
                            else "comparacion_marcha.gif")
    cuadros[0].save(archivo, save_all=True, append_images=cuadros[1:], duration=int(1000 / fps), loop=0)
    print(f"GIF guardado en {archivo} ({len(cuadros)} cuadros)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agachada", choices=["c8", "pc"], default="c8",
                        help="marcha agachada: c8 (Steele 2010, datos completos) o pc (rodilla de P4)")
    parser.add_argument("--lento", type=float, default=1.0)
    parser.add_argument("--gif", action="store_true", help="guardar un GIF en resultados/ en vez de abrir el visor")
    parser.add_argument("--sin-mejora", action="store_true",
                        help="el niño con exo con la misma marcha que sin exo (sin la mejora de postura de Lerner)")
    parser.add_argument("--musculos", action="store_true",
                        help="C8 medido | simulado con músculos sin exo | simulado con músculos con exo")
    parser.add_argument("--reflejo", type=float, default=None, help="rigidez del reflejo para --musculos")
    parser.add_argument("--perfil", choices=["exo correctivo", "exo Lerner", "exo propuesto"], default=None)
    args = parser.parse_args()
    if args.musculos:
        import marcha_musculos_6y as mm
        comp = ComparacionMusculos(args.reflejo or mm.RIGIDEZ_VER, args.perfil or mm.PERFIL_VER)
    else:
        comp = Comparacion(args.agachada, not args.sin_mejora)
    if args.gif:
        gif(comp, args.lento)
    else:
        ver(comp, args.lento)


if __name__ == "__main__":
    main()

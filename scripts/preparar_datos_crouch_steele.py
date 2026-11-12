"""
Ciclo completo de marcha agachada de un mismo niño (las dos piernas, pelvis y
tronco) con fuerzas de reacción del suelo y centro de presión, para
marcha_6y.py.

Fuente: Steele et al. 2010, "Muscle contributions to support and progression
during single-limb stance in crouch gait", J Biomech 43:2099-2105; datos en
SimTK (proyecto crouchgait, paquete "Crouch Gait Simulations"): 10 niños con
marcha agachada leve, Vicon a 120 Hz y plataformas AMTI.

Por niño se usan:
  - <S>_IK.mot   cinemática inversa (misma convención que el modelo de 6 años:
                 flexión de rodilla negativa, flexión de cadera y dorsiflexión
                 positivas);
  - <S>_GRF.mot  dos fuerzas: la 1 sobre calcn_r y la 2 sobre calcn_l, con su
                 centro de presión;
  - <S>_markerdata.trc  marcadores del tobillo y del dedo de cada pie, para
                 expresar el centro de presión respecto al pie;
  - <S>_model2392.osim  masa del niño.

Eventos:
  - contacto inicial y despegue de cada pie: cuando su fuerza vertical cruza
    20 N en la plataforma;
  - el siguiente contacto del mismo pie (fin del ciclo) y el contacto del pie
    izquierdo dentro del ciclo derecho: el instante en que cadera, rodilla y
    tobillo de esa pierna vuelven a la postura que tenían al pisar la
    plataforma (coincide con los contactos que dan los marcadores).

Cada pie pisó una plataforma una sola vez, a veces en zancadas distintas: la
cinemática de las dos piernas es de la misma zancada (la del apoyo derecho),
la fuerza de cada pie es de su propio apoyo en la plataforma, normalizada a
su ciclo. Por eso, en el doble apoyo, la suma de los dos pies puede pasar de
1,5 veces el peso; en apoyo simple (un pie) la fuerza es la medida.

Salida (data/marcha/crouch_steele/):
  <S>_cinematica.csv   pct (0-100 del ciclo derecho) y coordenadas en grados
  <S>_fuerzas.csv      pct (0-100 del ciclo de cada pierna); por pierna:
                       (fuerzas divididas por la masa y por factor_peso, para que
                       la fuerza vertical media de los dos pies sea el peso)
                       fuerza adelante, arriba, hacia afuera (N/kg) y centro de
                       presión en metros: cop_adelante (delante del marcador
                       del tobillo) y cop_afuera (hacia afuera del marcador del
                       dedo, que va sobre la línea media del pie)
  <S>_parametros.csv   masa, zancada, despegue_r, despegue_l, desfase_l,
                       velocidad, rodilla en apoyo (media), distancia
                       tobillo-dedo de los marcadores
  resumen.csv          una fila por niño

Uso (desde la carpeta del repo):
  python scripts/preparar_datos_crouch_steele.py --origen RUTA/CrouchGaitSimulations
(la carpeta que contiene C1, C2... ya descomprimidos, o los C1.zip... sin descomprimir)
"""

import argparse
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

SALIDA = Path(__file__).resolve().parents[1] / "data" / "marcha" / "crouch_steele"
UMBRAL_N = 20.0
PCT = np.arange(0, 101)
COORDS = ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
          "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
          "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
          "lumbar_extension", "lumbar_bending", "lumbar_rotation"]


# ---------------------------------------------------------------- lectura
class Sujeto:
    """Lee los archivos de un niño desde una carpeta o desde su zip."""

    def __init__(self, origen, nombre):
        self.nombre = nombre
        carpeta = Path(origen) / nombre / nombre
        zip_ = Path(origen) / f"{nombre}.zip"
        if carpeta.is_dir():
            self._leer = lambda f: (carpeta / f).read_text(errors="replace")
        elif zip_.is_file():
            z = zipfile.ZipFile(zip_)
            nombres = {Path(n).name: n for n in z.namelist() if n.count("/") <= 1 or True}
            self._leer = lambda f: z.read(next(n for n in z.namelist()
                                               if n.endswith(f"{nombre}/{f}") or n == f)).decode(errors="replace")
        else:
            raise FileNotFoundError(f"No encuentro {carpeta} ni {zip_}")

    def mot(self, f):
        L = self._leer(f).splitlines()
        i = next(k for k, l in enumerate(L) if l.strip().lower() == "endheader") + 1
        while not L[i].strip():
            i += 1
        cols = [c.strip() for c in L[i].split("\t") if c.strip()]
        datos = np.array([[float(x) for x in l.split()] for l in L[i + 1:] if l.strip()])
        return cols, datos

    def trc(self, f):
        L = self._leer(f).splitlines()
        nombres = [n.strip() for n in L[3].split("\t")[2:] if n.strip()]
        filas = [[float(x) if x.strip() else np.nan for x in l.split("\t")] for l in L[5:] if l.strip()]
        n = min(len(r) for r in filas)
        A = np.array([r[:n] for r in filas])
        return A[:, 1], {nm: A[:, 2 + 3 * k:5 + 3 * k] / 1000.0
                         for k, nm in enumerate(nombres) if 5 + 3 * k <= n}

    def masa(self):
        raiz = ET.fromstring(self._leer(f"{self.nombre}_model2392.osim"))
        return sum(float(b.findtext("mass")) for b in raiz.iter("Body") if b.find("mass") is not None)


# ---------------------------------------------------------------- eventos
def apoyos(t, fy):
    """(contacto, despegue) de cada apoyo completo en la plataforma."""
    on = fy > UMBRAL_N
    e = np.diff(on.astype(int))
    hs, to = t[1:][e == 1], t[1:][e == -1]
    return [(h, to[to > h][0]) for h in hs if np.any(to > h)]


def misma_postura(t, angs, ref, desde, hasta):
    """Instante en [desde, hasta] en que los ángulos vuelven a 'ref' y su error (°)."""
    tt = np.arange(desde, min(hasta, t[-1]), 1 / 240)
    if len(tt) == 0:
        return None, np.inf
    d = np.sqrt(sum((np.interp(tt, t, a) - r) ** 2 for a, r in zip(angs, ref)))
    k = int(np.argmin(d))
    return tt[k], d[k]


def marcador(M, lado, tipo):
    for nombre in {"ankle": (f"{lado}ANK", f"{lado}A"), "toe": (f"{lado}TOE", f"{lado}T")}[tipo]:
        if nombre in M:
            return M[nombre]
    raise KeyError(f"sin marcador de {tipo} {lado}")


# ---------------------------------------------------------------- un niño
def procesar(origen, nombre):
    s = Sujeto(origen, nombre)
    cols, ik = s.mot(f"{nombre}_IK.mot")
    ci = {c: i for i, c in enumerate(cols)}
    ti = ik[:, 0]
    q = lambda c: ik[:, ci[c]]
    _, grf = s.mot(f"{nombre}_GRF.mot")
    tg = grf[:, 0]
    tm, M = s.trc(f"{nombre}_markerdata.trc")
    masa = s.masa()

    sentido = np.sign(np.polyfit(ti, q("pelvis_tx"), 1)[0])   # +1 si camina hacia +x
    piernas = {}
    for lado, c0 in (("r", 1), ("l", 7)):
        ap = apoyos(tg, grf[:, c0 + 1])
        if not ap:
            return None, f"{nombre}: el pie {lado} no pisó la plataforma"
        h, to = ap[0]
        angs = [q(f"{j}_{lado}") for j in ("hip_flexion", "knee_angle", "ankle_angle")]   # grados
        ref = [np.interp(h, ti, a) for a in angs]
        fin, err = misma_postura(ti, angs, ref, h + 0.6, h + 1.5)
        if fin is None or err > 8:
            return None, f"{nombre}: no hay ciclo completo del pie {lado}"
        z = fin - h
        if not 50 <= 100 * (to - h) / z <= 75:
            return None, f"{nombre}: apoyo {lado} incompleto en la plataforma ({100*(to-h)/z:.0f}% del ciclo)"
        piernas[lado] = dict(h=h, to=to, z=z, angs=angs, ref=ref, c0=c0)

    # ciclo derecho: cinemática de las dos piernas de esa zancada
    r = piernas["r"]
    tt = r["h"] + PCT / 100 * r["z"]
    cine = {c: np.interp(tt, ti, q(c)) for c in COORDS}   # el .mot ya está en grados
    for c in ("pelvis_tilt", "pelvis_list", "pelvis_rotation"):
        cine[c] = cine[c] - cine[c].mean()    # sin la orientación media respecto al laboratorio
    l = piernas["l"]
    hl, _ = misma_postura(ti, l["angs"], l["ref"], r["h"] + 0.25 * r["z"], r["h"] + 0.75 * r["z"])
    desfase = 100 * (hl - r["h"]) / r["z"]

    # fuerzas y centro de presión de cada pie, en su propio ciclo
    fuerzas = {"pct": PCT}
    for lado, p in piernas.items():
        tp = p["h"] + PCT / 100 * p["z"]
        en_apoyo = tp <= p["to"]
        F = np.array([np.interp(tp, tg, grf[:, p["c0"] + k]) for k in range(3)]).T
        cop = np.array([np.interp(tp, tg, grf[:, p["c0"] + 3 + k]) for k in range(3)]).T
        tob = np.array([np.interp(tp, tm, marcador(M, lado.upper(), "ankle")[:, k]) for k in range(3)]).T
        dedo = np.array([np.interp(tp, tm, marcador(M, lado.upper(), "toe")[:, k]) for k in range(3)]).T
        largo = np.nanmedian(sentido * (dedo[en_apoyo, 0] - tob[en_apoyo, 0]))
        afuera = sentido * (1 if lado == "r" else -1)          # +z es la derecha
        fuerzas[f"fuerza_adelante_{lado}"] = np.where(en_apoyo, sentido * F[:, 0], 0) / masa
        fuerzas[f"fuerza_arriba_{lado}"] = np.where(en_apoyo, F[:, 1], 0) / masa
        fuerzas[f"fuerza_afuera_{lado}"] = np.where(en_apoyo, afuera * F[:, 2], 0) / masa
        fuerzas[f"cop_adelante_{lado}"] = np.where(en_apoyo, sentido * (cop[:, 0] - tob[:, 0]), np.nan)
        fuerzas[f"cop_afuera_{lado}"] = np.where(en_apoyo, afuera * (cop[:, 2] - dedo[:, 2]), np.nan)
        p["largo"] = largo

    # en una marcha estable la fuerza vertical media de los dos pies es el peso:
    # se normaliza con ese peso "dinámico" (corrige la masa del modelo escalado)
    total = fuerzas["fuerza_arriba_r"] + np.interp((PCT - desfase) % 100, PCT, fuerzas["fuerza_arriba_l"], period=100)
    factor = total.mean() / 9.81
    for k in [k for k in fuerzas if k.startswith("fuerza_")]:
        fuerzas[k] = fuerzas[k] / factor

    param = dict(masa=masa, factor_peso=factor, zancada=r["z"], zancada_l=l["z"],
                 despegue_r=100 * (r["to"] - r["h"]) / r["z"],
                 despegue_l=100 * (l["to"] - l["h"]) / l["z"],
                 desfase_l=desfase,
                 velocidad=abs(np.interp(r["h"] + r["z"], ti, q("pelvis_tx")) - np.interp(r["h"], ti, q("pelvis_tx"))) / r["z"],
                 rodilla_apoyo_r=-cine["knee_angle_r"][:61].mean(),
                 rodilla_apoyo_l=-np.interp((PCT[:61] + desfase) % 100, PCT, cine["knee_angle_l"]).mean(),
                 tobillo_dedo=np.mean([piernas[x]["largo"] for x in piernas]))
    return (cine, fuerzas, param), None


def guardar(nombre, cine, fuerzas, param):
    SALIDA.mkdir(parents=True, exist_ok=True)
    def csv(f, d, nota):
        np.savetxt(SALIDA / f, np.column_stack(list(d.values())), delimiter=",", fmt="%.5f",
                   header=f"Steele et al. 2010 (SimTK crouchgait), niño {nombre}. {nota} "
                          "Generado por preparar_datos_crouch_steele.py\n" + ",".join(d), comments="# ")
    csv(f"{nombre}_cinematica.csv", {"pct": PCT, **cine}, "Ciclo derecho, grados.")
    csv(f"{nombre}_fuerzas.csv", fuerzas, "Ciclo propio de cada pierna; fuerzas en N/kg; centro de presion en m.")
    csv(f"{nombre}_parametros.csv", {k: np.array([v]) for k, v in param.items()}, "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origen", type=Path, required=True,
                        help="carpeta con C1..C10 (carpetas o .zip) del paquete Crouch Gait Simulations")
    args = parser.parse_args()
    filas = []
    for i in range(1, 11):
        nombre = f"C{i}"
        try:
            res, motivo = procesar(args.origen, nombre)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue
        if res is None:
            print(f"  descartado: {motivo}")
            continue
        cine, fuerzas, param = res
        guardar(nombre, cine, fuerzas, param)
        doble = (fuerzas["fuerza_arriba_r"]
                 + np.interp((PCT - param["desfase_l"]) % 100, PCT, fuerzas["fuerza_arriba_l"], period=100)).max() / 9.81
        print(f"  {nombre}: {param['masa']:.1f} kg, zancada {param['zancada']:.2f} s, "
              f"despegue R {param['despegue_r']:.0f}% L {param['despegue_l']:.0f}%, "
              f"desfase L {param['desfase_l']:.1f}%, rodilla en apoyo R {param['rodilla_apoyo_r']:.0f}° "
              f"L {param['rodilla_apoyo_l']:.0f}°, fuerza media/peso del modelo {param['factor_peso']:.2f}, "
              f"suma máx de los dos pies {doble:.2f} × peso")
        filas.append([i] + list(param.values()))
    if filas:
        np.savetxt(SALIDA / "resumen.csv", np.array(filas), delimiter=",", fmt="%.4f",
                   header="Steele et al. 2010, resumen por niño. Generado por preparar_datos_crouch_steele.py\n"
                          + ",".join(["nino"] + list(param)), comments="# ")
    print(f"Listo: {SALIDA}")


if __name__ == "__main__":
    main()

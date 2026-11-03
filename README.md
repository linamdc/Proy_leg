# Proy_leg

Proyecto de Wearable Robotics: simulación biomecánica de la pierna con los modelos musculoesqueléticos de MyoSuite (`myo_sim`) en MuJoCo.

## Instalación

Requisitos: Python 3.10 o superior (probado con 3.13).

1. Clonar este repositorio.
2. Clonar `myo_sim` dentro de la carpeta del repo (no se sube a este repo, cada quien lo clona):

   ```bash
   git clone https://github.com/MyoHub/myo_sim.git
   ```

3. Instalar `myo_sim` en modo editable. Esto instala también `mujoco` y `numpy`:

   ```bash
   pip install -e myo_sim
   ```

4. Instalar las demás librerías que usan los scripts:

   ```bash
   pip install matplotlib openpyxl
   ```

   `openpyxl` solo hace falta para `preparar_datos_marcha.py`, que lee los Excel originales de marcha.

Versiones con las que se ha probado: `mujoco` 3.16.0, `numpy` 2.2.4, `matplotlib` 3.10.1, `openpyxl` 3.1.5.

Para comprobar que todo quedó bien instalado:

```bash
python -c "import myo_sim; m, d = myo_sim.load('myolegs'); print(m.nu, 'músculos')"
```

Debe imprimir `80 músculos`. Si sale `ModuleNotFoundError: No module named 'myo_sim'`, repite el paso 3 desde la carpeta del repo. Esto pasa, por ejemplo, si moviste la carpeta `myo_sim` después de instalarla.

## Scripts

```bash
python scripts/ver_pierna.py                     # abre el visor de MuJoCo con el modelo myolegs
python scripts/dif_legs.py                       # compara articulaciones y músculos de myolegs y myolegs26
python scripts/prueba_poplitea.py                # prueba del ángulo poplíteo: tabla + gráfica
python scripts/prueba_poplitea.py --ver severo   # además muestra la pose final (normal | moderado | severo)
python scripts/modelo_nino.py                    # modelo escalado al niño (P4): tabla de la versión sana y los escenarios de PC
python scripts/modelo_nino.py --ver sano                    # visor en la pose del ángulo poplíteo
python scripts/modelo_nino.py --ver pc --escenario severo   # sin_acortamiento | moderado | severo
```

### Pierna del niño como XML de MuJoCo

En `modelos/pierna_nino/` está la pierna derecha del niño como XML independiente (no necesita `myo_sim`), un archivo por caso: `pierna_nino_sano.xml`, `pierna_nino_pc_sin_acortamiento.xml`, `pierna_nino_pc_moderado.xml` y `pierna_nino_pc_severo.xml`. Tienen la pelvis fija, los 40 músculos de la pierna, dos IMU (muslo y pierna), cámaras y los keyframes `home` y `popliteo`.

Los casos de PC tienen además una versión con exoesqueleto (`pierna_nino_pc_<escenario>_exo.xml`): motor AK60-6 coaxial con la rodilla, barras laterales, valvas posteriores, almohadilla anterior y articulación libre del tobillo (1,0 kg). El actuador `exo_rodilla` da torque en la rodilla, en N·m: positivo es extensor y se satura a ±6 N·m.

```bash
python -m mujoco.viewer --mjcf=modelos/pierna_nino/pierna_nino_sano.xml   # abrir un caso
python scripts/exportar_pierna.py                                         # regenerarlos tras cambiar modelo_nino.py
python scripts/mover_pierna.py --caso pc_moderado_exo                     # verla moverse: asistencia ON/OFF cada 2 s
```

Los XML se generan: no se editan a mano.

### Ciclo de marcha

`marcha.py` mueve la pierna siguiendo los ángulos medidos en cada % del ciclo de marcha (la pelvis está fija, así que camina en el aire):

```bash
python scripts/marcha.py --caso sano              # Senden, 3–6 años, lenta
python scripts/marcha.py --caso pc_moderado       # rodilla de P4 (Lerner 2017)
python scripts/marcha.py --caso pc_moderado_exo   # + perfil de torque del exo de Lerner
python scripts/marcha.py --caso sano --lento 3    # en cámara lenta
```

De P4 solo se tiene la rodilla: la cadera y el tobillo de los casos de PC son los de Senden (supuesto). Las curvas están en `data/marcha/`. Para regenerarlas desde los Excel originales (Senden en OSF 3xqew y el digitalizado de Lerner), corre `python scripts/preparar_datos_marcha.py`.

Para usar el modelo del niño desde otro script en `scripts/`:

```python
from modelo_nino import cargar_nino, spec_nino
model, data = cargar_nino("sano")
model, data = cargar_nino("pc", "moderado")   # sin_acortamiento | moderado | severo
spec = spec_nino("pc", "severo")              # MjSpec editable, para agregarle el exoesqueleto antes de compilar
```

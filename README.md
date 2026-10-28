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
   pip install matplotlib
   ```

Versiones con las que se ha probado: `mujoco` 3.16.0, `numpy` 2.2.4, `matplotlib` 3.10.1.

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
```

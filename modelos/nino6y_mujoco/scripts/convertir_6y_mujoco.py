"""
Convierte el modelo OpenSim del niño de 6 años (marcha, dos piernas) a MuJoCo
con MyoConverter (los tres pasos y el reporte PDF de validación).

Uso (en el entorno de MyoConverter):
  python convertir_6y_mujoco.py MODELO.osim CARPETA_GEOMETRIA CARPETA_SALIDA
"""
import sys
from myoconverter.O2MPipeline import O2MPipeline

osim_file, geometry_folder, output_folder = sys.argv[1:4]

kwargs = dict(
    convert_steps=[1, 2, 3],
    muscle_list=None,
    osim_data_overwrite=True,
    conversion=True,
    validation=True,
    generate_pdf=True,
    speedy=False,
    add_ground_geom=True,
    treat_as_normal_path_point=False,
)

O2MPipeline(osim_file, geometry_folder, output_folder, **kwargs)

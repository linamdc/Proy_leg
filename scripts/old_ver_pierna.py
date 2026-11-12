import mujoco
import mujoco.viewer
import myo_sim

model, data = myo_sim.load("myolegs")
print(f"Articulaciones: {model.njnt}, músculos: {model.nu}")

mujoco.viewer.launch(model, data)

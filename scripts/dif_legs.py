import myo_sim

for nombre in ["myolegs", "myolegs26"]:
    model, data = myo_sim.load(nombre)
    musculos = [model.actuator(i).name for i in range(model.nu)]
    print(f"\n{nombre}: {model.njnt} joints, {model.nu} músculos")
    print(musculos)
    
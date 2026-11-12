import opensim as osim, mujoco, numpy as np, sys
osim.Logger.setLevelString("off")
om=osim.Model('osim/nino6y_marcha.osim'); s=om.initSystem()
mm=mujoco.MjModel.from_xml_path(sys.argv[1]); d=mujoco.MjData(mm)
cs=om.getCoordinateSet()
eqs=[(mm.jnt_qposadr[mm.eq_obj1id[i]], mm.jnt_qposadr[mm.eq_obj2id[i]], mm.eq_data[i][:5][::-1].copy())
     for i in range(mm.neq) if mm.eq_type[i]==mujoco.mjtEq.mjEQ_JOINT]
musc=[om.getMuscles().get(i).getName() for i in range(om.getMuscles().getSize())]
GAIT={'hip_flexion':(-0.35,0.8),'hip_adduction':(-0.2,0.2),'hip_rotation':(-0.25,0.25),'knee_angle':(-1.25,0.0),'ankle_angle':(-0.45,0.35)}
rng=np.random.default_rng(1); err={n:[] for n in musc}; rel={n:[] for n in musc}
indep=['hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r',
       'hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l']
for k in range(60):
    q={c:0.0 for c in indep}
    for c in indep:
        co=cs.get(c); lo,hi=GAIT.get(c.rsplit('_',1)[0],(co.getRangeMin(),co.getRangeMax())); q[c]=rng.uniform(max(lo,co.getRangeMin()),min(hi,co.getRangeMax()))
    for c in cs: c.setValue(s,q.get(c.getName(),0.0),False)
    om.realizePosition(s)
    mujoco.mj_resetDataKeyframe(mm,d,0)
    for j in range(mm.njnt):
        n=mm.joint(j).name
        if n in q or n.startswith(('pelvis_t','pelvis_r','pelvis_l','lumbar')) and n!='pelvis_ty':
            d.qpos[mm.jnt_qposadr[j]]=q.get(n,0.0)
    for _ in range(3):
        for a,b,c in eqs: d.qpos[a]=np.polyval(c,d.qpos[b])
    mujoco.mj_forward(mm,d)
    for n in musc:
        lo=om.getMuscles().get(n).getLength(s); lm=d.ten_length[mm.tendon(n+'_tendon').id]
        err[n].append(abs(lm-lo)); rel[n].append(abs(lm-lo)/lo)
tab=sorted(((max(err[n])*100, max(rel[n])*100, n) for n in musc), reverse=True)
print("músculo        error máx (cm)   (%)")
for e,r,n in tab[:int(sys.argv[2]) if len(sys.argv)>2 else 14]: print(f"{n:14s} {e:6.2f}        {r:5.1f}")
print("músculos con error < 0,5 cm:", sum(e<0.5 for e,_,_ in tab), "de", len(tab))

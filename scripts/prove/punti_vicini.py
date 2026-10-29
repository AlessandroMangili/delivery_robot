import sys,time,numpy as np,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
durata=float(sys.argv[1]) if len(sys.argv)>1 else 10.0; out=sys.argv[2] if len(sys.argv)>2 else '/tmp/punti_vicini.png'
rclpy.init(); n=Node('punti_vicini'); P=[]
def cb(m):
    A=np.frombuffer(m.data,np.uint8).reshape(-1,m.point_step); off={f.name:f.offset for f in m.fields}
    xyz=np.stack([A[:,off[k]:off[k]+4].copy().view(np.float32).ravel() for k in 'xyz'],1).astype(float)
    P.append(xyz[np.isfinite(xyz).all(1)])
n.create_subscription(PointCloud2,'/velodyne_points',cb,qos_profile_sensor_data)
t0=time.time()
while time.time()-t0<durata: rclpy.spin_once(n,timeout_sec=0.2)
if not P: print('nessuna nuvola ricevuta'); sys.exit(2)
X=np.concatenate(P); r=np.linalg.norm(X,axis=1); z=X[:,2]+0.535
print('nuvole %d, punti %d, portata minima %.3f m'%(len(P),len(X),r.min()))
vic=r<0.9
print('punti entro 0,9 m dal sensore: %d (%.1f per nuvola)'%(vic.sum(),vic.sum()/len(P)))
corpo=(np.abs(X[:,0])<0.48)&(np.abs(X[:,1])<0.36)
print('punti sopra l impronta del robot (|x|<0,48, |y|<0,36): %d (%.1f per nuvola) -> se sono tanti e fissi, il LiDAR vede pezzi del robot'%(corpo.sum(),corpo.sum()/len(P)))
for a,b in ((0.0,0.4),(0.4,0.6),(0.6,0.9)):
    s=(r>=a)&(r<b); print('  portata %.1f-%.1f m: %d punti, altezza da terra %s'%(a,b,s.sum(),'%.2f-%.2f m'%(z[s].min(),z[s].max()) if s.any() else '-'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(7,7)); s=r<1.5
ax.scatter(X[s,0],X[s,1],s=2,c=z[s],cmap='viridis'); ax.add_patch(plt.Rectangle((-0.43,-0.31),0.86,0.62,fill=False,ec='red'))
ax.add_patch(plt.Circle((0,0),0.9,fill=False,ec='orange',ls='--')); ax.set_aspect('equal'); ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5)
ax.set_title('punti entro 1,5 m, colore = altezza; rosso = robot, arancione = 0,9 m'); plt.savefig(out,dpi=90); print('figura',out)

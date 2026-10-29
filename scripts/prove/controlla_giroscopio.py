import sys,math,time,rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
durata=float(sys.argv[1]) if len(sys.argv)>1 else 10.0
rclpy.init(); n=Node('controlla_giroscopio'); z=[]
n.create_subscription(Imu,'/mavros/imu/data',lambda m:z.append(m.angular_velocity.z),qos_profile_sensor_data)
t0=time.time()
while time.time()-t0<durata: rclpy.spin_once(n,timeout_sec=0.2)
if len(z)<20: print('IMU: pochi messaggi (%d): MAVROS e lo stream rate sono attivi?'%len(z)); sys.exit(2)
m=sum(z)/len(z); s=math.sqrt(sum((v-m)**2 for v in z)/len(z))
print('giroscopio z a robot fermo: media %+.2f gradi/s, rumore %.2f gradi/s, %d messaggi in %.0f s'%(math.degrees(m),math.degrees(s),len(z),durata))
if abs(math.degrees(m))>0.5: print('NON VA: robot fermo ma il giroscopio segna rotazione. Lascia il robot fermo e ricalibra il giroscopio della Pixhawk prima di partire.'); sys.exit(1)
print('OK')

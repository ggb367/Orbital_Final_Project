import numpy as np
import matplotlib.pyplot as plt
import utils.helpers as hlp
import matplotlib as mpl

pos_vel = np.load('part2.npz')

r_vec = pos_vel['r']
v_vec = pos_vel['v']
print(hlp.cart2elm(r_vec[-1, :], v_vec[-1, :], hlp.earth.mu))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(r_vec[:, 0], r_vec[:, 1], r_vec[:, 2])
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = hlp.earth.radius*np.cos(u)*np.sin(v)
y = hlp.earth.radius*np.sin(u)*np.sin(v)
z = hlp.earth.radius*np.cos(v)
ax.plot_wireframe(x, y, z, color='#FF6C2C')
ax.set_xlim(-20000, 20000)
ax.set_ylim(-20000, 20000)
ax.set_zlim(-20000, 20000)
ax.set_xlabel("I, km")
ax.set_ylabel("J, km")
ax.set_zlabel("K, km")
ax.set_title("Three-quarters view of orbit")
w,h = mpl.figure.figaspect(1)
plt.figure(2, figsize=(w, h))
mpl.style.use('seaborn')
plt.plot(r_vec[:, 0], r_vec[:, 1])
plt.ylabel("J, km")
plt.xlabel("I, km")
plt.grid(True)
plt.title("Orbit in I-J Plane")
plt.axis('equal')

plt.figure(3, figsize=(w, h))
plt.plot(r_vec[:, 1], r_vec[:, 2])
plt.ylabel("K, km")
plt.xlabel("J, km")
plt.grid(True)
plt.title("Orbit in J-K Plane")
plt.axis('equal')

plt.figure(4, figsize=(w, h))
plt.plot(r_vec[:, 0], r_vec[:, 2])
plt.ylabel("K")
plt.xlabel("I")
plt.grid(True)
plt.title("Orbit in I-K Plane")
plt.axis('equal')


plt.show()

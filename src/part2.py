import numpy as np
import pandas as pd
from skyfield.api import load

import utils.helpers as hlp
import utils.propagator as prop
import matplotlib.pyplot as plt

# Remember: All of this is in the J2000 Frame

ts = load.timescale()
t_p = ts.utc(2020, 3, 19, 22, 49) #epoch

class conditions:
    A_m = 0.01/1000**2
    p_srp = 4.57e-6 * 1000 ** 2
    url = 'https://raw.githubusercontent.com/ggb367/Spring-2020/master/366L/hw7/density.csv'
    density_table = pd.read_csv(url)
    epoch = t_p._utc_float()  # days
    C_r = 1.5
    C_D = 2
    # set to 0 to turn off J2 or J3
    J2 = hlp.earth.J2
    J3 = hlp.earth.J3
    # set to False to turn off the moon, sun, drag, or srp
    sun = False
    moon = False
    drag = False
    srp = False

T0 = 0
TF = (365)*24*3600
dT = 1200

elements = [2.5*hlp.earth.radius, 0.5, 77, 0, 296, 0]

r_0, v_0 = hlp.elm2cart(elements, hlp.earth.mu)
r_vec, v_vec = prop.high_fidelity_orbit_prop(r_0, v_0, T0, TF, dT, conditions)

file = open('part2.npz', 'wb')
np.savez(file, r=r_vec, v=v_vec)

print('Complete!')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(r_vec[:, 0], r_vec[:, 1], r_vec[:, 2])
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = hlp.earth.radius*np.cos(u)*np.sin(v)
y = hlp.earth.radius*np.sin(u)*np.sin(v)
z = hlp.earth.radius*np.cos(v)
ax.plot_wireframe(x, y, z, color="blue")
ax.set_xlim(-20000, 20000)
ax.set_ylim(-20000, 20000)
ax.set_zlim(-20000, 20000)
ax.set_xlabel("I")
ax.set_ylabel("J")
ax.set_zlabel("K")

plt.figure(2)
plt.plot(r_vec[:, 0], r_vec[:, 1])
plt.ylabel("J")
plt.xlabel("I")

plt.figure(3)
plt.plot(r_vec[:, 1], r_vec[:, 2])
plt.ylabel("K")
plt.xlabel("J")

plt.figure(4)
plt.plot(r_vec[:, 0], r_vec[:, 2])
plt.ylabel("K")
plt.xlabel("I")


plt.show()


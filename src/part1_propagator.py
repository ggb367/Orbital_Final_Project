import numpy as np
import pandas as pd
from skyfield.api import load

import utils.helpers as hlp
import utils.propagator as prop

# Remember: All of this is in the J2000 Frame

ts = load.timescale()
t_p = ts.utc(2020, 3, 1, 12) #epoch

class conditions:
    A_m = 0.01/1000**2
    p_srp = 4.57e-6 * 1000 ** 2
    url = 'https://raw.githubusercontent.com/ggb367/Spring-2020/master/366L/hw7/density.csv'
    density_table = pd.read_csv(url)
    epoch = t_p._utc_float()  # days
    C_r = 1.5
    C_D = 2
    # set to 0 to turn off J2 or J3
    J2 = 0#hlp.earth.J2
    J3 = 0#hlp.earth.J3
    # set to False to turn off the moon, sun, drag, or srp
    sun = False
    moon = False
    drag = False
    srp = True


elements = np.empty([4, 6])
elements[0, :] = [6763, 0.001, 50, 0, 0, 0] # LEO
elements[1, :] = [26560, 0.001, 55, 0, 0, 0] # MEO
elements[2, :] = [42164, 0.01, 0.5, -120, 0, 0] # GEO
elements[3, :] = [26000, 0.72, 75, 90, -90, 0] # Molyniya

types = ['LEO', 'MEO', 'GEO', 'Molyniya']

T0 = 0
TF = 5*24*3600
dT = 600
for row in range(np.shape(elements)[0]):
    r_0, v_0 = hlp.elm2cart(elements[row, :], hlp.earth.mu)
    r_vec, v_vec = prop.high_fidelity_orbit_prop(r_0, v_0, T0, TF, dT, conditions)

    with open('SRP_'+types[row]+'.npz', 'wb') as f:
        np.savez(f, x=r_vec, y=v_vec)
        # np.save('Data/true_'+types[row]+'_vel', v_vec)
    # df = pd.DataFrame(data={'Position': r_vec, 'Velocity': v_vec})
    # df.to_csv('Data/true_'+types[row]+'.csv')
    print("iteration complete!")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot3D(r_vec[:, 0], r_vec[:, 1], r_vec[:, 2])
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = hlp.earth.radius*np.cos(u)*np.sin(v)
# y = hlp.earth.radius*np.sin(u)*np.sin(v)
# z = hlp.earth.radius*np.cos(v)
# ax.plot_wireframe(x, y, z, color="blue")
# ax.set_xlim(-20000, 20000)
# ax.set_ylim(-20000, 20000)
# ax.set_zlim(-20000, 20000)
#
# plt.show()

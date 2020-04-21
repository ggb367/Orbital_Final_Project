import matplotlib.pyplot as plt
import pandas as pd

import utils.helpers as hlp
import utils.propagator as prop
from skyfield.api import load

# Remember: All of this is in the J2000 Frame

ts = load.timescale()
t_p = ts.utc(2020, 3, 1, 12) #epoch

class conditions:
    A_m = 0.01/1000**2
    p_srp = 4.57e-6 * 1000 ** 2
    url = 'https://raw.githubusercontent.com/ggb367/Spring-2020/master/366L/hw7/density.csv'
    density_table = pd.read_csv(url)
    epoch = t_p._utc_float()  # days
    # set Coeff of Reflectivity to 0 to turn of SRP
    C_r = 1.5
    # C_r = 0
    # set Coeff of Drag to 0 to turn off atmospheric drag
    C_D = 2
    # C_D = 0
    # set to 0 to turn off J2 or J3
    J2 = hlp.earth.J2
    J3 = hlp.earth.J3
    # J2 = 0
    # J3 = 0
    # set to False to turn off the moon or sun
    sun = True
    moon = True


elements = [6763, 0.001, 50, 0, 0, 0] # LEO
# elements = [26560, 0.001, 55, 0, 0, 0] # MEO
# elements = [42164, 0.01, 0.5, -120, 0, 0] # GEO
# elements = [26000, 0.72, 75, 90, -90, 0] # Molniya

T0 = 0
TF = 5*24*3600
dT = 600
r_0, v_0 = hlp.elm2cart(elements, hlp.earth.mu)
r_vec, v_vec = prop.high_fidelity_orbit_prop(r_0, v_0, T0, TF, dT, conditions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(r_vec[:, 0], r_vec[:, 1], r_vec[:, 2])

plt.show()

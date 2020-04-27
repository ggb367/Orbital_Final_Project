import warnings

import numpy as np
import numpy.linalg as lg
from scipy.integrate import ode

import utils.helpers as hlp

def high_fidelity_orbit_prop(r_0, v_0, T0, tF, dT, conds):
    """

    :rtype: numpy.ndarray, numpy.ndarray
    :type conds: class
    :type dT: float or int
    :type tF: float
    :type T0: float
    :type v_0: numpy.ndarray
    :type r_0: numpy.ndarray

    """
    def high_fidelity_orbit(t, Y, mu):
        dY = np.empty([6, 1])
        dY[0] = Y[3]
        dY[1] = Y[4]
        dY[2] = Y[5]
        r = lg.norm(Y[0:3])

        sat2sun_norm, sat2sun, sun_range_norm, sun_range = hlp.sun_3body_pert(t, Y[0:3], conds.epoch)
        if conds.drag:
            a_d = hlp.drag_pert(Y[0:3], [Y[3:6]], conds.density_table, conds.C_D, conds.A_m)
            a_d = np.squeeze(a_d)
        else:
            a_d = np.zeros([3,])

        a_j = hlp.J2J3_Pert(Y[0:3], conds.J2, conds.J3)
        if conds.srp:
            a_srp = hlp.SRP_Pert(Y[0:3], sun_range, conds.C_r, conds.A_m)
            a_srp = np.squeeze(a_srp)
        else:
            a_srp = np.zeros([3,])
        if conds.sun:
            a_sun = [hlp.sun.mu*(sat2sun[0]/((sat2sun_norm)**3)-sun_range[0]/sun_range_norm**3),
                     hlp.sun.mu*(sat2sun[1]/((sat2sun_norm)**3)-sun_range[1]/sun_range_norm**3),
                     hlp.sun.mu*(sat2sun[2]/((sat2sun_norm)**3)-sun_range[2]/sun_range_norm**3)]
        else:
            a_sun = np.zeros([3,])
        if conds.moon:
            sat2moon_norm, sat2moon, moon_range_norm, moon_range = hlp.moon_3body_pert(t, Y[0:3], conds.epoch)
            a_moon = [hlp.moon.mu*(sat2moon[0]/((sat2moon_norm)**3)-moon_range[0]/moon_range_norm**3),
                      hlp.moon.mu*(sat2moon[1]/((sat2moon_norm)**3)-moon_range[1]/moon_range_norm**3),
                      hlp.moon.mu*(sat2moon[2]/((sat2moon_norm)**3)-moon_range[2]/moon_range_norm**3)]
        else:
            a_moon = np.zeros([3,])
        # print(moon_range_norm)
        dY[3] = (-mu*Y[0]/r**3)+a_sun[0]+a_moon[0]+a_d[0]+a_srp[0]+a_j[0]
        dY[4] = (-mu*Y[1]/r**3)+a_sun[1]+a_moon[1]+a_d[1]+a_srp[1]+a_j[1]
        dY[5] = (-mu*Y[2]/r**3)+a_sun[2]+a_moon[2]+a_d[2]+a_srp[2]+a_j[2]
        if lg.norm(Y[0:3])<hlp.earth.radius:
            warnings.warn('The orbit radius is smaller than the radius of the earth, the spacecraft has crashed!')
        return dY

    def derivFcn(t, y):
        return high_fidelity_orbit(t, y, hlp.earth.mu)

    Y_0 = np.concatenate([r_0, v_0], axis=0)
    rv = ode(derivFcn)

    #  The integrator type 'dopri5' is the same as MATLAB's ode45()!
    #  rtol and atol are the relative and absolute tolerances, respectively
    rv.set_integrator('dopri5', rtol=1e-10, atol=1e-20)
    rv.set_initial_value(Y_0, T0)
    output = []
    output.append(np.insert(Y_0, 0, T0))

    # Run the integrator and populate output array with positions and velocities
    while rv.successful() and rv.t<tF:  # rv.successful() and
        rv.integrate(rv.t+dT)
        output.append(np.insert(rv.y, 0, rv.t))

    if not rv.successful() and rv.t<tF:
        warnings.warn("Runge Kutta Failed!", RuntimeWarning)
    #  Convert the output to a numpy array for later use
    output = np.array(output)

    r_vec = np.empty([np.shape(output)[0]-1, 3])
    v_vec = np.empty([np.shape(output)[0]-1, 3])

    for i in range(np.shape(output)[0]-1):
        r_vec[i, 0] = output[i, 1]
        r_vec[i, 1] = output[i, 2]
        r_vec[i, 2] = output[i, 3]
        v_vec[i, 0] = output[i, 4]
        v_vec[i, 1] = output[i, 5]
        v_vec[i, 2] = output[i, 6]
    return r_vec, v_vec

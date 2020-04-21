import math as m

import numpy as np
import numpy.linalg as lg

def AU2km(AU):
    return AU*149597870.7

class earth:
    mu = 398600.4415
    semimajor = AU2km(1.000001018)
    p_srp = 4.57e-9
    J2 = 0.10826267e-2
    J3 = -0.2532327e-5
    radius = 6378.1363

    def eccentricty(T_TDB):
        return 0.01670862-0.000042037*T_TDB-0.0000001236*T_TDB**2+0.00000000004*T_TDB**3

    def inclination(T_TDB, deg=False):
        if deg:
            return 0+0.0130546*T_TDB-0.00000931*T_TDB**2-0.000000034*T_TDB**3
        else:
            return np.deg2rad(0+0.0130546*T_TDB-0.00000931*T_TDB**2-0.000000034*T_TDB**3)

    def RAAN(T_TDB, deg=False):
        if deg:
            return 174.873174-0.2410908*T_TDB+0.00004067*T_TDB**2-0.000001327*T_TDB**3
        else:
            return np.deg2rad(174.873174-0.2410908*T_TDB+0.00004067*T_TDB**2-0.000001327*T_TDB**3)

    def ARG_PERIHELION(T_TDB, deg=False):
        if deg:
            return 102.937348+0.3225557*T_TDB+0.00015026*T_TDB**2+0.000000478*T_TDB**3
        else:
            return np.deg2rad(102.937348+0.3225557*T_TDB+0.00015026*T_TDB**2+0.000000478*T_TDB**3)

    def Mean_Long(T_TDB, deg=False):
        if deg:
            return 100.466449+35999.3728519*T_TDB-0.00000568*T_TDB**2
        else:
            return np.deg2rad(100.466449+35999.3728519*T_TDB-0.00000568*T_TDB**2)

    def obliquity(T_TT, deg=False):
        if deg:
            return 23.439279-0.0130102*T_TT-5.086e-8*T_TT**2+5.565e-7*T_TT**3+1.6e-10*T_TT**4+1.21e-11*T_TT**5
        else:
            return np.deg2rad(
                23.439279-0.0130102*T_TT-5.086e-8*T_TT**2+5.565e-7*T_TT**3+1.6e-10*T_TT**4+1.21e-11*T_TT**5)

class sun:
    mu = 1.32712440042e11
    radius = 6.957e5

class moon:
    mu = 3903

def cart2elm(r, v, mu, deg=True):  # transform position and velocity vectors to classical orbital elements
    h = np.cross(r, v)
    r_norm = lg.norm(r)
    v_norm = lg.norm(v)
    eccent = np.cross(v, h)/mu-np.divide(r, r_norm)  # eccentricity
    eccent_norm = lg.norm(eccent)
    energy = (v_norm**2)/2-mu/r_norm
    h_norm = lg.norm(h)
    k = (h_norm**2)/(r_norm*mu)-1
    if energy<0:
        a = -mu/(2*energy)
    elif -10e-12<energy<10e-12:
        a = m.inf
    else:
        a = mu/(2*energy)
    i = np.arccos(np.dot(h, [0, 0, 1])/h_norm)
    n = np.cross([0, 0, 1], h)
    n_norm = lg.norm(n)
    if eccent_norm<10e-12 or eccent_norm>10e-12:
        nu = np.arccos(k/eccent_norm)
        if np.dot(r, v)<0:
            nu = 2*m.pi-nu
        RAAN = np.arccos(np.dot(n, [1, 0, 0])/n_norm)
        omega = np.arccos(np.dot(n, eccent)/(eccent_norm*n_norm))
    if eccent_norm<10e-12 and i<10e-12:
        RAAN = 0
        omega = 0
        nu = np.arccos(r[1]/r_norm)
        if r[1]<0:
            nu = 2*m.pi-nu
    elif eccent_norm<10e-12:
        omega = 0
        RAAN = np.arccos(np.dot(n, [1, 0, 0])/n_norm)
        nu = np.arccos(np.dot((n/n_norm), r)/r_norm)
        if r[2]<0:
            nu = 2*m.pi-nu
    elif i<10e-12:
        RAAN = 0
        omega = np.arccos(np.dot(eccent, [1, 0, 0])/eccent_norm)
        if energy[1]<0:
            omega = 2*m.pi-omega
    if deg:
        nu = 180*nu/m.pi
        i = 180*i/m.pi
        RAAN = 180*RAAN/m.pi
        omega = 180*omega/m.pi
    E = [a, eccent_norm, i, RAAN, omega, nu]
    for element in E:
        if not isinstance(element, float):
            print(E)
            raise TypeError("One of the elements is not a float!")
    return np.array(E)

def elm2cart(E, mu, deg=True):  # transform classical orbital elements to cartesian position and velocity vectors
    # E - [a, e, i, RAAN, omega, nu]
    a = E[0]
    e = E[1]
    if deg:
        i = m.pi*E[2]/180
        RAAN = m.pi*E[3]/180
        omega = m.pi*E[4]/180
        nu = m.pi*E[5]/180
    else:
        i = E[2]
        RAAN = E[3]
        omega = E[4]
        nu = E[5]
    p = a*(1-e**2)
    r_pqw = np.array([(p/(1+e*np.cos(nu)))*np.cos(nu), (p/(1+e*np.cos(nu)))*np.sin(nu), 0])
    v_pqw = np.array([np.sqrt(mu/p)*(-np.sin(nu)), np.sqrt(mu/p)*(e+np.cos(nu)), 0])
    # R_3(-RAAN)R_1(-i)R_3(-omega)
    c1 = np.cos(-omega)
    c2 = np.cos(-i)
    c3 = np.cos(-RAAN)
    s1 = np.sin(-omega)
    s2 = np.sin(-i)
    s3 = np.sin(-RAAN)
    q1 = np.array([c1*c3-c2*s1*s3, c3*s1+c1*c2*s3, s3*s2])
    q2 = np.array([-c1*s3-c3*c2*s1, c1*c2*c3-s1*s3, c1*s2])
    q3 = np.array([s1*s2, -c1*s2, c2])
    Q = np.array([q1, q2, q3])
    r = np.matmul(Q, r_pqw)
    v = np.matmul(Q, v_pqw)
    return r, v

def R1(phi):  # returns R1 transform matrix
    return np.array(
            [np.array([1, 0, 0]), np.array([0, np.cos(phi), np.sin(phi)]), np.array([0, -np.sin(phi), np.cos(phi)])])

def R2(phi):  # returns R2 transform matrix
    return np.array(
            [np.array([np.cos(phi), 0, -np.sin(phi)]), np.array([0, 1, 0]), np.array([np.sin(phi), 0, np.cos(phi)])])

def R3(phi):  # returns R3 transform matrix
    return np.array(
            [np.array([np.cos(phi), np.sin(phi), 0]), np.array([-np.sin(phi), np.cos(phi), 0]), np.array([0, 0, 1])])

def deg2rad(degree, minutes=0.0, seconds=0.0):  # transform an array of degrees to radians
    if isinstance(degree, int) or isinstance(degree, float):
        return (degree+minutes/60+seconds/3600)*np.pi/180
    elif isinstance(degree, list) or isinstance(degree, np.ndarray):
        output = np.empty(np.size(input))
        for i in range(np.size(input)):
            output[i] = input[i]*np.pi/180
        return output
    else:
        raise TypeError("degree must be a int, float, list, or ndarray, you used a %s", str(type(degree)))

def rad2deg(x: float) -> float:  # transforms a float degree to float radians
    return x*180/m.pi

def up_shadow(r_sc, sun_pos):  # for a given spacecraft vector position and sun vector postion, returns percent of spacecraft in shadow
    a = np.arcsin(sun.radius/lg.norm(sun_pos+r_sc))
    b = np.arcsin(earth.radius/lg.norm(r_sc))
    c = np.arccos(np.dot(r_sc, (sun_pos+r_sc))/(lg.norm(r_sc)*lg.norm(sun_pos+r_sc)))
    if c<np.abs(a-b):
        return 0
    elif a+b<=c:
        return 1
    else:
        x = (c**2+a**2-b**2)/(2*c)
        y = np.sqrt(a**2-x**2)
        A = a**2*np.arccos(x/a)+b**2*np.arccos((c-x)/b)-c*y
        return 1-A/(np.pi*a**2)

def SRP_Pert(r, r_sun, C_r, A_m):  # returns a vecotr of solar radiation pressure perturbation
    gamma = up_shadow(r, r_sun)
    if gamma==0:
        return [0, 0, 0]  # save a few cpu cycles
    return np.array(np.multiply(-1*earth.p_srp*C_r*A_m*gamma, np.divide(r_sun+r, lg.norm(r_sun+r))))

def drag_pert(r, v, density_table, C_D, A_m):  # returns a vector of drag perturbation
    if isinstance(r, np.ndarray) or isinstance(r, list):
        radius = np.linalg.norm(r)-earth.radius

    base_alt = np.array(density_table['Base Altitude'])
    scl_hgt = np.array(density_table['Scale Height'])
    nom_dens = np.multiply(np.array(density_table['Nominal Density']), 1000**3)
    del density_table
    idx = ((np.divide(np.abs(base_alt-radius), 10)).astype(int)).argmin()  # return the base altitude index
    density = nom_dens[idx]*np.exp(-(radius-base_alt[idx])/scl_hgt[idx])
    return np.array(np.multiply(-.5*C_D*A_m*density*lg.norm(v), v))

def J20002GCRF( ):  # returns a matrix that transfromfrom the J2000 frame to the GCRF frame
    delta = deg2rad(0, seconds=0.0146)
    zeta = deg2rad(0, seconds=-0.16617)
    eta = deg2rad(0, seconds=-0.0068192)
    return np.matmul(R3(-delta), np.matmul(R2(-zeta), R1(eta)))

def sun_3body_pert(t, r, epoch):  # returns the thrid body perturbations as a result of the sun
    t = t/86400+epoch
    sun_range, _ = PlanetRV(t)
    sun_range = np.matmul(J20002GCRF(), sun_range)
    sun_range = np.multiply(sun_range, -1)
    sat2sun = sun_range-r
    sat2sun_norm = lg.norm(sat2sun)
    sun_range_norm = lg.norm(sun_range)
    return sat2sun_norm, sat2sun, sun_range_norm, sun_range

def moon_3body_pert(t, r, epoch):  # returns the thrid body perturbations as a result of the sun
    # TODO: FIX THIS SHIT
    t = t/86400+epoch - 2400000.5# MJD
    T = MJDCenturies(t)
    lam = np.mod(np.deg2rad(218.32+481267.8813*T+
                            6.29*np.sin(np.deg2rad(134.9+477198.85*T))-
                            1.27*np.sin(np.deg2rad(259.2-413335.38*T))+
                            0.66*np.sin(np.deg2rad(235.7+890534.23*T))+
                            0.21*np.sin(np.deg2rad(269.9+954397.7*T))-
                            0.19*np.sin(np.deg2rad(357.5+35999.05*T))-
                            0.11*np.sin(np.deg2rad(186.6+966404.05*T))), 2*np.pi)
    phi = np.deg2rad(5.13*np.sin(np.deg2rad(93.3+483202.03*T))
                     +0.28*np.sin(np.deg2rad(228.2+960400.87*T))
                     -0.28*np.sin(np.deg2rad(318.3+6003.18*T))
                     -0.17*np.sin(np.deg2rad(217.6-407332.20*T)))
    parallax = np.deg2rad(0.9508+
                          5.18e-2*np.cos(np.deg2rad(134.9+477198.85*T))+
                          9.5e-3*np.cos(np.deg2rad(259.2-413335.38*T))+
                          7.8e-3*np.cos(np.deg2rad(235.7+890534.23*T))+
                          2.8e-3*np.cos(np.deg2rad(269.9+954397.7*T)))
    eps = np.deg2rad(23.439291-0.0130042*T-1.64e-7*T**2+5.04e-7*T**3)
    r_moon = earth.radius/np.sin(parallax)
    moon_range_hat = np.array([np.cos(phi)*np.cos(lam),
                               np.cos(eps)*np.cos(phi)*np.sin(lam)-np.sin(eps)*np.sin(phi),
                               np.sin(eps)*np.cos(phi)*np.sin(lam)+np.cos(eps)*np.sin(phi)])
    moon_range = np.multiply(r_moon, moon_range_hat)
    sat2moon = moon_range-r
    sat2moon_norm = lg.norm(sat2moon)
    moon_range_norm = lg.norm(moon_range)
    return sat2moon_norm, sat2moon, moon_range_norm, moon_range

def J2J3_Pert(r, J2, J3):  # returns the vector of j2 and j3 perturbations on a sattelite
    r_norm = lg.norm(r)
    a_2 = np.multiply(3*earth.mu*J2*earth.radius**2/(2*r_norm**5),
                      [r[0]*(5*(r[2]/r_norm)**2-1), r[1]*(5*(r[2]/r_norm)**2-1),
                       r[2]*(5*(r[2]/r_norm)**2-3)])
    a_3 = np.multiply(-5*J3*earth.mu*earth.radius**3/(2*r_norm**7),
                      [r[0]*(3*r[2]-7*r[2]**3/r_norm**2), r[1]*(3*r[2]-7*r[2]**3/r_norm**2),
                       6*r[2]**2-7*r[2]**4/r_norm**2-(3/5)*r_norm**2])
    a_p = a_2+a_3
    return np.array(a_p)

def MJDCenturies(MJD: float) -> float:  #converts MJD to MJD centuries
    return (MJD-51544.5)/36525

def KepEqtnE(M, e):
    if -np.pi<M<0 or M>np.pi:
        E = M-e
    else:
        E = M+e
    E_old = E
    count = 0
    while (count<10e4):
        E = E_old+(M-E_old+e*np.sin(E_old))/(1-e*np.cos(E_old))
        count = count+1
        if (abs(E-E_old)<10e-6):
            break
        E_old = E
    return E

def PlanetRV(JD_TDB, MJD=False):
    if not MJD:
        JD_TDB = JD_TDB-2400000.5
    T_TDB = MJDCenturies(JD_TDB)
    M = earth.Mean_Long(T_TDB)-earth.ARG_PERIHELION(T_TDB)
    arg_periapsis = (earth.ARG_PERIHELION(T_TDB)-earth.RAAN(T_TDB))
    eccentric_anomaly = KepEqtnE(M, earth.eccentricty(T_TDB))
    # elements - a e i RAAN arg peri nu
    nu = 2*m.atan2(np.sqrt(1+earth.eccentricty(T_TDB))*np.tan(eccentric_anomaly/2), np.sqrt(1-earth.eccentricty(T_TDB)))
    r, v = elm2cart(
            [earth.semimajor, earth.eccentricty(T_TDB), earth.inclination(T_TDB), earth.RAAN(T_TDB), arg_periapsis, nu],
            sun.mu, deg=False)
    r = np.matmul(R1(-earth.obliquity(T_TDB)), r)
    v = np.matmul(R1(-earth.obliquity(T_TDB)), v)
    return r, v

def s_prime(centuries_tt):
    return deg2rad(0, seconds=-0.000047*centuries_tt)

def precession_nutation(X, Y, s):
    a = (0.5+0.125*(X**2+Y**2))
    return np.matmul(np.array([[1-a*X**2, -a*X*Y, X], [-a*X*Y, 1-a*Y**2, Y], [-X, -Y, 1-a*(X**2+Y**2)]]), R3(s))

def sun_pos(JulianDate, AU=False):  # returns the sun's position for a given Julian Date
    JulianDate = JulianDate-2400000.5
    T = MJDCenturies(JulianDate)
    longitude_sun = deg2rad(280.46+36000.771*T)
    M = deg2rad(357.52772333+35999.0534*T)
    longitude_ecliptic = longitude_sun+np.deg2rad(1.914666471*np.sin(M)+0.019994643*np.sin(2*M))
    radius = 1.000140612-0.016708617*np.cos(M)-0.000139589*np.cos(2*M)
    obliquity = deg2rad(23.439291-0.0130042*T)
    if AU:
        return np.array([radius*np.cos(longitude_ecliptic), radius*np.cos(obliquity)*np.sin(longitude_ecliptic),
                         radius*np.sin(obliquity)*np.sin(longitude_ecliptic)])
    else:
        return np.multiply(np.array(
                [radius*np.cos(longitude_ecliptic), radius*np.cos(obliquity)*np.sin(longitude_ecliptic),
                 radius*np.sin(obliquity)*np.sin(longitude_ecliptic)]), 149597870)


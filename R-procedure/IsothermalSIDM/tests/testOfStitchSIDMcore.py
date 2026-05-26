"""
This code should test the methods of  finding central density (rho0) and central velocity dispersion (sigma0).
Using `stitchSIDM` in files such as: `IsothermalSIDMModel.py` or `IsothermalSIDMModelWithBaryons.py` we for
instances can change algorithm of minimization of finding rho0 and sigma0 (also accuracy of searching).

Also, I did that because I have some intuition that how we calculate the enclosed mass, can influence
on our results. And it turns out that is true. All three methods can be checked by changing method of
calculating enclosed mass in function `deltaSqare`.
"""
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
# ------------ FROM FILES ------------ #
import config as cfg  # constants
import auxiliaryFunctions as aux  # helpful functions
import units as uni
from IsothermalSIDMModel import stitchSIDM
# from IsothermalSIDMModelWithBaryons import stitchSIDM
from NFWProfile import NFWProfile, r1  # CDM profile (halo)

start = timer()
SavePlot = False
# --- cosmological parameters
sigma_m = 10.0  # [cm^2/g] annihilation cross-section
sigma_m_SI = sigma_m * 10**(-4) * 10**3  # [m^2/kg]
M_vir = 1*10**10.0  # [M_sun] Viral mass
# const_c = 15.0  # [dimensionless] concentration of DM
const_c = aux.DMc_gravo(M_vir, 0, cfg.const_h)  # [dimensionless] concentration of DM

# --------------- CREATING NFW CLASS --------------- #
"""
The initial profile of DM (we starting our evolution from that)
"""
NFW_profile = NFWProfile(M_vir, const_c)
r_s = NFW_profile.r_s
rho_s = NFW_profile.rho_s
rounded_r_s = cfg.rounded_number(r_s, 3)
rounded_rho_s = cfg.rounded_number(rho_s, 3)

# --- select period of time and find r1
time_tilda = 380.0  # [300, 440] [dimensionless]
time_gyr = uni.convert_time_tilda(time_tilda, rho_s, r_s, sigma_m)  # [Gyr]
r_1 = r1(NFW_profile, sigmamx=sigma_m, tage=time_gyr)
print(f'r_s: {"{:.3f}".format(r_s)} [kpc]')
print(f'r_1: {"{:.3f}".format(r_1)} [kpc]')
print(f'rho_s: {"{:.0f}".format(rho_s)} [M_sun/kpc^3]')

# --------------- FIND central density and velocity dispersion --------------- #
N = 500
print("***************************************************************************************************************")
IsoSIDM_1 = stitchSIDM(r_1, NFW_profile, N=N)
print(f'N = {N}')
print(f'M: {"{:.0f}".format(IsoSIDM_1[5][-1])} = 10^{"{:.4f}".format(np.log10(IsoSIDM_1[5][-1]))} [M_sun]')
print(f'central density is {"{:.3f}".format(IsoSIDM_1[0])} = 10^{"{:.5f}".format(np.log10(IsoSIDM_1[0]))} [M_sun/kpc^3]')
print(f'central density is {"{:.3f}".format(uni.rho_tilda(IsoSIDM_1[0], rho_s))} [M_sun/kpc^3]')
print(f'central velocity dispersion is {"{:.3f}".format(IsoSIDM_1[1])} [M_sun/kpc^3]')
print("***************************************************************************************************************")

N = 1000
IsoSIDM_2 = stitchSIDM(r_1, NFW_profile, N=N)
print(f'N = {N}')
print(f'central density is {"{:.3f}".format(IsoSIDM_2[0])} = 10^{"{:.5f}".format(np.log10(IsoSIDM_2[0]))} [M_sun/kpc^3]')
print(f'central density is {"{:.3f}".format(uni.rho_tilda(IsoSIDM_2[0], rho_s))} [M_sun/kpc^3]')
print(f'central velocity dispersion is {"{:.3f}".format(IsoSIDM_2[1])} [M_sun/kpc^3]')
print("***************************************************************************************************************")

N = 10000
IsoSIDM_3 = stitchSIDM(r_1, NFW_profile, N=N)
print(f'N = {N}')
print(f'central density is {"{:.3f}".format(IsoSIDM_3[0])} = 10^{"{:.5f}".format(np.log10(IsoSIDM_3[0]))} [M_sun/kpc^3]')
print(f'central density is {"{:.3f}".format(uni.rho_tilda(IsoSIDM_3[0], rho_s))} [rho_s]')
print(f'central velocity dispersion is {"{:.3f}".format(IsoSIDM_3[1])} [M_sun/kpc^3]')
print("***************************************************************************************************************")

N = 50000
IsoSIDM_4 = stitchSIDM(r_1, NFW_profile, N=N)
print(f'M: {"{:.0f}".format(IsoSIDM_4[5][-1])} = 10^{"{:.4f}".format(np.log10(IsoSIDM_4[5][-1]))} [M_sun]')
print(f'N = {N}')
print(f'central density is {"{:.3f}".format(IsoSIDM_4[0])} = 10^{"{:.5f}".format(np.log10(IsoSIDM_4[0]))} [M_sun/kpc^3]')
print(f'central density is {"{:.3f}".format(uni.rho_tilda(IsoSIDM_4[0], rho_s))} [rho_s]')
print(f'central velocity dispersion is {"{:.3f}".format(IsoSIDM_4[1])} [M_sun/kpc^3]')

end = timer()
print("script working:", end - start, "s")

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(IsoSIDM_1[4], IsoSIDM_1[2], label='Density Profile: N = 500')
plt.plot(IsoSIDM_4[4], IsoSIDM_4[2], "--", label='Density Profile: N = 50000')
# plt.plot(IsoSIDM_2[4], IsoSIDM_2[2], "--", label='Density Profile: N = 1000')

plt.yscale('log', base=10)
plt.xscale('log', base=10)

plt.xlabel('Radius (kpc)')
plt.ylabel('Dark Matter Density (rho_dm)')
plt.title('Dark Matter Density Profile')
plt.grid(True)
plt.legend()
if SavePlot is True:
    plt.savefig('miso-density.png', dpi=300)
    plt.close()
else:
    plt.show()

# Plot the solution
plt.figure(figsize=(8, 6))
plt.plot(IsoSIDM_1[4], IsoSIDM_1[5], label='Mass Profile: N = 500')
plt.plot(IsoSIDM_4[4], IsoSIDM_4[5], "--", label='Mass Profile: N = 50000')
# plt.plot(IsoSIDM_2[4], IsoSIDM_2[5], "--", label='Mass Profile: N = 1000')

plt.yscale('log', base=10)
plt.xscale('log', base=10)

plt.xlabel('Radius (kpc)')
plt.ylabel('Dark Matter mass (M(<r))')
plt.title('Dark Matter mass Profile')
plt.grid(True)
plt.legend()
if SavePlot is True:
    plt.savefig('miso-mass.png', dpi=300)
    plt.close()
else:
    plt.show()


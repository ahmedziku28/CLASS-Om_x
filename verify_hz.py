import numpy as np
import matplotlib.pyplot as plt
from classy import Class

# ── parameters ───────────────────────────────────────────
a_exo   = -100.0
b_exo   = 1.0
z_c     = 30.0
sigma_z = 6.0
H0      = 67.32
h       = H0 / 100.0
Omega_b   = 0.02238 / h**2
Omega_cdm = 0.1201  / h**2
Omega_m   = Omega_b + Omega_cdm

# ── Python reference functions ────────────────────────────
def gaussian(z, z_c, sz):
    return np.exp(-0.5*(z-z_c)**2/sz**2)

def amplitude(z, a, b):
    return a*(1+z) + b*z

def window(z, a, b, z_c, sz):
    amp     = amplitude(z, a, b)
    amp_0   = a*(1+z)
    exponent = (z_c**2 - (z-z_c)**2) / (2*sz**2)
    return (amp/amp_0) * np.exp(exponent)

def Omega_x0_func(a, z_c, sz):
    return a * gaussian(0, z_c, sz)

def Hz_python(z, H0, Om_m, Om_r, Om_L, a, b, z_c, sz):
    Omx0 = Omega_x0_func(a, z_c, sz)
    W    = window(z, a, b, z_c, sz)
    E2   = Om_m*(1+z)**3 + Om_r*(1+z)**4 + Om_L + Omx0*W
    return H0 * np.sqrt(np.maximum(E2, 0))

# ── Run CLASS first to get exact Omega_r ─────────────────
cosmo = Class()
cosmo.set({
    'H0':              H0,
    'omega_b':         0.02238,
    'omega_cdm':       0.1201,
    'N_ur':            3.046,
    'tau_reio':        0.0544,
    'a_exo':           a_exo,
    'b_exo':           b_exo,
    'z_c_exo':         z_c,
    'sigma_z_exo':     sigma_z,
    'output':          '',
    'background_verbose': 3,
})
cosmo.compute()

# get exact radiation from CLASS (photons + massless neutrinos)
# Omega_g  = cosmo.Omega_g()    # photons only
# Omega_ur = cosmo.Omega_ur()   # ultra-relativistic relics (massless nu)
Omega_r  = cosmo.Omega_r()
# print(f"CLASS Omega_g  = {Omega_g:.6e}")
# print(f"CLASS Omega_ur = {Omega_ur:.6e}")
print(f"CLASS Omega_r  = {Omega_r:.6e}")

# get H(z) from CLASS
z_arr    = np.linspace(0, 60, 500)
Hz_class = np.array([cosmo.Hubble(z) * 299792.458 for z in z_arr])
print(f"CLASS Omega_x0 = {cosmo.Omega0_exo:.6e}")

cosmo.struct_cleanup()
cosmo.empty()

# ── Now compute Python reference with exact same Omega_r ──
Omx0 = Omega_x0_func(a_exo, z_c, sigma_z)
Om_L = 1.0 - Omega_m - Omega_r - Omx0
print(f"\nPython: Omega_x0 = {Omx0:.6e}")
print(f"Python: Omega_r  = {Omega_r:.6e}")
print(f"Python: Omega_L  = {Om_L:.10f}")

Hz_py = Hz_python(z_arr, H0, Omega_m, Omega_r, Om_L,
                  a_exo, b_exo, z_c, sigma_z)

# ── comparison ────────────────────────────────────────────
rel_diff = (Hz_class - Hz_py) / Hz_py
print(f"\nMax relative difference: {np.max(np.abs(rel_diff))*1e6:.3f} ppm")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].semilogy(z_arr, Hz_py,    'b-',  lw=2,   label='Python')
axes[0].semilogy(z_arr, Hz_class, 'r--', lw=1.5, label='CLASS')
axes[0].axvline(z_c, color='gray', ls=':', alpha=0.5, label=f'z_c={z_c}')
axes[0].set_xlabel('z')
axes[0].set_ylabel('H(z)  [km/s/Mpc]')
axes[0].set_title('Hubble rate')
axes[0].legend()

axes[1].plot(z_arr, rel_diff * 1e6, 'k-', lw=1.5)
axes[1].axhline(0, color='gray', ls='--', lw=0.8)
axes[1].set_xlabel('z')
axes[1].set_ylabel('(H_CLASS - H_py) / H_py  [ppm]')
axes[1].set_title('Relative difference — should be < 1 ppm')

plt.tight_layout()
plt.savefig('verify_hz.png', dpi=150)
print("Saved verify_hz.png")
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import qutip

# ══════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════
DEFG_FILE = "dEFG_dQk.npz"

# ══════════════════════════════════════════════════════════════════════════
# PARAMETERS  (all from DFT or CIF)
# ══════════════════════════════════════════════════════════════════════════
E2QQ_MHZ  = -608.794743   # MHz
ETA_Q     =  0.940691
A_ISO_MHZ = -9.9235       # MHz
I_SPIN    = 5 / 2
Q_153Eu   = 2.41e-28      # C·m²
T_BATH    = 4.2           # K
GAM_PH_CM = 20.0          # cm⁻¹  (Raman linewidth of low-freq modes)

bath_spins_raw = [
    ('H', 0.5, -4.377754,  1.147464,  3.671829),
    ('H', 0.5,  2.705193,  1.730396, -5.172580),
    ('H', 0.5, -4.524319,  3.360769,  2.498627),
    ('H', 0.5, -3.772097,  4.853370,  0.664598),
    ('H', 0.5,  0.772214,  3.436250, -4.251066),
    ('H', 0.5, -2.312284,  5.238476, -1.335962),
    ('H', 0.5,  2.782100, -0.272415, -3.256752),
    ('H', 0.5, -0.553727,  4.406667, -2.875001),
    ('H', 0.5, -3.379545, -1.115950,  3.816080),
    ('H', 0.5, -2.464911, -3.032421,  3.784891),
    ('H', 0.5,  0.555278, -4.133950,  0.877101),
    ('H', 0.5, -0.915200, -5.147595,  3.001573),
    ('N', 1.0, -1.384994,  2.250116, -0.628577),
    ('N', 1.0, -2.227990,  0.593825,  1.386424),
    ('N', 1.0, -2.126693, -1.528136,  2.322551),
    ('N', 1.0,  0.880523,  1.522349,  2.381702),
    ('N', 1.0, -0.022078,  2.509503, -2.486584),
    ('N', 1.0, -1.423406, -1.498989, -2.100499),
    ('N', 1.0,  0.593269,  1.338372, -2.167569),
    ('N', 1.0, -1.183328, -1.864174,  1.399839),
]

GAMMA = {
    '153Eu': 6.6252e6,
    'H':     2.6752e8,
    'N':    -1.9338e7,
}

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════
hbar      = constants.hbar
k_B       = constants.k
c_SI      = constants.c
amu       = constants.u
mu0       = constants.mu_0
cm2rad    = c_SI * 100 * 2 * np.pi
DIM       = int(2 * I_SPIN + 1)
GAM_PH    = GAM_PH_CM * cm2rad
SCALE_SI  = 1e20 / (1e-10 * np.sqrt(amu))
PREFACTOR = constants.e * Q_153Eu / (2 * I_SPIN * (2 * I_SPIN - 1) * hbar)
ANG2M     = 1e-10
mu0_4pi   = mu0 / (4 * np.pi)

# ══════════════════════════════════════════════════════════════════════════
# SPIN OPERATORS  (QuTiP)
# ══════════════════════════════════════════════════════════════════════════
Jx = qutip.spin_Jx(I_SPIN)
Jy = qutip.spin_Jy(I_SPIN)
Jz = qutip.spin_Jz(I_SPIN)
Jp = qutip.spin_Jp(I_SPIN)
Jm = qutip.spin_Jm(I_SPIN)
Id = qutip.qeye(DIM)

Ivec_qt = [Jx, Jy, Jz]

def T_cart_qt(i, j):
    d = 1.0 if i == j else 0.0
    return (0.5 * (Ivec_qt[i] * Ivec_qt[j] + Ivec_qt[j] * Ivec_qt[i])
            - d * I_SPIN * (I_SPIN + 1) / 3.0 * Id)

T_cart_np = np.array([[T_cart_qt(i, j).full() for j in range(3)]
                       for i in range(3)])

# ══════════════════════════════════════════════════════════════════════════
# HAMILTONIAN  (QuTiP)
# ══════════════════════════════════════════════════════════════════════════
pf_qt = (2 * np.pi * E2QQ_MHZ * 1e6) / (4 * I_SPIN * (2 * I_SPIN - 1))

H_qt = (pf_qt * (3 * Jz * Jz - I_SPIN * (I_SPIN + 1) * Id
                  + ETA_Q * (Jp * Jp + Jm * Jm))
        + 2 * np.pi * A_ISO_MHZ * 1e6 * Jz)

evals, ekets = H_qt.eigenstates()
U    = np.column_stack([e.full().flatten() for e in ekets])
OAB  = evals[np.newaxis, :] - evals[:, np.newaxis]

Iz_eb = U.conj().T @ Jz.full() @ U
Ip_eb = U.conj().T @ Jp.full() @ U
Im_eb = U.conj().T @ Jm.full() @ U

# ══════════════════════════════════════════════════════════════════════════
# DFT MODES
# ══════════════════════════════════════════════════════════════════════════
data    = np.load(DEFG_FILE)
dV_all  = data['dV_dQk'].astype(float)
freqs   = data['freqs_cm'].astype(float)
N_modes = dV_all.shape[0]
omega_k = freqs * cm2rad
xk      = hbar * omega_k / (k_B * T_BATH)
n_k     = np.where(xk < 50, 1.0 / np.expm1(np.clip(xk, 1e-10, 50)), 0.0)

# ══════════════════════════════════════════════════════════════════════════
# HELPER: build phonon W matrix with tunable parameters
# ══════════════════════════════════════════════════════════════════════════
T_eb_qt = np.array([[U.conj().T @ T_cart_np[i, j] @ U
                      for j in range(3)] for i in range(3)])

H_diag_qt = qutip.Qobj(np.diag(evals), dims=[[DIM], [DIM]])


def build_W_ph(gam_ph_cm_val=GAM_PH_CM, sec_cutoff_val=0.1,
               extra_modes=None):
    """
    Build the phonon W matrix.

    Parameters
    ----------
    gam_ph_cm_val : float  Raman linewidth in cm⁻¹
    sec_cutoff_val : float  secular approximation cutoff (rad/s units passed
                            directly to bloch_redfield_tensor)
    extra_modes : list of (freq_cm, dV_tensor) pairs to append
    """
    gam_ph_val = gam_ph_cm_val * cm2rad

    # Build coupling operators
    use_dV = list(dV_all)
    use_ok = list(omega_k)
    use_nk = list(n_k)

    if extra_modes is not None:
        for (fc, dV_extra) in extra_modes:
            ok_e = fc * cm2rad
            xk_e = hbar * ok_e / (k_B * T_BATH)
            nk_e = 1.0 / np.expm1(np.clip(xk_e, 1e-10, 50)) if xk_e < 50 else 0.0
            use_dV.append(dV_extra)
            use_ok.append(ok_e)
            use_nk.append(nk_e)

    a_ops_local = []
    for k in range(len(use_ok)):
        A_k = np.zeros((DIM, DIM), dtype=complex)
        for i in range(3):
            for j in range(3):
                A_k += use_dV[k][i, j] * T_eb_qt[i, j]
        A_k_qt = qutip.Qobj(A_k * PREFACTOR * SCALE_SI, dims=[[DIM], [DIM]])

        ok, nk = use_ok[k], use_nk[k]
        def make_S(ok=ok, nk=nk, gph=gam_ph_val):
            def S(omega):
                Lp = (gph / np.pi) / ((omega + ok)**2 + gph**2)
                Lm = (gph / np.pi) / ((omega - ok)**2 + gph**2)
                return (hbar / (2 * ok)) * ((nk + 1) * Lp + nk * Lm)
            return S
        a_ops_local.append([A_k_qt, make_S()])

    R_qt_l, _ = qutip.bloch_redfield_tensor(H_diag_qt, a_ops_local,
                                             sec_cutoff=sec_cutoff_val,
                                             sparse_eigensolver=False)
    R_np_l = R_qt_l.full()

    W = np.zeros((DIM, DIM))
    for a in range(DIM):
        for b in range(DIM):
            if a != b:
                W[a, b] = np.real(R_np_l[a * DIM + a, b * DIM + b])
    for b in range(DIM):
        W[b, b] = -np.sum(W[:, b])
    return W


# ══════════════════════════════════════════════════════════════════════════
# BASELINE W_ph  (nominal parameters)
# ══════════════════════════════════════════════════════════════════════════
W_ph = build_W_ph()

# ══════════════════════════════════════════════════════════════════════════
# BATH SPIN CORRELATION TIMES
# ══════════════════════════════════════════════════════════════════════════
def compute_tau_bath(species, positions_m, gamma_rad):
    n  = len(positions_m)
    M2 = 0.0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            r = np.linalg.norm(positions_m[i] - positions_m[j])
            if r > 1e-12:
                M2 += r**(-6)
    I_bath = 0.5 if species == 'H' else 1.0
    M2 *= (3/5) * mu0_4pi**2 * gamma_rad**4 * hbar**2 * I_bath * (I_bath + 1)
    return 1.0 / np.sqrt(M2)

H_pos_m = np.array([[x,y,z] for (el,I,x,y,z) in bath_spins_raw if el=='H']) * ANG2M
N_pos_m = np.array([[x,y,z] for (el,I,x,y,z) in bath_spins_raw if el=='N']) * ANG2M

tau_H = compute_tau_bath('H', H_pos_m, GAMMA['H'])
tau_N = compute_tau_bath('N', N_pos_m, GAMMA['N'])

# ══════════════════════════════════════════════════════════════════════════
# DIPOLAR W MATRIX  (custom Lorentzian bath spectral density)
# ══════════════════════════════════════════════════════════════════════════
def dipolar_W_contribution(bath_spins, gamma_bath, I_bath, tau_bath):
    S_fac  = I_bath * (I_bath + 1)
    W_dip  = np.zeros((DIM, DIM))
    Ip2_eb = Ip_eb @ Ip_eb
    Im2_eb = Im_eb @ Im_eb

    for pos_m in bath_spins:
        r_vec = np.array(pos_m)
        r     = np.linalg.norm(r_vec)
        if r < 1e-12: continue
        costh = r_vec[2] / r
        sinth = np.sqrt(max(0.0, 1 - costh**2))
        b     = mu0_4pi * GAMMA['153Eu'] * gamma_bath * hbar / r**3

        geom = {
            0: (1 - 3*costh**2)**2,
            1: (9/4) * sinth**2 * costh**2,
            2: (9/16) * sinth**4,
        }

        for a in range(DIM):
            for b_idx in range(DIM):
                if a == b_idx: continue
                omega_ab = abs(OAB[b_idx, a])
                if omega_ab < 1e3: continue
                J     = S_fac * 2 * tau_bath / (1 + omega_ab**2 * tau_bath**2)
                mel_0 = abs(Iz_eb[a, b_idx])**2
                mel_1 = abs(Ip_eb[a, b_idx])**2 + abs(Im_eb[a, b_idx])**2
                mel_2 = abs(Ip2_eb[a, b_idx])**2 + abs(Im2_eb[a, b_idx])**2
                W_dip[a, b_idx] += (b**2 * J * (S_fac/3) *
                    (geom[0]*mel_0 + geom[1]*mel_1 + geom[2]*mel_2))

    for a in range(DIM):
        for b_idx in range(DIM):
            if a == b_idx: continue
            omega_ab = OAB[b_idx, a]
            if abs(omega_ab) < 1e3: continue
            x    = hbar * omega_ab / (k_B * T_BATH)
            bose = np.exp(x) if x < 50 else np.inf
            if bose < np.inf:
                w_avg = 0.5 * (W_dip[a, b_idx] + W_dip[b_idx, a])
                W_dip[a, b_idx] = w_avg * 2 / (1 + 1/bose)
                W_dip[b_idx, a] = w_avg * 2 / (1 + bose)

    for b_idx in range(DIM):
        W_dip[b_idx, b_idx] = -np.sum(W_dip[:, b_idx])
    return W_dip


def build_W_dip(tau_H_val, tau_N_val):
    H_spins_m = [np.array([x,y,z])*ANG2M for (el,I,x,y,z) in bath_spins_raw if el=='H']
    N_spins_m = [np.array([x,y,z])*ANG2M for (el,I,x,y,z) in bath_spins_raw if el=='N']
    W_H = dipolar_W_contribution(H_spins_m, GAMMA['H'], 0.5, tau_H_val)
    W_N = dipolar_W_contribution(N_spins_m, GAMMA['N'], 1.0, tau_N_val)
    W   = W_H + W_N
    np.fill_diagonal(W, 0.0)
    for b_idx in range(DIM):
        W[b_idx, b_idx] = -np.sum(W[:, b_idx])
    return W


H_spins_m = [np.array([x,y,z])*ANG2M for (el,I,x,y,z) in bath_spins_raw if el=='H']
N_spins_m = [np.array([x,y,z])*ANG2M for (el,I,x,y,z) in bath_spins_raw if el=='N']

W_dip_H = dipolar_W_contribution(H_spins_m, GAMMA['H'], 0.5, tau_H)
W_dip_N = dipolar_W_contribution(N_spins_m, GAMMA['N'], 1.0, tau_N)
W_dip   = W_dip_H + W_dip_N
for b_idx in range(DIM):
    W_dip[b_idx, b_idx] = 0.0
for b_idx in range(DIM):
    W_dip[b_idx, b_idx] = -np.sum(W_dip[:, b_idx])

# ══════════════════════════════════════════════════════════════════════════
# TOTAL W  →  T1 COMPONENTS
# ══════════════════════════════════════════════════════════════════════════
W_total = W_ph + W_dip


def extract_T1s(W):
    ev    = np.linalg.eigvals(W)
    rates = np.sort([-np.real(v) for v in ev
                     if -np.real(v) > np.max(np.abs(np.real(ev))) * 1e-6
                     and abs(np.imag(v)) < abs(np.real(v)) * 0.01])
    return [1/r for r in rates], list(rates)


T1_ph,  rates_ph  = extract_T1s(W_ph)
T1_tot, rates_tot = extract_T1s(W_total)

T1_long_ph,  T1_short_ph  = T1_ph[0],  T1_ph[-1]
T1_long_tot, T1_short_tot = T1_tot[0], T1_tot[-1]

# ══════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print(" Eu(dpphen)(NO3)3  —  153Eu NQR spin-lattice relaxation")
print("="*68)
print(f"\n  e²qQ = {E2QQ_MHZ:.3f} MHz    η = {ETA_Q:.6f}    A_iso = {A_ISO_MHZ:.4f} MHz")
print(f"  T    = {T_BATH} K    γ_ph = {GAM_PH_CM:.0f} cm⁻¹    DFT modes: {N_modes}")
print(f"  Bath: {len(H_spins_m)} ¹H (τ={tau_H*1e6:.1f} μs)  +  {len(N_spins_m)} ¹⁴N (τ={tau_N*1e6:.0f} μs)")

print(f"\n  NQR transition frequencies:")
for i, e in enumerate(evals):
    print(f"    |{i}⟩  {e/(2*np.pi*1e6):+9.4f} MHz")

print(f"\n  ┌────────────────────────────────────────────────────────┐")
print(f"  │                  T1_long    T1_short    Ratio          │")
print(f"  │  ──────────────────────────────────────────────────    │")
print(f"  │  Experiment       41.39 s      0.31 s   134×           │")
print(f"  │  Phonon only    {T1_long_ph:>7.2f} s    {T1_short_ph:>6.3f} s  {T1_long_ph/T1_short_ph:>5.1f}×          │")
print(f"  │  + dipolar bath {T1_long_tot:>7.2f} s    {T1_short_tot:>6.3f} s  {T1_long_tot/T1_short_tot:>5.1f}×          │")
print(f"  └────────────────────────────────────────────────────────┘")

print(f"\n  T1 components (full model):")
for i, (r, t) in enumerate(zip(rates_tot, T1_tot)):
    print(f"    λ{i+1}  {r:.4e} s⁻¹   T1 = {t:.3f} s")

W_ph_od  = W_ph.copy();  np.fill_diagonal(W_ph_od,  0)
W_dip_od = W_dip.copy(); np.fill_diagonal(W_dip_od, 0)
print(f"\n  Max off-diagonal rates:")
print(f"    Phonon:   {W_ph_od.max():.3e} s⁻¹")
print(f"    Dipolar:  {W_dip_od.max():.3e} s⁻¹")


# ══════════════════════════════════════════════════════════════════════════
# Secular cutoff sweep
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("  SECULAR CUTOFF SWEEP")
print("="*68)
sec_cutoffs  = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
sec_T1_long  = []
sec_T1_short = []
sec_ratio    = []

for sc in sec_cutoffs:
    W_sc   = build_W_ph(sec_cutoff_val=sc) + W_dip
    T1_sc, _ = extract_T1s(W_sc)
    if len(T1_sc) >= 2:
        sec_T1_long.append(T1_sc[0])
        sec_T1_short.append(T1_sc[-1])
        sec_ratio.append(T1_sc[0] / T1_sc[-1])
    else:
        sec_T1_long.append(np.nan)
        sec_T1_short.append(np.nan)
        sec_ratio.append(np.nan)
    print(f"  sec_cutoff={sc:5.3f}  T1_long={sec_T1_long[-1]:7.2f} s  "
          f"T1_short={sec_T1_short[-1]:7.3f} s  ratio={sec_ratio[-1]:6.1f}×")


# ══════════════════════════════════════════════════════════════════════════
# γ_ph sweep + 3. Raman % effect
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("  γ_ph SWEEP  (Raman linewidth sensitivity)")
print("="*68)
gam_vals_cm  = [1, 5, 10, 20, 50, 100, 200, 500]
gam_T1_long  = []
gam_T1_short = []
gam_ratio    = []

for gc in gam_vals_cm:
    W_gc   = build_W_ph(gam_ph_cm_val=gc) + W_dip
    T1_gc, _ = extract_T1s(W_gc)
    if len(T1_gc) >= 2:
        gam_T1_long.append(T1_gc[0])
        gam_T1_short.append(T1_gc[-1])
        gam_ratio.append(T1_gc[0] / T1_gc[-1])
    else:
        gam_T1_long.append(np.nan)
        gam_T1_short.append(np.nan)
        gam_ratio.append(np.nan)
    print(f"  γ_ph={gc:4d} cm⁻¹  T1_long={gam_T1_long[-1]:7.2f} s  "
          f"T1_short={gam_T1_short[-1]:7.3f} s  ratio={gam_ratio[-1]:6.1f}×")

# Raman % effect: change from 1 cm-1 to 500 cm-1
raman_effect_long  = 100 * (gam_T1_long[-1]  - gam_T1_long[0])  / gam_T1_long[0]
raman_effect_short = 100 * (gam_T1_short[-1] - gam_T1_short[0]) / gam_T1_short[0]
print(f"\n  [Raman % effect: γ_ph 1→500 cm⁻¹]")
print(f"    T1_long  shift: {raman_effect_long:+.1f}%")
print(f"    T1_short shift: {raman_effect_short:+.1f}%")
print(f"    The Raman linewidth γ_ph has a {'large' if abs(raman_effect_long)>20 else 'modest'} "
      f"effect on T1_long ({abs(raman_effect_long):.1f}% variation) and "
      f"a {'large' if abs(raman_effect_short)>20 else 'modest'} effect on T1_short "
      f"({abs(raman_effect_short):.1f}% variation) over the full range 1–500 cm⁻¹.")


# ══════════════════════════════════════════════════════════════════════════
# Artificial low-frequency phonon test
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("  ARTIFICIAL LOW-FREQUENCY PHONON TEST")
print("="*68)
# Use mean dV tensor of all modes as a proxy coupling for the injected mode
dV_mean = dV_all.mean(axis=0)

lf_freqs_cm = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
lf_T1_long  = []
lf_T1_short = []
lf_ratio    = []

for lf in lf_freqs_cm:
    extra = [(lf, dV_mean)]
    W_lf  = build_W_ph(extra_modes=extra) + W_dip
    T1_lf, _ = extract_T1s(W_lf)
    if len(T1_lf) >= 2:
        lf_T1_long.append(T1_lf[0])
        lf_T1_short.append(T1_lf[-1])
        lf_ratio.append(T1_lf[0] / T1_lf[-1])
    else:
        lf_T1_long.append(np.nan)
        lf_T1_short.append(np.nan)
        lf_ratio.append(np.nan)
    print(f"  LF phonon @ {lf:5.1f} cm⁻¹  T1_long={lf_T1_long[-1]:7.2f} s  "
          f"T1_short={lf_T1_short[-1]:7.3f} s  ratio={lf_ratio[-1]:6.1f}×")

print(f"\n  Baseline (no LF phonon): T1_long={T1_long_tot:.2f} s  "
      f"T1_short={T1_short_tot:.3f} s  ratio={T1_long_tot/T1_short_tot:.1f}×")


# ══════════════════════════════════════════════════════════════════════════
# τ_H and τ_N ± factor-of-2 sensitivity
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("  τ_H / τ_N  SENSITIVITY  (±2×)")
print("="*68)

tau_factors = [0.25, 0.5, 1.0, 2.0, 4.0]
tau_labels  = ['÷4', '÷2', '×1 (nominal)', '×2', '×4']

sens_results = {}
for label, factor_H, vary_H in [('τ_H varied', True, True),
                                  ('τ_N varied', False, True)]:
    row_long  = []
    row_short = []
    row_ratio = []
    for fac in tau_factors:
        t_H = tau_H * (fac if vary_H and label == 'τ_H varied' else 1.0)
        t_N = tau_N * (fac if not vary_H and label == 'τ_N varied' else 1.0)
        # For τ_N varied, fix τ_H and vice-versa
        if label == 'τ_H varied':
            t_H = tau_H * fac
            t_N = tau_N
        else:
            t_H = tau_H
            t_N = tau_N * fac
        W_s = W_ph + build_W_dip(t_H, t_N)
        T1_s, _ = extract_T1s(W_s)
        if len(T1_s) >= 2:
            row_long.append(T1_s[0]);  row_short.append(T1_s[-1])
            row_ratio.append(T1_s[0]/T1_s[-1])
        else:
            row_long.append(np.nan);  row_short.append(np.nan);  row_ratio.append(np.nan)
    sens_results[label] = (row_long, row_short, row_ratio)
    print(f"\n  {label}:")
    for lbl, tl, ts, tr in zip(tau_labels, row_long, row_short, row_ratio):
        marker = ' ◄ nominal' if lbl == '×1 (nominal)' else ''
        print(f"    {lbl:18s}  T1_long={tl:7.2f} s  T1_short={ts:7.3f} s  ratio={tr:6.1f}×{marker}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════
t_arr = np.concatenate([np.linspace(0, 2, 80),
                        np.linspace(2, 100, 80)[1:],
                        np.linspace(100, 400, 60)[1:]])

# ─── Figure 1: 6-panel ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    r"Eu(dpphen)(NO$_3$)$_3$ — $^{153}$Eu NQR spin-lattice relaxation"
    "\n"
    r"$e^2qQ=$" + f"{E2QQ_MHZ:.1f} MHz,  "
    r"$\eta=$"  + f"{ETA_Q:.4f},  "
    r"$T=$"     + f"{T_BATH} K",
    fontsize=12, fontweight='bold')

# (a) Recovery curves
ax = axes[0, 0]
h_ph  = sum(np.exp(-t_arr/t1) for t1 in T1_ph)  / len(T1_ph)
h_tot = sum(np.exp(-t_arr/t1) for t1 in T1_tot) / len(T1_tot)
ax.semilogy(t_arr, h_ph,  'steelblue',  lw=2.5, label='Phonon only')
ax.semilogy(t_arr, h_tot, 'darkorange', lw=2.5, label='+ Dipolar bath')
ax.axvline(41.39, color='crimson', ls=':', lw=1.5, label='Exp. T1_long = 41 s')
ax.axvline(0.31,  color='navy',    ls=':', lw=1.5, label='Exp. T1_short = 0.31 s')
ax.set_xlabel('Time (s)'); ax.set_ylabel('h(t)')
ax.set_title('(a) Recovery curve')
ax.legend(fontsize=8); ax.set_xlim(0, 300); ax.set_ylim(1e-3, 1)
ax.grid(True, alpha=0.2)

# (b) T1 components
ax = axes[0, 1]
ax.bar(np.arange(1, len(T1_ph)+1)  - 0.2, T1_ph,  width=0.35,
       color='steelblue',  alpha=0.8, label='Phonon')
ax.bar(np.arange(1, len(T1_tot)+1) + 0.2, T1_tot, width=0.35,
       color='darkorange', alpha=0.8, label='+ Dipolar')
ax.axhline(41.39, color='crimson', ls='--', lw=1.5, label='Exp. 41 s')
ax.axhline(0.31,  color='navy',    ls='--', lw=1.5, label='Exp. 0.31 s')
ax.set_xlabel('Eigenmode (1 = slowest)'); ax.set_ylabel('T1 (s)')
ax.set_title('(b) T1 components')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

# (c) W_total heatmap
ax = axes[0, 2]
W_show = np.abs(W_total.copy()); np.fill_diagonal(W_show, 0)
im = ax.imshow(np.log10(W_show + 1e-30), cmap='viridis', aspect='auto')
plt.colorbar(im, ax=ax, label=r'$\log_{10}|W_{ab}|$ (s⁻¹)')
lbl = [f'|{i}⟩' for i in range(DIM)]
ax.set_xticks(range(DIM)); ax.set_xticklabels(lbl)
ax.set_yticks(range(DIM)); ax.set_yticklabels(lbl)
ax.set_title('(c) Rate matrix W (off-diagonal)')

# (d) Bath spin geometry
ax = axes[1, 0]
H_xyz = np.array([[x,y,z] for (el,I,x,y,z) in bath_spins_raw if el=='H'])
N_xyz = np.array([[x,y,z] for (el,I,x,y,z) in bath_spins_raw if el=='N'])
ax.scatter(H_xyz[:,0], H_xyz[:,1], s=80,  c='steelblue', label='¹H (I=½)',
           edgecolors='k', lw=0.5)
ax.scatter(N_xyz[:,0], N_xyz[:,1], s=120, c='tomato', marker='D',
           label='¹⁴N (I=1)', edgecolors='k', lw=0.5)
ax.scatter(0, 0, s=250, c='gold', marker='*', label='Eu',
           edgecolors='k', lw=0.8, zorder=5)
ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
ax.set_title('(d) Bath spin geometry (xy projection)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_aspect('equal')

# (e) Dipolar rate per bath spin
ax = axes[1, 1]
W_per, labels, colors = [], [], []
for idx, (el, I_b, x, y, z) in enumerate(bath_spins_raw):
    pos  = [np.array([x,y,z]) * ANG2M]
    tau  = tau_H if el == 'H' else tau_N
    W_s  = dipolar_W_contribution(pos, GAMMA[el], I_b, tau)
    W_per.append(np.sum(np.abs(W_s[~np.eye(DIM, dtype=bool)])))
    labels.append(f"{el}{idx+1}\n{np.sqrt(x**2+y**2+z**2):.1f}Å")
    colors.append('steelblue' if el == 'H' else 'tomato')
ax.bar(range(len(W_per)), W_per, color=colors, edgecolor='k', lw=0.5)
ax.set_xticks(range(len(W_per)))
ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
ax.set_ylabel('Total rate contribution (s⁻¹)')
ax.set_title('(e) Dipolar contribution per bath spin')
ax.grid(True, alpha=0.2, axis='y')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='steelblue', label='¹H'),
                   Patch(facecolor='tomato',    label='¹⁴N')], fontsize=8)

# (f) Ratio summary
ax = axes[1, 2]
models = ['Experiment', 'Phonon\nonly', '+ Dipolar\nbath', '+ Crystal\nphonons\n(needed)']
ratios = [134, T1_long_ph/T1_short_ph, T1_long_tot/T1_short_tot, 134]
colors_f = ['crimson', 'steelblue', 'darkorange', 'gray']
bars   = ax.bar(models, ratios, color=colors_f, edgecolor='k', lw=0.8)
bars[-1].set_alpha(0.3)
ax.axhline(134, color='crimson', ls='--', lw=1.5, alpha=0.5)
for bar, ratio in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width()/2, ratio + 1.5,
            f'{ratio:.1f}×', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_ylabel(r'$T_{1,\mathrm{long}}\,/\,T_{1,\mathrm{short}}$')
ax.set_title('(f) T1 ratio progression')
ax.grid(True, alpha=0.2, axis='y'); ax.set_ylim(0, 155)

plt.tight_layout()
plt.show()


# ─── Figure 2: More analyses ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle(
    r"Eu(dpphen)(NO$_3$)$_3$ — Sensitivity & parameter sweep analyses",
    fontsize=13, fontweight='bold')

# ── Panel A: Secular cutoff sweep ────────────────────────────────────────
ax = axes2[0, 0]
ax.semilogx(sec_cutoffs, sec_T1_long,  'o-', color='steelblue',  lw=2,
            ms=7, label=r'$T_{1,\mathrm{long}}$')
ax.semilogx(sec_cutoffs, sec_T1_short, 's-', color='darkorange', lw=2,
            ms=7, label=r'$T_{1,\mathrm{short}}$')
ax2r = ax.twinx()
ax2r.semilogx(sec_cutoffs, sec_ratio, '^--', color='forestgreen', lw=1.5,
              ms=6, alpha=0.7, label='Ratio')
ax2r.set_ylabel(r'$T_{1,\mathrm{long}}/T_{1,\mathrm{short}}$', color='forestgreen')
ax2r.tick_params(axis='y', labelcolor='forestgreen')
ax.axvline(0.1, color='gray', ls=':', lw=1.5, label='Nominal (0.1)')
ax.axhline(41.39, color='crimson',  ls='--', lw=1, alpha=0.6)
ax.axhline(0.31,  color='navy',     ls='--', lw=1, alpha=0.6)
ax.set_xlabel(r'Secular cutoff (rad s$^{-1}$)')
ax.set_ylabel('T1 (s)')
ax.set_title('(A) Secular cutoff sweep')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper left')
ax.grid(True, alpha=0.2)

# ── Panel B: γ_ph sweep ──────────────────────────────────────────────────
ax = axes2[0, 1]
ax.semilogx(gam_vals_cm, gam_T1_long,  'o-', color='steelblue',  lw=2,
            ms=7, label=r'$T_{1,\mathrm{long}}$')
ax.semilogx(gam_vals_cm, gam_T1_short, 's-', color='darkorange', lw=2,
            ms=7, label=r'$T_{1,\mathrm{short}}$')
ax_r = ax.twinx()
ax_r.semilogx(gam_vals_cm, gam_ratio, '^--', color='forestgreen', lw=1.5,
              ms=6, alpha=0.7, label='Ratio')
ax_r.set_ylabel(r'$T_{1,\mathrm{long}}/T_{1,\mathrm{short}}$', color='forestgreen')
ax_r.tick_params(axis='y', labelcolor='forestgreen')
ax.axvline(GAM_PH_CM, color='gray', ls=':', lw=1.5, label=f'Nominal ({GAM_PH_CM} cm⁻¹)')
ax.axhline(41.39, color='crimson',  ls='--', lw=1, alpha=0.6)
ax.axhline(0.31,  color='navy',     ls='--', lw=1, alpha=0.6)
ax.set_xlabel(r'$\gamma_\mathrm{ph}$ (cm$^{-1}$)')
ax.set_ylabel('T1 (s)')
ax.set_title(r'(B) $\gamma_\mathrm{ph}$ sweep (Raman linewidth)')
# Annotate Raman % effect
ax.text(0.98, 0.05,
        f"Raman effect (1→500 cm⁻¹):\n"
        f"  T₁_long:  {raman_effect_long:+.1f}%\n"
        f"  T₁_short: {raman_effect_short:+.1f}%",
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper right',
          bbox_to_anchor=(0.98, 0.98))
ax.grid(True, alpha=0.2)

# ── Panel C: Low-frequency phonon test ───────────────────────────────────
ax = axes2[1, 0]
ax.semilogx(lf_freqs_cm, lf_T1_long,  'o-', color='steelblue',  lw=2,
            ms=7, label=r'$T_{1,\mathrm{long}}$')
ax.semilogx(lf_freqs_cm, lf_T1_short, 's-', color='darkorange', lw=2,
            ms=7, label=r'$T_{1,\mathrm{short}}$')
ax3r = ax.twinx()
ax3r.semilogx(lf_freqs_cm, lf_ratio, '^--', color='forestgreen', lw=1.5,
              ms=6, alpha=0.7, label='Ratio')
ax3r.set_ylabel(r'$T_{1,\mathrm{long}}/T_{1,\mathrm{short}}$', color='forestgreen')
ax3r.tick_params(axis='y', labelcolor='forestgreen')
ax.axhline(T1_long_tot,  color='steelblue',  ls=':', lw=1.5, alpha=0.5,
           label='Baseline (no LF)')
ax.axhline(T1_short_tot, color='darkorange', ls=':', lw=1.5, alpha=0.5)
ax.axhline(41.39, color='crimson', ls='--', lw=1, alpha=0.6)
ax.axhline(0.31,  color='navy',    ls='--', lw=1, alpha=0.6)
ax.set_xlabel(r'Injected LF phonon frequency (cm$^{-1}$)')
ax.set_ylabel('T1 (s)')
ax.set_title('(C) Artificial low-frequency phonon test\n(mean-coupling mode injected)')
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax3r.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
ax.grid(True, alpha=0.2)

# ── Panel D: τ_H / τ_N factor-of-2 sensitivity ───────────────────────────
ax = axes2[1, 1]
x_pos    = np.log2(tau_factors)  # -2, -1, 0, 1, 2
x_labels = tau_labels

tH_long,  tH_short,  tH_ratio  = sens_results['τ_H varied']
tN_long,  tN_short,  tN_ratio  = sens_results['τ_N varied']

ax.plot(x_pos, tH_long,  'o-',  color='steelblue',   lw=2, ms=7,
        label=r'$T_{1,\mathrm{long}}$ (τ$_H$ varied)')
ax.plot(x_pos, tH_short, 's--', color='steelblue',   lw=2, ms=7, alpha=0.6,
        label=r'$T_{1,\mathrm{short}}$ (τ$_H$ varied)')
ax.plot(x_pos, tN_long,  'o-',  color='tomato',      lw=2, ms=7,
        label=r'$T_{1,\mathrm{long}}$ (τ$_N$ varied)')
ax.plot(x_pos, tN_short, 's--', color='tomato',      lw=2, ms=7, alpha=0.6,
        label=r'$T_{1,\mathrm{short}}$ (τ$_N$ varied)')

ax4r = ax.twinx()
ax4r.plot(x_pos, tH_ratio, '^:', color='steelblue', lw=1.5, ms=5, alpha=0.5)
ax4r.plot(x_pos, tN_ratio, '^:', color='tomato',    lw=1.5, ms=5, alpha=0.5)
ax4r.set_ylabel(r'$T_{1,\mathrm{long}}/T_{1,\mathrm{short}}$ (dotted)',
                color='gray', fontsize=8)
ax4r.tick_params(axis='y', labelcolor='gray')

ax.axhline(41.39, color='crimson', ls='--', lw=1, alpha=0.6, label='Exp. 41 s')
ax.axhline(0.31,  color='navy',    ls='--', lw=1, alpha=0.6, label='Exp. 0.31 s')
ax.axvline(0, color='gray', ls=':', lw=1.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_xlabel(r'Scaling factor for $\tau$ (log$_2$ axis)')
ax.set_ylabel('T1 (s)')
ax.set_title(r'(D) $\tau_H$ and $\tau_N$ ± factor-of-2 sensitivity')
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print("\nCOMPLETE")

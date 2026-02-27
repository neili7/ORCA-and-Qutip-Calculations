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
    # (element, I, x[Å], y[Å], z[Å]) — relative to Eu, from DFT cluster
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
    '153Eu': 6.6252e6,    # rad/s/T
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
# PHONON W MATRIX  (QuTiP Bloch-Redfield tensor)
# ══════════════════════════════════════════════════════════════════════════
T_eb_qt = np.array([[U.conj().T @ T_cart_np[i, j] @ U
                      for j in range(3)] for i in range(3)])

H_diag_qt = qutip.Qobj(np.diag(evals), dims=[[DIM], [DIM]])

a_ops = []
for k in range(N_modes):
    A_k = np.zeros((DIM, DIM), dtype=complex)
    for i in range(3):
        for j in range(3):
            A_k += dV_all[k, i, j] * T_eb_qt[i, j]
    A_k_qt = qutip.Qobj(A_k * PREFACTOR * SCALE_SI, dims=[[DIM], [DIM]])

    ok, nk, gph = omega_k[k], n_k[k], GAM_PH
    def make_S(ok=ok, nk=nk, gph=gph):
        def S(omega):
            Lp = (gph / np.pi) / ((omega + ok)**2 + gph**2)
            Lm = (gph / np.pi) / ((omega - ok)**2 + gph**2)
            return (hbar / (2 * ok)) * ((nk + 1) * Lp + nk * Lm)
        return S
    a_ops.append([A_k_qt, make_S()])

R_qt, _ = qutip.bloch_redfield_tensor(H_diag_qt, a_ops,
                                       sec_cutoff=0.1,
                                       sparse_eigensolver=False)
R_np = R_qt.full()

W_ph = np.zeros((DIM, DIM))
for a in range(DIM):
    for b in range(DIM):
        if a != b:
            W_ph[a, b] = np.real(R_np[a * DIM + a, b * DIM + b])
for b in range(DIM):
    W_ph[b, b] = -np.sum(W_ph[:, b])

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
# FIGURE
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    r"Eu(dpphen)(NO$_3$)$_3$ — $^{153}$Eu NQR spin-lattice relaxation"
    "\n"
    r"$e^2qQ=$" + f"{E2QQ_MHZ:.1f} MHz,  "
    r"$\eta=$"  + f"{ETA_Q:.4f},  "
    r"$T=$"     + f"{T_BATH} K",
    fontsize=12, fontweight='bold')

t_arr = np.concatenate([np.linspace(0, 2, 80),
                        np.linspace(2, 100, 80)[1:],
                        np.linspace(100, 400, 60)[1:]])

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
colors = ['crimson', 'steelblue', 'darkorange', 'gray']
bars   = ax.bar(models, ratios, color=colors, edgecolor='k', lw=0.8)
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
print("\nCOMPLETE")

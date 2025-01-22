import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 9.81
L = 10.0
Nx = 200
dx = L / Nx
x_grid = np.linspace(0.5*dx, L - 0.5*dx, Nx)

# Temps
tmax = 7
CFL  = 0.5
dt_max = 0.01  # borne supérieure
dt = 1e-3      # dt initial

def topo(x):
    """
    Topographie en deux zones:
      - De x=0 à x=4 : pente linéaire décroissant de 0.5 à 0.
      - Au-delà de x=4 : fond plat à 0.
    """
    x_pente = 4.0
    bmax = 0.5
    b = np.zeros_like(x)
    mask_pente = (x < x_pente)
    # Sur la pente : b(x) = 0.5 - 0.5*(x/4)
    b[mask_pente] = bmax * (1.0 - x[mask_pente]/x_pente)
    return b

b = topo(x_grid)

h_init = np.zeros(Nx)
u_init = np.zeros(Nx)

x_disc = 4

for i in range(Nx):
    if x_grid[i] < x_disc:
        h_init[i] = 0.9 - b[i]  # hauteur "initiale" ajustée de la topographie
        u_init[i] = 0.45
    else:
        h_init[i] = 0.1
        u_init[i] = 0.1

def make_U(h, u):
    return np.vstack((h, h*u))

U_rus = make_U(h_init, u_init)
U_hll = make_U(h_init, u_init)
U_roe = make_U(h_init, u_init)

h_in, u_in = h_init[0], u_init[0]
U_left = np.array([h_in, h_in*u_in])  # vecteur (h, hu)

def compute_u(U):
    h  = U[0,:]
    hu = U[1,:]
    u  = np.zeros_like(h)
    mask = (h>1e-10)
    u[mask] = hu[mask]/h[mask]
    return u

def compute_flux(U):
    """
    F(U) = ( h*u, h*u^2 + 0.5*g*h^2 ).
    """
    h  = U[0,:]
    u  = compute_u(U)
    F = np.zeros_like(U)
    F[0,:] = h*u
    F[1,:] = h*u**2 + 0.5*g*h**2
    return F

def source_topo(U, b, dx):
    """
    Terme source = -g h d(b)/dx sur la 2e eq.
    """
    h = U[0,:]
    Nx_local = len(h)
    db_dx = np.zeros(Nx_local)
    for i in range(1, Nx_local-1):
        db_dx[i] = (b[i+1] - b[i-1])/(2*dx)
    db_dx[0]   = (b[1] - b[0])/dx
    db_dx[-1]  = (b[-1] - b[-2])/dx

    S = np.zeros_like(U)
    S[1,:] = -g * h * db_dx
    return S

def flux_rusanov(U_L, U_R):
    F_L = compute_flux(U_L.reshape(2,1))[:,0]
    F_R = compute_flux(U_R.reshape(2,1))[:,0]
    hL, huL = U_L
    hR, huR = U_R
    uL = huL/(hL + 1e-10)
    uR = huR/(hR + 1e-10)
    cL = np.sqrt(g*hL)
    cR = np.sqrt(g*hR)
    smax = max(abs(uL)+cL, abs(uR)+cR)
    return 0.5*(F_L + F_R) - 0.5*smax*(U_R - U_L)

def flux_hll(U_L, U_R):
    F_L = compute_flux(U_L.reshape(2,1))[:,0]
    F_R = compute_flux(U_R.reshape(2,1))[:,0]
    hL, huL = U_L
    hR, huR = U_R
    uL = huL/(hL + 1e-10)
    uR = huR/(hR + 1e-10)
    cL = np.sqrt(g*hL)
    cR = np.sqrt(g*hR)
    SL = min(uL-cL, uR-cR)
    SR = max(uL+cL, uR+cR)
    if SL>=0:
        return F_L
    elif SR<=0:
        return F_R
    else:
        return (SR*F_L - SL*F_R + SL*SR*(U_R - U_L)) / (SR-SL)

def flux_roe(U_L, U_R):
    F_L = compute_flux(U_L.reshape(2,1))[:,0]
    F_R = compute_flux(U_R.reshape(2,1))[:,0]
    hL, huL = U_L
    hR, huR = U_R
    uL = huL/(hL + 1e-10)
    uR = huR/(hR + 1e-10)
    # moyenne de Roe
    sqrt_hL = np.sqrt(hL + 1e-10)
    sqrt_hR = np.sqrt(hR + 1e-10)
    h_tilde = sqrt_hL * sqrt_hR
    u_tilde = (sqrt_hL*uL + sqrt_hR*uR) / (sqrt_hL + sqrt_hR + 1e-10)
    c_tilde = np.sqrt(g*h_tilde)
    dU = U_R - U_L
    smax = abs(u_tilde)+c_tilde
    return 0.5*(F_L+F_R) - 0.5*smax*dU

def update_timestep(U, flux_func, dt):
    """
    Mise à jour d'un pas de temps pour un schéma explicite
    U_j^{n+1} = U_j^n - dt/dx (F_{j+1/2}-F_{j-1/2}) + dt * S(b).
    + Dirichlet à gauche (on impose U_left).
    """
    N = U.shape[1]

    # 1) flux numerique
    Fnum = np.zeros((2, N+1))
    for j in range(1, N):
        Fnum[:, j] = flux_func(U[:, j-1], U[:, j])

    # BC gauche : Dirichlet => flux = flux_func(U_left, U[:,0])
    Fnum[:,0]   = flux_func(U_left, U[:,0])

    # BC droite : on peut faire flux nul ou un "copy"
    Fnum[:, -1] = flux_func(U[:, -1], U[:, -1])

    # 2) source
    S_topo = source_topo(U, b, dx)

    # 3) update
    U_new = U.copy()
    for j in range(N):
        U_new[:, j] = ( U[:, j]
                        - (dt/dx)*(Fnum[:, j+1] - Fnum[:, j])
                        + dt*S_topo[:, j] )
    # on ré-impose la condition Dirichlet à gauche
    U_new[:, 0] = U_left
    # on peut forcer la frontière droite à rester la même qu'en j-1
    U_new[:, -1] = U_new[:, -2]

    return U_new

time_data = []
U_rus_data = []
U_hll_data = []
U_roe_data = []

t = 0.0
U_rus_current = U_rus.copy()
U_hll_current = U_hll.copy()
U_roe_current = U_roe.copy()

while t < tmax:
    # stockage
    time_data.append(t)
    U_rus_data.append(U_rus_current.copy())
    U_hll_data.append(U_hll_current.copy())
    U_roe_data.append(U_roe_current.copy())

    # calcul dt en CFL (ex: on se base sur la solution Rusanov)
    h_loc = U_rus_current[0,:]
    u_loc = compute_u(U_rus_current)
    c_loc = np.sqrt(g*h_loc)
    smax  = np.max(np.abs(u_loc)+c_loc)
    if smax < 1e-8:
        smax = 1e-8
    dt_cfl = CFL * dx / smax
    dt = min(dt_cfl, dt_max)

    if t+dt > tmax:
        dt = tmax - t

    # mise a jour
    U_rus_current = update_timestep(U_rus_current, flux_rusanov, dt)
    U_hll_current = update_timestep(U_hll_current, flux_hll, dt)
    U_roe_current = update_timestep(U_roe_current, flux_roe, dt)

    t += dt

# dernier snapshot
time_data.append(t)
U_rus_data.append(U_rus_current.copy())
U_hll_data.append(U_hll_current.copy())
U_roe_data.append(U_roe_current.copy())

time_data   = np.array(time_data)
U_rus_data  = np.array(U_rus_data)  # shape=(n_snap, 2, Nx)
U_hll_data  = np.array(U_hll_data)
U_roe_data  = np.array(U_roe_data)

def get_h_plus_b(U_data):
    return U_data[:, 0, :] + b  # shape=(n_snap, Nx)

def get_u(U_data):
    # on doit calculer la vitesse snapshot par snapshot
    n_snap = U_data.shape[0]
    Nx_loc = U_data.shape[2]
    out = np.zeros((n_snap, Nx_loc))
    for i in range(n_snap):
        h_i  = U_data[i,0,:]
        hu_i = U_data[i,1,:]
        mask = (h_i>1e-10)
        out[i,mask] = hu_i[mask]/h_i[mask]
    return out

rus_hb = get_h_plus_b(U_rus_data)  # shape=(n_snap, Nx)
hll_hb = get_h_plus_b(U_hll_data)
roe_hb = get_h_plus_b(U_roe_data)

rus_u  = get_u(U_rus_data)
hll_u  = get_u(U_hll_data)
roe_u  = get_u(U_roe_data)

all_hb = np.concatenate([rus_hb, hll_hb, roe_hb], axis=0)
all_u  = np.concatenate([rus_u,   hll_u,   roe_u],   axis=0)
min_hb = np.min(all_hb)
max_hb = np.max(all_hb)
min_u  = np.min(all_u)
max_u  = np.max(all_u)

delta_hb = 0.1*(max_hb - min_hb) if max_hb>min_hb else 0.1
delta_u  = 0.1*(max_u - min_u)   if max_u>min_u else 0.1

ylim_h  = (min_hb - delta_hb, max_hb + delta_hb)
ylim_u  = (min_u  - delta_u , max_u  + delta_u )


fig, axs = plt.subplots(2, 1, figsize=(8,6))
ax_h, ax_u = axs

# On tracera 3 courbes pour la hauteur + topographie
line_rus_h, = ax_h.plot(x_grid, rus_hb[0,:], label='Rusanov', color='blue')
line_hll_h, = ax_h.plot(x_grid, hll_hb[0,:], label='HLL', linestyle='--', color='red')
line_roe_h, = ax_h.plot(x_grid, roe_hb[0,:], label='Roe', linestyle='-.', color='green')
line_b,     = ax_h.plot(x_grid, b, 'k:', label='topo')
ax_h.legend()
ax_h.set_xlim(0, L)
ax_h.set_ylim(ylim_h)
ax_h.set_ylabel('h + b')

# Pareil pour la vitesse
line_rus_u, = ax_u.plot(x_grid, rus_u[0,:], color='blue')
line_hll_u, = ax_u.plot(x_grid, hll_u[0,:], linestyle='--', color='red')
line_roe_u, = ax_u.plot(x_grid, roe_u[0,:], linestyle='-.', color='green')
ax_u.set_xlim(0, L)
ax_u.set_ylim(ylim_u)
ax_u.set_ylabel('u (m/s)')
ax_u.set_xlabel('x (m)')

time_text = ax_h.text(0.05, 0.9, '', transform=ax_h.transAxes)

def init_anim():
    line_rus_h.set_ydata(rus_hb[0,:])
    line_hll_h.set_ydata(hll_hb[0,:])
    line_roe_h.set_ydata(roe_hb[0,:])

    line_rus_u.set_ydata(rus_u[0,:])
    line_hll_u.set_ydata(hll_u[0,:])
    line_roe_u.set_ydata(roe_u[0,:])

    time_text.set_text(f't = {time_data[0]:.3f}s')
    return (line_rus_h, line_hll_h, line_roe_h,
            line_rus_u, line_hll_u, line_roe_u, time_text)

def update_anim(frame):
    line_rus_h.set_ydata(rus_hb[frame,:])
    line_hll_h.set_ydata(hll_hb[frame,:])
    line_roe_h.set_ydata(roe_hb[frame,:])

    line_rus_u.set_ydata(rus_u[frame,:])
    line_hll_u.set_ydata(hll_u[frame,:])
    line_roe_u.set_ydata(roe_u[frame,:])

    time_text.set_text(f't = {time_data[frame]:.3f}s')
    return (line_rus_h, line_hll_h, line_roe_h,
            line_rus_u, line_hll_u, line_roe_u, time_text)

ani2 = animation.FuncAnimation(
    fig, update_anim, frames=len(time_data),
    interval=50, blit=True, init_func=init_anim
)

ani2.save("SimulationExp2.mp4", fps=60)

plt.tight_layout()
plt.show()

# Extrait d'une figure simple à t=0
plt.plot(x_grid, rus_hb[0,:], label='Rusanov (t=0)', color='blue')
plt.plot(x_grid, b, 'k:', label='Topographie')
plt.title("Hauteur initiale + topographie")
plt.xlabel("x (m)")
plt.ylabel("h + b (m)")
plt.legend()
plt.show()

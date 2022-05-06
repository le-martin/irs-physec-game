#%%
from datetime import datetime
from fractions import Fraction
import itertools

# import nashpy as nash
import numpy as np
import pygambit
import scipy.linalg as la
from scipy.io import loadmat


# Input parameters
l_theta_bob = 5
n_bob = 4
l_theta_eve = 5
n_eve = 4
mat_file = 'h_data.mat'

#TODO: Import those as well
psd_n0 = -174
psd_n0 = 10 ** ((psd_n0-30)/10)
p_max = 46
p_max = 10 ** ((p_max-30)/10)
m = 3  # number of antennas


# Channel matrices generated using the above input parameters
channel = loadmat(mat_file)
h_AB = channel['h_AB']
h_AE = channel['h_AE']
h_A_IB = channel['h_A_IB']
h_A_IE = channel['h_A_IE']
h_IB_B = channel['h_IB_B']
h_IB_E = channel['h_IB_E']
h_IE_B = channel['h_IE_B']
h_IE_E = channel['h_IE_E']
noise = channel['Noise'][0]


def get_feasible_thetas(step: int) -> np.ndarray:
    feasible_thetas = [np.exp(2j * i * np.pi / step) for i in range(step)]

    return feasible_thetas


def get_beamforming(noise, p_max, m, h_AB, h_AE, h_A_IB, h_A_IE, h_IB_B, h_IB_E, h_IE_B, h_IE_E, theta_B_mat,
                    theta_E_mat):
    w_mat = []
    for theta_B, theta_E in zip(theta_B_mat, theta_E_mat):
        h_B = h_AB + h_IB_B @ np.diag(theta_B) @ h_A_IB + h_IE_B @ np.diag(theta_E) @ h_A_IE
        h_B_tilde = (1 / noise[0]) * h_B.conj().T @ h_B
        h_E = h_AE + h_IE_E @ np.diag(theta_E) @ h_A_IE + h_IB_E @ np.diag(theta_B) @ h_A_IB
        h_E_tilde = (1 / noise[1]) * h_E.conj().T @ h_E
        u_term = np.linalg.inv(h_E_tilde + p_max ** -1 * np.eye(m)) @ (h_B_tilde + p_max ** -1 * np.eye(m))
        [e_u, v_u] = la.eig(h_B_tilde + p_max ** -1 * np.eye(m), h_E_tilde + p_max ** -1 * np.eye(m))
        v = np.max(e_u)
        index = np.argmax(e_u)
        vec_eig = v_u[:, index]
        w = p_max * vec_eig / la.norm(vec_eig)
        w_mat.append(w)

    return np.array(w_mat)


# Feasible Thetas for Bob and Eve
feasible_thetas_bob = [np.exp(2j * idx * np.pi / l_theta_bob) for idx in range(l_theta_bob)]
feasible_thetas_eve = [np.exp(2j * idx * np.pi / l_theta_eve) for idx in range(l_theta_eve)]

theta_combinations_bob = list(itertools.product(*([feasible_thetas_bob]*n_bob)))
theta_combinations_eve = list(itertools.product(*([feasible_thetas_eve]*n_eve)))
theta_combinations = np.array(list(itertools.product(theta_combinations_bob, theta_combinations_eve)))


# Compute optimal beam forming for all theta combinations
w_mat = get_beamforming(noise=noise, p_max=p_max, m=m, h_AB=h_AB, h_AE=h_AE, h_A_IB=h_A_IB, h_A_IE=h_A_IE,
                        h_IB_B=h_IB_B, h_IE_E=h_IE_E, h_IB_E=h_IB_E, h_IE_B=h_IE_B,
                        theta_B_mat=theta_combinations[:, 0], theta_E_mat=theta_combinations[:, 1])

h_B_mat = [h_AB + h_IB_B @ np.diag(theta_combinations[i, 0, :]) @ h_A_IB + h_IE_B @ np.diag(theta_combinations[i, 1, :]) @ h_A_IE for i in range(len(theta_combinations))]
r_Bob = np.array([np.log2(1 + np.abs(h_B @ w) ** 2 / noise[0]) for h_B, w in zip(h_B_mat, w_mat)])
h_E_mat = [h_AE + h_IE_E @ np.diag(theta_combinations[i, 1, :]) @ h_A_IE + h_IB_E @ np.diag(theta_combinations[i, 0, :]) @ h_A_IB for i in range(len(theta_combinations))]
r_Eve = np.array([np.log2(1 + np.abs(h_E @ w) ** 2 / noise[1]) for h_E, w in zip(h_E_mat, w_mat)])
r_s = r_Bob - r_Eve

payoff_mat = r_s.reshape(len(theta_combinations_bob), len(theta_combinations_eve))

#%%
start = datetime.now()

print(0)
make_fraction = np.vectorize(lambda x: Fraction(x))
payoff_mat2 = make_fraction(payoff_mat)

print(1)
g = pygambit.Game.from_arrays(payoff_mat2, -payoff_mat2)
g.title = "RIS-PHYSEC"
g.players[0].label = "Bob"
g.players[1].label = "Eve"
for i in range(len(g.players[0].strategies)):
    g.players[0].strategies[i].label = str(theta_combinations_bob[i])
for i in range(len(g.players[1].strategies)):
    g.players[1].strategies[i].label = str(theta_combinations_eve[i])

print(2)
# nash_eq = pygambit.nash.lcp_solve(g)
solver = pygambit.nash.ExternalLPSolver()
nash_eq = solver.solve(g)
#%%
print(3)
nash_eq2 = []
for ne in range(len(nash_eq)):
    ne_profile = []
    for player in range(len(g.players)):
        player_profile = []
        for prob in nash_eq[ne][g.players[player]]:
            player_profile.append(float(prob))
        ne_profile.append(player_profile)
    nash_eq2.append(ne_profile)

end = datetime.now() - start
print(f"Simulation took {end} seconds.")
print(nash_eq[0].payoff())

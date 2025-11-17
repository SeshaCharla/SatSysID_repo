import numpy as np
import cvxpy as cp
from scipy.stats import halfnorm, goodness_of_fit, gaussian_kde

# ==============================================================================

def solve_QP(Phi:np.ndarray, H:np.ndarray, W:np.ndarray, verbose=False):
        """ Solve the quadratic programming problem with Phi and H """
        P = 2 * Phi.T @ (W**2) @ Phi
        q = 2 * (H.T @ (W**2) @ Phi).T
        h = np.vstack([-H, np.zeros([3, 1])])
        parm_signs = np.eye(3)
        parm_signs[0, 0] = -1
        G = np.vstack([-Phi, -parm_signs])
        # ===
        # Convex optimization problem
        theta = cp.Variable([3, 1])
        objective = cp.Minimize( (1/2)*cp.quad_form(theta, P) - q.T @ theta )
        constraints = [G @ theta <= h]
        prob = cp.Problem(objective=objective, constraints=constraints)
        # Convex optimization problem
        # theta = cp.Variable([3, 1])
        # objective = cp.Minimize(cp.sum_squares(Phi@theta-H))
        # constraints = [Phi@theta >= H,
        #                theta[0, 0] <= 0,
        #                theta[1, 0] >= 0,
        #                theta[2, 0] >= 0]
        # prob = cp.Problem(objective=objective, constraints=constraints)
        prob.solve(solver='MOSEK', verbose=verbose) #,
        #===
        # Solution
        return theta.value

# ==============================================================================

def solve_LP(Phi:np.ndarray, H:np.ndarray, W:np.ndarray, verbose=False):
        """ Solve the quadratic programming problem with Phi and H """
        # Convex optimization problem
        theta = cp.Variable([3, 1])
        objective = cp.Minimize(cp.sum(W@Phi@theta))
        constraints = [Phi@theta >= H,
                       theta[0, 0] <= 0,
                       theta[1, 0] >= 0,
                       theta[2, 0] >= 0]
        prob = cp.Problem(objective=objective, constraints=constraints)
        prob.solve(solver="MOSEK", verbose=verbose) #solver='MOSEK',
        #===
        # Solution
        return theta.value


# ==============================================================================

def PhiSat_mat(T, F, u1):
        """Returns the regression matrix for the given series or T, F and u1
           eta[k+1] = (u1[k]/F[k]) * [T**2 T 1] * [th1, th2 th3]^T
        """
        N = len(T)
        PhiSat = np.zeros([N, 3])
        # Looping
        for i in range(N):
                PhiSat[i, :] = (u1[i]/F[i]) * np.array([T[i]**2, T[i], 1])
        #===
        return PhiSat

# ==============================================================================

def scale2lambda(scale:float):
        """ Converts the scale parameter to lambda """
        return np.sqrt(np.pi/2)/(scale)

# =======================================================

def fit_dist(eps:np.ndarray, eps_max:float=8):
        """ Fits a half normal distribution to epsilon """
        eps_stat =  [eps[i] for i in range(len(eps)) if (eps[i] < eps_max and eps[i]>=0)]
        known_parms = {'loc' : 0}
        res = goodness_of_fit(halfnorm, eps_stat, known_params=known_parms, statistic='ks')
        return res

# ===============================================================================================

def Fisher_Information(lmbd:float, Phi:np.ndarray, indices:np.ndarray)->np.ndarray:
        """ Returns the Fisher Information Matrix for the given lambda and regression matrix Phi """
        _, m = np.shape(Phi)
        Phi_I = np.zeros([len(indices), m])
        for i in range(len(indices)):
                Phi_I[i, :] = Phi[indices[i], :]
        I_theta = (2*lmbd**2/np.pi) * (Phi_I.T @ Phi_I)
        return I_theta

# ==============================================================================================

def W_kde(eta:np.ndarray, u1:np.ndarray, u2:np.ndarray, T:np.ndarray, F:np.ndarray)->np.ndarray:
        """ Returns the diagonal weight matrix square for uniform sampling in the given state/input range"""
        pdf = gaussian_kde([eta, u1, u2, T, F])
        N = np.size(T)
        w = np.array([1/(pdf([eta[i], u1[i], u2[i], T[i], F[i]])) for i in range(N)])
        w_norm = w/np.sum(w)
        return np.diag(w_norm.flatten())

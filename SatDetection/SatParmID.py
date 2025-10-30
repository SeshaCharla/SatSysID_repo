# Saturated System Parameter Identification
import numpy as np
import cvxpy as cp
from scipy.optimize import lsq_linear, linprog
# ===========================================

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
# ===============================================================================

def solve_LP(eta, T, F, u1):
        """ Solves the bounding linear programming problem for the given time-series data """
        # Linear programming problem
        Phi_sat = PhiSat_mat(T, F, u1)
        eta_mat = np.matrix(eta).T
        theta = cp.Variable([3, 1])
        problem = cp.Problem(cp.Minimize(cp.sum(Phi_sat[:-1, :] @ theta)),
                                [
                                        Phi_sat[:-1, :] @ theta >= eta_mat[1:, :],      #\eta[k+1] <= \phi[k]^T \theta
                                        theta[0, 0] <= 0,
                                        theta[2, 0] >= 0
                                ]
                        )
        problem.solve()
        return np.matrix(theta.value).T
# =========================================================================================

def solve_lsq(eta_sat, phi_sat):
        """ Solves the bounding linear programming problem for the given time-series data """
        # Least-squares problem
        so
        theta = cp.Variable([3, 1])
        problem = cp.Problem(cp.Minimize(cp.sum_squares(eta_sat - (phi_sat @ theta))),
                                [
                                        theta[0, 0] <= 0,
                                        theta[2, 0] >= 0
                                ]
                        )
        problem.solve()
        return np.matrix(theta.value)
# =========================================================================================


def pred_sat_response(theta, T, F, u1):
        """ Predicted saturated systems response in time an returns array of eta_sat """
        phi_sat = PhiSat_mat(T, F, u1)
        eta_sat_hat = np.zeros(np.shape(T))
        eta_sat_hat[1:] = (phi_sat[0:-1, :] @ theta).flatten()
        eta_sat_hat[0] = eta_sat_hat[1]
        return eta_sat_hat
# ==========================================================================================

def get_sat_indeces(eta, eta_sat_hat, eps):
        """ Returns the regressor for the saturated system """
        indeces = [i for i in range(len(eta)) if np.abs(eta_sat_hat[i]-eta[i]) <= eps ]
        return indeces

# =======================================================================================
def Phi_alpha(T):
        """Returns the aging factor regression matrix for the given series or T, F and u1
           eta[k+1] = (u1[k]/F[k]) * [T**2 T 1] * [th1, th2 th3]^T
        """
        N = len(T)
        PhiSat = np.zeros([N, 3])
        # Looping
        for i in range(N):
                PhiSat[i, :] = np.array([T[i]**2, T[i], 1])
                # PhiSat[i, :] = np.array([T[i], 1, 0])
        #===
        return PhiSat

import numpy as np
import SatSysID_funcs as sf

class SatSys_ssd:
        """ The class that holds the data and methods for ssd SatSysID """

        def __init__(self, ssd_data, name):
                """ Initiates the class with ssd data """
                self.ssd = ssd_data
                self.name = name
                # Calculating theta
                self.theta_LP, self.idx = self.detect_sat()
                self.theta_QP, self.theta_stats = self.get_theta_stats()

        #==============================================================================================

        def get_theta_stats(self):
                """ Get the statistics of the paramter estimates by solving the quadratic programming problem """
                # Getting the relavent data
                eta = self.ssd['eta'][self.idx+1]
                u1 = self.ssd['u1'][self.idx]
                u2 = self.ssd['u2'][self.idx]
                F = self.ssd["F"][self.idx]
                T = self.ssd["T"][self.idx]
                Phi = sf.PhiSat_mat(T, F, u1)
                H = np.matrix(eta).T
                W = sf.W_kde(eta, u2, T, F)
                # Solving the Quadratic Program
                theta_QP = sf.solve_QP(Phi, H, W)
                # Calculating epsilon
                eta_hat = (Phi @ theta_QP).flatten()
                eps = (eta_hat - eta)
                # Fitting distribution
                hfn_fit = sf.fit_dist(eps, eps_max=2)
                hfn_lambda = sf.scale2lambda(hfn_fit.fit_result.params[1])
                var_eps = hfn_fit.fit_result.params[1]**2 * (1 - 2/np.pi)
                exp_eps = 1/hfn_lambda
                I_theta = sf.Fisher_Information(hfn_lambda, Phi)
                C_theta = np.linalg.inv(I_theta)
                sigma_eta_sat = np.sqrt( np.array( [ (Phi[j,:]@ C_theta @ Phi[j,:].T)
                                                        for j in range(np.shape(Phi)[0]-1) ] ))
                # Putting the caclulations into dictionary
                theta_stats = dict()
                theta_stats["eps"] = eps
                theta_stats["hfn_fit"] = hfn_fit
                theta_stats['hfn_lambda'] = hfn_lambda
                theta_stats["var_eps"] = var_eps
                theta_stats['exp_eps'] = exp_eps
                theta_stats['I_theta'] = I_theta
                theta_stats['C_theta'] = C_theta
                theta_stats['sigma_eta_sat'] = sigma_eta_sat
                # ====
                return theta_QP, theta_stats



        # ==========================================================================================

        def detect_sat(self):
                """ Detect the saturated segments and return the LP solution and """
                Phi = sf.PhiSat_mat(self.ssd['T'], self.ssd['F'], self.ssd['u1'])
                H = np.matrix(self.ssd['eta']).T
                theta_LP = sf.solve_LP(Phi[0:-1,:], H[1:, :], verbose=False)
                # Calculate the error
                eta_hat = (Phi[0:-1, :] @ theta_LP).flatten()
                eps_bimodal = (eta_hat - self.ssd['eta'][1:])
                idx = [i for i in range(len(eps_bimodal)) if eps_bimodal[i] <= 2 and eps_bimodal[i] >= 0]
                return theta_LP, np.array(idx)

        # ===========================================================================================

        def predict_eta_sat(self, ssd_ref):
                """ Predicts the response of this particular system to reference inputs ssd_ref"""
                self.ssd_ref = ssd_ref
                Phi_ref = sf.PhiSat_mat(self.ssd_ref['T'], self.ssd_ref['F'], self.ssd_ref['u1'])
                self.eta_pred = (Phi_ref[0:-1, :] @ self.theta_QP).flatten()
                self.sigma_pred = np.sqrt( np.array( [ (Phi_ref[j,:] @ self.theta_stats['C_theta'] @ Phi_ref[j,:].T)
                                                        for j in range(np.shape(Phi_ref)[0]-1) ] ))

        # =================================================================================================

        def temp_var(self):
                """ Get the temperature variation of the Max. sigma """
                N = 1000
                self.T_lin = np.linspace(np.min(self.ssd['T']), np.max(self.ssd['T']), N)
                Phi = np.matrix([np.array([Ti**2, Ti, 1]) for Ti in self.T_lin])
                self.gamma_max = (Phi @ self.theta_QP).flatten()
                self.sigma_gamma_max = np.sqrt( np.array( [ (Phi[j,:] @ self.theta_stats['C_theta'] @  Phi[j,:].T)[0,0]
                                                        for j in range(N) ] ))

        # ===================================================================================================

        def calc_Tw(self, theta_ref):
                """ Calculates the wald test's test-statistic given theta_ref """
                theta_diff = (self.theta_QP - theta_ref)
                self.Tw = (theta_diff.T @ self.theta_stats['I_theta'] @ theta_diff)[0,0]
                return self.Tw

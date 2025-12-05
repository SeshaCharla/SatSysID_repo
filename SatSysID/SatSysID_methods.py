import numpy as np
import SatSysID_funcs as sf
import matplotlib.pyplot as plt

class SatSys_ssd:
        """ The class that holds the data and methods for ssd SatSysID """

        def __init__(self, ssd_data, name):
                """ Initiates the class with ssd data """
                self.ssd = ssd_data
                self.name = name
                self.eps_max = 4
                if 'rmc' in name.lower():
                        self.T0 = 10
                        self.Tr = 6
                else:
                        self.T0 = 5
                        self.Tr = 5
                # Calculating theta
                self.theta_LP, self.eps_bimodal, self.idx, self.Phi = self.detect_sat()
                self.theta_stats = self.get_theta_stats()
                # self.min_max = self.min_max_data()

        #==============================================================================================

        def min_max_data(self):
                """ Returns the dictionary for the min_max data """
                min_max = dict()
                for key in ['u1', 'u2', 'eta', 'T', 'F']:
                        min_max[key] = ( np.min(self.ssd[key][self.idx]), np.max(self.ssd[key][self.idx]) )
                return min_max

        # ===========================================================================================================

        def get_theta_stats(self):
                """ Get the statistics of the paramter estimates """
                Phi = self.Phi[self.idx,:]
                hfn_fit = sf.fit_dist(self.eps_bimodal, eps_max=self.eps_max)
                hfn_lambda = sf.scale2lambda(hfn_fit.fit_result.params[1])
                var_eps = hfn_fit.fit_result.params[1]**2 * (1 - 2/np.pi)
                exp_eps = 1/hfn_lambda
                I_theta = sf.Fisher_Information(hfn_lambda, Phi)
                C_theta = np.linalg.inv(I_theta)
                sigma_eta_sat = np.sqrt( np.array( [ (Phi[j,:]@ C_theta @ Phi[j,:].T)
                                                        for j in range(np.shape(Phi)[0]-1) ] ))
                # Putting the caclulations into dictionary
                theta_stats = dict()
                theta_stats["eps"] = self.eps_bimodal[self.idx]
                theta_stats["hfn_fit"] = hfn_fit
                theta_stats['hfn_lambda'] = hfn_lambda
                theta_stats["var_eps"] = var_eps
                theta_stats['exp_eps'] = exp_eps
                theta_stats['I_theta'] = I_theta
                theta_stats['C_theta'] = C_theta
                theta_stats['sigma_eta_sat'] = sigma_eta_sat
                # ====
                return theta_stats

        # ==========================================================================================

        def detect_sat(self):
                """ Detect the saturated segments and return the LP solution and """
                Phi = sf.PhiSat_mat(self.ssd['T'], self.ssd['F'], self.ssd['u1'], T0=self.T0, Tr=self.Tr)
                H = self.ssd['eta']
                LP_res = sf.solve_LP(Phi[0:-1,:], H[1:], verbose=False)
                theta_LP = np.matrix(LP_res.x).T
                # Calculate the error
                eta_hat = np.asarray(Phi[0:-1, :] @ theta_LP).flatten()
                eps_bimodal = (eta_hat - self.ssd['eta'][1:])
                idx = [i for i in range(len(eps_bimodal)) if eps_bimodal[i] <= self.eps_max and eps_bimodal[i] >= 0]
                return theta_LP, eps_bimodal, np.array(idx), Phi

        # ===========================================================================================

        def predict_eta_sat(self, ssd_ref):
                """ Predicts the response of this particular system to reference inputs ssd_ref"""
                Phi_ref = sf.PhiSat_mat(ssd_ref['T'], ssd_ref['F'], ssd_ref['u1'], self.T0, self.Tr)
                self.eta_pred = np.asarray(Phi_ref[0:-1, :] @ self.theta_LP).flatten()
                self.sigma_pred = np.sqrt( np.array( [ (Phi_ref[j,:] @ self.theta_stats['C_theta'] @ Phi_ref[j,:].T)
                                                        for j in range(np.shape(Phi_ref)[0]-1) ] ))

        # =================================================================================================

        def temp_var(self, plot=True):
                """ Get the temperature variation of the Max. sigma """
                N = 1000
                T_lin = np.linspace(self.min_max['T'][0], self.min_max['T'][1], N)
                Phi = np.matrix([np.array([(2*(Ti-self.T0)/self.Tr)**2 - 1, ((Ti-self.T0)/self.Tr), 1]) for Ti in T_lin])
                gamma_max = (Phi @ self.theta_LP).A1
                sigma_gamma_max = np.sqrt( np.array( [ (Phi[j,:] @ self.theta_stats['C_theta'] @  Phi[j,:].T)[0,0]
                                                        for j in range(N) ] ))
                if plot:
                        plt.plot(T_lin, gamma_max, label=self.name + r'$\hat \eta_{sat}$ $\pm \sigma$')
                        plt.fill_between(T_lin, gamma_max-sigma_gamma_max, gamma_max+sigma_gamma_max,
                                         alpha=0.2)
                return T_lin, gamma_max, sigma_gamma_max

        # ===================================================================================================

        def calc_Tw(self, theta_ref):
                """ Calculates the wald test's test-statistic given theta_ref """
                theta_diff = (self.theta_QP - theta_ref)
                self.Tw = (theta_diff.T @ self.theta_stats['I_theta'] @ theta_diff)[0,0]
                return self.Tw

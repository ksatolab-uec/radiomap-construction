##################################################################
# Radio Map Construction with Regression Kriging
# Written by Koya SATO, Ph.D.
# Requirements:
# - Python 3.x
# - numpy
# - scipy
# - matplotlib
##################################################################

# The MIT License (MIT)
#
# Copyright (c) 2021 Koya SATO.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize

''' sampling via multivariate normal distribution with no trend '''
"""
cov: variance-covariance matrix
"""
def gen_multivariate_normal(cov):
    print("--gen_multivariate_normal()--")
    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal(len(cov))

    return np.dot(L, z)

'''variance-covariance matrix for log-normal shadowing'''
"""
x, y:  vector for measurement location
dcor:  correlation distance of shadowing [m]
sigma: standard deviation [dB]
"""
def gen_varcov_matrix(x, y, dcor, sigma):
    print("--gen_varcov_matrix()--")
    dmat = distance(x, y, x[:, np.newaxis], y[:, np.newaxis]) # distance matrix
    tmp  = 0.6931471805599453 / dcor                          # np.log(2.0)/dcor

    return sigma * sigma * np.exp(-dmat * tmp)

'''for measurement location'''
"""
n_node:   number of nodes
len_area: area length [m]
"""
def gen_location_vector(n_node, len_area):
    print("--gen_location_vector()--")
    x = np.random.uniform(0.0, len_area, n_node)
    y = np.random.uniform(0.0, len_area, n_node)
    
    return x, y

''' gen empirical semivariogram via binning '''
def gen_emprical_semivar(data, d_max, num):
    def gen_combinations(arr):
      r, c = np.triu_indices(len(arr), 1)

      return np.stack((arr[r], arr[c]), 1)

    print("--gen_emprical_semivar()--")
    d_semivar   = np.linspace(0.0, d_max, num)
    SPAN        = d_semivar[1] - d_semivar[0]

    indx        = gen_combinations(np.arange(N))
    
    '''gen semivariogram clouds'''
    d           = distance(data[indx[:, 0], 0], data[indx[:, 0], 1], data[indx[:, 1], 0], data[indx[:, 1], 1])
    indx        = indx[d<=d_max]
    d           = d[d <= d_max]
    semivar     = (data[indx[:, 0], 2] - data[indx[:, 1], 2])**2

    '''average semivariogram clouds via binning'''
    semivar_avg = np.empty(num)
    for i in range(num):
        d1 = d_semivar[i] - 0.5*SPAN
        d2 = d_semivar[i] + 0.5*SPAN
        indx_tmp = (d1 < d) * (d <= d2) #index within calculation span
        semivar_tr = semivar[indx_tmp]
        semivar_avg[i] = semivar_tr.mean()

    return d_semivar[np.isnan(semivar_avg) == False], 0.5 * semivar_avg[np.isnan(semivar_avg) == False]

'''theoretical semivariogram (exponential)'''
def semivar_exp(d, nug, sill, ran):
  return np.abs(nug) + np.abs(sill) * (1.0-np.exp(-d/(np.abs(ran))))

'''fitting emperical semivariotram to theoretical model'''
def fit_semivar(d, data):
    def obj_func(x):
        theorem = semivar_exp(d, x[0], x[1], x[2])

        return ((data-theorem)**2).sum()

    x0  = np.random.uniform(0.0, 1.0, 3)
    res = minimize(obj_func, x0, method='nelder-mead')
    for i in range(5):
        x0      = np.random.uniform(0.0, 1.0, 3)
        res_tmp = minimize(obj_func, x0, method='nelder-mead')
        if res.fun > res_tmp.fun:
            res = res_tmp

    return np.abs(res.x)

def ordinary_kriging(mat, x_vec, y_vec, z_vec, x_rx, y_rx, nug, sill, ran):
    vec              = np.ones(len(z_vec)+1, dtype=np.float)

    d_vec            = distance(x_vec, y_vec, x_rx, y_rx)
    vec[:len(z_vec)] = semivar_exp(d_vec, nug, sill, ran)
    weight           = np.linalg.solve(mat, vec)
    est              = (z_vec * weight[:len(z_vec)]).sum()

    return est

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

'''matrix for weight calculation in Ordinary Kriging'''
def gen_mat_for_kriging(x_vec, y_vec, z_vec, nug, sill, ran):
    mat = distance(x_vec, y_vec, x_vec[:, np.newaxis], y_vec[:, np.newaxis])
    mat = semivar_exp(mat, nug, sill, ran)
    mat = np.vstack((mat, np.ones(len(z_vec))))
    mat = np.hstack((mat, np.ones([len(z_vec)+1, 1])))
    mat[len(z_vec)][len(z_vec)] = 0.0

    return mat

def pathloss(d, eta):
    return 10.0 * eta * np.log10(d+1.0) #+1: to avoid diverse of path loss

'''Ordinary Least Squares for Path Loss Modeling'''
def OLS(d, p):
    A       = np.vstack([-10.0*np.log10(d+1.0), np.ones(len(p))]).T
    m, c    = np.linalg.lstsq(A, p, rcond=None)[0]
    return m, c

if __name__=="__main__":
    '''measurement configuration'''
    LEN_AREA    = 200.0            #area length [m]
    N           = 50              #number of samples
    DCOR        = 20.0             #correlation distance [m]
    STDEV       = 8.0              #standard deviation
    TX_X        = 0.0              #x coordinate of transmitter
    TX_Y        = 0.5 * LEN_AREA   #y coordinate of transmitter
    PTX         = 30.0             #transmission power [dBm]
    ETA         = 3.0              #path loss index

    '''parameters for semivariogram modeling and Kriging'''
    D_MAX       = LEN_AREA * np.sqrt(2.0) #maximum distance in semivariogram modeling
    N_SEMIVAR   = 20                      #number for averaging empirical semivariograms
    
    '''get measurement dataset'''
    x, y    = gen_location_vector(N, LEN_AREA)
    cov     = gen_varcov_matrix(x, y, DCOR, STDEV)
    z       = gen_multivariate_normal(cov)  #correlated shadowing vector[dB]

    d       = distance(TX_X, TX_Y, x, y)
    l       = pathloss(d, ETA)              #[dB]
    prx     = PTX - l + z                   #received signal power [dBm]

    '''path loss modeling'''
    eta_est, ptx_est = OLS(d, prx)

    '''regression-based shadowing extraction'''
    pmean_est   = ptx_est - pathloss(d, eta_est)
    shad_est    = prx - pmean_est

    '''get empirical semivariogram model'''
    data        = np.vstack([x, y, shad_est]).T
    d_sv, sv    = gen_emprical_semivar(data, D_MAX, N_SEMIVAR)
    param       = fit_semivar(d_sv, sv)

    '''plot empirical/theoretical semivariogram'''
    d_fit = np.linspace(0.0, D_MAX, 1000)
    y_fit = semivar_exp(d_fit, param[0], param[1], param[2])
    plt.plot(d_sv, sv, 'o', label="Empirical")
    plt.plot(d_fit, y_fit, label="Fitted")
    plt.title("Semivariogram")
    plt.xlabel("Distance between measurement points [m]")
    plt.ylabel("Semivariogram")
    plt.legend()
    plt.show()

    '''radio map construction'''
    N_DIV   = 30 #number of grids in each axis
    x_valid = np.linspace(0, LEN_AREA, N_DIV)
    y_valid = np.linspace(0, LEN_AREA, N_DIV)
    X, Y    = np.meshgrid(x_valid, y_valid)
    prx_map = np.zeros([len(x_valid), len(y_valid)])
    
    mat     = gen_mat_for_kriging(x, y, prx, param[0], param[1], param[2])

    for i in range(len(y_valid)):
        for j in range(len(x_valid)):
            d_tmp           = distance(TX_X, TX_Y, x_valid[j], y_valid[i])
            pmean           = ptx_est - pathloss(d_tmp, eta_est)
            prx_map[i][j]   = pmean + ordinary_kriging(mat, x, y, shad_est, x_valid[j], y_valid[i], param[0], param[1], param[2])

    '''plot results'''

    fig, axs = plt.subplots(1, 2, figsize=(9, 6))

    im0 = axs[0].scatter(x, y, s=80, c=prx, cmap='jet', vmin=-40, vmax=0)
    im1 = axs[1].pcolor(X, Y, prx_map, cmap='jet', vmin=-40, vmax=0)

    axs[0].set_title("Dataset")
    axs[0].set_xlim([0.0, LEN_AREA])
    axs[0].set_ylim([0.0, LEN_AREA])
    axs[0].set_xlabel("$x$ [m]")
    axs[0].set_ylabel("$y$ [m]")
    axs[0].set_aspect('equal')
    divider = make_axes_locatable(axs[0])
    cax     = divider.append_axes("right", size="5%", pad=0.1)
    cb      = fig.colorbar(im0, ax=axs[0], cax=cax)
    cb.set_label("Received Signal Power [dBm]")

    axs[1].set_title("Kriging-based Radio Map")
    axs[1].set_xlim([0.0, LEN_AREA])
    axs[1].set_ylim([0.0, LEN_AREA])
    axs[1].set_xlabel("$x$ [m]")
    axs[1].set_ylabel("$y$ [m]")
    axs[1].set_aspect('equal')
    divider = make_axes_locatable(axs[1])
    cax     = divider.append_axes("right", size="5%", pad=0.1)
    cb      = fig.colorbar(im1, ax=axs[1], cax=cax)
    cb.set_label("Received Signal Power [dBm]")

    plt.tight_layout()
    # plt.show()
    plt.savefig("example.png", bbox_inches="tight")
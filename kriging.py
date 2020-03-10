##################################################################
# Demonstration of Radio Map Construction with Regression Kriging
# Written by Koya SATO
# 2020.02.22 ver.1.0
# Verification:
# - Ubuntu18.04 (Docker container)
# - Python     3.7.3
# - numpy      1.16.2
# - scipy      1.2.1
# - matplotlib 3.0.3
##################################################################

# The MIT License (MIT)
#
# Copyright (c) 2020 Koya SATO.
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
from scipy.optimize import minimize

''' sampling via multivariate normal distribution with no trend '''
def genMultivariate(cov):
    print("--genMultivariate()--")
    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal(len(cov))
    return np.dot(L, z)

'''gen variance-covariance matrix'''
def genCovMat(x, y, dcor, sigma):
    print("--genCovMat()--")
    dmat = distance(x, y, x[:, np.newaxis], y[:, np.newaxis]) # distance matrix
    tmp = 0.6931471805599453 / dcor # np.log(2.0)/dcor

    return sigma * sigma * np.exp(-dmat * tmp)

'''measurement location '''
def genMeasurementLocation(size, len_area):
    print("--genMeasurementLocation()--")
    x = np.random.uniform(0.0, len_area, size)
    y = np.random.uniform(0.0, len_area, size)
    return x, y

''' gen empirical semivariogram via binning '''
def genSemivar(data, d_max, num):
    def genCombinations(arr):
      r, c = np.triu_indices(len(arr), 1)
      return np.stack((arr[r], arr[c]), 1)

    print("--genSemivar()--")
    d_semivar = np.linspace(0.0, d_max, num)
    SPAN = d_semivar[1] - d_semivar[0]

    indx = genCombinations(np.arange(N))
    
    d = distance(data[indx[:, 0], 0], data[indx[:, 0], 1], data[indx[:, 1], 0], data[indx[:, 1], 1])
    indx = indx[d<=d_max]
    d = d[d <= d_max]
    semivar = (data[indx[:, 0], 2] - data[indx[:, 1], 2])**2

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
def semivarFitting(d, data):
    def objFunc(x):
        theorem = semivar_exp(d, x[0], x[1], x[2])
        return ((data-theorem)**2).sum()

    x0 = np.random.uniform(0.0, 1.0, 3)
    res = minimize(objFunc, x0, method='nelder-mead')
    for i in range(5):
        x0 = np.random.uniform(0.0, 1.0, 3)
        res_tmp = minimize(objFunc, x0, method='nelder-mead')
        if res.fun > res_tmp.fun:
            res = res_tmp
    return np.abs(res.x)

def ordinaryKriging(mat, x_vec, y_vec, z_vec, x_rx, y_rx, nug, sill, ran):
    vec = np.ones(len(z_vec)+1, dtype=np.float)

    d_vec = distance(x_vec, y_vec, x_rx, y_rx)
    vec[:len(z_vec)] = semivar_exp(d_vec, nug, sill, ran)
    weight = np.linalg.solve(mat, vec)
    est = (z_vec * weight[:len(z_vec)]).sum()

    return est

def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

'''matrix for weight calculation in Ordinary Kriging'''
def genMat(x_vec, y_vec, z_vec, nug, sill, ran):
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
    A = np.vstack([-10.0*np.log10(d+1.0), np.ones(len(p))]).T
    m, c = np.linalg.lstsq(A, p)[0]
    return m, c

if __name__=="__main__":
    '''measurement configuration'''
    LEN_AREA = 200.0 #area length [m]
    N = 100 #number of samples
    DCOR = 20.0 #correlation distance [m]
    STDEV = 8.0 #standard deviation
    TX_X = 0.0 #x coordinate of transmitter
    TX_Y = 0.5 * LEN_AREA #y coordinate of transmitter
    PTX = 30.0 #transmission power [dBm]
    ETA = 3.0 #path loss index

    '''parameters for semivariogram modeling and Kriging'''
    D_MAX = LEN_AREA * np.sqrt(2.0) #maximum distance in semivariogram modeling
    N_SEMIVAR = 20 #number of points for averaging empirical semivariograms
    
    '''get measurement dataset'''
    x, y = genMeasurementLocation(N, LEN_AREA) #get N-coodinates for measurements
    cov = genCovMat(x, y, DCOR, STDEV) #gen variance-covariance matrix
    z = genMultivariate(cov) #gen measurement samples based on multivariate normal distribution

    d = distance(TX_X, TX_Y, x, y) #distance between received points and transmission point
    l = pathloss(d, ETA) #path loss
    prx = PTX - l + z #received signal power

    '''Path loss modeling'''
    eta_est, ptx_est = OLS(d, prx)

    '''Shadowing extraction'''
    pmean_est = ptx_est - pathloss(d, eta_est)
    shad_est = prx - pmean_est

    '''get empirical semivariogram model'''
    data = np.vstack([x, y, shad_est]).T
    d_sv, sv = genSemivar(data, D_MAX, N_SEMIVAR)
    param = semivarFitting(d_sv, sv)

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

    '''Radio Map Construction'''
    N_DIV = 30 #number of grids in each axis
    x_valid = np.linspace(0, LEN_AREA, N_DIV)
    y_valid = np.linspace(0, LEN_AREA, N_DIV)
    X, Y = np.meshgrid(x_valid, y_valid)
    prx_map = np.zeros([len(x_valid), len(y_valid)])
    
    mat = genMat(x, y, prx, param[0], param[1], param[2])

    for i in range(len(y_valid)):
        for j in range(len(x_valid)):
            pmean = ptx_est - pathloss(distance(TX_X, TX_Y, x_valid[j], y_valid[i]), eta_est)
            prx_map[i][j] = pmean + ordinaryKriging(mat, x, y, shad_est, x_valid[j], y_valid[i], param[0], param[1], param[2])

    '''plot results'''
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=1.0)
    ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect=1.0)

    ax1.scatter(x, y, s=80, c=prx, cmap='jet')
    ax1.set_title("Dataset")
    ax2.pcolor(X, Y, prx_map, cmap='jet')
    ax2.set_title("Kriging-based Map")
    plt.show()
    # plt.savefig("example.png")
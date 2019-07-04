################################################################
# Demonstration of Semivariogram Modeling and Ordinary Kriging
# Written by Koya SATO
# 2019.07.04 ver.1.0
# Verification:
# - Windows10 Home x64
# - Python     3.7.3
# - numpy      1.16.2
# - matplotlib 3.0.3
################################################################

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

''' sampling by multivariate normal distribution '''
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

if __name__=="__main__":
    '''measurement configuration'''
    LEN_AREA = 1000.0 #area length [m]
    N = 128 #number of samples
    DCOR = 100.0 #correlation distance [m]
    STDEV = 8.0 #standard deviation

    '''parameters for semivariogram modeling'''
    D_MAX = 500.0 #maximum distance in semivariogram modeling
    N_SEMIVAR = 20 #number of points for averaging empirical semivariograms

    '''get measurement dataset'''
    x, y = genMeasurementLocation(N, LEN_AREA) #get N-coodinates for measurements
    cov = genCovMat(x, y, DCOR, STDEV) #gen variance-covariance matrix
    z = genMultivariate(cov) #gen measurement samples based on multivariate normal distribution

    '''get empirical semivariogram model'''
    data = np.vstack([x, y, z]).T
    d_sv, sv = genSemivar(data, D_MAX, N_SEMIVAR)

    '''plot empirical/theoretical semivariogram'''
    param = semivarFitting(d_sv, sv)
    d_fit = np.linspace(0.0, D_MAX, 1000)
    y_fit = semivar_exp(d_fit, param[0], param[1], param[2])
    plt.plot(d_sv, sv, 'o', label="Empirical")
    plt.plot(d_fit, y_fit, label="Fitted")
    plt.title("Semivariogram")
    plt.xlabel("Distance between measurement points [m]")
    plt.ylabel("Semivariogram")
    plt.legend()
    plt.show()

    '''Ordinary Kriging'''
    N_DIV = 30
    x_valid = np.linspace(0, LEN_AREA, N_DIV)
    y_valid = np.linspace(0, LEN_AREA, N_DIV)
    X, Y = np.meshgrid(x_valid, y_valid)
    z_map = np.zeros([len(x_valid), len(y_valid)])

    mat = genMat(x, y, z, param[0], param[1], param[2])
    for i in range(len(y_valid)):
        for j in range(len(x_valid)):
            z_map[i][j] = ordinaryKriging(mat, x, y, z, x_valid[j], y_valid[i], param[0], param[1], param[2])

    '''plot results'''
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1, adjustable='box', aspect=1.0)
    ax2 = fig.add_subplot(1,2,2, adjustable='box', aspect=1.0)

    ax1.scatter(x, y, s=80, c=z, cmap='jet')
    ax1.set_title("Dataset")
    ax2.pcolor(X, Y, z_map, cmap='jet')
    ax2.set_title("Kriging-based Map")
    plt.show()
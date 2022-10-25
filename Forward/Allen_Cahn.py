"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch

Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
"""
import sys
import time
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.pyplot import MultipleLocator
import dapinn as dde
import numpy as np
from scipy.io import loadmat
from dapinn.backend import tf


def gen_testdata():
    data = loadmat("./Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    x_2 = X[:, 0:1] * X[:, 0:1]
    t_2 = X[:, 1:2] * X[:, 1:2]
    x_3 = X[:, 0:1] * X[:, 0:1] * X[:, 0:1]
    X_N = np.hstack((x_3, t_2,x_2))
    X = np.insert(X, [2], X_N, axis=1)
    X[:, [1, 4]] = X[:, [4, 1]]
    y = u.flatten()[:, None]
    return X, y


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
d = 0.001

def DAPINNpde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=4)
    dy_t2 = dde.grad.jacobian(y, x, i=0, j=3)
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    du_x1x2 = dde.grad.hessian(y, x, i=0, j=1)
    du_x2 = dde.grad.jacobian(y, x, i=0, j=1)
    du_x1x3 = dde.grad.hessian(y, x, i=0, j=2)
    du_x2x3 = dde.grad.hessian(y, x, i=1, j=2)
    du_x3x3 = dde.grad.hessian(y, x, i=2, j=2)
    du_x3 = dde.grad.jacobian(y, x, i=0, j=2)
    return dy_t+2*x[:, 4:5]*dy_t2 - d * (du_xx + \
4 * x[:, 0:1]**2 * du_x2x2 + 9 * x[:,0:1] **4 * du_x3x3 + 4 * x[:, 0:1] * du_x1x2 + 6 * x[:,0:1]**2 * du_x1x3 + \
                12 * x[:, 0:1] **3 * du_x2x3 + 2 * du_x2 + 6 * x[:,0:1] * du_x3) - 5 * (y - y**3)


def output_transform(x, y):
    return x[:, 0:1]**2 * tf.cos(np.pi * x[:, 0:1]) + x[:, 4:5] * (1 - x[:, 0:1]**2) * y


DAPINNdata = dde.data.TimePDE(geomtime, DAPINNpde, [], num_domain=400,mode="x3tt" )
DAPINNnet = dde.nn.FNN([5] + [30] * 3 + [1], "tanh", "Glorot normal")
DAPINNnet.apply_output_transform(output_transform)
DAPINNmodel = dde.Model(DAPINNdata, DAPINNnet)
time_start = time.time()  # 记录开始时间
DAPINNmodel.compile("adam", lr=1e-3)
DAPINNmodel.train(epochs=30000)
DAPINNmodel.compile("L-BFGS")
losshistory, train_state = DAPINNmodel.train()

X, y_true = gen_testdata()
y_pred = DAPINNmodel.predict(X)
f = DAPINNmodel.predict(X, operator=DAPINNpde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)
dapinnx3error=dde.metrics.l2_relative_error(y_true, y_pred)

#DAPINN fourier
def gen_testdatafourier():
    data = loadmat("./Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]


    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    t_2=X[:, 1:2]*X[:, 1:2]
    sin = np.sin(np.pi *X[:, 0:1])
    cos = np.cos(np.pi *X[:, 0:1])
    X_N = np.hstack((cos,t_2, sin))
    X = np.insert(X, [2], X_N, axis=1)
    X[:, [1, 4]] = X[:, [4, 1]]
    y = u.flatten()[:, None]
    return X, y


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def DAPINNFOURIERpde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=4)
    dy_t2 = dde.grad.jacobian(y, x, i=0, j=3)
    dy_x2= dde.grad.jacobian(y, x, i=0, j=1)
    dy_x3 = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    dy_x3x3 = dde.grad.hessian(y, x, i=2, j=2)
    dy_xx2 = dde.grad.hessian(y, x, i=0, j=1)
    dy_x1x3 = dde.grad.hessian(y, x, i=0, j=2)
    dy_x2x3 = dde.grad.hessian(y, x, i=1, j=2)
    return dy_t+2*x[:, 4:5]*dy_t2 - d * (dy_xx + (np.pi**2 *tf.cos(x[:,0:1])**2)*dy_x2x2+(np.pi**2 *tf.sin(x[:,0:1])**2)*dy_x3x3 + 2*np.pi *tf.cos(x[:,0:1])*dy_xx2 -\
                 2*np.pi *tf.sin(x[:,0:1])*dy_x1x3  -2*np.pi *np.pi *tf.cos(x[:,0:1])*tf.sin(x[:,0:1])*dy_x2x3  -\
                 x[:,1:2]*np.pi *np.pi *dy_x2-x[:,2:3]*np.pi *np.pi *dy_x3) - 5 * (y - y**3)


DAPINNFOURIERdata = dde.data.TimePDE(geomtime, DAPINNFOURIERpde, [], num_domain=400,mode="fourier tt" )
DAPINNFOURIERnet = dde.nn.FNN([5] + [30] * 3 + [1], "tanh", "Glorot normal")
DAPINNFOURIERnet.apply_output_transform(output_transform)
DAPINNFOURIERmodel = dde.Model(DAPINNFOURIERdata, DAPINNFOURIERnet)
time_start = time.time()  # 记录开始时间
DAPINNFOURIERmodel.compile("adam", lr=1e-3)
DAPINNFOURIERmodel.train(epochs=30000)
DAPINNFOURIERmodel.compile("L-BFGS")
DAPINNFOURIERmodel.train()

X, y_true = gen_testdatafourier()
y_pred = DAPINNFOURIERmodel.predict(X)
f = DAPINNFOURIERmodel.predict(X, operator=DAPINNFOURIERpde)

print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)
dapinnfouriererror=dde.metrics.l2_relative_error(y_true, y_pred)

#PINN

def gen_testdata():
    data = loadmat("./Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
d = 0.001

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - d * dy_xx - 5 * (y - y**3)

# Hard restraints on initial + boundary conditions
# Backend tensorflow.compat.v1 or tensorflow
def output_transform(x, y):
    return x[:, 0:1]**2 * tf.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y


data = dde.data.TimePDE(geomtime, pde, [], num_domain=400)

net = dde.nn.FNN([2] + [30] * 3 + [1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)
model = dde.Model(data, net)
time_start = time.time()  # 记录开始时间
model.compile("adam", lr=1e-3)
model.train(epochs=30000)
model.compile("L-BFGS")
model.train()


X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)
pinnerror=dde.metrics.l2_relative_error(y_true, y_pred)

# print error
print("daPINN with Fourier:L2 relative error of u",
      dapinnfouriererror)
print("daPINN with x^3 :L2 relative error of u",
      dapinnx3error)
print("PINN :L2 relative error of u",
      pinnerror)

def plt1d(x,y_ture,y_predict,name):
    # 1D
    y_ture_1=y_ture[:,25]
    y_predict_1=y_predict[:,25]
    y_ture_2=y_ture[:,50]
    y_predict_2=y_predict[:,50]
    y_ture_3=y_ture[:,75]
    y_predict_3=y_predict[:,75]
    plt.figure()
    plt.plot(x, y_ture_1, "-k", label="True")
    plt.plot(x, y_predict_1, "--r", label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(name+' t=0.25')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(x, y_ture_2, "-k", label="True")
    plt.plot(x, y_predict_2, "--r", label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(name+' t=0.5')
    plt.show()
    plt.figure()
    plt.plot(x, y_ture_3, "-k", label="True")
    plt.plot(x, y_predict_3, "--r", label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(name+' t=0.75')
    plt.legend()
    plt.show()

def DApinngen_plt_x():
    X, y_true = gen_testdatafourier()
    y_pred = DAPINNFOURIERmodel.predict(X)
    x_1  = np.linspace(-1, 1, 201)
    t_1 = np.linspace(0, 1, 101)

    error = abs(y_true - y_pred)
    y_pred = np.reshape(y_pred[:, 0], (101, 201)).T
    error = np.reshape(error[:, 0], (101, 201)).T
    y_true = np.reshape(y_true[:, 0], (101, 201)).T
    return y_pred, error, y_true, x_1, t_1



def gen_plt_x():
    X, y_ture = gen_testdata()
    y_pred = model.predict(X)
    x_1= np.linspace(-1, 1, 201)
    t_1= np.linspace(0, 1, 101)

    error = abs(y_ture-y_pred)
    y_pred = np.reshape(y_pred[:, 0], (101, 201)).T
    error = np.reshape(error[:, 0], (101, 201)).T
    y_ture=np.reshape(y_ture[:,0], (101, 201)).T
    return y_pred,error,y_ture,x_1,t_1


y_predict, error, y_true, x_1, t_1 = DApinngen_plt_x()
plt1d(x_1,y_true,y_predict,'DAPINN')

y_predict, error, y_true, x_1, t_1 = gen_plt_x()
plt1d(x_1,y_true,y_predict,'PINN')


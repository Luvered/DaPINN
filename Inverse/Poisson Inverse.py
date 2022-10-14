"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch

Implementation for the diffusion-reaction system with a space-dependent reaction rate in paper https://arxiv.org/abs/2111.02801.
"""
import dapinn as dde
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from deepxde.backend import tf
import csv
import math

# cos
def k(xi):
    x = xi
    return np.sin(x) + 2*np.sin(2*x) + 3 * np.sin(3 * x) + 4 * np.sin(4 * x)# + 8 * np.sin(8 * x)


geom = dde.geometry.Interval(0, 3)

def bc0(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)


def func0(x):
    return 0


def bcp(x, on_boundary):
    return on_boundary and np.isclose(x[0], 3)


def funcp(x):
    return 3


def sol(xi):
    x = xi
    return x + np.sin(x) + 1/2*np.sin(2*x) + 1/3 * np.sin(3 * x) + 1/4 * np.sin(4 * x)# + 1/8 * np.sin(8 * x)


bc0 = dde.icbc.DirichletBC(geom, func0, bc0)
bcp = dde.icbc.DirichletBC(geom, funcp, bcp)

num = 60
xvals = np.linspace(0, 3, num)

# 1. PINN

def pde(x, y):
    u, f = y[:, 0:1], y[:, 1:2]
    du_xx = dde.grad.hessian(y, x, component=0)
    return -du_xx - f

x_2 = xvals * xvals
x_2 = x_2.reshape((np.size(xvals, axis=0), 1))
xval2 = xvals.reshape((np.size(xvals, axis=0), 1))
xval2 = np.hstack((xval2, x_2))

ob_x, ob_u = xval2[:, 0:1], np.reshape(sol(xvals), (-1, 1))
observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)

data = dde.data.PDE(
    geom,
    pde,
    bcs=[bc0, bcp, observe_u],
    num_domain=num,
    num_boundary=30,
    train_distribution="uniform",
    num_test=1000,
)

net = dde.nn.PFNN([1, [30, 30], [30, 30], [30, 30], [30, 30], 2], "tanh", "Glorot uniform")

model = dde.Model(data, net)

model.compile("adam", lr=5e-5)

A = np.linspace(0, 3, 1000)
x = np.vstack((A))
losshistory, train_state = model.train(iterations=40000)

yhat = model.predict(x)
uhato, khato = yhat[:, 0:1], yhat[:, 1:2]

ktrue = k(x)
utrue = sol(x[:, 0:1])


# 2. DAPINN

def pde(x, y):
    u, f = y[:, 0:1], y[:, 1:2]
    dy_xx = dde.grad.hessian(y, x, i=0, j=0, component=0)
    dy_x2x2 = dde.grad.hessian(y, x, i=1, j=1, component=0)
    dy_xx2 = dde.grad.hessian(y, x, i=0, j=1, component=0)
    dy_x2 = dde.grad.jacobian(y, x, i=0, j=1)
    return -(dy_xx + 4 * x[:, 0:1] * dy_xx2 + 4 * x[:, 1:2] * dy_x2x2 + 2 * dy_x2) - f


x_2 = xvals * xvals
x_2 = x_2.reshape((np.size(xvals, axis=0), 1))
xval2 = xvals.reshape((np.size(xvals, axis=0), 1))
xval2 = np.hstack((xval2, x_2))

ob_x, ob_u = xval2, np.reshape(sol(xvals), (-1, 1))
observe_u = dde.icbc.PointSetBC(ob_x, ob_u, component=0)

data = dde.data.PDE(
    geom,
    pde,
    bcs=[bc0, bcp, observe_u],
    NN_type=2,
    num_domain=num,
    num_boundary=30,
    train_distribution="uniform",
    num_test=1000,
)

net = dde.nn.PFNN([2, [30, 30], [30, 30], [30, 30], [30, 30], 2], "tanh", "Glorot uniform")

model = dde.Model(data, net)

model.compile("adam", lr=5e-5)

A = np.linspace(0,3,1000)
A2 = A**2
x = np.vstack((A,A2)).T

losshistory, train_state = model.train(iterations=20000)

yhat = model.predict(x)
uhat, khat = yhat[:, 0:1], yhat[:, 1:2]

ktrue = k(x)
utrue = sol(x[:, 0:1])


print("—————————— PINN ——————————")
print("L2 relative error of f(x):",dde.metrics.l2_relative_error(khato, ktrue[:, 0:1]))
print("L2 relative error of u(x):",dde.metrics.l2_relative_error(uhato, utrue))

print("—————————— DAPINN ——————————")
print("L2 relative error of f(x):", dde.metrics.l2_relative_error(khat, ktrue[:, 0:1]))
print("L2 relative error of u(x):",dde.metrics.l2_relative_error(uhat, utrue))


plt.figure()
plt.plot(x[:,0:1], ktrue[:, 0:1],color='k', label="f_true",linewidth=1)
plt.plot(x[:,0:1], khat[:, 0:1], linestyle="--",color='b', label="f_DaPINN")
plt.plot(x[:,0:1], khato, linestyle="--",color='r', label="f_PINN")
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('FIG4_A.pdf', format='pdf', dpi=1000)
utrue = sol(x[:, 0:1])

plt.figure()
plt.plot(x[:,0:1], utrue,color='k', label="u_true",linewidth=1)
plt.plot(x[:,0:1], uhat, linestyle="--",color='b', label="u_DaPINN")
plt.plot(x[:,0:1], uhato, linestyle="--",color='r', label="u_PINN")
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x)')
plt.savefig('FIG4_B.pdf', format='pdf', dpi=1000)

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

import dapinn as dde
import numpy as np
import time
import csv


def heat_solution(x):
    """Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    x : np.ndarray
    t : np.ndarray
    """
    t = np.array([x[:, -1]]).T
    x=x[:,0:1]
    return np.exp(-(n ** 2 * np.pi ** 2 * a * t) / (L ** 2)) * np.sin(n * np.pi * x / L)


# Problem parameters:
a = 0.4  # Thermal diffusivity
L = 1  # Length of the bar
n = 1  # Frequency of the sinusoidal initial conditions

##dapinn x2
def dapinnx2pde(x, y):
    """Expresses the PDE residual of the heat equation."""
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    dy_xx2 = dde.grad.hessian(y, x, i=0, j=1)
    dy_x2=dde.grad.jacobian(y, x, i=0, j=1)
    return (dy_t)- a * (
                dy_xx + 4 * x[:, 0:1] * dy_xx2 + 4 * (x[:, 0:1]**2 ) * dy_x2x2 + 2 * dy_x2)

# Computational geometry:
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)


dapinnx2data = dde.data.TimePDE(
    geomtime,
    dapinnx2pde,
    [bc, ic],
    num_domain=105,
    num_boundary=15,
    num_initial=15,
    num_test=2540,
    solution=heat_solution,
    mode="x2t"
)
dapinnx2net = dde.nn.FNN([3] + [20] *4 + [1], "tanh", "Glorot normal")
dapinnx2model = dde.Model(dapinnx2data, dapinnx2net)
time_start = time.time()  # 记录开始时间
# Build and train the model:
dapinnx2model.compile("adam", lr=1e-4, metrics=["l2 relative error"])
dapinnx2model.train(iterations=100000)
dapinnx2model.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state = dapinnx2model.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ',time_sum)
# Plot/print the results
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
dapinnx2error = best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)
##dapinn x3
def dapinnx3pde(x, y):
    """Expresses the PDE residual of the heat equation."""
    dy_t = dde.grad.jacobian(y, x, i=0, j=3)

    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    du_x1x2 = dde.grad.hessian(y, x, i=0, j=1)
    du_x2=dde.grad.jacobian(y, x, i=0, j=1)
    du_x1x3 =dde.grad.hessian(y, x, i=0, j=2)
    du_x2x3= dde.grad.hessian(y, x, i=1, j=2)
    du_x3x3 =dde.grad.hessian(y, x, i=2, j=2)
    du_x3 = dde.grad.jacobian(y, x, i=0, j=2)
    return dy_t- a * (du_xx + 4 * x[:, 0:1]**2 * du_x2x2 + 9 * x[:, 0:1]**4 * du_x3x3 + \
    4 * x[:,0:1]* du_x1x2+6*x[:, 0:1]**2*du_x1x3+12*x[:,0:1]**3*du_x2x3+ 2 * du_x2+6*x[:, 0:1]*du_x3 )



# Computational geometry:
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)

# Define the PDE problem and configurations of the network:
dapinnx3data = dde.data.TimePDE(
    geomtime,
    dapinnx3pde,
    [bc, ic],
    num_domain=105,
    num_boundary=15,
    num_initial=15,
    num_test=2540,
    solution=heat_solution,mode="x3t"
)
dapinnx3net = dde.nn.FNN([4] + [20] * 4 + [1], "tanh", "Glorot normal")
dapinnx3model = dde.Model(dapinnx3data, dapinnx3net)
time_start = time.time()  # 记录开始时间
# Build and train the model:
dapinnx3model.compile("adam", lr=1e-4, metrics=["l2 relative error"])
dapinnx3model.train(iterations=100000)
dapinnx3model.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state = dapinnx3model.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ',time_sum)
# Plot/print the results
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
dapinnx3error=best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)

##pinn

def PINNpde(x, y):
    """Expresses the PDE residual of the heat equation."""
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a * dy_xx
# 创建列表，保存header内容
# Computational geometry:
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
    lambda _, on_initial: on_initial,
)



# Define the PDE problem and configurations of the network:
PINNdata = dde.data.TimePDE(
    geomtime,
    PINNpde,
    [bc, ic],
    num_domain=105,
    num_boundary=15,
    num_initial=15,
    num_test=2540,
    solution=heat_solution
)

PINNnet = dde.nn.FNN([2] + [20] * 4 + [1], "tanh", "Glorot normal")
PINNmodel = dde.Model(PINNdata, PINNnet)
time_start = time.time()  # 记录开始时间
# Build and train the model:
PINNmodel.compile("adam", lr=1e-4, metrics=["l2 relative error"])
PINNmodel.train(iterations=100000)
PINNmodel.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state = PINNmodel.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ',time_sum)
# Plot/print the results
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
pinnerror=best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)




#
# print error
print("daPINN with x^2:L2 relative error of u",
      dapinnx2error)
print("daPINN with x^3 :L2 relative error of u",
      dapinnx3error)
print("PINN :L2 relative error of u",
      pinnerror)
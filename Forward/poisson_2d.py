from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time
import numpy as np
import dapinn as dde
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

a = 10

def f(x, y):
    u_xx = (
        16**a
        * a
        * (a * (1 - 2 * x) ** 2 - 2 * x**2 + 2 * x - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((x - 1) ** 2 * x**2)
    )
    u_yy = (
        16**a
        * a
        * (a * (1 - 2 * y) ** 2 - 2 * y**2 + 2 * y - 1)
        * ((x - 1) * x * (y - 1) * y) ** a
        / ((y - 1) ** 2 * y**2)
    )
    return -u_xx - u_yy


def output_transform(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]

    return x_in * y_in * (1 - x_in) * (1 - y_in) * y

def gen_test_x(num):
    x = np.linspace(0, 1, num)
    y = np.linspace(0, 1, num)
    l = []

    for i in range(len(y)):
        for j in range(len(x)):
            l.append([x[j], y[i]])
    t = np.array(l)
    x_2 = t[:, 0:1]
    y_2 = t[:, 1:2]
    X_1 = np.hstack((x_2, y_2))
    X = np.insert(t, [2], X_1, axis=1)
    return X

def gen_test_x2(num):
    x = np.linspace(0, 1, num)
    y = np.linspace(0, 1, num)
    l = []

    for i in range(len(y)):
        for j in range(len(x)):
            l.append([x[j], y[i]])
    t = np.array(l)
    x_2 = t[:, 0:1] * t[:, 0:1]
    y_2 = t[:, 1:2] * t[:, 1:2]
    X_1 = np.hstack((x_2, y_2))
    X = np.insert(t, [2], X_1, axis=1)
    return X




def sol(t):
    x = t[:, 0:1]
    y = t[:, 1:2]

    return (16 * x * y * (1 - x) * (1 - y)) ** a

##dapinn copy

def DApde(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]
    du_x2x2 = dde.grad.hessian(y, x, i=2, j=2)
    du_xx2 = dde.grad.hessian(y, x, i=0, j=2)
    du_y2y2 = dde.grad.hessian(y, x, i=3, j=3)
    du_yy2 = dde.grad.hessian(y, x, i=1, j=3)
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    return (du_xx + 2*du_xx2 + du_x2x2) + \
           (du_yy + 2*du_yy2 + du_y2y2) + \
           f(x_in, y_in)




geom = dde.geometry.Rectangle([0, 0], [1, 1])
DAdata = dde.data.PDE(geom, DApde, [],solution=sol, num_domain=500, num_test=10000,mode = "copy2xy")
DAnet = dde.maps.FNN([4] + [50] * 3 + [1], "tanh", "Glorot normal")
DAnet.apply_output_transform(output_transform)

DAPINNmodel = dde.Model(DAdata, DAnet)
time_start = time.time()  # 记录开始时间
DAPINNmodel.compile("adam", lr=0.0001,metrics=["l2 relative error"])
losshistory, train_state = DAPINNmodel.train(epochs=100000, callbacks=[])
DAPINNmodel.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state =DAPINNmodel.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)
# 259.08
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
dapinncopyerror=best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)
##DAPINN X2
def dapinnx2pde(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]
    du_x2x2 = dde.grad.hessian(y, x, i=2, j=2)
    du_xx2 = dde.grad.hessian(y, x, i=0, j=2)
    du_x2 = dde.grad.jacobian(y, x, i=0, j=2)
    du_y2y2 = dde.grad.hessian(y, x, i=3, j=3)
    du_yy2 = dde.grad.hessian(y, x, i=1, j=3)
    du_y2 = dde.grad.jacobian(y, x, i=0, j=3)

    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    return (du_xx + 4 * x[:, 0:1] * du_xx2 + 4 * (x[:, 0:1] ** 2) * du_x2x2 + 2 * du_x2) + \
           (du_yy + 4 * x[:, 1:2] * du_yy2 + 4 * (x[:, 1:2] ** 2) * du_y2y2 + 2 * du_y2) + \
           f(x_in, y_in)
geom = dde.geometry.Rectangle([0, 0], [1, 1])
dax2data = dde.data.PDE(geom, dapinnx2pde, [],solution=sol, num_domain=500, num_test=10000,mode="x2xy")
n = 10

dax2net = dde.maps.FNN([4] + [50] * 3 + [1], "tanh", "Glorot normal")
dax2net.apply_output_transform(output_transform)

DAPINNX2model = dde.Model(dax2data, dax2net)
time_start = time.time()  # 记录开始时间
DAPINNX2model.compile("adam", lr=0.001,metrics=["l2 relative error"])
DAPINNX2model.train(epochs=100000, callbacks=[])
DAPINNX2model.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state =DAPINNX2model.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)
# 259.08
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
dapinnx2error=best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)
#PINN



def pinnpde(x, y):
    x_in = x[:, 0:1]
    y_in = x[:, 1:2]
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, i=1, j=1)
    return du_xx + du_yy + f(x_in, y_in)




geom = dde.geometry.Rectangle([0, 0], [1, 1])
PINNdata = dde.data.PDE(geom, pinnpde, [], num_domain=500,solution=sol, num_test=10000)
PINNnet = dde.maps.FNN([2] + [50] *3  + [1], "tanh", "Glorot normal")
PINNnet.apply_output_transform(output_transform)

PINNmodel = dde.Model(PINNdata, PINNnet)
time_start = time.time()  # 记录开始时间
PINNmodel.compile("adam", lr=0.0001,metrics=["l2 relative error"])
PINNmodel.train(epochs=100000, callbacks=[])
PINNmodel.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state =PINNmodel.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)# 175.62s
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step= np.argmin(loss_train)
best_metrics=np.array(losshistory.metrics_test)[best_step,:]
pinnerror=best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)


#print error
print("daPINN with copy :L2 relative error of u",
    dapinncopyerror)
print("daPINN with x^2 :L2 relative error of u",
    dapinnx2error)
print("PINN :L2 relative error of u",
    pinnerror)
#PLOT
plt.rcParams.update({"font.size": 22})
##dapinn copy

X = gen_test_x(100)
y = DAPINNX2model.predict(X).tolist()

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(y[i][0])
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("DAPINNX2 Prediction")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
im.set_clim(0, 1)
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()
########################################


y = DAPINNX2model.predict(X).tolist()
y_true = sol(X)

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(abs(y[i][0] - y_true[i][0]))
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("DAPINNX2 Absolute Error of u")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()

##dapinnx2

X = gen_test_x2(100)
y = DAPINNX2model.predict(X).tolist()

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(y[i][0])
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("DAPINNX2 Prediction")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
im.set_clim(0, 1)
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()
########################################


y = DAPINNX2model.predict(X).tolist()
y_true = sol(X)

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(abs(y[i][0] - y_true[i][0]))
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("DAPINNX2 Absolute Error of u")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()
##pinn

X = gen_test_x2(100)
X=X[:,0:2]
y = PINNmodel.predict(X).tolist()

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(y[i][0])
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("PINN Prediction")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
im.set_clim(0, 1)
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()
########################################



y = PINNmodel.predict(X).tolist()
y_true = sol(X)

disp = []
prev = X[0][1]
temp = []

for i in range(len(y)):

    if X[i][1] == prev:
        temp.append(abs(y[i][0] - y_true[i][0]))
    else:

        prev = X[i][1]

        temp2 = []
        for elem in temp:
            temp2.append((elem))
        disp.append(temp2)
        temp.clear()

        temp.append(y[i][0])
disp.reverse()
plt.figure(figsize=(7, 7))
plt.xlabel("x")
plt.ylabel("y")
plt.title("PINN Absolute Error of u")

ax = plt.gca()
im = ax.imshow(disp, extent=(0, 1, 0, 1))
ax.set_aspect(1)

divider = make_axes_locatable(ax)
width = ax.get_position().width
height = ax.get_position().height
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(im, cax=cax)
plt.show()

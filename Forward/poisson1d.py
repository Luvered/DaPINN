"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import dapinn as dde
import matplotlib.pyplot as plt
import numpy as np
from dapinn.backend import tf




def gen_test_x(num):
    x = np.linspace(0, np.pi, num)
    x_2 = x
    l = []
    for i in range(len(x_2)):
            l.append([x[i], x_2[i]])
    x1 = np.array(l)

    return x1


def gen_test_x2(num):
    x = np.linspace(0, np.pi, num)
    x_2 = x*x
    l = []
    for i in range(len(x_2)):
            l.append([x[i], x_2[i]])
    x1 = np.array(l)

    return x1


def boundary(x, on_boundary):
    return on_boundary


def func(x):
    sol = x[:,0:1] + 1 / 8 * np.sin(8 * x[:,0:1])+ 1 / 7 * np.sin(7 * x[:,0:1])
    for i in range(1, 4):
        sol += 1 / i * np.sin(i * x[:,0:1])
    return sol
#DaPINN with copy

def DaPINNcopypde(x, y):
    du_xx = dde.grad.hessian(y, x,  i=0,j=0)
    du_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    du_x1x2 = dde.grad.hessian(y, x, i=0, j=1)

    f = 8 * tf.sin(8 * x[:,0:1])+7 * tf.sin(7 * x[:,0:1])
    for i in range(1, 4):
        f += i * tf.sin(i*x[:,0:1])
    return -(du_xx +  du_x2x2 + 2 * du_x1x2) - f

geom = dde.geometry.Interval(0, np.pi)
bc = dde.icbc.DirichletBC(geom, func, boundary)
data = dde.data.PDE(geom, DaPINNcopypde, bc, 24, 2 , solution=func, num_test=2000,mode="copy2")

layer_size = [2] + [20] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

DaPINNcopymodel = dde.Model(data, net)
DaPINNcopymodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
DaPINNcopymodel.train(iterations=20000)
DaPINNcopymodel.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state=DaPINNcopymodel.train()
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step= np.argmin(loss_train)
best_metrics=np.array(losshistory.metrics_test)[best_step,:]
dapinncopyerror=best_metrics[0]
print(
    "DaPINN with copy:L2 relative error of u",
    best_metrics[0]
)


#DaPINN with x^2


def DaPINNpde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0,j=0)
    dy_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    dy_x2 = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx2 = dde.grad.hessian(y, x, i=0, j=1)
    f = 8 * tf.sin(8 * x[:,0:1])+7 * tf.sin(7 * x[:,0:1])
    for i in range(1, 4):
        f += i * tf.sin(i *x[:,0:1])
    return -(dy_xx + 4 * x[:, 0:1] * dy_xx2 + 4 * (x[:, 0:1]**2) * dy_x2x2 + 2 * dy_x2) - f





#DaPINN

geom = dde.geometry.Interval(0, np.pi)
bc = dde.icbc.DirichletBC(geom, func, boundary)
DaPINNx2data = dde.data.PDE(geom, DaPINNpde, bc, 24, 2 , solution=func, num_test=2000,mode="x2")

layer_size = [2] + [20] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
DaPINNx2net = dde.nn.FNN(layer_size, activation, initializer)
DaPINNx2model = dde.Model(DaPINNx2data, DaPINNx2net)
DaPINNx2model.compile("adam", lr=0.001, metrics=["l2 relative error"])
DaPINNx2model.train(iterations=20000)
DaPINNx2model.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state=DaPINNx2model.train()


loss_train = np.sum(losshistory.loss_train, axis=1)
best_step= np.argmin(loss_train)
best_metrics=np.array(losshistory.metrics_test)[best_step,:]
dapinnx2error=best_metrics[0]
print(
    "DaPINN with x^2: L2 relative error of u",
    best_metrics[0]
)



#pinn

def PINNpde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    f = 8 * tf.sin(8 * x)+7 * tf.sin(7 * x)
    for i in range(1, 4):
        f += i * tf.sin(i * x)
    return -dy_xx - f

def func(x):
    sol = x + 1 / 8 * np.sin(8 * x)+1 / 7 * np.sin(7 * x)
    for i in range(1, 4):
        sol += 1 / i * np.sin(i * x)
    return sol

geom = dde.geometry.Interval(0, np.pi)
bc = dde.icbc.DirichletBC(geom, func, boundary)
PINNdata = dde.data.PDE(geom, PINNpde, bc, 24, 2, solution=func, num_test=2000)

layer_size = [1] + [20] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
PINNnet = dde.nn.FNN(layer_size, activation, initializer)

PINNmodel = dde.Model(PINNdata, PINNnet)
PINNmodel.compile("adam", lr=0.001, metrics=["l2 relative error"])
PINNmodel.train(iterations=20000)
PINNmodel.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state=PINNmodel.train()
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
pinnerror=best_metrics[0]
print(
    "PINN :L2 relative error of u",
    best_metrics[0]
)

# Print ERROR
print("daPINN with copy :L2 relative error of u",
    dapinncopyerror)
print("daPINN with x^2 :L2 relative error of u",
    dapinnx2error)
print("PINN :L2 relative error of u",
    pinnerror)
# Plot PDE residual
def _pack_data(train_state):
    def merge_values(values):
        if values is None:
            return None
        return np.hstack(values) if isinstance(values, (list, tuple)) else values

    y_train = merge_values(train_state.y_train)
    y_test = merge_values(train_state.y_test)
    best_y = merge_values(train_state.best_y)
    best_ystd = merge_values(train_state.best_ystd)
    return y_train, y_test, best_y, best_ystd

x = gen_test_x(500)
x2 = gen_test_x2(500)
y_train, y_test, best_y, best_ystd = _pack_data(train_state)
ye=func(x[:,0:1])
y1 = PINNmodel.predict(x[:,0:1])
y2 = DaPINNcopymodel.predict(x)
y3 = DaPINNx2model.predict(x2)
fig=plt.figure()
ax=plt.axes()
ax.plot(train_state.X_train[:, 0], y_train[:, 0], "ok", label="Train")
ax.plot(x[:,0:1],ye,c='k',label="ture" )
ax.plot(x[:,0:1],y1,linestyle='--',c=(51/255,102/255,153/255),label="PINN" )
ax.plot(x[:,0:1],y2,linestyle='--',c=(230/255,190/255,47/255),label="DaPINN with x")
ax.plot(x[:,0:1],y3,linestyle='--',c=(239/255,0/255,0/255),label="DaPINN with x^2")
plt.xlim(0.75,2)
plt.ylim(1.7,2.7)
plt.xlabel('x', fontsize =20)
plt.ylabel('u', fontsize =20)
ax.grid()
plt.legend(loc='lower right',framealpha=1,frameon=True);
plt.tight_layout()
plt.show()

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""
import dapinn as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
from dapinn.backend import tf
from matplotlib import pyplot as plt
# Backend pytorch
# import torch
# Backend paddle
# import paddle


C = dde.Variable(2.0)

# 1.PINN

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    # Backend tensorflow.compat.v1 or tensorflow
    return (
            dy_t
            - C * dy_xx
            + tf.exp(-x[:, 1:])
            * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
    )

def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

Num = 5

observe_x = np.vstack((np.linspace(-1, 1, num=Num), np.full((Num), 1))).T
observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x), component=0)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    NN_type=0,
    num_domain=20,
    num_boundary=10,
    num_initial=5,
    anchors=observe_x,
    solution=func,
    num_test=10000,
)

layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C
)
variable = dde.callbacks.VariableValue(C, period=1000)
losshistory, train_state = model.train(iterations=3 * 10000, callbacks=[variable])

dde.saveplot(losshistory, train_state, issave=False, isplot=False)

X = np.vstack((np.linspace(-1, 1, num=100), np.full((100), 0.5))).T
y_true = func(X)
y_pred = model.predict(X)

X = np.vstack((np.linspace(-1, 1, num=10), np.full((10), 0.5))).T
y_true = func(X)
y_pred = model.predict(X)


x_flat = np.linspace(-1, 1, 80)
t_flat = np.linspace( 0, 1, 40)
x, t = np.meshgrid(x_flat, t_flat)

u = model.predict(np.stack([x.flatten(), t.flatten()], axis=-1))
uhat = func(np.stack([x.flatten(), t.flatten()], axis=-1))
u = u.reshape(x.shape)
uhat = uhat.reshape(x.shape)

A = np.random.rand(1000)*2-1
B = np.abs(np.random.rand(1000))
X = np.vstack((A,B)).T
y_true = func(X)
y_pred = model.predict(X)
f = model.predict(X, operator=pde)

mr = np.mean(np.absolute(f))
mse = dde.metrics.l2_relative_error(y_true, y_pred)
ce = abs(variable.value[0]-1)

fig1 = plt.figure()
plt.title('solution of u(x,t)')
plt.pcolormesh(x,t,uhat,cmap='inferno')
cb=plt.colorbar()
cb.set_label('u(x,t)')
plt.xlabel('x')
plt.ylabel('t')

fig2 = plt.figure()
plt.title('PINN: error')
plt.pcolormesh(x,t,u,cmap='inferno')
cb=plt.colorbar()
cb.set_label('u(x,t)')
plt.xlabel('x')
plt.ylabel('t')

fig3 = plt.figure()
plt.title('PINN: u(x)')
plt.pcolormesh(x,t,u-uhat,cmap='inferno') 
cb=plt.colorbar()
cb.set_label('error(x,t)')
plt.xlabel('x')
plt.ylabel('t')

# 2. DaPINN

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_x2=dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_xx2 = dde.grad.hessian(y, x, i=0, j=1)
    dy_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    return (
        dy_t
        - C * (dy_xx + 4 * x[:, 0:1] * dy_xx2 + 4 * x[:, 1:2] * dy_x2x2 + 2 * dy_x2)
        + tf.exp(-x[:, 2:3])
        * (tf.sin(np.pi * x[:, 0:1]) - np.pi ** 2 * tf.sin(np.pi * x[:, 0:1]))
    )

def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 2:3])
         
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

Num = 5

observe_x = np.vstack((np.linspace(-1, 1, num=Num), np.linspace(-1, 1, num=Num)**2, np.full((Num), 1))).T
observe_y = dde.icbc.PointSetBC(observe_x, func(observe_x))
observe_x = np.vstack((np.linspace(-1, 1, num=Num), np.full((Num), 1))).T

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, observe_y],
    NN_type=1,
    num_domain = 20,
    num_boundary = 10,
    num_initial = 5,
    anchors=observe_x,
    solution=func,
    num_test=10000,
    train_distribution='uniform'
)

layer_size = [3] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)

model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], external_trainable_variables=C
)

variable = dde.callbacks.VariableValue(C, period=1000, filename="variables"+str(0)+".dat")

losshistory, train_state = model.train(iterations=30000, callbacks=[variable])


dde.saveplot(losshistory, train_state, issave=False, isplot=False)

A = np.random.rand(1000)*2-1
A2 = A**2
B = np.abs(np.random.rand(1000))
X = np.vstack((A,A2,B)).T
y_true = func(X)
y_pred = model.predict(X)
f = model.predict(X, operator=pde)


print("—————————— PINN ——————————")
print("Mean residual:", mr)
print("L2 relative error:", mse)
print("C error", ce)

print("—————————— DaPINN ——————————")
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
print("C error",abs(variable.value[0]-1))

np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

X = np.vstack((np.linspace(-1, 1, num=100), np.linspace(-1, 1, num=100)**2, np.full((100), 0.5))).T
y_true = func(X)
y_pred = model.predict(X)

# plot
x_flat = np.linspace(-1, 1, 80)
t_flat = np.linspace( 0, 1, 40)
x, t = np.meshgrid(x_flat, t_flat)

u = model.predict(np.stack([x.flatten(), x.flatten()**2, t.flatten()], axis=-1))
uhat = func(np.stack([x.flatten(), x.flatten()**2, t.flatten()], axis=-1))
u = u.reshape(x.shape)
uhat = uhat.reshape(x.shape)

fig4 = plt.figure()
plt.title('DaPINN: u(x,t)')
plt.pcolormesh(x,t,u,cmap='inferno')
cb=plt.colorbar()
cb.set_label('u(x,t)')
plt.xlabel('x')
plt.ylabel('t')

fig5 = plt.figure()
plt.title('DaPINN: error')
plt.pcolormesh(x,t,u-uhat,cmap='inferno') 
cb=plt.colorbar()
cb.set_label('error(x,t)')
plt.xlabel('x')
plt.ylabel('t')


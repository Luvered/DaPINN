"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.pyplot import MultipleLocator
import dapinn as dde
import numpy as np
import time


def DApinngen_testdata():
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    x_2 = X[:, 0:1] * X[:, 0:1]
    t_2 = X[:, 1:2] * X[:, 1:2]
    x_3 = X[:, 0:1] * X[:, 0:1] * X[:, 0:1]
    X_N = np.hstack((x_3, t_2,x_2))
    X = np.insert(X, [2], X_N, axis=1)
    X[:, [1, 4]] = X[:, [4, 1]]
    y = exact.flatten()[:, None]
    return X, y


def DApinnpde(x, y):
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
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

    return dy_t+2*x[:, 4:5]*dy_t2+ y * (du_x + 2 * x[:, 0:1] * du_x2 + 3 * x[:, 0:1] ** 2 * du_x3) - 0.01 / np.pi * (du_xx + \
           4 * x[:,0:1] ** 2 * du_x2x2 + 9 * x[:, 0:1] ** 4 * du_x3x3 + 4 * x[:,0:1] * du_x1x2 + 6 * x[:,0:1] ** 2 * du_x1x3 + \
           12 * x[:,0:1] ** 3 * du_x2x3 + 2 * du_x2 + 6 * x[:,0:1] * du_x3)


def DApinngen_randomnumtest(num):
    X = geomtime.random_points(num)
    x_2 = X[:, 0:1] * X[:, 0:1]
    t_2 = X[:, 1:2] * X[:, 1:2]
    x_3 = X[:, 0:1] * X[:, 0:1] * X[:, 0:1]
    X_N = np.hstack((x_3, t_2,x_2))
    X = np.insert(X, [2], X_N, axis=1)
    X[:, [1, 4]] = X[:, [4, 1]]
    return X


def DApinngen_plt_x():
    X, y_true = DApinngen_testdata()
    y_pred = DApinnmodel.predict(X)
    x_1  = np.linspace(-1, 1, 256)
    t_1 = np.linspace(0, 0.99, 100)

    error = abs(y_true - y_pred)
    y_pred = np.reshape(y_pred[:, 0], (100, 256)).T
    error = np.reshape(error[:, 0], (100, 256)).T
    y_true = np.reshape(y_true[:, 0], (100, 256)).T
    return y_pred, error, y_true, x_1, t_1

def plt1d(x,y_ture,y_predict):
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
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(x, y_ture_2, "-k", label="True")
    plt.plot(x, y_predict_2, "--r", label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(x, y_ture_3, "-k", label="True")
    plt.plot(x, y_predict_3, "--r", label="Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def plot(y1, error, y_true, x_1, t_1):


    # 进行绘图
    cmax = np.max(y1)
    cmin = np.min(y1)
    fig, ax = plt.subplots()
    levels = np.arange(cmin * 1.01, cmax * 1.01, (cmax - cmin) / 1000)  # 对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
    cs = ax.contourf(t_1, x_1, y1, levels, cmap=plt.get_cmap('Spectral'))
    # 添加colorbar
    cbar = fig.colorbar(cs, fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))  # 对colorbar的大小进行设置
    # 设置颜色条的刻度
    tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
    cbar.locator = tick_locator
    cbar.ax.tick_params(labelsize=12.5)
    # 设置颜色条的title
    cbar.ax.set_title('u', fontsize=12.5)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # 设置坐标刻度及间隔
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    x_major_locator = MultipleLocator(0.2)  # 刻度间隔
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # 设置坐标轴标签
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 14,
             }
    ax.set_xlabel('$\t{t}$', font3)
    ax.set_ylabel('$\t{x}$', font3)
    # 设置坐标刻度字体大小
    ax.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    # 设置图像像素及大小
    plt.rcParams['figure.figsize'] = (6.0, 4.5)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.show()



    # 进行绘图
    cmax = np.max(error)
    cmin = np.min(error)
    fig, ax = plt.subplots()
    levels = np.arange(cmin, cmax, (cmax - cmin) / 1000)  # 对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
    cs = ax.contourf(t_1, x_1, error, levels, cmap=plt.get_cmap('Spectral'))
    # 添加colorbar
    cbar = fig.colorbar(cs, fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))  # 对colorbar的大小进行设置
    # 设置颜色条的刻度
    tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
    cbar.locator = tick_locator
    cbar.ax.tick_params(labelsize=12.5)
    # 设置颜色条的title
    cbar.ax.set_title('error', fontsize=12.5)
    cbar.update_ticks()  # 显示colorbar的刻度值
    # 设置坐标刻度及间隔
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    x_major_locator = MultipleLocator(0.2)  # 刻度间隔
    y_major_locator = MultipleLocator(0.2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # 设置坐标轴标签
    font3 = {'family': 'Arial',
             'weight': 'normal',
             'size': 14,
             }
    ax.set_xlabel('$\t{t}$', font3)
    ax.set_ylabel('$\t{x}$', font3)
    # 设置坐标刻度字体大小
    ax.tick_params(labelsize=14)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    # 设置图像像素及大小
    plt.rcParams['figure.figsize'] = (6.0, 4.5)
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率

    plt.show()
    return 0


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

DApinndata = dde.data.TimePDE(
    geomtime, DApinnpde, [bc, ic], num_domain=2000, num_boundary=100, num_initial=200,mode="x3tt"
)
net = dde.nn.FNN([5] + [40] * 4 + [1], "tanh", "Glorot normal")
DApinnmodel = dde.Model(DApinndata, net)

time_start = time.time()  # 记录开始时间

DApinnmodel.compile("adam", lr=1e-3)
losshistory, train_state = DApinnmodel.train(epochs=20000)
DApinnmodel.compile("L-BFGS")
DApinnmodel.train()

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)

#
X, y_true = DApinngen_testdata()
y_pred = DApinnmodel.predict(X)

f = DApinnmodel.predict(X, operator=DApinnpde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
dapinnx3error=dde.metrics.l2_relative_error(y_true, y_pred)
###############################
#PINN
def gen_testdata():
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx


def gen_plt_x():
    X, y_ture = gen_testdata()
    y_pred = model.predict(X)
    x_1= np.linspace(-1, 1, 256)
    t_1= np.linspace(0, 0.99, 100)

    error = abs(y_ture-y_pred)
    y_pred = np.reshape(y_pred[:, 0], (100, 256)).T
    error = np.reshape(error[:, 0], (100, 256)).T
    y_ture=np.reshape(y_ture[:,0], (100, 256)).T
    return y_pred,error,y_ture,x_1,t_1

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(
    geomtime, pde, [bc, ic], num_domain=2000, num_boundary=100, num_initial=200
)
net = dde.nn.FNN([2] + [40] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

time_start = time.time()  # 记录开始时间

model.compile("adam", lr=1e-3)
model.train(iterations=20000)
model.compile("L-BFGS")
model.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)

X, y_true = gen_testdata()
y_pred = model.predict(X)
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
pinnerror=dde.metrics.l2_relative_error(y_true, y_pred)


# print error
print("daPINN with x^3 :L2 relative error of u",
      dapinnx3error)
print("PINN :L2 relative error of u",
      pinnerror)


# plots
##########################################
y_predict, error, y_true, x_1, t_1 = DApinngen_plt_x()
plt1d(x_1,y_true,y_predict)
plot(y_predict, error, y_true, x_1, t_1)
y_predict, error, y_true, x_1, t_1 = gen_plt_x()
plt1d(x_1,y_true,y_predict)
plot(y_predict, error, y_true, x_1, t_1)
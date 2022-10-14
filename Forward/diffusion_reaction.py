"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

import numpy as np
import dapinn as dde
from dapinn.backend import tf
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from matplotlib.pyplot import MultipleLocator
import time
# aPINN fourier
def daPINNfourierpde(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 3:4]
    dy_t = dde.grad.jacobian(y, x, i=0, j=3)
    dy_x2= dde.grad.jacobian(y, x, i=0, j=1)
    dy_x3 = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    dy_x3x3 = dde.grad.hessian(y, x, i=2, j=2)
    dy_xx2 = dde.grad.hessian(y, x, i=0, j=1)
    dy_x1x3 = dde.grad.hessian(y, x, i=0, j=2)
    dy_x2x3 = dde.grad.hessian(y, x, i=1, j=2)
    r = tf.exp(-t_in) * (
            3 * tf.sin(2 * x_in) / 2
            + 8 * tf.sin(3 * x_in) / 3
            + 15 * tf.sin(4 * x_in) / 4
            + 63 * tf.sin(8 * x_in) / 8
    )

    return dy_t  - (dy_xx + (tf.cos(x[:,0:1])**2)*dy_x2x2+(tf.sin(x[:,0:1])**2)*dy_x3x3 + 2*tf.cos(x[:,0:1])*dy_xx2 -\
                 2*tf.sin(x[:,0:1])*dy_x1x3  -2*tf.cos(x[:,0:1])*tf.sin(x[:,0:1])*dy_x2x3  -\
                 x[:,1:2]*dy_x2-x[:,2:3]*dy_x3) - r


def solution(a):
    x, t = a[:, 0:1], a[:, 3:4]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) * val


def icfunc(x):
    return (
            tf.sin(8 * x) / 8
            + tf.sin(1 * x) / 1
            + tf.sin(2 * x) / 2
            + tf.sin(3 * x) / 3
            + tf.sin(4 * x) / 4
    )


def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 3:4]

    return (x_in - np.pi) * (x_in + np.pi) * (1 - tf.exp(-t_in)) * y + icfunc(x_in)


# Computational geometry:
geom = dde.geometry.Interval(-np.pi, np.pi)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

daPINNfourierdata = dde.data.TimePDE(
    geomtime,
    daPINNfourierpde,
    [],
    num_domain=60,
    # train_distribution="uniform",
    solution=solution,
    num_test=10000,mode="fourier t"
)

layer_size = [4] + [20] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
daPINNfouriernet = dde.maps.FNN(layer_size, activation, initializer)

daPINNfouriernet.apply_output_transform(output_transform)
daPINNfouriermodel = dde.Model(daPINNfourierdata, daPINNfouriernet)
time_start = time.time()  # 记录开始时间
daPINNfouriermodel.compile("adam", lr=0.0001, metrics=["l2 relative error"])
daPINNfouriermodel.train(epochs=100000)
daPINNfouriermodel.compile("L-BFGS",metrics=["l2 relative error"])
losshistory, train_state = daPINNfouriermodel.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)# 训练时间:  239.08803939819336
# Plot/print the results
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
dapinnfouriererror=best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)

# daPINN x^3
def daPINNx3pde(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 3:4]
    dy_t = dde.grad.jacobian(y, x, i=0, j=3)
    du_xx = dde.grad.hessian(y, x, i=0, j=0)
    du_x2x2 = dde.grad.hessian(y, x, i=1, j=1)
    du_x1x2 = dde.grad.hessian(y, x, i=0, j=1)
    du_x2 = dde.grad.jacobian(y, x, i=0, j=1)
    du_x1x3 = dde.grad.hessian(y, x, i=0, j=2)
    du_x2x3 = dde.grad.hessian(y, x, i=1, j=2)
    du_x3x3 = dde.grad.hessian(y, x, i=2, j=2)
    du_x3 = dde.grad.jacobian(y, x, i=0, j=2)

    r = tf.exp(-t_in) * (
            3 * tf.sin(2 * x_in) / 2
            + 8 * tf.sin(3 * x_in) / 3
            + 15 * tf.sin(4 * x_in) / 4
            + 63 * tf.sin(8 * x_in) / 8
    )

    return (dy_t) - (du_xx + 4 * x[:, 0:1]**2 * du_x2x2 + 9 * x[:, 0:1] **4 * du_x3x3 + 4 * x[:, 0:1] * du_x1x2 + 6 * x[:, 1:2] * du_x1x3 + 12 * x[:, 2:3]  * du_x2x3 + 2 * du_x2 + 6 * x[:,0:1] * du_x3) - r



# Computational geometry:
geom = dde.geometry.Interval(-np.pi, np.pi)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

daPINNx3data = dde.data.TimePDE(
    geomtime,
    daPINNx3pde,
    [],
    num_domain=60,
    # train_distribution="uniform",
    solution=solution,
    num_test=10000,mode="x3t"
    )

layer_size = [4] + [20] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
daPINNx3net = dde.maps.FNN(layer_size, activation, initializer)

daPINNx3net.apply_output_transform(output_transform)
daPINNx3model = dde.Model(daPINNx3data, daPINNx3net)
time_start = time.time()  # 记录开始时间
daPINNx3model.compile("adam", lr=0.0001, metrics=["l2 relative error"])
daPINNx3model.train(epochs=100000)
daPINNx3model.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state=daPINNx3model.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)  # 训练时间:    223.429025888443

# Plot/print the results
loss_train = np.sum(losshistory.loss_train, axis=1)
best_step = np.argmin(loss_train)
best_metrics = np.array(losshistory.metrics_test)[best_step, :]
dapinnx3error=best_metrics[0]
print(
    "L2 relative error of u",
    best_metrics[0]
)

# PINN
def PINNpde(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    r = tf.exp(-t_in) * (
        3 * tf.sin(2 * x_in) / 2
        + 8 * tf.sin(3 * x_in) / 3
        + 15 * tf.sin(4 * x_in) / 4
        + 63 * tf.sin(8 * x_in) / 8
    )


    return [dy_t - dy_xx - r]


def solution(a):
    x, t = a[:, 0:1], a[:, 1:2]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) * val


def icfunc(x):
    return (
        tf.sin(8 * x) / 8
        + tf.sin(1 * x) / 1
        + tf.sin(2 * x) / 2
        + tf.sin(3 * x) / 3
        + tf.sin(4 * x) / 4
    )


def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    return (x_in - np.pi) * (x_in + np.pi) * (1 - tf.exp(-t_in)) * y + icfunc(x_in)



# Computational geometry:
geom = dde.geometry.Interval(-np.pi, np.pi)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
x = geomtime.random_points(10000)
data = dde.data.TimePDE(
    geomtime,
    PINNpde,
    [],
    num_domain=60,
    # train_distribution="uniform",
    solution=solution,
    num_test=10000,
)

layer_size = [2] + [20] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

net.apply_output_transform(output_transform)

PINNmodel = dde.Model(data, net)
time_start = time.time()  # 记录开始时间
PINNmodel.compile("adam", lr=0.0001, metrics=["l2 relative error"])

PINNmodel.train(epochs=100000)
PINNmodel.compile("L-BFGS", metrics=["l2 relative error"])
losshistory, train_state = PINNmodel.train()
time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('训练时间: ', time_sum)  # 训练时间:  239.08803939819336
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
print("daPINN with Fourier:L2 relative error of u",
      dapinnfouriererror)
print("daPINN with x^3 :L2 relative error of u",
      dapinnx3error)
print("PINN :L2 relative error of u",
      pinnerror)

# plots dapinn fourier
##########################################
def gen_plt_xfourier(num):
    x = np.linspace(-np.pi, np.pi, num)
    t = np.linspace(0, 1, num)
    l = []

    for i in range(len(t)):
        for j in range(len(x)):
            l.append([x[j], t[i]])
    l = np.array(l)
    x_2 = np.sin(l[:, 0])
    x_3 = np.cos(l[:, 0])

    X_N_1 = np.vstack(( x_3,x_2))
    X_N_1 = X_N_1.T
    l = np.insert(l, [2], X_N_1, axis=1)
    l[:, [1, 3]] = l[:, [3, 1]]

    y = PINNmodel.predict(l)
    y_true = solution(l)
    error = abs(y_true-y)
    y = np.reshape(y[:, 0], (num, num))
    error = np.reshape(error[:, 0], (num, num))
    y_true=np.reshape(y_true[:,0], (num, num))
    return y,error,y_true,x,t

y1, error, y_true, x_1, t_1 = gen_plt_xfourier(500)

#进行绘图

cmax=np.max(y1)
cmin=np.min(y1)
fig, ax = plt.subplots()
levels = np.arange(cmin,cmax,(cmax-cmin)/1000.0)#对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
cs = ax.contourf(x_1 , t_1, y1,levels,cmap=plt.get_cmap('Spectral'))
#添加colorbar
cbar = fig.colorbar(cs,fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))#对colorbar的大小进行设置
#设置颜色条的刻度
tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=12.5)
#设置颜色条的title
cbar.ax.set_title('u',fontsize=12.5)
cbar.update_ticks()#显示colorbar的刻度值
#设置坐标刻度及间隔
ax.set_xlim(-3.14,3.14)
ax.set_ylim(0,1)
x_major_locator=MultipleLocator(1)#刻度间隔
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
#设置坐标轴标签
font3 = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 14,
         }
ax.set_xlabel('$\t{x}$',font3)
ax.set_ylabel('$\t{t}$',font3)
#设置坐标刻度字体大小
ax.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
#设置图像像素及大小
plt.rcParams['figure.figsize']=(8.0,6)
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
fig.savefig('FIG9_f.eps', format='eps', dpi=100)
plt.show()
'''


'''
#进行绘图
cmax=np.max(error)
cmin=np.min(error)
fig, ax = plt.subplots()
levels = np.arange(cmin,cmax,(cmax-cmin)/1000)#对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
cs = ax.contourf(x_1,t_1 , error,levels,cmap=plt.get_cmap('Spectral'))
#添加colorbar
cbar = fig.colorbar(cs,fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))#对colorbar的大小进行设置
#设置颜色条的刻度
tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=12.5)
#设置颜色条的title
cbar.ax.set_title('error',fontsize=12.5)
cbar.update_ticks()#显示colorbar的刻度值
#设置坐标刻度及间隔
ax.set_xlim(-3.14,3.14)
ax.set_ylim(0,1)
x_major_locator=MultipleLocator(1)#刻度间隔
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
#设置坐标轴标签
font3 = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 14,
         }
ax.set_xlabel('$\t{x}$',font3)
ax.set_ylabel('$\t{t}$',font3)
#设置坐标刻度字体大小
ax.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
#设置图像像素及大小
plt.rcParams['figure.figsize']=(6.0,4.5)
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
fig.savefig('FIG9_i.eps', format='eps', dpi=300)
plt.show()

# plots dapinn x^3
##########################################


def gen_plt_xdaPINNx3(num):
    x = np.linspace(-np.pi, np.pi, num)
    t = np.linspace(0, 1, num)
    l = []

    for i in range(len(t)):
        for j in range(len(x)):
            l.append([x[j], t[i]])
    l = np.array(l)
    x_2 = l[:, 0] * l[:, 0]
    x_3 = l[:, 0] * l[:, 0] * l[:, 0]

    X_N_1 = np.vstack(( x_3,x_2))
    X_N_1 = X_N_1.T
    l = np.insert(l, [2], X_N_1, axis=1)
    l[:, [1, 3]] = l[:, [3, 1]]

    y = daPINNx3model.predict(l)
    y_true = solution(l)
    error = abs(y_true-y)
    y = np.reshape(y[:, 0], (num, num))
    error = np.reshape(error[:, 0], (num, num))
    y_true=np.reshape(y_true[:,0], (num, num))
    return y,error,y_true,x,t


y1,error,y_true,x_1,t_1 = gen_plt_xdaPINNx3(500)

#进行绘图

cmax=np.max(y1)
cmin=np.min(y1)
fig, ax = plt.subplots()
levels = np.arange(cmin,cmax,(cmax-cmin)/1000.0)#对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
cs = ax.contourf(x_1 , t_1, y1,levels,cmap=plt.get_cmap('Spectral'))
#添加colorbar
cbar = fig.colorbar(cs,fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))#对colorbar的大小进行设置
#设置颜色条的刻度
tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=12.5)
#设置颜色条的title
cbar.ax.set_title('u',fontsize=12.5)
cbar.update_ticks()#显示colorbar的刻度值
#设置坐标刻度及间隔
ax.set_xlim(-3.14,3.14)
ax.set_ylim(0,1)
x_major_locator=MultipleLocator(1)#刻度间隔
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
#设置坐标轴标签
font3 = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 14,
         }
ax.set_xlabel('$\t{x}$',font3)
ax.set_ylabel('$\t{t}$',font3)
#设置坐标刻度字体大小
ax.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
#设置图像像素及大小
plt.rcParams['figure.figsize']=(8.0,6)
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
fig.savefig('FIG9_e.eps', format='eps', dpi=100)
plt.show()
'''

'''
#进行绘图
cmax=np.max(error)
cmin=np.min(error)
fig, ax = plt.subplots()
levels = np.arange(cmin,cmax,(cmax-cmin)/1000)#对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
cs = ax.contourf(x_1,t_1 , error,levels,cmap=plt.get_cmap('Spectral'))
#添加colorbar
cbar = fig.colorbar(cs,fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))#对colorbar的大小进行设置
#设置颜色条的刻度
tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=12.5)
#设置颜色条的title
cbar.ax.set_title('error',fontsize=12.5)
cbar.update_ticks()#显示colorbar的刻度值
#设置坐标刻度及间隔
ax.set_xlim(-3.14,3.14)
ax.set_ylim(0,1)
x_major_locator=MultipleLocator(1)#刻度间隔
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
#设置坐标轴标签
font3 = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 14,
         }
ax.set_xlabel('$\t{x}$',font3)
ax.set_ylabel('$\t{t}$',font3)
#设置坐标刻度字体大小
ax.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
#设置图像像素及大小
plt.rcParams['figure.figsize']=(6.0,4.5)
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
fig.savefig('FIG9_h.eps', format='eps', dpi=300)
plt.show()


# plots PINN
##########################################


def gen_plt_x(num):
    x = np.linspace(-np.pi, np.pi, num)
    t = np.linspace(0, 1, num)
    l = []

    for i in range(len(t)):
        for j in range(len(x)):
            l.append([x[j], t[i]])
    y = PINNmodel.predict(np.array(l))
    y_true = solution(np.array(l))
    error = abs(y_true-y)
    y = np.reshape(y[:, 0], (num, num))
    error = np.reshape(error[:, 0], (num, num))
    y_true=np.reshape(y_true[:,0], (num, num))
    return y,error,y_true,x,t


y1,error,y_true,x_1,t_1 = gen_plt_x(500)

cmax=np.max(y1)
cmin=np.min(y1)
fig, ax = plt.subplots()
levels = np.arange(cmin,cmax,(cmax-cmin)/1000.0)#对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
cs = ax.contourf(x_1 , t_1, y1,levels,cmap=plt.get_cmap('Spectral'))
#添加colorbar
cbar = fig.colorbar(cs,fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))#对colorbar的大小进行设置
#设置颜色条的刻度
tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=12.5)
#设置颜色条的title
cbar.ax.set_title('u',fontsize=12.5)
cbar.update_ticks()#显示colorbar的刻度值
#设置坐标刻度及间隔
ax.set_xlim(-3.14,3.14)
ax.set_ylim(0,1)
x_major_locator=MultipleLocator(1)#刻度间隔
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
#设置坐标轴标签
font3 = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 14,
         }
ax.set_xlabel('$\t{x}$',font3)
ax.set_ylabel('$\t{t}$',font3)
#设置坐标刻度字体大小
ax.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
#设置图像像素及大小
plt.rcParams['figure.figsize']=(8.0,6)
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
fig.savefig('FIG9_d.eps', format='eps', dpi=300)
plt.show()


#进行绘图
cmax=np.max(error)
cmin=np.min(error)
fig, ax = plt.subplots()
levels = np.arange(cmin,cmax,(cmax-cmin)/1000)#对颜色渐进细致程度进行设置，其中0与z_max是色条显示的数据范围，0.005是颜色显示的细致程度
cs = ax.contourf(x_1,t_1 , error,levels,cmap=plt.get_cmap('Spectral'))
#添加colorbar
cbar = fig.colorbar(cs,fraction=0.1, pad=0.15, shrink=0.9, anchor=(0.0, 0.3))#对colorbar的大小进行设置
#设置颜色条的刻度
tick_locator = ticker.MaxNLocator(nbins=6)  # colorbar上的刻度值个数
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=12.5)
#设置颜色条的title
cbar.ax.set_title('error',fontsize=12.5)
cbar.update_ticks()#显示colorbar的刻度值
#设置坐标刻度及间隔
ax.set_xlim(-3.14,3.14)
ax.set_ylim(0,1)
x_major_locator=MultipleLocator(1)#刻度间隔
y_major_locator=MultipleLocator(0.2)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
#设置坐标轴标签
font3 = {'family' : 'Arial',
         'weight' : 'normal',
         'size'   : 14,
         }
ax.set_xlabel('$\t{x}$',font3)
ax.set_ylabel('$\t{t}$',font3)
#设置坐标刻度字体大小
ax.tick_params(labelsize=14)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Arial') for label in labels]
#设置图像像素及大小
plt.rcParams['figure.figsize']=(6.0,4.5)
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
fig.savefig('FIG9_g.eps', format='eps', dpi=300)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 08:36:08 2022

@author: saura
"""



import numpy as np
import statistics as st
import pandas as pd
import matplotlib.pyplot as plt
import pysindy as ps
from pysindy.differentiation import SpectralDerivative
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.differentiation import FiniteDifference

seg_length=11
"""first data series"""

data = pd.read_csv(r'C:/Users/saura/OneDrive - Delft University of Technology/Documents/TU Delft MSc/Additional graduation project/Sensor data/with friction/Thesis sensor data/150gm/20.csv' )
df=pd.DataFrame(data,columns=['SensorDistance0','SensorDistance2','Protocoltimestamp'])
u1s1=(df.SensorDistance0).values-st.mean((df.SensorDistance0).values)
x1s1=(df.SensorDistance2).values-st.mean((df.SensorDistance2).values)
#t1=(df.Protocoltimestamp).values
x1s1=0.001*np.array(x1s1[0:8000:1])
u1s1=0.001*np.array(u1s1[0:8000:1])

for i in range(0,len(x1s1)-2,1):
    if x1s1[i]*x1s1[i+1]<0:
        firstzero=i
        break

x1s1=np.array(x1s1[firstzero:-1:1])
u1s1=np.array(u1s1[firstzero:-1:1])

""" Windowing"""
"""
window1=np.hanning(len(x1))
x1=np.multiply(x1,window1)
u1=np.multiply(u1,window1)
"""
"""
x1=np.pad(x1,(10000,10000),'constant')
u1=np.pad(u1,(10000,10000),'constant')
"""

t1s1=0.0005*np.array(range(0,len(x1s1),1))
sd=SmoothedFiniteDifference(smoother_kws={'window_length':seg_length})
x2s1=sd._differentiate(x1s1, t1s1)
u2s1=sd._differentiate(u1s1,t1s1)
xs1_start=(np.array([x1s1[0], x2s1[0]]))
u1s1sim=u1s1
t1s1sim=t1s1


x1s1=np.transpose(np.array([x1s1[:]]))
u1s1=np.transpose(np.array([u1s1[:]]))
x2s1=np.transpose(np.array([x2s1[:]]))
u2s1=np.transpose(np.array([u2s1[:]]))

t1s1=0.0005*np.array(np.transpose(np.array(range(0,len(x1s1),1))))
xs1_train=np.concatenate((x1s1,x2s1),axis=1)

""""Second data series"""

data = pd.read_csv(r'C:/Users/saura/OneDrive - Delft University of Technology/Documents/TU Delft MSc/Additional graduation project/Sensor data/with friction/Thesis sensor data/150gm/25.csv' )
df=pd.DataFrame(data,columns=['Protocoltimestamp','SensorDistance0','SensorDistance2'])
u1s2=(df.SensorDistance0).values-st.mean((df.SensorDistance0).values)
x1s2=(df.SensorDistance2).values-st.mean((df.SensorDistance2).values)
#t2=(df.Protocoltimestamp).values
x1s2=0.001*np.array(x1s2[0:8000:1])
u1s2=0.001*np.array(u1s2[0:8000:1])

for i in range(0,len(x1s2)-2,1):
    if x1s2[i]*x1s2[i+1]<0:
        firstzero=i
        break

x1s2=np.array(x1s2[firstzero:-1:1])
u1s2=np.array(u1s2[firstzero:-1:1])

""" Windowing"""
"""
window2=np.hanning(len(x2))
x2=np.multiply(x2,window2)
u2=np.multiply(u2,window2)
"""
"""
x2s2=np.pad(x2s2,(10000,10000),'constant')
u2s2=np.pad(u2s2,(10000,10000),'constant')
"""

t1s2=0.0005*np.array(range(0,len(x1s2),1))
sd=SmoothedFiniteDifference(smoother_kws={'window_length':seg_length})
x2s2=sd._differentiate(x1s2, t1s2)
u2s2=sd._differentiate(u1s2, t1s2)

xs2_start=(np.array([x1s2[0], x2s2[0]]))

u1s2sim=u1s2
t1s2sim=t1s2


x1s2=np.transpose(np.array([x1s2[:]]))
u1s2=np.transpose(np.array([u1s2[:]]))
x2s2=np.transpose(np.array([x2s2[:]]))
u2s2=np.transpose(np.array([u2s2[:]]))

t1s2=0.0005*np.array(np.transpose(np.array(range(0,len(x1s2),1))))
xs2_train=np.concatenate((x1s2,x2s2),axis=1)

""""Third data series"""

data = pd.read_csv(r'C:/Users/saura/OneDrive - Delft University of Technology/Documents/TU Delft MSc/Additional graduation project/Sensor data/with friction/Thesis sensor data/150gm/32.csv' )
df=pd.DataFrame(data,columns=['Protocoltimestamp','SensorDistance0','SensorDistance2'])
u1s3=(df.SensorDistance0).values-st.mean((df.SensorDistance0).values)
x1s3=(df.SensorDistance2).values-st.mean((df.SensorDistance2).values)
#t3=(df.Protocoltimestamp).values
x1s3=0.001*np.array(x1s3[0:8000:1])
u1s3=0.001*np.array(u1s3[0:8000:1])

for i in range(0,len(x1s3)-2,1):
    if x1s3[i]*x1s3[i+1]<0:
        firstzero=i
        break

x1s3=np.array(x1s3[firstzero:-1:1])
u1s3=np.array(u1s3[firstzero:-1:1])


""" Windowing"""
"""
window3=np.hanning(len(x3))
x3=np.multiply(x3,window3)
u3=np.multiply(u3,window3)
"""
"""
x1s3=np.pad(x1s3,(10000,10000),'constant')
u1s3=np.pad(u1s3,(10000,10000),'constant')
"""
t1s3=0.0005*np.array(range(0,len(x1s3),1))
sd=SmoothedFiniteDifference(smoother_kws={'window_length':seg_length})
x2s3=sd._differentiate(x1s3, t1s3)
u2s3=sd._differentiate(u1s3, t1s3)

xs3_start=(np.array([x1s3[0], x2s3[0]]))
u1s3sim=u1s3
t1s3sim=t1s3

x1s3=np.transpose(np.array([x1s3[:]]))
u1s3=np.transpose(np.array([u1s3[:]]))
x2s3=np.transpose(np.array([x2s3[:]]))
u2s3=np.transpose(np.array([u2s3[:]]))

t1s3=0.0005*np.array(np.transpose(np.array(range(0,len(x1s3),1))))
xs3_train=np.concatenate((x1s3,x2s3),axis=1)

""""Fourth data series"""

data = pd.read_csv(r'C:/Users/saura/OneDrive - Delft University of Technology/Documents/TU Delft MSc/Additional graduation project/Sensor data/with friction/Thesis sensor data/150gm/38.8.csv' )
df=pd.DataFrame(data,columns=['Protocoltimestamp','SensorDistance0','SensorDistance2'])
u1s4=(df.SensorDistance0).values-st.mean((df.SensorDistance0).values)
x1s4=(df.SensorDistance2).values-st.mean((df.SensorDistance2).values)
#t=(df.Protocoltimestamp).values
x1s4=0.001*np.array(x1s4[0:8000:1])
u1s4=0.001*np.array(u1s4[0:8000:1])

for i in range(0,len(x1s4)-2,1):
    if x1s4[i]*x1s4[i+1]<0:
        firstzero=i
        break

x1s4=np.array(x1s4[firstzero:-1:1])
u1s4=np.array(u1s4[firstzero:-1:1])


""" Windowing"""
"""
window4=np.hanning(len(x4))
x4=np.multiply(x4,window4)
u4=np.multiply(u4,window4)
"""
"""
x4=np.pad(x4,(10000,10000),'constant')
u4=np.pad(u4,(10000,10000),'constant')
"""
t1s4=0.0005*np.array(range(0,len(x1s4),1))
sd=SmoothedFiniteDifference(smoother_kws={'window_length':seg_length})
x2s4=sd._differentiate(x1s4, t1s4)
u2s4=sd._differentiate(u1s4, t1s4)

xs4_start=(np.array([x1s4[0], x2s4[0]]))
u1s4sim=u1s4
t1s4sim=t1s4

x1s4=np.transpose(np.array([x1s4[:]]))
u1s4=np.transpose(np.array([u1s4[:]]))
x2s4=np.transpose(np.array([x2s4[:]]))
u2s4=np.transpose(np.array([u2s4[:]]))

t1s4=0.0005*np.array(np.transpose(np.array(range(0,len(x1s4),1))))
xs4_train=np.concatenate((x1s4,x2s4),axis=1)

"""Creating list of different data"""
x=[xs1_train,xs2_train,xs3_train,xs4_train]
"""
x=np.concatenate(x)
t=[t1,t2,t3,t4]
length=len(t1)
tinp=0.0005*np.array(range(0,len(np.concatenate(t)),1))
"""
def u_fun():
    return [np.concatenate((u1s1,u2s1),axis=1),np.concatenate((u1s2,u2s2),axis=1),np.concatenate((u1s3,u2s3),axis=1),np.concatenate((u1s4,u2s4),axis=1)]


u=u_fun()




#x=np.concatenate((x1,x2,x3,x4),axis=0)
#u=np.concatenate((u1,u2,u3,u4),axis=0)

#t=0.0005*np.array(np.transpose(np.array(range(0,len(x),1))))

#t=(np.array(t[199999:-1:1])-t[199999])
"""

X=np.stack(x,axis=-1)
"""
"""
fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
fig.suptitle('Experiment data for top storey displacement')

ax1.plot(t1, x1)
ax1.set_title("60.39 Hz")
ax1.set(xlabel='Time (s)',ylabel='Top storey disp. (m)')
ax2.plot(t4, x4)
ax2.set_title("77.50 Hz")
ax2.set(xlabel='Time (s)',ylabel='Top storey disp. (m)')
ax3.plot(t3, x3)
ax3.set_title("138.34 Hz")
ax3.set(xlabel='Time (s)',ylabel='Top storey disp. (m)')
ax4.plot(t2, x2)
ax4.set_title("157.26 Hz")
ax4.set(xlabel='Time (s)',ylabel='Top storey disp. (m)')



fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
fig.suptitle('Experiment data for top storey displacement')

ax1.plot(t1, u1)
ax1.set_title("60.39 Hz")
ax1.set(xlabel='Time (s)',ylabel='Input (m)')
ax2.plot(t4, u4)
ax2.set_title("77.50 Hz")
ax2.set(xlabel='Time (s)',ylabel='Input (m)')
ax3.plot(t3, u3)
ax3.set_title("138.34 Hz")
ax3.set(xlabel='Time (s)',ylabel='Input (m)')
ax4.plot(t2, u2)
ax4.set_title("157.26 Hz")
ax4.set(xlabel='Time (s)',ylabel='Input (m)')
"""
"""
differentiation_method=ps.differentiation.SmoothedFiniteDifference(),
feature_library=ps.feature_library.PolynomialLibrary(degree=2),
optimizer=ps.optimizers.STLSQ(threshold=0.05,max_iter=50,fit_intercept=True),


"""

"""
lib_func=[
    lambda x:x,
    lambda x:np.sign(x),
        ]



lib_func_names=[
    lambda x:x,
    lambda x:"sign("+x+")",
   
    ]
"""
"""
sindy_library = ps.WeakPDELibrary(
    library_functions=x_lib_func,
    spatiotemporal_grid=t,
    function_names=lib_func_names,
    is_uniform=True,
    K=100,
    
)
"""
"""
#Functons for sindypi library

lib_func=[
    lambda x:x,
    lambda x:np.sign(x),
        ]

dot_lib_func=[
    lambda x:x,
    ]

lib_func_names=[
    lambda x:x,
    lambda x:"sign("+x+")",
    lambda x:x,
    ]


sindy_library=ps.SINDyPILibrary(
    library_functions=lib_func,
    x_dot_library_functions=dot_lib_func,
    t=np.concatenate((t1s1,t1s2,t1s3,t1s4)),                                   #check this always
    function_names=lib_func_names,
    differentiation_method=ps.SpectralDerivative(),
    interaction_only=False
    )
"""

"""
sindy_library=ps.PDELibrary(library_functions=lib_func,  
                            function_names=lib_func_names, 
                            derivative_order=0, 
                           temporal_grid=np.concatenate((t1s1,t1s2,t1s3,t1s4)),
                            include_bias=True,
                          implicit_terms=True,
                         )

"""

#for custom library

x_lib_func=[
    lambda x:x,
#    lambda u: u,
    lambda x:np.sign(x),
#    lambda x:np.tanh(x),    
    ]

lib_func_names =[
    lambda x:x,
 #   lambda u:u,
    lambda x: 'sign('+x+')', 
 #    lambda x:'tanh('+x+')',
  ]

sindy_library = ps.CustomLibrary(
    library_functions=x_lib_func, function_names=lib_func_names
)

"""
sindy_opt=ps.SR3(threshold=0.01,
                   thresholder="l0", 
                   max_iter=10000, 
               #    normalize_columns=True, 
                   tol=1e-7,
                  # trimming_fraction=0.1
                  )
"""

sindy_opt=ps.STLSQ(threshold=0.01,max_iter=10000)

"""
sindy_opt = ps.SINDyPI(
    threshold=0.01,                 # check this value 1e-6
    tol=1e-8,
    thresholder="l1",
    max_iter=100000,
  #  normalize_columns=True
)
"""
"""
sindy_opt=ps.FROLS(kappa=10)
""" 
"""
model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
#    feature_names=['x','u'],
   differentiation_method=ps.SmoothedFiniteDifference()
    )
"""


model = ps.SINDy(
    optimizer=sindy_opt,
    feature_library=sindy_library,
    feature_names=['x1','x2','u1','u2'],
   differentiation_method=ps.SpectralDerivative(),
    )

dt=0.0005
"""
The OG working
model.fit(x, t=dt, u=u,multiple_trajectories=True)
model.print()
"""
#for new trial
model.fit(x,t=dt,u=u,multiple_trajectories=True)
model.print()
"""
simoriginal=model.simulate(xs1_start,t=t1s1sim,u=u1s1sim)
plt.figure()
plt.plot(t1s1[0:-1],simoriginal[:,0],"k--",label="SINDy model",linewidth=3)
plt.xlabel("t")
plt.ylabel("x")
plt.plot(t1s1,x1s1,"b",label="experimental data")
plt.legend()
plt.show()
"""
"""
diffu= model.differentiate(u1,t1)
plt.figure()
plt.plot(t1,diffu)
"""
"""
simoriginal=model.simulate([x1_start],t=t1,u=u1sim)
plt.figure()
plt.plot(t1[0:-1],simoriginal,"k--",label="SINDy model",linewidth=3)
plt.xlabel("t")
plt.ylabel("x")
plt.plot(t1,x1,"b",label="data")


datatest = pd.read_csv(r'C:/Users/saura/OneDrive - Delft University of Technology/Documents/TU Delft MSc/Additional graduation project/Sensor data/with friction/sample.xlsx' )
dftest=pd.DataFrame(datatest,columns=['Protocoltimestamp','SensorDistance0','SensorDistance2',])
utest=(dftest.SensorDistance0).values-st.mean((dftest.SensorDistance0).values)
xtest=(dftest.SensorDistance2).values-st.mean((dftest.SensorDistance2).values)
ttest=(dftest.Protocoltimestamp).values

xtest=0.001*np.array(xtest[199999:-1:1])
utest=0.001*np.array(utest[199999:-1:1])
ttest=(np.array(ttest[199999:-1:1])-ttest[199999])

x0_test=xtest[0]
u0_test=utest[0]
sim=model.simulate([x0_test],t=ttest,u=utest)

plt.figure()
plt.plot(ttest[0:-1],sim,"k--",label="SINDy model",linewidth=3)
plt.xlabel("t")
plt.ylabel("x")
plt.plot(ttest,xtest,"b",label="experimental data")
plt.legend()
plt.show()
"""
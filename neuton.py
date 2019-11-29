# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:32:02 2019

@author: Stalker
"""

import sympy as sm
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D 
import numpy as np



def Rx(q):
    r = sm.Matrix([[1 ,     0     ,    0     ],
                   [0 ,  sm.cos(q),-sm.sin(q)],
                   [0 ,  sm.sin(q), sm.cos(q)]])
    return r
def Ry(q):
    r = sm.Matrix([[sm.cos(q), -sm.sin(q), 0],
                   [sm.sin(q),  sm.cos(q), 0],
                   [      0  ,     0     , 1]])
    return r
def Rz(q):
    r = sm.Matrix([[sm.cos(q) , 0, sm.sin(q)],
                   [    0     , 1,      0   ],
                   [-sm.sin(q), 0, sm.cos(q)]])
    return r



m1 = 1
m2 = 0.5
m3 = 0.5
m = [m1,m2,m3]
a1 = 1
a2 = 0.5
A1 = 1
A2 = 1
A3 = 1
t = sm.symbols('t')
q1 = A1*sm.sin(t)
q2 = A2*sm.cos(2*t)
q3 = A3*sm.sin(3*t)


R01 = Rz(q1)*Rx(sm.pi/2)
R12 = Rz(q2)*Ry(sm.pi/2)*Rz(sm.pi/2)
R23 = sm.Matrix([[1,0,0],
                 [0,1,0],
                 [0,0,1]])
R = [R01] + [R12] + [R23]
I1 = sm.Matrix([[m[0]*a1**2/12,0,0],
                [0,m[0]*a1**2/12,0],
                [0,       0     ,0]])
I2 = sm.Matrix([[m[1]*a2**2/12,0,0],
                [0,m[1]*a2**2/12,0],
                [0,       0     ,1]])
I3 = sm.Matrix([[m[2]*(q3)**2/12,        0         , 0],
                [0                 ,m[2]*(q3)**2/12, 0], 
                [0                 ,       0       , 0]])
I = [I1,I2,I3] 
q = sm.Matrix([[q1],[q2],[q3]])
dq = sm.diff(q)
ddq = sm.diff(dq)
c1 = a1/2
c2 = a2/2
c3 = (q3)/2
r01 = sm.Matrix([[0],[0],[a1]])
r12 = sm.Matrix([[a2],[0],[0]])
r23 = sm.Matrix([[0],[0],[q3]])
rc1 = sm.Matrix([[0],[0],[c1]])
rc2 = sm.Matrix([[c2],[0],[0]])
rc3 = sm.Matrix([[0],[0],[c3]])
r = [r01] + [r12] +[r23]
rc = [rc1] +[rc2] +[rc3]
w0 = sm.Matrix([[0],[0],[dq[0,0]]])
z0 = sm.Matrix([[0],[0],[1]])
a0 = sm.Matrix([[0],[0],[0]])
ac0 = sm.Matrix([[0],[0],[0]])
v0 = sm.Matrix([[0],[0],[0]])
vc0 = sm.Matrix([[0],[0],[0]])
g0 = sm.Matrix([[0],[0],[-9.81]])
alpa0 = sm.diff(w0)
w = [w0]
z = [z0,z0,z0]
a = [a0]
ac = [ac0]
v = [v0]
vc = [vc0]
g = [R01.T*g0]
alpa = [alpa0]
akor = 0 
acen = 0
for i in range(1, 3):
    if i != 2:
        wi = R[i].T*(w[i-1] + dq[i,0]*z[i-1])
        w.append(wi)
        vi = R[i].T*v[i-1] + wi.cross(r[i]) 
        v.append(vi)
        vci = R[i].T*vc[i-1] + wi.cross(rc[i])
        vc.append(vci)
        dw = sm.diff(wi)
        alpa.append(dw)
        ai = R[i].T*a[i-1]+ dw.cross(r[i])+wi.cross(wi.cross(r[i]))
        a.append(ai)
        acc = ai + dw.cross(rc[i])+ wi.cross(wi.cross(rc[i]))
        ac.append(acc)
        gi = R[i].T*g[i-1]
        g.append(gi)
    else:
        wi = R[i].T*w[i-1]
        w.append(wi)
        vi = R[i].T*v[i-1] + dq[2,0]*z[i]+ wi.cross(r[i])
        v.append(vi)
        vci = R[i].T*vc[i-1] + wi.cross(rc[i])
        vc.append(vci)
        dw = sm.diff(wi)
        alpa.append(dw)
        acor = 2*dq[2,0]*wi.cross(z[i-1])
        ai = R[i].T*a[i-1] + ddq[2,0]*z[i-1] +2*dq[2,0]*wi.cross(z[i-1]) +  dw.cross(r[i]) + wi.cross(wi.cross(r[i]))
        a.append(ai)
        acc = ai + ddq[2,0]*z[i-1] +2*dq[2,0]*wi.cross(z[i]) + dw.cross(rc[i])+ wi.cross(wi.cross(rc[i]))
        acen = wi.cross(wi.cross(rc[i]))
        ac.append(acc)
        gi = R[i].T*g[i-1]
        g.append(gi)

fd = sm.Matrix([[0],[0],[0]])
tord = sm.Matrix([[0],[0],[0]])
f = [fd]   
tor = [tord]
i = 2     
while i > -1:
    fi = R[i]*f[0] + m[i]*ac[i] - m[i]*g[i]
    f = [fi] + f
    tori = R[i]*tor[0] - fi.cross(rc[i]) + (R[i]*f[0]).cross(-rc[i]) + alpa[i] + w[i].cross(I[i]*w[i])
    tor = tor + [tori]
    i -= 1


tm = np.linspace( 0, 4*np.pi, 1000)
def massiv(f, tm):
    fn = np.zeros(tm.shape[0], dtype="float32")
    for i in range(tm.shape[0]):
        fn[i] = f.subs(t, tm[i])
    return fn

w3x = massiv(w[2][0],tm) 
w3y = massiv(w[2][1],tm)
w3z = massiv(w[2][2],tm)
v3x = massiv(vc[2][0], tm)
v3y = massiv(vc[2][1], tm)
v3z = massiv(vc[2][2], tm)
acx = massiv(ac[2][0],tm)
acy = massiv(ac[2][1],tm)
acz = massiv(ac[2][2],tm)
fcor = m3*acor
fcorx = massiv(fcor[0],tm)
fcory = massiv(fcor[1],tm)
fcorz = massiv(fcor[2],tm)
fcen = m3*acen
fcenx = massiv(fcen[0],tm)
fceny = massiv(fcen[1],tm)
fcenz = massiv(fcen[2],tm)
fg = g[2]
fgx = massiv(fg[0],tm)
fgy = massiv(fg[1],tm)
fgz = massiv(fg[2],tm)



fig = plt.figure(1)
plt.title("Angular velocity along x for q3")
plt.scatter(tm, w3x, s = 5)
plt.xlabel("t, sec")
plt.ylabel("w3x, rad/sec")
plt.show()


fig = plt.figure(2)
plt.title("Angular velocity along y for q3")
plt.scatter(tm, w3y, s = 5)
plt.xlabel("t, sec")
plt.ylabel("w3x, rad/sec")
plt.show()

fig = plt.figure(3)
plt.title("Angular velocity along z for q3")
plt.scatter(tm, w3z, s = 5)
plt.xlabel("t, sec")
plt.ylabel("w3z, rad/sec")
plt.show()

fig = plt.figure(4)
plt.title("Linear velocity along x for q3")
plt.scatter(tm, v3x, s = 5)
plt.xlabel("t")
plt.ylabel("v3z, m/sec")
plt.show()


fig = plt.figure(5)
plt.title("Linear velocity along y for q3")
plt.scatter(tm, v3y, s = 5)
plt.xlabel("t, sec")
plt.ylabel("v3y, m/sec")
plt.show()


fig = plt.figure(6)
plt.title("Linear velocity along z for q3")
plt.scatter(tm, v3z, s = 5)
plt.xlabel("t, sec")
plt.ylabel("v3z, m/sec")
plt.show()


fig = plt.figure(7)
plt.title("Accerleration along x for q3")
plt.scatter(tm, acx, s = 5)
plt.xlabel("t, sec")
plt.ylabel("ac3x, m/sec^2")
plt.show()


fig = plt.figure(8)
plt.title("Accerleration along y for q3")
plt.scatter(tm, acy, s = 5)
plt.xlabel("t, sec")
plt.ylabel("ac3y, m/sec^2")
plt.show()

fig = plt.figure(9)
plt.title("Accerleration along z for q3")
plt.scatter(tm, acz, s = 5)
plt.xlabel("t, sec")
plt.ylabel("ac3z, m/sec^2")
plt.show()

fig = plt.figure(10)
plt.title("Coriolis force along x for q3")
plt.scatter(tm, fcorx, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fcorx, N")
plt.show()


fig = plt.figure(11)
plt.title("Coriolis force along y for q3")
plt.scatter(tm, fcory, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fcory, N")
plt.show()

fig = plt.figure(12)
plt.title("Coriolis force along z for q3")
plt.scatter(tm, fcorz, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fcorz, N")
plt.show()

fig = plt.figure(13)
plt.title("Centrifugal forc along x for q3")
plt.scatter(tm, fcenx, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fcenx, N")
plt.show()


fig = plt.figure(14)
plt.title(" Centrifugal forc along y for q3")
plt.scatter(tm, fceny, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fceny, N")
plt.show()

fig = plt.figure(15)
plt.title("Centrifugal forc along z for q3")
plt.scatter(tm, fcenz, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fcenz, N")
plt.show()


fig = plt.figure(16)
plt.title("Gravity force along x for q3")
plt.scatter(tm, fgx, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fgx, N")
plt.show()


fig = plt.figure(17)
plt.title("Gravity force along y for q3")
plt.scatter(tm, fgy, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fgy, N")
plt.show()

fig = plt.figure(18)
plt.title("Gravity force along z for q3")
plt.scatter(tm, fgz, s = 5)
plt.xlabel("t, sec")
plt.ylabel("fgz, N")
plt.show()



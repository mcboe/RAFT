#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:16:48 2021

@author: erbc925n
"""
# =============================================================================
# =============================================================================
#                          HUANG & EATOCK TAYLOR SCRIPT
#  SECOND-ORDER SUM-FREQUENCY FORCE AND MOMENT CALCULATION FOR TRUNCATED  
#  CYLINDER IN FINITE DEPTH BASED ON:
#     Huang & Eatock Taylor 1996, "Semi-analytical solution for second-order
#  wave diffraction by a truncated cylinder in monochromatic waves"
# 
#  PRODUCES DIAGONAL QTFs FOR A SPECIFIED RANGE OF PULSATION
# 
#   Required inputs:
#   - Radius
#   - Draft                  
#   - Depth                   
#   - Range of pulsations or frequencies
# 
#   !!! Note !!! : Convention used in paper:
#    - Phase conv: e^(iwt) --> here e^(-iwt)
#    - Moment calculated at MWL
# 
# 
# Author: Elie RONGE:           elie-externe.ronge@edf.fr 
# Contact: Chirstophe PEYRARD:  christophe.peyrard@edf.fr
# =============================================================================
# =============================================================================
# Preamble
from timeit import default_timer as timer
import copy
import scipy as sp
import os
import numpy as np
from scipy import special as spe
from scipy.optimize import newton_krylov
from scipy.optimize import fsolve
from scipy import integrate
import matplotlib.pyplot as plt
import sys
import logging

start = timer()

tan = np.tan
cos = np.cos
sin = np.sin
cosh = np.cosh
sinh = np.sinh
tanh = np.tanh
exp = np.exp

h1v = spe.hankel1
h1vp = spe.h1vp
jv =spe.jv
jvp = spe.jvp
kv = spe.kv
kvp = spe.kvp
iv = spe.iv
ivp = spe.ivp

pi = np.pi

  
# =============================================================================
# INPUT DATA
# =============================================================================
# Environment parameters
g = 9.81 #                  gravity acceleration    [m.s-2]
rho = 1025 #                sea water density       [kg.m-3]


# Case parameters

a = 9#                     cylinder radius         [m]
b = 45 #                      #cylinder draft         [m]
h = 120 #                   water depth             [m]
d = h-b                     #water depth below keel  [m]

freq_min = 0.02 #           minimum frequency       [Hz]
freq_max = 0.5 #           maximum frequency       [Hz]

# # v_min = 0.5/a
# # v_max = 3.0/a

# k_min = 0.25/a
# k_max = 2.75/a

# om_min = np.sqrt(k_min*g*tanh(k_min*h)) # minimum pulsation       [rad.s-1]
# om_max = np.sqrt(k_max*g*tanh(k_max*h))  # maximum pulsation       [rad.s-1]

om_min = freq_min*2*pi # minimum pulsation       [rad.s-1]
om_max = freq_max*2*pi  # maximum pulsation       [rad.s-1]

nfreq = 3 #                Number of frequency in range
omega = np.linspace(om_min, om_max, nfreq) #        [rad.s-1]



# Convergence parameters:
quad_conv = 0.01  # convergence on sum for quadratic force
k2m_conv = 0.01   # convergence on evanescent modes fourier sum
q_conv = 0.1      # convergence on sum for q1.5 term
fs_conv = 0.
5     # convergence on truncated free surface integration


# Plot logical - Activation not recommended for large number of frequencies due to number of plots generated (2 per frequency)

plotfreesurface  =  False #If activated, will plot non-homogoneous diffracted potential function over truncated free surface for each frequency - assign figure number in fs_figno = 
checkevmodedecay = False # If activated, will plot evanescent modes terms over truncated free surface for each frequency - assign figure number in evmodes_figno

# Ouput location and output files
#path = '/home/erbc925n/CALHYPSO/Run_Local/Tyrion_Etude_ER/TP_Study_3rd/'
#resultfile = 'tptyrion0.03-0.5hz_output2.txt'  # This is name of result file
#logfile = 'tptyrion0.03-0.5hz_output2message.txt' # This is name of log file


# Console log
#os.chdir(path)
#if os.path.isfile(logfile):
#    os.remove(logfile) #This deletes existing log file before run
# Set-up of logfile format
#targets = logging.StreamHandler(sys.stdout), logging.FileHandler(logfile)
#logging.basicConfig(format='%(message)s',level=logging.INFO, handlers = targets)

# =============================================================================
#  FUNCTIONS
# =============================================================================

# Wave length Function - Gets  wavenumber and wavelength from frequency and water 
# depth based on linear wave dispertion equation
def wavelength(freq,depth): 
    h = depth        
    pi = np.pi
    f = lambda k: (2*pi*freq)**2 - g*k*np.tanh(k*h)
    x = np.linspace(0, 2*pi/2,100)
    k = sp.optimize.newton_krylov(f,x)
    L = 2*pi/k
    
    k = np.median(k)
    L = np.median(L)
    
    return k,L

# Function finds wave number for pulsation of twice the given frequency 
def findk2(omega, h):  
    f = lambda k : (2*omega)**2 - g*k*np.tanh(k*h)    
    k2 = newton_krylov(f,0.4)
        
    return k2
 
# Find evanecent modes wave numbers 
def findk2m(omega, h, n1, n2, plotprint=False): 
    # THIS ROUTINE FINDS TEH SOLUTION
    f = lambda k : (2*omega)**2 + g*k*np.tan(k*h)
    n = n1
    k2m = []
    logging.info('start - discretsising function with step: 0.5\u03c0/h*1/100')
    for n in range(n1, n2):
        
        try:
            dk = 0.5*np.pi/h/100
        
            k_min = (n-0.5)*np.pi/h
            k_max = (n)*np.pi/h
            k_arr = np.arange(k_min+dk, k_max,dk)
            k_r = fsolve(f, k_arr)
            k_r.sort()
            mask = k_r <= (k_max)
            mask = mask & (k_r >=  k_min)
            k_r = k_r [mask]
            k_r = np.round(k_r, decimals=5)
            k_r = np.unique(k_r)
            k_r2 = newton_krylov(f, k_r)
            mask = k_r2 <= (k_max)
            mask = mask & (k_r2 >=  k_min)
            k_r2 = k_r2[mask]
            k_r2.sort()
            k_r2 = np.round(k_r2, decimals=5)
            k_r = float(np.unique(k_r2))
            k2m.append(k_r)

        except:
            logging.info('newton_krylov_failed to converge - discretsising function with step/4')
            dk = dk/4

            try: 
                
                k_min = (n-0.5)*np.pi/h
                k_max = (n)*np.pi/h
                k_arr = np.arange(k_min+dk, k_max,dk)
                k_r = fsolve(f, k_arr)
                k_r.sort()
                mask = k_r <= (k_max)
                mask = mask & (k_r >=  k_min)
                k_r = k_r [mask]
                k_r = np.round(k_r, decimals=5)
                k_r = np.unique(k_r)
                k_r2 = newton_krylov(f, k_r)
                mask = k_r2 <= (k_max)
                mask = mask & (k_r2 >=  k_min)
                k_r2 = k_r2[mask]
                k_r2.sort()
                k_r2 = np.round(k_r2, decimals=5)
                k_r = float(np.unique(k_r2))
                k2m.append(k_r)

            except:
                logging.info('newton_krylov_failed to converge twice - function discrete step/16')
                dk = dk/4

                try: 
                    k_min = (n-0.5)*np.pi/h
                    k_max = (n)*np.pi/h
                    k_arr = np.arange(k_min+dk, k_max,dk)
                    k_r = fsolve(f, k_arr)
                    k_r.sort()
                    mask = k_r <= (k_max)
                    mask = mask & (k_r >=  k_min)
                    k_r = k_r [mask]
                    k_r = np.round(k_r, decimals=5)
                    k_r = np.unique(k_r)
                    k_r2 = newton_krylov(f, k_r)
                    mask = k_r2 <= (k_max)
                    mask = mask & (k_r2 >=  k_min)
                    k_r2 = k_r2[mask]
                    k_r2.sort()
                    k_r2 = np.round(k_r2, decimals=5)
                    k_r = float(np.unique(k_r2))
                    k2m.append(k_r)
                 
                except:
                    logging.info('newton_krylov_failed to converge - approximation made based on equal spacing of roots')
                   
                    k2m_approx = k2m[-1]+ (k2m[n-2]-k2m[n-3])
                    k2m.append(k2m_approx)

    
    k2m = np.array(k2m)
    if plotprint:
        k_arr = np.arange((0.5)*np.pi/h, n2*np.pi/h, dk)  
        plt.figure(), plt.plot(k_arr, f(k_arr))
        plt.plot(k2m, f(np.array(k2m)), 'rx')
        plt.ylim([-500,500]), plt.title('k2m root finding'), plt.xlabel('f(k2m)')
        plt.ylabel('k2m')
    
    
    return np.array(k2m)  

# Huang&Taylor function: calculates the second-order sum-frequency force using Huang&Taylor 1996 approach
def HuangTaylor(omega,a,h, evmodes_figno = 1, fs_figno = 2):
    freq = omega/(np.pi*2)
    f = freq
    om = omega
    v = om**2/g
    k = wavelength(f, h)[0]
    L = wavelength(f, h)[1]
    
    infostring = [' =============================================================================\n',\
          'HUANG & EATOCK TAYLOR ANALYTICAL SOLUTION FOR:\n', \
          'a = %.3f m, b = %.3f m, h = %.3f m, omega = %.3f rad/s\n'%(a,b,h,omega), \
          'k = %.3f m-1, L = %.3f m, ka = %.3f\n'%(k, L, k*a), \
          ' =============================================================================\n']
    logging.info(''.join(infostring))
    
    # Quadratic of 1st order potential
    
    n = 0
    sig_fx = []
    sig_my = []
    fm_21_conv = 1

    # =============================================================================
    # Convergence of quadratic force
    # =============================================================================
    # logging.info('= = = = = = = = = =\nQuadratic convergence check:\n= = = = = = = = = =')
    
    # while fm_21_conv >= quad_conv:
        
    #     # Appendix C
    #     # Equation C4
    #     fx_s = (-1)**n/(h1vp(n, k*a, n=1)*h1vp(n+1, k*a, n=1))*(2-2*k*b/sinh(2*k*h)\
    #          +(sinh(2*k*h)-sinh(2*k*d))/sinh(2*k*h)+n*(n+1)/(k*a)**2*\
    #           ((sinh(2*k*h)-sinh(2*k*d))/sinh(2*k*h)+2*k*b/np.sinh(2*k*h)))
            
    #     # Equations C4 and C5
    #     z_kh = 1/4 + (2*k*b*sinh(2*k*h)-cosh(2*k*h)+cosh(2*k*d))/(8*(k*b)**2)
    #     my_s = (-1)**n/(h1vp(n, k*a, n=1)*h1vp(n+1, k*a, n=1))*\
    #          (1 + 2*k*b/np.sinh(2*k*h)*((n*(n+1)/(k*a)**2+1)*z_kh - 1/2))
        
    #     # Appends each term of Fourier sum
    #     sig_fx.append([fx_s])
    #     sig_my.append([my_s])
        
        
    #     # Convergence check
    #     if n > 0:
        
    #         conv_real1 = (np.sum(np.real(sig_fx))-np.sum(np.real(sig_fx[:-1])))/np.sum(np.real(sig_fx[:-1]))
    #         conv_real2 = (np.sum(np.real(sig_my))-np.sum(np.real(sig_my[:-1])))/np.sum(np.real(sig_my[:-1]))
            
    #         conv_imag1 = (np.sum(np.imag(sig_fx))-np.sum(np.imag(sig_fx[:-1])))/np.sum(np.imag(sig_fx[:-1]))
    #         conv_imag2 = (np.sum(np.imag(sig_my))-np.sum(np.imag(sig_my[:-1])))/np.sum(np.imag(sig_my[:-1]))
            
    #         fm_21_conv = np.max([np.abs(conv_real1), np.abs(conv_imag1), np.abs(conv_real2), np.abs(conv_imag2)])
        
        
    #     logging.info('Quadratic iteration n = %.i -  convergence: %.3e'%(n, fm_21_conv))
    #     n += 1             

    # Fx_21 = 2*1j/(np.pi*(k*a)**2)  * np.sum(sig_fx)
    # My_21 = 4*1j/(np.pi*(k*a)**2)  * np.sum(sig_my)*b/h
    
    # # =============================================================================
    # # #  SECOND ORDER POTENTIAL
    # # =============================================================================
    # # Incident Potential
    
    # #  Incident potential Eq.2.28 - force and moment Eq.3.19 and Eq.3.20
    # Fx_2I = lambda z: -3/2*1j*np.pi*k*tanh(k*h)*cosh(2*k*(z+h))*spe.j1(2*k*a)/((np.sinh(k*h))**4)
    # My_2I = lambda z: -3/2*1j*np.pi*k*tanh(k*h)*cosh(2*k*(z+h))*spe.j1(2*k*a)/((np.sinh(k*h))**4)*(b+z)
    
    # # Integration over cylinder length
    # real_part = integrate.quad(lambda x: np.real(Fx_2I(x)), -b, 0)[0]
    # imag_part = integrate.quad(lambda x: np.imag(Fx_2I(x)), -b, 0)[0]
    # Fx_2I = real_part + 1j * imag_part

    # real_partM = integrate.quad(lambda x: np.real(My_2I(x)), -b, 0)[0]/h
    # imag_partM = integrate.quad(lambda x: np.imag(My_2I(x)), -b, 0)[0]/h
    # My_2I = real_part + 1j * imag_part
    # #Fx_2I = integrate.quad(Fx_2I, -b, 0)[0]
    # #My_2I = integrate.quad(My_2I, -b, 0)[0]/h
    
    # # =============================================================================
    # # Evanescent modes (k2m) convergence check
    # # =============================================================================
    # # Diffracted potential
    k2 = findk2(om, h)
    
    
    logging.info('= = = = = = = = = =\nEvanecent modes convergence check:\n= = = = = = = = = =')    
    conv_val = 1
    k2m_start = 1
    k2m_no = 20
    conv_val = 1
    iconv = 0
    k2m_l = []
    k2m_conv2 = copy.deepcopy(k2m_conv)
    
    # Calculates evanescent modes based on convergence of homogoneous diffracted potential
    while conv_val >= k2m_conv2:
        logging.info('k2m iteration: %.i - convergence%.3e'%(iconv, conv_val))
        iconv +=1
        # Extends array of evanescent modes
        k2m_l.extend(findk2m(om, h, k2m_start, k2m_no))
        k2m = np.array(k2m_l)
        l_k2m = len(k2m)
        k2m_start = k2m_no
        k2m_no += 10
        
    #     # Homogoneous diffracted potential - Eq. 2.36-Eq.2.37
    #     alph_m = lambda km: (2*k*sinh(2*k*h)-4*omega**2/g*cosh(2*k*h))/(4*k**2+km**2)
        
    #     #  Calculate list of each force/moment evanescent mode terms of 
    #     #  homogoneous diffracted potential Eq.2.36 - Eq.2.37
    #     #  !!! the integration over cylinder length and factor in Eq.3.19-Eq.3.20 are already applied
    #     fx2h_ev = [3*pi*1j*k**2*tanh(k*h)*jvp(1,2*k*a)/sinh(k*h)**4*alph_m(k2m[m])* kv(1,k2m[m]*a)/kvp(1,k2m[m]*a)\
    #                *4*k2m[m]/(2*k2m[m]*h+sin(2*k2m[m]*h))*cos(k2m[m]*h)/k2m[m]**2*(sin(k2m[m]*h)-sin(k2m[m]*d)) for m in range(l_k2m)]
    #     my2h_ev = [3*pi*1j*k**2*tanh(k*h)*jvp(1,2*k*a)/sinh(k*h)**4*alph_m(k2m[m])* kv(1,k2m[m]*a)/kvp(1,k2m[m]*a)\
    #                *4*k2m[m]/(2*k2m[m]*h+sin(2*k2m[m]*h))*cos(k2m[m]*h)/k2m[m]* \
    #                  (b*sin(k2m[m]*h)/k2m[m]-cos(k2m[m]*d)/k2m[m]**2+cos(k2m[m]*h)/k2m[m]**2) for m in range(l_k2m)]
        
    #     # List of the converging sum of the evanescent modes
    #     fx2h_s = []
    #     my2h_s = []

    #     fx2h_val = 0
    #     my2h_val = 0
        
    #     # psi1_val = 0
    #     # psi5_val = 0
    
    #     for i in range(len(k2m)-1):
            
    #         fx2h_val += fx2h_ev[i]
    #         my2h_val += my2h_ev[i]
            
    #         # This keeps a record of sum of evanescent modes for convergence
    #         fx2h_s.append(fx2h_val)
    #         my2h_s.append(my2h_val)
 
        
    #     fx2h_s = np.array(fx2h_s)
    #     my2h_s = np.array(my2h_s)

    #     # Convergence check
    #     fxh_ev_conv = np.abs((fx2h_s[1:]-fx2h_s[:-1])/fx2h_s[:-1])
    #     myh_ev_conv = np.abs((my2h_s[1:]-my2h_s[:-1])/my2h_s[:-1])
        
    #     # psi1_ev_conv = np.abs((psi1_s[1:]-psi1_s[:-1])/psi1_s[:-1])
    #     # psi5_ev_conv = np.abs((psi5_s[1:]-psi5_s[:-1])/psi5_s[:-1])
        
    #     conv_val = np.max([fxh_ev_conv[-1], myh_ev_conv[-1]])
        
    #     # Criterias for decreasing convergence requirement after amount of iteration
    #     if iconv == 5:
    #         k2m_conv2 = 0.001
    #     elif iconv == 10:
    #         k2m_conv2 = 0.002
        
    #     if iconv >= 35:
    #         logging.info('max iteration reached - convergence error is:\n%.5f'%conv_val)
    #         break
    
    # infostring = ['Convergence ends:\n','Fx2H evanecent modes conv.: %.4e\n'%(fxh_ev_conv[-1]),\
    #        'm = %.i'%(len(k2m))]
    # logging.info(''.join(infostring))
    # if conv_val >= 0.01:
    #     logging.info('Convergence error on fourier sum - code stops - revise parameters')

  
    
  
    # # Calculates propagating mode of homogoneous potential based on Eq.2.36 - 2.37
    # # !!! the integration over cylinder length and factor in Eq.3.19-Eq.3.20 are already applied
    # alph_0 = (2*k*sinh(2*k*h)-4*omega**2/g*cosh(2*k*h))/(4*k**2-k2**2)

    # gam_0 = h/2*(1+sinh(2*k2*h)/(2*k2*h))
    
    # Z_0 = lambda z: 1/np.sqrt(gam_0)*cosh(k2*(z+h))
    
    # Fx2h_0 = 3*pi*1j*k**2*tanh(k*h)*jvp(1,2*k*a)/sinh(k*h)**4*alph_0* h1v(1,k2*a)/h1vp(1,k2*a)\
    #                    *4*k2/(2*k2*h+sinh(2*k2*h))*cosh(k2*h)/k2**2*(sinh(k2*h)-sinh(k2*d))
    # My2h_0 = 3*pi*1j*k**2*tanh(k*h)*jvp(1,2*k*a)/sinh(k*h)**4*alph_0* h1v(1,k2*a)/h1vp(1,k2*a)\
    #                    *4*k2/(2*k2*h+sinh(2*k2*h))*cosh(k2*h)/k2*\
    #                     (b*sinh(k2*h)/k2+cosh(k2*d)/k2**2-cosh(k2*h)/k2**2)
    
    #Fx_2H = Fx2h_0+fx2h_s[-1]
    #My_2H = My2h_0/h+my2h_s[-1]/h
    # My_2H term is non-dimensionalised with a factor of 1/h, this is 
    # different to paper where 1/b is used
   
    
   
    
    
    # =============================================================================
    # Preparation of terms of non-homogoneous diffracted potential
    # =============================================================================
    logging.info('= = = = = = = = = = \nNon-homogoneous diffracted potential convergence check on q1 term:\n= = = = = = = = = =')   
    
    
    
    #  Functions
    gam_0 = h/2*(1+sinh(2*k2*h)/(2*k2*h))
   
    Z_0 = lambda z: 1/np.sqrt(gam_0)*cosh(k2*(z+h))
    gam = lambda km: h/2*(1+sin(2*km*h)/(2*km*h))
    Z_m = lambda z, km: 1/np.sqrt(gam(km))*cos(km*(z+h))
    
    
    e_n = lambda n: 1 if n == 0 else 2
    
    
    S_n = lambda n,r:   (1j)*g/(omega)*jvp(n, k*a)/h1vp(n,k*a) * h1v(n,k*r)*exp(1j*n*pi/2)
    S_ndr = lambda n,r: (1j)*g/(omega)*jvp(n, k*a)/h1vp(n,k*a) * k*h1vp(n,k*r)*exp(1j*n*pi/2)
    T_n = lambda n,r: -(1j)*g/(omega)*(1j)**n*jv(n, k*r)
    T_ndr = lambda n,r: -(1j)*g/(omega)*(1j)**n*k*jvp(n, k*r)
    
    f_l1 = []
    f_l2 = [] 
    f_l3 = []
    f_l4 = [] 
    fs1 = [0]
    fs2 = [0]
    fs3 = [0]
    fs4 = [0]
    qr_l = []
    infval = 3
    qstep = 1
    q_range = list(range(-infval,infval+1))
    nmax = 7
    iconv2 = 0
    q_conv = 0.001
    conv_q1 = 1
    while conv_q1 >= q_conv:
        logging.info('convergence iteration - step %.i:'%iconv2)
        qr_l.append(q_range)
        fm = lambda r, n: e_n(n)*np.sum([1j*omega/(2*g)*k**2*(3*tanh(k*h)**2-1)*(S_n(n-q,r)*S_n(q,r)+2*T_n(n-q,r)*S_n(q, r)) \
                       + 1j*omega/g*(S_ndr(n-q,r)*S_ndr(q, r)-q*(n-q)/r**2*S_n(n-q,r)*S_n(q,r) \
                                     + 2*(T_ndr(n-q, r)*S_ndr(q, r) -q*(n-q)/r**2*T_n(n-q,r)*S_n(q,r))) for q in q_range])
        f_l1.append(fm(a, 1))
        f_l2.append(fm(30*a, 1))
        f_l3.append(fm(a, nmax))
        f_l4.append(fm(30*a, nmax))
        fs1.append(np.sum(f_l1))
        fs2.append(np.sum(f_l2))
        fs3.append(np.sum(f_l3))
        fs4.append(np.sum(f_l4))
        conv_q1 = np.max([np.abs((fs1[-1] - fs1[-2])/fs1[-2]),np.abs((fs2[-1] - fs2[-2])/fs2[-2]),np.abs((fs3[-1] - fs3[-2])/fs3[-2]),np.abs((fs4[-1] - fs4[-2])/fs4[-2])])
        q_range = list(range(-infval-qstep, -infval))+list(range(infval+1, infval+qstep+1))
        infval += qstep
        if iconv2 > 100:
            logging.info('max iteration reached - convergence error is:\n%.5f'%conv_q1)
            break
        iconv2 += 1
    
    q_range = list(range(-infval, infval+1))
    
    logging.info('Convergence on q1 sum truncation ends - q infval = %.i, convergence:%.3e'%(infval, conv_q1))
      
        
    
    # =============================================================================
    # Check decay of evanecent modes
    # =============================================================================
    
    # if checkevmodedecay:
    
    #     psi_ev = lambda r: (np.sum([lamb_1m[i]*np.cos(-1j*k2m[i]*h)/(-1j*k2m[i])*spe.kv(1, -1j*k2m[i]*r) \
    #                                    /spe.kvp(1, -1j*k2m[i]*a, n=1) for i in range(len(k2m))]))
        
    #     q11_ev = lambda r: np.sum([(-1)**n*((gam_n(n,r)*gam_n(n+1,r) - spe.jv(n, k*r)*spe.jv(n+1, k*r)) *  \
    #                                       (n*(n+1)/(k*r)**2 -1/2 + 3/2* (np.tanh(k*h))**2)) for n in range(nmax)])
        
    #     kr_l = np.arange(1*k*a, 100*k*a, 0.02)
    #     q = []
    #     psi= []
    #     for i, kr in enumerate(kr_l):
    
    #         q.append(q11_ev(kr/k))   
    #         psi.append(psi_ev(kr/k)) 
            
    #     plt.figure(evmodes_figno)
    #     plt.subplot(1,2,1), plt.xlabel('ka'), plt.ylabel('real amplitude')
    #     plt.plot(kr_l, np.real(psi), kr_l, np.real(q)), plt.legend(['Psi_ev','q_ev'])
    #     plt.subplot(1,2,2), plt.xlabel('ka'), plt.ylabel('imag amplitude')
    #     plt.plot(kr_l, np.imag(psi), kr_l, np.imag(q)), plt.legend(['Psi_ev','q_ev'])
    #     plt.tight_layout()
    #     plt.suptitle('q1-psi1 evanescent modes decay')
    # =============================================================================
    # Free surface loop
    # =============================================================================
    logging.info('= = = = = = = = = = \nDiffracted potential free surface integral:\n= = = = = = = = = =')   
     
    m_l = len(k2m)
    
    q_n = lambda r, n: e_n(n)*np.sum([1j*omega/(2*g)*k**2*(3*tanh(k*h)**2-1)*(S_n(n-q,r)*S_n(q,r)+2*T_n(n-q,r)*S_n(q, r)) \
                       + 1j*omega/g*(S_ndr(n-q,r)*S_ndr(q, r)-q*(n-q)/r**2*S_n(n-q,r)*S_n(q,r) \
                                     + 2*(T_ndr(n-q, r)*S_ndr(q, r) -q*(n-q)/r**2*T_n(n-q,r)*S_n(q,r))) for q in q_range])
    
    
    # phi_p1 = lambda r, n: r*q_n(r,n)*((1j*pi/2*(jv(n,k2*a)-jvp(n, k2*a)/h1vp(n, k2*a)*h1v(n, k2*a)) \
    #                             *h1v(n, k2*r)*Z_0(0)*integrate.quadrature(Z_0, -b, 0)[0]) \
    #                    + (np.sum((iv(n,k2m*a)-ivp(n, k2m*a)/kvp(n, k2m*a)*kv(n, k2m*a)) \
    #                             *kv(n, k2m*r)*Z_m(0, k2m)\
    #                             *np.array([integrate.quadrature(Z_m, -b, 0, args=k2m[m])[0] for m in range(m_l)]))))
    Z_0m = lambda z: 1/np.sqrt(gam_0)*cosh(k2*(z+h))*(b+z)
    Z_mm = lambda z, km: 1/np.sqrt(gam(km))*cos(km*(z+h))*(b+z)
    phi_pm = lambda r, n: r*q_n(r,n)*((1j*pi/2*(jv(n,k2*a)-jvp(n, k2*a)/h1vp(n, k2*a)*h1v(n, k2*a)) \
                                *h1v(n, k2*r)*Z_0(0)*integrate.quadrature(Z_0m, -b, 0)[0]) \
                        + (np.sum((iv(n,k2m*a)-ivp(n, k2m*a)/kvp(n, k2m*a)*kv(n, k2m*a)) \
                                *kv(n, k2m*r)*Z_m(0, k2m)\
                                *np.array([integrate.quadrature(Z_mm, -b, 0, args=k2m[m])[0] for m in range(m_l)]))))
    
    # phi_p1 = lambda r, n: r*q_n(r,n)*((1j*pi/2*(jv(n,k2*a)-jvp(n, k2*a)/h1vp(n, k2*a)*h1v(n, k2*a)) \
    #                             *h1v(n, k2*r)*Z_0(0)*integrate.quadrature(Z_0, -b, 0)[0]) \
    #                    + (np.sum((iv(n,k2m*a)-ivp(n, k2m*a)/kvp(n, k2m*a)*kv(n, k2m*a)) \
    #                             *kv(n, k2m*r)*Z_m(0,k2m)*np.array([integrate.quadrature(Z_m, -b, 0, args = k2m[m])[0] for m in range(m_l)]))))   
        
    phi_p1 = lambda r, n: r*q_n(r,n)*((1j*pi/2*(jv(n,k2*a)-jvp(n, k2*a)/h1vp(n, k2*a)*h1v(n, k2*a)) \
                                *h1v(n, k2*r)*4*k2/(2*k2*h+sinh(2*k2*h))*cosh(k2*h)\
                                     *(sinh(k2*h)/k2 - sinh(k2*d)/k2)) \
                       + (np.sum((iv(n,k2m*a)-ivp(n, k2m*a)/kvp(n, k2m*a)*kv(n, k2m*a)) \
                                *kv(n, k2m*r)*4*k2m/(2*k2m*h+sin(2*k2m*h))*cos(k2m*h)\
                                *(sin(k2m*h)/k2m - sin(k2m*d)/k2m))))       
    
    phi_p1m = lambda r, n: r*q_n(r,n)*((1j*pi/2*(jv(n,k2*a)-jvp(n, k2*a)/h1vp(n, k2*a)*h1v(n, k2*a)) \
                                *h1v(n, k2*r)*4*k2/(2*k2*h+sinh(2*k2*h))*cosh(k2*h)\
                                *(b*sinh(k2*h)/k2 + cosh(k2*d)/k2**2 -cosh(k2*h)/k2**2) \
                       + (np.sum((iv(n,k2m*a)-ivp(n, k2m*a)/kvp(n, k2m*a)*kv(n, k2m*a)) \
                                *kv(n, k2m*r)*4*k2m/(2*k2m*h+sin(2*k2m*h))*cos(k2m*h)\
                                *(b*sin(k2m*h)/k2m - cos(k2m*d)/k2m**2  + cos(k2m*h)/k2m**2)))))   
    
    def fs_int_def(nka_min, nka_max):
        # Discretisation step for function
        dkr = min(0.01,   0.01*k*a)
        kr_l = np.arange(nka_min*a*k, nka_max*a*k, dkr)
        f = []
        m = []
        
        
        # This calculates the function [f_2p,m_2p] = k*r*q1*[psi1,psi5]
        for i, kr in enumerate(kr_l):
            # This print radial distance in form kr for multiple of 10 of ka
            logging.info('Loop over free surface, ka value:%.3f'%kr)  if (kr_l[i] %(10*k*a)) <= (kr_l[i-1]%(10*k*a)) else 0
                
                
            f.append(phi_p1(kr/k,1))
            m.append(phi_p1m(kr/k, 1))
            
        
        f = np.array(f)*2*pi*1j*om/g
        m = np.array(m)*2*pi*1j*om/g
        
        real1 = np.real(f)
        imag1 = np.imag(f)*1j
        real2 = np.real(m)
        imag2 = np.imag(m)*1j
        
        if plotfreesurface:
            plt.figure(fs_figno)
            plt.subplot(1,2,1), plt.plot(kr_l/(k*a), f), plt.xlabel('r/a'), plt.ylabel('Re(Fpx)')
            plt.subplot(1,2,2), plt.plot(kr_l/(k*a), f), plt.xlabel('r/a'), plt.ylabel('Im(Fpx)')
            plt.tight_layout()
            plt.suptitle('Free surface function: k.r.q.psi')
        
        # This integrates the function over the free surfqce section nka_min - nka_max
        Fx_r = integrate.simpson(imag1, dx = dkr/k) + integrate.simpson(real1, dx = dkr/k)
        My_r = integrate.simpson(imag2, dx = dkr/k) + integrate.simpson(real2, dx = dkr/k)
        return Fx_r, My_r

    start_kr = 1
    trunc_1 = 10 


    Fx_2P, My_2P = fs_int_def(start_kr, trunc_1)
    
    conv = 1
    iconv2 = 0
    # Step of truncation loop over free surface
    Dkr = 10 # Scalar to k.a
    
    fx2p_intsum = [Fx_2P]
    my2p_intsum = [My_2P]
    
    # Loop over free surface to satisfy convergence parameters set above
    while conv > fs_conv:

        mean_f1 = np.mean(fx2p_intsum)
        mean_m1 = np.mean(my2p_intsum)
        
        Fx_2P_dk, My_2P_dk = fs_int_def(trunc_1, trunc_1+Dkr)
        
        trunc_1 = trunc_1+Dkr
        Fx_2P += Fx_2P_dk
        My_2P += My_2P_dk
        
        fx2p_intsum.append(Fx_2P)
        my2p_intsum.append(My_2P)
        
        mean_f2 = np.mean(fx2p_intsum)
        mean_m2 = np.mean(my2p_intsum)
        
        conv = np.max([np.abs(np.real((mean_f2-mean_f1)/mean_f1)), np.imag((mean_f2-mean_f1)/mean_f1),np.abs(np.real((mean_m2-mean_m1)/mean_m1)), np.imag((mean_m2-mean_m1)/mean_m1)])
        logging.info('convergence at: %.6f'%conv)
        iconv2 +=  1 
        if iconv2 > 150:
            logging.info('Maximum iteration reached, min convergence:%.3f'%conv)
            break
    Fx_2P = - mean_f2
    My_2P = - mean_m2/h
   
    infostring = ['= = = = = = = = = = \nConvergence on Free Surface reached - convergence:%.3e\n'%conv, \
          'Convergence summary:\n', \
          'Quadratic iteration n = %.i -  convergence: %.3e\n'%(n, fm_21_conv) ,\
          # 'Convergence ends:\n','Fx2H evanecent modes conv.: %.4e\n Psi evanescent modes conv. conv: %.4e\n'%(fxh_ev_conv[-1], psi1_ev_conv[-1]), \
          # 'My2H evanecent modes conv.: %.4e\n Psi5 evanescent modes conv. conv: %.4e\n'%(myh_ev_conv[-1], psi5_ev_conv[-1]), \
          'Convergence on q1 sum truncation ends - convergence:%.3e\n'%conv_q1, \
          '= = = = = = = = = = \n']
    logging.info(''.join(infostring))
    
    Fx_2 = Fx_2P
    My_2 = My_2P
    Fx_2_tot = Fx_2 
    My_2_tot = My_2
                                                                                                                
    return [om, Fx_2_tot, My_2_tot, Fx_2, My_2, Fx_2P, My_2P]



# =============================================================================
#  LOOP THROUGH PULSATION RANGE
# =============================================================================
ht_mat = []
for i, om in enumerate(omega):
    ht_mat.append(HuangTaylor(om, a, h, evmodes_figno=i*2+1,fs_figno=i*2+2))   
ht_mat = np.array(ht_mat)


# =============================================================================
# # RECORD RESULTS IN NPY ARRAY FORMAT
# =============================================================================

#np.save(resultfile.replace('.txt', ''), ht_mat)

# =============================================================================
# # RECORD RESULTS IN TXT FILE
# =============================================================================
# ht_header = "Variables= 'Pulsation[rad/s]', 'Fx2_tot', 'My2_tot', 'Fx2', 'My2','Fx2I', 'My2I', 'Fx2H', 'My2H', 'Fx2P', 'My2P', 'Fx21', 'My21'\n"

# with open(resultfile, 'w') as fl:
#     lines = []
#     lines.append(ht_header)
#     for row in ht_mat:
#         line = '  '
#         for elem in row:
#             line += ' %.6e+%.6ej  '%(np.real(elem), np.imag(elem)) if elem >0 else '%.6e+%.6ej  '%(np.real(elem), np.imag(elem))
#         line += '\n'
#         lines.append(line)
#     fl.writelines(lines)


# =============================================================================
# WRITE SUM-FREQUENCY QTFs FOR CALHYPSO INPUT
# =============================================================================

# CHANGE INDEX FOR PARTIAL QTFS e.g. ky_mat[:,2] and ky_mat[:,3] for 2nd-order potential load only

Fx2tot = ht_mat[:,1]*rho*a*g
My2tot = ht_mat[:,2]*rho*a*g*h


qtf_re_dof = [np.zeros([len(omega), len(omega)]) for i in range(6)]
qtf_im_dof = [np.zeros([len(omega), len(omega)]) for i in range(6)]


np.fill_diagonal(qtf_re_dof[0], np.real(Fx2tot)); np.fill_diagonal(qtf_im_dof[0], np.imag(Fx2tot))
np.fill_diagonal(qtf_re_dof[4], np.real(My2tot)); np.fill_diagonal(qtf_im_dof[4], np.imag(My2tot))


#  Imag part
lines = []
head = ['QTF+  Imag part FX\n', '----------------------------\n','QTF+  Imag part FY\n','----------------------------\n', \
        'QTF+  Imag part FZ\n', '----------------------------\n','QTF+  Imag part MX\n', '----------------------------\n', \
        'QTF+  Imag part MY\n', '----------------------------\n','QTF+  Imag part MZ\n', '----------------------------\n']
for i in range(6):
    lines.append(head[i*2])
    lines.append(head[i*2+1])
    for j, row in enumerate(qtf_im_dof[i]):
        line = '%.6e' %omega[j]
        for i, elem in enumerate(row):
            if elem > 0:
                line = line + ' '
            line = line + ' %.7e' % elem
        line = line + '\n'
        lines.append(line)

with open('QTFPI.dat', 'w') as fl:
    fl.writelines(lines)

# Real part
lines = []
head = ['QTF+  Real part FX\n', '----------------------------\n','QTF+  Real part FY\n','----------------------------\n', \
        'QTF+  Real part FZ\n', '----------------------------\n','QTF+  Real part MX\n', '----------------------------\n', \
        'QTF+  Real part MY\n', '----------------------------\n','QTF+  Real part MZ\n', '----------------------------\n']
for i in range(6):
    lines.append(head[i*2])
    lines.append(head[i*2+1])
    for j, row in enumerate(qtf_re_dof[i]):
        line = '%.6e' %omega[j]
        for elem in row:
            if elem > 0:
                line = line + ' '
            line = line + ' %.7e' % elem
        line = line + '\n'
        lines.append(line)
#os.chdir(path)
with open('QTFPR.dat', 'w') as fl:
    fl.writelines(lines)


end = timer()
time = end-start

logging.info('=============================================================================\nSCRIPT FINISHED - RUNTIME: %i s'%time)




logging.shutdown()


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib
matplotlib.use('Agg') #don't want a display
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
import subprocess
import io
import base64
import numpy as np
#np.set_printoptions(threshold=np.nan)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from qutip import *
import os
import traceback
from qutip import loky_pmap
import subprocess
from IPython.display import Image, display
from PIL import Image
from scipy import integrate
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
plt.close('all')


# Model parameters
omega_D = 0.5
omega_D_backup = omega_D
kappa=0.01 # Decay rate 0.01
temp=0.0 # k_b * T
Omega_hamiltonian=0.05
U_hamiltonian=0.05
n_th_a=0.0 #If temp=0 need this line to avoid div. by 0
#n_th_a = 1/(np.exp(1/temp)-1)  # temperature with average of x excitations 
N = 10         # fock state basis
a  = destroy(N) # annihilation operator
U_values = np.linspace(0,U_hamiltonian,2)
Omega_values = np.linspace(0,Omega_hamiltonian,2) 

# collapse operators
c_op_list = []
c_op_list.append(np.sqrt(kappa * (1 + n_th_a)) * a*a)  # decay operators
c_op_list.append(np.sqrt(kappa*n_th_a) * a.dag()*a.dag())  # excitation operators

# initial state
disp = 2.0
D1 = displace(N, disp)
D2 = displace(N, -disp)
psi = (1/np.sqrt(2))*(D1 + D2) * basis(N, 0)

#rho0 = coherent_dm(N,disp)  # start with a coherent state 
#rho0 = fock(N,2) #or fock state
rho0 = psi * psi.dag()  # start with a superposition of two coherent states

# setting the time
maxtime = 500 # Specified in periods of the driver 
tres = 50 # Specified in terms of the drive period
tlist = np.linspace(0, maxtime*(2*np.pi), tres*maxtime+1) # maxtime and t resolution 
driven_tlist = np.linspace(0, maxtime*(2*np.pi/omega_D), tres*maxtime+1) 
plotinterval=1 # Plot once every drive period 

# Set up plotting
s_res = 400
s_res += 1
range_l = -4.50
range_h = 4.50 # may need to be floats
xvecS = np.linspace(range_l,range_h,s_res)
xvec = np.linspace(range_l,range_h,s_res) 
pvec = np.linspace(range_l,range_h,s_res) # Need to be symmetrical

def plot_setup_contour():
    plt.figure(figsize=(15, 15))
    fig=plt.gcf()
    axes = plt.gca()  
    return fig, axes


# Hamiltonians
H_0 = a.dag() * a + .5                                             
H_D = (1-omega_D)*(a.dag() * a + .5) + (Omega_hamiltonian/2)*(a*a + a.dag()*a.dag()) 
H_U = (U_hamiltonian/2) * a.dag() * a.dag() * a * a   
H_D_U = H_D + H_U


# Lindblad evolutions
result = mesolve(H_0, rho0, tlist, c_op_list, [], progress_bar=True)
result_U = mesolve(H_0+H_U, rho0, tlist, c_op_list, [], progress_bar=True)
result_Omega = mesolve(H_D, rho0, driven_tlist, c_op_list, [], progress_bar=True)
result_Omega_U = mesolve(H_D_U, rho0, driven_tlist, c_op_list, [], progress_bar=True)


# This value determines how densly the Wigner current vector field is plotted
factor=int(round((s_res-1)/40))


def compute_wigner(r, dir):
    for U in U_values:
        for Omega in Omega_values:
            fig, axes = plot_setup_contour()  
            plt.clf()
            staten=r-tres*maxtime-2

            if Omega == 0:
                omega_D = 0
            else:
                omega_D = omega_D_backup
            
            s_resn = int(round((s_res-1)/factor))

            if U == 0.000 and Omega == 0.000:
                W = wigner(result.states[staten], xvec, xvec)
            if U == 0.05 and Omega == 0.000:
                W = wigner(result_U.states[staten], xvec, xvec)
            if U == 0.000 and Omega == 0.092:
                W = wigner(result_Omega.states[staten], xvec, xvec)
            if U == 0.05 and Omega == 0.092:
                W = wigner(result_Omega_U.states[staten], xvec, xvec)
                

                
            WP = np.zeros((s_res, s_res))
            WX = np.zeros((s_res, s_res))
            WPX = np.zeros((s_res, s_res))
            W2P = np.zeros((s_res, s_res))
            W2X = np.zeros((s_res, s_res))
            W3P = np.zeros((s_res, s_res))
            W3X = np.zeros((s_res, s_res))
            
            for i in range(1,s_res-1,factor):  # Implicitly enforces Dirichlet boundary condition    
                    for j in range(1,s_res-1,factor):
                        W3P[i,j] = (W.T[i,j+2] - 2*W.T[i,j+1] + 2*W.T[i,j-1] - W.T[i,j-2]) / (2*((range_h-range_l)/(s_res-1))**3)
                        W3X[i,j] = (W.T[i+2,j] - 2*W.T[i+1,j] + 2*W.T[i-1,j] - W.T[i-2,j]) / (2*((range_h-range_l)/(s_res-1))**3)
                        W2P[i,j] = (W.T[i,j+1] - 2*W.T[i,j] + W.T[i,j-1])/(((range_h-range_l)/(s_res-1))**2) # These are finite difference methods for calc. derivatives
                        W2X[i,j] = (W.T[i+1,j] - 2*W.T[i,j] + W.T[i-1,j])/(((range_h-range_l)/(s_res-1))**2)
                        WP[i,j] = (W.T[i,j+1] - W.T[i,j])/((range_h-range_l)/(s_res-1)) 
                        WX[i,j] = (W.T[i+1,j] - W.T[i,j])/((range_h-range_l)/(s_res-1))
                        WPX[i, j] = (W.T[i+1,j+1] - W.T[i+1,j-1] - W.T[i-1,j+1] + W.T[i-1,j-1]) / (4*((range_h-range_l) / (s_res-1))**2)

        
            J_sys_xhat = np.zeros((s_res,s_res))
            J_sys_phat = np.zeros((s_res,s_res))
            J_env_xhat = np.zeros((s_res,s_res))
            J_env_phat = np.zeros((s_res,s_res))
            J_xhat = np.zeros((s_res,s_res))
            J_phat = np.zeros((s_res,s_res))
            J_damp_xhat = np.zeros((s_res,s_res))
            J_damp_phat = np.zeros((s_res,s_res))
            J_diff_xhat = np.zeros((s_res,s_res))
            J_diff_phat = np.zeros((s_res,s_res))


            for i in range(1,s_res-1,factor):  # row, column indexing is easier in python 
                for j in range(1,s_res-1,factor):
                    J_sys_xhat[i,j] = ((1-omega_D-Omega-U)*pvec[j] + ((U*pvec[j]**3)/(2)) + ((U*pvec[j]*xvec[i]**2)/(2)))*W.T[i,j] - ((U*pvec[j])/(8))*W2X[i,j] - ((U*pvec[j])/(8))*W2P[i,j]
                    J_sys_phat[i,j] = ((omega_D+U-Omega-1)*xvec[i] - ((U*xvec[i]**3)/(2)) - ((U*xvec[i]*pvec[j]**2)/(2)))*W.T[i,j] + ((U*xvec[i])/(8))*W2P[i,j] - ((U*xvec[i])/(8))*W2X[i,j]

                    J_damp_xhat[i,j] = -kappa/2 * ( (xvec[i]*pvec[j]**2 + xvec[i]**3)*W.T[i,j] + (pvec[j]**2 + xvec[i]**2 - (1/4))*WX[i,j] + (xvec[i]/4)*W2X[i,j] + (pvec[j]/4)*WPX[i,j])
                    J_damp_phat[i,j] = -kappa/2 * ( (pvec[j]*xvec[i]**2 + pvec[j]**3)*W.T[i,j] + (xvec[i]**2 + pvec[j]**2 - (1/4))*WP[i,j] + (pvec[j]/4)*W2P[i,j] + (xvec[i]/4)*WPX[i,j])
                    
                    J_diff_xhat[i,j] = - kappa*n_th_a * ((xvec[i]**2 + pvec[j]**2)*WX[i,j])
                    J_diff_phat[i,j] = - kappa*n_th_a * ((xvec[i]**2 + pvec[j]**2)*WP[i,j])                                        

                    J_env_xhat[i,j] = J_damp_xhat[i,j] + J_diff_xhat[i,j]
                    J_env_phat[i,j] = J_damp_phat[i,j] + J_diff_phat[i,j]
        
            J_xhat=J_sys_xhat+J_env_xhat
            J_phat=J_sys_phat+J_env_phat

            p = Path(dir)
            p.mkdir(exist_ok=True, parents=True)
            namef = f"coherent_2D_U{U}_Omega{Omega}_{str(r).zfill(5)}.npz"
            np.savez(p.joinpath(namef), 
                    J_xhat=J_xhat, 
                    J_phat=J_phat,
                    W = W,
                    Omega = Omega,
                    U = U,
                    r=r,
                    tau = ((r-1)/tres),
                    omega_D = omega_D)

    pass
    

            
    


def makePlots(polku):
    # Polku: path to result directory
    for p in Path(polku).iterdir():
        tulokset = np.load(p)
        J_xhat = tulokset["J_xhat"]
        J_phat = tulokset["J_phat"]
        W = tulokset["W"]
        U = tulokset["U"]
        Omega = tulokset["Omega"]
        tau = tulokset["tau"]
        r = tulokset["r"]

        fig, axes = plot_setup_contour()  
        plt.clf()

        # Grid for the Wigner current
        X,Y = np.meshgrid(xvec[1:s_res-1:factor],xvec[1:s_res-1:factor])

        # Delete 0 length arrows
        xt = J_xhat[1:s_res:factor,1:s_res:factor].T
        pt = J_phat[1:s_res:factor,1:s_res:factor].T

        # Logical values for Wigner current plotting
        bx = np.logical_or(xt > 2E-3, xt < -2E-3) #for full
        bp = np.logical_or(pt > 2E-3, pt < -2E-3) #for full
        bxp=np.logical_or(bx,bp)

        # Plotting
        supp=plt.pcolor(xvec,xvec,W,vmin=-0.36, vmax=0.36,cmap=plt.cm.bwr_r) #range is determined by W
        supp2=plt.quiver(X[bxp],Y[bxp],xt[bxp],pt[bxp],angles='xy',scale=2) #for full second number is 2
        qk=plt.quiverkey(supp2, -4.1, 4.1, 0.1, "0.1",coordinates='data',fontproperties={'weight': 'bold','size':22}) #for full


        fig = plt.gcf()
        axes = plt.gca() 
        axes.set_xlim([-4.5,4.5])
        axes.set_ylim([-4.5,4.5])
        axes.add_patch(
            Rectangle(
                (3.25, -4.30),   # (x,y)
                1,          # width
                1,          # height
                fill=False, color='k', linestyle='solid', linewidth=3     ))
        axes.text(2.9, -3.4, '$\hbar$', style='oblique', color='k', fontsize=22)
        axes.text(-4.3, -4.3,r'$\tau$ = %d' %tau, style='oblique', color='k', fontsize=22)
        axes.text(-4.3, -4.0, r'$\Omega = %.2f$' % Omega, style='oblique', color='k', fontsize=22)
        axes.text(-4.3, -3.7, r'$U = %.2f$' % U, style='oblique', color='k', fontsize=22)
        axes.grid(False) # Display grid
        namef = f"coherent_2D_U{U}_Omega{Omega}_{str(r).zfill(5)}.png"
        p = Path(polku).joinpath(namef)
        fig.savefig(p, bbox_inches="tight")
        plt.close('all')


#import os
#from PIL import Image

# def combine_images_to_grid(U_values, Omega_values, grid_name):
#     # Extract `r` from the grid_name (e.g., "combined_grid_00001.png")
#     r_value = grid_name.split('_')[-1].split('.')[0]  # Extracts "00001" from "combined_grid_00001.png"
    
#     # Sort U and Omega to ensure proper order
#     U_values = sorted(U_values)  # Ensure U increases bottom-to-top
#     Omega_values = sorted(Omega_values)  # Ensure Omega increases left-to-right

#     images = []
#     for U in reversed(U_values):  # Reverse U to go from bottom-to-top
#         row_images = []
#         for Omega in Omega_values:  # Omega increases left-to-right
#             # Construct the filename based on `U`, `Omega`, and `r`
#             filename = f"coherent_2D_U{U}_Omega{Omega}_{str(r).zfill(5)}.png"
#             print(f"Looking for file: {filename}")  # Debugging
#             if os.path.exists(filename):
#                 row_images.append(Image.open(filename))
#             else:
#                 print(f"File not found: {filename}")  # Debugging for missing files
#         if row_images:
#             # Combine the row of images horizontally
#             row_image = concat_images_horizontally(row_images)
#             images.append(row_image)
    
#     # Combine all rows vertically
#     if images:
#         final_image = concat_images_vertically(images)
#         final_image.save(grid_name)
#         print(f"Combined grid saved as '{grid_name}'.")
#     else:
#         print(f"No images found for grid {grid_name}.")



# def concat_images_horizontally(images):
#     widths, heights = zip(*(img.size for img in images))
#     total_width = sum(widths)
#     max_height = max(heights)
#     new_image = Image.new("RGBA", (total_width, max_height))
#     x_offset = 0
#     for img in images:
#         new_image.paste(img, (x_offset, 0))
#         x_offset += img.width
#     return new_image

# def concat_images_vertically(images):
#     widths, heights = zip(*(img.size for img in images))
#     max_width = max(widths)
#     total_height = sum(heights)
#     new_image = Image.new("RGBA", (max_width, total_height))
#     y_offset = 0
#     for img in images:
#         new_image.paste(img, (0, y_offset))
#         y_offset += img.height
#     return new_image



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--folder", help="output directory", required=True)
    parsed = parser.parse_args()

    #computetimes = range(1,tres*maxtime+2,plotinterval)
    computetimes = range(1,5,plotinterval)
    results =  loky_pmap(partial(compute_wigner, dir=parsed.folder), computetimes, progress_bar=True)    

    # Plotting starts here
    makePlots(parsed.folder)
    


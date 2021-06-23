#!/usr/bin/python
import numpy as np
#from numpy import linalg
import math, sys, os, time
'''
2019 - 04 - 05.
DFT code by Viktor Cikojevi\'c. In case you find any bugs please report them on one of my e-mails :-) 
cikojevic.viktor@gmail.com
viktor.cikojevic@upc.edu
'''
print("SIM. started at ", time.ctime())
start_time = time.time()

r_0 = 387.672168  # https://www.wolframalpha.com/input/?i=161.9+atomic+mass+unit+*+mu_0+*+%289.93+bohr+magneton%29%5E2+%2F+%284+pi+hbar%5E2%29+in+bohr+radius 
mu_m = 1.E-06 / (r_0 * 0.529E-10)

a_11 = 60 / r_0
a_22 = 60 / r_0
a_12 = 60 / r_0 #variable

g_11 = 4*np.pi*a_11
g_22 = 4*np.pi*a_22
g_12 = 4*np.pi*a_12 #mass is the same


nparticles = 2000
np1, np2 = 2000    , 1.E-05
m2 = 1.
# [ 10.70953852  21.41907704   0.           0.         220.00181649  21.34811108]
# [ 10.70953853  21.41907706   0.           0.         220.00540222  21.35340471]
# 
nxyz = np.ones(3) * 32 #if you want 1D or 2D, just change this array
nxyz = np.array(nxyz, dtype=np.int32)
L = np.array([3, 3, 26]) * mu_m #if you want 1D or 2D, just change this array. This means between -L/2 and L/2

init_rho = 0.2* mu_m
init_z =   3  * mu_m

L_ok = L * 0.8 # np.array([10000., 6000., 6000.])

t_equil = 1000
t_prod  = 0. #np.inf for infinite      #total propagation time. Count from the real (imaginary) timestep if you propagate in real (imaginary) time
deltat_prod    = 0.1 # real timestep
deltat_equil = 1 # this is abs value of imaginary part


printToStdoutEvery = 20 # (int)(t_equil/deltat_equil / 100)
printDenEvery      = 20 #  (int)(t_equil/deltat_equil / 100)


kill_fac = 0.95

def init_kill_matrix(x, y, z):
	matrix = 1.
	if(N_DIM == 1):
		matrix = np.logical_or(np.abs(x) > L_ok[0]/2.)
	if(N_DIM == 2):
		matrix = np.logical_or(np.abs(x) > L_ok[0]/2., np.abs(y) > L_ok[1]/2.)
	if(N_DIM == 3):
		matrix = np.logical_or(np.abs(x) > L_ok[0]/2., np.abs(y) > L_ok[1]/2.)
		matrix = np.logical_or(matrix,   np.abs(z) > L_ok[2]/2.)
	return np.invert(matrix) + matrix*kill_fac # this matrix is 1 inside box, and kill_fac outside

def potential_external(x, y, z):
	return 0. # 0.5 * (x**2/a_ho[0]**4 + y**2/a_ho[1]**4 + z**2/a_ho[2]**4)
	

roll_dist = 0

def init_psi_1(x, y, z):
	
	return np.exp(-0.5 * 0.25 * (30)**2 * ((x/L[0])**2 + (y/L[1])**2 + (z/L[2])**2) ) + 0.j
	#return np.roll(np.exp(-0.5  * ((x/init_rho)**2 + (y/init_rho)**2 + (z/init_z)**2) ) + 0.j, roll_dist, axis=2)
	
def init_psi_2(x, y, z):
	return np.exp(-0.5 * 0.25 * (30)**2 * ((x/L[0])**2 + (y/L[1])**2 + (z/L[2])**2) ) + 0.j
	#return np.roll(np.exp(-0.5  * ((x/init_rho)**2 + (y/init_rho)**2 + (z/init_z)**2) ) + 0.j, -roll_dist, axis=2)
	



################################################################################
################################################################################
################################################################################
################################################################################


#exit()

imProp = False
#setting up auxiliary variables
N_DIM = len(nxyz)
LHalf = L / 2
x_lin = np.linspace(-LHalf[0], LHalf[0], num = nxyz[0], endpoint=False)
y_lin = np.linspace(-LHalf[1], LHalf[1], num = nxyz[1], endpoint=False)
z_lin = np.linspace(-LHalf[2], LHalf[2], num = nxyz[2], endpoint=False)

dx = L/nxyz
d3r = np.prod(dx)

kx = np.fft.fftfreq(nxyz[0], dx[0]/( 2. * np.pi))
ky = np.fft.fftfreq(nxyz[1], dx[1]/( 2. * np.pi))
kz = np.fft.fftfreq(nxyz[2], dx[2]/( 2. * np.pi))

d3k = (kx[1] - kx[0]) * (ky[1] - ky[0]) * (kz[1] - kz[0])
x, y, z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')


pot_ext = potential_external(x, y, z)



### global variables
lhy_energy, mu_a, mu_b = 1, 1, 1
sumk2_1 = (kx**2 + ky**2 + kz**2)/2
sumk2_2 = (kx**2 + ky**2 + kz**2)/2/m2
kinprop_1, kinprop_2 = 1., 1.
dt_equil = -1.j*deltat_equil
dt_prod = deltat_prod
kill_matrix = init_kill_matrix(x, y, z)
psi_1, psi_2 = 1, 1
ft_dip = 1.
###
print(" *** Parameters of the simulation ***")
print("nxyz ", nxyz)
print("L ", L)
print("d3r " , d3r)
print("d3k", d3k )
x2 = x**2
z2 = z**2

kvec = np.sqrt(kx**2 + ky**2 + kz**2)
ft_dip = 4*np.pi/3  * (2*kz**2 - kx**2 - ky**2) / kvec**2
#krho = np.sqrt(kx**2 + ky**2)
#ft_dip += 4*np.pi/3 * np.exp(-zcut*krho) * (krho**2/kvec**2 * np.cos(kz*zcut) - kz*krho/kvec**2 * np.sin(kz*zcut)) # sin(theta)=rho/r, cos(theta)=z/r
#ft_dip += 4*np.pi/3 * zcut / np.sqrt(zcut**2 + rcut**2)
ft_dip  = np.nan_to_num(ft_dip, posinf=0)

print("Loading interpolating pickles ... ")
import pickle
with open('interpolator_lhy_energy.pkl', 'rb') as f:
    lhy_energy = pickle.load(f)
with open('interpolator_mu_a.pkl', 'rb') as f:
    mu_a = pickle.load(f)
with open('interpolator_mu_b.pkl', 'rb') as f:
    mu_b = pickle.load(f)
print("pickles sucessfully loaded!")
#Trotter operator, O(dt^2) global error


def get_phis(den1, den2):
	phi_1 = np.fft.ifftn(ft_dip * np.fft.fftn(den1))
	phi_2 = np.fft.ifftn(ft_dip * np.fft.fftn(den2))
	return phi_1, phi_2
	#p = phi_1 + phi_2
	#return p, p

r_0 = 387.672168 
sclen = 60. / r_0 # units of a0
alpha = 2 * np.pi * sclen
g = 4*np.pi*sclen
a_dd = 1./3 
eps_dd = a_dd / sclen
beta  = 32*g*sclen**1.5 /(3 * np.sqrt(np.pi)) * (1+1.5*eps_dd**2) * 2./5
gamma = 3./2

def energy_density_interaction(n1, n2, phi_1, phi_2):
	en_mf  = 0.5*g_11*n1**2 + 0.5*g_22*n2**2 + g_12*n1*n2 + 0.j
	en_mf += (0.5*phi_1 + phi_2)*n1 + (0.5*phi_2)*n2
	en_lhy = lhy_energy(n1, n2, grid=False)
	return np.array([d3r * np.sum(en_mf.real), d3r * np.sum(en_lhy)])


def dEps_dPsi(na, nb, phi_a, phi_b):
	lmu_a = g_11*na + g_12*nb  + phi_a  + phi_b
	lmu_b = g_22*nb + g_12*na  + phi_b  + phi_a


	lmu_a += mu_a(na, nb, grid=False)
	lmu_b += mu_b(na, nb, grid=False)
	return lmu_a, lmu_b

def chemical_potential(psi_1, psi_2):
	den_1 = np.abs(psi_1) ** 2
	den_2 = np.abs(psi_2) ** 2
	phi_1, phi_2 = get_phis(den_1, den_2)
	pot_1, pot_2 = dEps_dPsi(den_1, den_2, phi_1, phi_2)
	c1 = np.sum(pot_1 * den_1 * d3r) / np1 
	c2 = np.sum(pot_2 * den_2 * d3r) / np2
	return np.array([c1.real, c2.real])


def T2_operator(psi_1, psi_2, den_1, den_2, phi_1, phi_2):
	global dt, pot_ext, imProp
	#exp(-1/2 * i dt * V)  
	pot_1, pot_2 = dEps_dPsi(den_1, den_2, phi_1, phi_2)
	psi_1 *= np.exp(-0.5j * pot_1 * dt)
	psi_2 *= np.exp(-0.5j * pot_2 * dt)
	
	psi_1 = np.fft.fftn(psi_1)
	psi_2 = np.fft.fftn(psi_2)
	psi_1 *= kinprop_1
	psi_2 *= kinprop_2
	psi_1 = np.fft.ifftn(psi_1)
	psi_2 = np.fft.ifftn(psi_2)

	den_1 = np.abs(psi_1) ** 2
	den_2 = np.abs(psi_2) ** 2
	phi_1, phi_2 = get_phis(den_1, den_2)
	pot_1, pot_2 = dEps_dPsi(den_1, den_2, phi_1, phi_2)
	psi_1 *= np.exp(-0.5j * pot_1 * dt)
	psi_2 *= np.exp(-0.5j * pot_2 * dt)
	
	if(imProp): #normalize		
		psi_1 *= np.sqrt(np1)/np.sqrt(d3r * np.sum(np.abs(psi_1) ** 2))
		psi_2 *= np.sqrt(np2)/np.sqrt(d3r * np.sum(np.abs(psi_2) ** 2))
	
	return psi_1, psi_2
	


def energy(psi_1, psi_2, den_1, den_2, phi_1, phi_2):
	p_ext = np.array([d3r * np.sum(den_1 * pot_ext), d3r * np.sum(den_2 * pot_ext)])
	p_int = energy_density_interaction(den_1, den_2, phi_1, phi_2)
	k_en = np.array([ d3r * np.sum((np.conj(psi_1) * np.fft.ifftn(np.fft.fftn(psi_1) * sumk2_1)).real), 
					  d3r * np.sum((np.conj(psi_2) * np.fft.ifftn(np.fft.fftn(psi_2) * sumk2_2)).real)])
	return np.concatenate((k_en, p_ext, p_int), axis=None)

output_dir = 'snapshots_time_evolution_0'; num=0
while os.path.exists(output_dir): num+=1; output_dir="snapshots_time_evolution_"+str(num)
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(output_dir+'/npy_files'): os.makedirs(output_dir+'/npy_files')
file_en_equil = open(output_dir + '/en_equil.dat', 'w', buffering=1)
file_en_prod = open(output_dir + '/en_prod.dat', 'w', buffering=1)
delta_t_crit=0.


np.save(f"{output_dir}/npy_files/x_axis", x_lin )  # use exponential notation
np.save(f"{output_dir}/npy_files/y_axis", y_lin )  # use exponential notation
np.save(f"{output_dir}/npy_files/z_axis", z_lin )  # use exponential notation
np.save(f"{output_dir}/npy_files/L_box", L)



def dft_simulation(t_max,delta_t):
	global psi_1, psi_2	
	timestep = 0
	time = 0.
	while time <= t_max:
		den_1, den_2 = np.abs(psi_1)**2, np.abs(psi_2)**2
		phi_1, phi_2 = get_phis(den_1, den_2)
		
		if(timestep % printToStdoutEvery == 0 or timestep==0):
			if(imProp == False):
				nc1, nc2 = d3r * np.sum(den_1), d3r * np.sum(den_2)
			else:
				nc1, nc2 = np1, np2
			en_contributions = energy(psi_1, psi_2, den_1, den_2, phi_1, phi_2)
			x2_a = np.sqrt(d3r * np.sum(x2 * den_1) / nc1 / L[0]**2)
			x2_b = np.sqrt(d3r * np.sum(x2 * den_2) / nc2 / L[0]**2)
			z2_a = np.sqrt(d3r * np.sum(z2 * den_1) / nc1 / L[2]**2)
			z2_b = np.sqrt(d3r * np.sum(z2 * den_2) / nc2 / L[2]**2)
			en = np.sum(en_contributions)
			string_out= f"{time:.3e} {en:.5e} {en_contributions[0]+en_contributions[1]:.5e} {en_contributions[4]:.5e} {en_contributions[5]:.5e}\
						  {nc1+nc2:.5e} \t {x2_a:.4e} {x2_b:.4e} \t {z2_a:.4e} {z2_b:.4e} \t {np.max(den_1):.4e} {np.max(den_2):.4e} \n"
			if(imProp):
				file_en_equil.write(string_out)
			else:
				file_en_prod.write(string_out)
			print(string_out)

		if(timestep % printDenEvery == 0 or timestep==0):
			timestep /= printDenEvery
			if(imProp==True):
				file=output_dir + '/npy_files/psi_equil_%i' % timestep
			else:
				file=output_dir + '/npy_files/psi_prod_%i' % timestep
			#if(imProp==False):
			#np.save(file, psi)   # use exponential notation
			np.save(file + "_xy_1", dx[2] * np.sum(den_1, axis=2) )  # use exponential notation
			np.save(file + "_xy_2", dx[2] * np.sum(den_2, axis=2) )  # use exponential notation
			#np.save(file + "_xz", dx[1] * np.sum(den, axis=1) )  # use exponential notation
			#np.save(file + "_yz", dx[0] * np.sum(den, axis=0) )  # use exponential notation
			np.save(file + "_x_1", np.swapaxes(den_1,0,2)[int(nxyz[2]/2)][int(nxyz[1]/2)])   # use exponential notation
			np.save(file + "_x_2", np.swapaxes(den_2,0,2)[int(nxyz[2]/2)][int(nxyz[1]/2)])   # use exponential notation

			#np.save(file + "_x_1", np.sum(den_1, axis=(1, 2)))   # use exponential notation
			#np.save(file + "_x_2", np.sum(den_2, axis=(1, 2)))   # use exponential notation
			#np.save(file + "_z_1", np.sum(den_1, axis=(1, 0)))   # use exponential notation
			#np.save(file + "_z_2", np.sum(den_2, axis=(1, 0)))   # use exponential notation
			
			#np.save(file + "_y", np.swapaxes(den,1,2)[int(nxyz[0]/2)][int(nxyz[2]/2)])   # use exponential notation
			np.save(file + "_z_1", den_1[int(nxyz[0]/2)][int(nxyz[1]/2)])   # use exponential notation
			np.save(file + "_z_2", den_2[int(nxyz[0]/2)][int(nxyz[1]/2)])   # use exponential notation
			#np.save(file + "_yz", np.sum(den, axis=0))   # use exponential notation			
			timestep *= printDenEvery
		psi_1, psi_2 = T2_operator(psi_1, psi_2, den_1, den_2, phi_1, phi_2)
		if(imProp == False):
			psi_1 *= kill_matrix 
			psi_2 *= kill_matrix 
		
		time += delta_t
		timestep += 1
		
	return 0



print("Initializing wf's... ")
psi_1, psi_2 = init_psi_1(x, y, z), init_psi_2(x, y, z)
psi_1 *= np.sqrt(np1)/np.sqrt(d3r * np.sum(np.abs(psi_1) ** 2))  #normalize
psi_2 *= np.sqrt(np2)/np.sqrt(d3r * np.sum(np.abs(psi_2) ** 2))  #normalize
print("wfs initiated ")

print(np1, np2)
den1, den2 = np.abs(psi_1)**2, np.abs(psi_2)**2

print(f"\nMAX DENSITY IS {np.max(den1):.3e} {np.max(den2):.3e}\n")


phi1, phi2 = get_phis(den1, den2)
print(f"max density is {np.max(den1+den2):.3e}")
ec = energy(psi_1, psi_2, den1, den2, phi1, phi2)
print(ec/10.840325135236936, np.sum(ec), np.sum(ec) / -5.89393e+00)


print(f"{chemical_potential(psi_1, psi_2)}")
print(f"\nMAX PHI IS {np.max(phi1):.3e} {np.max(phi2):.3e}\n")
#exit()



print("Initializing kinetic energy propagator ... ")
imProp = True
dt = dt_equil
kinprop_1 = np.exp(-1j * dt * sumk2_1)
kinprop_2 = np.exp(-1j * dt * sumk2_2)
print(" *** EQUILIBRATION ... **** ")
dft_simulation(t_equil, deltat_equil)
print("EQUILIBRATION ENDED at ", time.ctime())






'''
print(" *** EQUILIBRATION with 10x smaller timestep, starting from previous w.f. ... **** ")
dt_equil = dt_equil * 0.1
dt = dt_equil
kinprop_1 = np.exp(-1j * dt * sumk2_1)
t_equil = 5000
printToStdoutEvery = (int)(t_equil/deltat_equil / 100)
printDenEvery      = (int)(t_equil/deltat_equil / 100)
output_dir = 'snapshots_time_evolution_1'
os.makedirs(output_dir)
os.makedirs(output_dir+'/npy_files')
file_en_equil = open(output_dir + '/en_equil.dat', 'w', buffering=1)
file_en_prod = open(output_dir + '/en_prod.dat', 'w', buffering=1)

dft_simulation(t_equil, deltat_equil)
'''


print( f"Equilibration took {time.time() - start_time} seconds")



imProp = False
dt = dt_prod
kinprop_1 = np.exp(-1j * dt * sumk2_1)
kinprop_2 = np.exp(-1j * dt * sumk2_2)

print(" *** PRODUCTION ... **** ")
print(" *** delta_t_crit = %.5e" % delta_t_crit)
dft_simulation(t_prod, deltat_prod)

#np.save("psi", cp.asnumpy(psi))

print("SIM. ENDED at ", time.ctime())
print( time.time() - start_time)

	


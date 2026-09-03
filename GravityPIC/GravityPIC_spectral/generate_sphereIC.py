import numpy as np 
from scipy.spatial import cKDTree
import scipy
import h5py

rng = np.random.default_rng()
FloatType = np.float32  # double precision: np.float64, for single use np.float32
IntType = np.int32

R = 1
N = 1000
Mtot = 1
out_file = "output_sphere_lev1/sphere1000_ics.txt"

#position of center of box
centerx = 2
centery = 2
centerz = 2

center_pos = np.array([centerx,centery,centerz])

#set all to 0.25 for no deformation
# eigenvalues of deformation tensor
alpha = 0.25
beta = 0.25
gamma = 0.25

Vel = np.zeros((N,3), dtype=FloatType)
ids = np.arange(N)
mass_weight = Mtot / N

space=((R**3)*(4*np.pi/3)/N)**(1/3)

l=[]

while len(l) < N:
    x, y, z = rng.uniform(-R, R, size=3)
    r2 = x*x + y*y + z*z
    if r2 < R**2:
        l.append([x, y, z])
        
pos=np.array(l)

def repulse(pos, ccut, strength):
    
    N = pos.shape[0]
    
    # --- find all pairs within rcut ---
    rcut = space*ccut
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=rcut, output_type='ndarray')  # shape (Npairs, 2)

    disp = np.zeros_like(pos)

    #quit when everyone is beyond ccut*mean away from eachother
    if pairs.size == 0:
        return pos, disp

    i, j = pairs[:, 0], pairs[:, 1]
    sep = pos[i] - pos[j]                     
    r = np.linalg.norm(sep, axis=1)          

    #get displacement vector
    rhat = sep / r[...,None]         

    #displacement magnitude
    mag = strength * (1.0 - r / rcut) ** 2    

    # push i away from j, and j away from i
    contrib = mag[:, None] * rhat             
    
    np.add.at(disp, i, contrib)
    np.add.at(disp, j, -contrib)
    
    return pos + disp, disp  

def cart_to_sph(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.linalg.norm(pos, axis=1)
    theta = np.arccos(np.clip(z / r, -1, 1))   
    phi = np.arctan2(y, x)  
    return np.column_stack((r, theta, phi))

def sph_to_cart(pos):
    r, theta, phi = pos[:, 0], pos[:, 1], pos[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))


def relax_loop(pos, iterate, ccut=1.25, strength=1):
    #run loop for set number of iterations or quit if displacement becomes negligible
    for i in np.arange(iterate):
        strength_scaled = strength * 0.98**i

        #project to surface of sphere if it escaped
        posnew, disp = repulse(pos, ccut, strength_scaled)
        pos_sph=cart_to_sph(posnew)
        pos_sph[:, 0] = np.minimum(pos_sph[:, 0], R)
        projected=sph_to_cart(pos_sph)
        
        #order by ascending radius
        idx = np.argsort(pos_sph[:, 0])
        pos_sph_sorted = pos_sph[idx]
        
        #new radius will be chosen so that radial dist is exactly r^3
        percentile = (np.arange(N) + 0.5) / N
        pos_sph_sorted[:,0] = R * percentile**(1/3)
        
        uncentered=sph_to_cart(pos_sph_sorted)
        
        final_pos=uncentered-np.average(uncentered, axis=0)
                
        if np.average(np.sum((disp*disp),axis=-1))< 1e-3*space:
            print("stopped at" + str(i))
            break
        pos = final_pos
        
    return final_pos, disp

final_pos, disp=relax_loop(pos, 100)

def sphericalize_preserve_volume(pos):
    N, d = pos.shape
    #compute second order lagrangian tensor
    Cij = (pos.T @ pos) / N      

    #compute whitening xform matrix
    C=np.linalg.det(Cij)
    C_inv_half = scipy.linalg.fractional_matrix_power(Cij, -0.5)
    M=C**(1/6)*C_inv_half

    xformed_pos = pos @ M                   
    
    return xformed_pos

xformed_pos = sphericalize_preserve_volume(final_pos)

# multiply all by a scalar to ensure everything remains inside the sphere
rmax=max(cart_to_sph(xformed_pos)[:,0])
final_pos=xformed_pos* (R / rmax)

# eigenvalues of deformation tensor
lambda_i = [alpha, beta, gamma]

# Zeldovich deformation and velocity assignment 
new_pos = (1 / (2 * alpha)) * (1 - np.array(lambda_i) / (2 * alpha)) * final_pos
new_vel = np.sqrt(2 * alpha) * (1 - np.array(lambda_i) / alpha) * final_pos

# subtract off COM and COM vel

final_pos = new_pos - np.average(new_pos, axis=0)
final_vel = new_vel - np.average(new_vel, axis=0)

weights = np.full((N, 1), mass_weight)

# sphere centered at 0. We want it to be at the center of the box
final_pos += center_pos

data = np.hstack([final_pos, final_vel, weights])

with open(out_file, "w") as f:
    f.write(f"{N}\n")
    for row in data:
        f.write(" ".join(f"{val:.17g}" for val in row) + "\n")

print(f"Wrote {N} particles to {out_file}")
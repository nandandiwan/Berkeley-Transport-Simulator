import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from device import Device


class PoissonSolver:
    def __init__(self, device_obj : Device):
        self.device = device_obj
        self.nx = self.device.nx
        self.nz = self.device.nz
        self.dx = self.device.dx
        self.dz = self.device.dz
        self.q = self.device.q


    def _get_1d_index(self, i, j):
        return i * self.nz + j

    
    
    def solve_initial_poisson_equation_SparseMat(self):

        device_state = self.device

        # Define the grid parameters
        dx, dz = device_state.dx, device_state.dz   # grid step size
        ratio_dx_dz = dx/dz
        Nx, Nz = device_state.potential.shape  # number of grids

        # Define the structure parameters


        # Define the electical parameters
        V_gate = device_state.VG
        V_drain = device_state.Vd
        V_source = device_state.Vs
        gate_start_idx_x = self.nx // 3
        gate_end_idx_x = 2 * (self.nx // 3)
        epsilon = device_state.epsilon.copy()

        n_points = Nx * Nz
        G = np.zeros(n_points)

        def index(i, j):
            return i + j * Nx

        i_list_for_coo_matrix = []
        j_list_for_coo_matrix = []
        data_list_for_coo_matrix = []
        
        for j in range(Nz):
            for i in range(Nx):
                
                idx = index(i, j)
                
                if i == 0:
                    i_list_for_coo_matrix.append(idx)
                    j_list_for_coo_matrix.append(idx)
                    data_list_for_coo_matrix.append(1.0)
                    G[idx] = V_source
                        
                elif i == Nx-1:
                    
                    i_list_for_coo_matrix.append(idx)
                    j_list_for_coo_matrix.append(idx)
                    data_list_for_coo_matrix.append(1.0)
                    G[idx] = V_drain
                    
                    
                elif j == self.nz - 1:
                    if gate_start_idx_x <= i < gate_end_idx_x:  # Middle third (Dirichlet)
                        i_list_for_coo_matrix.append(idx)
                        j_list_for_coo_matrix.append(idx)
                        data_list_for_coo_matrix.append(1.0)
                        G[idx] = V_gate
                    else:  # Outer thirds (Neumann: dV/dz = 0 => V_i,nz-1 = V_i,nz-2)
                        i_list_for_coo_matrix.extend([idx, idx,])
                        j_list_for_coo_matrix.extend([index(i, j-1), idx,])
                        data_list_for_coo_matrix.extend(
                            [1,
                             -1]
                        ) 
                        G[idx] = 0
                elif j == 0:
                    if gate_start_idx_x <= i < gate_end_idx_x:  # Middle third (Dirichlet)
                        i_list_for_coo_matrix.append(idx)
                        j_list_for_coo_matrix.append(idx)
                        data_list_for_coo_matrix.append(1.0)
                        G[idx] = V_gate
                    else:  # Outer thirds (Neumann: dV/dz = 0 => V_i,nz-1 = V_i,nz-2)
                        i_list_for_coo_matrix.extend([idx, idx,])
                        j_list_for_coo_matrix.extend([index(i, j+1), idx,])
                        data_list_for_coo_matrix.extend(
                            [1,
                             -1]
                        ) 
                        G[idx] = 0                 
            

                else: # volume region sparse matrix
                    i_list_for_coo_matrix.extend([idx, idx, idx, idx, idx])
                    j_list_for_coo_matrix.extend([index(i+1, j), index(i, j+1), index(i-1, j), index(i, j-1), idx])
                    data_list_for_coo_matrix.extend(
                        [(epsilon[i, j] + epsilon[i,j-1])/2 / ratio_dx_dz**2, 
                         (epsilon[i-1, j] + epsilon[i,j])/2,
                         (epsilon[i-1, j] + epsilon[i-1,j-1])/2 / ratio_dx_dz**2,
                         (epsilon[i, j-1] + epsilon[i-1,j-1])/2, 
                         -(epsilon[i,j] + epsilon[i-1, j] + epsilon[i-1, j-1] + epsilon[i, j-1])/2 * (1+1/ratio_dx_dz**2)]
                    )

                    G[idx] = 0 # RHS of Poisson equation

        A = sp.coo_matrix((data_list_for_coo_matrix, (i_list_for_coo_matrix, j_list_for_coo_matrix)))

        # transform matrix to CSR representation
        A = A.tocsr()

        # solve this linear system
        Init_V = spsolve(A, G)

        # reshape the solution vector to matrix
        Init_V = Init_V.reshape((Nz, Nx)).T

        return Init_V
    def solve_poisson_equation_SparseMat(self):  

        device_state = self.device

        # Define the grid parameters
        dx, dz = device_state.dx, device_state.dz   # grid step size
        ratio_dx_dz = dx/dz
        Nx, Nz = device_state.potential.shape  # number of grids


        # Define the physics constants
        V_therm = device_state.V_thermal
        e_charge = device_state.q 
        epsilon_0 = device_state.epsilon_0

        # Define the device state parameters
        epsilon = device_state.epsilon.copy()
        gate_start_idx_x = self.nx // 3
        gate_end_idx_x = 2 * (self.nx // 3)
        V_grid = device_state.potential.copy()

        ### charge density calculation
        rho = np.zeros(V_grid.shape)
        rho[:, 1:-1] = \
                e_charge * (device_state.p - device_state.n + device_state.doping_profile)[:, 1:-1]
        

        n_ = np.zeros(rho.shape)
        n_[:, 1:-1] = device_state.n[:, 1:-1]  ### exclude Silicon-IL interface
        
        p_ = np.zeros(rho.shape)
        p_[:, 1:-1] = device_state.p[:, 1:-1] ### exclude Silicon-IL interface

        # Construct the sparse matrix:A and right-hand-side vector:G
        n_points = Nx * Nz
        G = np.zeros(n_points)

        def index(i, j):
            return i + j * Nx

        i_list_for_coo_matrix = []
        j_list_for_coo_matrix = []
        data_list_for_coo_matrix = []

        for j in range(Nz):
            for i in range(Nx):
                
                idx = index(i, j)
                
                if i == 0:
                    i_list_for_coo_matrix.append(idx)
                    j_list_for_coo_matrix.append(idx)
                    data_list_for_coo_matrix.append(1.0)
                    G[idx] = 0
                        
                elif i == Nx-1:
                    
                    i_list_for_coo_matrix.append(idx)
                    j_list_for_coo_matrix.append(idx)
                    data_list_for_coo_matrix.append(1.0)
                    G[idx] = 0
                    
                    
                elif j == self.nz - 1:
                    if gate_start_idx_x <= i < gate_end_idx_x:  # Middle third (Dirichlet)
                        i_list_for_coo_matrix.append(idx)
                        j_list_for_coo_matrix.append(idx)
                        data_list_for_coo_matrix.append(1.0)
                        G[idx] = 0
                    else:  # Outer thirds (Neumann: dV/dz = 0 => V_i,nz-1 = V_i,nz-2)
                        i_list_for_coo_matrix.extend([idx, idx,])
                        j_list_for_coo_matrix.extend([index(i, j-1), idx,])
                        data_list_for_coo_matrix.extend(
                            [1,
                             -1]
                        ) 
                        G[idx] = 0
                elif j == 0:
                    if gate_start_idx_x <= i < gate_end_idx_x:  # Middle third (Dirichlet)
                        i_list_for_coo_matrix.append(idx)
                        j_list_for_coo_matrix.append(idx)
                        data_list_for_coo_matrix.append(1.0)
                        G[idx] = 0
                    else:  # Outer thirds (Neumann: dV/dz = 0 => V_i,nz-1 = V_i,nz-2)
                        i_list_for_coo_matrix.extend([idx, idx,])
                        j_list_for_coo_matrix.extend([index(i, j+1), idx,])
                        data_list_for_coo_matrix.extend(
                            [1,
                             -1]
                        ) 
                        G[idx] = 0     
                else: # volume region sparse matrix
                    a1 = (epsilon[i, j] + epsilon[i,j-1])/2 / ratio_dx_dz**2
                    a2 = (epsilon[i-1, j] + epsilon[i,j])/2
                    a3 = (epsilon[i-1, j] + epsilon[i-1,j-1])/2 / ratio_dx_dz**2
                    a4 = (epsilon[i, j-1] + epsilon[i-1,j-1])/2
                    a0 = -(epsilon[i,j] + epsilon[i-1, j] + epsilon[i-1, j-1] + epsilon[i, j-1])/2 * (1+1/ratio_dx_dz**2)
                    i_list_for_coo_matrix.extend([idx, idx, idx, idx, idx])
                    j_list_for_coo_matrix.extend([index(i+1, j), index(i, j+1), index(i-1, j), index(i, j-1), idx])
                    data_list_for_coo_matrix.extend(
                        [a1, 
                         a2,
                         a3,
                         a4, 
                         a0 - 1/V_therm * e_charge * (p_[i,j] + n_[i,j]) * dz**2 / epsilon_0]
                    )

                    G[idx] = -rho[i,j]* dz**2 / epsilon_0 - \
                        (
                            a1*V_grid[i+1,j] + a2*V_grid[i,j+1] + \
                         a3*V_grid[i-1,j] + a4*V_grid[i,j-1] + \
                           a0*V_grid[i,j]
                        ) # RHS of Poisson equation
                    
        A = sp.coo_matrix((data_list_for_coo_matrix, (i_list_for_coo_matrix, j_list_for_coo_matrix)))

        # convert this matrix into csr representation
        A = A.tocsr()

        # solve this linear system
        Delta_V = spsolve(A, G)

        # reshape the solution vector to array
        Delta_V = Delta_V.reshape((Nz, Nx)).T

        return Delta_V
    
    def solve_poisson_equation(self, tol = 1e-5):
        err = 1
        MAX_ITERATIONS = 100
        self.device.potential = self.solve_initial_poisson_equation_SparseMat()
        self.device.Ev = self.device.potential
        self.device.Ec = self.device.Ev + self.device.Eg
    
        self.device.update_carrier_concentrations_fermi()
        
        
        for iteration in range(MAX_ITERATIONS):
            deltaV = self.solve_poisson_equation_SparseMat()
            self.device.potential += deltaV
            self.device.Ev = self.device.potential
            self.device.Ec = self.device.Ev + self.device.Eg
            err = np.max(np.abs(deltaV))
            if err > tol:
                print(f"Poisson solver converged in {iteration} iterations")
                break
            
            

          
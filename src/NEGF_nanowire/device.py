import numpy as np
import scipy.constants as spc
from NEGF_device_generation import UnitCell
class Device:
    def __init__(self, channel_length = 0.5431 *10e-9, channel_width = 0.5431 *15e-9, channel_thickness = 0.5431 *3e-9, 
                 nx=40, ny = 40, nz=50, T=300.0, material_params=None, equilibrium=False):
        # Physical constants
        self.T = T  # Use the passed temperature parameter
        self.q = spc.e
        self.kbT = spc.Boltzmann * self.T  # Keep in Joules
        self.kbT_eV = spc.Boltzmann * self.T / self.q  # Also store in eV for convenience
        
        self.hbar = spc.hbar
        self.m0 = spc.m_e

        self.epsilon_0 = spc.epsilon_0
        self.a = .5431e-10
        self.V_thermal = self.kbT_eV  # Thermal voltage in eV
                
        
        self.channel_length = channel_length
        self.channel_thickness = channel_thickness
        self.channel_width = channel_width
        
        
        self.unitCell = UnitCell(channel_length, channel_width, channel_thickness, equilibrium_GF=equilibrium)
        # hamiltonian parameters
        self.unitX = self.unitCell.Nx
        self.unitY = self.unitCell.Ny
        self.unitZ = self.unitCell.Nz
        
        
        # voltage 
        self.VG = 0.0
        self.Vs = 0
        self.Vd = 0
    

        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.dx = self.channel_length / (self.nx - 1) if self.nx > 1 else self.channel_length
        self.dy = self.channel_width / (self.ny - 1) if self.ny > 1 else self.channel_width
        self.dz = self.channel_thickness / (self.nz - 1) if self.nz > 1 else self.channel_thickness
        
        if material_params is None:
            self.epsilon_rel = 11.7  # Relative permittivity of Silicon
            self.Eg = 1.12   # Bandgap of Silicon [J]
            self.xi = 4.05    # Electron affinity of Silicon [J] (energy from vacuum=0 to Ec)
            self.Nc = 2.8e25 # Conduction band effective density of states [m^-3] for Si @300K
            self.Nv = 1.04e25 # Valence band effective density of states [m^-3] for Si @300K
            self.me_eff = 0.26 # Effective mass for electrons (conductivity effective mass for Si)
            self.mh_eff = 0.49 # Effective mass for holes (conductivity effective mass for Si)
        else:
            self.epsilon_rel = material_params.get("epsilon_rel", 11.7)
            self.Eg = material_params.get("Eg", 1.12 * self.q)
            self.xi = material_params.get("xi", 4.05 * self.q)
            self.Nc = material_params.get("Nc", 2.8e25)
            self.Nv = material_params.get("Nv", 1.04e25)
            self.me_eff = material_params.get("me_eff", 0.26)
            self.mh_eff = material_params.get("mh_eff", 0.49)

        self.epsilon_val = self.epsilon_rel  # Absolute permittivity [F/m]

        # Doping parameters
        self.N_donor = 1.0e23  # Donor concentration for N-type regions [m^-3] (e.g., 1e17 cm^-3)
        self.N_acceptor = 1.0e21  # Acceptor concentration for P-type regions [m^-3] (e.g., 1e15 cm^-3)

        # --- Core Arrays ---
        self.potential = np.zeros((self.nx, self.ny, self.nz))  # Electric potential [V]
        self.Ec = np.zeros((self.nx, self.ny, self.nz))         # Conduction band edge [J]
        self.Ev = np.zeros((self.nx, self.ny, self.nz))         # Valence band edge [J]
        self.doping_profile = np.zeros((self.nx, self.ny, self.nz)) # Net doping (Nd-Na) [m^-3]
        self.epsilon = np.full((self.nx - 1, self.ny - 1, self.nz - 1), self.epsilon_val) # Permittivity map [F/m]

        self.n = np.zeros((self.nx, self.ny, self.nz))          # Electron concentration [m^-3]
        self.p = np.zeros((self.nx, self.ny, self.nz))          # Hole concentration [m^-3]
        self.Efn = np.zeros((self.nx, self.ny, self.nz))        # Electron quasi-Fermi level [J]
        self.Efp = np.zeros((self.nx, self.ny, self.nz))        # Hole quasi-Fermi level [J]


        self._initialize_doping_profile()
        self._initialize_band_edges_flat_band() # Initial guess for Ec, Ev


    def _initialize_doping_profile(self):
        if self.nx == 0:
            return
        n_region1_end_idx = self.nx // 5
        p_region_end_idx = 4 * (self.nx // 5)
        for i in range(self.nx):
            if i < n_region1_end_idx:
                self.doping_profile[i, :, :] = self.N_donor
            elif i < p_region_end_idx:
                self.doping_profile[i, :,: ] = -self.N_acceptor
            else:
                self.doping_profile[i, :,:] = self.N_donor
    
    
    def _initialize_band_edges_flat_band(self):
        """Initializes Ec and Ev. Assumes V=0 initially. Ec = -xi - qV."""
        self.Ec[:, :,:] = -self.xi # Assuming xi is defined as energy from vacuum (0) to Ec
        self.Ev[:, :,:] = self.Ec - self.Eg



    def update_carrier_concentrations_fermi(self):
        """
        Updates electron (n) and hole (p) concentrations.
        Uses Fermi-Dirac statistics if fdint_half is available, otherwise Boltzmann.
        This method should be called within a self-consistent loop.
        """
        # For Fermi-Dirac (requires fdint library):
        # eta_c = (self.Efn - self.Ec) / (self.kb * self.T)
        # eta_v = (self.Ev - self.Efp) / (self.kb * self.T)
        # try:
        #     self.n = self.Nc * fdint_half(eta_c)
        #     self.p = self.Nv * fdint_half(eta_v)
        # except NameError: # fdint_half not imported
        #     print("Warning: fdint_half not available. Using Boltzmann approximation for n, p.")
        exp_n_arg = np.clip((self.Efn - self.xi - self.Ec) / (self.kbT), -700, 700)
        exp_p_arg = np.clip((self.Ev - self.xi - self.Efp) / (self.kbT), -700, 700)
        
        
        self.n = self.Nc * np.exp(exp_n_arg)
        self.p = self.Nv * np.exp(exp_p_arg)

    def update_quasi_fermi_level(self):
        epsilon = 1e-10  # Small number to prevent log(0) or division by zero

 
        n_density_from_negf = np.maximum(self.n, epsilon)
        p_density_from_negf = np.maximum(self.p, epsilon)

        ratio_n_Nc = n_density_from_negf / (self.Nc + epsilon)
        self.Efn = self.Ec + (self.kbT) * np.log(np.maximum(ratio_n_Nc, epsilon))
        ratio_p_Nv = p_density_from_negf / (self.Nv + epsilon)
  
        self.Efp = self.Ev - (self.kbT) * np.log(np.maximum(ratio_p_Nv, epsilon))
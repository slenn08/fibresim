import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from enum import Enum

class ParameterFields(Enum):
    """
    Enum to identify specific fields within the FibreParameters class
    """
    ATTENUATION = 1
    BETA1 = 2
    BETA2 = 3
    BETA3 = 4
    NL_COEF = 5
    FIBRE_LENGTH = 6
    DZ = 7
    N_MODES = 8
    COUPLING_MATRICES = 9


class BatchFibreParams():
    def __init__(self):
        """
        Initialises the parameters to null values.
        """
        self.params = {
            ParameterFields.ATTENUATION: None,
            ParameterFields.BETA1: None,
            ParameterFields.BETA2: None,
            ParameterFields.BETA3: None,
            ParameterFields.NL_COEF: None,
            ParameterFields.FIBRE_LENGTH: None,
            ParameterFields.DZ: None,
            ParameterFields.N_MODES: None,
            ParameterFields.COUPLING_MATRICES: None,
        }


    def __getitem__(self, idx): 
        if isinstance(idx, int) or isinstance(idx, slice):
            new_params = BatchFibreParams()
            for k in self.params:
                new_params[k] = self.params[k][idx]
            return new_params

        else:
            return self.params[idx]


    def __setitem__(self, param, value):
        self.params[param] = value

    
    def concat_params(self, other_params):
        for k in self.params:
            if self.params[k] is None:
                self.params[k] = other_params[k]
            else:
                self.params[k] = torch.concat((self.params[k], other_params[k]))


class SingleFibreParameters():
    """
    Class to store parameters for simulated fibre. Has functionality to
    directly modify parameters, and also to load in parameters from a .mat file.
    """

    def __init__(self, matfile=None):
        """
        Initialises the parameters to null values. If a matfile is specified, it
        will be loaded and the parameters imported.

        Parameters:
            matfile: string
                The .mat file to load. If left empty, null values will be given to
                all parameters. Default value: None
        """
        self.params = {
            ParameterFields.ATTENUATION: None,
            ParameterFields.BETA1: None,
            ParameterFields.BETA2: None,
            ParameterFields.BETA3: None,
            ParameterFields.NL_COEF: None,
            ParameterFields.FIBRE_LENGTH: None,
            ParameterFields.DZ: None,
            ParameterFields.N_MODES: None,
            ParameterFields.COUPLING_MATRICES: None,
        }

        if matfile:
            self.load_mat(matfile)


    def load_mat(self, matfile):
        """
        Loads fibre parameters from a mat file. 

        Parameters:
            matfile: string
                The matfile to load parameters from.
        """
        mat_data = scipy.io.loadmat(matfile)
        fibre_params = mat_data["FibreP"][0][0]

        # Repeat ensures that different polarisations of same mode has same parameters
        # i.e. np.repeat([1,2,3],2) = [1,1,2,2,3,3]
        att = fibre_params["att"].flatten()
        att = torch.tensor(np.repeat(att,2), dtype=torch.float)
        dmd = fibre_params["beta1"].flatten()
        dmd = torch.tensor(np.repeat(dmd,2), dtype=torch.float)
        beta2 = fibre_params["beta2"].flatten()
        beta2 = torch.tensor(np.repeat(beta2,2), dtype=torch.float)
        beta3 = fibre_params["beta3"].flatten()
        beta3 = torch.tensor(np.repeat(beta3,2), dtype=torch.float)

        L = torch.tensor(fibre_params["L"].flatten()[0])
        dz = torch.tensor(int(fibre_params["dz"].flatten()[0]))
        nModes = torch.tensor(fibre_params["tnom"].flatten()[0] * 2)

        # nlcoef = fibre_params["nlCoef"].flatten()[:nModes//2]
        # nlcoef = torch.tensor(np.repeat(nlcoef,2), dtype=torch.float)#/(10*np.log10(np.exp(1)))/1e3
        # Only works for 1 mode 2 pol
        nlcoef = fibre_params["nlCoef"].flatten()[0]
        nlcoef = torch.tensor([[nlcoef, 2*nlcoef/3],[2*nlcoef/3, nlcoef]], dtype=torch.float)

        Q_import = fibre_params["CoupMatLPabVect"][0]
        Q = torch.zeros((L//dz, nModes, nModes), dtype=torch.complex64)
        for i,q in enumerate(Q_import):
            #Q[i] = torch.eye(nModes)
            Q[i] = torch.from_numpy(q[0])
        
        self.params[ParameterFields.ATTENUATION] = att
        self.params[ParameterFields.BETA1] = dmd
        self.params[ParameterFields.BETA2] = beta2
        self.params[ParameterFields.BETA3] = beta3
        self.params[ParameterFields.NL_COEF] = nlcoef
        self.params[ParameterFields.FIBRE_LENGTH] = L
        self.params[ParameterFields.DZ] = dz
        self.params[ParameterFields.N_MODES] = nModes
        self.params[ParameterFields.COUPLING_MATRICES] = Q

    def __getitem__(self, param):
        return self.params[param]

    def __setitem__(self, param, value):
        self.params[param] = value

    def __repr__(self):
        s = ""
        for parameter in ParameterFields:
            if parameter is not ParameterFields.COUPLING_MATRICES:
                s += str(parameter) + " = " + str(self.params[parameter])
            else:
                s += str(parameter) + " = Tensor of shape " + str(self.params[parameter].shape)
            s += "\n"
        return s


class NonLinearFibre():
    # Based on  this implementation
    # https://github.com/beyondexabit/SDM-DSP-Sandbox/blob/ChannelEstimation_FracTxRx/SDMSimul/fibre_design_all_FEM/mmf_NL_xM_2pol.m
    """
    Simulates non-linear fibre transmission using fixed step SSFM.
    """
    def __init__(self, fibre_params, nSamples, freq_bins, batch_size=1, device=torch.device("cpu")):
        """
        Generates parameters needed to simulate fibre.

        Parameters:
            fibre_params: FibreParameters
                Holds the necessary fibre parameters 
            nSamples: int
                Number of samples in each signal being propagated through the fibre.
            freq_bins: torch.tensor, shape (nSamples, )
                Frequency bins for the signal
            batch_size: int
                Size of the batch. Default: 1
            device: torch.device
                Default: cpu
        """
        self.fibre_params = fibre_params
        self.nSamples = nSamples
        nModes = fibre_params[ParameterFields.N_MODES][0]
        dz = fibre_params[ParameterFields.DZ][0]
        length = fibre_params[ParameterFields.FIBRE_LENGTH][0]
        self.steps = int(length // dz)

        self.half_step = torch.zeros(batch_size, nModes, nSamples, dtype=torch.complex64, device=device)

        for m in range(nModes):
            attenuation = fibre_params[ParameterFields.ATTENUATION][:, m].unsqueeze(1)
            b1 = fibre_params[ParameterFields.BETA1][:, m].unsqueeze(1)
            b2 = fibre_params[ParameterFields.BETA2][:, m].unsqueeze(1)
            b3 = fibre_params[ParameterFields.BETA3][:, m].unsqueeze(1)

            A = torch.exp(-attenuation/2 * dz/2)
            theta = (-b1*(2*torch.pi*freq_bins) - b2/2*(2*torch.pi*freq_bins)**2 - b3/6*(2*torch.pi*freq_bins)**3) * dz/2
            self.half_step[:, m] = A * (torch.cos(theta) + 1j * torch.sin(theta))

            # D = (-attenuation/2 - 
            #      1j*b1*(2*torch.pi*freq_bins) - 
            #      1j*b2/2*(2*torch.pi*freq_bins)**2 -
            #      1j *b3/6*(2*torch.pi*freq_bins)**3
            #     ) * dz/2
            # self.half_step[:, m] = torch.exp(D)
            

    def simulate(self, A0):
        """
        Parameters:
            A0: torch.tensor, shape (batch_size, n_samples, n_modes):
                Complex signal to be transmitted
            steps: int
                The number of steps to simulate
        
        Returns:
            A: torch.tensor, shape (batch_size, n_samples, n_modes):
                Complex signal after transmission through the fibre
        """
        A = torch.transpose(A0, 1, 2)

        for step in range(0, self.steps):
            # Coupling in the time domain
            coup_mats = self.fibre_params[ParameterFields.COUPLING_MATRICES][:, step]
            A = torch.bmm(coup_mats, A)

            # Dispesion step in frequency domain
            A_fft = torch.fft.fft(A,dim=2)
            A_fft = self.half_step * A_fft

            # Non-linear step in time domain
            A = torch.fft.ifft(A_fft, dim=2)
            theta = -torch.bmm(self.fibre_params[ParameterFields.NL_COEF], torch.abs(A)**2) * self.fibre_params[ParameterFields.DZ].unsqueeze(-1).unsqueeze(-1)
            A = A * (torch.cos(theta) + 1j * torch.sin(theta))
            #A = A * torch.exp(-1j * torch.bmm(self.fibre_params[ParameterFields.NL_COEF], torch.abs(A)**2) * self.fibre_params[ParameterFields.DZ][0])

            # Dispesion step in frequency domain
            A_fft = torch.fft.fft(A,dim=2)
            A_fft = self.half_step * A_fft

            A = torch.fft.ifft(A_fft, dim=2)

        return torch.transpose(A, 1, 2)


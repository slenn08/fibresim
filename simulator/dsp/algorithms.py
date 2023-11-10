import torch
from abc import abstractmethod, ABC

from fibres import ParameterFields

class DSP(ABC):
    def __init__(self, fibre_params, nSamples, freq_bins):
        """
        Generates parameters needed to simulate fibre.

        Parameters:
            fibre_params: FibreParameters
                Holds the necessary fibre parameters 
            nSamples: int
                Number of samples in each signal being propagated through the fibre.
            freq_bins: torch.tensor, shape (nSamples, )
                Frequency bins for the signal
        """
        self.dz = fibre_params[ParameterFields.DZ].item()
        self.steps = int(fibre_params[ParameterFields.FIBRE_LENGTH].item() // self.dz)
        
        self.fibre_params = fibre_params
        self.nSamples = nSamples
        self.nModes = fibre_params[ParameterFields.N_MODES]

        self.D = torch.zeros(self.nModes, nSamples, dtype=torch.complex64)

        for m in range(self.nModes):
            attenuation = fibre_params[ParameterFields.ATTENUATION][m]
            b1 = fibre_params[ParameterFields.BETA1][m]
            b2 = fibre_params[ParameterFields.BETA2][m]
            b3 = fibre_params[ParameterFields.BETA3][m]
            self.D[m] = (-attenuation/2 - 
                         1j*b1*(2*torch.pi*freq_bins) - 
                         1j*b2/2*(2*torch.pi*freq_bins)**2 -
                         1j*b3/6*(2*torch.pi*freq_bins)**3)
    
    @abstractmethod
    def simulate(self, A):
        pass


class OptimalDBP(DSP):
    def __init__(self, fibre_params, nSamples, freq_bins):
        """
        Generates parameters needed to simulate fibre.

        Parameters:
            fibre_params: FibreParameters
                Holds the necessary fibre parameters 
            nSamples: int
                Number of samples in each signal being propagated through the fibre.
            freq_bins: torch.tensor, shape (nSamples, )
                Frequency bins for the signal
        """
        super().__init__(fibre_params, nSamples, freq_bins)


    def simulate(self, Al):
        """
        Parameters:
            Al: torch.tensor, shape (n_samples, n_modes):
                Complex signal after transmission through the fibre
        
        Returns:
            A0: torch.tensor, shape (n_samples, n_modes):
                Complex signal that was transmitted
        """
        self.half_step = torch.exp(self.D * self.dz/2)
        
        A = Al.T
        for step in range(self.steps-1, -1, -1):
            # Reverse dispersion step in frequency domain
            A_fft = torch.fft.fft(A, dim=1)
            A_fft = A_fft / self.half_step

            # Reverse non-linearity step in time domain
            A = torch.fft.ifft(A_fft, dim=1)
            A = A / torch.exp(-1j * self.fibre_params[ParameterFields.NL_COEF].matmul(torch.abs(A)**2) * self.dz)
            
            # Reverse dispersion step in frequency domain
            A_fft = torch.fft.fft(A, dim=1)
            A_fft = A_fft / self.half_step

            A = torch.fft.ifft(A_fft, dim=1)

        # for step in range(end_step-1, start_step-1, -1):
            A = torch.conj(self.fibre_params[ParameterFields.COUPLING_MATRICES][step].T).matmul(A)

        return A.T


class DBPAndLinear(DSP):
    def __init__(self, fibre_params, nSamples, freq_bins):
        """
        Generates parameters needed to simulate fibre.

        Parameters:
            fibre_params: FibreParameters
                Holds the necessary fibre parameters 
            nSamples: int
                Number of samples in each signal being propagated through the fibre.
            freq_bins: torch.tensor, shape (nSamples, )
                Frequency bins for the signal
        """
        super().__init__(fibre_params, nSamples, freq_bins)


    def simulate_span(self, Al):
        """
        Parameters:
            Al: torch.tensor, shape (n_samples, n_modes):
                Complex signal after transmission through the fibre
        
        Returns:
            A0: torch.tensor, shape (n_samples, n_modes):
                Complex signal that was transmitted
        """
        A = Al.T

        self.half_step = torch.exp(self.D * self.dz/2)

        # A = torch.exp(self.D.real * self.dz/2)
        # theta = self.D.imag * self.dz/2
        # self.half_step = A * (torch.cos(theta) + 1j * torch.sin(theta))

        # Removes dispersion, nonlinearities
        for step in range(self.steps-1, -1, -1):
            # Reverse dispersion step in frequency domain
            A_fft = torch.fft.fft(A, dim=1)
            A_fft = A_fft / self.half_step

            # Reverse non-linearity step in time domain
            A = torch.fft.ifft(A_fft, dim=1)
            A = A / torch.exp(-1j * self.fibre_params[ParameterFields.NL_COEF].matmul(torch.abs(A)**2) * self.dz)
            
            # Reverse dispersion step in frequency domain
            A_fft = torch.fft.fft(A, dim=1)
            A_fft = A_fft / self.half_step

            A = torch.fft.ifft(A_fft, dim=1)


        # Adds dispersion
        A_fft = torch.fft.fft(A, dim=1)
        for step in range(self.steps-1, -1, -1):
            A_fft = A_fft * self.half_step * self.half_step
        A = torch.fft.ifft(A_fft, dim=1)
        

        # Removes dispersion and coupling
        # dz = self.fibre_params[ParameterFields.DZ]
        # steps = self.fibre_params[ParameterFields.FIBRE_LENGTH] // dz
        # self.half_step = torch.exp(self.D * dz/2)
        for step in range(self.steps-1, -1, -1):
            A_fft = torch.fft.fft(A, dim=1)
            A_fft = A_fft / self.half_step
            A_fft = A_fft / self.half_step

            A = torch.fft.ifft(A_fft, dim=1)
            A = torch.conj(self.fibre_params[ParameterFields.COUPLING_MATRICES][step]).T.matmul(A)

        return A.T
    

class OptimalLinear(DSP):
    def __init__(self, fibre_params, nSamples, freq_bins):
        """
        Generates parameters needed to simulate fibre.

        Parameters:
            fibre_params: FibreParameters
                Holds the necessary fibre parameters 
            nSamples: int
                Number of samples in each signal being propagated through the fibre.
            freq_bins: torch.tensor, shape (nSamples, )
                Frequency bins for the signal
        """
        super().__init__(fibre_params, nSamples, freq_bins)


    def simulate_span(self, Al):
        """
        Parameters:
            Al: torch.tensor, shape (n_samples, n_modes):
                Complex signal after transmission through the fibre
        
        Returns:
            A0: torch.tensor, shape (n_samples, n_modes):
                Complex signal that was transmitted
        """
        A = Al.T

        self.half_step = torch.exp(self.D * self.dz/2)

        # Removes dispersion and coupling
        for step in range(self.steps-1, -1, -1):
            # Reverse dispersion step in frequency domain
            A_fft = torch.fft.fft(A, dim=1)
            A_fft = A_fft / (self.half_step*self.half_step)

            # # Reverse coupling
            A = torch.fft.ifft(A_fft, dim=1)
            A = torch.conj(self.fibre_params[ParameterFields.COUPLING_MATRICES][step]).T.matmul(A)
            #A = torch.linalg.inv(self.fibre_params[ParameterFields.COUPLING_MATRICES][step]).matmul(A)

        return A.T

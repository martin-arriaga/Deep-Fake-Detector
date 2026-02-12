import cv2
import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt
"""
2d Discrete Cosine Transform Engine 
Core Concept: look for artifacts left behind from upsampling low resolution
images (Diffusion models and generative adversarial models)
"""
class SpectralEngine:
    @staticmethod
    def get_2d_dct(image):
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows_dct= dct(grey_image, type=2, norm = 'ortho', axis = 0)
        columns_dct= dct(rows_dct, type = 2, norm='ortho',axis = 1)
        return columns_dct
    @staticmethod
    def log_scale(image):
        return np.log(np.abs(image)+1e-9)

    def visualize_spectral_fingerprint(self, dct):
        logspec = self.log_scale(dct)
        plt.imshow(logspec, cmap='magma')
        plt.title('Log Spectral Fingerprint')
        plt.colorbar()
        plt.show()


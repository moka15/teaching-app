import matplotlib.pyplot as plt
import numpy as np
from shiny.express import input, render, ui

ui.input_slider("radius", "Radius [nm]", 1, 100, 20)
ui.input_slider("wavelength", "Wavelength [nm]", 1, 10, 1)

@render.plot(alt="A histogram")  
def plot():

    radius = input.radius()
    wavelength = input.wavelength()

    delta = 1e-5  # n = 1 - delta + i*beta
    beta = 0

    dynamic_range = 5

    z_det = 0.1 #[m]

    msft_npix_real = 32
    msft_npix_fourier = 128
    box_delta = 2 * radius / (msft_npix_real - 1)
    k_0 = 2 * np.pi / wavelength  # [1/m]

    msft_real_axis = np.linspace(-1, 1, msft_npix_real) * radius
    X, Y, Z = np.meshgrid(msft_real_axis, msft_real_axis, msft_real_axis)
    msft_density = (delta - 1j*beta) * (X ** 2 + Y ** 2 + Z ** 2 < radius ** 2)

    Q_perp_cut = np.fft.fftshift(np.fft.fftfreq(msft_npix_fourier, d=box_delta / (2 * np.pi)))
    Q_X, Q_Y = np.meshgrid(Q_perp_cut, Q_perp_cut, indexing='ij')
    Q_Z = k_0 - np.emath.sqrt(k_0 ** 2 - Q_X ** 2 - Q_Y ** 2)
    Q_angles = np.arcsin(np.sqrt(Q_X ** 2 + Q_Y ** 2) / k_0)

    padding = msft_npix_fourier // 2 - msft_npix_real // 2

    ## Polarization map
    PolMap = 1 - (Q_Y/k_0)**2
    PolMap[Q_X**2 + Q_Y**2 > k_0**2] = np.nan

    msft_image = np.fft.fftshift(np.fft.fft2(np.sum(msft_density, axis=0), s=(msft_npix_fourier, msft_npix_fourier))) * box_delta**2 * 1/(2 * np.pi)
    #cull evanescent waves
    msft_image[Q_X**2 + Q_Y**2 > k_0**2] = np.nan
    msft_image *= k_0/z_det
    msft_image = PolMap * np.abs(msft_image**2)

    fig, ax = plt.subplots()
    max_val = np.nanmax(np.log10(msft_image))
    ax.imshow(np.log10(msft_image), cmap='turbo', vmin=max_val - dynamic_range, vmax=max_val)
    return fig  
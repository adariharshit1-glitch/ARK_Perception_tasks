import cv2
import numpy as np


def loading_image(path):
    image = cv2.imread(path)
    if image is None:
        print("loading fail:", path)
    return image

def CVT_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def morphology(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned

def snr(noisy, filtered):

    noisy = noisy.astype(np.float32)
    filtered = filtered.astype(np.float32)

    noise = noisy - filtered

    signal_power = np.mean(filtered ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)

def snr_blind(image):

    image = image.astype(np.float32)

    signal_power = np.var(image)

    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    noise_power = np.var(laplacian)

    if noise_power == 0:
        return float('inf')

    return 10 * np.log10(signal_power / noise_power)

def save_image(path, image):
    cv2.imwrite(path, image)


ironMan_input = "data/iron_man_noisy.jpg"
ironMan_output = "output/iron_man_denoised.jpg"

ironMan_image = loading_image(ironMan_input)

ironMan_gray = CVT_gray(ironMan_image)

ironMan_median = median_filter(ironMan_gray, 3)

ironMan_clean = morphology(ironMan_median, 3)

snr_iron = snr(ironMan_gray, ironMan_clean)
snr_iron_blind = snr_blind(ironMan_clean)

print("Processed:", ironMan_input)
print("Median Kernel: 3")
print("Morph Kernel: 3")
print("SNR:", round(snr_iron, 3), "dB")
print("SNR (Blind):", round(snr_iron_blind, 3), "dB\n")

save_image(ironMan_output, ironMan_clean)


nature_input = "data/nature_noisy.jpg"
nature_output = "output/nature_denoised.jpg"

nature_image = loading_image(nature_input)

nature_filtered = median_filter(nature_image, 5)

nature_gray_original = CVT_gray(nature_image)
nature_gray_filtered = CVT_gray(nature_filtered)

snr_nature = snr(nature_gray_original, nature_gray_filtered)
snr_nature_blind = snr_blind(nature_gray_filtered)

print("Processed:", nature_input)
print("Median Kernel: 5")
print("SNR:", round(snr_nature, 3), "dB")
print("SNR (Blind):", round(snr_nature_blind, 3), "dB\n")

save_image(nature_output, nature_filtered)
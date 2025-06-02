from PIL import Image
from scipy.stats import skew, kurtosis
import numpy as np

# ===================== FUNGSI DASAR =====================

def gambar_ke_array(img):
    lebar, tinggi = img.size
    pixels = list(img.getdata())
    return [pixels[i * lebar:(i + 1) * lebar] for i in range(tinggi)]

def rgb_ke_hsv_pixel(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    maks = max(r, g, b)
    mins = min(r, g, b)
    delta = maks - mins

    if delta == 0:
        h = 0
    elif maks == r:
        h = (60 * ((g - b) / delta) + 360) % 360
    elif maks == g:
        h = (60 * ((b - r) / delta) + 120) % 360
    else:
        h = (60 * ((r - g) / delta) + 240) % 360

    return h

# ===================== SEGMENTASI =====================

def segmentasi_hue_multi(img, hue_ranges):
    rgb_array = gambar_ke_array(img)
    mask = []

    for row in rgb_array:
        mask_row = []
        for r, g, b in row:
            h = rgb_ke_hsv_pixel(r, g, b)
            cocok = any(start <= h <= end for (start, end) in hue_ranges)
            mask_row.append(1 if cocok else 0)
        mask.append(mask_row)
    return mask

def segmentasi_gelap(img, ambang=80):
    gray_mask = []
    rgb_array = gambar_ke_array(img)
    for row in rgb_array:
        row_mask = []
        for r, g, b in row:
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            row_mask.append(1 if gray <= ambang else 0)
        gray_mask.append(row_mask)
    return gray_mask

def gabungkan_mask(mask1, mask2):
    return [[1 if m1 or m2 else 0 for m1, m2 in zip(r1, r2)] for r1, r2 in zip(mask1, mask2)]

def morfologi_lengkap(mask, kernel_size=3):
    return erosi(dilasi(erosi(mask, kernel_size), kernel_size), kernel_size)

def erosi(mask, kernel_size=3):
    offset = kernel_size // 2
    hasil = []
    for i in range(len(mask)):
        row = []
        for j in range(len(mask[0])):
            valid = True
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    ni, nj = i + dx, j + dy
                    if not (0 <= ni < len(mask) and 0 <= nj < len(mask[0])) or mask[ni][nj] == 0:
                        valid = False
            row.append(1 if valid else 0)
        hasil.append(row)
    return hasil

def dilasi(mask, kernel_size=3):
    offset = kernel_size // 2
    hasil = []
    for i in range(len(mask)):
        row = []
        for j in range(len(mask[0])):
            aktif = False
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < len(mask) and 0 <= nj < len(mask[0]) and mask[ni][nj] == 1:
                        aktif = True
            row.append(1 if aktif else 0)
        hasil.append(row)
    return hasil

def segmentasi_overripe_komplit(img):
    mask1 = segmentasi_hue_multi(img, [(10, 30), (260, 310)])
    mask2 = segmentasi_gelap(img, 80)
    mask3 = segmentasi_hue_multi(img, [(40, 90)])

    combined = gabungkan_mask(mask1, mask2)
    final = gabungkan_mask(combined, mask3)
    return morfologi_lengkap(final, 3)

# ===================== EKSTRAKSI FITUR =====================

def ekstraksi_fitur_warna(img, mask):
    rgb_array = gambar_ke_array(img)
    hue, sat, val = [], [], []

    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                r, g, b = rgb_array[i][j]
                r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
                maks, mins = max(r_, g_, b_), min(r_, g_, b_)
                delta = maks - mins

                # Hue
                if delta == 0:
                    h = 0
                elif maks == r_:
                    h = (60 * ((g_ - b_) / delta) + 360) % 360
                elif maks == g_:
                    h = (60 * ((b_ - r_) / delta) + 120) % 360
                else:
                    h = (60 * ((r_ - g_) / delta) + 240) % 360
                hue.append(h)

                # Saturation dan Value
                s = 0 if maks == 0 else delta / maks
                v = maks
                sat.append(s)
                val.append(v)

    return {
        "mean_hue": np.mean(hue) if hue else 0,
        "std_hue": np.std(hue) if hue else 0,
        "skewness_hue": skew(hue) if hue else 0,
        "kurtosis_hue": kurtosis(hue) if hue else 0,
        "mean_saturation": np.mean(sat) if sat else 0,
        "std_saturation": np.std(sat) if sat else 0,
        "mean_value": np.mean(val) if val else 0,
        "std_value": np.std(val) if val else 0
    }

def segmentasi_ke_grayscale(img, mask):
    rgb_array = gambar_ke_array(img)
    gray_array = []
    for i in range(len(mask)):
        row = []
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                r, g, b = rgb_array[i][j]
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            else:
                gray = 0
            row.append(gray)
        gray_array.append(row)
    return gray_array

def hitung_glcm(grayscale_array, level=8):
    glcm = [[0]*level for _ in range(level)]
    tinggi, lebar = len(grayscale_array), len(grayscale_array[0])

    def quantize(x): return min(x * level // 256, level - 1)

    for i in range(tinggi):
        for j in range(lebar - 1):
            g1 = quantize(grayscale_array[i][j])
            g2 = quantize(grayscale_array[i][j + 1])
            glcm[g1][g2] += 1

    total = sum(map(sum, glcm))
    return [[v / total if total else 0 for v in row] for row in glcm]

def ekstrak_fitur_glcm(glcm):
    contrast = energy = homogeneity = correlation = 0
    N = len(glcm)
    mu_i = [sum(glcm[i][j] * j for j in range(N)) for i in range(N)]
    mu_j = [sum(glcm[i][j] * i for i in range(N)) for j in range(N)]
    mean_i, mean_j = np.mean(mu_i), np.mean(mu_j)
    std_i, std_j = np.std(mu_i), np.std(mu_j)

    for i in range(N):
        for j in range(N):
            p = glcm[i][j]
            contrast += p * ((i - j)**2)
            energy += p**2
            homogeneity += p / (1 + abs(i - j))
            if std_i and std_j:
                correlation += ((i - mean_i) * (j - mean_j) * p) / (std_i * std_j)
    return {
        "Contrast": contrast,
        "Energy": energy,
        "Homogeneity": homogeneity,
        "Correlation": correlation
    }

# ===================== FUNGSI UTAMA UNTUK DEPLOY =====================

def ekstrak_semua_fitur(img: Image.Image) -> (dict, str):
    # Resize dulu
    img = img.resize((224, 224))

    # Buat semua mask
    mask_r = segmentasi_hue_multi(img, [(30, 60)])
    mask_u = segmentasi_hue_multi(img, [(70, 130)])
    mask_o = segmentasi_overripe_komplit(img)

    # Hitung piksel aktif
    total_r = sum(map(sum, mask_r))
    total_u = sum(map(sum, mask_u))
    total_o = sum(map(sum, mask_o))

    kelas = max(
        [("ripe", total_r), ("unripe", total_u), ("overripe", total_o)],
        key=lambda x: x[1]
    )[0]

    # Ambil mask dominan
    mask = {"ripe": mask_r, "unripe": mask_u, "overripe": mask_o}[kelas]

    # Ekstraksi
    fitur_warna = ekstraksi_fitur_warna(img, mask)
    gray_array = segmentasi_ke_grayscale(img, mask)
    glcm = hitung_glcm(gray_array)
    fitur_glcm = ekstrak_fitur_glcm(glcm)

    fitur = {**fitur_warna, **fitur_glcm}
    return fitur, kelas

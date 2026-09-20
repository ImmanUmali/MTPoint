import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

def v_normalize(x, x_min, x_max):
    return np.clip((2*x - (x_min + x_max)) / (x_max - x_min), -1, 1)

def v_denormalize(z, x_min, x_max):
    return (np.clip(z, -1, 1) * (x_max - x_min) + (x_min + x_max)) / 2

def apply_gaussian_filter(d, size=25, sigma=3):
    # Gaussian filter
    arr = np.arange(size // 2 * (-1), size // 2 + 1)
    gf = np.exp(-np.power(arr, 2) / (2 * sigma**2))
    gf = np.flip(gf / gf.sum())

    return np.convolve(
        np.pad(d, size // 2, mode='edge'), 
        gf, 
        mode='valid'
    )

def find_divisors(N):
    for i in range(int(np.sqrt(N)), 0, -1):
        if N % i == 0:
            return i, N // i

def task_index(t_cue, p, n):
    return [0.05, 0.175, 0.35, 0.6].index(t_cue) * 6 + [1.25, 1.8].index(p) * 3 + [1, 2, 4].index(n)

def task_cond(idx):
    return [0.05, 0.175, 0.35, 0.6][idx//6], [1.25, 1.8][(idx%6)//3], [1, 2, 4][idx%3]


def compute_r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    return r_squared
        

if __name__ == "__main__":
    import psutil
    print(psutil.cpu_count(logical=False))
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def quantize_kl(weights):
    """
    计算kl散度量化
    :param weights: 输入分布
    :return:
    """
    weights_scales = np.zeros(weights.shape)
    absmax = max(weights)
    n = len(weights)
    




def quantize_aciq(weights):
    """
    计算aciq量化
    :param weights: 输入分布
    :return: scale + threshold
    """
    weight_scale = np.zeros(weights.shape)
    absmax = max(weights)
    for i in range(len(weights)):
        threshold = compute_aciq_gaussian_clip(absmax, len(weights))
        weight_scale[i] = 127 / threshold
    return weight_scale, threshold


def compute_aciq_gaussian_clip(absmax, N, num_bits=8):
    alpha_gaussian = [0, 1.71063519, 2.15159277, 2.55913646,
                      2.93620062, 3.28691474, 3.6151146, 3.92403714]
    gaussian_const = (0.5 * 0.35) * \
        (1 + math.sqrt(3.14159265358979323846 * math.log10(4)))
    std = (absmax * 2 * gaussian_const) / math.sqrt(2 * math.log10(N))

    return alpha_gaussian[num_bits-1] * std


def compute_mean_variance(gaussian_distribution):
    """
    Compute the variance
    :param gaussian_distribution: 数据分布
    :return: 均值+方差    
    """
    n = len(gaussian_distribution)
    mean = sum(gaussian_distribution) / n
    variance = 0
    for i in range(n):
        variance += pow(gaussian_distribution[i] - mean, 2)
    variance /= n
    return mean, variance

if __name__ == "__main__":
    print("Starting aciq!")
    plt.figure(figsize=(10, 6))

    plt.subplot(311)
    layer_weights = np.random.normal(0, 1, 300)
    mean, var = compute_mean_variance((layer_weights))
    print("origin ditribute mean = {}, var = {}".format(mean, var))

    n, bins, _ = plt.hist(layer_weights, bins=50, rwidth=0.8,
                          density=True, stacked=True, alpha=0.7, align='mid')
    y = norm.pdf(bins, 0, 1)
    plt.plot(bins, y, color='r')
    plt.title("distribute")
    plt.xlabel("value")
    plt.ylabel("probability")

    plt.subplot(312)
    layer_weights_scale, layer_threshold = quantize_aciq(layer_weights)
    print("aciq threshold = {}".format(layer_threshold))
    aciq_layer_weghts = layer_weights * layer_weights_scale
    aciq_mean, aciq_var = compute_mean_variance(aciq_layer_weghts)
    print("origin ditribute mean = {}, var = {}".format(aciq_mean, aciq_var))

    n, bins, _ = plt.hist(aciq_layer_weghts, bins=50,
                          rwidth=0.8, density=True, stacked=True, alpha=0.7, align='mid')
    y = norm.pdf(bins, 0, 1)
    plt.plot(bins, y, color='r')
    plt.title("distribute")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(313)
    plt.plot(layer_weights, aciq_layer_weghts)
    plt.title("quant value")
    plt.xlabel("fp32 data")
    plt.ylabel("int8 data")
    plt.grid()

    plt.show()

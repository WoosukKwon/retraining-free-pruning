import numpy as np


def fit_piecewise_linear(lut):
    best = {"error": 100}

    for threshold in range(1, len(lut) + 1):
        c = lut[:threshold].sum() / threshold
        x_i = lut[threshold:] - c
        i = np.arange(1, len(x_i) + 1)
        if len(i) == 0:
            slope = 0
        else:
            slope = (i * x_i).sum() / (i * i).sum()
        slope = 0 if slope < 0 else slope

        approximated = [c] * threshold
        for i in range(1, len(lut) - threshold + 1):
            approximated.append(slope * i + c)
        approximated = np.asarray(approximated)

        squared_error = ((lut - approximated) * (lut - approximated)).sum()
        if squared_error < best["error"]:
            best["error"] = squared_error
            best["threshold"] = threshold
            best["c"] = c
            best["slope"] = slope
            best["approximated"] = approximated

    return best

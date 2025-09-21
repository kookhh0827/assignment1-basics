import math


# cosine annealing with linear warmup
def cosine_annealing_with_linear_warmup(t, eta_max, eta_min, Tw, Tc):
  if t < Tw:
    return eta_max * t / Tw
  elif t > Tc:
    return eta_min
  else:
    return eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw))) / 2
    
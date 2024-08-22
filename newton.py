# x_t = x_{t-1} - f'(x_{t-1})/f''(x_{t-1})
# implement newton's method for: f(x) = cosx

import numpy as np

x = 1
while abs(np.sin(x)/np.cos(x)) > 0.01:
    x = x - np.sin(x)/np.cos(x)
    print(x, abs(np.sin(x)/np.cos(x)))
print("result", x)
'''
-0.5574077246549021 0.6233441765797427
0.06593645192484066 0.06603217384416574
-9.572191932508134e-05 9.572191961743796e-05
'''
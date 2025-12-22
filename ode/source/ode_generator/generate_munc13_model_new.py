import generate_ode_system as gos
import sys


# Open a file for writing
file = open('munc13_model_new_3.txt', 'w')

# Redirect stdout to the file
sys.stdout = file

gos.generate_ode_system([
    'S + R <-> M, kfsr, krsr',
    'M + M <-> D, gamma*kfmm, krmm',
    'S + X -> W + Y, kfc',
    'M + X -> C + Y, gamma*kfc',
    'D + X -> C + C + Y + Y - X, 2*gamma*kfc',
    'S + Y -> W + Z, kfc',
    'M + Y -> C + Z, gamma*kfc',
    'D + Y -> C + C + Z + Z - Y, 2*gamma*kfc',
    'S + Z -> W, kfc',
    'M + Z -> C, gamma*kfc',
    'D + Z -> C + C - Z, 2*gamma*kfc',
    ])

# Close the file
file.close()

# Restore stdout to its default value
sys.stdout = sys.__stdout__

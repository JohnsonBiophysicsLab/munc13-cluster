import generate_ode_system as gos
import sys


# Open a file for writing
file = open('munc13_model_new_2_noc.txt', 'w')

# Redirect stdout to the file
sys.stdout = file

gos.generate_ode_system([
    'S + R <-> M, kfsr, krsr',
    'M + M <-> D, gamma*kfmm, krmm',
    ])

# Close the file
file.close()

# Restore stdout to its default value
sys.stdout = sys.__stdout__

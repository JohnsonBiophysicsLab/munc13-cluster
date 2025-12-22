import generate_ode_system as gos
import sys


# Open a file for writing
file = open('munc13_model_dimer.txt', 'w')

# Redirect stdout to the file
sys.stdout = file

gos.generate_ode_system(['M + M <-> MM, kf0, kr0',
                         'M + R <-> MR, kf2, kr2',
                         'MR + M <-> MMR, kf0, kr0',
                         'MR + MR <-> MMRR, gamma*kf0, kr0',
                         'MM + R <-> MMR, 2*kf2, kr2',
                         'MMR + R <-> MMRR, gamma*kf2, kr2',])

# Close the file
file.close()

# Restore stdout to its default value
sys.stdout = sys.__stdout__

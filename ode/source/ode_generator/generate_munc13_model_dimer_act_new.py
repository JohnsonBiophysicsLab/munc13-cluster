import generate_ode_system as gos
import sys


# Open a file for writing
file = open('munc13_model_dimer_act.txt', 'w')

# Redirect stdout to the file
sys.stdout = file

gos.generate_ode_system(['M + M <-> MM, kf0, kr0',
                         'M + R <-> MR, kf2, kr2',
                         'M + C -> MC, kf1',
                         'M + MC <-> 2MC, kf3, kr3',
                         'M + MC <-> MMC, kf0, kr0',
                         'M + MMC <-> MMC + MC, 2*kf3, kr3',
                         'M + MR <-> MMR, kf0, kr0',
                         'M + MRC <-> MRC + MC, kf3, kr3',
                         'M + MRC <-> MMRC, kf0, kr0',
                         'M + MMRC <-> MMRC + MC, 2*kf3, kr3',
                         'R + MM <-> MMR, 2*kf2, kr2',
                         'R + MC <-> MRC, gamma*kf2, kr2',
                         'R + MMC <-> MMRC, gamma*kf2, kr2',
                         'C + MM -> MMC, 2*kf1',
                         'C + MR -> MRC, gamma*kf1',
                         'C + MMR -> MMRC, gamma*kf1',
                         'MM + MC <-> MC + MMC, kf3, kr3',
                         'MM + MRC <-> MRC + MMC, kf3, kr3',
                         'MM + MMC <-> 2MMC, 2*kf3, kr3',
                         'MM + MMRC <-> MMRC + MMC, 2*kf3, kr3',
                         'MR + MC <-> MC + MRC, gamma*kf3, kr3',
                         'MR + MMC <-> MMC + MRC, 2*gamma*kf3, kr3',
                         'MR + MRC <-> 2MRC, gamma*kf3, kr3',
                         'MR + MMRC <-> MMRC + MRC, 2*gamma*kf3, kr3',
                         'MC + MMR <-> MC + MMRC, gamma*kf3, kr3',
                         'MMC + MMR <-> MMC + MMRC, 2*gamma*kf3, kr3',
                         'MMR + MRC <-> MRC + MMRC, gamma*kf3, kr3',
                         'MMR + MMRC <-> 2MMRC, 2*gamma*kf3, kr3',
                         ])

# Close the file
file.close()

# Restore stdout to its default value
sys.stdout = sys.__stdout__

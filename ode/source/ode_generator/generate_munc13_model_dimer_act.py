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
                         'M + MMRRC <-> MMRRC + MC, 2*kf3, kr3',
                         'M + MMRCC <-> MMRCC + MC, 2*kf3, kr3',
                         'M + MMRRCC <-> MMRRCC + MC, 2*kf3, kr3',
                         'M + MMCC <-> MMCC + MC, 2*kf3, kr3',
                         'R + MM <-> MMR, 2*kf2, kr2',
                         'R + MC <-> MRC, gamma*kf2, kr2',
                         'R + MMR <-> MMRR, gamma*kf2, kr2',
                         'R + MMC <-> MMRC, 2*gamma*kf2, kr2',
                         'R + MMRC <-> MMRRC, gamma*kf2, kr2',
                         'R + MMRCC <-> MMRRCC, gamma*kf2, kr2',
                         'R + MMCC <-> MMRCC, 2*gamma*kf2, kr2',
                         'C + MM -> MMC, 2*kf1',
                         'C + MR -> MRC, gamma*kf1',
                         'C + MMC -> MMCC, gamma*kf1',
                         'C + MMR -> MMRC, 2*gamma*kf1',
                         'C + MMRC -> MMRCC, gamma*kf1',
                         'C + MMRRC -> MMRRCC, gamma*kf1',
                         'C + MMRR -> MMRRC, 2*gamma*kf1',
                         'MM + MC <-> MC + MMC, kf3, kr3',
                         'MM + MRC <-> MRC + MMC, kf3, kr3',
                         'MM + MMC <-> 2MMC, 2*kf3, kr3',
                         'MM + MMRC <-> MMRC + MMC, 2*kf3, kr3',
                         'MM + MMCC <-> MMCC + MMC, 2*kf3, kr3',
                         'MM + MMRRC <-> MMRRC + MMC, 2*kf3, kr3',
                         'MM + MMRCC <-> MMRCC + MMC, 2*kf3, kr3',
                         'MM + MMRRCC <-> MMRRCC + MMC, 2*kf3, kr3',
                         'MR + MR <-> MMRR, gamma*kf0, kr0',
                         'MR + MC <-> MMRC, gamma*kf0, kr0',
                         'MR + MC <-> MC + MRC, gamma*kf3, kr3',
                         'MR + MMC <-> MMC + MRC, 2*gamma*kf3, kr3',
                         'MR + MRC <-> 2MRC, gamma*kf3, kr3',
                         'MR + MRC <-> MMRRC, gamma*kf0, kr0',
                         'MR + MMRC <-> MMRC + MRC, 2*gamma*kf3, kr3',
                         'MR + MMCC <-> MMCC + MRC, 2*gamma*kf3, kr3',
                         'MR + MMRRC <-> MMRRC + MRC, 2*gamma*kf3, kr3',
                         'MR + MMRCC <-> MMRCC + MRC, 2*gamma*kf3, kr3',
                         'MR + MMRRCC <-> MMRRCC + MRC, 2*gamma*kf3, kr3',
                         'MC + MC <-> MMCC, gamma*kf0, kr0',
                         'MC + MMR <-> MC + MMRC, gamma*kf3, kr3',
                         'MC + MRC <-> MMRCC, gamma*kf0, kr0',
                         'MC + MMRR <-> MC + MMRRC, gamma*kf3, kr3',
                         'MMC + MMR <-> MMC + MMRC, 2*gamma*kf3, kr3',
                         'MMC + MMRR <-> MMC + MMRRC, 2*gamma*kf3, kr3',
                         'MMR + MRC <-> MRC + MMRC, gamma*kf3, kr3',
                         'MMR + MMRC <-> 2MMRC, 2*gamma*kf3, kr3',
                         'MMR + MMCC <-> MMCC + MMRC, 2*gamma*kf3, kr3',
                         'MMR + MMRRC <-> MMRRC + MMRC, 2*gamma*kf3, kr3',
                         'MMR + MMRCC <-> MMRCC + MMRC, 2*gamma*kf3, kr3',
                         'MMR + MMRRCC <-> MMRRCC + MMRC, 2*gamma*kf3, kr3',
                         'MRC + MRC <-> MMRRCC, gamma*kf0, kr0',
                         'MRC + MMRR <-> MRC + MMRRC, gamma*kf3, kr3',
                         'MMRC + MMRR <-> MMRC + MMRRC, 2*gamma*kf3, kr3',
                         'MMRR + MMRCC <-> MMRCC + MMRRC, 2*gamma*kf3, kr3',
                         'MMRR + MMRRC <-> 2MMRRC, 2*gamma*kf3, kr3',
                         'MMRR + MMRRCC <-> MMRRCC + MMRRC, 2*gamma*kf3, kr3'
                         ])

# Close the file
file.close()

# Restore stdout to its default value
sys.stdout = sys.__stdout__

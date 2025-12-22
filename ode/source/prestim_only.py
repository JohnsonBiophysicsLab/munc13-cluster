import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from munc13 import Munc13, Solver

import sys
print(sys.path)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Main function below.
# It produces as output the file named by default ../data/optimizedParms_clusterRun.txt
#this is written to in the function write_sortedParms in the Solver class
#This default name can be overwritten by passing a different name to the Solver constructor
if __name__ == "__main__":
    start_time = time.time()
    # Parameter definitions
    print("Start main, initialize model and solver...")
    parameter_ranges = {
        "kfsr":      {"min": 0.001, "max": 10},   # kfSR uM-1s-1
        #"krsr_nostim":      {"min": 0.1,   "max": 1000}, # krSR_nostim
        "krsr":       {"min": 0.001,   "max": 1000}, # krSR_stim s-1
        "kfmm":      {"min": 0.1, "max": 10},   # kfMM uM-1s-1
        "krmm":      {"min": 0.01,   "max": 10}, # krMM s-1
        "kfmx":      {"min": 0.001, "max": 10},   # kf1x uM-1s-1
        "krmx":      {"min": 0.01,   "max": 1000}, # kr1x s-1
       # "kfc_nostim":      {"min": 0.001, "max": 10},   # kfc_nostim
        "kfc":      {"min": 0.001, "max": 10},   # kfc_stim uM-1s-1
        "krc":      {"min": 0.01, "max": 1000},   # krc
        "kfxx":      {"min": 0.001, "max": 10},   # kx2 uM-1s-1
        "krxx":      {"min": 0.01,   "max": 1000}, # krx2 s-1
        #"sig":       {"min": 1,   "max": 10}, # sig, scale factor >1
        "eLoop":    {"min": 0.001, "max": 1}, #exp(free energy <0)
        "eDF":      {"min": 0.0001,   "max": 1}, # exp(free energy kT units <0).
        "kfdd":     {"min": 0.01,   "max": 1}, # kfdd unimolecular: s-1
        "stimUpSR":       {"min": 1,   "max": 100}, # stimUpSR: scale factor >1
        "S0":        {"min": 0.001, "max": 0.5},   # S0 (uM)
        "R0":        {"min": 0.5, "max": 1000},   # R0 (/um^2)
        #"D1":        {"min": 0.05,   "max": 5}, # D1
        #"D1_over_D2":        {"min": 1.5,   "max": 5}, # D2
        "X0":        {"min": 0.01,   "max": 50}, # X0  (/um^2)
         
    }

    # Order in which the solver will read parameters from a candidate
    # so, e.g. candidate[0] = kfsr.
    params_to_optimize = np.array([
        "kfsr","krsr","kfmm","krmm","kfmx","krmx","kfc","krc","kfxx","krxx","eLoop","eDF","kfdd","stimUpSR","S0","R0","X0"
    ])
    print("number of parameters to optimize: ", params_to_optimize.size)
    # GA settings
    popSize = 5000
    nGen = 5

    # Instantiate the model and solver, including max time limit.
    maxTime = 800.0
    model = Munc13(parameter_ranges, params_to_optimize, t_max=maxTime)
    #the solver can be passed a specific filename for the solutions.
    random_number = np.random.randint(1, 10000)
    filename = f"../data/testParms_preStimOnly_{random_number}.txt"
    print("Output filename: ", filename)
    solver = Solver(model, populationSize=popSize, NGEN=nGen, outfileName=filename)

    # For testing: simulate and plot one candidate solution
    testOne=False
    #Set testOne to false to run the optimizer.
    if(testOne):
        print('Test one candidate solution...')
        #test_candidate = [1.7010477963667472, 0.001689591986722821, 0.10235722071503092, 5.615119684346966, 2.8583908752630456, 0.027546902406542565, 2.203977862345212, 0.04862994328083227, 0.03843319096381628, 762.9136096968069, 9.08072531453085, 0.002375058508699627, 0.10664103571145198, 2.125872789394077, 0.01, 17.53699421106815, 11.611444433666234]
        test_candidate = [0.016829832020490265, 0.00395936538873811, 0.008500338099183543, 0.07125294855359987, 1.1410896484560995, 0.046911314688838794, 2.5134894424798437, 0.01, 2.6281477842717402, 2.704397340886635, 64.75892464777132, 0.7504104571009792, 0.17723463652848184, 64.19919670691235, 0.1, 9.535154024381814, 0.345296743329276]
        model.test_candidate_and_mutant(test_candidate)
        viableFitness = []
    else:
        # Run the GA
        print("Run the GA ", filename)
        viablePoints, viableFitness = solver.run()

    #look at one solution
    #    y1= model.simulate_pre(viablePoints[0])
    #    model.plot_mycluster_time(y1, figsize=(8,6), dpi=300)

    
    # Print top solution info
    if len(viableFitness) > 0:
        best_fit = max(viableFitness, key=lambda x: x[0])
        print(f"Best fitness from the run: {best_fit[0]}")
        print("Best parameters:", viablePoints[viableFitness.index(best_fit)])
    else:
        print("No viable solutions found.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_minutes = elapsed_time / 60
    print(f"Total execution time: {elapsed_time_minutes:.2f} minutes")
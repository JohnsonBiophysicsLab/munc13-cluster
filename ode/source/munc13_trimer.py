import os
import math
import numpy as np
import pandas as pd
import scipy.integrate
import matplotlib.pyplot as plt
import time
import seaborn as sns
from matplotlib.patches import Patch

# DEAP imports
from deap import base, creator, tools, algorithms

# Multiprocessing for parallel execution
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Munc13:
    def __init__(self, parameter_ranges, params, mode=0, t_max=50000, experiment_dt=0.5):
        """
        A model class for simulating Munc13 dynamics (pre- and post-stimulation).
        
        :param parameter_ranges: dict with min/max for each parameter to be optimized
        :param params: list/array of parameter names (strings) in the order they appear in candidate solutions
        :param mode: an integer to select among multiple evaluation modes
        """
        self.n_params = len(params)
        self.params_to_optimize = params
        self.parameter_ranges = parameter_ranges
        self.t_max = t_max
        #self.t_max_pre = t_max_pre
        self.h = 0.01  # 3D to 2D length scale in um
        
        
       # self.Atotal = 1 / 1.176        # (1/cluster density) area per cluster in um^2, pre-stim (determined by experimental cluster density)
        #self.Acluster = 0.01862     # cluster area, pre-stim
       # self.V = self.Atotal * VAratio
        #self.gamma = self.V / (self.Atotal * self.h)

        #self.AtotalDense = 1 / 1.675   # (1/cluster density) area per cluster in um^2, post-stim (determined by experimental cluster density)
        #self.AclusterDense = 0.02087 # cluster area, post-stim
        #self.VDense = self.AtotalDense * VAratio
        #self.gammaDense = self.VDense / (self.AtotalDense * self.h)

        #self.densIncrease = 3

        self.cellVolume = 4.0/3.0* 3.14* 10.0**3   # in um^3
        self.cellArea = 4.0* 3.14 * 10.0**2     # in um^2
        VAratio = self.cellVolume/self.cellArea # Vol/Area ratio, in units of um
        self.gamma=VAratio/self.h
        print(f"Gamma is {self.gamma}")
        #the reported cluster density is integrated over 320s of time, so
        #to estimate the instantaneous density, this must be lowered,
        #lifetime of clusters is ~10s, hence the factor 10/320
        self.density_exp_post = 1.675*10/320  # clusters per um^2, post-stim
        self.density_exp_pre = 1.176*10/320  # clusters per um^2, pre-stim
        self.recruitmentStim = 3 #increase of munc13 on the membrane upon stimulation
        # diffusion constants
        self.D_exp_pre = 0.04255
        self.D_exp_pre_sem = 0.003675
        self.D_model_pre = None

        self.D_exp_post = 0.035611
        self.D_exp_post_sem = 0.002845
        self.D_model_post = None

        self.D_exp_DC2A_pre = 0.07652157
        self.D_exp_DC2A_pre_sem = 0.003385
        self.D_model_DC2A_pre = None

        self.D_exp_DC2A_post = 0.080462
        self.D_exp_DC2A_post_sem = 0.00273
        self.D_model_DC2A_post = None

        self.D_exp_shRIM_pre = 0.04514325
        self.D_exp_shRIM_pre_sem = 0.006545
        self.D_model_shRIM_pre = None

        self.D_exp_shRIM_post = 0.033447
        self.D_exp_shRIM_post_sem = 0.005401
        self.D_model_shRIM_post = None

        self.D_exp_DC2B_pre = 0.04692072
        self.D_exp_DC2B_pre_sem = 0.003362
        self.D_model_DC2B_pre = None

        self.D_exp_DC2B_post = 0.040632
        self.D_exp_DC2B_post_sem = 0.003063
        self.D_model_DC2B_post = None

        self.D_model_endogenous_pre = None
        self.D_model_endogenous_post = None

        self.bestSolution = None
        self.filteredSolutions = None

        self.D_munc = None
        self.D_R = None

        self.final_d_pre = None
        self.final_d_post = None

        # Experimental data setup, NOT USED
        targetIncrease = 15.1
        targetIncreasePost = 11.4
        n_time_points = int(self.t_max / experiment_dt)
        #n_time_points_pre = int(self.t_max_pre / experiment_dt)
        self.timePoints = [i * experiment_dt for i in range(n_time_points + 1)]
       # self.timePointsPre = [i * experiment_dt for i in range(n_time_points_pre + 1)]

        # Load experimental data
        self.expdata = pd.read_table(
            os.path.join("../data/wt_result_nonorm95.txt"),
            header=None, sep=r"\s+",
            names=["time", "expInt", "expSEM"]
        )
        self.expTime = self.expdata.time
        N = len(self.expdata)

        # Adjust intensities for modeling
        self.delIntensity = (self.expdata.expInt[N - 1] - targetIncrease * self.expdata.expInt[0]) / (1.0 - targetIncrease)
        self.delIntensityPost = (self.expdata.expInt[N - 1] - targetIncreasePost * self.expdata.expInt[0]) / (1.0 - targetIncreasePost)

        self.mode = mode
        self.modes = [self.eval_both_clusterModel]

        # Minimum threshold for a "viable" solution
        self.threshold = -500


    # -----------------------------
    #     ODE system definitions
    # -----------------------------
    


    


    def munc13_clusterOde(self, t, y, params):
        """
        Maggie Johnson, Oct 2025
        Munc13 model with explicit clustering. 
        We will no longer enforce a specific cluster regime, so we want the
        density of the clusters to match those observed experimentally.

        y[0]  = S, Munc13 in solution
        y[1]  = R, General recruiter on the membrane
        y[2]  = M, Munc13 on the membrane (out of the cluster)
        y[3]  = D, Munc13 dimer on the membrane (out of the cluster)
        y[4]  = X, New--Cluster nucleator, also in 2D but dilute (well-mixed)
        y[5]  = MX, Munc13 on the membrane + X 
        y[6]  = M2X, two muncs+ X
        y[7]  = M3X, three muncs+X
        y[8]  = M3X2, three muncs + two X
        y[9]  = M3X3, three muncs + three X
        y[10] = M3X2M, three muncs + two X + one M
        y[11] = M3X2M2, three muncs + two X + two M
        y[12] = M3X2M3, three muncs + two X + three M
        y[13] = M3X3M, three muncs + three X + one M
        y[14] = M3X3M2, three muncs + three X + two M
        y[15] = M3X3M3, three muncs + three X + three M
        y[16] = M3X3M3M, three muncs + three X + three M+one M
        y[17] = M3X3M3M2, three muncs + three X + three M+two M
        y[18] = M3X3M3M3, three muncs + three X + three M+three M
        y[19]  = SX, Munc13 on the membrane + X 
        y[20]  = S2X, two muncs+ X
        y[21]  = S3X, three muncs+X
        y[22]  = S3X2, three muncs + two X
        y[23]  = S3X3, three muncs + three X
        y[24] = S3X2S, three muncs + two X + one M
        y[25] = S3X2S2, three muncs + two X + two M
        y[26] = S3X2S3, three muncs + two X + three M
        y[27] = S3X3S, three muncs + three X + one M
        y[28] = S3X3S2, three muncs + three X + two M
        y[29] = S3X3S3, three muncs + three X + three M
        y[30] = S3X3S3S, three muncs + three X + three M+one M
        y[31] = S3X3S3S2, three muncs + three X + three M+two M
        y[32] = S3X3S3S3, three muncs + three X + three M+three M
        y[33]  = MSX, two muncs+ X
        y[34]  = M2SX, three muncs+X
        y[35]  = M2SX2, three muncs + two X
        y[36]  = M2SX3, three muncs + three X
        y[37] = M2SX2M, three muncs + two X + one M
        y[38] = M2SX2M2, three muncs + two X + two M
        y[39] = M2SX2M3, three muncs + two X + three M
        y[40] = M2SX3M, three muncs + three X + one M
        y[41] = M2SX3M2, three muncs + three X + two M
        y[42] = M2SX3M3, three muncs + three X + three M
        y[43] = M2SX3M3M, three muncs + three X + three M+one M
        y[44] = M2SX3M3M2, three muncs + three X + three M+two M
        y[45] = M2SX3M3M3, three muncs + three X + three M+three M

        kfsr, krsr is for recruitment to the membrane
        kfmm, krmm is for dimerization of munc13
        kf1x, kr1x is for binding to the cluster nucleator X of a monomer
        kfc, krc is for binding of a monomer to the cluster nucleator X with a monomer already present
        krmmTrimer is the off-rate for a monomer from the trimer.
        kx2, krx2 is the rate of adding another nucleator X to the cluster
        krxTrimer is the off-rate of X from the trimer.

        Reaction equations:
        1-19
            S + R <-> M, kfsr, krsr
            M + M <-> D, gamma*kfmm, krmm
            S + X <-> SX, kf1x, kr1x
            M + X <-> MX, gamma*kf1x, kr1x
            D + X -> M2X, 2*gamma*kf1x
            SX + R <-> MX, gamma*kfsr, krsr
            M + MX <-> M2X, gamma*kfc, krc
            M + M2X <-> M3X, gamma*kfc, krmmTrimer
            M3X + X <-> M3X2, gamma*kx2, krx2
            M3X2 + X <-> M3X3, gamma*kx2, krxTrimer
            M3X2 + M <-> M3X2M, gamma*kf1x, kr1x
            M3X2M + M <-> M3X2M2, gamma*kfc, krc
            M3X2M2 + M <-> M3X2M3, gamma*kfc, krmmTrimer
            M3X3 + M <-> M3X3M, gamma*kf1x, kr1x
            M3X3M + M <-> M3X3M2, gamma*kfc, krc
            M3X3M2 + M <-> M3X3M3, gamma*kfc, krmmTrimer
            M3X3M3 + M <-> M3X3M3M, gamma*kf1x, kr1x
            M3X3M3M + M <-> M3X3M3M2, gamma*kfc, krc
            M3X3M3M2 + M <-> M3X3M3M3, gamma*kfc, krmmTrimer
            repeated for dimer recruitmen
            20-30
            D + MX -> M3X, 2*gamma*kfc
            D + M2X -> M3X+M, 2*gamma*kfc
            M3X2 + D -> M3X2M2, 2*gamma*kf1x
            M3X2M + D -> M3X2M3, 2*gamma*kfc
            M3X2M2 + D -> M3X2M3+M, 2*gamma*kfc
            M3X3 + D -> M3X3M2, 2*gamma*kf1x
            M3X3M + D -> M3X3M3, 2*gamma*kfc
            M3X3M2 + D -> M3X3M3M, 2*gamma*kfc
            M3X3M3 + D -> M3X3M3M2, 2*gamma*kf1x
            M3X3M3M + D -> M3X3M3M3, 2*gamma*kfc
             M3X3M3M2 + D -> M3X3M3M3+M, 2*gamma*kfc. r30
            
            Now recruit S to the cluster, with R at the same time.
            31-41
            S+R + MX -> M2X, gamma*kfsr*kfc/krsr
            S+R + M2X -> M3X, gamma*kfsr*kfc/krsr
            M3X2 + S+R -> M3X2M, gamma*kfsr*kfc/krsr
            M3X2M + S+R -> M3X2M2, gamma*kfsr*kfc/krsr
            M3X2M2 + S+R -> M3X2M3, gamma*kfsr*kfc/krsr
            M3X3 + S+R -> M3X3M, gamma*kfsr*kfc/krsr
            M3X3M + S+R -> M3X3M2, gamma*kfsr*kfc/krsr
            M3X3M2 + S+R -> M3X3M3, gamma*kfsr*kfc/krsr
            M3X3M3 + S+R -> M3X3M3M, gamma*kfsr*kfc/krsr
            M3X3M3M + S+R -> M3X3M3M2, gamma*kfsr*kfc/krsr
            M3X3M3M2 + S+R -> M3X3M3M3, gamma*kfsr*kfc/krsr

            repeated for S recruitment
            S + SX <-> S2X, kfc, krc
            S + S2X <-> S3X, kfc, krmmTrimer
            S3X + X <-> S3X2, kx2, krx2
            S3X2 + X <-> S3X3, kx2, krxTrimer
            S3X2 + S <-> S3X2S, kf1x, kr1x
            S3X2S + S <-> S3X2S2, kfc, krc
            S3X2S2 + S <-> S3X2S3, kfc, krmmTrimer
            S3X3 + S <-> S3X3S, kf1x, kr1x
            S3X3S + S <-> S3X3S2, kfc, krc
            S3X3S2 + S <-> S3X3S3, kfc, krmmTrimer
            S3X3S3 + S <-> S3X3S3S, kf1x, kr1x
            S3X3S3S + S <-> S3X3S3S2, kfc, krc
            S3X3S3S2 + S <-> S3X3S3S3, kfc, krmmTrimer
            Next we need to add in reactions for the S clusters to recruit R in bulk, and thus no mixed clusters exist

             kfsr, krsr is for recruitment to the membrane
            kfmm, krmm is for dimerization of munc13
            kf1x, kr1x is for binding to the cluster nucleator X of a monomer
            kfc, krc is for binding of a monomer to the cluster nucleator X with a monomer already present
            krmmTrimer is the off-rate for a monomer from the trimer.
            kx2, krx2 is the rate of adding another nucleator X to the cluster
            krxTrimer is the off-rate of X from the trimer.
        """
        kfsr = params[0]
        krsr = params[1]
        gamma = self.gamma
        kfmm = params[2]
        krmm = params[3]
        kf1x = params[4]
        kr1x = params[5]
        kfc = params[6]
        krc = params[7]
        kd = krc/kfc*1e-6; #units of M required.
        krmmTrimer = krc*kd; #units of 1/s with c0=1M divided out


        krmmTrimer = krmmTrimer*params[8] #this is the cooperativity factor
        kx2 = params[9]
        krx2 = params[10]
        kdx=kx2/krx2*1e-6; #units of M required.
        krxTrimer=krx2*kdx; #units of 1/s with c0=1M divided out
        #krxTrimer = params[12]
        enhanceRate = 10 #scalare that increase the on-rate when more than one X is present

        r1 = kfsr*y[0]*y[1]
        r1b = krsr*y[2]
        r2 = gamma*kfmm*y[2]*y[2]
        r2b= krmm*y[3]
        r3 = kf1x*y[0]*y[4]
        r3b = kr1x*y[19]
        r4 = gamma*kf1x*y[2]*y[4]
        r4b = kr1x*y[5]
        r5 = 2*gamma*kf1x*y[3]*y[4]#goes to 5
        r6 = gamma*kfsr*y[19]*y[1]
        r6b = krsr*y[5]
        r7 = gamma*kfc*y[2]*y[5]
        r7b = krc*y[6]
        r8 = gamma*kfc*y[2]*y[6]
        r8b = krmmTrimer*y[7]
        r9 = gamma*kx2*y[7]*y[4]
        r9b = krx2*y[8]
        r10 = gamma*kx2*y[8]*y[4]
        r10b = krxTrimer*y[9]
        r11 = gamma*kf1x*y[8]*y[2]*enhanceRate
        r11b = kr1x*y[10]
        r12 = gamma*kfc*y[10]*y[2]*enhanceRate
        r12b = krc*y[11]
        r13 = gamma*kfc*y[11]*y[2]*enhanceRate
        r13b = krmmTrimer*y[12]
        r14 = gamma*kf1x*y[9]*y[2]*enhanceRate
        r14b = kr1x*y[13]
        r15 = gamma*kfc*y[13]*y[2]*enhanceRate
        r15b = krc*y[14]
        r16 = gamma*kfc*y[14]*y[2]*enhanceRate
        r16b = krmmTrimer*y[15]
        r17 = gamma*kf1x*y[15]*y[2]*enhanceRate
        r17b = kr1x*y[16]
        r18 = gamma*kfc*y[16]*y[2]*enhanceRate
        r18b = krc*y[17]
        r19 = gamma*kfc*y[17]*y[2]*enhanceRate
        r19b = krmmTrimer*y[18]
        r20 = 2*gamma*kfc*y[3]*y[5]
        r21 = 2*gamma*kfc*y[3]*y[6]
        r22 = 2*gamma*kf1x*y[8]*y[3]*enhanceRate
        r23 = 2*gamma*kfc*y[10]*y[3]*enhanceRate
        r24 = 2*gamma*kfc*y[11]*y[3]*enhanceRate
        r25 = 2*gamma*kf1x*y[9]*y[3]*enhanceRate
        r26 = 2*gamma*kfc*y[13]*y[3]*enhanceRate
        r27 = 2*gamma*kfc*y[14]*y[3]*enhanceRate
        r28 = 2*gamma*kf1x*y[15]*y[3]*enhanceRate
        r29 = 2*gamma*kfc*y[16]*y[3]*enhanceRate
        r30 = 2*gamma*kfc*y[17]*y[3]*enhanceRate
        #reactions with S+R
        rateSR=gamma*kfsr*kfc/krsr
        rateSRX=gamma*kfsr/krsr*kf1x

        r31 = rateSR*y[0]*y[1]*y[5]
        r32 = rateSR*y[0]*y[6]*y[1]
        r33 = rateSRX*y[8]*y[0]*y[1]#*enhanceRate
        r34 = rateSR*y[10]*y[0]*y[1]#*enhanceRate
        r35 = rateSR*y[11]*y[0]*y[1]#*enhanceRate
        r36 = rateSRX*y[9]*y[0]*y[1]#*enhanceRate
        r37 = rateSR*y[13]*y[0]*y[1]#*enhanceRate
        r38 = rateSR*y[14]*y[0]*y[1]#*enhanceRate
        r39 = rateSRX*y[15]*y[0]*y[1]#*enhanceRate
        r40 = rateSR*y[16]*y[0]*y[1]#*enhanceRate
        r41 = rateSR*y[17]*y[0]*y[1]#*enhanceRate
        
        #for now leave out recruitment of S to the clusters.
        #alternatively, make it a triple reactino--so R is also required.
        #the on-rate will be gamma*kfc*kfr/kbr. Reverse is then just M falling off

        #Also currently leave out the fully S bound state, in the absence of R.

         




        dylist = []
        #add each new species reaction rate.
        dylist.append(-r1+r1b-r3+r3b-r31-r32-r33-r34-r35-r36-r37-r38-r39-r40-r41)#0
        dylist.append(-r1+r1b-r6+r6b-r31-r32-r33-r34-r35-r36-r37-r38-r39-r40-r41)#1
        dylist.append(+r1-r1b-2*r2+2*r2b-r4+r4b-r7+r7b-r8+r8b-r11+r11b-r12+r12b-r13+r13b-r14+r14b-r15+r15b-r16+r16b-r17+r17b-r18+r18b-r19+r19b+r21+r24+r30)#2
        dylist.append(+r2-r2b-r5-r20-r21-r22-r23-r24-r25-r26-r27-r28-r29-r30)#3
        dylist.append(-r3+r3b-r4+r4b-r5-r9+r9b-r10+r10b)#4
        dylist.append(+r4-r4b+r6-r6b-r7+r7b-r20-r31)#5
        dylist.append(+r5+r7-r7b-r8+r8b-r21-r32+r31)#6 M2X
        dylist.append(r8-r8b-r9+r9b+r20+r21+r32)#7 M3X
        dylist.append(r9-r9b-r10+r10b-r11+r11b-r22-r33)#8
        dylist.append(r10-r10b-r14+r14b-r25-r36)#9 M3X3
        dylist.append(r11-r11b-r12+r12b-r23-r34+r33)#10 M3X2M
        dylist.append(r12-r12b-r13+r13b-r24+r22-r35+r34)#11 M3X2M2
        dylist.append(r13-r13b+r23+r24+r35)#12 M3X2M3
        dylist.append(r14-r14b-r15+r15b-r26-r37+r36)#13 M3X3M
        dylist.append(r15-r15b-r16+r16b-r27+r25-r38+r37)#14 M3X3M2
        dylist.append(r16-r16b-r17+r17b-r28+r26-r39+r38)#15 M3X3M3
        dylist.append(r17-r17b-r18+r18b-r29+r27-r40+r39)#16 M3X3M3M
        dylist.append(r18-r18b-r19+r19b-r30+r28-r41+r40)#17 M3X3M3M2
        dylist.append(r19-r19b+r29+r30+r41)#18 M3X3M3M3
        dylist.append(+r3-r3b-r6+r6b)#19 SX



        return np.array(dylist)
    # ------------------------------------------------------
    #        Utilities to compute Munc13 on cluster/membrane
    # ------------------------------------------------------
    def calculate_munc13_on_membrane(self, copies):
        """
        Munc13 in cluster (copies/um^2) = C/Acluster
        Munc13 on the membrane but not in cluster (copies/um^2) = (M + 2*D)/Atotal
        """
      
        memMunc = (copies[2] + 2 * copies[3]+ copies[5] + 2*copies[6]+ 3*copies[7]+\
                    3*copies[8]+ 3*copies[9]+ 4*copies[10]+ 5*copies[11]+ 6*copies[12]+\
                          4*copies[13]+ 5*copies[14]+ 6*copies[15]+ 7*copies[16]+\
                              8*copies[17]+ 9*copies[18]+ copies[19]) 
        densMunc=memMunc/self.cellArea
        return densMunc


   

    

    

    def compute_background_from_target_increase(self, target_increase):
        """
        Compute background (delta) such that:
            (I_final - delta) / (I_0 - delta) = target_increase
        """
        I0 = self.expdata.expInt.iloc[0]
        If = self.expdata.expInt.iloc[-1]
        delta = (If - target_increase * I0) / (1.0 - target_increase)
        return delta


    
    
    
     # --------------------------------------------
    #          Cost (Chi) and viability checks
    # --------------------------------------------
    def costChi_cluster(self, Y, postBool):
        """
        Compute a chi-square-like cost for the "pre" data.
        Based on the cluster forming ODE model.
        Y is the solution in vector form, so all species are in units of uM. 
        if postBool is True, evaluate for post-stim condition.
        if postBool is False, evaluate for pre-stim condition.
        """
        #Calculate the density of clusters on the membrane
        #all of these are time-dependent arrays, we want only the final steady-state values.
        density=self.calculate_cluster_density(Y)
        #we want to add in a term that penalizes a high numbers
        #of small clusters on the membrane.
        #these are clusters that have nucleated with X, but have less than 6 copies
        smallClusterDens=self.calculate_small_cluster_density(Y)
        #We want this to be close to zero, and not close to -1. 
        #due to numerical precision, the density of the clusters can be slightly negative
        #So, we need to force them to zero such that chi is always negative
        
        chiSmallClust=-smallClusterDens[-1]/(density[-1]+1)
        if(density[-1]<0):
            chiSmallClust=self.threshold*10 #prevent 0 densities from being favorable.
            print(f"WARNING: density of clusters is negative {density[-1]}")
        elif(smallClusterDens[-1]<0):
            chiSmallClust=0 #in this case density is positive but small clusters are at 0.
            print(f"WARNING2: density of small clusters is negative {smallClusterDens[-1]}")
        

        

        if postBool:
            #evaluate for the post-stim cluster density
            densExpt=self.density_exp_post
            chiDens=self.costChi_densityCluster(density[-1],densExpt)
            
        else:
            densExpt=self.density_exp_pre
            chiDens=self.costChi_densityCluster(density[-1],densExpt)

        
        costSum=0
        #we may need to add in a term to the cost function to select
        #against a high density of sub-clusters on the membrane,
        #that would be all species with say 3-6 munc13.
        #this is called chiSmallClust, we can weight it less than the experimental comparison.
        weightSmallClust=0.1
        print(f"Small cluster density chi: Simulated {smallClusterDens[-1]}, high density {density[-1]}, Chi {chiSmallClust}")
        #calculate the percent of munc13 on the membrane that is in clusters.
        percClusterTotal, percClustMem=self.calc_percentages_cluster(Y)
        print(f"Percent of munc13 in clusters: {percClustMem*100}%")
        #implement a relu penalty if percent in clusters is greater than 40%
        chiPercClust=max(0, percClustMem-0.4)*-10

        costSum=chiDens[0]+weightSmallClust*chiSmallClust + chiPercClust # this is already negative
        return [costSum]


    def calculate_cluster_density(self,Y):
        """
        given the solution of the cluster model, compute the density of clusters on the membrane
        """
        copies = Y *self.cellVolume * 602 #converts from uM to copy numbers.
        #we have to define what is a cluster. We can say that it requires 3 copies of X.
        #and at least 6 copies of munc13 (either S or M)
        cluster_copies = copies[15]+copies[16]+copies[17]+copies[18]+copies[12]
               #now convert to density
        memDensity = cluster_copies/(self.cellArea) #
        return memDensity
    



    def calculate_small_cluster_density(self,Y):
        """
        given the solution of the cluster model, compute the density of clusters on the membrane
        """
        copies = Y *self.cellVolume * 602 #converts from uM to copy numbers.
        #This is now the inverse of the cluster density,
        # it is all species that have nucleated with X, but are too small
        # so they have less than 6 copies of munc13 (either S or M)
        cluster_copies = copies[5]+copies[6]+copies[7]+copies[8]+copies[9]+\
                        copies[10]+copies[11]+copies[13]+copies[14]
               #now convert to density
        memDensity = cluster_copies/(self.cellArea) #
        return memDensity


    def costChi_densityCluster(self, simDens, expDens):
        """
        Compute a chi-square-like cost for the cluster density.
        We are passing in the computed densities from the simulation.
        The target densities are also passed in.
        Use only the steady state value or final time of the simDens.
        """
        
        # Calculate differences for the final density
        diff1 = (simDens - expDens) ** 2
       
        # Weight prefactors to normalize the terms
        # Normalize by the square of expected values to make terms dimensionless and comparable
        weight1 = 1.0 / (expDens ** 2)
       
        sum_diff = weight1 * diff1 
        print(f"Cluster density chi: Simulated {simDens}, Experimental {expDens}, Chi { -sum_diff}")
        #we want to maximize the fitness, so return the negative value.
        return [-sum_diff]

    def costChi_recruitmentStim(self, Ypre, Ypost):
        """
        Compute a chi-square-like cost for the increase in the 
        amount of munc13 recruited to the membrane upon stimulation.
        This should be based on total munc13, in the clustered and nonclustered phases.
        """
        copiesPre = Ypre * self.cellVolume * 602 #converts from uM to copy numbers.
        #calculate the total density of munc13 on the membrane
        memDensMuncPre =   self.calculate_munc13_on_membrane(copiesPre)

        copiesPost = Ypost * self.cellVolume * 602 #converts from uM to copy numbers.
        #calculate the total density of munc13 on the membrane
        memDensMuncPost =   self.calculate_munc13_on_membrane(copiesPost)

        #ratio of densities on membrane at steady-state.
        recruitStim=memDensMuncPost[-1]/memDensMuncPre[-1]

        # Calculate differences just for the last point.
        diff1 = (recruitStim - self.recruitmentStim) ** 2
       
        # Weight prefactors to normalize the terms
        # Normalize by the square of expected values to make terms dimensionless and comparable
        weight = 1.0 / (self.recruitmentStim ** 2)
       
        sum_diff = weight * diff1
        print(f"Recruitment upon stimulation, sim value {recruitStim}, exp value {self.recruitmentStim}, chi: Chi {-sum_diff}")
        return [-sum_diff]

    def isViableFitness(self, fit):
        return fit >= self.threshold


    def isViable(self, point):
        fitness = self.eval_both_clusterModel(point)
        return self.isViableFitness(fitness[0])


    # ---------------------------------------------------
    #   Main evaluation: combine pre and post conditions
    # ---------------------------------------------------
  

    def eval_both_clusterModel(self, candidate):
        """
        Simulate the model in pre and post conditions
        Evaluate the candidate solution to compute fitness
        Evaluate the fitness using the chi terms

        Include the evaluation of the different mutants.
        """
        solutionPre, solutionPost = self.simulate(candidate)

        # Chi from pre
        chiPre = self.costChi_cluster(solutionPre, False)
        # Chi from post
        chiPost = self.costChi_cluster(solutionPost, True)

        chiRecruitStim= self.costChi_recruitmentStim(solutionPre,solutionPost)
        
        chiTotal = chiPre[0] + chiPost[0]+chiRecruitStim[0]

        
        

        # Now evaluate the mutants, which will require new simulations
        '''
        # diffusion constants of mutant chi
        candidate_dc2a = list(candidate)
        candidate_dc2a[3] = 0
        _, _, _, _, DpreDC2A, DpostDC2A = self.simulate(candidate_dc2a)
        
        chiDiffusionDC2A = 250*self.costChiDiffusionDC2A(DpreDC2A, DpostDC2A)
        
        chiTotal += chiDiffusionDC2A[0]
        
        candidate_shRIM = list(candidate)
        candidate_shRIM[3] *= 2
        _, _, _, _, DpreshRIM, DpostshRIM = self.simulate(candidate_shRIM)
        
        chiDiffusionshRIM = 250*self.costChiDiffusionshRIM(DpreshRIM, DpostshRIM)
        
        chiTotal += chiDiffusionshRIM[0]
        
        candidate_dc2b = list(candidate)
        candidate_dc2b[0] *= 0.5
        _, _, _, _, DpreDC2B, DpostDC2B = self.simulate(candidate_dc2b)
        
        chiDiffusionDC2B = 250*self.costChiDiffusionDC2B(DpreDC2B, DpostDC2B)
        
        chiTotal += chiDiffusionDC2B[0]
        '''
        if(chiTotal > 0):
            print(f"Positive chi found! {chiTotal}")
            print(f"Parameters: {candidate}")
            print(f"Chi pre: {chiPre}, Chi post: {chiPost}, Chi recruit: {chiRecruitStim}")
            print("ERROR")
            print("-------ERROR------")
            return [self.threshold*10]
        
        print(f"Total chi: {chiTotal}")

        return [chiTotal]
    # -----------------------------------------
    #      ODE simulation (pre/post)
    # -----------------------------------------
    def simulate_pre(self, candidate):
        """
   
        The candidate contains the sampled parameters.
        Their order is defined in the array "params_to_optimize"
        So it does not need to match the order in the ODE function.
        The list of params below needs to be passed to the ODE funtion, however

        Even if you put constraints on the candidates to be optimized, the ODE still takes in the same
        list of parameters. So 'candidate' can change shape, but the ODE parameters should not.

        """
        kfsr=candidate[0] #form SR
        krsr=candidate[1] #dissociate SR
        kfmm=candidate[2] #form D
        krmm=candidate[3] #dissociate D
        kf1x=candidate[4] #form MX
        kr1x=candidate[5] #dissociate MX
        kfc=candidate[6] #bind to cluster with M and X present
        krc=candidate[7] #reverse of above
        krmmTriCoop=candidate[8] #off rate from trimer
        kx2=candidate[9] #add second X to trimer
        krx2=candidate[10] #off rate of second X from trimer
       # krxTrimer=candidate[12] #off rate of third X from trimer
        S0=candidate[11] #initial Solution Munc13 (S), uM
        R0=candidate[12] #initial R, /um^2
        #D1=candidate[9] #Monomer (M) diffusion constant on membrane, um^2/s
        #D2=candidate[9] / candidate[10] #Dimer (D) diffusion constant on membrane, um^2/s
        X0 = candidate[13] #initial X /um^2
        # convert to uM
        R0 = R0*self.cellArea/self.cellVolume/602.0 
        X0 = X0*self.cellArea/self.cellVolume/602.0 

        rateParams=np.array([kfsr, krsr, kfmm, krmm, kf1x, kr1x, kfc, krc, krmmTriCoop, kx2, krx2])
        

        initValues=np.zeros(20)
        initValues[0]=S0
        initValues[1]=R0
        initValues[4]=X0
        
        solution = scipy.integrate.solve_ivp(fun=self.munc13_clusterOde, method='BDF', t_span=(0, self.t_max), y0=initValues, t_eval=self.timePoints, args=(rateParams,), rtol=1e-7, atol=1e-9)

        #D = self.calc_diffusion(D1, D2, solution.y)

        return solution.y


    def simulate_post(self, candidate):
        """
   
        The candidate contains the sampled parameters.
        Their order is defined in the array "params_to_optimize"
        So it does not need to match the order in the ODE function.
        The list of params below needs to be passed to the ODE funtion, however
        Even if you put constraints on the candidates to be optimized, the ODE still takes in the same
        list of parameters. So 'candidate' can change shape, but the ODE parameters should not.

        """
        kfsr=candidate[0]*candidate[16] #form SR
        krsr=candidate[1] #dissociate SR
        kfmm=candidate[2] #form D
        krmm=candidate[3] #dissociate D
        kf1x=candidate[4] #form MX
        kr1x=candidate[5] #dissociate MX
        kfc=candidate[6]*candidate[15] #bind to cluster with M and X present
        krc=candidate[7] #reverse of above
        krmmTriCoop=candidate[8] #off rate from trimer
        kx2=candidate[9] #add second X to trimer
        krx2=candidate[10] #off rate of second X from trimer
       # krxTrimer=candidate[12] #off rate of third X from trimer
        S0=candidate[11] #initial Solution Munc13 (S), uM
        R0=candidate[12] #initial R, /um^2
        #D1=candidate[9] #Monomer (M) diffusion constant on membrane, um^2/s
        #D2=candidate[9] / candidate[10] #Dimer (D) diffusion constant on membrane, um^2/s
        #Following stimulation, here we allow X0 to increase.
        X0 = candidate[13]*candidate[14] #initial X /um^2
        # convert to uM
        R0 = R0*self.cellArea/self.cellVolume/602.0 
        X0 = X0*self.cellArea/self.cellVolume/602.0 

        rateParams=np.array([kfsr, krsr, kfmm, krmm, kf1x, kr1x, kfc, krc, krmmTriCoop, kx2, krx2])
        

        initValues=np.zeros(20)
        initValues[0]=S0
        initValues[1]=R0
        initValues[4]=X0
        
        solution = scipy.integrate.solve_ivp(fun=self.munc13_clusterOde, method='BDF', t_span=(0, self.t_max), y0=initValues, t_eval=self.timePoints, args=(rateParams,), rtol=1e-7, atol=1e-9)

        #D = self.calc_diffusion(D1, D2, solution.y)

        return solution.y


    def simulate(self, candidate):
        """
        Wrapper to run both pre- and post-stimulation solvers.
        """
        solutionPre = self.simulate_pre(candidate)
        solutionPost = self.simulate_post(candidate)
        return [solutionPre, solutionPost]


    def calc_diffusion(self, D1, D2, solution):
        """Calculate the average of Munc13 on membrane diffusion constants"""
        D_average = (solution[2] * D1 + 2*solution[3] * D2) / (solution[2] + 2*solution[3] + solution[5] + solution[7])

        return D_average[-1]

    def test_mass_conservation(self, Y):
        """Test that the total mass of Munc13 is conserved in the ODE solution"""
        copies = Y * self.cellVolume * 602
        totalMunc=copies[0][0]
        totalR=copies[1][0]
        totalX=copies[4][0]
        #calculate the total number of Munc13 in the system at each time point
        muncVsTime=copies[0]+copies[2]+2*copies[3]+\
                    copies[5]+2*copies[6]+3*copies[7]+\
                    3*copies[8]+3*copies[9]+4*copies[10]+\
                    5*copies[11]+6*copies[12]+\
                    4*copies[13]+5*copies[14]+6*copies[15]+\
                    7*copies[16]+8*copies[17]+9*copies[18]+\
                    copies[19]
        #calculate the total number of R in the system at each time point
        RvsTime=copies[1]+muncVsTime-copies[0]-copies[19] 
        #calculate the total number of X in the system at each time point
        XvsTime=copies[4]+copies[5]+copies[6]+copies[7]+2*copies[8]+3*copies[9]+\
                    2*copies[10]+2*copies[11]+2*copies[12]+\
                    3*copies[13]+3*copies[14]+3*copies[15]+\
                    3*copies[16]+3*copies[17]+3*copies[18]+\
                    copies[19]
        
        
        print(f"Initial total Munc13 {totalMunc}, final total Munc {muncVsTime[-1]}")
        print(f"Initial total R {totalR}, final total R {RvsTime[-1]}")
        print(f"Initial total X {totalX}, final total X {XvsTime[-1]}")

    def plot_freespecies_time(self, Y, figsize=(4, 3), fontsize=12, dpi=300):
        """Plot the time course of the cluster model solution"""
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        copies= Y * self.cellVolume * 602  # converts from uM to copy numbers.
        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, copies[0], linestyle="-", label="Solution copies", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[1], linestyle="-", label="Recruiter copies", color=c_post, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[2], linestyle="-", label="Membrane monomer", color="blue", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[3], linestyle="-", label="Membrane dimer", color="cyan", alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
       #
        ax.set_ylabel("Copy numbers")
        #ax.set_xlim(0, 10)
        #ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        #fig.savefig("../fig/fig_total_recruitment_wt.svg") 
        #fig.savefig("../fig/fig_total_recruitment_wt.png", dpi=dpi) 
        plt.show()

    def plot_mycluster_time(self, Y, figsize=(4, 3), fontsize=12, dpi=300):
        """Plot the time course of the cluster model solution"""
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        copies= Y * self.cellVolume * 602  # converts from uM to copy numbers.
        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, copies[12]+copies[15]+copies[16]+copies[17]+copies[18], linestyle="-", label="copies of clusters", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, 6*copies[12]+6*copies[15]+7*copies[16]+8*copies[17]+9*copies[18], linestyle="-", label="munc in larger clusters", color="green", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[5], linestyle="-", label="cluster monomers", color=c_post, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[5]+copies[6]+copies[7]+copies[8]+copies[9]+copies[10]+copies[11]+copies[13]+copies[14], linestyle="-", label="copies of small clusters", color="blue", alpha=0.95, zorder=3)
        
        #ax.plot(self.timePoints, copies[2], linestyle="-", label="Membrane monomer", color="blue", alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, copies[3], linestyle="-", label="Membrane dimer", color="cyan", alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
       #
        ax.set_ylabel("Copy numbers")
        #ax.set_xlim(0, 10)
        #ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        #fig.savefig("../fig/fig_total_recruitment_wt.svg") 
        #fig.savefig("../fig/fig_total_recruitment_wt.png", dpi=dpi) 
        plt.show()
    
    def plot_each_cluster_time(self, Y, figsize=(5, 4), fontsize=12, dpi=300):
        """Plot the time course of the cluster model solution"""
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        copies= Y * self.cellVolume * 602  # converts from uM to copy numbers.
        #plt.figure()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, copies[2], linestyle="-", label="Monomer on membrane", color="black", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[3], linestyle="-", label="dimer on membrane", color="green", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[5], linestyle="-", label="cluster 1", color="cyan", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[6], linestyle="-", label="cluster 2", color="blue", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[7], linestyle="-", label="cluster 3", color="pink", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[8], linestyle="-", label="cluster 3X2", color="purple", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[9], linestyle="-", label="cluster 3X3", color="gray", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[10], linestyle="-", label="cluster 4", color="lime", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[11], linestyle="-", label="cluster 5", color="gold", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[12], linestyle="-", label="cluster 6", color="olive", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[13], linestyle="-", label="cluster 4X3", color="yellow", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[14], linestyle="-", label="cluster 5X3", color="brown", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[15], linestyle="-", label="cluster 6X3", color="teal", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[16], linestyle="-", label="cluster 7X3", color="indigo", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[17], linestyle="-", label="cluster 8X3", color=(0.1, 0.1, 0.5), alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[18], linestyle="-", label="cluster 9X3", color=(0.1, 0.2, 0.8), alpha=0.95, zorder=3)
        
        #ax.plot(self.timePoints, copies[2], linestyle="-", label="Membrane monomer", color="blue", alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, copies[3], linestyle="-", label="Membrane dimer", color="cyan", alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
       #
        ax.set_ylabel("Copy numbers")
        #ax.set_xlim(0, 10)
        #ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        #fig.savefig("../fig/fig_total_recruitment_wt.svg") 
        #fig.savefig("../fig/fig_total_recruitment_wt.png", dpi=dpi) 
        plt.show(block=True)


    def calc_diffusion_dilute(self, D1, D2, solution):
        """Calculate the average of Munc13 in dilute diffusion constants"""
        D_average = (solution[2] * D1 + 2 * solution[3] * D2) / (solution[2] + 2 * solution[3])
        percentages_M = (solution[2]) / (solution[2] + 2 * solution[3])
        percentages_D = (2*solution[3]) / (solution[2] + 2 * solution[3])
        # print(f"Percentages of Munc13 in M: {percentages_M[-1]}, in D: {percentages_D[-1]}")

        return D_average[-1]


    def calc_percentages_cluster(self, solution):
        """Calcualte the percentage of Munc13 copies in cluster
            out of total Munc13 on membrane
            out of total Munc13 in system
            """
        copies=solution * self.cellVolume * 602
        clusterCopies=6*copies[15]+7*copies[16]+8*copies[17]+9*copies[18]+6*copies[12]
        memDens=self.calculate_munc13_on_membrane(copies)
        memCopies=memDens[-1]*self.cellArea
        totalCopies=np.sum(copies[-1])
        P_cluster=clusterCopies/totalCopies
        P_cluster_normMem=clusterCopies/memCopies
        

        return P_cluster[-1], P_cluster_normMem[-1]

    def calc_populations(self, solution):
        P_cluster = (solution[5] + solution[7]) / (solution[2] + 2 * solution[3] + solution[5] + solution[7])
        P_monomer = (solution[2]) / (solution[2] + 2 * solution[3] + solution[5] + solution[7])
        P_dimer = (2 * solution[3]) / (solution[2] + 2 * solution[3] + solution[5] + solution[7])

        return P_monomer[-1], P_dimer[-1], P_cluster[-1]
    
    def calc_3D_2D_percentages_cluster(self, solution):
        """Calcualte the percentage of Munc13 copies in cluster from 3D or 2D"""
        P_3D_average = (solution[5]) / (solution[5] + solution[7])
        P_2D_average = (solution[7]) / (solution[5] + solution[7])

        return P_3D_average[-1], P_2D_average[-1]

    def test_one_candidate(self):
        """For testing: simulate and plot one candidate solution"""
        candidate0 = [0.1, 10, 1.0, 10, 1, 10, 10, 0.1, 0.1, 100, 10, 0.5, 1000, 500, 3]
        candidate1 = [0.017382105557085994, 0.23534111769875998, 0.01809755627495464, 2.9069951706107138, 0.4118893484422218, 23.833737227748962, 3.2097323891634066, 7.61349042068835, 0.20456435773642842, 2.1055727878389083, 0.08637081907922083, 0.01918319760855923, 664.9476748277353, 20.641137609115198, 4.100426010572915]
        candidate2 = [0.036342205816724996, 229.34496283712963, 0.003411526300668634, 361.07621931704523, 0.05974152710656036, 3.243719173819602, 2.1426595404688658, 0.8442928496038155, 0.10030058793884863, 1.6505255606288565, 0.010406454389152936, 6.915039534428586, 224.83673900344007, 11.965218000934748, 2.280116783250255]
        sol = self.simulate_pre(candidate2)
        print(sol[0])

        print(self.timePoints[-1])
        self.test_mass_conservation(sol)
        self.plot_freespecies_time(sol, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_mycluster_time(sol, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_each_cluster_time(sol, figsize=(5, 4), fontsize=12, dpi=300)

        # calculate fitness of this solution
        fit = self.eval_both_clusterModel(candidate1)
        print("fitness of this candidate: ", fit)

    def filter_and_store_solutions(self, best=100, totalTime=30.0, dt=0.1):
        np.random.seed(42)
        self.t_max = totalTime
        n_time_points = int(totalTime / dt)
        self.timePoints = [i * dt for i in range(n_time_points + 1)]
    
        df = pd.read_csv("../data/optimizedParms_cluster.txt", sep=",", engine="python")
        df.columns = df.columns.str.strip()
        df = df.sort_values(by="Rank")
    
        param_cols = [col for col in df.columns if col not in ["Rank", "Fitness"]]
    
        qualified = []
        for _, row in df.iterrows():
            param_vector = [float(row[col]) for col in param_cols]
            intensRatio, intensRatioPost, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(param_vector)
            tmp = np.mean(post/pre)
            # print("Average munc13 on membrane, NOSTIM/STIM: ", tmp)
            if True:#preCluster[-1] >= 30 and tmp > 1.5:
                qualified.append(param_vector)
                if len(qualified) == best:
                    break
                else:
                    if len(qualified) % 10 == 0:
                        # print(f"{len(qualified)} solutions found")
                        pass
    
        self.filteredSolutions = qualified
        self.bestSolution = qualified[0] if qualified else None


    

    def plot_total_recruitment_wt(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
       # ax.plot(self.timePoints, pre / self.Atotal, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
       # ax.plot(self.timePoints, post / self.AtotalDense, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
       #
        ax.set_ylabel("Unc13A density on membrane\n(copies/um^2)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_total_recruitment_wt.svg") 
        fig.savefig("../fig/fig_total_recruitment_wt.png", dpi=dpi) 
        plt.show()
        
    def plot_cluster_recruitment_wt(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, preCluster, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, postCluster, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A copies in cluster")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 50)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_cluster_recruitment_wt.svg") 
        fig.savefig("../fig/fig_cluster_recruitment_wt.png", dpi=dpi) 
        plt.show()

    def plot_diffusivity_wt(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        fig, ax = plt.subplots(figsize=figsize)

        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        self.D_model_pre = d_pre
        self.D_model_post = d_post

        print("Experimental:", self.D_exp_pre, self.D_exp_post)
        print("Model:", self.D_model_pre, self.D_model_post)

        mutants = ['WT']  #, r'$\Delta$C2A', 'shRNA RIM2', r'$\Delta$C2B']

        # Data
        D_exp_pre   = [self.D_exp_pre,]  #   self.D_exp_DC2A_pre,   self.D_exp_shRIM_pre,   self.D_exp_DC2B_pre]
        D_exp_post  = [self.D_exp_post,]  #  self.D_exp_DC2A_post,  self.D_exp_shRIM_post,  self.D_exp_DC2B_post]
        D_model_pre  = [self.D_model_pre,]  #  self.D_model_DC2A_pre,  self.D_model_shRIM_pre,  self.D_model_DC2B_pre]
        D_model_post = [self.D_model_post,]  # self.D_model_DC2A_post, self.D_model_shRIM_post, self.D_model_DC2B_post]

        D_exp_pre_sem  = [self.D_exp_pre_sem, ]  #  self.D_exp_DC2A_pre_sem,  self.D_exp_shRIM_pre_sem,  self.D_exp_DC2B_pre_sem]
        D_exp_post_sem = [self.D_exp_post_sem, ]  # self.D_exp_DC2A_post_sem, self.D_exp_shRIM_post_sem, self.D_exp_DC2B_post_sem]

        n = len(mutants)

        # Layout: [Exp(NO), Exp(STIM)]   gap   [Model(NO), Model(STIM)]
        bar_width   = 0.18
        small_gap   = 0.10   # separation between Exp pair and Model pair
        group_gap   = 0.60   # separation between mutant groups
        group_width = 4*bar_width + small_gap

        centers = np.arange(n) * (group_width + group_gap)
        lefts = centers - group_width/2

        x_exp_no   = lefts + 0*bar_width
        x_exp_stim = lefts + 1*bar_width
        x_mod_no   = lefts + 2*bar_width + small_gap
        x_mod_stim = lefts + 3*bar_width + small_gap

        # Colors and styles
        c_no, c_stim = 'black', 'red'
        hatch_model = '///'

        # EXP: solid fills (with error bars)
        ax.bar(x_exp_no,   D_exp_pre,  width=bar_width, yerr=D_exp_pre_sem,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, D_exp_post, width=bar_width, yerr=D_exp_post_sem, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   D_model_pre,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, D_model_post, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels([], fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_diffusivity_wt.svg") 
        fig.savefig("../fig/fig_diffusivity_wt.png", dpi=dpi) 
        plt.show()

    def plot_population_wt(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate)
        P_monomer_pre, P_dimer_pre, P_cluster_pre = self.calc_populations(solutionPre)
        P_monomer_post, P_dimer_post, P_cluster_post = self.calc_populations(solutionPost)

        print("Population NO STIM: Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_pre*100, P_dimer_pre*100, P_cluster_pre*100))
        print("Population STIM:    Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_post*100, P_dimer_post*100, P_cluster_post*100))

        # plot two pie charts side by side
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        labels = ['M', 'D', 'C']
        sizes_pre = [P_monomer_pre, P_dimer_pre, P_cluster_pre]
        sizes_post = [P_monomer_post, P_dimer_post, P_cluster_post]
        colors = ['#ff9999','#66b3ff','#99ff99']
        explode = (0.05, 0.05, 0.05)  # explode all slices slightly
        wedges1, texts1, autotexts1 = ax[0].pie(sizes_pre, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        wedges2, texts2, autotexts2 = ax[1].pie(sizes_post, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        # draw circle for donut shape
        centre_circle0 = plt.Circle((0,0), 0.70, fc='white')
        centre_circle1 = plt.Circle((0,0), 0.70, fc='white')
        ax[0].add_artist(centre_circle0)
        ax[1].add_artist(centre_circle1)
        ax[0].set_title('NO STIM', fontsize=fontsize)
        ax[1].set_title('STIM', fontsize=fontsize)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax[0].axis('equal')  
        ax[1].axis('equal')
        fig.tight_layout()
        fig.savefig("../fig/fig_population_wt.svg") 
        fig.savefig("../fig/fig_population_wt.png", dpi=dpi)
        plt.show()

    def plot_total_recruitment_dc2a(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c3 = candidate[3]
        candidate[3] = 0  # Set kfmm to 0 for deltaC2A scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[3] = c3

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
       # ax.plot(self.timePoints, pre / self.Atotal, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, post / self.AtotalDense, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A density on membrane\n(copies/$\mu$m$^2$)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_total_recruitment_dc2a.svg") 
        fig.savefig("../fig/fig_total_recruitment_dc2a.png", dpi=dpi) 
        plt.show()

    def plot_cluster_recruitment_dc2a(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        
        c3 = candidate[3]
        candidate[3] = 0  # Set kfmm to 0 for deltaC2A scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[3] = c3

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, preCluster, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, postCluster, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A copies in cluster")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 50)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_cluster_recruitment_dc2a.svg") 
        fig.savefig("../fig/fig_cluster_recruitment_dc2a.png", dpi=dpi) 
        plt.show()

    def plot_diffusivity_dc2a(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        fig, ax = plt.subplots(figsize=figsize)

        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        self.D_model_pre = d_pre
        self.D_model_post = d_post

        c3 = candidate[3]
        candidate[3] = 0  # Set kfmm to 0 for deltaC2A scenario
        solutionPre, solutionPost, _, _, d_pre, d_post = self.simulate(candidate)

        self.D_model_DC2A_pre = d_pre
        self.D_model_DC2A_post = d_post

        candidate[3] = c3

        print("Experimental:", self.D_exp_DC2A_pre, self.D_exp_DC2A_post)
        print("Model:", self.D_model_DC2A_pre, self.D_model_DC2A_post)

        mutants = ['WT', r'$\Delta$C2A',] # 'shRNA RIM2', r'$\Delta$C2B']

        # Data
        D_exp_pre   = [self.D_exp_pre, self.D_exp_DC2A_pre,] #   self.D_exp_shRIM_pre,   self.D_exp_DC2B_pre]
        D_exp_post  = [self.D_exp_post, self.D_exp_DC2A_post,] #  self.D_exp_shRIM_post,  self.D_exp_DC2B_post]
        D_model_pre  = [self.D_model_pre, self.D_model_DC2A_pre,] #  self.D_model_shRIM_pre,  self.D_model_DC2B_pre]
        D_model_post = [self.D_model_post, self.D_model_DC2A_post,] # self.D_model_shRIM_post, self.D_model_DC2B_post]

        D_exp_pre_sem  = [self.D_exp_pre_sem, self.D_exp_DC2A_pre_sem,] #  self.D_exp_shRIM_pre_sem,  self.D_exp_DC2B_pre_sem]
        D_exp_post_sem = [self.D_exp_post_sem, self.D_exp_DC2A_post_sem,] # self.D_exp_shRIM_post_sem, self.D_exp_DC2B_post_sem]

        n = len(mutants)

        # Layout: [Exp(NO), Exp(STIM)]   gap   [Model(NO), Model(STIM)]
        bar_width   = 0.18
        small_gap   = 0.10   # separation between Exp pair and Model pair
        group_gap   = 0.60   # separation between mutant groups
        group_width = 4*bar_width + small_gap

        centers = np.arange(n) * (group_width + group_gap)
        lefts = centers - group_width/2

        x_exp_no   = lefts + 0*bar_width
        x_exp_stim = lefts + 1*bar_width
        x_mod_no   = lefts + 2*bar_width + small_gap
        x_mod_stim = lefts + 3*bar_width + small_gap

        # Colors and styles
        c_no, c_stim = 'black', 'red'
        hatch_model = '///'

        # EXP: solid fills (with error bars)
        ax.bar(x_exp_no,   D_exp_pre,  width=bar_width, yerr=D_exp_pre_sem,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, D_exp_post, width=bar_width, yerr=D_exp_post_sem, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   D_model_pre,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, D_model_post, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels(mutants, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_diffusivity_dc2a.svg") 
        fig.savefig("../fig/fig_diffusivity_dc2a.png", dpi=dpi) 
        plt.show()

    def plot_population_dc2a(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c3 = candidate[3]
        candidate[3] = 0  # Set kfmm to 0 for deltaC2A scenario

        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate)
        P_monomer_pre, P_dimer_pre, P_cluster_pre = self.calc_populations(solutionPre)
        P_monomer_post, P_dimer_post, P_cluster_post = self.calc_populations(solutionPost)

        candidate[3] = c3

        print("Population NO STIM: Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_pre*100, P_dimer_pre*100, P_cluster_pre*100))
        print("Population STIM:    Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_post*100, P_dimer_post*100, P_cluster_post*100))

        # plot two pie charts side by side
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        labels = ['M', 'D', 'C']
        sizes_pre = [P_monomer_pre, P_dimer_pre, P_cluster_pre]
        sizes_post = [P_monomer_post, P_dimer_post, P_cluster_post]
        colors = ['#ff9999','#66b3ff','#99ff99']
        explode = (0.05, 0.05, 0.05)  # explode all slices slightly
        wedges1, texts1, autotexts1 = ax[0].pie(sizes_pre, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        wedges2, texts2, autotexts2 = ax[1].pie(sizes_post, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        # draw circle for donut shape
        centre_circle0 = plt.Circle((0,0), 0.70, fc='white')
        centre_circle1 = plt.Circle((0,0), 0.70, fc='white')
        ax[0].add_artist(centre_circle0)
        ax[1].add_artist(centre_circle1)
        ax[0].set_title('NO STIM', fontsize=fontsize)
        ax[1].set_title('STIM', fontsize=fontsize)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax[0].axis('equal')  
        ax[1].axis('equal')
        fig.tight_layout()
        fig.savefig("../fig/fig_population_dc2a.svg") 
        fig.savefig("../fig/fig_population_dc2a.png", dpi=dpi)
        plt.show()

    def plot_total_recruitment_shrim(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c3 = candidate[3]
        candidate[3] = c3 * 2  # Set kfmm to 2x for shRNA RIM2 scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[3] = c3

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
       # ax.plot(self.timePoints, pre / self.Atotal, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, post / self.AtotalDense, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A density on membrane\n(copies/$\mu$m$^2$)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_total_recruitment_shrim.svg") 
        fig.savefig("../fig/fig_total_recruitment_shrim.png", dpi=dpi) 
        plt.show()

    def plot_cluster_recruitment_shrim(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        
        c3 = candidate[3]
        candidate[3] = c3 * 2  # Set kfmm to 2x for shRNA RIM2 scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[3] = c3

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, preCluster, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, postCluster, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A copies in cluster")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 50)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_cluster_recruitment_shrim.svg") 
        fig.savefig("../fig/fig_cluster_recruitment_shrim.png", dpi=dpi) 
        plt.show()

    def plot_diffusivity_shrim(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        fig, ax = plt.subplots(figsize=figsize)

        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        self.D_model_pre = d_pre
        self.D_model_post = d_post

        c3 = candidate[3]
        candidate[3] = c3 * 2  # Set kfmm to 2x for shRNA RIM2 scenario
        solutionPre, solutionPost, _, _, d_pre, d_post = self.simulate(candidate)

        self.D_model_shRIM_pre = d_pre
        self.D_model_shRIM_post = d_post

        candidate[3] = c3

        print("Experimental:", self.D_exp_shRIM_pre, self.D_exp_shRIM_post)
        print("Model:", self.D_model_shRIM_pre, self.D_model_shRIM_post)

        mutants = ['WT', 'shRNA RIM2',] # r'$\Delta$C2A', r'$\Delta$C2B']

        # Data
        D_exp_pre   = [self.D_exp_pre, self.D_exp_shRIM_pre,] #   self.D_exp_shRIM_pre,   self.D_exp_DC2B_pre]
        D_exp_post  = [self.D_exp_post, self.D_exp_shRIM_post,] #  self.D_exp_shRIM_post,  self.D_exp_DC2B_post]
        D_model_pre  = [self.D_model_pre, self.D_model_shRIM_pre,] #  self.D_model_shRIM_pre,  self.D_model_DC2B_pre]
        D_model_post = [self.D_model_post, self.D_model_shRIM_post,] # self.D_model_shRIM_post, self.D_model_DC2B_post]

        D_exp_pre_sem  = [self.D_exp_pre_sem, self.D_exp_shRIM_pre_sem,] #  self.D_exp_shRIM_pre_sem,  self.D_exp_DC2B_pre_sem]
        D_exp_post_sem = [self.D_exp_post_sem, self.D_exp_shRIM_post_sem,] # self.D_exp_shRIM_post_sem, self.D_exp_DC2B_post_sem]

        n = len(mutants)

        # Layout: [Exp(NO), Exp(STIM)]   gap   [Model(NO), Model(STIM)]
        bar_width   = 0.18
        small_gap   = 0.10   # separation between Exp pair and Model pair
        group_gap   = 0.60   # separation between mutant groups
        group_width = 4*bar_width + small_gap

        centers = np.arange(n) * (group_width + group_gap)
        lefts = centers - group_width/2

        x_exp_no   = lefts + 0*bar_width
        x_exp_stim = lefts + 1*bar_width
        x_mod_no   = lefts + 2*bar_width + small_gap
        x_mod_stim = lefts + 3*bar_width + small_gap

        # Colors and styles
        c_no, c_stim = 'black', 'red'
        hatch_model = '///'

        # EXP: solid fills (with error bars)
        ax.bar(x_exp_no,   D_exp_pre,  width=bar_width, yerr=D_exp_pre_sem,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, D_exp_post, width=bar_width, yerr=D_exp_post_sem, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   D_model_pre,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, D_model_post, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels(mutants, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_diffusivity_shrim.svg") 
        fig.savefig("../fig/fig_diffusivity_shrim.png", dpi=dpi) 
        plt.show()

    def plot_population_shrim(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c3 = candidate[3]
        candidate[3] = c3 * 2  # Set kfmm to 2x for shRNA RIM2 scenario

        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate)
        P_monomer_pre, P_dimer_pre, P_cluster_pre = self.calc_populations(solutionPre)
        P_monomer_post, P_dimer_post, P_cluster_post = self.calc_populations(solutionPost)

        candidate[3] = c3

        print("Population NO STIM: Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_pre*100, P_dimer_pre*100, P_cluster_pre*100))
        print("Population STIM:    Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_post*100, P_dimer_post*100, P_cluster_post*100))

        # plot two pie charts side by side
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        labels = ['M', 'D', 'C']
        sizes_pre = [P_monomer_pre, P_dimer_pre, P_cluster_pre]
        sizes_post = [P_monomer_post, P_dimer_post, P_cluster_post]
        colors = ['#ff9999','#66b3ff','#99ff99']
        explode = (0.05, 0.05, 0.05)  # explode all slices slightly
        wedges1, texts1, autotexts1 = ax[0].pie(sizes_pre, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        wedges2, texts2, autotexts2 = ax[1].pie(sizes_post, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        # draw circle for donut shape
        centre_circle0 = plt.Circle((0,0), 0.70, fc='white')
        centre_circle1 = plt.Circle((0,0), 0.70, fc='white')
        ax[0].add_artist(centre_circle0)
        ax[1].add_artist(centre_circle1)
        ax[0].set_title('NO STIM', fontsize=fontsize)
        ax[1].set_title('STIM', fontsize=fontsize)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax[0].axis('equal')  
        ax[1].axis('equal')
        fig.tight_layout()
        fig.savefig("../fig/fig_population_shrim.svg") 
        fig.savefig("../fig/fig_population_shrim.png", dpi=dpi)
        plt.show()

    def plot_total_recruitment_dc2b(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c0 = candidate[0]
        candidate[0] = c0 * 0.5  # Set kfsr to 0.5x for DeltaC2B scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[0] = c0

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
       # ax.plot(self.timePoints, pre / self.Atotal, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, post / self.AtotalDense, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A density on membrane\n(copies/$\mu$m$^2$)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_total_recruitment_dc2b.svg") 
        fig.savefig("../fig/fig_total_recruitment_dc2b.png", dpi=dpi) 
        plt.show()

    def plot_cluster_recruitment_dc2b(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        
        c0 = candidate[0]
        candidate[0] = c0 * 0.5  # Set kfsr to 0.5x for DeltaC2B scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[0] = c0

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, preCluster, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, postCluster, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A copies in cluster")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 50)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_cluster_recruitment_dc2b.svg") 
        fig.savefig("../fig/fig_cluster_recruitment_dc2b.png", dpi=dpi) 
        plt.show()

    def plot_diffusivity_dc2b(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        fig, ax = plt.subplots(figsize=figsize)

        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        self.D_model_pre = d_pre
        self.D_model_post = d_post

        c0 = candidate[0]
        candidate[0] = c0 * 0.5  # Set kfsr to 0.5x for DeltaC2B scenario
        solutionPre, solutionPost, _, _, d_pre, d_post = self.simulate(candidate)

        self.D_model_DC2B_pre = d_pre
        self.D_model_DC2B_post = d_post

        candidate[0] = c0

        print("Experimental:", self.D_exp_DC2B_pre, self.D_exp_DC2B_post)
        print("Model:", self.D_model_DC2B_pre, self.D_model_DC2B_post)

        mutants = ['WT', r'$\Delta$C2B',] # r'$\Delta$C2A', r'$\Delta$C2B']

        # Data
        D_exp_pre   = [self.D_exp_pre, self.D_exp_DC2B_pre,] #   self.D_exp_shRIM_pre,   self.D_exp_DC2B_pre]
        D_exp_post  = [self.D_exp_post, self.D_exp_DC2B_post,] #  self.D_exp_shRIM_post,  self.D_exp_DC2B_post]
        D_model_pre  = [self.D_model_pre, self.D_model_DC2B_pre,] #  self.D_model_shRIM_pre,  self.D_model_DC2B_pre]
        D_model_post = [self.D_model_post, self.D_model_DC2B_post,] # self.D_model_shRIM_post, self.D_model_DC2B_post]

        D_exp_pre_sem  = [self.D_exp_pre_sem, self.D_exp_DC2B_pre_sem,] #  self.D_exp_shRIM_pre_sem,  self.D_exp_DC2B_pre_sem]
        D_exp_post_sem = [self.D_exp_post_sem, self.D_exp_DC2B_post_sem,] # self.D_exp_shRIM_post_sem, self.D_exp_DC2B_post_sem]

        n = len(mutants)

        # Layout: [Exp(NO), Exp(STIM)]   gap   [Model(NO), Model(STIM)]
        bar_width   = 0.18
        small_gap   = 0.10   # separation between Exp pair and Model pair
        group_gap   = 0.60   # separation between mutant groups
        group_width = 4*bar_width + small_gap

        centers = np.arange(n) * (group_width + group_gap)
        lefts = centers - group_width/2

        x_exp_no   = lefts + 0*bar_width
        x_exp_stim = lefts + 1*bar_width
        x_mod_no   = lefts + 2*bar_width + small_gap
        x_mod_stim = lefts + 3*bar_width + small_gap

        # Colors and styles
        c_no, c_stim = 'black', 'red'
        hatch_model = '///'

        # EXP: solid fills (with error bars)
        ax.bar(x_exp_no,   D_exp_pre,  width=bar_width, yerr=D_exp_pre_sem,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, D_exp_post, width=bar_width, yerr=D_exp_post_sem, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   D_model_pre,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, D_model_post, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels(mutants, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_diffusivity_dc2b.svg") 
        fig.savefig("../fig/fig_diffusivity_dc2b.png", dpi=dpi) 
        plt.show()

    def plot_population_dc2b(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c0 = candidate[0]
        candidate[0] = c0 * 0.5  # Set kfsr to 0.5x for DeltaC2B scenario

        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate)
        P_monomer_pre, P_dimer_pre, P_cluster_pre = self.calc_populations(solutionPre)
        P_monomer_post, P_dimer_post, P_cluster_post = self.calc_populations(solutionPost)

        candidate[0] = c0

        print("Population NO STIM: Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_pre*100, P_dimer_pre*100, P_cluster_pre*100))
        print("Population STIM:    Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_post*100, P_dimer_post*100, P_cluster_post*100))

        # plot two pie charts side by side
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        labels = ['M', 'D', 'C']
        sizes_pre = [P_monomer_pre, P_dimer_pre, P_cluster_pre]
        sizes_post = [P_monomer_post, P_dimer_post, P_cluster_post]
        colors = ['#ff9999','#66b3ff','#99ff99']
        explode = (0.05, 0.05, 0.05)  # explode all slices slightly
        wedges1, texts1, autotexts1 = ax[0].pie(sizes_pre, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        wedges2, texts2, autotexts2 = ax[1].pie(sizes_post, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        # draw circle for donut shape
        centre_circle0 = plt.Circle((0,0), 0.70, fc='white')
        centre_circle1 = plt.Circle((0,0), 0.70, fc='white')
        ax[0].add_artist(centre_circle0)
        ax[1].add_artist(centre_circle1)
        ax[0].set_title('NO STIM', fontsize=fontsize)
        ax[1].set_title('STIM', fontsize=fontsize)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax[0].axis('equal')  
        ax[1].axis('equal')
        fig.tight_layout()
        fig.savefig("../fig/fig_population_dc2b.svg") 
        fig.savefig("../fig/fig_population_dc2b.png", dpi=dpi)
        plt.show()

    def plot_total_recruitment_endogenous(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c7 = candidate[7]
        candidate[7] = c7 * 0.1  # Set munc13 concentration to 0.1x for endogenous munc13 scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[7] = c7

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        #ax.plot(self.timePoints, pre / self.Atotal, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, post / self.AtotalDense, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A density on membrane\n(copies/$\mu$m$^2$)")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_total_recruitment_endogenous.svg") 
        fig.savefig("../fig/fig_total_recruitment_endogenous.png", dpi=dpi) 
        plt.show()

    def plot_cluster_recruitment_endogenous(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        
        c7 = candidate[7]
        candidate[7] = c7 * 0.1  # Set munc13 concentration to 0.1x for endogenous munc13 scenario
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        candidate[7] = c7

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.timePoints, preCluster, linestyle="-", label="NO STIM", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, postCluster, linestyle="-", label="STIM", color=c_post, alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Unc13A copies in cluster")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 50)
        ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_cluster_recruitment_endogenous.svg") 
        fig.savefig("../fig/fig_cluster_recruitment_endogenous.png", dpi=dpi) 
        plt.show()

    def plot_diffusivity_endogenous(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)
        fig, ax = plt.subplots(figsize=figsize)

        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        self.D_model_pre = d_pre
        self.D_model_post = d_post

        c7 = candidate[7]
        candidate[7] = c7 * 0.1  # Set munc13 concentration to 0.1x for endogenous munc13 scenario
        solutionPre, solutionPost, _, _, d_pre, d_post = self.simulate(candidate)

        self.D_model_endogenous_pre = d_pre
        self.D_model_endogenous_post = d_post

        candidate[7] = c7

        # print("Experimental:", self.D_exp_endogenous_pre, self.D_exp_endogenous_post)
        print("Model:", self.D_model_endogenous_pre, self.D_model_endogenous_post)

        mutants = ['WT', 'Endogenous',] # r'$\Delta$C2A', r'$\Delta$C2B']

        # Data
        D_exp_pre   = [self.D_exp_pre, 0,] #   self.D_exp_shRIM_pre,   self.D_exp_DC2B_pre]
        D_exp_post  = [self.D_exp_post, 0,] #  self.D_exp_shRIM_post,  self.D_exp_DC2B_post]
        D_model_pre  = [self.D_model_pre, self.D_model_endogenous_pre,] #  self.D_model_shRIM_pre,  self.D_model_DC2B_pre]
        D_model_post = [self.D_model_post, self.D_model_endogenous_post,] # self.D_model_shRIM_post, self.D_model_DC2B_post]

        D_exp_pre_sem  = [self.D_exp_pre_sem, 0,] #  self.D_exp_shRIM_pre_sem,  self.D_exp_DC2B_pre_sem]
        D_exp_post_sem = [self.D_exp_post_sem, 0,] # self.D_exp_shRIM_post_sem, self.D_exp_DC2B_post_sem]

        n = len(mutants)

        # Layout: [Exp(NO), Exp(STIM)]   gap   [Model(NO), Model(STIM)]
        bar_width   = 0.18
        small_gap   = 0.10   # separation between Exp pair and Model pair
        group_gap   = 0.60   # separation between mutant groups
        group_width = 4*bar_width + small_gap

        centers = np.arange(n) * (group_width + group_gap)
        lefts = centers - group_width/2

        x_exp_no   = lefts + 0*bar_width
        x_exp_stim = lefts + 1*bar_width
        x_mod_no   = lefts + 2*bar_width + small_gap
        x_mod_stim = lefts + 3*bar_width + small_gap

        # Colors and styles
        c_no, c_stim = 'black', 'red'
        hatch_model = '///'

        # EXP: solid fills (with error bars)
        ax.bar(x_exp_no,   D_exp_pre,  width=bar_width, yerr=D_exp_pre_sem,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, D_exp_post, width=bar_width, yerr=D_exp_post_sem, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   D_model_pre,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, D_model_post, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels(mutants, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend()
        fig.tight_layout()

        fig.savefig("../fig/fig_diffusivity_endogenous.svg") 
        fig.savefig("../fig/fig_diffusivity_endogenous.png", dpi=dpi) 
        plt.show()

    def plot_population_endogenous(self, select=3, figsize=(4, 3), fontsize=12, dpi=300):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        candidate = self.filteredSolutions[select]
        print("candinate: ", candidate)

        c7 = candidate[7]
        candidate[7] = c7 * 0.1  # Set munc13 concentration to 0.1x for endogenous munc13 scenario

        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate)
        P_monomer_pre, P_dimer_pre, P_cluster_pre = self.calc_populations(solutionPre)
        P_monomer_post, P_dimer_post, P_cluster_post = self.calc_populations(solutionPost)

        candidate[7] = c7

        print("Population NO STIM: Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_pre*100, P_dimer_pre*100, P_cluster_pre*100))
        print("Population STIM:    Monomer: %.2f%%, Dimer: %.2f%%, Cluster: %.2f%%" % (P_monomer_post*100, P_dimer_post*100, P_cluster_post*100))

        # plot two pie charts side by side
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        labels = ['M', 'D', 'C']
        sizes_pre = [P_monomer_pre, P_dimer_pre, P_cluster_pre]
        sizes_post = [P_monomer_post, P_dimer_post, P_cluster_post]
        colors = ['#ff9999','#66b3ff','#99ff99']
        explode = (0.05, 0.05, 0.05)  # explode all slices slightly
        wedges1, texts1, autotexts1 = ax[0].pie(sizes_pre, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        wedges2, texts2, autotexts2 = ax[1].pie(sizes_post, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140, pctdistance=0.85, textprops={'fontsize': fontsize * 0.8})
        # draw circle for donut shape
        centre_circle0 = plt.Circle((0,0), 0.70, fc='white')
        centre_circle1 = plt.Circle((0,0), 0.70, fc='white')
        ax[0].add_artist(centre_circle0)
        ax[1].add_artist(centre_circle1)
        ax[0].set_title('NO STIM', fontsize=fontsize)
        ax[1].set_title('STIM', fontsize=fontsize)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax[0].axis('equal')  
        ax[1].axis('equal')
        fig.tight_layout()
        fig.savefig("../fig/fig_population_endogenous.svg") 
        fig.savefig("../fig/fig_population_endogenous.png", dpi=dpi)
        plt.show()

    # plot the total munc13 in cluster, compared with experimental kinetic, nerdss simulaition data
    def plot_cluster_kinetic(self, best=300, select=297, totalTime=10.0, dt=0.1, figsize=(9,6), fontsize=12, dpi=600):
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        idx = select
        candidate = self.filteredSolutions[idx]
        print("candinate: ", candidate)

        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        data = {
            "timePoints": self.timePoints,
            "preCluster": preCluster,
            "postCluster": postCluster,
            "pre": pre,
            "post": post
        }
        df = pd.DataFrame(data)
        df.to_csv(f"../../NERDSS/data/munc13_copies_ode.csv", index=False)
        
        data = {
            "timePoints": self.timePoints,
            "preEq": preEq,
            "postEq": postEq
        }
        df = pd.DataFrame(data)
        df.to_csv(f"../../NERDSS/data/munc13_copies_eq_ode.csv", index=False)

        df_pre = pd.read_csv("../../NERDSS/data/stochastic_pre.csv")
        df_post = pd.read_csv("../../NERDSS/data/stochastic_post.csv")

        # ----- Publication-oriented defaults -----
        plt.rcParams.update({
            "figure.dpi": 300, "savefig.dpi": 300,
            "font.size": fontsize, "axes.labelsize": fontsize, "axes.titlesize": fontsize,
            "legend.fontsize": fontsize, "xtick.labelsize": fontsize, "ytick.labelsize": fontsize,
            "axes.linewidth": 0.8, "lines.linewidth": 1.4,
            "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in vector exports
        })

        fig, ax = plt.subplots(figsize=figsize)

        c_pre  = "black"   # NO STIM
        c_post = "red"     # STIM

        # --- PRE (NO STIM): scatter means + SEM band + ODE line + experiment---
        pre_t   = df_pre["timePoints"].to_numpy()
        pre_m   = df_pre["munc13_cluster_mean"].to_numpy()
        pre_sem = df_pre["munc13_cluster_sem"].to_numpy()

        # Errorbar points
        err1 = ax.errorbar(
            pre_t, pre_m, yerr=pre_sem,
            fmt="o", ms=3, mfc="white", mew=0.9,
            capsize=2, elinewidth=0.9,
            label="NO STIM: NERDSS",
            color=c_pre, ecolor=c_pre, linestyle="none", zorder=2
        )

        # ODE line
        line_pre, = ax.plot(self.timePoints, preCluster, linestyle="-", label="NO STIM: ODE", color=c_pre, alpha=0.95, zorder=3)

        # plot experiment
        # NO STIM
        solutionPre, solutionPost, _, _, Dpre, Dpost = self.simulate(candidate)
        _, _, preCluster, postCluster, pre, post, preEq, postEq,_,_,_,_,_,_ = self.calculate_results(candidate)
        mc, md = self.get_cluster_time_series(solutionPre, False)
        fold_increase = self.get_simulated_fold_change(mc, md)
        delta = self.compute_background_from_target_increase(fold_increase)

        relInt = (self.expdata.expInt - delta) / (self.expdata.expInt[0] - delta)
        sem = self.expdata.expSEM / (self.expdata.expInt[0] - delta)
        time = self.expdata.time
        ax.errorbar(time, relInt * preCluster[0], yerr=sem * preCluster[0], linestyle="--", color=c_pre, alpha=0.95, label='NO STIM: Exp')

        # --- POST (STIM): scatter means + SEM band + ODE line + experiemnt---
        post_t   = df_post["timePoints"].to_numpy()
        post_m   = df_post["munc13_cluster_mean"].to_numpy()
        post_sem = df_post["munc13_cluster_sem"].to_numpy()

        err2 = ax.errorbar(
            post_t, post_m, yerr=post_sem,
            fmt="s", ms=3, mfc="white", mew=0.9,
            capsize=2, elinewidth=0.9,
            label="STIM: NERDSS",
            color=c_post, ecolor=c_post, linestyle="none", zorder=2
        )

        line_post, = ax.plot(self.timePoints, postCluster, linestyle="-", label="STIM: ODE", color=c_post, alpha=0.95, zorder=3)

        # STIM
        mc, md = self.get_cluster_time_series(solutionPost, True)
        fold_increase = self.get_simulated_fold_change(mc, md)
        delta = self.compute_background_from_target_increase(fold_increase)

        relInt = (self.expdata.expInt - delta) / (self.expdata.expInt[0] - delta)
        sem = self.expdata.expSEM / (self.expdata.expInt[0] - delta)
        time = self.expdata.time
        ax.errorbar(time, relInt * postCluster[0], yerr=sem * postCluster[0], linestyle="--", color=c_post, alpha=0.95, label='STIM: Exp')
    
        # ----- Axes cosmetics -----
        ax.set_xlabel("Time (s)")                       # adjust units/label to yours
        ax.set_ylabel("Unc13A Copies in Cluster Region")   # adjust units/label to yours

        ax.legend(fontsize=12)

        fig.tight_layout()

        # Save as vector + raster
        fig.savefig("../fig/munc13_timecourse.svg")  # vector for journals
        fig.savefig("../fig/munc13_timecourse.png", dpi=dpi)  # raster fallback
        plt.show()

    # plot the diffusion constants of dilute phase; plot the percentage of cluster copies
    def plot_diffusion_dilute_percentage_cluster(self, select=298, figsize=(2.8, 7), fontsize=18, dpi=600):
        """
        Average diffusion of Munc13 copies in the dilute phase and percentage of Munc13 copies in the cluster.
        """
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        if not hasattr(self, "filteredSolutions") or not self.filteredSolutions:
            print("Filtered solutions not found. Run `filter_and_store_solutions()` first.")
            return

        idx = select
        candidate = self.filteredSolutions[idx]
        print(candidate)
        
        D1 = candidate[9]  # Monomer diffusion constant on membrane
        D2 = candidate[9] / candidate[10] # Dimer diffusion constant on membrane
        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate)

        D_dilute_pre = self.calc_diffusion_dilute(D1, D2, solutionPre)
        D_dilute_post = self.calc_diffusion_dilute(D1, D2, solutionPost)
        perc_cluster_pre = self.calc_percentages_cluster(solutionPre)
        perc_cluster_post = self.calc_percentages_cluster(solutionPost)
        _, perc_cluster_pre_2D = self.calc_3D_2D_percentages_cluster(solutionPre)
        _, perc_cluster_post_2D = self.calc_3D_2D_percentages_cluster(solutionPost)

        print(f"Average Diffusion in Dilute Phase: No Stim={D_dilute_pre:.4f}, Stim={D_dilute_post:.4f}")
        print(f"Percentage of Munc13 in Cluster: No Stim={perc_cluster_pre:.4f}, Stim={perc_cluster_post:.4f}")
        print(f"Percentage of Munc13 in Cluster From 2D: No Stim={perc_cluster_pre:.4f}, Stim={perc_cluster_post:.4f}")

        categories = ["NO STIM", "STIM"]
        x = np.arange(len(categories))
        width = 0.35

        # Plot: Diffusion constants in dilute phase
        fig, axe = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
        axe.bar(x, [D_dilute_pre, D_dilute_post], width, capsize=5, label="Model", color=['black', 'red'], alpha=1.0)
    
        axe.set_xticks(x)
        axe.set_xticklabels(categories, fontsize=fontsize)
        axe.set_ylabel(r"$\mathrm{Average\ Diffusion\ out\ of\ Cluster}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        # axe.set_title("Average Diffusion in Dilute Phase", fontsize=fontsize)
        fig.tight_layout()
        fig.savefig("../data/dilute_diffusion.png", dpi=dpi, bbox_inches="tight")
        plt.show()

        # Plot: Percentage of Munc13 copies in the cluster
        fig, axe = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
        axe.bar(x, [perc_cluster_pre*100, perc_cluster_post*100], width, capsize=5, label="Model", color=['black', 'red'], alpha=1.0)
    
        axe.set_xticks(x)
        axe.set_xticklabels(categories, fontsize=fontsize)
        axe.set_ylabel(r"$\mathrm{Unc13A\ in\ Cluster}\ (\%)$", fontsize=fontsize)
        # axe.set_title("Percentage of Munc13 in Cluster", fontsize=fontsize)
        fig.tight_layout()
        fig.savefig("../data/cluster_perc.png", dpi=dpi, bbox_inches="tight")
        plt.show()

        # Plot: Percentage of Munc13 from 2D in the cluster
        fig, axe = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
        axe.bar(x, [perc_cluster_pre_2D*100, perc_cluster_post_2D*100], width, capsize=5, label="Model", color=['black', 'red'], alpha=1.0)
    
        axe.set_xticks(x)
        axe.set_xticklabels(categories, fontsize=fontsize)
        axe.set_ylabel(r"$\mathrm{2D\ Lateral\ Recruitment\ to\ Cluster}\ (\%)$", fontsize=fontsize)
        # axe.set_title("Percentage of Munc13 in Cluster from 2D", fontsize=fontsize)
        fig.tight_layout()
        fig.savefig("../data/cluster_from2D.png", dpi=dpi, bbox_inches="tight")
        plt.show()

    # plot the distribution of optimized parameters
    def plot_parms(self, best=100, select=100, totalTime=10.0, dt=0.1,
               parameter_ranges=None, percent=None, figsize=(16,12),
               fontsize=18, dpi=600, bins_per_decade=10):
        """
        Create figure: hist of parameter values with even-width bars in log space.
        """
        from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

        # ------------------------------ style -----------------------------------
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        if parameter_ranges is None:
            parameter_ranges = {
                "kfsr":          {"min": 0.001, "max": 10},
                "krsr_nostim":   {"min": 0.1,   "max": 1000},
                "krsr_stim":     {"min": 0.1,   "max": 1000},
                "kfmm":          {"min": 0.001, "max": 10},
                "krmm":          {"min": 0.01,  "max": 100},
                "kfc_nostim":    {"min": 0.001, "max": 10},
                "kfc_stim":      {"min": 0.001, "max": 10},
                "S0":            {"min": 0.01,  "max": 10},
                "R0":            {"min": 100,   "max": 500},
                "D1":            {"min": 0.05,  "max": 5},
                "D1_over_D2":    {"min": 1.5,   "max": 5},
                "X":             {"min": 10,    "max": 10},     # constant
            }

        # helper: log-spaced bin edges (even in log space)
        def _log_bins(lo, hi, bpd=bins_per_decade):
            lo = max(lo, np.finfo(float).tiny)
            exp_lo = np.floor(np.log10(lo))
            exp_hi = np.ceil(np.log10(hi))
            n = max(1, int((exp_hi - exp_lo) * bpd))
            return np.logspace(exp_lo, exp_hi, n + 1)

        # --------------------- model times for completeness ---------------------
        self.t_max = totalTime
        n_time_points = int(totalTime / dt)
        self.timePoints = [i * dt for i in range(n_time_points + 1)]

        # --------------------- load and pick solutions --------------------------
        df = pd.read_csv("../data/optimizedParms.txt", sep=",", engine="python")
        df.columns = df.columns.str.strip()
        df = df.sort_values(by="Rank")

        solutions = []
        for _, row in df.iterrows():
            rank = int(row["Rank"])
            fitness = float(row["Fitness"])
            param_cols_all = [c for c in df.columns if c not in ["Rank", "Fitness"]]
            param_vector = [float(row[c]) for c in param_cols_all]
            solutions.append((rank, fitness, param_vector))

        if percent is not None:
            n_solutions = len(solutions)
            n_to_consider = max(1, int(n_solutions * percent / 100))
            best = n_to_consider
            select = n_to_consider

        print(f"Total solutions loaded: {len(solutions)}")
        if percent:
            print(f"Filtering solutions within {percent}% of best fitness.")
        print(f"Considering top {best} solutions by rank.")

        filtered_solutions = [sol for sol in solutions if sol[0] < best]
        if len(filtered_solutions) < select:
            select = len(filtered_solutions)

        rng = np.random.default_rng()
        random_indices = rng.choice(len(filtered_solutions), size=select, replace=False)
        print(f"Randomly selected {select} solutions for parameter histograms.")

        # --------------------------- prepare data -------------------------------
        param_cols = [c for c in df.columns if c not in ["Rank", "Fitness"]]
        n_params = len(param_cols)

        param_matrix = np.array([filtered_solutions[i][2] for i in random_indices],
                                dtype=float)  # shape: (select, n_params)

        parms_name_map = {
            'kfsr': r'$kf_{SR}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krsr_nostim': r'$kr_{SR,\mathrm{NOSTIM}}\; (\mathrm{s}^{-1})$',
            'krsr_stim': r'$kr_{SR,\mathrm{STIM}}\; (\mathrm{s}^{-1})$',
            'kfmm': r'$kf_{MM}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krmm': r'$kr_{MM}\; (\mathrm{s}^{-1})$',
            'kfc_nostim': r'$kf_{C,\mathrm{NOSTIM}}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'kfc_stim': r'$kf_{C,\mathrm{STIM}}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'S0': r'$S_{0}\; (\mu\mathrm{M})$',
            'R0': r'$R_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            'D1': r'$D_{1}\; (\mu\mathrm{m}^{2}/\mathrm{s})$',
            'D1_over_D2': r'$D_{1} / D_{2}$',
            'X': r'$X$'
        }

        # ----------------------------- plotting --------------------------------
        fig1, axes1 = plt.subplots(3, 4, figsize=figsize, dpi=dpi)
        axes1 = axes1.flatten()

        for idx, ax in enumerate(axes1):
            if idx >= n_params:
                ax.set_visible(False)
                continue

            col_name = param_cols[idx]
            vals = np.asarray(param_matrix[:, idx], dtype=float)

            # keep positives only (log scale)
            positive = vals[vals > 0]
            if positive.size == 0:
                ax.text(0.5, 0.5, "no positive values", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(parms_name_map.get(col_name, col_name))
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                continue

            # choose plotting range
            if col_name in parameter_ranges:
                p = parameter_ranges[col_name]
                lo = p["min"] * 0.1
                hi = p["max"] * 10
            else:
                raise ValueError(f"Parameter '{col_name}' not found in provided ranges.")

            # log-spaced bins -> even widths in log space
            bins = _log_bins(lo, hi, bpd=bins_per_decade)
            ax.hist(positive, bins=bins, edgecolor="black", alpha=0.7)

            # log axis and tidy ticks
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
            ax.xaxis.set_major_formatter(LogFormatterMathtext())
            ax.xaxis.set_minor_locator(LogLocator(base=10.0,
                                                subs=np.arange(2, 10) * 0.1,
                                                numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x', which='major', labelsize=fontsize-2)
            ax.tick_params(axis='x', which='minor', length=3)

            # helpful light grid
            ax.grid(True, which='both', axis='x', alpha=0.15)

            # show allowed range bounds if provided
            if col_name in parameter_ranges:
                ax.axvline(parameter_ranges[col_name]["min"], color='red', ls='--', lw=2)
                ax.axvline(parameter_ranges[col_name]["max"], color='red', ls='--', lw=2)

            ax.set_title(parms_name_map.get(col_name, col_name))
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        fig1.tight_layout(rect=[0, 0, 1, 0.97])
        fig1.savefig("../fig/params-distribution.png", dpi=dpi, bbox_inches="tight")

    def plot_parameter_ranges_summary(
        self,
        best=100,
        select=100,
        totalTime=10.0,
        dt=0.1,
        parameter_ranges=None,
        percent=10,                    # use top `percent`% of solutions by Rank
        figsize=(11, 7),
        fontsize=16,
        dpi=600,
        bar_width=0.7,
        save_path="../fig/params-summary.png",
    ):
        """
        Summary plot of parameter ranges:
        - Grey bar: allowed range (from `parameter_ranges` or data span fallback).
        - Light blue bar: range across the top `percent`% solutions by Rank.
        - Dark blue line: best (lowest-Rank) value.

        `percent=10` on a file with 1000 rows uses the best 100 rows by Rank.
        """
        from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D

        # ------------------------------ style -----------------------------------
        sns.set_style("ticks")
        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        if parameter_ranges is None:
            parameter_ranges = {
                "kfsr":          {"min": 0.001, "max": 10},
                "krsr_nostim":   {"min": 0.1,   "max": 1000},
                "krsr_stim":     {"min": 0.1,   "max": 1000},
                "kfmm":          {"min": 0.001, "max": 10},
                "krmm":          {"min": 0.01,  "max": 100},
                "kfc_nostim":    {"min": 0.001, "max": 10},
                "kfc_stim":      {"min": 0.001, "max": 10},
                "S0":            {"min": 0.01,  "max": 10},
                "R0":            {"min": 100,   "max": 500},
                "D1":            {"min": 0.05,  "max": 5},
                "D1_over_D2":    {"min": 1.5,   "max": 5},
                "X":             {"min": 10,    "max": 10},     # constant
            }

        # --------------------- model times for completeness ---------------------
        self.t_max = totalTime
        n_time_points = int(totalTime / dt)
        self.timePoints = [i * dt for i in range(n_time_points + 1)]

        # --------------------- load solutions -----------------------------------
        df = pd.read_csv("../data/optimizedParms.txt", sep=",", engine="python")
        df.columns = df.columns.str.strip()

        # Parameter columns = everything except bookkeeping
        param_cols = [c for c in df.columns if c not in ("Rank", "Fitness")]
        if not param_cols:
            raise ValueError("No parameter columns found in optimizedParms.txt")

        # Best-by-rank subset size
        n_total = len(df)
        n_best = max(1, int(np.floor(n_total * (percent / 100.0))))
        # Best subset by Rank (smaller Rank = better)
        best_df = df.nsmallest(n_best, "Rank")
        best_row = df.nsmallest(1, "Rank").iloc[0]

        print(f"Total solutions loaded: {n_total}")
        print(f"Using top {percent}% by Rank -> {n_best} solutions.")

        # -------------------- labels --------------------------------------------
        parms_name_map = {
            'kfsr': r'$kf_{SR}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krsr_nostim': r'$kr_{SR,\mathrm{NOSTIM}}\; (\mathrm{s}^{-1})$',
            'krsr_stim': r'$kr_{SR,\mathrm{STIM}}\; (\mathrm{s}^{-1})$',
            'kfmm': r'$kf_{MM}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krmm': r'$kr_{MM}\; (\mathrm{s}^{-1})$',
            'kfc_nostim': r'$kf_{C,\mathrm{NOSTIM}}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'kfc_stim': r'$kf_{C,\mathrm{STIM}}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'S0': r'$S_{0}\; (\mu\mathrm{M})$',
            'R0': r'$R_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            'D1': r'$D_{1}\; (\mu\mathrm{m}^{2}/\mathrm{s})$',
            'D1_over_D2': r'$D_{1} / D_{2}$',
            'X': r'$X$'
        }

        tiny = np.finfo(float).tiny

        # -------------------- allowed ranges (grey) ------------------------------
        pr = {}
        for p in param_cols:
            if p in parameter_ranges:
                lo = max(parameter_ranges[p]["min"], tiny)
                hi = parameter_ranges[p]["max"]
            else:
                raise ValueError(f"Parameter '{p}' not found in provided ranges.")
            if hi <= lo or np.isclose(np.log10(hi) - np.log10(lo), 0.0):
                lo *= 0.9
                hi *= 1.1
            pr[p] = {"min": lo, "max": hi}

        # ------------- range across the best `percent`% (light blue) ------------
        subset_ranges = {}
        for p in param_cols:
            col = best_df[p].to_numpy(float)
            col = col[np.isfinite(col) & (col > 0)]
            if col.size == 0:
                v = max(float(best_row[p]), tiny)
                wlo, whi = v * 0.95, v * 1.05
            else:
                wlo, whi = col.min(), col.max()
            # clamp to allowed, pad if collapsed
            wlo = max(wlo, pr[p]["min"])
            whi = min(whi, pr[p]["max"])
            if whi <= wlo:
                wlo *= 0.95
                whi *= 1.05
            subset_ranges[p] = (wlo, whi)

        # global y span (from allowed)
        y_min = min(pr[p]["min"] for p in param_cols)
        y_max = max(pr[p]["max"] for p in param_cols)

        # ----------------------------- plot -------------------------------------
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        x = np.arange(len(param_cols), dtype=float)
        half = bar_width / 2.0

        for i, p in enumerate(param_cols):
            alo, ahi = pr[p]["min"], pr[p]["max"]
            wlo, whi = subset_ranges[p]
            best_val = max(float(best_row[p]), tiny)

            # grey allowed range
            ax.add_patch(
                patches.Rectangle((i - half, alo), bar_width, ahi - alo,
                                facecolor="lightgray", edgecolor="none", alpha=1.0)
            )
            # light-blue: best `percent`% by Rank
            ax.add_patch(
                patches.Rectangle((i - half, wlo), bar_width, whi - wlo,
                                facecolor="#7ec8ff", edgecolor="none", alpha=0.85)
            )
            # dark-blue horizontal line for the single best (lowest Rank)
            ax.hlines(best_val, i - bar_width * 0.35, i + bar_width * 0.35,
                    colors="#1f4ea8", linewidth=2.2)

        # y-axis formatting (log)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1,
                                            numticks=12))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", axis="y", alpha=0.15)

        # x labels
        ax.set_xlim(-0.5, len(param_cols) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([parms_name_map.get(p, p) for p in param_cols],
                        rotation=35, ha="right")
        ax.tick_params(axis="x", labelsize=fontsize-2)
        ax.tick_params(axis="y", labelsize=fontsize-2)
        ax.set_ylabel("Parameter-Dependent Units", fontsize=fontsize)

        # legend
        legend_elems = [
            patches.Patch(facecolor="lightgray", edgecolor="none", label="Allowed range"),
            patches.Patch(facecolor="#7ec8ff", edgecolor="none",
                        label=f"Top {percent}% by Rank"),
            Line2D([0], [0], color="#1f4ea8", lw=2.2, label="Best value"),
        ]
        ax.legend(handles=legend_elems, loc="upper right", frameon=False,
                fontsize=fontsize-2)

        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved summary to: {save_path}")

    def plot_all_diffusion(self, figsize=(8, 7), fontsize=18, dpi=600):
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        mutants = ['WT', r'$\Delta$C2A', 'shRNA RIM2', r'$\Delta$C2B']

        # Data
        D_exp_pre   = [self.D_exp_pre,   self.D_exp_DC2A_pre,   self.D_exp_shRIM_pre,   self.D_exp_DC2B_pre]
        D_exp_post  = [self.D_exp_post,  self.D_exp_DC2A_post,  self.D_exp_shRIM_post,  self.D_exp_DC2B_post]
        D_model_pre  = [self.D_model_pre,  self.D_model_DC2A_pre,  self.D_model_shRIM_pre,  self.D_model_DC2B_pre]
        D_model_post = [self.D_model_post, self.D_model_DC2A_post, self.D_model_shRIM_post, self.D_model_DC2B_post]

        D_exp_pre_sem  = [self.D_exp_pre_sem,  self.D_exp_DC2A_pre_sem,  self.D_exp_shRIM_pre_sem,  self.D_exp_DC2B_pre_sem]
        D_exp_post_sem = [self.D_exp_post_sem, self.D_exp_DC2A_post_sem, self.D_exp_shRIM_post_sem, self.D_exp_DC2B_post_sem]

        n = len(mutants)

        # Layout: [Exp(NO), Exp(STIM)]   gap   [Model(NO), Model(STIM)]
        bar_width   = 0.18
        small_gap   = 0.10   # separation between Exp pair and Model pair
        group_gap   = 0.60   # separation between mutant groups
        group_width = 4*bar_width + small_gap

        centers = np.arange(n) * (group_width + group_gap)
        lefts = centers - group_width/2

        x_exp_no   = lefts + 0*bar_width
        x_exp_stim = lefts + 1*bar_width
        x_mod_no   = lefts + 2*bar_width + small_gap
        x_mod_stim = lefts + 3*bar_width + small_gap

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Colors and styles
        c_no, c_stim = 'black', 'red'
        hatch_model = '///'

        # EXP: solid fills (with error bars)
        ax.bar(x_exp_no,   D_exp_pre,  width=bar_width, yerr=D_exp_pre_sem,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, D_exp_post, width=bar_width, yerr=D_exp_post_sem, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   D_model_pre,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, D_model_post, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels(mutants, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0)

        # Legend: show meaning of color and style separately (cleaner)
        color_handles = [
            Patch(facecolor=c_no,   edgecolor='black', label='NO STIM'),
            Patch(facecolor=c_stim, edgecolor='black', label='STIM')
        ]
        style_handles = [
            Patch(facecolor='black', edgecolor='black', label='Experiment'),
            Patch(facecolor='white', edgecolor='black', hatch=hatch_model, label='Model')
        ]

        # Move legends above the plot
        leg1 = ax.legend(
            handles=color_handles,
            fontsize=fontsize*0.75,
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.35, 1.15)  # left group
        )
        ax.add_artist(leg1)

        ax.legend(
            handles=style_handles,
            fontsize=fontsize*0.75,
            frameon=False,
            loc='upper center',
            bbox_to_anchor=(0.65, 1.15)  # right group
        )

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()

    def save_individual_solution_figures_combined(self, output_dir="../fig/individual_figures", best=100, totalTime=10.0, dt=0.1, figsize=(16, 20), dpi=300):
        os.makedirs(output_dir, exist_ok=True)
        self.filter_and_store_solutions(best=best, totalTime=totalTime, dt=dt)

        for idx, candidate_params in enumerate(self.filteredSolutions):
            print(f"Generating combined figure for solution #{idx}")

            fig, axes = plt.subplots(nrows=5, ncols=2, figsize=figsize, dpi=dpi)
            axes = axes.flatten()

            self._draw_plot_copies(candidate_params, axes[0])
            self._draw_plot_kinetic(candidate_params, axes[1])
            self._draw_plot_diffusion_comparison(candidate_params, axes[2])
            self._draw_plot_diffusion_dilute_only(candidate_params, axes[3])
            self._draw_plot_cluster_percentage_total(candidate_params, axes[4])
            self._draw_plot_cluster_percentage_2D(candidate_params, axes[5])
            self._draw_plot_DC2A(candidate_params, axes[6])
            self._draw_plot_shRIM(candidate_params, axes[7])
            self._draw_plot_lower_lipid_binding(candidate_params, axes[8])
            self._draw_plot_lower_conc_munc13(candidate_params, axes[9])

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"solution_{idx:03d}.png"), bbox_inches="tight")
            plt.close(fig)

    def _draw_plot_copies(self, candidate_params, ax):
        _, _, _, _, pre, post, preEq, postEq, _, _, _, _, _, _ = self.calculate_results(candidate_params)
       # ax.plot(self.timePoints, pre / self.Atotal, label='NO STIM')
        #ax.plot(self.timePoints, post / self.AtotalDense, label='STIM')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Munc13$_{mem}$ (copies/$\mu$m$^2$)')
        ax.set_title("Membrane Munc13")
        ax.legend()

    def _draw_plot_kinetic(self, candidate_params, ax):
        _, _, preCluster, postCluster, _, _, preEq, postEq, _, _, _, _, _, _ = self.calculate_results(candidate_params)
        ax.plot(self.timePoints, preCluster, label='NO STIM')
        ax.plot(self.timePoints, postCluster, label='STIM')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Munc13 in Cluster')
        ax.set_title("Clustered Munc13 Kinetics")
        ax.legend()

    def _draw_plot_diffusion_comparison(self, candidate_params, ax):
        _, _, _, _, d_pre, d_post = self.simulate(candidate_params)
        model_mean = [d_pre, d_post]
        model_sem = [0, 0]  # No SEM unless you're running replicates

        exp_vals = [self.D_exp_pre, self.D_exp_post]
        exp_sem = [self.D_exp_pre_sem, self.D_exp_post_sem]

        categories = ["NO STIM", "STIM"]
        x = np.arange(len(categories))
        width = 0.35
        fontsize = 12

        ax.bar(x - width/2, exp_vals, width, yerr=exp_sem, capsize=5,
            label="Experiment", color='red', alpha=0.8)
        ax.bar(x + width/2, model_mean, width, capsize=5,
            label="Model", color='blue', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=fontsize)
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_title("Average Diffusion: Sim vs Exp")
        ax.legend(fontsize=fontsize, loc='best')

    def _draw_plot_diffusion_dilute_only(self, candidate_params, ax):
        D1 = candidate_params[9]
        D2 = D1 / candidate_params[10]
        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate_params)
        D_dilute_pre = self.calc_diffusion_dilute(D1, D2, solutionPre)
        D_dilute_post = self.calc_diffusion_dilute(D1, D2, solutionPost)

        ax.bar([0, 1], [D_dilute_pre, D_dilute_post],
            tick_label=["NO STIM", "STIM"], color='skyblue')
        ax.set_ylabel("Dilute Diffusion (µm²/s)")
        ax.set_title("Dilute Diffusion")

    def _draw_plot_cluster_percentage_total(self, candidate_params, ax):
        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate_params)
        perc_cluster_pre = self.calc_percentages_cluster(solutionPre)
        perc_cluster_post = self.calc_percentages_cluster(solutionPost)

        ax.bar([0, 1], [perc_cluster_pre, perc_cluster_post],
            tick_label=["NO STIM", "STIM"], color='salmon')
        ax.set_ylabel("Clustered % (Total)")
        ax.set_title("Total Clustered Percentage")

    def _draw_plot_cluster_percentage_2D(self, candidate_params, ax):
        solutionPre, solutionPost, _, _, _, _ = self.simulate(candidate_params)
        _, perc_cluster_pre_2D = self.calc_3D_2D_percentages_cluster(solutionPre)
        _, perc_cluster_post_2D = self.calc_3D_2D_percentages_cluster(solutionPost)

        ax.bar([0, 1], [perc_cluster_pre_2D, perc_cluster_post_2D],
            tick_label=["NO STIM", "STIM"], color='lightgreen')
        ax.set_ylabel("Clustered % (2D only)")
        ax.set_title("2D Clustered Percentage")

    def _draw_plot_DC2A(self, candidate_params, ax):
        candidate = list(candidate_params)
        c3 = candidate[3]
        candidate[3] = 0
        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        candidate[3] = c3

        model_mean = [d_pre, d_post]
        model_sem = [0, 0]

        exp_vals = [self.D_exp_DC2A_pre, self.D_exp_DC2A_post]
        exp_sem = [self.D_exp_DC2A_pre_sem, self.D_exp_DC2A_post_sem]

        categories = ["NO STIM", "STIM"]
        x = np.arange(len(categories))
        width = 0.35
        fontsize = 12

        ax.bar(x - width/2, exp_vals, width, yerr=exp_sem, capsize=5,
            label="Experiment", color='red', alpha=0.8)
        ax.bar(x + width/2, model_mean, width, capsize=5,
            label="Model", color='purple', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=fontsize)
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_title("DC2A Diffusion")
        ax.legend(fontsize=fontsize, loc='best')

    def _draw_plot_shRIM(self, candidate_params, ax):
        candidate = list(candidate_params)
        c3 = candidate[3]
        candidate[3] *= 2
        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        candidate[3] = c3

        model_mean = [d_pre, d_post]
        model_sem = [0, 0]

        exp_vals = [self.D_exp_shRIM_pre, self.D_exp_shRIM_post]
        exp_sem = [self.D_exp_shRIM_pre_sem, self.D_exp_shRIM_post_sem]

        categories = ["NO STIM", "STIM"]
        x = np.arange(len(categories))
        width = 0.35
        fontsize = 12

        ax.bar(x - width/2, exp_vals, width, yerr=exp_sem, capsize=5,
            label="Experiment", color='red', alpha=0.8)
        ax.bar(x + width/2, model_mean, width, capsize=5,
            label="Model", color='orange', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=fontsize)
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_title("shRIM Diffusion")
        ax.legend(fontsize=fontsize, loc='best')

    def _draw_plot_lower_lipid_binding(self, candidate_params, ax):
        candidate = list(candidate_params)
        c0 = candidate[0]
        candidate[0] *= 0.5
        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        candidate[0] = c0

        model_mean = [d_pre, d_post]
        model_sem = [0, 0]

        if d_pre > d_post:
            print("lower lipid binding d_pre > d_post")
            print(candidate)

        exp_vals = [self.D_exp_DC2B_pre, self.D_exp_DC2B_post]
        exp_sem = [self.D_exp_DC2B_pre_sem, self.D_exp_DC2B_post_sem]

        categories = ["NO STIM", "STIM"]
        x = np.arange(len(categories))
        width = 0.35
        fontsize = 12

        ax.bar(x - width/2, exp_vals, width, yerr=exp_sem, capsize=5,
            label="Experiment", color='red', alpha=0.8)
        ax.bar(x + width/2, model_mean, width, capsize=5,
            label="Model", color='green', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=fontsize)
        ax.set_ylabel(r"$\mathrm{Average\ Diffusion}\ (\mu\mathrm{m}^2/\mathrm{s})$", fontsize=fontsize)
        ax.set_title("Lower Lipid Binding (DC2B)")
        ax.legend(fontsize=fontsize, loc='best')

    def _draw_plot_lower_conc_munc13(self, candidate_params, ax):
        candidate = list(candidate_params)
        c7 = candidate[7]
        candidate[7] *= 0.1  # reduce Munc13 initial conc
        _, _, _, _, d_pre, d_post = self.simulate(candidate)
        ax.bar([0, 1], [d_pre, d_post], tick_label=["NO STIM", "STIM"], color='gray')
        ax.set_title("Lower Munc13 Concentration")
        candidate[7] = c7


class Solver:
    def __init__(self, model, populationSize=500, NGEN=5, outfileName="../data/optimizedParms_clusterRun.txt"):
        """
        Sets up a DEAP-based Genetic Algorithm to optimize parameters for Munc13 model.
        :param model: A Munc13 model instance
        :param populationSize: number of individuals in each GA generation
        :param NGEN: number of generations
        """
        self.model = model
        self.populationSize = populationSize
        self.NGEN = NGEN
        self.indpb = 0.75  # Probability of mutation per gene
        self.outfileName = outfileName
        # Create DEAP classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Candidate", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        # Register "gene generator"
        self.toolbox.register("candidate", self.generateCandidate)
        # Define how to build a population of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.candidate)
        # Genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutateCandidate, indpb=self.indpb, mult=0.5)
        self.toolbox.register("select", tools.selRoulette)

        # Register the main evaluation function with DEAP
        # (We only pick one from model.modes if you have multiple)
        self.toolbox.register("evaluate", self.model.eval_both_clusterModel)

        # Create a multiprocessing pool for parallel fitness evaluation
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        self.toolbox.register("map", self.pool.map)

    def generateCandidate(self):
        """
        Generate one random candidate solution by sampling each parameter
        in log10 space between [log10(min), log10(max)].
        """
        candidate = []
        for i in range(self.model.n_params):
            param_name = self.model.params_to_optimize[i]
            low = self.model.parameter_ranges[param_name]["min"]
            high = self.model.parameter_ranges[param_name]["max"]
            logval = np.random.uniform(math.log10(low), math.log10(high))
            candidate.append(10 ** logval)
        return creator.Candidate(candidate)

    def mutateCandidate(self, candidate, indpb, mult):
        """
        Mutates each gene with probability (1 - indpb). 
        The new value is scaled by 'rnd2' in [1-mult, 1+mult].
        Clamps to min/max if out of range.
        """
        for idx, val in enumerate(candidate):
            if np.random.rand() >= indpb:
                scale_factor = np.random.uniform(1 - mult, 1 + mult)
                candidate[idx] = val * scale_factor

                # Clamp to [min, max]
                param_name = self.model.params_to_optimize[idx]
                min_val = self.model.parameter_ranges[param_name]["min"]
                max_val = self.model.parameter_ranges[param_name]["max"]
                candidate[idx] = max(min_val, min(candidate[idx], max_val))
        return (candidate,)

    def findNominalValues(self):
        """
        Main GA loop: 
          - create initial population
          - run for NGEN generations
          - track any 'viable' solutions above the threshold
        Returns final sets of (viablePoints, viableFitness, twoDVals).
        """
        # Initialize population
        population = self.toolbox.population(n=self.populationSize)

        # Will store all viable solutions across generations
        all_viable_points = []
        all_viable_fitness = []

        # GA main loop
        for gen in range(self.NGEN):
            # Variation (recombination+mutation)
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.8, mutpb=0.2) #0.8, 0.02
            # Evaluate in parallel
            fits = list(self.toolbox.map(self.toolbox.evaluate, offspring))

            for fit_val, ind in zip(fits, offspring):
                cost, = fit_val
                ind.fitness.values = (cost,)

                # Check viability
                #print("cost: ", cost)
                if cost >= self.model.threshold:
                    # Store if not already in the set
                    if ind not in all_viable_points:
                        all_viable_points.append(ind[:])  # store a copy
                        all_viable_fitness.append([cost])

            # Selection step
            population = self.toolbox.select(offspring, k=len(population))

            if gen % 2 == 0:
                print(f"=== Generation {gen} ===")
                print(f"Number of viable points so far: {len(all_viable_points)}")

        return (all_viable_points, all_viable_fitness)

    def write_sortedParms(self, viablePoints, viableFitness):
        """
        Sort viable solutions by fitness, write the top solutions to a file.
        """
        #outfile = "../data/optimizedParms_clusterRun.txt"

        outfile = self.outfileName
        print(f"Writing results to {outfile}")

        # Prepare arrays for sorting
        fit_values = [vf[0] for vf in viableFitness]
        neg_fit_values = [-v for v in fit_values]

        # Sort indices by negative fitness (so best->worst)
        idx_sorted = np.argsort(neg_fit_values)

        with open(outfile, "w") as f:
            # CSV header
            header = (
                "Rank, Fitness, kfsr, krsr, kfmm, krmm, kf1x, kr1x, kfc, krc, krmmTriCoop, kx2, krx2, S0, R0, X0, X0_stimScale\n"
            )
            f.write(header)

            for rank, idx in enumerate(idx_sorted):
                fitness_val = fit_values[idx]
                opt_params = viablePoints[idx]

                # Unpack your optimized parameters in the known order
                # the order is set in self.model.params_to_optimize
                # so it matches the candidates, and not the order of the ODE parameters
                kfsr = opt_params[0]
                krsr = opt_params[1]
                #krsr_stim = params[2]
                kfmm = opt_params[2]
                krmm = opt_params[3]
                kf1x = opt_params[4]
                kr1x = opt_params[5]
                kfc = opt_params[6]
                krc = opt_params[7]
                krmmTriCoop = opt_params[8]
                kx2 = opt_params[9]
                krx2 = opt_params[10]

                S0 = opt_params[11]
                R0 = opt_params[12]
                
                X0 = opt_params[13]
                X0_stimScale = opt_params[14]

                # Write a line per solution
                f.write(
                    f"{rank}, {fitness_val:.4f}, "
                    f"{kfsr:.5g}, {krsr:.5g}, {kfmm:.5g}, "
                    f"{krmm:.5g}, {kf1x:.5g}, {kr1x:.5g}, "
                    f"{kfc:.5g}, {krmmTriCoop:.5g}, {kx2:.5g}, "
                    f"{krx2:.5g}, {krc:.5g}, {S0:.5g}, "
                    f"{R0:.5g}, {X0:.5g}, {X0_stimScale:.5g}\n"
                )

        print("Finished writing parameter file.")

    def run(self):
        """
        A convenience method to run the entire pipeline:
          1) findNominalValues (GA)
          2) sort & write results
        """
        viablePoints, viableFitness = self.findNominalValues()
        self.write_sortedParms(viablePoints, viableFitness)

        # IMPORTANT: close the pool to avoid zombie processes
        self.pool.close()
        self.pool.join()

        print("Done with all GA generations and file writing.")
        return viablePoints, viableFitness
    
#this is the main function to run the GA on the model defined below.
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
        "kfmm":      {"min": 0.001, "max": 10},   # kfMM uM-1s-1
        "krmm":      {"min": 0.01,   "max": 1000}, # krMM s-1
        "kf1x":      {"min": 0.001, "max": 10},   # kf1x uM-1s-1
        "kr1x":      {"min": 0.01,   "max": 1000}, # kr1x s-1
       # "kfc_nostim":      {"min": 0.001, "max": 10},   # kfc_nostim
        "kfc":      {"min": 0.1, "max": 10},   # kfc_stim uM-1s-1
        "krc":      {"min": 0.01, "max": 100},   # krc
        
        "krmmTriCoop":  {"min": 0.1, "max": 1},   # scale factor for trimer off rate, 10 weakens the loop, 0.1 strengthens it
        "kx2":      {"min": 0.01, "max": 10},   # kx2 uM-1s-1
        "krx2":      {"min": 0.01,   "max": 1000}, # krx2 s-1
        "S0":        {"min": 0.001, "max": 10},   # S0 (uM)
        "R0":        {"min": 1, "max": 1000},   # R0 (/um^2)
        #"D1":        {"min": 0.05,   "max": 5}, # D1
        #"D1_over_D2":        {"min": 1.5,   "max": 5}, # D2
        "X0":        {"min": 0.1,   "max": 500}, # X0  (/um^2)
        "X0_stimScale": {"min": 1,   "max": 10}, # SCALAR to increase X0 after stim
        "kfc_stimScale": {"min": 1,   "max": 100}, # SCALAR to increase kfc after stim
        "kfsr_stimScale": {"min": 1,   "max": 100}, # SCALAR to increase kfsr after stim
    }

    # Order in which the solver will read parameters from a candidate
    params_to_optimize = np.array([
        "kfsr","krsr","kfmm","krmm","kf1x","kr1x","kfc","krc","krmmTriCoop","kx2","krx2","S0","R0","X0","X0_stimScale","kfc_stimScale","kfsr_stimScale"
    ])
    print("number of parameters to optimize: ", params_to_optimize.size)
    # GA settings
    popSize = 5000
    nGen = 5

    # Instantiate the model and solver, including max time limit.
    maxTime = 1000.0
    model = Munc13(parameter_ranges, params_to_optimize, t_max=maxTime)
    #the solver can be passed a specific filename for the solutions.
    random_number = np.random.randint(1, 10000)
    filename = f"../data/testParms_clusterRun_{random_number}.txt"
    print("Output filename: ", filename)
    solver = Solver(model, populationSize=popSize, NGEN=nGen, outfileName=filename)

    # For testing: simulate and plot one candidate solution
   # print('Test one candidate solution...')
   # model.test_one_candidate()

    print("Run the GA ", filename)
    # Run the GA
    viablePoints, viableFitness = solver.run()

    #look at one solution
    #y1= model.simulate_pre(viablePoints[0])
    #model.plot_mycluster_time(y1, figsize=(8,6), dpi=300)

    
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

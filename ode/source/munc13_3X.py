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
        #We need to reduce the cluster density--the one they report
        #of ~1 cluster/um2 is integrated over 320s of time.
        #the lifetime of each cluster is 10s, hence the below estimate of instantaneous density.

        #these numbers are from the boosh analysis, but the mutants use nastic
        #self.density_exp_post = 1.675*10/320  # clusters per um^2, post-stim
        #self.density_exp_pre = 1.176*10/320  # clusters per um^2, pre-stim
        #NASTIC: 0.776±0.0995. STIM: 1.321±0.169
        self.density_exp_pre = 0.78*10/320
        self.density_exp_post = 1.32*10/320

        self.recruitmentStim = 3 #increase of munc13 on the membrane upon stimulation
        
        #Same observables as above, but for the delta C2A mutant with impaired dimerization.     
        self.density_c2a_pre = 0.35*10/320
        self.density_c2a_post = 0.58*10/320
        self.recruitmentStimC2A = 3

        self.DtM = 0.08 #um2/s
        self.DtX = 0.001 #um2/s
        self.DtD = 0.04 #um2/s twice as slow as DM.

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
        self.modes = [self.fitness_function_to_call]

        # Minimum threshold for a "viable" solution
        self.threshold = -500


    # -----------------------------
    #     ODE system definitions
    # -----------------------------
    
       

    def munc13_hexamerOde_loopCa(self, t, y, params):
        """
        Maggie Johnson, Oct 2025
        Munc13 model with explicit clustering. 
        We will no longer enforce a specific cluster regime, so we want the
        density of the clusters to match those observed experimentally.
        To simplify the assembly pathways, 
        We require X to be present to nucleate a cluster.
        There is stoichiometry of 2M to 1X.
        The Ms can exist as 2M or convert to a dimer D. The dimer state
        has a more stable free energy, including in the cluster state an extra stability.
        Hence removing the dimerization interaction does impact clustering.
        We assume the nucleators X are required to reach the hexamer,
        so you cannot have 6 Ms without 3 Xs.
        
        THIS IS CHANGED FROM ABOVE TO REMOVE THE EXTRA STABILIZATION BY dGC UNTIL LOOP CLOSURE.
        Now the extra stabilitation will come from eMLoop, so dGC is only during growth.
        For the D-hexamer, the loop stability is eDLoop.
        This is CHANGED from munc13_hexamerOde_loop also by the addition of a new membrane specie C

        y[0]  = S, Munc13 in solution
        y[1]  = R, General recruiter on the membrane
        y[2]  = M, Munc13 on the membrane (out of the cluster)
        y[3]  = D, Munc13 dimer on the membrane (out of the cluster)
        y[4]  = X, New--Cluster nucleator, also in 2D but dilute (well-mixed)
        y[5]  = MX, Munc13 on the membrane + X 
        y[6]  = SX, Munc13 in solution+ X
        y[7]  = M2X, two muncs + X 
        y[8]  = DX, dimer + X
        y[9]  = M2X2, two muncs + two X
        y[10] = DX2, dimer + two X
        y[11] = M3X2, three muncs + two X 
        y[12] = DMX2, dimer+munc+ two X
        y[13] = M4X2, four munc + two X
        y[14] = DM2X2. dimer + two munc + two X
        y[15] = D2X2, two dimer + two X
        y[16] = M4X3, four muncs + three X
        y[17] = D2X3, two dimers + three X
        y[18] = M5X3, five muncs + three X
        y[19] = D2MX3, two dimers + munc + three X
        y[20] = M6X3, six muncs + three X
        y[21] = D2M2X3, two dimers + two muncs + three X
        y[22]  = D3X3, three dimers + three X
        y[23] = C, the inhibiting channel.
        y[24] = CM, channel + monomer
        y[25] = CD, channel + dimer
        y[26] = CHEX, channel + hexamer
        y[27] = S2, dimer in solution.
        
                
                Reaction equations:
        1-19
            1. S + R <-> M, kfsr, krsr
            2. M + M <-> D, gamma*kfmm, krmm
            3. M + X <-> MX, gamma*kfmx, krmx
            4. S + X <-> SX, kfmx, krmx
            5. SX + R <-> MX, gamma*kfsr, krsr
            6. M + MX <-> M2X, gamma*kfc, krc
            7. D + X <-> DX, gamma*kfmx*S, krmx*S*np.exp(dGC)*eDF 
            8. M2X<->DX kfDD, krDD dG=dMM+eDF
            9.  X + M2X <-> M2X2, gamma*kfxx, krxx
            10. DX+ X <->DX2, gamma*S*kfxx, S*krxx
            11. M2X2 <-> DX2, kfDD, krDD
           12. M2X2+M <-> M3X2 gamma*kfmx, krmx
           13. DX2+M <-> DMX2 gamma*kfmx, krmx
          ** 14. M3X2+M <->M4X2 gamma*kfc, krc
          ** 15)	DMX2+M <->DM2X2 gamma*kfc, krc
            16)	DM2X2 <->D2X2 kfDD, krDD
          ** 17)	DX2+D<->D2X2 gamma*kfmx, krmx*exp(dGC)*eDF (slower than r15) 
            18)	D2X2+X <-> D2X3 gamma*S*kfxx, S*krxx*exp(dGX)
          ** 19)	D2X3+D<->D3X3  gamma*kfmx, krmx*exp(dGC)*eDF*eDLoop. 
            20)	M4X2+X <->M4X3 gamma*kfxx, krxx*exp(dGXX) 
            21)	M4X3+M <->M5X3 gamma*kfmx, krmx
           ** 22)	M5X3+M <->M6X3 gamma*kfc, krc*eMLoop
            23)	D2X3+M <->D2MX3 gamma*kfmx, krmx
           ** 24)	D2MX3+M <->D2M2X3 gamma*kfc, krc*eMLoop #loop closure with 6. 
            25)	D2M2X3<->D3X3 kfDD, krDD*eDLoop/eMLoop #switch from M to D loop stability.
            26)	S+R+MX->M2X  gamma*kfc*kfsr/krsr
            27)	S+R+M2X2->M3X2 gamma*kfmx*kfsr/krsr
            28)	S+R+DX2 ->DMX2 gamma*kfmx*kfsr/krsr
            29)	S+R+M3X2->M4X2 gamma*kfc*kfsr/krsr
            30)	S+R+DMX2->DM2X2 gamma*kfc*kfsr/krsr
            31)	S+R+M4X3->M5X3 gamma*kfmx*kfsr/krsr
            32)	S+R+D2X3->D2MX3 gamma*kfmx*kfsr/krsr
            33)	S+R+M5X3->M6X3 gamma*kfc*kfsr/krsr
            34)	S+R+D2MX3->D2M2X3 gamma*kfc*kfsr/krsr
            #ALLOW X to add onto clusters even if monomers are attached.

            35) DX+DX <->D2X2 dGXX: gamma*kfxx, krxx
            36) M2X+M2X <->M4X2 dGXX: gamma*kfxx, krxx
            37) D2X2+DX <->D3X3 2dGXX+2dGC: gamma*kfxx, krxx*exp(dGX)*eDLoop
            38) M4X2 + M2X <->M6X3 2dGXX + 2dGC: gamma*kfxx, krxx*exp(dGX)*eMLoop
            #Include mixtures of D and M
            39) DX + M2X <->DM2X2 dGXX: gamma*kfxx, krxx
            40) D2X2+M2X <->D2M2X3: 2*dGXX: gamma*kfxx, krxx*exp(dGX)*eMLoop
            41) DM2X2 + DX <-> D2M2X3 2*dGXX: gamma*kfxx, krxx*exp(dGX)*eMLoop
            #reactions with the channel C
            42) C+ M <-> CM dGCM, kfcm, krcm
            43) C + D <-> CD dGCD, kfcd, krcd, should be stronger than dGCM
            44) C + D3X3 <-> CHEX dGCHEX, kfchex, krchex
            45) C+ M6X3 <-> CHEX dGCHEX, kfchex, krchex
           #do not include 46, because the backward flux creates mixed species
           that do not exist without dimer reactions.
             46) C+D2M2X3 <-> CHEX dGCHEX, kfchex, krchex

             #add in solution dimerization and dimer recruitment to 2R
             47) S+S <->S2 kfmm, krmm
             48) S2 + 2R <-> D 2*gamma*kfsr*1e-6 (units uM-2 s-1), krsr*exp(dGSR). 
             exp(dGSR)= krsr/kfsr/c0, where c0=1e6uM. 
             S2 + R  <->S2R 2*kfsr, krsr: 
             i S2R+R <-> S2R2 gamma*kfsr, 2*krsr
             S2*R/S2R = krsr/(2kfsr) (uM)
             ii S2R = S2*R/krsr*2*kfsr 
             iii S2R*R/S2R2 = 2*krsr/(gamma*kfsr) (uM)
             insert ii into iii:
             S2*R*R/S2R2 = 2*krsr/(gamma*kfsr*2)*c0*exp(dG) (uM^2)
             --> too fast onS2*R^2 <->S2R2 with on rate: gamma*kfsr^2*2/krsr, off rate: 2*krsr
             on rate: 2*gamma*kfsr/c0, where c0=1M or 1e6uM. off rate: 2*krsr*exp(dG)
        -----------          

        
        """
        kfsr = params[0]
        krsr = params[1]
        gamma = self.gamma
        kfmm = params[2]
        krmm = params[3]
        kfmx = params[4]
        krmx = params[5]
        kfc = params[6]
        krc = params[7]
        
        kfxx = params[8]
        krxx = params[9]
        #sig=params[10] #scale factor to accelerate dimer to cluster
        eMLoop = params[10] # monomer loop stability
        
        eDF = params[11] #cooperative free energy for dimer contacts with the cluster.
        kfdd = params[12] #rate of transitioning in the cluster from 2 monomers to dimers.
        
        kfcm = params[13]
        krcm = params[14]
        eCA = params[15]
        kclus = params[16]
        eDLoop = params[17] #dimer loop stability.
        # krdc = params[12] constrainted by free energies.
        #dG values are in units of kT
        #dGM = np.log(krmm*1e-6/kfmm) #K_D is in units of M, divide out c0=1M standard state. 
        dGC = np.log(krc*1e-6/kfc) #K_D is in units of M, divide out c0=1M standard state. 
        if(kfmm==0):
            dGM = 100
        else:
            dGM = np.log(krmm*1e-6/kfmm) #K_D in units of M
        
        if(kfmx==0):
            dGMX = 100
        else:
            dGMX = np.log(krmx*1e-6/kfmx) #K_D in units of M
        
        #dGSR = np.log(krsr*1e-6/kfsr) #K_D in units of M
        #dGSR changes under stimulation.
        dGXX = np.log(krxx*1e-6/kfxx) #K_D in units of M
        krdd = kfdd*np.exp(dGM)*eDF  #the ratio of on/off for the 2M<->D transition is controlled by other free energies
        krdd2 = kfdd*np.exp(dGM)*eDF #All additions of D use eDF.
        #chgDG=eMLoop/eDF #replace the eDF stabilization with the eMLoop stabilization
        
        #krxTrimer = params[12]
        #enhanceRate = 10 #scalare that increase the on-rate when more than one X is present
        #krmxDimer=krmx*sig*np.exp(dGC)*eDF # off rate for MX slowed by C and F=dG
        #kfast=max(kfc, kfmx)  #two interfaces, choose faster rate
        #kr_cmx = kfast*np.exp(dGMX+dGC) #reverse reaction with MX and C bonds=dG

        #leave stoichiometric factors for the dy expressions
        r1 = kfsr*y[0]*y[1] #in 3D
        r1b = krsr*y[2]
        r2 = gamma*kfmm*y[2]*y[2]
        r2b = krmm*y[3]
        r3 = gamma*kfmx*y[2]*y[4]
        r3b = krmx*y[5]
        r4 = kfmx*y[0]*y[4]  #in 3D
        r4b = krmx*y[6]
        r5 = gamma*kfsr*y[6]*y[1]
        r5b = krsr*y[5]
        r6 = gamma*kfc*y[2]*y[5]
        r6b = krc*y[7]
        r7 = gamma*kfmx*y[3]*y[4]
        r7b = krmx*np.exp(dGC)*eDF*y[8]
        r8 = kfdd*y[7]
        r8b = krdd*y[8]
        r9 = gamma*kfxx*y[4]*y[7]
        r9b = krxx*y[9]
        r10 = gamma*kfxx*y[4]*y[8]   
        r10b = krxx*y[10]
        r11 = kfdd*y[9]
        r11b = krdd*y[10]
        r12 = gamma*kfmx*y[9]*y[2]
        r12b = krmx*y[11]
        r13 = gamma*kfmx*y[2]*y[10]
        r13b = krmx*y[12]
        r14 = gamma*kfc*y[2]*y[11]
        r14b = krc*y[13]
        r15 = gamma*kfc*y[2]*y[12]
        r15b = krc*y[14]
        r16 = kfdd*y[14]
        r16b = krdd2*y[15]
        r17 = gamma*kfmx*y[10]*y[3]
        r17b = krmx*np.exp(dGC)*eDF*y[15]
        r18 = gamma*kfxx*y[15]*y[4]
        r18b = krxx*np.exp(dGXX)*y[17]
        r19 = gamma*kfmx*y[17]*y[3]
        r19b = krmx*np.exp(3*dGC)*eDF*eDLoop*y[22]
        r20 = gamma*kfxx*y[13]*y[4]
        r20b = krxx*np.exp(dGXX)*y[16]
        r21 = gamma*kfmx*y[16]*y[2]
        r21b = krmx*y[18]
        r22 = gamma*kfc*y[2]*y[18]
        r22b = krc*y[20]*eMLoop
        r23 = gamma*kfmx*y[17]*y[2]
        r23b = krmx*y[19]
        r24 = gamma*kfc*y[19]*y[2]
        r24b = krc*y[21]*eMLoop
        r25 = kfdd*y[21]
        r25b = krdd2*y[22]*eDLoop/eMLoop
        r26 = 0# gamma*kfc*kfsr/krsr*y[0]*y[1]*y[5]
        r27 = 0#gamma*kfmx*kfsr/krsr*y[0]*y[1]*y[9]
        r28 = 0#gamma*kfmx*kfsr/krsr*y[0]*y[1]*y[10]
        r29 = 0#gamma*kfc*kfsr/krsr*y[0]*y[1]*y[11]
        r30 = 0#gamma*kfc*kfsr/krsr*y[0]*y[1]*y[12]
        r31 = 0#gamma*kfmx*kfsr/krsr*y[0]*y[1]*y[16]
        r32 = 0#gamma*kfmx*kfsr/krsr*y[0]*y[1]*y[17]
        r33 = 0#gamma*kfc*kfsr/krsr*y[0]*y[1]*y[18]
        r34 = 0#gamma*kfc*kfsr/krsr*y[0]*y[1]*y[19]

        r35 = gamma*kfxx*y[8]*y[8]
        r35b = y[15]*krxx 
        r36 = gamma*kfxx*y[7]*y[7]
        r36b = y[13]*krxx
        r37 = gamma*kfxx*y[8]*y[15]
        r37b = y[22]*krxx*np.exp(dGXX)*eDLoop
        r38 = gamma*kfxx*y[7]*y[13]
        r38b = y[20]*krxx*np.exp(dGXX)*eMLoop
        r39 = gamma*kfxx*y[8]*y[7]
        r39b = y[14]*krxx
        r40 = gamma*kfxx*y[15]*y[7]
        r40b = y[21]*krxx*np.exp(dGXX)*eMLoop
        r41 = gamma*kfxx*y[8]*y[14]
        r41b = y[21]*krxx*np.exp(dGXX)*eMLoop

        #add in reactions for binding the C channel specie on the membrane.
        r42 = y[23]*y[2]*gamma*kfcm
        r42b = y[24]*krcm
        r43 = y[23]*y[3]*gamma*kfcm
        r43b = y[25]*krcm*eCA
        r44 = y[23]*y[22]*gamma*kclus
        #r44b = y[26]*krclus make it irreversible, 
        #otherwise, the reverse reactions have to be distinguished due to D vs M.
        r45 = y[23]*y[20]*gamma*kclus
        #r45b = y[26]*krclus
        r47 = y[0]*y[0]*kfmm
        r47b = y[27]*krmm  
        r48 = y[27]*y[1]*y[1]*2*gamma*kfsr*1e-6
        r48b = y[3]*krsr*krsr/kfsr*1e-6 #y[3]*krsr*np.exp(dGSR) 

        dylist = []
        #add each new species reaction rate.
        dylist.append(-r1+r1b-r4+r4b-r26-r27-r28-r29-r30-r31-r32-r33-r34-2*r47+2*r47b)#0 S
        dylist.append(-r1+r1b-r5+r5b-r26-r27-r28-r29-r30-r31-r32-r33-r34-2*r48+2*r48b)#1 R
        dylist.append(+r1-r1b-2*r2+2*r2b-r3+r3b-r6+r6b-r12+r12b-r13+r13b-r14+r14b-r15+r15b-r21+r21b-r22+r22b-r23+r23b-r24+r24b-r42+r42b)#2 M
        dylist.append(+r2-r2b-r7+r7b-r17+r17b-r19+r19b-r43+r43b+r48-r48b)#3 D
        dylist.append(-r3+r3b-r4+r4b-r7+r7b-r9+r9b-r10+r10b-r18+r18b-r20+r20b)#4 X
        dylist.append(r3-r3b+r5-r5b-r6+r6b-r26)#5 MX
        dylist.append(r4-r4b-r5+r5b)#6 SX
        dylist.append(r6-r6b-r8+r8b-r9+r9b+r26-2*r36+2*r36b-r38+r38b-r39+r39b-r40+r40b)#7 M2X
        dylist.append(r7-r7b+r8-r8b-r10+r10b-2*r35+2*r35b-r37+r37b-r39+r39b-r41+r41b)#8 DX
        dylist.append(r9-r9b-r11+r11b-r12+r12b-r27)#9 M2X2
        dylist.append(r10-r10b+r11-r11b-r13+r13b-r17+r17b-r28)#10 DX2
        dylist.append(r12-r12b-r14+r14b-r29+r27)#11 M3X2
        dylist.append(r13-r13b-r15+r15b-r30+r28)#12 DMX2
        dylist.append(r14-r14b-r20+r20b+r29+r36-r36b-r38+r38b)#13 M4X2
        dylist.append(r15-r15b-r16+r16b+r30+r39-r39b-r41+r41b)#14 DM2X2
        dylist.append(r16-r16b+r17-r17b-r18+r18b+r35-r35b-r37+r37b-r40+r40b)#15 D2X2
        dylist.append(r20-r20b-r21+r21b-r31)#16 M4X3
        dylist.append(r18-r18b-r19+r19b-r23+r23b-r32)#17 D2X3
        dylist.append(r21-r21b-r22+r22b-r33+r31)#18 M5X3
        dylist.append(r23-r23b-r24+r24b-r34+r32)#19 D2MX3
        dylist.append(r22-r22b+r33+r38-r38b-r45) #20 M6X3
        dylist.append(r24-r24b-r25+r25b+r34+r40-r40b+r41-r41b) #21 D2M2X3
        dylist.append(r19-r19b+r25-r25b+r37-r37b-r44) #22 D3X3
        dylist.append(-r42+r42b-r43+r43b-r44-r45)#23 C
        dylist.append(r42-r42b) #24 CM
        dylist.append(r43-r43b) #25 CD
        dylist.append(r44+r45) #26 CHEX
        dylist.append(r47-r47b-r48+r48b) #27 S2

        return np.array(dylist)
    "END OF HEXAMER ODE loopCa"

   
    # ------------------------------------------------------
    #        Utilities to compute Munc13 on cluster/membrane
    # ------------------------------------------------------
    def calculate_munc13_on_membrane(self, copies):
        """
        Calculate total copies of munc13 on the membrane.
        """
      
        memMunc = (copies[2] + 2 * copies[3]+ copies[5] + 2*copies[7]+2*copies[8]+\
                    2*copies[9]+2*copies[10]+3*copies[11]+3*copies[12]+\
                    4*copies[13]+ 4*copies[14]+ 4*copies[15]+ 4*copies[16]+\
                    4*copies[17]+ 5*copies[18]+ 5*copies[19]+ 6*copies[20]+\
                          6*copies[21]+6*copies[22]+copies[24]+2*copies[25]+6*copies[26])
                              
        #densMunc=memMunc/self.cellArea
        return memMunc


   

    def calculate_lifetime_of_clusters(self, sol, candidate):
        """
        Calculate the average lifetime of clusters 
        We will use our formula for the dwell time
        based on a two-state model, either bound in a cluster,
        or unbound states that can transition into the cluster. 
        Do not include any transitions that move from within the cluster 
        to the cluster, as those don't produce flux into the bound state.
        """
        #get the final concentrations of each cluster species
      
        #off rates of X from each species
        
        
        kfmx=candidate[4] #form MX
        
        kfc=candidate[6] #bind to cluster with M and X present
       
        
        kfxx=candidate[8] #add second X to trimer
        
        gamma = self.gamma
        #numerator is sum over all bound states with correct stoichiometry
        #D3X3, D2M2X3, M6X3, M5X3, D2MX3, CHEX.
        y = sol[:,-1]
        #include both 5-mer and 6-mers in the bound state
        numerator46 = 6*(y[22] + y[21] + y[20] +y[26]) + 5*(y[19] + y[18])
        #include only 6-mers in the bound state 
        numerator56 = 6*(y[22] + y[21] + y[20] +y[26]) #+ y[19] + y[18] 
        #denominator is reactive flux of reactants multiplied by on rates. 
        #only reactions that go to the bound state, with reactants not in the bound state.
        #r19, (r21,  r23 are 4->5). (r24 and r22 are 5->6), so remove 5 from the clusters.

        #r37, r38, r40, r41
        #transitions must go from 4 (or less) to the bound state of 5-mer or 6-mer.
        #cannot go from 5-mer to 6-mer in this calculation, as 5 is in the bound state.
        denominator46 = gamma*kfmx*y[17]*y[3]+gamma*kfmx*y[16]*y[2]+\
                    gamma*kfmx*y[17]*y[2]+gamma*kfxx*y[8]*y[15]+\
                    gamma*kfxx*y[7]*y[13]+gamma*kfxx*y[15]*y[7]+\
                    gamma*kfxx*y[8]*y[14]
        #transitions must go to a 6-mer
        denominator56 = gamma*kfmx*y[17]*y[3]+gamma*kfc*y[19]*y[2]+\
                    gamma*kfc*y[2]*y[18]+gamma*kfxx*y[8]*y[15]+\
                    gamma*kfxx*y[7]*y[13]+gamma*kfxx*y[15]*y[7]+\
                    gamma*kfxx*y[8]*y[14]

        tau46 = numerator46/(6*denominator46) if denominator46>0 else 0
        tau56 = numerator56/(6*denominator56) if denominator56>0 else 0    
        return tau46, tau56
    

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
    def costChi_cluster(self, Y, densExpt):
        """
        Compute a chi-square-like cost for the "pre" data.
        Based on the cluster forming ODE model.
        Y is the solution in vector form, so all species are in units of uM. 
        densExpt is the expected value of cluster density from the 
        experimental condition
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
        
        #the small cluster density should be 10 x lower than the large density
        #chiSmallClust=-smallClusterDens[-1]/(density[-1]+1)
        factorLarger = 10
        chiSmallClust=-1*max(0, smallClusterDens[-1]*factorLarger-density[-1])
        
        #chiSmallClust = np.log10(smallClusterDens[-1])
        if(density[-1]<0):
            chiSmallClust=self.threshold*10 #prevent 0 densities from being favorable.
            print(f"WARNING: density of clusters is negative {density[-1]}")
        elif(smallClusterDens[-1]<0):
            chiSmallClust=0 #in this case density is positive but small clusters are at 0.
            print(f"WARNING2: density of small clusters is negative {smallClusterDens[-1]}")
        
        #add in a steady-state term to select for solutions that aren't still changing in time.
        #both the high density clusters and the low density clusters
        L=len(density)
        n75=np.int64(L*0.75)
        ssDelta=density[-1]-density[n75]
        #chiSS=-ssDelta*ssDelta/(density[-1]*density[-1])
        chiSS=-ssDelta*ssDelta/(densExpt*densExpt)
        ssDeltaLow=smallClusterDens[-1]-smallClusterDens[n75]
        chiSS=chiSS-ssDeltaLow*ssDeltaLow/(densExpt*densExpt)

       
        #densExpt=self.density_exp_post
        chiDens=self.costChi_densityCluster(density[-1],densExpt)
            
    
        
        costSum=0
        #we may need to add in a term to the cost function to select
        #against a high density of sub-clusters on the membrane,
        #that would be all species with say 3-6 munc13.
        #this is called chiSmallClust, we can weight it less than the experimental comparison.
        weightSmallClust=10.0 #this is the inverse of the target maximums
        #if we want small clusters to be <1e-4, set the weight to 1e4
        print(f"Small cluster density chi: Simulated {smallClusterDens[-1]}, high density {density[-1]}, Chi {chiSmallClust}")
        #calculate the percent of munc13 on the membrane that is in clusters.
        percClusterTotal, percClustMem, percMem, pMono, pDimer=self.calc_percentages_cluster(Y)
        print(f"Percent of mem munc13 in clusters: {percClustMem*100}%. Percent munc13 on the membrane: {percMem*100}%. Monomer mem: {pMono*100}%. Dimer mem: {pDimer*100}%")
        #implement a relu penalty if percent in clusters is greater than 40%
        weightPerc = 10
        chiPercClust=max(0, percClustMem-0.4)*-weightPerc
        #implement a relu penalty if percent in clusters is less than 5%
        chiPercClust =chiPercClust + max(0, 0.05 - percClustMem)*-weightPerc

        weightChiSS = 10
        print(f"Change in density over last 25%: {ssDelta}, associated chi: {chiSS*weightChiSS}")

        costSum=chiDens[0]+weightSmallClust*chiSmallClust + chiPercClust +chiSS*weightChiSS # this is already negative
        return [costSum, density[-1]]


    def calculate_cluster_density(self,Y):
        """
        given the solution of the cluster model, compute the density of clusters on the membrane
        """
        copies = Y *self.cellVolume * 602 #converts from uM to copy numbers.
        #we have to define what is a cluster. We can say that it requires 3 copies of X.
        #and at least 6 copies of munc13 (either S or M)
        #20-22 have 6 munc. 16-22 have 3 X. 18 and 19 have 5 munc.
        cluster_copies = copies[20]+copies[21]+copies[22]+copies[18]+copies[19]+copies[26]
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
        #we can ignore species that have only 1 X as being too small 
        cluster_copies = copies[11]+copies[12]+copies[13]+\
                        copies[14]+copies[15]+copies[16]+copies[17]
                        
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
        weight1 = 5.0 / (expDens ** 2)
       
        sum_diff = weight1 * diff1 
        print(f"Cluster density chi: Simulated {simDens}, Experimental {expDens}, Chi { -sum_diff}")
        #we want to maximize the fitness, so return the negative value.
        return [-sum_diff]

    def costChi_recruitmentStim(self, Ypre, Ypost, observedStim):
        """
        Compute a chi-square-like cost for the increase in the 
        amount of munc13 recruited to the membrane upon stimulation.
        This should be based on total munc13, in the clustered and nonclustered phases.
        """
        copiesPre = Ypre[:,-1] * self.cellVolume * 602 #converts from uM to copy numbers.
        #calculate the total density of munc13 on the membrane
        #print(copiesPre) 
        memCopyMuncPre =   self.calculate_munc13_on_membrane(copiesPre)

        copiesPost = Ypost[:,-1] * self.cellVolume * 602 #converts from uM to copy numbers.
        #calculate the total density of munc13 on the membrane
        #print(copiesPost)       
        memCopyMuncPost =   self.calculate_munc13_on_membrane(copiesPost)

        #ratio of densities on membrane at steady-state.
        recruitStim=memCopyMuncPost/memCopyMuncPre
        if(np.isnan(recruitStim)):
            recruitStim = 1.0
        # Calculate differences just for the last point.
        diff1 = (recruitStim - observedStim) ** 2
       
        # Weight prefactors to normalize the terms
        # Normalize by the square of expected values to make terms dimensionless and comparable
        weight = 10.0 / (observedStim ** 2)
        sum_diff=np.array(1)
        sum_diff = weight * diff1
        print(f"Recruitment copies pre {memCopyMuncPre}, and copies post {memCopyMuncPost}")

        print(f"Recruitment upon stimulation, sim value {recruitStim}, exp value {observedStim}, chi: Chi {-sum_diff}")
        return [-sum_diff]

    def isViableFitness(self, fit):
        return fit >= self.threshold


    def isViable(self, point):
        fitness = self.fitness_function_to_call(point)
        return self.isViableFitness(fitness[0])


    # ---------------------------------------------------
    #   Main evaluation: combine pre and post conditions
    # ---------------------------------------------------
  
    def fitness_function_to_call(self, candidate):
        #use this more general function so you can change the evaluation function
        #easily
        fitness = np.array(1)
        #fitness=self.eval_stimOnly(candidate)
        fitness = self.eval_noStim(candidate)    
        return [fitness[0]]

    def eval_both_clusterModel(self, candidate):
        """
        Simulate the model in pre and post conditions
        Evaluate the candidate solution to compute fitness
        Evaluate the fitness using the chi terms

        Include the evaluation of the different mutants.
        """
        solutionPre, solutionPost = self.simulate(candidate)

        # Chi from pre
        densExpt=self.density_exp_pre
        print("CHI PRE STIMULATION")
        chiPre, endDensityPre = self.costChi_cluster(solutionPre, densExpt)
        # Chi from post
        print("**CHI POST STIMULATION")
        chiPost, endDensityPost = self.costChi_cluster(solutionPost, self.density_exp_post)

        #chi from total change on the membrane
        #print(f" before chi recruitment stim, M on membrane: , {solutionPre[2][-1]}, and post {solutionPost[2][-1]}")
        chiRecruitStim= self.costChi_recruitmentStim(solutionPre,solutionPost, self.recruitmentStim)
        
        #add a term to select for weaker response under dimer mutation
        chiTotal=np.array(1)
        
        chiTotal = chiPre + chiPost+ chiRecruitStim[0]

        
        

        # Now evaluate the mutants, which will require new simulations
        #first is the mutant C2A, which eliminates dimerization.
        candidate_dc2a=list(candidate)
        candidate_dc2a[2] = 0 #this sets kfmm to zero.
        candidate_dc2a[12] = 0 #this sets kfdd to zero (no in cluster transition to 2M->D)
        
        mutantC2A_pre, mutantC2A_post = self.simulate(candidate_dc2a)
        # Chi from pre
        print("MUTANT CHI PRE STIMULATION")
        chiPre,endDensityMutantPre = self.costChi_cluster(mutantC2A_pre, self.density_c2a_pre)
        # Chi from post
        print("**MUTANT CHI POSTSTIMULATION")
        chiPost,endDensityMutantPost = self.costChi_cluster(mutantC2A_post, self.density_c2a_post)

        #chi from total change on the membrane
        chiRecruitStim= self.costChi_recruitmentStim(mutantC2A_pre,mutantC2A_post, self.recruitmentStimC2A)
        
        #add a term to select for weaker response under dimer mutation
        #add a term to select for weaker response under dimer mutation
        
        #add a term to ensure that the mutant has a lower density of clusters than the dimer.
        #say it should be at least 1.5 times lower. 
        chiMutant = -1*max(0, endDensityMutantPre-endDensityPre*1.5)
        weightMutant = 10 

        
        chiTotal = chiTotal+ chiPre + chiPost+ chiMutant*weightMutant + chiRecruitStim[0]

        
    
        if(chiTotal > 0):
            print(f"Positive chi found! {chiTotal}")
            print(f"Parameters: {candidate}")
            print(f"Chi pre: {chiPre}, Chi post: {chiPost}, Chi recruit: {chiRecruitStim}")
            print("ERROR")
            print("-------ERROR------")
            return [self.threshold*10]
        
        print(f"Total chi: {chiTotal}")

        return [chiTotal]
    

    def eval_stimOnly(self, candidate):
        """
        Simulate the model in pre and post conditions
        Evaluate the candidate solution to compute fitness
        Evaluate the fitness using the chi terms

        no mutants!
        """
        solutionPre, solutionPost = self.simulate(candidate)

        # Chi from pre
        densExpt=self.density_exp_pre
        print("CHI PRE STIMULATION")
        chiPre, endDensityPre = self.costChi_cluster(solutionPre, densExpt)
        # Chi from post
        print("**CHI POST STIMULATION")
        chiPost, endDensityPost = self.costChi_cluster(solutionPost, self.density_exp_post)

        #chi from total change on the membrane
        #print(f" before chi recruitment stim, M on membrane: , {solutionPre[2][-1]}, and post {solutionPost[2][-1]}")
        chiRecruitStim= self.costChi_recruitmentStim(solutionPre,solutionPost, self.recruitmentStim)
        
        #add a term to select for weaker response under dimer mutation
        chiTotal=np.array(1)
        
        chiTotal = chiPre + chiPost+ chiRecruitStim[0]

        
    
    
        if(chiTotal > 0):
            print(f"Positive chi found! {chiTotal}")
            print(f"Parameters: {candidate}")
            print(f"Chi pre: {chiPre}, Chi post: {chiPost}, Chi recruit: {chiRecruitStim}")
            print("ERROR")
            print("-------ERROR------")
            return [self.threshold*10]
        
        print(f"Total chi: {chiTotal}")

        return [chiTotal]
    
    def eval_noStim(self, candidate):
        """
        Simulate the model in pre conditions only.
        Evaluate the candidate solution to compute fitness
        Evaluate the fitness using the chi terms

        Include the evaluation of the different mutants.
        """
        solutionPre = self.simulate_pre(candidate)

        # Chi from pre
        densExpt=self.density_exp_pre
        chiPre, endDensityPre = self.costChi_cluster(solutionPre, densExpt)
        
        #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPre, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
        chiLifetime = -1*max(0,5 - tau56) -1*max(0,tau56 - 50)


        #chiTotal = chiPre[0] 
        chiTotal = chiPre+chiLifetime

        # Now evaluate the mutants, which will require new simulations
        #first is the mutant C2A, which eliminates dimerization.
        candidate_dc2a=list(candidate)
        candidate_dc2a[2] = 0 #this sets kfmm to zero.
        candidate_dc2a[12] = 0 #this sets kfdd to zero (no in cluster transition to 2M->D)
        
        mutantC2A_pre = self.simulate_pre(candidate_dc2a)
        # Chi from pre
        chiPre, endDensityMutant = self.costChi_cluster(mutantC2A_pre, self.density_c2a_pre)
       
        #add a term for cluster lifetime.
        tau46, tau56 = self.calculate_lifetime_of_clusters(mutantC2A_pre, candidate_dc2a)
        print(f"MUTANT Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
        chiLifetime = -1*max(0,5 - tau46) -1*max(0,tau46 - 50)
        
        #add a term to ensure that the mutant has a lower density of clusters than the dimer.
        #say it should be at least 1.5 times lower. 
        chiMutant = -1*max(0, endDensityMutant-endDensityPre*1.5)
        weightMutant = 10 

        chiTotal = chiTotal+ chiPre +chiMutant*weightMutant

        
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
        numVar = 28 #number of variables we are tracking. 
        kfsr=candidate[0] #form SR
        krsr=candidate[1] #dissociate SR
        kfmm=candidate[2] #form D
        krmm=candidate[3] #dissociate D
        kfmx=candidate[4] #form MX
        krmx=candidate[5] #dissociate MX
        kfc=candidate[6] #bind to cluster with M and X present
        krc=candidate[7] #reverse of above
        
        kfxx=candidate[8] #add second X to trimer
        krxx=candidate[9] #off rate of second X from trimer
        eMLoop=candidate[10] #speed up for dimers
        eDF = candidate[11] #exponential of free energy benefit eDF<=1 for dimer in cluster, slows off rate.
        kfdd = candidate[12] #rate of converting from 2M to D in the cluster.
        stimUpSR= candidate[13] #scalar increase of recruitment to membrane.
        S0=candidate[14] #initial Solution Munc13 (S), uM
        R0=candidate[15] #initial R, /um^2
        #D1=candidate[9] #Monomer (M) diffusion constant on membrane, um^2/s
        #D2=candidate[9] / candidate[10] #Dimer (D) diffusion constant on membrane, um^2/s
        X0 = candidate[16] #initial X /um^2
        C0 = candidate[17]
        # convert to uM
        R0 = R0*self.cellArea/self.cellVolume/602.0 
        X0 = X0*self.cellArea/self.cellVolume/602.0 
        kfcm = candidate[18]
        krcm = candidate[19]
        eCA = candidate[20]
        kclus = candidate[21]
        eDLoop = candidate[22]
        #rateParams=np.array([kfsr, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd])
        rateParams=np.array([kfsr, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd, kfcm, krcm, eCA, kclus, eDLoop])
        
        

        initValues=np.zeros(numVar)
        initValues[0]=S0
        initValues[1]=R0
        initValues[4]=X0
        initValues[23]=C0
        #print("initial values of X0", initValues[4])
        #print("rate parameters array:", rateParams)
        solution = scipy.integrate.solve_ivp(fun=self.munc13_hexamerOde_loopCa, method='BDF', t_span=(0, self.t_max), y0=initValues, t_eval=self.timePoints, args=(rateParams,), rtol=1e-7, atol=1e-9)

        #D = self.calc_diffusion(D1, D2, solution.y)

        return solution.y

    def simulate_post(self, candidate, solution_pre):
        """
   
        The candidate contains the sampled parameters.
        Their order is defined in the array "params_to_optimize"
        So it does not need to match the order in the ODE function.
        The list of params below needs to be passed to the ODE funtion, however

        Even if you put constraints on the candidates to be optimized, the ODE still takes in the same
        list of parameters. So 'candidate' can change shape, but the ODE parameters should not.

        """
        numVar = 28 #number of variables we are tracking. 
        kfsr=candidate[0] #form SR
        krsr=candidate[1] #dissociate SR
        kfmm=candidate[2] #form D
        krmm=candidate[3] #dissociate D
        kfmx=candidate[4] #form MX
        krmx=candidate[5] #dissociate MX
        kfc=candidate[6] #bind to cluster with M and X present
        krc=candidate[7] #reverse of above
        
        kfxx=candidate[8] #add second X to trimer
        krxx=candidate[9] #off rate of second X from trimer
        eMLoop=candidate[10] #speed up for dimers
        eDF = candidate[11] #free energy benefit eDF<0 for dimer in cluster.
        kfdd = candidate[12] #rate of converting from 2M to D in the cluster.
        stimUpSR= candidate[13] #scalar increase of recruitment to membrane.
        S0=candidate[14] #initial Solution Munc13 (S), uM
        R0=candidate[15] #initial R, /um^2
        #D1=candidate[9] #Monomer (M) diffusion constant on membrane, um^2/s
        #D2=candidate[9] / candidate[10] #Dimer (D) diffusion constant on membrane, um^2/s
        X0 = candidate[16] #initial X /um^2
        C0 = candidate[17]#initial C /um^2 
        # convert to uM
        R0 = R0*self.cellArea/self.cellVolume/602.0 
        X0 = X0*self.cellArea/self.cellVolume/602.0 
        kfcm = candidate[18]
        krcm = candidate[19]
        eCA = candidate[20]
        kclus = candidate[21]
        eDLoop = candidate[22]
        
        #rateParams=np.array([kfsr*stimUpSR, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd])
        
        rateParams=np.array([kfsr*stimUpSR, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd, kfcm, krcm, eCA, kclus, eDLoop])
        
        #for the post stimulation, increase the on-rate to the recruiter.
        #rateParams=np.array([kfsr*stimUpSR, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd])
        
        #for the post stimulation, use the pre-stim solution as the initial conditions.
        initValues=solution_pre[:,-1]
      
        
        solution = scipy.integrate.solve_ivp(fun=self.munc13_hexamerOde_loopCa, method='BDF', t_span=(0, self.t_max), y0=initValues, t_eval=self.timePoints, args=(rateParams,), rtol=1e-7, atol=1e-9)

        #D = self.calc_diffusion(D1, D2, solution.y)

        return solution.y
    
    


    def simulate(self, candidate):
        """
        Wrapper to run both pre- and post-stimulation solvers.
        """
        solutionPre = self.simulate_pre(candidate)
        solutionPost = self.simulate_post(candidate, solutionPre)
        return [solutionPre, solutionPost]


    

    def test_mass_conservation(self, Y):
        """Test that the total mass of Munc13 is conserved in the ODE solution
            Just look at the final time point.
        """
        #final copy numbers.
        copies = Y[:,-1] * self.cellVolume * 602
        #initial values.
        totalMunc=Y[0][0]* self.cellVolume * 602
        totalR=Y[1][0]* self.cellVolume * 602
        totalX=Y[4][0]* self.cellVolume * 602
        totalC = Y[23][0]* self.cellVolume * 602
        #calculate the total number of Munc13 in the system at each time point
        memMunc=self.calculate_munc13_on_membrane(copies)
        muncFinal=memMunc+copies[0]+copies[6]
        
        #calculate the total number of R in the system at each time point
        RvsTime=copies[1]+memMunc
        #calculate the total number of X in the system at each time point
        XvsTime=copies[4]+copies[5]+copies[6]+copies[7]+copies[8]+2*copies[9]+\
                    2*copies[10]+2*copies[11]+2*copies[12]+\
                    2*copies[13]+2*copies[14]+2*copies[15]+\
                    3*copies[16]+3*copies[17]+3*copies[18]+\
                    3*copies[19]+3*copies[20]+3*copies[21]+\
                    3*copies[22]+3*copies[26]
        
        CvsTime = copies[23]+copies[24]+copies[25]+copies[26]
        
        print(f"Initial total Munc13 {totalMunc}, final total Munc {muncFinal}")
        print(f"Initial total R {totalR}, final total R {RvsTime}")
        print(f"Initial total X {totalX}, final total X {XvsTime}")
        print(f"Initial total C {totalC}, final total C {CvsTime}")


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
        #        ax.plot(self.timePoints, copies[0], linestyle="-", label="Solution copies", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[1], linestyle="-", label="Recruiter copies", color=c_post, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[2], linestyle="-", label="Membrane monomer", color="blue", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[3], linestyle="-", label="Membrane dimer", color="cyan", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[4], linestyle="-", label="Nucleator X", color="green", alpha=0.95, zorder=3)
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
        ax.plot(self.timePoints,copies[20]+copies[21]+copies[22]+copies[18]+copies[19], linestyle="-", label="copies of clusters", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, 6*(copies[20]+copies[21]+copies[22])+5*(copies[18]+copies[19]), linestyle="-", label="munc in larger clusters", color="green", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[5], linestyle="-", label="MX", color=c_post, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[11]+copies[12]+copies[13]+copies[14]+copies[15]+copies[16]+copies[17], linestyle="-", label="copies of small clusters", color="blue", alpha=0.95, zorder=3)
        
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
        #ax.plot(self.timePoints, copies[2], linestyle="-", label="Monomer on membrane", color="black", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[5], linestyle="-", label="MX", color="cyan", alpha=0.95, zorder=3)
       # ax.plot(self.timePoints, copies[4], linestyle="-", label="X", color="blue", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[7], linestyle="-", label="M2X-one", color="pink", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[9], linestyle="-", label="M2X2-two", color="gray", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[11], linestyle="-", label="M3X2", color="gold", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[13], linestyle="-", label="M4X2", color="yellow", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[16], linestyle="-", label="M4X3", color="indigo", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[18], linestyle="-", label="M5X3", color=(0.1, 0.2, 0.8), alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[20], linestyle="-", label="M6X3", color=(0.7, 0.2, 0.8), alpha=0.95, zorder=3)
        
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

    def plot_DX_cluster_time(self, Y, figsize=(5, 4), fontsize=12, dpi=300):
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
        ax.plot(self.timePoints, copies[3], linestyle="-", label="dimer on membrane", color="green", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[8], linestyle="-", label="DX-one", color="purple", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[10], linestyle="-", label="DX2-two", color="lime", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[12], linestyle="-", label="DMX2", color="olive", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[14], linestyle="-", label="DM2X2", color="brown", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[15], linestyle="-", label="D2X2", color="teal", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[17], linestyle="-", label="D2X3", color=(0.1, 0.1, 0.5), alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[19], linestyle="-", label="D2MX3", color=(0.5, 0.2, 0.8), alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[21], linestyle="-", label="D2M2X3", color=(0.5, 0.1, 0.1), alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[22], linestyle="-", label="D3X3", color=(0.7, 0.2, 0.1), alpha=0.95, zorder=3)
        
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

    """MEJ
         Calculate the diffusivity of the full population based on the
         sizes of the species, only on the membrane"""
    def calculate_population_average_diffusion(self, sol):      
       
        copies = sol[:,-1] * self.cellVolume * 602
        #this is the total copies of munc13 on the membrane.
        #this needs to be stoichiometric, because they measure 
        #diffusivity of inidividual munc13, and not separate complexes.
        # So a complex with 6 munc13 is 6x more likely to be measured towards diffusivity. 
        memMunc = (copies[2] + 2 * copies[3]+ copies[5] + 2*copies[7]+2*copies[8]+\
                    2*copies[9]+2*copies[10]+3*copies[11]+3*copies[12]+\
                    4*copies[13]+ 4*copies[14]+ 4*copies[15]+ 4*copies[16]+\
                    4*copies[17]+ 5*copies[18]+ 5*copies[19]+ 6*copies[20]+\
                          6*copies[21]+6*copies[22])
        #this is the total copies of
        
        #Diffusion of complexes depends on DM, DX, DD
        #we are going to use a simple and very conservative model, where the radii
        #and not the mass sums linearly. If its the mass, you need 
        # a model for scaling of the radius, e.g. 1/3 for space filing, or 3/5 for 
        # unstructured. 
        DtM=self.DtM
        DtX=self.DtX
        DtD = self.DtD
        DtMX = 1.0/(1.0/self.DtM + 1.0/self.DtX)
        DtMMX = 1.0/(1.0/DtMX + 1.0/self.DtM)
        DtDX = 1.0/(1.0/self.DtX + 1.0/self.DtD)
        DtM2X2 = 1.0/(1.0/DtMMX + 1.0/self.DtX)
        DtDX2 = 1.0/(1.0/DtDX + 1.0/self.DtX)
        DtM3X2= 1.0/(1.0/DtM2X2+1.0/self.DtM)
        DtDMX2= 1.0/(1.0/DtDX2+1.0/DtM)
        DtM4X2= 1.0/(1.0/DtM3X2 + 1.0/DtM)
        DtDM2X2=1.0/(1.0/DtDMX2 + 1.0/DtM)
        DtD2X2=1.0/(1.0/DtDX2 + 1.0/DtD)
        DtM4X3=1.0/(1.0/DtM4X2 + 1.0/DtX)
        DtD2X3=1.0/(1.0/DtD2X2 + 1.0/DtX)
        DtM5X3=1.0/(1.0/DtM4X3 + 1.0/DtM)
        DtD2MX3=1.0/(1.0/DtD2X3 + 1.0/DtM)
        DtM6X3=1.0/(1.0/DtM5X3 + 1.0/DtM)
        DtD2M2X3=1.0/(1.0/DtD2MX3 + 1.0/DtM)
        DtD3X3=1.0/(1.0/DtD2X3 + 1.0/DtD)

        print(f"DtM, {DtM}. DtX {DtX}. DtD {DtD}")
        print(f"DtM6X3, {DtM6X3}. DtD3X3 {DtD3X3}. DtMX {DtMX}. DtDX {DtDX}")
        print(f"memMunc {memMunc}. M {copies[2]}. D {copies[3]}. ")

        averageD = (copies[2]*self.DtM + 2*copies[3]*self.DtD + copies[5]*DtMX+\
                    2*DtMMX*copies[7]+2*DtDX*copies[8]+\
                    2*DtM2X2*copies[9]+2*DtDX2*copies[10]+3*DtM3X2*copies[11]+3*DtDMX2*copies[12]+\
                    4*DtM4X2*copies[13]+ 4*DtDM2X2*copies[14]+ 4*DtD2X2*copies[15]+ 4*DtM4X3*copies[16]+\
                    4*DtD2X3*copies[17]+ 5*DtM5X3*copies[18]+ 5*DtD2MX3*copies[19]+ 6*DtM6X3*copies[20]+\
                          6*DtD2M2X3*copies[21]+6*DtD3X3*copies[22])/memMunc
        
        return averageD
    


    def calc_percentages_cluster(self, solution):
        """MEJ
            Calcualte the percentage of Munc13 copies in cluster
            out of total Munc13 on membrane
            out of total Munc13 in system
            """
        #just get the value at the end.
        copies=solution[:,-1] * self.cellVolume * 602
        #these all account for stoichiometry of munc13 in the species.
        clusterCopies=6*(copies[20]+copies[21]+copies[22]+copies[26])+5*(copies[18]+copies[19])
        memCopies=self.calculate_munc13_on_membrane(copies)
      
        muncTotal=memCopies+copies[0]+copies[6]
        #totalCopies=np.sum(copies[-1]) #this does not account for stoichiometry
        P_cluster=clusterCopies/muncTotal
        P_cluster_normMem=clusterCopies/memCopies
        P_mem = memCopies/muncTotal
        pMono = (copies[2]+copies[24])/memCopies
        pDimer = 2*(copies[3]+copies[25])/memCopies
        

        return P_cluster, P_cluster_normMem, P_mem, pMono, pDimer

    def calc_populations(self, solution):
        #this is not accurate!! DO NOT USE
        P_cluster = (solution[5] + solution[7]) / (solution[2] + 2 * solution[3] + solution[5] + solution[7])
        P_monomer = (solution[2]) / (solution[2] + 2 * solution[3] + solution[5] + solution[7])
        P_dimer = (2 * solution[3]) / (solution[2] + 2 * solution[3] + solution[5] + solution[7])

        return P_monomer[-1], P_dimer[-1], P_cluster[-1]
    
    def calc_3D_2D_percentages_cluster(self, solution):
        """Calcualte the percentage of Munc13 copies in cluster from 3D or 2D"""
        P_3D_average = (solution[5]) / (solution[5] + solution[7])
        P_2D_average = (solution[7]) / (solution[5] + solution[7])

        return P_3D_average[-1], P_2D_average[-1]

    def test_one_candidate(self, test_candidate):
        """For testing: simulate and plot one candidate solution"""
        lowKeep=[0.02365312119320792, 0.07324823081818993, 0.006781629952218706, 0.026145435035022414, 0.05962051912741756, 910.4432802859255, 1.0201102782119582, 2.1591353776886364, 5.548766168146259, 6.552086744803905, 1.369400021791212, 0.009392526937221047, 0.26663253781704555, 3.819454710837715, 0.4416628055713939, 36.84271293114108, 2.3713879028512737]
        #low1=[0.02365312119320792, 0.07324823081818993, 0.006781629952218706, 0.026145435035022414, 0.05, 910.4432802859255, 1.0201102782119582, 2.1591353776886364, 5.548766168146259, 6.552086744803905, 1.369400021791212, 0.009392526937221047, 0.26663253781704555, 3.819454710837715, 0.4416628055713939, 36.84271293114108, 2.37]
        #low1 = [0.002170279753376766, 0.0030985108546292737, 0.541089118807506, 36.32529528172233, 1.967527357011766, 7.145874551891691, 9.532372977709668, 5.781722248002691, 0.3859629243663857, 3.4783327757469267, 11.701267871430337, 0.6684945723147445, 0.6667433565458644, 12.502253663509027, 0.4171832802444095, 5.602251449211045, 0.6696123435578875]
        #low1 = [0.0030797294137653677, 0.3161086120633267, 0.001, 49.24493095416059, 0.6161974948556822, 34.99198510338638, 0.2778800667029034, 0.12840543417676628, 10, 56.23293899824067, 6.974798271558043, 0.5585814977426274, 0.09863048436762115, 3.484748825675551, 7.4277084334964805, 157.33532277929788, 0.266428887539505]
        #low1 = [0.49335715569731065, 1.347543094580936, 0.001, 47.12925771187148, 0.018594310247031155, 0.01603485903182298, 7.038960289433359, 0.02886043244602297, 6.948907555262286, 4.7041029781798676, 12.54929757390953, 0.554932576424846, 0.05905615952760187, 4.6039748546372445, 0.011467935727766825, 716.4098928992836, 0.1447290585619685]
        #low1=[5.743226759831292, 0.0018309892557324039, 0.03285912981166581, 0.1983236052173084, 0.018049508326647788, 0.04842961665796711, 0.11611942775698278, 0.019124251367490287, 2.049294616105438, 0.22599680930156066, 9.33694669734479, 1, 0.01, 12.86442741404803, 1.4002213127797607, 1.4872831077940427, 27.8125416304795]
        sol = self.simulate_pre(test_candidate)
        print("S vs time: ", sol[0])

        print(self.timePoints[-1])
        self.test_mass_conservation(sol)
        self.plot_freespecies_time(sol, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_mycluster_time(sol, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_each_cluster_time(sol, figsize=(5, 4), fontsize=12, dpi=300)
        self.plot_DX_cluster_time(sol, figsize=(5, 4), fontsize=12, dpi=300)
        print("free nucleators X at the end pre: ",sol[4][-1])
        print("MX at the end pre: ",sol[5][-1])
        # calculate fitness of this solution
        fit = self.fitness_function_to_call(test_candidate)
        print("fitness of this candidate: ", fit)

        #simulate post:
        solPost = self.simulate_post(test_candidate, sol)
        print("S vs time: ", solPost[0])

        print(self.timePoints[-1])
        #self.test_mass_conservation(solPost)
        self.plot_freespecies_time(solPost, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_mycluster_time(solPost, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_each_cluster_time(solPost, figsize=(5, 4), fontsize=12, dpi=300)
        self.plot_DX_cluster_time(sol, figsize=(5, 4), fontsize=12, dpi=300)
        print("free nucleators X at the end post: ",solPost[4][-1])
        print("MX at the end post: ",solPost[5][-1])

    def test_candidate_and_mutant(self, test_candidate):
        """For testing: simulate and plot one candidate solution"""
        #lowKeep=[0.02365312119320792, 0.07324823081818993, 0.006781629952218706, 0.026145435035022414, 0.05962051912741756, 910.4432802859255, 1.0201102782119582, 2.1591353776886364, 5.548766168146259, 6.552086744803905, 1.369400021791212, 0.009392526937221047, 0.26663253781704555, 3.819454710837715, 0.4416628055713939, 36.84271293114108, 2.3713879028512737]
        #low1=[0.02365312119320792, 0.07324823081818993, 0.006781629952218706, 0.026145435035022414, 0.05, 910.4432802859255, 1.0201102782119582, 2.1591353776886364, 5.548766168146259, 6.552086744803905, 1.369400021791212, 0.009392526937221047, 0.26663253781704555, 3.819454710837715, 0.4416628055713939, 36.84271293114108, 2.37]
        #low1 = [0.002170279753376766, 0.0030985108546292737, 0.541089118807506, 36.32529528172233, 1.967527357011766, 7.145874551891691, 9.532372977709668, 5.781722248002691, 0.3859629243663857, 3.4783327757469267, 11.701267871430337, 0.6684945723147445, 0.6667433565458644, 12.502253663509027, 0.4171832802444095, 5.602251449211045, 0.6696123435578875]
        #low1 = [0.0030797294137653677, 0.3161086120633267, 0.001, 49.24493095416059, 0.6161974948556822, 34.99198510338638, 0.2778800667029034, 0.12840543417676628, 10, 56.23293899824067, 6.974798271558043, 0.5585814977426274, 0.09863048436762115, 3.484748825675551, 7.4277084334964805, 157.33532277929788, 0.266428887539505]
        #low1 = [0.49335715569731065, 1.347543094580936, 0.001, 47.12925771187148, 0.018594310247031155, 0.01603485903182298, 7.038960289433359, 0.02886043244602297, 6.948907555262286, 4.7041029781798676, 12.54929757390953, 0.554932576424846, 0.05905615952760187, 4.6039748546372445, 0.011467935727766825, 716.4098928992836, 0.1447290585619685]
        #low1=[5.743226759831292, 0.0018309892557324039, 0.03285912981166581, 0.1983236052173084, 0.018049508326647788, 0.04842961665796711, 0.11611942775698278, 0.019124251367490287, 2.049294616105438, 0.22599680930156066, 9.33694669734479, 1, 0.01, 12.86442741404803, 1.4002213127797607, 1.4872831077940427, 27.8125416304795]
        sol = self.simulate_pre(test_candidate)
        print("S vs time: ", sol[0])

        print(self.timePoints[-1])
        self.test_mass_conservation(sol)
        self.plot_freespecies_time(sol, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_mycluster_time(sol, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_each_cluster_time(sol, figsize=(5, 4), fontsize=12, dpi=300)
        self.plot_DX_cluster_time(sol, figsize=(5, 4), fontsize=12, dpi=300)
        print("free nucleators X at the end pre: ",sol[4][-1])
        print("MX at the end pre: ",sol[5][-1])
        # calculate fitness of this solution
        fit = self.fitness_function_to_call(test_candidate)
        print("fitness of this candidate: ", fit)

        #simulate post:
        candidate_dc2a=list(test_candidate)
        candidate_dc2a[2] = 0 #this sets kfmm to zero.
        candidate_dc2a[12] = 0 #this sets kfdd to zero (no in cluster transition to 2M->D)
        
        mutantC2A_pre = self.simulate_pre(candidate_dc2a)


        self.plot_freespecies_time(mutantC2A_pre, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_mycluster_time(mutantC2A_pre, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_each_cluster_time(mutantC2A_pre, figsize=(4, 3), fontsize=12, dpi=300)
        self.plot_DX_cluster_time(mutantC2A_pre, figsize=(4, 3), fontsize=12, dpi=300) 

        print("free nucleators X at the end post: ",mutantC2A_pre[4][-1])
        print("MX at the end post: ",mutantC2A_pre[5][-1])

    def filter_and_store_solutions(self, best=100, totalTime=30.0, dt=0.1):
        np.random.seed(42)
        self.t_max = totalTime
        n_time_points = int(totalTime / dt)
        self.timePoints = [i * dt for i in range(n_time_points + 1)]
    
        eDF = pd.read_csv("../data/optimizedParms_cluster.txt", sep=",", engine="python")
        eDF.columns = eDF.columns.str.strip()
        eDF = eDF.sort_values(by="Rank")
    
        param_cols = [col for col in eDF.columns if col not in ["Rank", "Fitness"]]
    
        qualified = []
        for _, row in eDF.iterrows():
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
        eDF = pd.DataFrame(data)
        eDF.to_csv(f"../../NERDSS/data/munc13_copies_ode.csv", index=False)
        
        data = {
            "timePoints": self.timePoints,
            "preEq": preEq,
            "postEq": postEq
        }
        eDF = pd.DataFrame(data)
        eDF.to_csv(f"../../NERDSS/data/munc13_copies_eq_ode.csv", index=False)

        eDF_pre = pd.read_csv("../../NERDSS/data/stochastic_pre.csv")
        eDF_post = pd.read_csv("../../NERDSS/data/stochastic_post.csv")

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
        pre_t   = eDF_pre["timePoints"].to_numpy()
        pre_m   = eDF_pre["munc13_cluster_mean"].to_numpy()
        pre_sem = eDF_pre["munc13_cluster_sem"].to_numpy()

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
        post_t   = eDF_post["timePoints"].to_numpy()
        post_m   = eDF_post["munc13_cluster_mean"].to_numpy()
        post_sem = eDF_post["munc13_cluster_sem"].to_numpy()

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
        eDF = pd.read_csv("../data/optimizedParms.txt", sep=",", engine="python")
        eDF.columns = eDF.columns.str.strip()
        eDF = eDF.sort_values(by="Rank")

        solutions = []
        for _, row in eDF.iterrows():
            rank = int(row["Rank"])
            fitness = float(row["Fitness"])
            param_cols_all = [c for c in eDF.columns if c not in ["Rank", "Fitness"]]
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
        param_cols = [c for c in eDF.columns if c not in ["Rank", "Fitness"]]
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
        parameter_ranges=None,
        percent=10,                    # use top `percent`% of solutions by Rank
        figsize=(11, 7),
        fontsize=16,
        dpi=600,
        bar_width=0.7,
        save_path="../fig/params-summary.png",
        inputFile = "../data/testParms_preStimOnly_4028.txt",
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

        

        # --------------------- load solutions -----------------------------------
        eDF = pd.read_csv(inputFile, sep=",", engine="python")
        eDF.columns = eDF.columns.str.strip()

        # Parameter columns = everything except bookkeeping
        param_cols = [c for c in eDF.columns if c not in ("Rank", "Fitness")]
        if not param_cols:
            raise ValueError("No parameter columns found in optimizedParms.txt")

        # Best-by-rank subset size
        n_total = len(eDF)
        n_best = max(1, int(np.floor(n_total * (percent / 100.0))))
        # Best subset by Rank (smaller Rank = better)
        best_eDF = eDF.nsmallest(n_best, "Rank")
        best_row = eDF.nsmallest(1, "Rank").iloc[0]

        print(f"Total solutions loaded: {n_total}")
        print(f"Using top {percent}% by Rank -> {n_best} solutions.")

        # -------------------- labels --------------------------------------------
        parms_name_map = {
            'kfsr': r'$kf_{SR}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krsr': r'$kr_{SR,\mathrm{NOSTIM}}\; (\mathrm{s}^{-1})$',
            
            'kfmm': r'$kf_{MM}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krmm': r'$kr_{MM}\; (\mathrm{s}^{-1})$',
            'kfmx': r'$kf_{MX}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krmx': r'$kr_{MX}\; (\mathrm{s}^{-1})$',
           
            'kfc': r'$kf_{C}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krc': r'$kr_{C}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'kfxx': r'$kf_{XX}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krxx': r'$kr_{XX}\; (\mathrm{s}^{-1})$',
            'kfdd': r'$kf_{DD}\; (\mathrm{s}^{-1})$',
            'eMLoop': r'$exp^{Mloop}\; $',
            'eDF': r'$exp^{DF}\; $',
            'eDLoop': r'$exp^{Dloop}\; $',
            'S0': r'$S_{0}\; (\mu\mathrm{M})$',
            'R0': r'$R_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            'X0': r'$X_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            'C0': r'$C_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            'kfcm': r'$kf_{CM}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krcm': r'$kr_{CM}\; (\mathrm{s}^{-1})$',
            'kfclus': r'$kf_{clus}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'eCA': r'$exp^{CA}\; $',

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
            col = best_eDF[p].to_numpy(float)
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
        y_min = 1e-3
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




"""
END OF MUNC13 CLASS DEFINITIONS.


DEFINE SOLVER CLASS FOR RUNNING THE OPTIMIZATION





"""

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
        self.toolbox.register("evaluate", self.model.fitness_function_to_call)

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
            print("fitness: ",fits[0])
            print("candidate: ",offspring[0])
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
                "Rank, Fitness, kfsr, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd, stimUpSR, S0, R0, X0, C0, kfcm, krcm, eCA, kclus, eDLoop\n"
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
              
                kfmm = opt_params[2]
                krmm = opt_params[3]
                kfmx = opt_params[4]
                krmx = opt_params[5]
                kfc = opt_params[6]
                krc = opt_params[7]
                
                kfxx = opt_params[8]
                krxx = opt_params[9]
                eMLoop = opt_params[10]
                eDF = opt_params[11]
                kfdd = opt_params[12]
                stimUpSR = opt_params[13]
                S0 = opt_params[14]
                R0 = opt_params[15]
                
                X0 = opt_params[16]
                C0 = opt_params[17]
                kfcm = opt_params[18]
                krcm = opt_params[19]
                eCA = opt_params[20]
                kclus = opt_params[21]
                eDLoop = opt_params[22]
                

                # Write a line per solution
                f.write(
                    f"{rank}, {fitness_val:.4f}, "
                    f"{kfsr:.5g}, {krsr:.5g}, {kfmm:.5g}, "
                    f"{krmm:.5g}, {kfmx:.5g}, {krmx:.5g}, "
                    f"{kfc:.5g}, {krc:.5g}, {kfxx:.5g}, "
                    f"{krxx:.5g}, {eMLoop:.5g}, {eDF:.5g}, "
                    f"{kfdd:.5g}, {stimUpSR:.5g}, "
                    f"{S0:.5g}, {R0:.5g}, {X0:.5g},{C0:.5g}, "
                    f"{kfcm:.5g},{krcm:.5g},{eCA:.5g}, "
                    f"{kclus:.5g}, {eDLoop:.5g}\n"
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
        "kfmm":      {"min": 0.01, "max": 10},   # kfMM uM-1s-1
        "krmm":      {"min": 0.01,   "max": 100}, # krMM s-1
        "kfmx":      {"min": 0.001, "max": 10},   # kf1x uM-1s-1
        "krmx":      {"min": 0.01,   "max": 1000}, # kr1x s-1
       # "kfc_nostim":      {"min": 0.001, "max": 10},   # kfc_nostim
        "kfc":      {"min": 0.001, "max": 10},   # kfc_stim uM-1s-1
        "krc":      {"min": 0.01, "max": 1000},   # krc
        "kfxx":      {"min": 0.001, "max": 10},   # kx2 uM-1s-1
        "krxx":      {"min": 0.01,   "max": 1000}, # krx2 s-1
        "eMLoop":      {"min": 0.001,   "max": 10}, # exp(free energy kT units <0).
        "eDF":      {"min": 0.001,   "max": 10}, # exp(free energy kT units <0).
        "kfdd":     {"min": 0.01,   "max": 1}, # kfdd unimolecular: s-1
        "stimUpSR":       {"min": 1,   "max": 100}, # stimUpSR: scale factor >1
        "S0":        {"min": 0.001, "max": 5},   # S0 (uM)
        "R0":        {"min": 0.1, "max": 10000},   # R0 (/um^2)
        #"D1":        {"min": 0.05,   "max": 5}, # D1
        #"D1_over_D2":        {"min": 1.5,   "max": 5}, # D2
        "X0":        {"min": 0.05,   "max": 100}, # X0  (/um^2)
        "C0":       {"min": 0.05,   "max":  10}, # X0  (/um^2)
        "kfcm":     {"min": 1e-26,   "max": 1e-26}, # kfcm: should be slow  (uM-1s-1)
        "krcm":     {"min": 0.1,   "max": 1000}, # krcm  (s-1)
        "eCA":      {"min": 0.001,   "max": 10}, # eCA  (exp(dG/kT)) benefit for dimer binding C.
        "kclus":    {"min": 1e-26,   "max": 1e-26}, # kclus (uM-1s-1)
        "eDLoop":   {"min": 0.001,  "max": 10} #stabilize loop of dimers >1 means destabilize
         
    }

    # Order in which the solver will read parameters from a candidate
    # so, e.g. candidate[0] = kfsr.
    #params_to_optimize = np.array([
    #    "kfsr","krsr","kfmm","krmm","kfmx","krmx","kfc","krc","kfxx","krxx","eMLoop","eDF","kfdd","stimUpSR","S0","R0","X0"
    #])
    params_to_optimizeCa = np.array([
        "kfsr","krsr","kfmm","krmm","kfmx","krmx","kfc","krc","kfxx","krxx","eMLoop","eDF","kfdd","stimUpSR","S0","R0","X0","C0","kfcm", "krcm","eCA","kclus","eDLoop"
    ])
    print("number of parameters to optimize: ", params_to_optimizeCa.size)
    # GA settings
    popSize = 50000
    nGen = 5

    # Instantiate the model and solver, including max time limit.
    maxTime = 1000.0
    model = Munc13(parameter_ranges, params_to_optimizeCa, t_max=maxTime)
    #the solver can be passed a specific filename for the solutions.
    random_number = np.random.randint(1, 10000)
    filename = f"../data/testParms_withCA_Lifetimes_{random_number}.txt"
    print("Output filename: ", filename)
    solver = Solver(model, populationSize=popSize, NGEN=nGen, outfileName=filename)

    # For testing: simulate and plot one candidate solution
    testOne=False
    #Set testOne to false to run the optimizer.
    if(testOne):
        print('Test one candidate solution...')
        test_candidate=[5.743226759831292, 0.0018309892557324039, 0.03285912981166581, 0.1983236052173084, 0.018049508326647788, 0.04842961665796711, 0.11611942775698278, 0.019124251367490287, 2.049294616105438, 0.22599680930156066, 9.33694669734479, 1, 0.01, 12.86442741404803, 1.4002213127797607, 1.4872831077940427, 27.8125416304795]
        model.test_one_candidate(test_candidate)
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

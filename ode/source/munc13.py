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

        self.recruitmentStim = 305219/101990 #increase of munc13 on the membrane upon stimulation
        
        #Same observables as above, but for the delta C2A mutant with impaired dimerization.     
        self.density_c2a_pre = 0.35*10/320
        self.density_c2a_post = 0.58*10/320
        self.recruitmentStimC2A = 256318/86340

        self.DtM = 0.08 #um2/s
        self.DtX = 0.001 #um2/s
        self.DtD = 0.025 #um2/s twice as slow as DM.
        self.DtQ = 0.08 #um2/s

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
        self.threshold = -50


    # -----------------------------
    #     ODE system definitions
    # -----------------------------
    
       

    def munc13_hexamerOde_1X(self, t, y, params):
        """
        Maggie Johnson, Oct 2025
        Munc13 model with explicit clustering. 
        
        This model is simplified in one way, such that only a single X is required.
        The new assumption is that clusters effectively exist without munc13, that is X.
        Then, X is a mimic of a multi-valent cluster.
        Each M that recruits forms a unique contact to X. 
        The M vs D species must have distinct on-rates, and the same 
        off rates, so that the lifetime is the same.
        Further, there are two 'classes' of recruiters. 
        One is R, that binds to X to nucleat.
        The other is Q. It is inhibited from clustering by the C2C domain.
        Thus X can still be in excess, because M bound to Q will not nucleate clusters.


        
        y[0]  = S, Munc13 in solution
        y[1]  = R, General recruiter on the membrane
        y[2]  = M, Munc13 on the membrane (out of the cluster)
        y[3]  = D, Munc13 dimer on the membrane (out of the cluster)
        y[4]  = X, Cluster nucleator, also in 2D but dilute (well-mixed)
        y[5]  = S2, dimer in solution
        y[6]  = MX, Munc13 on the membrane + X 
        y[7]  = SX, Munc13 in solution+ X
        y[8]  = M2X, two muncs + X 
        y[9]  = DX, dimer + X
        y[10]  = Q, sequestering specie
        y[11] = SQ, Munc13 + Q
        y[12] = S2Q2, Dimer + Q
        y[13] = M3X, 3
        y[14] = M4X, 4
        y[15] = M5X, 5
        y[16] = M6X, 6
        y[17] = D2X, 4
        y[18] = D3X, 6
        y[19] = DMX, 3
        y[20] = DM2X, 4
        y[21] = D2MX, 5
        y[22] = D2M2X, 6
        
        
        
                
                Reaction equations:
        1-19r, krsr
            1. S + R <-> M, kfsr, krsr
            2. M + M <-> D, gamma*kfmm, krmm
            3. M + X <-> MX, gamma*kfmx, krmx
            4. S + X <-> SX, kfmx, krmx
            5. SX + R <-> MX, gamma*kfsr, krsr
            6. M + MX <-> M2X, gamma*kfc, krc
            7. D + X <-> DX, gamma*kfmx*S*np.exp(-DF), krmx*S, this way S can negate dGC
            8. M2X<->DX kfDD, krDD dG=dMM+dF-dGC
            9. S+S <->S2 kfmm, krmm
            10. S2 + 2R <-> D 2*gamma*kfsr*1e-6 (units uM-2 s-1), krsr*exp(dGSR). 
            11. S+Q <->SQ kfq, krq
            12. S2+2Q <-> S2Q2 2*gamma*kfq*1e-6, krq*exp(dGQ)
            13. DX+D <->D2X gamma*kfc*S*np.exp(-DF), krc*S
            14. D2X+D <-> D3X gamma*kfc*S*np.exp(-DF), krc*S*np.exp(Loop)
            15. M2X+M <-> M3X gamma*kfc, krc
            16. M3X+M <-> M4X gamma*kfc, krc
            17. M4X+M <-> M5X gamma*kfc, krc
            18. M5X+M <-> M6X gamma*kfc, krc*np.exp(Loop)
            19. DX+ M <->DMX gamma*kfc, krc
            20. DMX+M <-> DM2X gamma*kfc, krc
            21. M2X+D <-> DM2X gamma*kfc*S*exp(-dF), krc*S
            22. DM2X+D <->D2M2X gamma*kfc*S*exp(-dF), krc*S*np.exp(Loop)
            23. DM2X <->D2X kfDD, krDD dG=dMM+dF-dGC
            24. D2M2X <->D3X kfDD, krDD dG=dMM+dF-dGC
            25. D2X+M <->D2MX gamma*kfc, krc
            26. D2MX+M <->D2M2X gamma*kfc, krx*np.exp(Loop)
            27. SQ+SQ<-> S2Q2 gamma*kfmm, krmm
            #explanation for S2+2R (three body)
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
        
        kfq = params[8]
        krq = params[9]
        #sig=params[10] #scale factor to accelerate dimer to cluster
        eLoop = params[10] # monomer loop stability
        #eLoop = np.exp(Loop), for favorable loop formation, eLoop<1
        eDF = params[11] #cooperative free energy for dimer contacts with the cluster.
        kfdd = params[12] #rate of transitioning in the cluster from 2 monomers to dimers.
        Sd = params[13] #scale factor to accelerate dimer binding to clusters
        #kfcm = params[13]
        #krcm = params[14]
        #eCA = params[15]
        #kclus = params[16]
        #eDLoop = params[17] #dimer loop stability.
        # krdc = params[12] constrainted by free energies.
        #dG values are in units of kT
        #dGM = np.log(krmm*1e-6/kfmm) #K_D is in units of M, divide out c0=1M standard state. 
        #dGC = np.log(krc*1e-6/kfc) #K_D is in units of M, divide out c0=1M standard state. 
        
        #we don't need these calculations, but keep in mind that if any of the rates
        #are set to zero, the krdd calculation will become infinity.
        if(kfmm==0):
            krdd = 0 #forward reaction can't happen, so neither can this.
        else:
            krdd = kfdd*eDF*kfc/krc*krmm/kfmm #same as above, factors of 1e-6 cancel
            #dGM = np.log(krmm*1e-6/kfmm) #K_D in units of M
        
        if(kfq==0):
            krq2 = 0 #forward reaction can't happen, so neither can this.
        else:
            krq2=krq*krq*1e-6/kfq #same as above.
            #dGM = np.log(krmm*1e-6/kfmm) #K_D in units of M
        
        
        #dGSR = np.log(krsr*1e-6/kfsr) #K_D in units of M
        #dGSR changes under stimulation.
        #dGM<0, dGC<0, both stabilizing. eDF=exp(dF) < 1 (for stabilizing) 
        #krdd = kfdd*np.exp(dGM-dGC)*eDF  #the ratio of on/off for the 2M<->D transition is controlled by other free energies
        
        
        #these operations below will fail if the on-rates are zero.
        #in that case, the off-rates are infinite. 
        #this should not be a problem, these unbinding reactions should not happen if forward reactions are zero.
        
        #krsr2=krsr*np.exp(dGSR) #unbinding from 2 Rs.
        krsr2=krsr*krsr*1e-6/kfsr #same as above expression, krsr*exp(dGSR)
        #chgDG=eMLoop/eDF #replace the eDF stabilization with the eMLoop stabilization
        #krq2=krq*np.exp(dGQ)
        krq2=krq*krq*1e-6/kfq #same as above.
        #krxTrimer = params[12]
        #enhanceRate = 10 #scalare that increase the on-rate when more than one X is present
        #krmxDimer=krmx*sig*np.exp(dGC)*eDF # off rate for MX slowed by C and F=dG
        #kfast=max(kfc, kfmx)  #two interfaces, choose faster rate
        #kr_cmx = kfast*np.exp(dGMX+dGC) #reverse reaction with MX and C bonds=dG

        #only input stoichiometric factors for the dy expressions, not here.
        r1 = kfsr*y[0]*y[1] #in 3D
        r1b = krsr*y[2]
        r2 = gamma*kfmm*y[2]*y[2]
        r2b = krmm*y[3]
        r3 = gamma*kfmx*y[2]*y[4]
        r3b = krmx*y[6]
        r4 = kfmx*y[0]*y[4]  #in 3D
        r4b = krmx*y[7]
        r5 = gamma*kfsr*y[7]*y[1]
        r5b = krsr*y[6]
        r6 = gamma*kfc*y[2]*y[6]
        r6b = krc*y[8]
        r7 = gamma*kfmx*y[3]*y[4]
        r7b = krmx*Sd*y[9]
        r8 = kfdd*y[8]
        r8b = krdd*y[9]
        r9 = y[0]*y[0]*kfmm
        r9b = krmm*y[5]
        r10 = y[5]*y[1]*y[1]*2*gamma*kfsr*1e-6  
        r10b = y[3]*krsr2
        r11 = y[0]*y[10]*kfq
        r11b = y[11]*krq
        r12 = y[5]*y[10]*y[10]*2*gamma*kfq*1e-6
        r12b = y[12]*krq2
        r13 = gamma*kfc*Sd*y[3]*y[9]/eDF
        r13b = y[17]*krc*Sd
        r14 = y[17]*y[3]*gamma*kfc*Sd/eDF
        r14b = y[18]*krc*Sd*eLoop
        r15 = gamma*kfc*y[2]*y[8]
        r15b = krc*y[13]
        r16 = gamma*kfc*y[2]*y[13]
        r16b = krc*y[14]
        r17 = gamma*kfc*y[2]*y[14]
        r17b = krc*y[15]
        r18 = gamma*kfc*y[2]*y[15]
        r18b = krc*y[16]*eLoop

        r19 = gamma*kfc*y[2]*y[9]
        r19b = krc*y[19]
        r20 = gamma*kfc*y[2]*y[19]
        r20b = krc*y[20]
        r21 = gamma*kfc*Sd*y[3]*y[8]/eDF
        r21b = krc*Sd*y[20]
        r22 = gamma*kfc*Sd*y[3]*y[20]/eDF
        r22b = krc*Sd*eLoop*y[22]
        r23 = kfdd*y[20]
        r23b = krdd*y[17]
        r24 = kfdd*y[22]
        r24b = krdd*y[18]
        r25 = gamma*kfc*y[2]*y[17]
        r25b = krc*y[21]
        r26 = gamma*kfc*y[2]*y[21]
        r26b = krc*eLoop*y[22]
        r27 = kfmm*y[11]*y[11]
        r27b = krmm*y[12]
        

        dylist = []
        #add each new species reaction rate.
        dylist.append(-r1+r1b-r4+r4b-2*r9+2*r9b-r11+r11b)#0 S
        dylist.append(-r1+r1b-r5+r5b-2*r10+2*r10b)#1 R
        dylist.append(r1-r1b-2*r2+2*r2b-r3+r3b-r6+r6b-r15+r15b-r16+r16b-r17+r17b-r18+r18b-r19+r19b-r20+r20b-r25+r25b-r26+r26b)#2 M
        dylist.append(+r2-r2b-r7+r7b+r10-r10b-r13+r13b-r14+r14b-r21+r21b-r22+r22b)#3 D
        dylist.append(-r3+r3b-r4+r4b-r7+r7b)#4 X
        dylist.append(+r9-r9b-r10+r10b-r12+r12b)#5 S2
        dylist.append(r3-r3b+r5-r5b-r6+r6b)#6 MX
        dylist.append(r4-r4b-r5+r5b)#7 SX
        dylist.append(r6-r6b-r8+r8b-r15+r15b-r21+r21b)#8 M2X
        dylist.append(r7-r7b+r8-r8b-r13+r13b-r19+r19b)#9 DX
        dylist.append(-r11+r11b-2*r12+2*r12b) #10 Q
        dylist.append(r11-r11b-2*r27+2*r27b) #11 SQ
        dylist.append(r12-r12b+r27-r27b) #12 S2Q
        dylist.append(r15-r15b-r16+r16b)#13 M3X
        dylist.append(r16-r16b-r17+r17b) #14 M4X
        dylist.append(r17-r17b-r18+r18b) #15 M5X
        dylist.append(r18-r18b)#16 M6X
        dylist.append(r13-r13b-r14+r14b+r23-r23b-r25+r25b) #17 D2X
        dylist.append(r14-r14b+r24-r24b) #18 D3X
        dylist.append(r19-r19b-r20+r20b) #19 DMX
        dylist.append(r20-r20b+r21-r21b-r22+r22b-r23+r23b) #20 DM2X
        dylist.append(r25-r25b-r26+r26b) #21 D2MX
        dylist.append(r22-r22b-r24+r24b+r26-r26b) #22 D2M2X
       

        return np.array(dylist)
    "END OF HEXAMER ODE loopCa"

   
    # ------------------------------------------------------
    #        Utilities to compute Munc13 on cluster/membrane
    # ------------------------------------------------------
    def calculate_munc13_on_membrane(self, copies):
        """
        Calculate total copies of munc13 on the membrane.
        For the Single X Model!
        """
      
        memMunc = (copies[2] + 2 * copies[3]+ copies[6]+\
                    copies[7] + 2*copies[8] + 2*copies[9] +\
                    copies[11]+ 2*copies[12] +\
                    3*copies[13] + 4*copies[14] + 5*copies[15] + 6*copies[16]+\
                    4*copies[17] + 6*copies[18] +\
                    3*copies[19] + 4*copies[20] +\
                    5*copies[21] + 6*copies[22]
                )
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
        For the Single X Model!
        """
        #get the final concentrations of each cluster species
      
        #off rates of X from each species
        
        
        #kfmx=candidate[4] #form MX
        
        kfc=candidate[6] #bind to cluster with M and X present
        Sd = candidate[13]
        eDF = candidate[11] 
        gamma = self.gamma
        #numerator is sum over all bound states with correct stoichiometry
        #D3X3, D2M2X3, M6X3, M5X3, D2MX3, CHEX.
        y = sol[:,-1]
        #include both 5-mer and 6-mers in the bound state
        numerator46 = 6*(y[22] + y[18] + y[16]) + 5*(y[21] + y[15])
        #include only 6-mers in the bound state 
        numerator56 = 6*(y[22] + y[18] + y[16]) #+ y[19] + y[18] 
        #denominator is reactive flux of reactants multiplied by on rates. 
        #only reactions that go to the bound state, with reactants NOT in the bound state.
        # 4->6: r22, r14
        # 5->6: r26, r18
        #4->5: r25, r17
        #cannot go from 5-mer to 6-mer in this calculation, as 5 is in the bound state.
        denominator46 = gamma*kfc*Sd*y[3]*y[20]/eDF+y[17]*y[3]*gamma*kfc*Sd/eDF+\
                        gamma*kfc*y[2]*y[17]+gamma*kfc*y[2]*y[14]
        #transitions must go to a 6-mer: r22, r14, 
        denominator56 = gamma*kfc*Sd*y[3]*y[20]/eDF+y[17]*y[3]*gamma*kfc*Sd/eDF+\
                        gamma*kfc*y[2]*y[21]+gamma*kfc*y[2]*y[15]

        tau46 = numerator46/(6*denominator46) if denominator46>0 else 0
        tau56 = numerator56/(6*denominator56) if denominator56>0 else 0    
        return tau46, tau56
  
    
    
    
     # --------------------------------------------
    #          Cost (Chi) and viability checks
    # --------------------------------------------
    def costChi_cluster(self, Y, densExpt, kdimer, weights):
        """
        Compute a chi-square-like cost for the simulated data.
        Based on the cluster forming ODE model.
        Y is the solution in vector form, so all species are in units of uM. 
        densExpt is the expected value of cluster density from the 
        experimental condition
        kdimer is the rate of dimer formation, kfmm. 
        """
        #Calculate the density of clusters on the membrane
        #all of these are time-dependent arrays, we want only the final steady-state values.
        density=self.calculate_cluster_density(Y)
        #we want to add in a term that penalizes a high numbers
        #of small clusters on the membrane.
        #these are clusters that have nucleated with X, but have less than 6 copies
        smallClusterDens, smallS=self.calculate_small_cluster_density(Y)
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
        
        weightSmallClust=weights['wSmallClust'] #this is the inverse of the target maximums
        
        #if we want small clusters to be <1e-4, set the weight to 1e4
        print(f"Small cluster density chi: Simulated {smallClusterDens[-1]}, high density {density[-1]}, Chi {chiSmallClust}")
        #calculate the percent of munc13 on the membrane that is in clusters.
        percClusterTotal, percClustMem, percMem, pMono, pDimer, pMX, pDX=self.calc_percentages_cluster(Y)
        print(f"Percent of mem munc13 in clusters: {percClustMem*100}%. Percent munc13 on the membrane: {percMem*100}%. Monomer mem: {pMono*100}%. Dimer mem: {pDimer*100}%")
        print(f"Percent of memMunc in MX {pMX*100}%, Percent in DX {pDX*100}%")
        #implement a relu penalty if percent in clusters is greater than 40%
        weightPerc = weights['wPercClust']
        upperLimit = 0.4 #penalize fraction of munc13 in clusters above this.
        lowerLimit = 0.05 #penalize fraction below this
        chiPercClust=-max(0, percClustMem-0.4)
        #implement a relu penalty if percent in clusters is less than 5%
        chiPercClust =chiPercClust -max(0, 0.05 - percClustMem)
        #implement a relu penalty if percent in dimers is less than 10%
        lowerDimer = 0.2
        chiDimer = -max(0, lowerDimer-pDimer)
        if(kdimer==0):
            weightDimer= 0 # do not penalize systems that cannot form dimers!
        else:
            weightDimer=weights['wDimer']

        
        weightChiSS = weights['wChiSS']
        print(f"Change in density over last 25%: {ssDelta}, associated chi: {chiSS*weightChiSS}")

        costSum=chiDens[0]+weightSmallClust*chiSmallClust +weightPerc*chiPercClust+\
            chiSS*weightChiSS+chiDimer*weightDimer # this is already negative
        
        return [costSum, density[-1]]


    def calculate_cluster_density(self,Y):
        """
        given the solution of the cluster model, compute the density of clusters on the membrane
        For model with 1 X!
        """
        copies = Y *self.cellVolume * 602 #converts from uM to copy numbers.

        #and at least 6 copies of munc13 (either S or M)
        #6-mers: 22, 18, 16. 

        cluster_copies = copies[22]+copies[18]+copies[16]#+copies[21]+copies[15]
               #now convert to density
        memDensity = cluster_copies/(self.cellArea) #
        return memDensity
    

    def pie_charts(self, sol, solPost, figsize, fileStr):
        # plot two pie charts side by side
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        fontsize= 10
        dpi=300
        labels = ['M', 'D', 'C']#,'SC','MX']
        #reads in the full array
        percClusterTotal, percClustMem, percMem, pMono, pDimer, pMX, pDX=self.calc_percentages_cluster(sol)
        smallClusterDens, smallS=self.calculate_small_cluster_density(sol)
        memCopies=self.calculate_munc13_on_membrane(sol[:,-1]*self.cellVolume*602.0)
        percX=pMX+pDX
        sizes_pre = [pMono, pDimer, percClustMem]#, smallS[-1]*self.cellArea/memCopies, percX]
        print('sizes pre stim, and sum: ', sizes_pre)
        #Now perform the same calculations for the post-stim
        percClusterTotal, percClustMem, percMem, pMono, pDimer, pMX, pDX=self.calc_percentages_cluster(solPost)
        smallClusterDens, smallS=self.calculate_small_cluster_density(solPost)
        memCopies=self.calculate_munc13_on_membrane(solPost[:,-1]*self.cellVolume*602.0)
        percX=pMX+pDX
        sizes_post = [pMono, pDimer, percClustMem]#, smallS[-1]*self.cellArea/memCopies, percX]
        print('sizes POST stim, and sum: ', sizes_post)


        
        colors = ['#ff9999',"#2270bd","#0d0e0d"]#, "#31f7f7", "#f7ed30"]
        #explode = (0.05, 0.05, 0.05,0.05, 0.05)  # explode all slices slightly
        wedges1, texts1, autotexts1 = ax[0].pie(sizes_pre,  labels=labels, colors=colors,
                 startangle=140,autopct='%1.1f%%', textprops={'fontsize': fontsize * 0.8})
        wedges2, texts2, autotexts2 = ax[1].pie(sizes_post,  labels=labels, colors=colors,
               startangle=140, autopct='%1.1f%%', textprops={'fontsize': fontsize * 0.8})
        # draw circle for donut shape
        #centre_circle0 = plt.Circle((0,0), 0.70, fc='white')
        #centre_circle1 = plt.Circle((0,0), 0.70, fc='white')
        #ax[0].add_artist(centre_circle0)
        #ax[1].add_artist(centre_circle1)
        ax[0].set_title('NO STIM', fontsize=fontsize)
        ax[1].set_title('STIM', fontsize=fontsize)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax[0].axis('equal')  
        ax[1].axis('equal')
        fig.tight_layout()
        #fig.savefig("../fig/fig_population_wt.svg") 
        fig.savefig(f"/Users/margaret/Dropbox/r2025/Munc13/IMAGES/pieChart_{fileStr}.png", dpi=dpi)
        plt.show()

    def calculate_small_cluster_density(self,Y):
        """
        given the solution of the cluster model, compute the density of clusters on the membrane
        for 1 X model!
        """
        copies = Y *self.cellVolume * 602 #converts from uM to copy numbers.
        #This is now the inverse of the cluster density,
        #species with between 3 and 5. 
        #5-mer: 21 and 15, 4-mer: 20, 17, 14, 3-mer: 13 and 19
        cluster_copies = copies[21]+copies[15]+copies[20]+copies[19]+\
                            copies[17]+copies[14]+copies[13]

        cluster_stoich_copies = 5*copies[21]+5*copies[15]+4*copies[20]+4*copies[19]+\
                            4*copies[17]+3*copies[14]+3*copies[13]            
               #now convert to density
        memDensity = cluster_copies/(self.cellArea) #
        return memDensity, cluster_stoich_copies/self.cellArea


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
        fitness = self.eval_clusterModel_withD(candidate)    
        return fitness

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
        wts=['wSmallClust','wPercClust','wDimer', 'wChiSS']
        weightVector =pd.DataFrame([[10.0, 50.0, 10.0, 10.0]], columns=wts)
        chiPre, endDensityPre = self.costChi_cluster(solutionPre, densExpt, candidate[2], weightVector)
        
         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPre, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
        upperBound = 20
        lowerBound = 5
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPre=chiPre+chiLifetime

        # Chi from post
        print("**CHI POST STIMULATION")
        chiPost, endDensityPost = self.costChi_cluster(solutionPost, self.density_exp_post, candidate[2], weightVector)
        
         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPost, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
    
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)

        chiPost=chiPost+chiLifetime
        #chi from total change on the membrane
        #print(f" before chi recruitment stim, M on membrane: , {solutionPre[2][-1]}, and post {solutionPost[2][-1]}")
        chiRecruitStim= self.costChi_recruitmentStim(solutionPre,solutionPost, self.recruitmentStim)
        
       
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
        chiPre,endDensityMutantPre = self.costChi_cluster(mutantC2A_pre, self.density_c2a_pre, candidate_dc2a[2], weightVector)
         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(mutantC2A_pre, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
    
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPre=chiPre+chiLifetime
       
        # Chi from post
        print("**MUTANT CHI POSTSTIMULATION")
        chiPost,endDensityMutantPost = self.costChi_cluster(mutantC2A_post, self.density_c2a_post, candidate_dc2a[2], weightVector)

        tau46, tau56 = self.calculate_lifetime_of_clusters(mutantC2A_post, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
    
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPost=chiPost+chiLifetime
        #chi from total change on the membrane
        chiRecruitStim= self.costChi_recruitmentStim(mutantC2A_pre,mutantC2A_post, self.recruitmentStimC2A)
        
        #add a term to select for weaker response under dimer mutation
        #add a term to select for weaker response under dimer mutation
        
        #add a term to ensure that the mutant has a lower density of clusters than the dimer.
        #say it should be at least 1.5 times lower. 
        factorLower = 1.5
        chiMutant = -1*max(0, factorLower*endDensityMutantPre-endDensityPre)
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

        return chiTotal
    
    def eval_clusterModel_withD(self, candidate):
        """
        Simulate the model in pre and post conditions
        Evaluate the candidate solution to compute fitness
        Evaluate the fitness using the chi terms

        Include the evaluation of the different mutants.
        Include a fitness term to constrain D vs Dpost WT.
        """
        solutionPre, solutionPost = self.simulate(candidate)

        # Chi from pre
        densExpt=self.density_exp_pre
        #print("CHI PRE STIMULATION")
        wts=['wSmallClust','wPercClust','wDimer', 'wChiSS']
        weightVector =pd.DataFrame([[10.0, 50.0, 0.0, 10.0]], columns=wts)
        
        chiPre, endDensityPre = self.costChi_cluster(solutionPre, densExpt, candidate[2], weightVector)
        
         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPre, candidate)
        #print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
        upperBound = 20
        lowerBound = 5
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPre=chiPre+chiLifetime

        # Chi from post
        #print("**CHI POST STIMULATION")
        chiPost, endDensityPost = self.costChi_cluster(solutionPost, self.density_exp_post, candidate[2], weightVector)
        
         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPost, candidate)
        #print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
    
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)

        chiPost=chiPost+chiLifetime
        #chi from total change on the membrane
        #print(f" before chi recruitment stim, M on membrane: , {solutionPre[2][-1]}, and post {solutionPost[2][-1]}")
        chiRecruitStim= self.costChi_recruitmentStim(solutionPre,solutionPost, self.recruitmentStim)
        
        #compute the diffusion per and post stim
        avgD = self.calculate_population_average_diffusion(solutionPre)
        avgDpost = self.calculate_population_average_diffusion(solutionPost)
        simRatio=avgD/avgDpost
        DWTratio=1.12 #the actual WT ratio is D/Dpost = 1.17
        #use the Relu term to penalize speed-ups, not larger slowdowns.
        chiDratio = -1*max(0,DWTratio-simRatio)
        chiTotal=np.array(1)
        #assign a weight to the Chi for diffusion slow-down upon stimulation
        weightD = 10.0

        chiTotal = chiPre + chiPost+ chiRecruitStim[0] + chiDratio*weightD

        
        

        # Now evaluate the mutants, which will require new simulations
        #first is the mutant C2A, which eliminates dimerization.
        candidate_dc2a=list(candidate)
        candidate_dc2a[2] = 0 #this sets kfmm to zero.
        candidate_dc2a[12] = 0 #this sets kfdd to zero (no in cluster transition to 2M->D)
        
        mutantC2A_pre, mutantC2A_post = self.simulate(candidate_dc2a)
        # Chi from pre
        #print("MUTANT CHI PRE STIMULATION")
        chiPre,endDensityMutantPre = self.costChi_cluster(mutantC2A_pre, self.density_c2a_pre, candidate_dc2a[2],weightVector)
         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(mutantC2A_pre, candidate)
        #print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
    
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPre=chiPre+chiLifetime
       
        # Chi from post
        #print("**MUTANT CHI POSTSTIMULATION")
        chiPost,endDensityMutantPost = self.costChi_cluster(mutantC2A_post, self.density_c2a_post, candidate_dc2a[2],weightVector)

        tau46, tau56 = self.calculate_lifetime_of_clusters(mutantC2A_post, candidate)
        #print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
    
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPost=chiPost+chiLifetime
        #chi from total change on the membrane
        chiRecruitStim= self.costChi_recruitmentStim(mutantC2A_pre,mutantC2A_post, self.recruitmentStimC2A)
        
        #add a term to select for weaker response under dimer mutation
        #add a term to select for weaker response under dimer mutation
        
        #add a term to ensure that the mutant has a lower density of clusters than the dimer.
        #say it should be at least 1.5 times lower. 
        factorLower = 1.5
        chiMutant = -1*max(0, factorLower*endDensityMutantPre-endDensityPre)
        weightMutant = 10 

        mutD = self.calculate_population_average_diffusion(mutantC2A_pre)
        mutDpost = self.calculate_population_average_diffusion(mutantC2A_post)
        mutRatio=mutD/avgD #ratio of mutant to WT
        Dratio=1.5 #the actual mut/WT ratio is D/DWt = 1.8
        #use the Relu term to penalize speed-ups, not larger slowdowns.
        chiDratio = -1*max(0,Dratio-mutRatio)
        
        
        chiTotal = chiTotal+ chiPre + chiPost+ chiMutant*weightMutant + chiRecruitStim[0] + chiDratio*weightD

        
    
        if(chiTotal > 0):
            print(f"Positive chi found! {chiTotal}")
            print(f"Parameters: {candidate}")
            print(f"Chi pre: {chiPre}, Chi post: {chiPost}, Chi recruit: {chiRecruitStim}")
            print("ERROR")
            print("-------ERROR------")
            return [self.threshold*10]
        
        print(f"Total chi: {chiTotal}")

        return chiTotal

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
        wts=['wSmallClust','wPercClust','wDimer', 'wChiSS']
        weightVector =pd.DataFrame([[10.0, 50.0, 10.0, 10.0]], columns=wts)
        
        chiPre, endDensityPre = self.costChi_cluster(solutionPre, densExpt, candidate[2], weightVector)

         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPre, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
        upperBound = 20
        lowerBound = 5
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPre=chiPre+chiLifetime
        # Chi from post
        print("**CHI POST STIMULATION")
        chiPost, endDensityPost = self.costChi_cluster(solutionPost, self.density_exp_post, candidate[2], weightVector)

         #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPost, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
    
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)
        chiPost=chiPost+chiLifetime
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
        wts=['wSmallClust','wPercClust','wDimer', 'wChiSS']
        weightVector =pd.DataFrame([[10.0, 50.0, 10.0, 10.0]], columns=wts)
        
        chiPre, endDensityPre = self.costChi_cluster(solutionPre, densExpt, candidate[2], weightVector)
        
        #add a term that measures the lifetime of the clusters
        tau46, tau56 = self.calculate_lifetime_of_clusters(solutionPre, candidate)
        print(f"Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
        upperBound = 20
        lowerBound = 5
        chiLifetime = -1*max(0,lowerBound - tau56) -1*max(0,tau56 - upperBound)


        #chiTotal = chiPre[0] 
        chiTotal = chiPre+chiLifetime

        # Now evaluate the mutants, which will require new simulations
        #first is the mutant C2A, which eliminates dimerization.
        candidate_dc2a=list(candidate)
        candidate_dc2a[2] = 0 #this sets kfmm to zero.
        candidate_dc2a[12] = 0 #this sets kfdd to zero (no in cluster transition to 2M->D)
        
        mutantC2A_pre = self.simulate_pre(candidate_dc2a)
        # Chi from pre
        chiPre, endDensityMutant = self.costChi_cluster(mutantC2A_pre, self.density_c2a_pre, candidate_dc2a[2], weightVector)
       
        #add a term for cluster lifetime.
        tau46, tau56 = self.calculate_lifetime_of_clusters(mutantC2A_pre, candidate_dc2a)
        print(f"MUTANT Cluster lifetimes: tau46 {tau46}, tau56 {tau56}")
        #we want tau56 to be at least 10 seconds and less than 50
        chiLifetime = -1*max(0,lowerBound - tau46) -1*max(0,tau46 - upperBound)
        
        #add a term to ensure that the mutant has a lower density of clusters than the dimer.
        #say it should be at least 1.5 times lower. 
        chiMutant = -1*max(0, endDensityMutant-endDensityPre*1.5)
        weightMutant = 10 

        chiTotal = chiTotal+ chiPre +chiLifetime+chiMutant*weightMutant

        
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
        numVar = 23 #number of variables we are tracking. 
        kfsr=candidate[0] #form SR
        krsr=candidate[1] #dissociate SR
        kfmm=candidate[2] #form D
        krmm=candidate[3] #dissociate D
        kfmx=candidate[4] #form MX
        krmx=candidate[5] #dissociate MX
        kfc=candidate[6] #bind to cluster with M and X present
        krc=candidate[7] #reverse of above
        
        kfq=candidate[8] #add second X to trimer
        krq=candidate[9] #off rate of second X from trimer
        eLoop=candidate[10] #speed up for dimers
        eDF = candidate[11] #exponential of free energy benefit eDF<=1 for dimer in cluster, slows off rate.
        kfdd = candidate[12] #rate of converting from 2M to D in the cluster.
        stimUpSR= candidate[14] #scalar increase of recruitment to membrane.
        S0=candidate[15] #initial Solution Munc13 (S), uM
        R0=candidate[16] #initial R, /um^2
        #D1=candidate[9] #Monomer (M) diffusion constant on membrane, um^2/s
        #D2=candidate[9] / candidate[10] #Dimer (D) diffusion constant on membrane, um^2/s
        X0 = candidate[17] #initial X /um^2
        Q0 = candidate[18]
        # convert to uM
        R0 = R0*self.cellArea/self.cellVolume/602.0 
        X0 = X0*self.cellArea/self.cellVolume/602.0 
        Q0 = Q0*self.cellArea/self.cellVolume/602.0
        Sd = candidate[13]
        
        #rateParams=np.array([kfsr, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd])
        rateParams=np.array([kfsr, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfq, krq, eLoop, eDF, kfdd, Sd])
        
        

        initValues=np.zeros(numVar)
        initValues[0]=S0
        initValues[1]=R0
        initValues[4]=X0
        initValues[10]=Q0
        #print("initial values of X0", initValues[4])
        #print("rate parameters array:", rateParams)
        solution = scipy.integrate.solve_ivp(fun=self.munc13_hexamerOde_1X, method='BDF', t_span=(0, self.t_max), y0=initValues, t_eval=self.timePoints, args=(rateParams,), rtol=1e-7, atol=1e-9)

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
        #numVar = 23 #number of variables we are tracking. 
        kfsr=candidate[0] #form SR
        krsr=candidate[1] #dissociate SR
        kfmm=candidate[2] #form D
        krmm=candidate[3] #dissociate D
        kfmx=candidate[4] #form MX
        krmx=candidate[5] #dissociate MX
        kfc=candidate[6] #bind to cluster with M and X present
        krc=candidate[7] #reverse of above
        
        kfq=candidate[8] #add second X to trimer
        krq=candidate[9] #off rate of second X from trimer
        eLoop=candidate[10] #speed up for dimers
        eDF = candidate[11] #free energy benefit eDF<0 for dimer in cluster.
        kfdd = candidate[12] #rate of converting from 2M to D in the cluster.
        stimUpSR= candidate[14] #scalar increase of recruitment to membrane.
        S0=candidate[15] #initial Solution Munc13 (S), uM
        R0=candidate[16] #initial R, /um^2
        #D1=candidate[9] #Monomer (M) diffusion constant on membrane, um^2/s
        #D2=candidate[9] / candidate[10] #Dimer (D) diffusion constant on membrane, um^2/s
        X0 = candidate[17] #initial X /um^2
        Q0 = candidate[18]#initial C /um^2 
        # convert to uM
        R0 = R0*self.cellArea/self.cellVolume/602.0 
        X0 = X0*self.cellArea/self.cellVolume/602.0 
        Q0 = Q0*self.cellArea/self.cellVolume/602.0
        Sd = candidate[13]
        #rateParams=np.array([kfsr*stimUpSR, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd])
        
        rateParams=np.array([kfsr*stimUpSR, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfq, krq, eLoop, eDF, kfdd, Sd])
        
        #for the post stimulation, increase the on-rate to the recruiter.
        #rateParams=np.array([kfsr*stimUpSR, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfxx, krxx, eMLoop, eDF, kfdd])
        
        #for the post stimulation, use the pre-stim solution as the initial conditions.
        initValues=solution_pre[:,-1]
      
        
        solution = scipy.integrate.solve_ivp(fun=self.munc13_hexamerOde_1X, method='BDF', t_span=(0, self.t_max), y0=initValues, t_eval=self.timePoints, args=(rateParams,), rtol=1e-7, atol=1e-9)

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
        totalQ = Y[10][0]* self.cellVolume * 602
        #calculate the total number of Munc13 in the system at each time point
        memMunc=self.calculate_munc13_on_membrane(copies)
        muncFinal=memMunc+copies[0]+2*copies[5]
        
        #calculate the total number of R in the system at each time point
        RvsTime=copies[1]+memMunc-copies[11]-2*copies[12]-copies[7]
        #calculate the total number of X in the system at each time point
        XvsTime=copies[4]+copies[6]+copies[7]+copies[8]+copies[9]+\
                  copies[13]+copies[14]+copies[15]+copies[16]+copies[17]+\
                  copies[18]+copies[19]+copies[20]+copies[21]+copies[22]
        
        QvsTime = copies[10]+copies[11]+2*copies[12]
        
        print(f"Initial total Munc13 {totalMunc}, final total Munc {muncFinal}")
        print(f"Initial total R {totalR}, final total R {RvsTime}")
        print(f"Initial total X {totalX}, final total X {XvsTime}")
        print(f"Initial total Q {totalQ}, final total C {QvsTime}")


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

    def plot_diffusion_vs_exp(self, D_metrics, whichRow):
        #make a plot that shows actual and predicted diffusion constants for
        fig, ax = plt.subplots(figsize=(3, 2))
        D=D_metrics['D'].iloc[whichRow]
        Dpost=D_metrics['Dpost'].iloc[whichRow]
        DC2A=D_metrics['C2AD'].iloc[whichRow]
        DC2Apost=D_metrics['C2ADpost'].iloc[whichRow]
        ax.plot(self.D_exp_pre,D, markersize=4, marker='o', color='black', alpha=0.95, zorder=3)
        ax.plot(self.D_exp_post,Dpost, markersize=4, marker='o', color='red', alpha=0.95, zorder=3)
        ax.plot(self.D_exp_DC2A_pre,DC2A, markersize=6, marker='+', color='black', alpha=0.95, zorder=3)
        ax.plot(self.D_exp_DC2A_post,DC2Apost, markersize=6, marker='+', color='red', alpha=0.95, zorder=3)
        ax.plot(np.arange(0, 10)*0.1,np.arange(0, 10)*0.1, linewidth=1, ls='-', color='black', alpha=0.95, zorder=3)

        ax.set_xlim(left=0, right=0.1)
        ax.set_ylim(bottom=0, top=0.1)
        ax.set_xlabel("Actual D ($\\mu$m$^2/s$)", fontsize=9)
        ax.set_ylabel("Predicted D ($\\mu$m$^2/s$)", fontsize=9)
        ax.tick_params(axis='both',labelcolor='black', labelsize=9)
        #ax.set_xticks([0, 1000,2000])
        # Get rid of bound box on the top.
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"/Users/margaret/Dropbox/r2025/Munc13/IMAGES/Diffusion_actual_vs_predicted_{whichRow}.png",dpi=300)
        plt.show()

    def plot_time_resolved_density(self, sol, solPost, fileStr):
        #create quality small figure of time-dependence of clusters
        fig, ax = plt.subplots(figsize=(2.5, 2))  # 2 inches wide, 2 inches tall

        copies = sol * self.cellVolume * 602
        clusters = copies[22] + copies[18] + copies[16]
        small_clusters = copies[21] + copies[20] + copies[19] + copies[17] + copies[15] + copies[14] + copies[13]
        copiesP = solPost * self.cellVolume * 602
        clustersP = copiesP[22] + copiesP[18] + copiesP[16]
        small_clustersP = copiesP[21] + copiesP[20] + copiesP[19] + copiesP[17] + copiesP[15] + copiesP[14] + copiesP[13]

        # ensure timePoints is a numpy array so adding a scalar works reliably
        tp = np.asarray(self.timePoints)
        fullTime = np.concatenate((tp, tp + tp[-1]))

        # concatenate pre/post series
        combine = np.concatenate((clusters, clustersP))
        small_combine = np.concatenate((small_clusters, small_clustersP))

        ax.plot(fullTime, combine / self.cellArea, linewidth=2,linestyle="-", label="clusters", color='blue', alpha=0.95, zorder=3)
        ax.plot(fullTime, small_combine / self.cellArea, linewidth=2, linestyle="--", label="small clusters", color='blue', alpha=0.95, zorder=3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=0.025)
        ax.set_xlabel("time (s)", fontsize=9)
        ax.set_ylabel("Cluster Density (/$\\mu$m$^2$)", color='blue',fontsize=9)
        ax.tick_params(axis='both',labelcolor='blue', labelsize=9)
        ax.set_xticks([0, 1000,2000])
        # Get rid of bound box on the top.
        ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        

        #now add the monomer density on the membrane, on the right axis
        # Second line (right y-axis)
        ax2 = ax.twinx()
        memCopies=self.calculate_munc13_on_membrane(copies)
        memCopiesPost=self.calculate_munc13_on_membrane(copiesP)
     
        combineMem = np.concatenate((memCopies, memCopiesPost))
        upperLim=combineMem[-1]/self.cellArea
        ax2.plot(fullTime, combineMem/self.cellArea, color='cyan', linewidth=2)
        ax2.set_ylabel("Munc13_mem (/$\\mu$m$^2$)", fontsize=9, color='cyan')
        ax2.tick_params( labelcolor='cyan', labelsize=8)
        ax2.set_ylim(bottom=0, top=4)

        plt.tight_layout()
        plt.savefig(f"/Users/margaret/Dropbox/r2025/Munc13/IMAGES/clusterDensity_vs_time_{fileStr}.png",dpi=300)
        plt.show()

    def plot_mycluster_time(self, Y, figsize=(4, 3), fontsize=12, dpi=300):
        """Plot the time course of the cluster 1X model solution"""
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
        clusters = copies[22]+copies[18]+copies[16]#
        small_clusters = copies[21]+copies[20]+copies[19]+copies[17]+copies[15]+copies[14]+copies[13]#3-5mers.
        ax.plot(self.timePoints,clusters, linestyle="-", label="copies of clusters", color=c_pre, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, 6*clusters, linestyle="-", label="munc in larger clusters", color="green", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[6], linestyle="-", label="MX", color=c_post, alpha=0.95, zorder=3)
        ax.plot(self.timePoints, small_clusters, linestyle="-", label="copies of small clusters", color="blue", alpha=0.95, zorder=3)
        
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
        """Plot the time course of the cluster 1X model solution"""
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
        #ax.plot(self.timePoints, copies[5], linestyle="-", label="MX", color="cyan", alpha=0.95, zorder=3)
       # ax.plot(self.timePoints, copies[4], linestyle="-", label="X", color="blue", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[8], linestyle="-", label="M2X", color="pink", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[13], linestyle="-", label="M3X", color="gray", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[14], linestyle="-", label="M4X", color="gold", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[15], linestyle="-", label="M5X", color="yellow", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[16], linestyle="-", label="M6X", color="indigo", alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, copies[18], linestyle="-", label="M5X3", color=(0.1, 0.2, 0.8), alpha=0.95, zorder=3)
        #ax.plot(self.timePoints, copies[20], linestyle="-", label="M6X3", color=(0.7, 0.2, 0.8), alpha=0.95, zorder=3)
        
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
        """Plot the time course of the cluster 1X model solution"""
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
        ax.plot(self.timePoints, copies[9], linestyle="-", label="DX", color="purple", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[17], linestyle="-", label="D2X", color="lime", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[18], linestyle="-", label="D3X", color="olive", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[19], linestyle="-", label="DMX", color=(0.5, 0.2, 0.8), alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[20], linestyle="-", label="DM2X", color="brown", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[21], linestyle="-", label="D2MX", color="teal", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[22], linestyle="-", label="D2M2X", color=(0.1, 0.1, 0.5), alpha=0.95, zorder=3)
        
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

    def plot_sequester_time(self, Y, figsize=(5, 4), fontsize=12, dpi=300):
        """Plot the time course of the 1X model solution"""
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
        ax.plot(self.timePoints, copies[11], linestyle="-", label="SQ", color="green", alpha=0.95, zorder=3)
        ax.plot(self.timePoints, copies[12], linestyle="-", label="S2Q2", color="purple", alpha=0.95, zorder=3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Copy numbers")
        #ax.set_xlim(0, 10)
        #ax.set_ylim(0, 600)
        ax.legend()
        fig.tight_layout()

        #fig.savefig("../fig/fig_total_recruitment_wt.svg") 
        #fig.savefig("../fig/fig_total_recruitment_wt.png", dpi=dpi) 
        plt.show(block=True)

    def plot_density_vs_exp(self, sol, solPost, fileStr, whichExp='WT', figsize=(2.5, 2), fontsize=9, dpi=300):
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

        #candidate = self.filteredSolutions[select]
        copies = sol[:,-1] * self.cellVolume * 602
        clusters = copies[22] + copies[18] + copies[16]
        copiesP = solPost[:,-1] * self.cellVolume * 602
        clustersP = copiesP[22] + copiesP[18] + copiesP[16]
        area=self.cellArea
        fig, ax = plt.subplots(figsize=figsize)

        
        
        mutants = ['WT']  #, r'$\Delta$C2A', 'shRNA RIM2', r'$\Delta$C2B']

        
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
        if whichExp=='C2A':
            densPre=self.density_c2a_pre
            densPost=self.density_c2a_post
        else:
            densPre=self.density_exp_pre
            densPost=self.density_exp_post

        ax.bar(x_exp_no,  densPre,  width=bar_width,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, densPost, width=bar_width, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   clusters/area,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, clustersP/area, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Cluster\ Density}\ (/\mu\mathrm{m}^2)$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels([], fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0, top=0.042)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.legend()
        fig.tight_layout()

        #fig.savefig("../fig/fig_diffusivity_wt.svg") 
        fig.savefig(f"/Users/margaret/Dropbox/r2025/Munc13/IMAGES/barplot_clusterDens_{fileStr}.png", dpi=dpi) 
        plt.show()

    def plot_lifetime_vs_exp(self, candidate, sol, solPost, fileStr, figsize=(2.5, 2), fontsize=9, dpi=300):
        
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

        #candidate = self.filteredSolutions[select]
        tau46, tau56 = self.calculate_lifetime_of_clusters(sol, candidate)
        tau46, tau56post = self.calculate_lifetime_of_clusters(solPost, candidate)
        
        
        fig, ax = plt.subplots(figsize=figsize)

        
        n = 1

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
        if fileStr=='C2A':
            lifePre=10
            lifePost = 10
        else:
            lifePre=10
            lifePost = 10

        ax.bar(x_exp_no,  lifePre,  width=bar_width,  capsize=4,
            color=c_no,   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim, lifePost, width=bar_width, capsize=4,
            color=c_stim, edgecolor='black', label='Exp (STIM)')

        # MODEL: hatched (use white fill + colored edge so the hatch stands out)
        ax.bar(x_mod_no,   tau56,  width=bar_width,
            facecolor='white', edgecolor=c_no,   hatch=hatch_model, label='Model (NO STIM)')
        ax.bar(x_mod_stim, tau56post, width=bar_width,
            facecolor='white', edgecolor=c_stim, hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Cluster\ lifetime}\ (s)$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels([], fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0, top=20)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.legend()
        fig.tight_layout()

        #fig.savefig("../fig/fig_diffusivity_wt.svg") 
        fig.savefig(f"/Users/margaret/Dropbox/r2025/Munc13/IMAGES/barplot_Lifetimes_{fileStr}.png", dpi=dpi) 
        plt.show()

    def plot_track_increase_vs_exp(self, sol, solPost, mut, mutPost, fileStr, figsize=(2.5, 2)):
        sns.set_style("ticks")
        dpi=300
        fontsize=9

        sns.set_context("paper", rc={
            "font.size": fontsize,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "font.family": "serif"
        })

        

        memCopies=self.calculate_munc13_on_membrane(sol[:,-1]*self.cellVolume *602)
        memCopiesPost=self.calculate_munc13_on_membrane(solPost[:,-1]*self.cellVolume *602)
        
        #WT 
        simIncreaseWT=memCopiesPost/memCopies

        #now dC2A
        memCopies=self.calculate_munc13_on_membrane(mut[:,-1]*self.cellVolume *602)
        memCopiesPost=self.calculate_munc13_on_membrane(mutPost[:,-1]*self.cellVolume *602)
        #C2A
        simIncreaseC2A=memCopiesPost/memCopies


        fig, ax = plt.subplots(figsize=figsize)

       
        n = 1

        # Layout: [Exp(NO), Exp(STIM)]   gap   [Model(NO), Model(STIM)]
        bar_width   = 0.18
        small_gap   = 0.10   # separation between Exp pair and Model pair
        group_gap   = 0.60   # separation between mutant groups
        group_width = 4*bar_width + small_gap

        centers = np.arange(n) * (group_width + group_gap)
        lefts = centers - group_width/2

        x_exp_no   = lefts + 0*bar_width
        x_exp_stim = lefts + 1*bar_width + small_gap
        x_mod_no   = lefts + 2*bar_width + group_gap
        x_mod_stim = lefts + 3*bar_width + group_gap + small_gap

        # Colors and styles
        c_no, c_stim = 'black', 'red'
        hatch_model = '///'

       

        ax.bar(x_exp_no, self.recruitmentStim,  width=bar_width,  capsize=4,
            color='red',   edgecolor='black', label='Exp (NO STIM)')
        ax.bar(x_exp_stim,  simIncreaseWT,  width=bar_width,
            facecolor='white', edgecolor='red',   hatch=hatch_model, label='Model (NO STIM)')
     
        #Now C2A
        ax.bar(x_mod_no,   self.recruitmentStimC2A,  width=bar_width,capsize=4,
            color='orange',   edgecolor='black', label='Exp (C2A)')
        ax.bar(x_mod_stim, simIncreaseC2A, width=bar_width,
            facecolor='white', edgecolor='orange', hatch=hatch_model, label='Model (STIM)')

        # Axes/labels
        ax.set_ylabel(r"$\mathrm{tracks\ Stim / tracks}$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels([], fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0, top=5)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.legend()
        fig.tight_layout()

        
        fig.savefig(f"/Users/margaret/Dropbox/r2025/Munc13/IMAGES/barplot_trackIncrease_{fileStr}.png", dpi=dpi) 
        plt.show()

    def plot_diffusion_as_barplot(self, D1, D1post, D2, D2post, fileStr):
        #make a bar plot of diffusion following lower
        fig, ax = plt.subplots(figsize=(3,2))
        bar_width=0.18

        fontsize=9
        gap=0.1
        c1=0.9
        hatch_model = '///'
        centers=[c1, c1+bar_width, c1+2*bar_width+gap, c1+3*bar_width+gap]
        ax.bar(centers[0],  D1,  width=bar_width,  capsize=4,
                    facecolor='white',   edgecolor='black', hatch=hatch_model)
        ax.bar(centers[1], D1post, width=bar_width, capsize=4,
                    facecolor='white', edgecolor='red', hatch=hatch_model)
        ax.bar(centers[2],  D2,  width=bar_width,  capsize=4,
                    facecolor='white',   edgecolor='black', hatch=hatch_model)
        ax.bar(centers[3], D2post, width=bar_width, capsize=4,
                    facecolor='white', edgecolor='red', hatch=hatch_model)
       
        # Axes/labels
        ax.set_ylabel(r"$\mathrm{Diffusion\ Constant}\ (/\mu\mathrm{m}^2/s)$", fontsize=fontsize)
        ax.set_xticks(centers)
        ax.set_xticklabels([], fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize * 0.8)
        ax.set_ylim(bottom=0, top=0.08)

        # Adjust layout to fit
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Cleanup
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        dpi=300
        fig.savefig(f"/Users/margaret/Dropbox/r2025/Munc13/IMAGES/diffusion_WT_vs_{fileStr}.png", dpi=dpi) 
            
    def calculate_population_average_diffusion(self, sol):     
        """MEJ
         Calculate the diffusivity of the full population based on the
         sizes of the species, only on the membrane
         For single X model
         """ 
       
        copies = sol[:,-1] * self.cellVolume * 602
        #this is the total copies of munc13 on the membrane.
        #this needs to be stoichiometric, because they measure 
        #diffusivity of inidividual munc13, and not separate complexes.
        # So a complex with 6 munc13 is 6x more likely to be measured towards diffusivity. 
        memMunc =self.calculate_munc13_on_membrane(copies)
        #this is the total copies of
        
        #Diffusion of complexes depends on DM, DX, DD
        #we are going to use a simple and very conservative model, where the radii
        #and not the mass sums linearly. If its the mass, you need 
        # a model for scaling of the radius, e.g. 1/3 for space filing, or 3/5 for 
        # unstructured. 
        DtM=self.DtM
        DtX=self.DtX
        DtD = self.DtD
        DtQ = self.DtQ
        DtS=DtM*100 #solution diffusion
        DtMX = 1.0/(1.0/self.DtM + 1.0/self.DtX)
        DtMMX = 1.0/(1.0/DtMX + 1.0/self.DtM)
        DtDX = 1.0/(1.0/self.DtX + 1.0/self.DtD)
        DtM3X = 1.0/(1.0/DtMMX + 1.0/self.DtM)
        DtM4X = 1.0/(1.0/DtM3X + 1.0/self.DtM)
        DtM5X= 1.0/(1.0/DtM4X+1.0/self.DtM)
        DtM6X= 1.0/(1.0/DtM5X+1.0/DtM)
        DtD2X= 1.0/(1.0/DtDX+ 1.0/DtD)
        DtD3X=1.0/(1.0/DtD2X + 1.0/DtD)
        DtDMX=1.0/(1.0/DtDX + 1.0/DtM)
        DtD2MX=1.0/(1.0/DtDMX + 1.0/DtD)
        DtD2M2X=1.0/(1.0/DtD2MX + 1.0/DtM)
        DtDM2X=1.0/(1.0/DtDMX + 1.0/DtM)
        DtSX = 1.0/(1.0/DtX + 1.0/DtS)
        DtSQ = 1.0/(1.0/DtQ + 1.0/DtS)
        DtS2Q2 = 1.0/(1.0/DtSQ + 1.0/DtSQ)
       

        print(f"DtM, {DtM}. DtX {DtX}. DtD {DtD}")
        print(f"DtM6X, {DtM6X}. DtD3X {DtD3X}. DtMX {DtMX}. DtDX {DtDX}")
        print(f"memMunc {memMunc}. M {copies[2]}. D {copies[3]}. ")

        averageD = (copies[2]*self.DtM + 2*copies[3]*self.DtD +copies[6]*DtMX+\
                    DtSX*copies[7]+2*DtMMX*copies[8]+\
                    2*DtDX*copies[9]+DtSQ*copies[11]+2*DtS2Q2*copies[12]+\
                    3*DtM3X*copies[13]+ 4*DtM4X*copies[14]+ 5*DtM5X*copies[15]+ 6*DtM6X*copies[16]+\
                    4*DtD2X*copies[17]+ 6*DtD3X*copies[18]+ 3*DtDMX*copies[19]+ 4*DtDM2X*copies[20]+\
                          5*DtD2MX*copies[21]+6*DtD2M2X*copies[22])/memMunc
        
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
        clusterCopies=6*(copies[22]+copies[18]+copies[16])#+5*(copies[18]+copies[19])
        memCopies=self.calculate_munc13_on_membrane(copies)
      
        muncTotal=memCopies+copies[0]+2*copies[5]
        #totalCopies=np.sum(copies[-1]) #this does not account for stoichiometry
        P_cluster=clusterCopies/muncTotal #fraction of total inclusters
        P_cluster_normMem=clusterCopies/memCopies #fraction of membrane bound in clusters
        P_mem = memCopies/muncTotal #fraction of total on the membrane.
        pMono = (copies[2]+copies[11])/memCopies #fraction on membrane as monomers, M and SQ
        pDimer = 2*(copies[3]+copies[12])/memCopies #fraction on membrane as dimers, D and S2Q2
        pMX = copies[6]/memCopies #fraction monomers bound to nucleator
        pDX = copies[9]/memCopies #fraction dimers bound to nucleator

        return P_cluster, P_cluster_normMem, P_mem, pMono, pDimer, pMX, pDX


    def compute_all_metrics(self, candidate):
        
        sol, solPost = self.simulate(candidate)
        candidate_dc2a=list(candidate)
        candidate_dc2a[2] = 0 #this sets kfmm to zero.
        candidate_dc2a[12] = 0 #this sets kfdd to zero (no in cluster transition to 2M->D)
        mut, mutPost = self.simulate(candidate_dc2a)
       

        avgD = self.calculate_population_average_diffusion(sol)
        avgDpost = self.calculate_population_average_diffusion(solPost)
        mutD = self.calculate_population_average_diffusion(mut)
        mutDpost = self.calculate_population_average_diffusion(mutPost)
        metricsD = np.array(4)
        metricsD=[avgD, avgDpost, mutD, mutDpost]
        #now calculate other fitness metrics
        tau46, tau56 = self.calculate_lifetime_of_clusters(sol, candidate)
        tau46, tau56post = self.calculate_lifetime_of_clusters(solPost, candidate)
        density=self.calculate_cluster_density(sol)
        densPost=self.calculate_cluster_density(solPost)
        smallClusterDens, smallS=self.calculate_small_cluster_density(sol)
        L=len(density)
        n75=np.int64(L*0.75)
        ssDelta=density[-1]-density[n75]     
        ssDeltaPost=densPost[-1]-densPost[n75]     
        percClusterTotal, percClustMem, percMem, pMono, pDimer, pMX, pDX=self.calc_percentages_cluster(sol)
        percClusterTotal, percClustMemPost, percMem, pMono, pDimerPost, pMX, pDX=self.calc_percentages_cluster(solPost)
        memCopyMuncPre =   self.calculate_munc13_on_membrane(sol[:,-1]*self.cellVolume*602)      
        memCopyMuncPost =   self.calculate_munc13_on_membrane(solPost[:,-1]*self.cellVolume*602)

        #ratio of densities on membrane at steady-state.
        recruitStim=memCopyMuncPost/memCopyMuncPre
        if(np.isnan(recruitStim)):
            recruitStim = 1.0
        metricWT = np.array(11)
        metricWT=[tau56, tau56post, density[-1], densPost[-1], recruitStim, percClustMem, percClustMemPost, pDimer, pDimerPost, ssDelta, ssDeltaPost]
    
        #now compute all the same metrics for the dC2A.
        #now calculate other fitness metrics

        tau46, tau56 = self.calculate_lifetime_of_clusters(mut, candidate)
        tau46, tau56post = self.calculate_lifetime_of_clusters(mutPost, candidate)
        density=self.calculate_cluster_density(mut)
        densPost=self.calculate_cluster_density(mutPost)
        smallClusterDens, smallS=self.calculate_small_cluster_density(mut)
        L=len(density)
        n75=np.int64(L*0.75)
        ssDelta=density[-1]-density[n75]     
        ssDeltaPost=densPost[-1]-densPost[n75]     
        percClusterTotal, percClustMem, percMem, pMono, pDimer, pMX, pDX=self.calc_percentages_cluster(mut)
        percClusterTotal, percClustMemPost, percMem, pMono, pDimerPost, pMX, pDX=self.calc_percentages_cluster(mutPost)
        memCopyMuncPre =   self.calculate_munc13_on_membrane(mut[:,-1]*self.cellVolume*602)      
        memCopyMuncPost =   self.calculate_munc13_on_membrane(mutPost[:,-1]*self.cellVolume*602)

        #ratio of densities on membrane at steady-state.
        recruitStim=memCopyMuncPost/memCopyMuncPre
        if(np.isnan(recruitStim)):
            recruitStim = 1.0
        metricC2A=np.array(11)
        metricC2A=[tau56, tau56post, density[-1], densPost[-1], recruitStim, percClustMem, percClustMemPost, pDimer, pDimerPost, ssDelta, ssDeltaPost]
        return metricWT, metricC2A, metricsD
        

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
        inputFile = "../data/dummy.txt",
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
        # Best subset by fitness, larger fitness = better)
        best_eDF = eDF.nlargest(n_best, "Fitness")
        best_row = eDF.nlargest(1, "Fitness").iloc[0]

        print(f"Total solutions loaded: {n_total}")
        print(f"Using top {percent}% by Fitness -> {n_best} solutions.")

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
            'kfq': r'$kf_{Q}\; (\mu\mathrm{M}^{-1}\,\mathrm{s}^{-1})$',
            'krq': r'$kr_{Q}\; (\mathrm{s}^{-1})$',
            'kfdd': r'$kf_{DD}\; (\mathrm{s}^{-1})$',
            'eLoop': r'$exp^{loop}\; $',
            'eDF': r'$exp^{DF}\; $',
            'Sd': r'$Sd\; $',
            'S0': r'$S_{0}\; (\mu\mathrm{M})$',
            'R0': r'$R_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            'X0': r'$X_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            'Q0': r'$Q_{0}\; (\mathrm{copies}/\mu\mathrm{m}^{2})$',
            #A': r'$exp^{CA}\; $',

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

    
    def plot_parameter_KD_summary(
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
        inputFile = "../data/dummy.txt",
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

        
        # Best-by-rank subset size
        n_total = len(eDF)
        n_best = max(1, int(np.floor(n_total * (percent / 100.0))))
        # Best subset by fitness (higher fitness=better)
        best_eDF = eDF.nlargest(n_best, "Fitness")
        best_rowP = eDF.nlargest(1, "Fitness").iloc[0]

        print(f"Total solutions loaded: {n_total}")
        print(f"Using top {percent}% by Fitness -> {n_best} solutions.")

        #-----------------#
        #create a new datastructure from best_eDF that evaluates the KD values for each row
        kdnames = ['kDsr','kDmm','kDmx','kDc','kDq']
        kdDF=pd.DataFrame(columns = kdnames)
        #calculate the KD values from kr/kf
        for idx, row in best_eDF.iterrows():
            # Access row data by column name, e.g. row['column_name']
            #print(idx, row)
            newRow=np.array(5) 
            newRow=[row['krsr']/row['kfsr'], row['krmm']/row['kfmm'], row['krmx']/row['kfmx'], row['krc']/row['kfc'], row['krq']/row['kfq']]        
            kdDF.loc[len(kdDF)]=newRow

        best_row=pd.DataFrame(columns = kdnames)
        newRow=[best_rowP['krsr']/best_rowP['kfsr'], best_rowP['krmm']/best_rowP['kfmm'], best_rowP['krmx']/best_rowP['kfmx'], best_rowP['krc']/best_rowP['kfc'], best_rowP['krq']/best_rowP['kfq']]        
        best_row.loc[len(best_row)]=newRow 

        # -------------------- labels --------------------------------------------
        parms_name_map = {
            'kDsr': r'$K_{D,SR}\; (\mu\mathrm{M})$',
            'kDmm': r'$K_{D,MM}\; (\mu\mathrm{M})$',
            'kDmx': r'$K_{D,MX}\; (\mu\mathrm{M})$',
            'kDc': r'$K_{D,C}\; (\mu\mathrm{M})$',
            'kDq': r'$K_{D,Q}\; (\mu\mathrm{M})$', 

        }

        tiny = np.finfo(float).tiny

        # -------------------- allowed ranges (grey) ------------------------------
        pr = {}
        for p in kdnames:
            lo=0.0001
            hi=100000
            pr[p] = {"min": lo, "max": hi}

        # ------------- range across the best `percent`% (light blue) ------------
        subset_ranges = {}
        for p in kdnames:
            col = kdDF[p].to_numpy(float)
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
        y_min = min(pr[p]["min"] for p in kdnames)
        y_max = max(pr[p]["max"] for p in kdnames)

        # ----------------------------- plot -------------------------------------
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        x = np.arange(len(kdnames), dtype=float)
        half = bar_width / 2.0

        for i, p in enumerate(kdnames):
            alo, ahi = pr[p]["min"], pr[p]["max"]
            wlo, whi = subset_ranges[p]
            print("p: ", p)
            print("best_row", best_row)

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
        ax.set_xlim(-0.5, len(kdnames) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([parms_name_map.get(p, p) for p in kdnames],
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
        self.indpb = 0.85  # Probability of mutation per gene
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
                "Rank, Fitness, kfsr, krsr, kfmm, krmm, kfmx, krmx, kfc, krc, kfq, krq, eLoop, eDF, kfdd, Sd, stimUpSR, S0, R0, X0, Q0\n"
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
                
                kfq = opt_params[8]
                krq = opt_params[9]
                eLoop = opt_params[10]
                eDF = opt_params[11]
                kfdd = opt_params[12]
                stimUpSR = opt_params[14]
                S0 = opt_params[15]
                R0 = opt_params[16]
                
                X0 = opt_params[17]
                Q0 = opt_params[18]
                Sd = opt_params[13]
                

                # Write a line per solution
                f.write(
                    f"{rank}, {fitness_val:.4f}, "
                    f"{kfsr:.5g}, {krsr:.5g}, {kfmm:.5g}, "
                    f"{krmm:.5g}, {kfmx:.5g}, {krmx:.5g}, "
                    f"{kfc:.5g}, {krc:.5g}, {kfq:.5g}, "
                    f"{krq:.5g}, {eLoop:.5g}, {eDF:.5g}, "
                    f"{kfdd:.5g}, {Sd:.5g}, {stimUpSR:.5g}, "
                    f"{S0:.5g}, {R0:.5g}, {X0:.5g},{Q0:.5g}\n"
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
        "kfmm":      {"min": 0.1, "max": 10},   # kfMM uM-1s-1
        "krmm":      {"min": 0.01,   "max": 10}, # krMM s-1
        "kfmx":      {"min": 0.001, "max": 10},   # kf1x uM-1s-1
        "krmx":      {"min": 0.01,   "max": 1000}, # kr1x s-1
       # "kfc_nostim":      {"min": 0.001, "max": 10},   # kfc_nostim
        "kfc":      {"min": 0.001, "max": 10},   # kfc_stim uM-1s-1
        "krc":      {"min": 0.01, "max": 1000},   # krc
        "kfq":      {"min": 0.001, "max": 10},   # kx2 uM-1s-1
        "krq":      {"min": 0.01,   "max": 1000}, # krx2 s-1
        "eLoop":      {"min": 0.0001,   "max": 10}, # exp(free energy kT units <0).
        "eDF":      {"min": 0.0001,   "max": 10}, # exp(free energy kT units <0).
        "kfdd":     {"min": 0.01,   "max": 1}, # kfdd unimolecular: s-1
        "Sd":     {"min": 0.1,   "max": 10}, # scalar to change on/off kinetics of dimer to cluster
        "stimUpSR":       {"min": 1,   "max": 100}, # stimUpSR: scale factor >1
        "S0":        {"min": 0.001, "max": 5},   # S0 (uM)
        "R0":        {"min": 0.1, "max": 10000},   # R0 (/um^2)
        #"D1":        {"min": 0.05,   "max": 5}, # D1
        #"D1_over_D2":        {"min": 1.5,   "max": 5}, # D2
        "X0":        {"min": 0.01,   "max": 100}, # X0  (/um^2)
        "Q0":       {"min": 0.01,   "max":  100}, # Q0  (/um^2)
       
         
    }

    # Order in which the solver will read parameters from a candidate
    # so, e.g. candidate[0] = kfsr.
   
    params_to_optimizeCa = np.array([
        "kfsr","krsr","kfmm","krmm","kfmx","krmx","kfc","krc","kfq","krq","eLoop","eDF","kfdd","Sd","stimUpSR","S0","R0","X0","Q0"
    ])
    print("number of parameters to optimize: ", params_to_optimizeCa.size)
    # GA settings
    popSize = 20000
    nGen = 5

    # Instantiate the model and solver, including max time limit.
    maxTime = 1000.0
    model = Munc13(parameter_ranges, params_to_optimizeCa, t_max=maxTime)
    #the solver can be passed a specific filename for the solutions.
    random_number = np.random.randint(1, 10000)
    filename = f"../data/testParms_Dterm_Lifetimes_{random_number}.txt"
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

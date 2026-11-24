import pathlib
import sys
from colorama import Fore, Back, Style, init

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.fig02_lindblad_decoherence import lindblad_stats
from scripts.fig03_mcwf_decoherence import mcwf_stats
from scripts.fig04_brune_scaling import scaling
from scripts.fig05_mcwf_decoherence_seed_dependence import seed_dependence
from scripts.fig06_mcwf_decoherence_Ntraj_dependence import Traj_dependence
from scripts.fig07_mcwf_advantages import mcwf_advantages
from mcwf import process
init(autoreset=True)
if __name__=="__main__":
    # print("---------------------------------------------------------------------------------------------\n")
    # print("                                            FIG. 2")
    # print("---------------------------------------------------------------------------------------------")
    # print(Fore.BLUE + '\n\n- Running Lindblad stats function - Will produce figure showing Parity, Purity, Mean photon Number, Wigner function')
    # lindblad_stats()
    # print("---------------------------------------------------------------------------------------------\n")

    # print("                                            FIG. 3")
    # print("---------------------------------------------------------------------------------------------")    
    # print(Fore.BLUE + '- Running MCWF stats function - Will produce figures showing Parity, Purity, Mean photon Number, Wigner function')
    # mcwf_stats()
    # print("---------------------------------------------------------------------------------------------\n")

    # print("                                            FIG. 4")
    # print("---------------------------------------------------------------------------------------------")
    # print(Fore.BLUE + '- Runnign Scaling benchmarks function - Will produce figures showing decoherenec decay for alpha sweeps, decoherence time benchmarks ')
    # scaling()
    # print("---------------------------------------------------------------------------------------------\n")

    # print("                                            FIG. 5")
    # print("---------------------------------------------------------------------------------------------")
    # print(Fore.BLUE + '- Runnign Seed Dependence stats script - Sweeps accros seeds and will produce figures showing Parity, Purity, Mean photon Number for dephasing-photon loss-both ')
    # seed_dependence()
    # print("---------------------------------------------------------------------------------------------\n")

    # print("                                            FIG. 6")
    # print("---------------------------------------------------------------------------------------------")
    # print(Fore.BLUE + '- Runnign trajectory number dependence stats script - Sweeps accros seeds and will produce figures showing Parity, Purity, Mean photon Number for dephasing-photon loss-both ')
    # Traj_dependence()
    # print("---------------------------------------------------------------------------------------------\n")

    # print("                                            FIG. MCWF")
    # print("---------------------------------------------------------------------------------------------")
    # print(Fore.RED + '- Runnign MCWF function - Will produce figures showing decoherenec decay for alpha sweeps, decoherence time benchmarks ')
    # process()
    # print("---------------------------------------------------------------------------------------------\n")

    print("                                            FIG. 7")
    print("---------------------------------------------------------------------------------------------")
    print(Fore.RED + '- Runnign MCWF advantage - Will produce figures showing Lindblad vs MCWF on time and memory complexity and the evolution of parit across time')
    
    mcwf_advantages()

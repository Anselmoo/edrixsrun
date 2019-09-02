__all__ = ['print_wavefunction', 'save_spc', 'save_rixs']
import numpy as np


def save_stk(en, stk):
    print(en, stk)
    # np.savetxt(fname=fname,X=[],delimiter='\t',header=header)


def print_wavefunction(eng, wfc, spin, lang, LS, fname):
    """
    Printing function for the wave function analysis

    Parameters
    ----------

    eng: 1d float numpy-array
        Contains the relative state-energies
    wfc: 2d float numpy-array
        List with MRCI-wavefunction
    spin: list with 3 x 1D-float numpy-array
        List of Sx, Sy, Sz, and S2 values
    lang: list with 3 x 1D-float numpy-array
        List of Lx, Ly, Lz, and L2 values
    LS: 1d float numpy-array
        LS-values for calculationg the J2 = L2 + S2 + 2 * LS
    fname: string
        Filename
    :return:
    """
    print_header = ("\tEnergy(eV)\tEnergy(eV)\t3d-alpha\t3d-beta\t\t2p-alpha\t2p-beta"
                    "\t\t < S_x > \t < S_y > \t < S_z > \t < S2 > "
                    "\t\t < L_x > \t < L_y > \t < L_z > \t < L2 > "
                    "\t\t < LS > \t < J2 >")
    print(print_header)

    write_header = ("Energy(eV)\tEnergy(eV)\t3d-alpha\t3d-beta\t2p-alpha\t2p-beta"
                    "\t < S_x > \t < S_y > \t < S_z > \t < S2 > "
                    "\t < L_x > \t < L_y > \t < L_z > \t < L2 > "
                    "\t < LS > \t < J2 >\n")

    # Exporting the wavefunction-decompensition to txt-file
    txtfile = open(fname, 'w+')
    txtfile.write(write_header)
    # Calculating the absolute energy starting with 0.00 eV
    en_0 = np.min(eng)
    for en, ci, sx, sy, sz, s2, lx, ly, lz, l2, ls in zip(eng, wfc, spin[0], spin[1], spin[2], spin[3], lang[0],
                                                          lang[1], lang[2], lang[3], LS):
        print_str = ("\t{:8.5f}\t{:8.5f}\t{}\t{}\t\t{}\t{}"
                     "\t\t{:8.5f}\t{:8.5f}\t{:8.5f}\t{:8.5f}"
                     "\t\t{:8.5f}\t{:8.5f}\t{:8.5f}\t{:8.5f}"
                     "\t\t{:8.5f}\t{:8.5f}".format(en, -en_0 + en,
                                                   str(ci[0:5])[1:-1], str(ci[5:10])[1:-1],
                                                   str(ci[10:13])[1:-1], str(ci[13:16])[1:-1],
                                                   sx, sy, sz, s2, lx, ly, lz, l2, ls, s2 + l2 + 2 * ls))
        print(print_str)

        write_str = ("{:8.5f}\t{:8.5f}\t{}\t{}\t{}\t{}"
                     "\t{:8.5f}\t{:8.5f}\t{:8.5f}\t{:8.5f}"
                     "\t{:8.5f}\t{:8.5f}\t{:8.5f}\t{:8.5f}"
                     "\t{:8.5f}\t{:8.5f}\n".format(en, -en_0 + en,
                                                   str(ci[0:5])[1:-1], str(ci[5:10])[1:-1],
                                                   str(ci[10:13])[1:-1], str(ci[13:16])[1:-1],
                                                   sx, sy, sz, s2, lx, ly, lz, l2, ls, s2 + l2 + 2 * ls))
        txtfile.write(write_str)
    txtfile.close()


def save_spc(en, spc, fname, shift=0., type='XAS'):
    """
    Saving-Function for the Absorption- or Emission-Spectra

    Parameters
    ----------

    en: 1d float numpy-array
        Contains the relative absorption- or emission-energies

    spc: float numpy-array-list (1d or 2d array)
        Contains the spc-intensities

    fname: str
        filename-part for saving the spc-data as ascii

    shift: float
        Relative energy to shift the absorption or emission-energies to the right edge-jump energy

    type: str
        str for labeling the ascii as XAS- or XES-type
    """
    # For loop for the number of polarizations
    for i, sp in enumerate(spc.T):  # Important transpose the list for the right iteration
        # Creating the filename
        save_str = ('{}_{}_{}.txt').format(fname, type, str(i + 1))
        # Saving via numpy
        header = ('Energy_{}\tIntensity_{}'.format(str(i + 1), str(i + 1)))
        np.savetxt(save_str, np.array([en + shift, sp.T]).T, delimiter='\t', header=header)


def save_rixs(inc, emi, rixs, fname, shift=0.):
    """
    Saving-Function for the Absorption- or Emission-Spectra

    Parameters
    ----------

    inc: 1d float numpy-array
        Contains the relative incident-(absorption) energies

    emi: 1d float numpy-array
        Contains the relative emission-energies

    rixs: float numpy-array-list (2d or 4d array)
        Contains the spc-RIXS-intensities

    fname: str
        filename-part for saving the spc-data as ascii

    shift: float
        Relative energy to shift the incident-energies to the right edge-jump energy
    """
    # For loop for the number of polarizations

    for i, sp in enumerate(rixs.T):

        # Initial the 2d-array for saving the 2d-RIXS-plane as three column array
        # Incident-Energy, Emission-Energy, RIXS-Intensity
        tmp_rixs = np.zeros((len(inc) * len(emi), 3), dtype=float)
        index = 0  # Index to catch the inc*emi-length
        for j in range(sp.shape[0]):
            for k in range(sp.shape[1]):
                tmp_rixs[index, 0] = inc[j] + shift  # Incident Energy
                tmp_rixs[index, 1] = emi[k]  # Emission Energy
                tmp_rixs[index, 2] = sp[j, k]  # RIXS-Intensity
                index += 1
        # Creating the filename
        save_str = ('{}_RIXS_{}.txt').format(fname, str(i + 1))
        header = ('Energy_Inc_{}\tEnergy_Ems_{}\tIntensity_{}'.format(str(i + 1), str(i + 1), str(i + 1)))
        np.savetxt(save_str, tmp_rixs, delimiter='\t', header=header)

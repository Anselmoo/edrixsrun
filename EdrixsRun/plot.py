__all__ =['xas_spectra','xes_spectra','plot_show','rixs_spectra','rxas_spectra','rxes_spectra']
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


def xas_spectra(en,spc,shift=0.,fname='XAS'):


    # plot XAS
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    plt.plot(en+shift,spc)
    # Here are the designs-settings
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    plt.xlabel(r'Energy of incident photon (eV)')
    plt.ylabel(r'XAS Intensity (a.u.)')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.13,
                        top=0.95, wspace=0.05, hspace=0.00)
    plt.savefig(fname+'_xas.pdf')

def xes_spectra(en,spc,shift=0.,fname='XES'):


    # plot XES
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    plt.plot(en+shift,spc)
    # Here are the designs-settings
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    plt.xlabel(r'Energy of emitted photon (eV)')
    plt.ylabel(r'XES Intensity (a.u.)')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.13,
                        top=0.95, wspace=0.05, hspace=0.00)
    plt.savefig(fname+'_xes.pdf')

def rixs_spectra(incident, emission, rixs, shift=0.,fname='RIXS'):
    # plot RIXS map
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    a, b, c, d = min(incident) + shift, max(incident) + shift, min(emission), max(emission)
    #print(rixs[0].shape)

    plt.imshow(rixs[:,:,0].T , extent=[a, b, c, d],
               origin='lower', aspect='auto', cmap='rainbow',
               interpolation='bicubic')
    # Here are the designs-settings
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.xlabel(r'Energy of incident photon (eV)')
    plt.ylabel(r'Energy loss (eV)')
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.13,
                        top=0.95, wspace=0.05, hspace=0.00)
    plt.savefig(fname+'_rixs.pdf')

def rxas_spectra(en,spc,shift=0.,fname='XAS'):


    # plot total sum of RXAS
    plt.figure()
    ax = plt.subplot(1, 1, 1)

    for i, sp in enumerate(spc.T):
        plt.plot(en+shift,np.sum(sp,axis=0))
    # Here are the designs-settings
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    #ax.yaxis.set_major_locator(MultipleLocator(2.5))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.xlabel(r'Energy of incident photon (eV)')
    plt.ylabel(r'$\Sigma$ XAS Intensity (a.u.)')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.13,
                        top=0.95, wspace=0.05, hspace=0.00)
    plt.savefig(fname+'_total_xas.pdf')

def rxes_spectra(en,spc,shift=0.,fname='XES'):


    # plot total sum of RXES
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    for i, sp in enumerate(spc.T):
        plt.plot(en+shift,np.sum(sp,axis=1))
    # Here are the designs-settings
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    #ax.yaxis.set_major_locator(MultipleLocator(2.5))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    plt.xlabel(r'Energy of emitted photon (eV)')
    plt.ylabel(r'$\Sigma$ XES Intensity (a.u.)')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.13,
                        top=0.95, wspace=0.05, hspace=0.00)
    plt.savefig(fname+'_total_rxes.pdf')

def plot_show():
    plt.show()
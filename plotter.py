import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_subplots(ax, file_path, plot_title, x_label, y_label, l_cut, color=None, label=None):
    # Load data from file
    data = np.loadtxt(file_path)
    
    # Extract columns
    lbin = data[:, 0]  # First column: lbin
    Cl = data[:, 1]    # Second column: Cl
    sigma = data[:, 2] # Third column: sigma (errors)

    mask = lbin < l_cut
    lbin = lbin[mask]
    Cl = Cl[mask]
    sigma = sigma[mask]
    
    # Plot
    ax.errorbar(lbin, Cl  ,yerr= sigma, fmt='o', label=label, color=color)
    ax.set_title(plot_title, fontsize=14)
    ax.grid(alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

# File paths and titles
file_paths = [
    "results/Clkg_A_v1.5_lmax6144_mapC2s0.05_comp0.2_cutoff0.2_desi_dr1_0.80_z_2.10_PR4_PR4mask_lmin5_binsize5_sepnormTrue.txt",
    "results/Clgg_A_v1.5_lmax6144_mapC2s0.05_comp0.2_cutoff0.2_desi_dr1_0.80_z_2.10_PR4_PR4mask_lmin5_binsize5_sepnormTrue.txt",
    "results/Clkk_A_v1.5_lmax6144_mbinary.txt"
]
titles = [
    r"$C_\ell^{\kappa g}$",
    r"$C_\ell^{gg}$",
    r"$C_\ell^{\kappa \kappa}$"
]
colors = ['darkorange', 'navy', '#FF00FF']

# Create figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot each data set in individual subplots
for i, (path, title, color) in enumerate(zip(file_paths, titles, colors)):
    load_and_plot_subplots(
        axs[i // 2, i % 2], path, title, x_label=r"$\ell$", y_label=r"$C_\ell$", l_cut=105, color=color
    )

# Plot all together in the fourth subplot
for path, title, color in zip(file_paths, titles, colors):
    load_and_plot_subplots(
        axs[1, 1], path, r"Combined $D_\ell$ Plots", x_label=r"$\ell$", y_label=r"$C_\ell$", l_cut=105, color=color, label=title
    )
axs[1, 1].legend(loc="best",fontsize=10)

# Adjust layout
plt.tight_layout()
plt.savefig("results/plots/C_ells.pdf", format="pdf")
plt.show()

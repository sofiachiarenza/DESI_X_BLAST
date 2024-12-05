import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

def load_and_plot_subplots(ax, file_path, plot_title, x_label, y_label, l_cut, color=None, label=None, theory_data=None, theory_label=None, show_theory_legend=True):
    # Load measured data
    data = np.loadtxt(file_path)
    
    # Extract columns
    lbin = data[:, 0]  # First column: lbin
    Cl = data[:, 1]    # Second column: Cl
    sigma = data[:, 2] # Third column: sigma (errors)

    mask = lbin < l_cut
    lbin = lbin[mask]
    Cl = Cl[mask]
    sigma = sigma[mask]
    
    # Plot measured data
    ax.errorbar(lbin, Cl, yerr=sigma, fmt='o', label=label, color=color)
    
    # Overlay theoretical data if provided
    if theory_data is not None:
        lbin_theory = theory_data[:, 0]
        Cl_theory = theory_data[:, 1]
        theory_legend_label = theory_label if show_theory_legend else "_nolegend_"
        ax.plot(lbin_theory, Cl_theory, linestyle='--', linewidth=2, label=theory_legend_label, color=color)

    ax.set_title(plot_title, fontsize=14)
    ax.set_yscale("log")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

# Load config file
config_file = 'config.yaml'  # First argument: path to config.yaml
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Extract parameters from config
data_type = config['data_type']
mask_name = config['mask']['mask_name']
ap_scale = config['mask']['ap_scale']
comp_s = config['mask']['completeness']
zmin = config['data']['zmin']
zmax = config['data']['zmax']
lmin = config['computation']['lmin']
binsize = config['computation']['binsize']
sepnorm = config['computation']['sepnorm']

# Convert sepnorm to string for filename and title
sepnorm_str = "True" if sepnorm else "False"

# Build the filename dynamically
redshift_info = f"zmin_{zmin:.1f}_zmax_{zmax:.1f}"
if data_type == "data":
    type_info = "data"
elif data_type == "mock":
    type_info = "mock"
else:
    type_info = "unknown"

# Overall title
overall_title = (f"Type: {type_info}, zmin: {zmin:.1f}, zmax: {zmax:.1f}, "
                 f"Apodization Scale: {ap_scale}, Completeness: {comp_s:.2f}, "
                 f"Sepnorm: {sepnorm_str}")

# File paths and titles for measured data
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

# Load theoretical data and unpack columns
lbin_theory, Cl_theory_kk, Cl_theory_kg, Cl_theory_gg = np.loadtxt(
    "results/binned_theory_Cl_kk_kg_gg_lmin5.txt", unpack=True
)

# Apply mask for l_cut to theoretical data
l_cut = 105
mask2 = lbin_theory < l_cut
lbin_theory = lbin_theory[mask2]
Cl_theory_kk = Cl_theory_kk[mask2]
Cl_theory_kg = Cl_theory_kg[mask2]
Cl_theory_gg = Cl_theory_gg[mask2]

# Map theoretical data to respective measurements
theory_columns = [Cl_theory_kg, Cl_theory_gg, Cl_theory_kk]

# Create figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot each dataset in individual subplots with theory overlay
for i, (path, title, color) in enumerate(zip(file_paths, titles, colors)):
    load_and_plot_subplots(
        axs[i // 2, i % 2],
        file_path=path,
        plot_title=title,
        x_label=r"$\ell$",
        y_label=r"$C_\ell$",
        l_cut=l_cut,
        color=color,
        label="Measured",
        theory_data=np.column_stack((lbin_theory, theory_columns[i])),
        theory_label="Theory"
    )

# Plot all datasets together in the fourth subplot
for i, (path, title, color) in enumerate(zip(file_paths, titles, colors)):
    load_and_plot_subplots(
        axs[1, 1],
        file_path=path,
        plot_title=r"Combined $C_\ell$ Plots",
        x_label=r"$\ell$",
        y_label=r"$C_\ell$",
        l_cut=l_cut,
        color=color,
        label=title,
        theory_data=np.column_stack((lbin_theory, theory_columns[i])),
        theory_label=f"Theory {title}",
        show_theory_legend=False
    )
axs[1, 1].legend(loc="best", fontsize=10)

# Add overall title
fig.suptitle(overall_title, fontsize=16, y=0.98)

# Adjust layout and dynamically save the figure
output_plot_path = f"results/plots/Cls_{type_info}_{redshift_info}_ap{ap_scale}_comp{comp_s:.2f}_{sepnorm_str}.pdf"
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the overall title
plt.savefig(output_plot_path, format="pdf")
plt.show()
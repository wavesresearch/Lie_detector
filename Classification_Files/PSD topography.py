import mne
import matplotlib.pyplot as plt
import os

export_dir = "/Users/emerickleclerc/Library/Mobile Documents/com~apple~CloudDocs/ENSTA/2A/STAGE/PROJET/Lie_detector-main/PSD Exports"  # Le dossier de destination
data_dir = "/Users/emerickleclerc/Library/Mobile Documents/com~apple~CloudDocs/ENSTA/2A/STAGE/PROJET/Lie_detector-main/Dataset"  # Dossier contenant les fichiers .fif

# Parcourir tous les fichiers du dossier
for file in os.listdir(data_dir):
    if file.endswith(".fif") and "eeg" in file.lower():  # Ne traiter que les fichiers eeg .fif (pas les eve)
        file_path = os.path.join(data_dir, file)
        base_name = os.path.splitext(file)[0]
        export_path = os.path.join(export_dir, base_name + "_PSD.png")

        print(f"Traitement de : {file}")

        raw = mne.io.read_raw_fif(file_path, preload=True)

        # Calcul de la densité spectrale de puissance (bande alpha 8–12 Hz)
        psd = raw.compute_psd(fmin=8, fmax=12)

        # Moyenne du PSD pour chaque canal
        psds_mean = psd.get_data().mean(axis=1)  # Moyenne sur les fréquences

        # Créer une figure haute résolution
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)  # Taille + résolution

        # Tracer la topomap avec colorbar et colormap personnalisée
        im, _ = mne.viz.plot_topomap(
            psds_mean,
            raw.info,
            axes=ax,
            cmap='jet',
            show=False
        )

        # Ajouter la colorbar à droite
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.6)
        cbar.set_label('Power Spectral Density (µV²/Hz)', fontsize=12)

        # Ajouter un titre
        ax.set_title("PSD Topography - Bande Alpha (8–12 Hz)", fontsize=14)

        # Ajouter le nom du fichier en bas
        fig.text(0.5, 0.02, f"Fichier EEG : {file}", ha='center', fontsize=10, style='italic')

        # Affichage
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])  # Laisse de la place pour le label

        # Export image
        plt.savefig(export_path, dpi=300)
        plt.close(fig)  # Ferme la figure après sauvegarde

        print(f"Image exportée avec succès vers : {export_path}")
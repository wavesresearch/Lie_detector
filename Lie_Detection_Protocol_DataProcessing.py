import numpy as np
import mne
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
from mne.time_frequency import tfr_multitaper, tfr_morlet, csd_multitaper
from mne.stats import permutation_cluster_test as pcluster_test
from mne.minimum_norm import apply_inverse, apply_inverse_epochs
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
import asrpy
import tkinter as tk 
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time 
import logging

class EEGPreProcessor:
    def __init__(self,raw_fnames):
        self.epochs_combined = None
        self.raw_fnames = raw_fnames
        self.event_id = dict(lie=50, true=51)
        self.tmin, self.tmax = 0.0, 1.0
        self.baseline = (None, 0.5)
        self.reject = dict(eeg=150e-6)
        self.extract_raw_data(0,0)
        mne.set_log_level(logging.WARNING)
        self.processing()        

    def extract_raw_data(self,val_preprocessing, val_ROI):
        self.raws = []
        self.all_epochs = []

        for raw_fname in self.raw_fnames:
            raw = mne.io.read_raw_fif(raw_fname, preload=True)

            events = mne.find_events(raw)

            ch_names=['F3','FC5','T7','C3','CP5','TP9', 'P5','P6', 'PO3', 'POz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'Pz', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2', 'CP6', 'TP10', 'PO7', 'PO8']
            self.channel_of_interest = ['CP5'] 

            # Sélection des canaux spécifiques
            raw_picks = raw.pick_channels(self.channel_of_interest)

            # sample = raw.get_data()
            # sample = sample[:-1]
            # normalized_data = self.normalisation(np.array(sample))
 
            # raw_normalized = mne.io.RawArray(np.concatenate([normalized_data,[sample]]), raw.info)

            # raw = raw_normalized
            
            if val_preprocessing == 1 :

                raw.filter(l_freq=0.5, h_freq=45.0)
            
                # raw.info['bads'] = ['T8','TP10'] 
                # raw.interpolate_bads(reset_bads=True)

                asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff = 20)
                asr.fit(raw)
                raw = asr.transform(raw)

                # ica = ICA(n_components=20, random_state = 97)
                # ica.fit(raw)
                # ica.apply(raw)

            if val_ROI:  # Vérifie si val_ROI est True

                sfreq = raw.info['sfreq']  # fréquence d'échantillonnage
                n_channels = self.data_rsc.shape[0]  # nombre de canaux (vertices dans la ROI)
                ch_names = [f'RSC_{i}' for i in range(n_channels)]  # Noms des canaux personnalisés

                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

                # Vérifiez que self.data_rsc n'est pas vidpipe et a les dimensions correctes
                if self.data_rsc is None or self.data_rsc.shape[0] == 0:
                    raise ValueError("self.data_rsc is empty or not properly formatted")

                # Créez un nouvel objet RawArray avec les données de la ROI et les 'times' synchronisés
                raw_rsc = mne.io.RawArray(self.data_rsc, info, copy=None, verbose=None)
                print(f"Created RawArray with {len(raw_rsc.info['ch_names'])} channels and sampling frequency {sfreq}")

                # Vérifiez que raw_rsc a été correctement créé
                if raw_rsc is None or len(raw_rsc.info['ch_names']) == 0:
                    raise ValueError("raw_rsc is empty or not properly created")

                # Trouvez les événements dans raw_rsc
                events_rsc = mne.find_events(raw)
                print("Events found in raw_rsc:")
                print(events_rsc)

                # Créer des epochs pour la région d'intérêt
                epochs_rsc = mne.Epochs(
                    raw_rsc, events_rsc, self.event_id, tmin=self.tmin, tmax=self.tmax, proj=True, baseline=self.baseline, reject=None, preload=True
                )
                print(f"Created epochs with {epochs_rsc.info['nchan']} channels")

                # Vérifiez que les epochs ont été correctement créées
                if len(epochs_rsc) == 0:
                    print("Warning: All epochs were dropped!")
                    epochs_rsc.plot_drop_log()

                self.raws.append(raw_rsc)
                self.all_epochs.append(epochs_rsc)

            else :
                events =mne.pick_events(events, include=[self.event_id['true'],self.event_id['lie']]) 
                epochs = mne.Epochs(raw, events, self.event_id, self.tmin, self.tmax, proj=True, baseline=self.baseline, reject=None, preload=True)
                
                self.raws.append(raw)
                self.all_epochs.append(epochs)

        # Vérifiez le nombre de canaux pour chaque raw ajouté
        for i, raw in enumerate(self.raws):
            print(f"Raw {i} has {len(raw.info['ch_names'])} channels")

        # Vérifiez le nombre de canaux pour chaque epochs ajouté
        for i, epochs in enumerate(self.all_epochs):
            print(f"Epochs {i} has {epochs.info['nchan']} channels")

                
        self.raw_combined = mne.concatenate_raws(self.raws)
        print("nombre de raw total : ", len(self.raw_combined))
        self.epochs_combined = mne.concatenate_epochs(self.all_epochs)
        print("nombre d'epochs total : ", len(self.epochs_combined))


    def processing(self):

        fig, axs = plt.subplots(2,2, figsize=(12, 16))
        fig.suptitle(f'Preprocessing Signals', fontsize=16)

        self.display_raw_data("Before preprocessing", axs[0, 0], axs[0, 1])
        val_preprocessing = 1

        self.extract_raw_data(val_preprocessing,0)
        self.display_raw_data("After processing", axs[1, 0], axs[1, 1])
        val_preprocessing = 0 

    def time_frequency_display(self, epochs):
        freqs = np.arange(2, 36)  # fréquences de 2 à 35 Hz
        vmin, vmax = -1, 1.5  # valeurs min et max pour les ERDS dans le plot
        baseline = (0, 0.5)  # intervalle de baseline (en s)
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, centre et max pour ERDS

        tmin, tmax = -1, 4  # intervalle de temps pour l'analyse

        # Sélectionner les canaux d'intérêt
        # epochs = epochs.pick(['CP6','CP5'])
        epochs =  epochs.pick(self.channel_of_interest)
        
        # Calculer le TFR pour chaque epoch
        tfr = mne.time_frequency.tfr_multitaper(
            epochs,
            freqs=freqs,
            n_cycles=freqs,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=2,
        )
        
        # Appliquer la baseline
        tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
        
        # Calculer la moyenne des TFR
        tfr_mean = tfr.average()
        
        # Afficher la moyenne des PSD
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 1]})
        for ch, ax in enumerate(axes[:-1]):  # pour chaque canal
            tfr_mean.plot([ch], cmap="RdBu_r", cnorm=cnorm, axes=ax, colorbar=False, show=False)
            ax.set_title(epochs.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"Moyenne des PSD {epochs} pour tous les epochs")
        plt.show()

    def temporal_display (self): 
   
        # Création d'une figure
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle('Temporal Representation of epochs', fontsize=16)

        # Utilisation de gridspec pour une disposition personnalisée
        gs = gridspec.GridSpec(6, 6, height_ratios=[3, 1, 3, 3, 1, 3], width_ratios=[20, 1, 20, 1, 20, 1])


        # Définition des axes pour chaque canal
        ax_colormaps = []
        ax_colorbars = []
        ax_signals = []
        for i in range(3):
            ax_colormaps.append(fig.add_subplot(gs[0, 2 * i]))
            ax_colorbars.append(fig.add_subplot(gs[0, 2 * i + 1]))
            ax_signals.append(fig.add_subplot(gs[1, 2 * i:2 * i + 2]))
            ax_colormaps.append(fig.add_subplot(gs[3, 2 * i]))
            ax_colorbars.append(fig.add_subplot(gs[3, 2 * i + 1]))
            ax_signals.append(fig.add_subplot(gs[4, 2 * i:2 * i + 2]))

        # Parcours des canaux
        for i, channel in enumerate(self.channel_of_interest):
            # Tracé de l'image pour les epochs de "lie" pour le canal actuel
            self.epochs_lie.plot_image(
                picks=channel,
                vmin=-10,
                vmax=10,
                cmap='PuBuGn',
                axes=[ax_colormaps[i * 2], ax_signals[i * 2], ax_colorbars[i * 2]],
                colorbar=True,
                show=False,
                title=f'Lie - {channel}'
            )

            # Tracé de l'image pour les epochs de "true" pour le canal actuel
            self.epochs_true.plot_image(
                picks=channel,
                vmin=-10,
                vmax=10,
                cmap='PuBuGn',
                axes=[ax_colormaps[i * 2 + 1], ax_signals[i * 2 + 1], ax_colorbars[i * 2 + 1]],
                colorbar=True,
                show=False,
                title=f' True - {channel}'
            )

            # Ajustement de la disposition et affichage
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajuste la disposition pour la suptitle

   


    def frequency_display (self):
        freqs = np.arange(1,50,1)

        fig, axes = plt.subplots(1,2)

        #for i, channel in enumerate(self.channel_of_interest):
        self.epochs_lie.plot_psd(fmin = 4.5, fmax = 35, picks = self.channel_of_interest, average = False, ax = axes[0], color = 'red' , show=False)
        self.epochs_true.plot_psd(fmin = 4.5, fmax = 35, picks = self.channel_of_interest , average = False,  ax = axes[1], color = 'green' , show=False)

        axes[0].set_title(f'PSD - Lie ')
        axes[1].set_title(f'PSD - True ')

        plt.tight_layout()


    def ERN_display(self):
        # Définir les bandes de fréquence
        bands = {
            'alpha': (8, 12),
            'beta': (13, 30),
            'theta': (4, 7)
        }

        # Créer une figure avec trois sous-graphes pour chaque bande de fréquence
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle('Filtered EEG Epochs for Alpha, Beta, and Theta Bands')

        sfreq = self.epochs_lie.info['sfreq']
        times = np.arange(self.epochs_lie.get_data().shape[-1]) / sfreq

        # Itérer sur chaque bande de fréquence
        for i, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
            # Filtrer les données pour chaque bande
            lie_filtered = mne.filter.filter_data(
                self.epochs_lie.get_data(),  # Utiliser les données brutes des epochs
                sfreq=self.epochs_lie.info['sfreq'], 
                l_freq=low_freq, 
                h_freq=high_freq
            )
            
            true_filtered = mne.filter.filter_data(
                self.epochs_true.get_data(),  # Utiliser les données brutes des epochs
                sfreq=self.epochs_true.info['sfreq'], 
                l_freq=low_freq, 
                h_freq=high_freq
            )
            
            # Tracer les données filtrées pour chaque epoch
            for epoch in range(lie_filtered.shape[0]):
                axes[i].plot(times, lie_filtered.T, color='red', alpha=0.5)
                axes[i].plot(times, true_filtered.T, color='blue', alpha=0.5)

            axes[i].set_title(f'{band_name.capitalize()} Band ({low_freq}-{high_freq} Hz)')
            axes[i].set_xlabel('Time (samples)')
            axes[i].set_ylabel('Amplitude')

        # Afficher la légende
        handles, labels = axes[-1].get_legend_handles_labels()
        axes[-1].legend(handles, ['epochs_lie', 'epochs_true'], loc='upper right')

        plt.tight_layout()
        plt.show()

    def visualize_data(self):
        eeg_preprocessor.ERN_display()
        eeg_preprocessor.temporal_display()
        eeg_preprocessor.frequency_display()
        eeg_preprocessor.time_frequency_display(self.epochs_lie)
        eeg_preprocessor.time_frequency_display(self.epochs_true)

    def choose_ROI(self):

        source = self.epochs_combined

        subjects_dir = 'C:/Users/andresfs/mne_data/MNE-sample-data/subjects'
        subject_name = "fsaverage"

        # Référencer l'EEG
        montage = mne.channels.make_standard_montage('standard_1005')
        source.set_montage(montage)
        source.set_eeg_reference('average', projection=True)

        # Calculer la covariance de bruit
        noise_cov = mne.make_ad_hoc_cov(source.info)

        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

        # Charger le modèle BEM
        bem = mne.read_bem_solution('C:/Users/andresfs/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif')

        # Configurer l'espace source
        src = mne.setup_source_space("fsaverage", spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)

        # Calculer la solution directe (forward solution)
        fwd = mne.make_forward_solution(source.info, trans="fsaverage", src=src, bem=bem, meg=False, eeg=True, mindist=5.0, n_jobs=None, verbose=False)

        # Calculer l'opérateur inverse
        inverse_operator = mne.minimum_norm.make_inverse_operator(source.info, fwd, noise_cov, loose=0.2, depth=0.8)

        # Appliquer l'inverse
        method = "sLORETA"
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        stc = apply_inverse_epochs(source, inverse_operator, lambda2, method=method, pick_ori=None, verbose=True)

        # Charger les labels FreeSurfer
        labels = mne.read_labels_from_annot(subject=subject_name, parc='aparc.a2009s', subjects_dir=subjects_dir)

        for label in labels: 
            print("label name : ", label.name)

        # Spécifier les numéros de labels d'intérêt
        label_numbers = [
            7080,
            7828,
            3089,
            5411,
            2602,
            5534,
            3352,
            7405,
            7763,
            2750,
            4437,
            2106,
            738,
            4397,
            3187,
            948,
            3942,
            7868,
            6707,
            4751,
            4687,
            3109,
            2947,
            3901,
            7138,
            6734,
        ]

        # Filtrer pour trouver les labels avec les numéros spécifiés
        rsc_label = [label for label in labels if any(vertex in label.vertices for vertex in label_numbers)]

        if not rsc_label:
            raise ValueError("No label matching the specified numbers found in the annotation.")

        # Restreindre les données aux labels spécifiés
        vertices_rsc = np.concatenate([label.vertices for label in rsc_label])

        self.data_rsc = []

        for stc_epoch in stc:
            stc_rsc = stc_epoch.in_label(rsc_label[0])
            # Extraire les données de la région d'intérêt
            self.data_rsc.append(stc_rsc.data)

        print("size of data_rsc", np.shape(self.data_rsc))


    def normalisation(self,X) : 

        self.Xmoy = np.mean(X,axis=1, keepdims=True)
        self.Xstd = np.std(X,axis=1, keepdims= True)
        X_norm = (X - self.Xmoy)/self.Xstd

        return X_norm
    

    def display_raw_data(self, subtitle, ax_psd, ax_evoked):
        if 'lie' in self.epochs_combined.event_id:
            self.epochs_lie = self.epochs_combined['lie']
        else:
            print("No events found for 'lie'. Skipping...")
            return

        if 'true' in self.epochs_combined.event_id:
            self.epochs_true = self.epochs_combined['true']
        else:
            print("No events found for 'true'. Skipping...")
            return

        print("-------------------------------------------------------------------------------------------------------------------------------")
        print("le nombre d'epochs lie est de :", len(self.epochs_lie))
        print("le nombre d'epochs true est de :", len(self.epochs_true))
        print("-------------------------------------------------------------------------------------------------------------------------------")

        ax_psd.set_title(f'{subtitle} - PSD')
        self.epochs_lie.plot_psd(fmin=0, fmax=self.raw_combined.info['sfreq'] / 2, ax=ax_psd, show=False)
        self.epochs_true.plot_psd(fmin=0, fmax=self.raw_combined.info['sfreq'] / 2, ax=ax_psd, show=False)
        ax_psd.legend(['lie', 'true'])

        lines = ax_psd.get_lines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                line.set_color('red')
            else:
                line.set_color('blue')

        # Récupération des données pour toutes les epochs
        data_lie = self.epochs_lie.get_data()
        data_true = self.epochs_true.get_data()

        times = self.epochs_lie.times  # Les temps sont les mêmes pour toutes les epochs

        channel_index = 0  

        # for epoch_data in data_lie:
        #     ax_evoked.plot(times, epoch_data[:,channel_index], color='red', alpha=0.3)
        # for epoch_data in data_true:
        #     ax_evoked.plot(times, epoch_data[:,channel_index], color='blue', alpha=0.3)

        # Calcul de la moyenne et de l'incertitude pour les tracés remplis (facultatif)
        evoked_lie = self.epochs_lie.average()
        evoked_true = self.epochs_true.average()

        data_to_plot_lie = evoked_lie.data[channel_index]
        data_to_plot_true = evoked_true.data[channel_index]
        
        ax_evoked.plot(times, data_to_plot_lie, color='red', label='lie')
        ax_evoked.plot(times,data_to_plot_true, color='blue', label='true')

        std_lie = np.std(data_lie, axis=0)
        std_true = np.std(data_true, axis=0)

        uncertainty_lie = std_lie[channel_index]
        uncertainty_true = std_true[channel_index]

        ax_evoked.fill_between(times, data_to_plot_lie - uncertainty_lie, data_to_plot_lie + uncertainty_lie, color='red', alpha=0.3)
        ax_evoked.fill_between(times, data_to_plot_true - uncertainty_true, data_to_plot_true + uncertainty_true, color='blue', alpha=0.3)

        ax_evoked.set_title(f'{subtitle} - Evoked')
        ax_evoked.legend()

class EEGClassifier:

    def __init__(self,epochs_combined):
        self.tmin, self.tmax = 0, 1.0
        self.event_id = dict(lie=50, true=51)
        self.temporal_feature_names = None
        self.frequencial_feature_names = None
        self.epochs_combined = epochs_combined
        self.labels = None 
        self.features = None
              
    def extract_temporal_features(self, X):

        print("Temporal features extraction ...")

        fs = 256  # Fréquence d'échantillonnage
        features = []
        feature_names = []

        nb_intervals = 6
        points = np.linspace(self.tmin, self.tmax, nb_intervals + 1)
        regions = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

        for epoch in X:
            epoch_features = []
            for start_time, end_time in regions:
                start_idx = int((start_time - self.tmin) * fs)
                end_idx = int((end_time - self.tmin) * fs)
                segment = epoch[:, start_idx:end_idx]

                mean = np.mean(segment, axis=1)
                std = np.std(segment, axis=1)
                derive_segment = np.gradient(segment, 1/fs, axis=1)
                derive_seconde_segment = np.gradient(derive_segment, 1/fs, axis=1)

                mobility = np.sqrt((np.var(derive_segment, axis=1)) / np.var(segment, axis=1))
                mobility_derive = np.sqrt((np.var(derive_seconde_segment, axis=1)) / np.var(derive_segment, axis=1))
                complexity = mobility_derive / mobility

                epoch_features.append(np.concatenate([mean, std, mobility, complexity]))

                channels = segment.shape[0]
                for ch in range(channels):
                    feature_names.extend([
                        f"mean_ch{ch}_region{start_time}-{end_time}",
                        f"std_ch{ch}_region{start_time}-{end_time}",
                        f"mobility_ch{ch}_region{start_time}-{end_time}",
                        f"complexity_ch{ch}_region{start_time}-{end_time}"
                    ])

            features.append(np.concatenate(epoch_features))

        self.temporal_feature_names = feature_names
        print(" Done")
        return np.array(features)

    def extract_frequencial_features(self, X):

        print("Frequencial features extraction ...")

        fs = 256
        nb_intervals = 6
        points = np.linspace(self.tmin, self.tmax, nb_intervals + 1)
        regions = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

        """
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
        """

        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
        }

        features = []
        feature_names = []
        n_fft = min(512, len(X))

        for epoch in X:
            epoch_features = []
            for start_time, end_time in regions:
                start_idx = int((start_time - self.tmin) * fs) 
                end_idx = int((end_time - self.tmin) * fs)
                segment = epoch[:, start_idx:end_idx]

                for ch_idx, channel in enumerate(segment):
                    psd, freqs = mne.time_frequency.psd_array_welch(channel, sfreq=fs, n_fft=n_fft, n_per_seg=n_fft)

                    for band_name, band in bands.items():
                        low, high = band
                        idx_band = np.logical_and(freqs >= low, freqs <= high)
                        band_power = np.sum(psd[idx_band])
                        epoch_features.append(band_power)
                        feature_names.append(f"{band_name}_ch{ch_idx}_region{start_time}-{end_time}")

            features.append(np.array(epoch_features))

        self.frequencial_feature_names = feature_names
        print(" Done")
        return np.array(features)

    def feature_extraction(self, val_ROI):
        if val_ROI:
            X = self.data_rsc
        else:
            X = self.epochs_combined.get_data()

        print("Nombre de données pour la classification : ", len(X))

        temporal_features = self.extract_temporal_features(X)
        frequencial_features = self.extract_frequencial_features(X)

        self.temporal_features = temporal_features
        self.frequencial_features = frequencial_features
        self.features = np.hstack([self.temporal_features, self.frequencial_features])
        self.feature_names = np.hstack([self.temporal_feature_names, self.frequencial_feature_names])

        print("Nombre de features temporelles : ", temporal_features.shape)
        print("Nombre de features fréquentielles : ", frequencial_features.shape)

        self.labels = self.epochs_combined.events[:, -1] == self.event_id['true']

        print("Nombre de labels :", len(self.labels))
        print("Nombre de True :", np.sum(self.labels))
        print("Nombre de Lie :", len(self.labels) - np.sum(self.labels))

    def classification(self, features, feature_names, feature_type):
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(features, self.labels, test_size=0.4, random_state=42)
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        print(f"Classification report for {feature_type} features:")
        print(classification_report(y_test, y_pred, zero_division=0))

        feature_selector = SelectFromModel(rf, prefit=True)
        selected_indices = feature_selector.get_support(indices=True)
        
        # Correcting the selected feature names indexing
        selected_feature_names = [feature_names[i] for i in selected_indices]
        feature_importances = rf.feature_importances_[selected_indices]

        importances_df = pd.DataFrame({
            "Features": selected_feature_names,
            "Importance": feature_importances
        })

        importances_df = importances_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

        filename = f'sorted_importance_{feature_type}.txt'
        importances_df.to_csv(filename, index=False, sep="\t")

        self.display_features(importances_df,feature_type)

        print(importances_df)

    def display_features(self, importances_df, feature_type):
        # Create a Tkinter window
        root = tk.Tk()
        root.title(f"Feature Importances ({feature_type.capitalize()} Features)")
        root.state('zoomed')  # Make the window full screen

        # Create a frame for the canvas with scrollbars
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a figure and axis with Matplotlib
        fig, ax = plt.subplots(figsize=(20, len(importances_df) * 0.5))
        fig.patch.set_facecolor('white')

        # Plot the bar chart
        y_pos = range(len(importances_df))
        ax.barh(y_pos, importances_df['Importance'], color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importances_df['Features'])
        ax.invert_yaxis()  # Invert y axis to have the highest importance on top
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importances ({feature_type.capitalize()} Features)')

        # Create a canvas to display the Matplotlib figure
        figure_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
        figure_canvas.draw()
        figure_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        root.mainloop()
    
    def confusion_matrix(self): 
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
        pipeline = make_pipeline(selector, SVC())

        # Initialize lists to store results
        conf_matrices = []
        accuracies = []

        # Calculate the number of rows and columns needed for the subplots
        num_folds = skf.get_n_splits(self.features, self.labels)
        num_cols = 2
        num_rows = int(np.ceil(num_folds / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
        fig.suptitle('Confusion Matrices - Cross Validation', fontsize=16)

        for fold_idx, (train_index, test_index) in enumerate(skf.split(self.features, self.labels)):
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            conf_matrices.append(conf_matrix)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            # Plot confusion matrix
            row_idx = fold_idx // num_cols
            col_idx = fold_idx % num_cols
            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f'Fold {fold_idx + 1}')

        # Adjust space between subplots
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.5, wspace=0.3)

        # Show average accuracy across folds
        mean_accuracy = np.mean(accuracies)
        print(f'Average accuracy across folds: {mean_accuracy:.2f}')
         

class EEGProcessor:

    def __init__(self, epochs_combined, epochs_lie, epochs_true):
        self.epochs_combined = epochs_combined
        self.epochs_lie = epochs_lie
        self.epochs_true = epochs_true
        mne.set_log_level(logging.WARNING)

    def source_localization(self, state, side):
        if state == "lie": 
            info = self.epochs_lie.info 
            epochs = self.epochs_lie
        elif state == "true":
            info = self.epochs_true.info 
            epochs = self.epochs_true
        else : 
            
            info = None 
            epochs = None

        # Ajouter une référence EEG moyenne
        epochs.set_eeg_reference('average', projection=True)

        # Calculer la covariance de bruit basée sur les données des epochs
        noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=None)

        # Calculer la covariance de bruit basée sur la baseline
        baseline_cov = mne.compute_covariance(epochs, tmin=-1.5, tmax=0)

        baseline_cov.plot(info, proj=True)

        # Utiliser un modèle de tête standard fourni par MNE
        subjects_dir = 'C:/Users/andresfs/mne_data/MNE-sample-data/subjects'
        
        mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)
        bem = mne.read_bem_solution('C:/Users/andresfs/mne_data/MNE-sample-data/subjects/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif')
        
        src = mne.setup_source_space("fsaverage", spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)
        fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=bem, meg=False, eeg=True, mindist=5.0, n_jobs=None, verbose=False)
        
        # Calculer la solution inverse
        inverse_operator = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
        
        if state == "lie": 
            evoked = epochs['lie'].average() 
        elif state == "true":
            evoked = epochs['true'].average() 
        else : 
            evoked = None 
        
        method = "dSPM"
        snr = 3.0
        lambda2 = 1.0 / snr**2
        stc, residual = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None, return_residual=True, verbose=True)

        if side == "left" : 
            vertno_max_lh, time_max = stc.get_peak(hemi="lh")
            hemi = "lh"
        elif side == "right" : 
            vertno_max_rh, time_max = stc.get_peak(hemi="rh")
            hemi = "rh"
        else : 
            print("Error : No side choose")


        surfer_kwargs = dict(
            hemi=hemi,
            subjects_dir=subjects_dir,
            clim=dict(kind="value", lims=[8, 12, 15]),
            views="lateral",
            initial_time=time_max,  
            time_unit="s",
            size=(800, 800),
            smoothing_steps=10,
        )

        brain = stc.plot(**surfer_kwargs)

        if side == "left"  : 
            brain.add_foci(vertno_max_lh, coords_as_verts=True, hemi="lh", color="blue", scale_factor=0.6, alpha=0.5)

        elif side == "right" : 
            brain.add_foci(vertno_max_rh, coords_as_verts=True, hemi="rh", color="red", scale_factor=0.6, alpha=0.5)

        brain.add_text(0.1, 0.9, "dSPM (plus location of maximal activation)", "title", font_size=14)

    def calculate_coherence(self):
        # Calculer la cohérence entre les canaux EEG
        fmin = 0.5
        fmax = 40.0
        csd = csd_multitaper(self.epochs_combined, fmin=fmin, fmax=fmax, tmin=self.epochs_combined.tmin, tmax=self.epochs_combined.tmax)
        csd.plot()    


if __name__ == "__main__":


    """
    "D:/EEG_project/Dataset/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.fif",
    "D:/EEG_project/Dataset/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-002_eeg.fif",
    "D:/EEG_project/Dataset/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-003_eeg.fif",
    "D:/EEG_project/Dataset/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-004_eeg.fif",

    "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-001_eeg.fif",
    "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-002_eeg.fif",
    "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-003_eeg.fif",
    "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-004_eeg.fif",
    """


    raw_fnames = [ 
            "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-001_eeg.fif",
            # "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-002_eeg.fif",
            # "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-003_eeg.fif",
            # "D:/EEG_project/Dataset/sub-P003/ses-S001/eeg/sub-P003_ses-S001_task-Default_run-004_eeg.fif",
     
              ]

    eeg_preprocessor = EEGPreProcessor(raw_fnames)
    eeg_preprocessor.visualize_data()

    # eeg_classifier = EEGClassifier(eeg_preprocessor.epochs_combined)
    # eeg_classifier.feature_extraction(0)
    # eeg_classifier.classification(eeg_classifier.temporal_features, eeg_classifier.temporal_feature_names, 'temporal')
    # eeg_classifier.classification(eeg_classifier.frequencial_features,eeg_classifier.frequencial_feature_names,  'frequencial')
    
    # eeg_processor = EEGProcessor(eeg_preprocessor.epochs_combined, eeg_preprocessor.epochs_lie, eeg_preprocessor.epochs_true)
    # eeg_processor.calculate_coherence()


    fig, ax = plt.subplots()
    plt.show()



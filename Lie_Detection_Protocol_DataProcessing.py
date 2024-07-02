import numpy as np
import mne
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mne.preprocessing import ICA
from mne.time_frequency import tfr_multitaper, tfr_morlet, csd_multitaper
from mne.stats import permutation_cluster_test as pcluster_test
import seaborn as sns

class EEGProcessor:
    def __init__(self, raw_fnames):
        self.raw_fnames = raw_fnames
        self.event_id = dict(start=49, lie=50, true=51)
        self.tmin, self.tmax = -1.5, 0.0
        self.baseline = (None, 0)
        self.reject = dict(eeg=150e-6)
        self.raw_combined = None
        self.epochs_combined = None
        self.features = None
        self.labels = None
        self.extract_raw_data(0)

    def extract_raw_data(self,val_preprocessing):
        self.raws = []
        all_epochs = []
        for raw_fname in self.raw_fnames:
            raw = mne.io.read_raw_fif(raw_fname, preload=True)

            if val_preprocessing == 1 :
                raw.filter(l_freq=1.5, h_freq=40.0)
                
                raw.info['bads'] = ['F4', 'FC6', 'FC3']
                raw.interpolate_bads(reset_bads=True)

                ica = ICA(n_components=20, random_state = 97)
                ica.fit(raw)
                ica.apply(raw)

            events = mne.find_events(raw)
            epochs = mne.Epochs(raw, events, self.event_id, self.tmin, self.tmax, proj=True, baseline=self.baseline,
                                reject=self.reject, preload=True)
            self.raws.append(raw)
            all_epochs.append(epochs)

        self.raw_combined = mne.concatenate_raws(self.raws)
        self.epochs_combined = mne.concatenate_epochs(all_epochs)


    def extract_temporal_features(self,X):
        fs = 256  # Fréquence d'échantillonnage
        features = []

        nb_intervals = 6 

        # Définir les régions temporelles en secondes
        points = np.linspace(self.tmin,self.tmax,nb_intervals+1)
        regions = [(points[i],points[i+1]) for i in range (len(points)-1)]

        for epoch in X:  
            epoch_features = []
            
            for start_time, end_time in regions:
                # Calculer les indices de début et de fin pour chaque segment
                start_idx = int((start_time + 1.5) * fs)
                end_idx = int((end_time + 1.5) * fs)
                
                segment = epoch[:, start_idx:end_idx]
                
                # Calculer les statistiques pour le segment
                mean = np.mean(segment, axis=1)
                var = np.var(segment, axis=1)
                std = np.std(segment, axis=1)
                
                epoch_features.append(np.concatenate([mean, var, std]))

            features.append(np.concatenate(epoch_features))

        return np.array(features)

    def feature_extraction(self):
        X = self.epochs_combined.get_data()
        self.features = self.extract_temporal_features(X)
        self.labels = self.epochs_combined.events[:, -1] == self.event_id['true']

    def classification(self):

        skf = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)
        pipeline = make_pipeline(PCA(), RandomForestClassifier())

        # Initialize lists to store results
        conf_matrices = []
        accuracies = []

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
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
            row_idx = fold_idx // 3
            col_idx = fold_idx % 3
            ax = axes[row_idx, col_idx]
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False, ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title(f'Fold {fold_idx + 1}')

        # Adjust space between subplots
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.5, wspace=0.3)

        # Show average accuracy across folds
        mean_accuracy = np.mean(accuracies)
        print(f'Average accuracy across folds: {mean_accuracy:.2f}')

        plt.show()


    def processing(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 16))
        fig.suptitle(f'Preprocessing Signals', fontsize=16)

        self.display_raw_data("Before preprocessing", axs[0, 0], axs[0, 1])

        val_preprocessing = 1
        self.extract_raw_data(val_preprocessing)

        self.display_raw_data("After processing", axs[1, 0], axs[1, 1])

        plt.show()

        val_preprocessing = 0 


    def display_raw_data(self, subtitle, ax_psd, ax_evoked):
        if 'lie' in self.epochs_combined.event_id:
            epochs_lie = self.epochs_combined['lie']
        else:
            print("No events found for 'lie'. Skipping...")
            return

        if 'true' in self.epochs_combined.event_id:
            epochs_true = self.epochs_combined['true']
        else:
            print("No events found for 'true'. Skipping...")
            return

        ax_psd.set_title(f'{subtitle} - PSD')
        epochs_lie.plot_psd(fmin=0, fmax=self.raw_combined.info['sfreq'] / 2, ax=ax_psd, show=False)
        epochs_true.plot_psd(fmin=0, fmax=self.raw_combined.info['sfreq'] / 2, ax=ax_psd, show=False)
        ax_psd.legend(['lie', 'true'])

        lines = ax_psd.get_lines()
        for i, line in enumerate(lines):
            if i % 2 == 0:
                line.set_color('red')
            else:
                line.set_color('blue')

        evoked_lie = epochs_lie.average()
        evoked_true = epochs_true.average()

        times = evoked_lie.times
        ax_evoked.plot(times, evoked_lie.data.mean(axis=0), color='red', label='lie')
        ax_evoked.plot(times, evoked_true.data.mean(axis=0), color='blue', label='true')

        std_lie = evoked_lie.data.std(axis=0)
        std_true = evoked_true.data.std(axis=0)

        ax_evoked.fill_between(times, evoked_lie.data.mean(axis=0) - std_lie, evoked_lie.data.mean(axis=0) + std_lie,
                               color='red', alpha=0.3)
        ax_evoked.fill_between(times, evoked_true.data.mean(axis=0) - std_true, evoked_true.data.mean(axis=0) + std_true,
                               color='blue', alpha=0.3)

        ax_evoked.set_title(f'{subtitle} - Evoked')
        ax_evoked.legend()

    def time_frequency (self): 

        freqs = np.arange(2, 36)  # fréquences de 2 à 35Hz
        vmin, vmax = -1.0, 1.5  # valeurs min et max pour les ERDS dans le graphique
        baseline = (-1, 0)  # intervalle de baseline (en s)
        cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # normalisation des couleurs

        kwargs = dict(
            n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
        )  # paramètres pour le test de cluster de permutation

        self.epochs_combined = self.epochs_combined.pick(["C3", "C4"])

        tmin, tmax = -1.5, 0

        # Calcul du TFR
        from mne.time_frequency import tfr_multitaper
        tfr = tfr_multitaper(
            self.epochs_combined,
            freqs=freqs,
            n_cycles=freqs,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=2,
        )

        # Réduire la période de temps et appliquer la baseline
        tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")

        # Parcourir les événements pour la visualisation
        for event in self.event_id:
            tfr_ev = tfr[event]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 1]})
            fig.canvas.manager.set_window_title(f"ERDS map ({event})")

            for ch, ax in enumerate(axes[:-1]):
                # Calculer les clusters positifs et négatifs
                _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], threshold=1, tail=1, **kwargs)
                _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], threshold=-1, tail=-1, **kwargs)

                # Combinaison des clusters
                c = np.stack(c1 + c2, axis=-1) if c1 or c2 else np.zeros(tfr_ev.data.shape[:2])
                p = np.concatenate((p1, p2))
                mask = c[..., p <= 0.05].any(axis=-1)

                # Tracer le TFR avec masquage
                tfr_ev.average().plot(
                    [ch],
                    cmap="RdBu_r",
                    cnorm=cnorm,
                    axes=ax,
                    colorbar=False,
                    show=False,
                    #mask=mask,
                    mask_style="mask",
                )

                ax.set_title(self.epochs_combined.ch_names[ch], fontsize=10)
                ax.axvline(0, linewidth=1, color="black", linestyle=":")  # événement
                if ch != 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels("")

        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"ERDS ({event})")
        plt.show()

if __name__ == "__main__":

    """
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-001_eeg.fif",
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-002_eeg.fif",
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-003_eeg.fif",
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-004_eeg.fif",
        
    """

    raw_fnames = [
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-001_eeg.fif",
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-002_eeg.fif",
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-003_eeg.fif",
        "D:/EEG_project/Dataset/sub-P002/ses-S001/eeg/sub-P002_ses-S001_task-Default_run-004_eeg.fif",

        "D:/EEG_project/Dataset/sub-P006/ses-S001/eeg/sub-P006_ses-S001_task-Default_run-001_eeg.fif",
        "D:/EEG_project/Dataset/sub-P006/ses-S001/eeg/sub-P006_ses-S001_task-Default_run-002_eeg.fif",
        "D:/EEG_project/Dataset/sub-P006/ses-S001/eeg/sub-P006_ses-S001_task-Default_run-004_eeg.fif"
    
    ]

    eeg_processor = EEGProcessor(raw_fnames)
    eeg_processor.processing()
    eeg_processor.feature_extraction()
    eeg_processor.classification()
    eeg_processor.time_frequency()
    #TODO : time frequency representation 
    #TODO : source localization 
    #TODO : calculate coherence  


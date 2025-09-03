from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QShortcut
import pylsl
import mne
import numpy as np
from collections import defaultdict
from time import time

class Controller:
    MARKER_START = 1
    TIME_TEST = 5000  # duration of the test
    DELAY_SESSION = 5000  # 30 seconds delay between 2 sessions
    NUMBER_TEST = 16 # 60 tests for 7 min session
    NUMBER_SESSION = 3
    iter = 0

    def __init__(self, view, model, mrk):
        self.view = view
        self.model = model
        self.mrkstream = mrk

        self.increment_test = 0
        self.increment_session = 0
        self.counter = 0

        self.flag_delay = 0
        self.var_delay = False

        self.iter_touch_select = 0
        self.var_delay_touch_select = 0 

        self.starting(self.initialization)

        self.eeg_inlet = None
        self.raw_data = []
        self.timestamps = []
        self.events = []
        self.current_session = 0
        self.setup_eeg_recording()

        
    def setup_eeg_recording(self):
        """Connect to EEG stream"""
        print("Resolving EEG stream...")
        streams = pylsl.resolve_streams(wait_time = 5.0)
        stream = [s for s in streams if s.name() == 'Explore_8547_ExG'][0]
        self.eeg_inlet = pylsl.StreamInlet(stream)
        print(f"Connected to EEG stream: {stream.name()}")

        self.eeg_inlet = pylsl.StreamInlet(stream)
        info = stream  # pylsl.StreamInfo
        sfreq = info.nominal_srate()
        if sfreq <= 0:
            sfreq = 256
            print(f" sfreq invalide, valeur forcée à {sfreq}")

        ch_names = ['F3','FC5','T7','C3','CP5','TP9', 'P5','P6', 'PO3', 'POz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'Pz', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2', 'CP6', 'TP10', 'PO7', 'PO8']

        self.info = mne.create_info(ch_names, sfreq, 'eeg')
        print("Channels:", ch_names)
        print("Sampling rate (sfreq):", sfreq)

        # Create info structure for MNE
        #sfreq = streams[0].nominal_srate()
        #sfreq = 256  # Adjust this to your actual sampling rate
        #ch_names = ['F3','FC5','T7','C3','CP5','TP9', 'P5','P6', 'PO3', 'POz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'Pz', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2', 'CP6', 'TP10', 'PO7', 'PO8'] # Adjust to your montage
        #self.info = mne.create_info(ch_names, sfreq, 'eeg')

    def session_delay(self):
        """Save data when session ends"""
        print("Saving session data...")
        self.save_session_data()
    
    # ... rest of existing session_delay code ...

    def save_session_data(self):
        if not self.raw_data:
            return

        data = np.array(self.raw_data).T  # Channels x Samples
        print("Data shape:", data.shape)

        events = np.array(self.events)
        if events.size == 0:
            print("Aucun événement à ajouter, saut de raw.add_events()")
            return

        if events.ndim == 1:
            events = events.reshape(1, 3)

        #  Créer un seul Info combiné
        all_ch_names = self.info['ch_names'] + ['STI 014']
        all_ch_types = ['eeg'] * len(self.info['ch_names']) + ['stim']
        combined_info = mne.create_info(all_ch_names, self.info['sfreq'], ch_types=all_ch_types)

        #  Ajouter canal stim vide
        data = np.vstack([data, np.zeros((1, data.shape[1]))])

        raw = mne.io.RawArray(data, combined_info)

        print("Events shape:", events.shape)
        raw.add_events(events, stim_channel='STI 014')

        filename = f"session_{self.current_session}_{time()}.fif"
        raw.save(filename, overwrite=True)
        print(f"Saved session data to {filename}")

        self.raw_data = []
        self.timestamps = [] 
        self.events = []
        self.current_session += 1


    def starting(self, function):
        # Start windows
        self.view.show_start_window()
        self.view.button_start.clicked.connect(function)

    def initialization(self):

        self.view.close_start_window()
        self.timer_progressbar = QTimer()
        self.timer_progressbar.timeout.connect(self.progress_bar)
        self.timer_progressbar.start(int(self.TIME_TEST * 0.01))

        # Initialize the LSL marker stream
        self.model.init_mrkstream(self.mrkstream)
        self.mrkstream.push_sample([str(self.MARKER_START)])  # The first marker START signal
        print("Emitting the Marker Start on the lsl")

        self.set_signals()

    def push_marker(self, marker_type):
        """Push marker with current EEG sample timestamp"""
        marker_code = {
            'START': 1,
            'LIE': 2,
            'TRUTH': 3
        }[marker_type]
    
        if self.timestamps:
            # Use last EEG timestamp as reference
            self.events.append([len(self.raw_data), 0, marker_code])
    
        self.mrkstream.push_sample([str(marker_code)])

    def set_signals(self):
        self.model.set_images(
            self.view.get_current_image_pathname(),
            self.view.button1.pathname,
            self.view.button2.pathname
        )

        self.view.button1.clicked.connect(self.model.first_image_check)
        self.view.button2.clicked.connect(self.model.second_image_check)

        self.view.signal_key_1.connect(self.model.first_image_check)
        self.view.signal_key_2.connect(self.model.second_image_check)

    
        self.view.button1.clicked.connect(lambda: self.push_marker('LIE'))
        self.view.button2.clicked.connect(lambda: self.push_marker('TRUTH'))

        self.view.signal_key_1.connect(lambda: self.push_marker('LIE'))
        self.view.signal_key_2.connect(lambda: self.push_marker('TRUTH'))

        self.flag_touch_select = self.model.get_touch_select()


    def next_turn(self):

        self.mrkstream.push_sample([str(self.MARKER_START)])
        print("Emitting the Marker START on the LSL")

        self.view.enable_interaction()
        current_image = self.model.set_random_image()
        self.view.update_current_image(current_image)
        self.view.update_symbols_layout(current_image)
        self.model.set_images(
            self.view.get_current_image_pathname(),
            self.view.get_image1_pathname(),
            self.view.get_image2_pathname()
        )
       

    def session_delay(self):

        print("Waiting for 30s")
        self.flag_delay = 1

        self.view.session_finish()
        self.view.disable_interaction()

        self.timer_delay_session = QTimer()
        self.timer_delay_session.setSingleShot(True)
        self.timer_delay_session.timeout.connect(self.end_session_delay)
        self.timer_delay_session.start(self.DELAY_SESSION)

    def end_session_delay(self):

        self.flag_delay = 0
        self.var_delay = True
        self.view.enable_interaction()
        self.increment_test = 0
        self.increment_session += 1
        self.iter = 0
        self.view.test_finish()

    def touch_select_delay(self):
        self.iter_touch_select = 1
        self.timer_delay_touch_select = QTimer()
        self.timer_delay_touch_select.setSingleShot(True)
        self.timer_delay_touch_select.timeout.connect(self.end_touch_select_delay)
        self.timer_delay_touch_select.start(1000/2) #Wait 1 second

    def end_touch_select_delay(self):
        self.var_delay_touch_select = 1



    def progress_bar(self):

        # Collect EEG samples
        sample, timestamp = self.eeg_inlet.pull_sample(timeout=0.0)
        if sample:
            self.raw_data.append(sample)
            self.timestamps.append(timestamp)

        if self.increment_session < self.NUMBER_SESSION:

            if self.flag_delay == 0 :
                self.view.indication() 

            if self.increment_test < self.NUMBER_TEST:

                self.flag_touch_select = self.model.get_touch_select()

                if self.counter >= 100 or self.flag_touch_select==1:

                    if self.flag_touch_select and self.iter_touch_select==0:
                        self.touch_select_delay()

                    elif self.var_delay_touch_select:
                        self.var_delay_touch_select = 0
                        self.iter_touch_select = 0
                        self.flag_touch_select = 0
                        self.counter = 0
                        self.increment_test += 1

                        self.next_turn()

                self.counter += 1
                self.view.progress.setValue(self.counter)

            else:

                if self.iter == 0 and not self.var_delay:
                    self.session_delay()
                    self.iter = 1
                if self.var_delay:
                    self.var_delay = False
                    self.iter = 0

        else : 
            self.view.experiment_finish()
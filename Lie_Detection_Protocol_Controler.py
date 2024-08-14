from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QShortcut
import pylsl

class Controller:
    MARKER_START = 1
    TIME_TEST = 5000  # duration of the test
    DELAY_SESSION = 3000  # 30 seconds delay between 2 sessions
    NUMBER_TEST = 1 # 60 tests for 7 min session
    NUMBER_SESSION = 1
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
        self.mrkstream.push_sample(pylsl.vectorstr([str(self.MARKER_START)]))  # The first marker START signal
        print("Emitting the Marker Start on the lsl")

        self.set_signals()

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

        self.flag_touch_select = self.model.get_touch_select()

    def next_turn(self):

        self.mrkstream.push_sample(pylsl.vectorstr([str(self.MARKER_START)]))
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
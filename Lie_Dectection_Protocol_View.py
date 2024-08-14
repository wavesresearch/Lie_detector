from PySide2.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QFrame, QPushButton, QProgressBar, \
    QSpacerItem, QSizePolicy
from PySide2.QtGui import QPixmap, QIcon, Qt
import random
from PySide2.QtCore import Signal, QSize


class View(QMainWindow):
    """
    signal for the two keyboard keys
    """
    signal_key_1 = Signal(bool)
    signal_key_2 = Signal(bool)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lie Detection Protocol")
        self.showFullScreen()
        self.symbols_pathnames = ["src/dog.png", "src/axe.png", "src/aircraft.png", "src/battery.png", "src/brain.png", "src/phone.png",
                                  "src/fish.png", "src/sun.png", "src/tree.png", "src/wave.png"]

        self.main_layout = QVBoxLayout()
        main_widget = QWidget()
        main_widget.setLayout(self.main_layout)
        self.setCentralWidget(main_widget)

        self.title_label = QLabel("Lie Detection Protocol")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 60px; font-weight: bold; font-family: Arial")
        self.main_layout.addWidget(self.title_label)

        self.text_label = QLabel("")
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("font-size:  30px; font-weight: bold; font-family: Arial")
        self.main_layout.addWidget(self.text_label)

        image_layout = QVBoxLayout()
        self.main_layout.addLayout(image_layout)

        image_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(600, 600)

        random_image_path = random.choice(self.symbols_pathnames)
        self.image_label.pathname = random_image_path

        pixmap = QPixmap(random_image_path)
        scaled_pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        image_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        #Create the progress bar
        self.progress = QProgressBar(self)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.main_layout.addWidget(self.progress)

        #Create the 2 buttons
        self.button1 = QPushButton()
        self.button1.pathname = None
        self.button2 = QPushButton()
        self.button2.pathname = None

        #Set the 2 buttons
        self.set_symbols_layout(random_image_path)

        #Set de Start window
        self.button_start = QPushButton("Press to start the experiment")
        self.start_window = StartWindow(self.button_start)

    

    def update_current_image(self, random_image_path):
        """
         This function updates the current image every 7 seconds.
        :param random_image_path:
        :return:
        """
        self.image_label.pathname = random_image_path
        pixmap = QPixmap(random_image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def update_symbols_layout(self, random_image_path):
        """
         This function updates the 2 images and 2 pathnames every 7 seconds.
        :param random_image_path:
        :return:
        """
        button_random = random.choice(self.list_button)
        other_button = self.list_button[0] if button_random == self.list_button[1] else self.list_button[1]

        button_random.pathname = random_image_path
        pixmap1 = QPixmap(random_image_path)
        icon1 = QIcon(pixmap1)
        button_random.setIcon(icon1)

        pathname = random.choice(self.symbols_pathnames)
        #avoid having the same buttons
        while pathname == random_image_path :
            pathname = random.choice(self.symbols_pathnames)

        other_button.pathname = pathname
        pixmap2 = QPixmap(pathname)
        icon2 = QIcon(pixmap2)
        other_button.setIcon(icon2)

        self.main_layout.addWidget(self.symbols_widget)

    def get_current_image_pathname(self):
        return self.image_label.pathname

    def get_image1_pathname(self):
        return self.button1.pathname

    def get_image2_pathname(self):
        return self.button2.pathname

    def set_symbols_layout(self,image_label_pathname):
        """
        This function defines the first two buttons for launching the experiment.
        :param image_label_pathname:
        :return:
        """
        symbols_layout = QHBoxLayout()
        self.symbols_widget = QFrame()
        self.symbols_widget.setLayout(symbols_layout)

        self.symbol_buttons = []

        self.list_button = []

        # First button
        random_button1 = random.choice(self.symbols_pathnames)
        #avoid having the same buttons
        while random_button1 == image_label_pathname:
            random_button1 = random.choice(self.symbols_pathnames)


        self.button1.pathname = random_button1
        self.pixmap1 = QPixmap(random_button1)
        self.icon1 = QIcon(self.pixmap1)

        self.button1.setIcon(self.icon1)
        self.button1.setFixedSize(300, 300)
        self.button1.setIconSize(QSize(210, 210))
        self.symbol_buttons.append(self.button1)
        symbols_layout.addWidget(self.button1)

        self.list_button.append(self.button1)

        # Second button
        self.button2.pathname = image_label_pathname
        self.pixmap2 = QPixmap(image_label_pathname)
        self.icon2 = QIcon(self.pixmap2)

        self.button2.setIcon(self.icon2)
        self.button2.setFixedSize(300, 300)
        self.button2.setIconSize(QSize(210, 210))
        self.symbol_buttons.append(self.button2)
        symbols_layout.addWidget(self.button2)

        self.list_button.append(self.button2)

    def test_finish(self):
        pass
        # self.text_label.setText("the test is over")
        #self.main_layout.addWidget(self.symbols_widget)


    def session_finish(self):
        """
        This function is called when the session is over  
        """
        self.text_label.setStyleSheet("color: black; font-size: 40px; font-weight: bold; font-family: Arial;")
        self.text_label.setText("The session is over, please wait 30 seconds")
        self.main_layout.addWidget(self.symbols_widget)

    def experiment_finish(self):
        """
        This function is called when the experiment is over and close the main window 
        """
        self.close()

    def disable_interaction(self):
        self.button1.setEnabled(False)
        self.button2.setEnabled(False)

    def enable_interaction(self):
        self.button1.setEnabled(True)
        self.button2.setEnabled(True)

    def indication(self):
        """
        This function tell instruction about the session you realised 
        """
        self.text_label.setStyleSheet("color: green; font-size: 40px; font-weight: bold; font-family: Arial;")
        self.text_label.setText("select the right answer")
        self.main_layout.addWidget(self.symbols_widget)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.signal_key_1.emit(True)
            self.disable_interaction()

        elif event.key() == Qt.Key_Enter:
            self.signal_key_2.emit(False)
            self.disable_interaction()

        else:
            super().keyPressEvent(event)

    
    def create_start_window(self,button_start):
        """Create and return an instance of the second window."""
        start_window = StartWindow(button_start)
        return start_window

    def show_start_window(self):
        """Show the second window."""
        self.start_window = self.create_start_window(self.button_start)
        self.start_window.show()
    
    def close_start_window(self):
        self.start_window.close()



class StartWindow(QWidget):
        def __init__(self,button_start):
            super().__init__()
            
            self.setWindowTitle("Start Window")
            self.setGeometry(150, 150, 400, 300)
            
            layout = QVBoxLayout()
            layout.addWidget(button_start)
            
            self.setLayout(layout)

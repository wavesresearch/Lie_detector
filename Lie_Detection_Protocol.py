import sys
from Lie_Dectection_Protocol_View import View
from Lie_Detection_Protocol_Model import Model
from Lie_Detection_Protocol_Controler import Controller
import pylsl
import matplotlib
matplotlib.use('Qt5Agg')
from PySide2.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    
    # 1. Initialize LSL Marker Stream
    print("Creating marker stream...")
    marker_info = pylsl.StreamInfo(
        name='Lie_Task_Markers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format=pylsl.cf_string,
        source_id='lie_exp_123'
    )
    marker_outlet = pylsl.StreamOutlet(marker_info)
    
    # 2. Set up MVC components
    view = View()
    model = Model()
    controller = Controller(view=view, model=model, mrk=marker_outlet)
    
    # 3. Connect cleanup handler
    def on_exit():
        if hasattr(controller, 'save_session_data'):
            controller.save_session_data()  # Save any remaining data
        print("Experiment ended. Data saved.")
    
    app.aboutToQuit.connect(on_exit)
    
    # 4. Start experiment
    view.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


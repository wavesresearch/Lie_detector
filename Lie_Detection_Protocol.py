import sys
from Lie_Dectection_Protocol_View import View
from Lie_Detection_Protocol_Model import Model
from Lie_Detection_Protocol_Controler import Controller
import pylsl



from PySide2.QtWidgets import QApplication

app = QApplication(sys.argv)

info = pylsl.stream_info('Lie_Task_Markers','Markers',1,0,pylsl.cf_string,'unsampledStream')
mrkstream = pylsl.stream_outlet(info,1,1)


view = View()
model = Model()
controller = Controller(view=view,model=model,mrk=mrkstream)

view.show()

sys.exit(app.exec_())


import os
import random
from PySide2.QtCore import QObject, Signal, Slot, Qt
from PySide2.QtWidgets import QWidget
import pylsl


class Model(QObject):

    MARKER_LIE = 2
    MARKER_TRUTH = 3

    def __init__(self) -> None:
        super().__init__()
        images_dir = "C:/Users/user/Documents/BrainKybLab/Lab Protocols/Lie_detector/Lie_detector_Quentin/color_image_bank" 
       
        self.symbols_pathnames = [
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.endswith(".png") and not f.startswith("._") ]
        
        self.flag_touch_select = 0 
        
        
    def init_mrkstream(self,mrk):
        self.mrkstream = mrk 
        
    def set_images(self,current_image_pathname,image1_pathname,image2_pathname):
        self.current_image = current_image_pathname
        self.image1 = image1_pathname
        self.image2= image2_pathname

    def first_image_check(self):
        self.flag_touch_select = 1
        if self.current_image == self.image1 :
            #print("YOU SAY THE TRUTH")
            self.mrkstream.push_sample([str(self.MARKER_LIE)])
            print("Emitting the Marker TRUE on the lsl")


        elif self.current_image != self.image1 :
            #print("YOU LIE")
            self.mrkstream.push_sample([str(self.MARKER_LIE)])
            print("Emitting the Marker LIE on the lsl")



    def second_image_check(self,mrkstream):
        self.flag_touch_select = 1
        if self.current_image == self.image2 :
            #print("YOU SAY THE TRUTH")
            self.mrkstream.push_sample([str(self.MARKER_LIE)])
            print("Emitting the Marker TRUE on the lsl")


        elif self.current_image != self.image2 : 
            #print("YOU LIE")
            self.mrkstream.push_sample([str(self.MARKER_LIE)])
            print("Emitting the Marker LIE on the lsl")



    def set_random_image(self):
        self.flag_touch_select = 0
        random_image_path = random.choice(self.symbols_pathnames)
        self.current_image = random_image_path
        return random_image_path  # = current_image
    
    def get_touch_select(self):
        return self.flag_touch_select





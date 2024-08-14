import pyxdf
import os
import mne
import os.path as op
import numpy as np

from mne.datasets import sample
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree, make_report,
                      find_matching_paths, get_entity_vals)



bids_root = 'D:/EEG_project/Dataset'
# Print the tree of files
print_dir_tree(bids_root, max_depth=None, return_str=False)


# Extract the .xdf files list
sessions = 'S001'
datatype = 'eeg'
extensions = [".xdf"]  # ignore other files
subjects = ['P007']
runs = ['001','002','003','004']
bids_paths = find_matching_paths(bids_root, datatypes=datatype,
                                 sessions=sessions, extensions=extensions, subjects=subjects, runs=runs)


print(bids_paths)


stream_markers = 'Lie_Task_Markers_' ########## To modify ########## To modify

stream_data = 'Explore_8547_ExG' ########## To modify ########## To modify
no_channels = 32

# Reading .xdf files, convert to .fif and save events file
for fname in bids_paths:
    # Reading .xdf file
    print("The filename is : ",fname)
    streams, header = pyxdf.load_xdf(fname)
    for stream in (streams):
        stream_name = stream['info']['name']
        print(stream_name)
 
        # Reading Markers    
        if stream_name[0] == stream_markers:
            # '0' marker is standard and '1' marker is oddball/target
            print('Processing Markers')
            markers = stream['time_series']
            time_stamps = stream['time_stamps']
            
        # Reading EEG
        elif stream_name[0] == stream_data:
            print('Processing Data')
            eeg_data = stream["time_series"].T
            eeg_data *= 1e-6 # uV -> V
            time_stamps_eeg = stream['time_stamps']
            sfreq = float(stream["info"]["nominal_srate"][0])
            eeg_data = eeg_data[0:no_channels,:]
            eeg_size = eeg_data.shape[1]

    # Finding the position of the markers
    pos = []
    for i in time_stamps:
        norm = np.zeros(time_stamps_eeg.shape[0])
        ind = 0
        for j in time_stamps_eeg:
            norm[ind] = abs (i-j)
            ind = ind + 1
        val = np.where(norm == np.amin(norm))
        pos.append(val[0][0])
    pos_markers = np.array(pos)

    # Creating the new markers
    markers_data = np.zeros(time_stamps_eeg.shape[0])
    for i in range(0,pos_markers.shape[0]):
        markers_data[pos_markers[i]] = ord(markers[i][0])
    markers_data = np.reshape(markers_data, (1, markers_data.shape[0]))

    # Creating Raw data
    data = np.concatenate((eeg_data, markers_data), axis=0)
    info = mne.create_info(data.shape[0], sfreq, ["eeg"]*(data.shape[0]-1) + ["stim"])
    raw = mne.io.RawArray(data, info)
    raw.plot(scalings=dict(eeg=100e-6),duration=10,highpass=0.5, lowpass=45)

    #Renaming channels
    #mapping = {'0':'FC2','1':'FC4','2':'FC6','3':'FCz','4':'C6','5':'C4','6':'C2','7':'TP8', '8':'CP2','9':'CP4','10':'FT7','11':'FT8','12':'FC5','13':'C5','14':'FC1','15':'C3', '16':'Cz', '17':'C1', '18':'CP1', '19':'CPz', '20':'CP3', '21':'CP5', '22':'F8', '23':'AF4', '24':'F4', '25':'AFz', '26':'AF3', '27':'F7', '28':'F3', '29':'FC3', '30':'CP6', '31':'TP7', '32':'stim'} ########## To modify ########## To modify
    #ch_names=['F3','FC5','T7','C3','CP5','TP9', 'P5','P6', 'PO3', 'POz', 'AFz', 'Fz', 'Cz', 'FC1', 'FC2', 'P1', 'Pz', 'P2', 'PO4', 'O1', 'Oz', 'O2', 'F4', 'FC6', 'T8', 'C4', 'CP1', 'CP2', 'CP6', 'TP10', 'PO7', 'PO8','stim']
    mapping = {'0': 'F3', '1': 'FC5', '2': 'T7', '3': 'C3', '4': 'CP5', '5': 'TP9', '6': 'P5', '7': 'P6', '8': 'PO3', '9': 'POz', '10': 'AFz', '11': 'Fz', '12': 'Cz', '13': 'FC1', '14': 'FC2', '15': 'P1', '16': 'Pz', '17': 'P2', '18': 'PO4', '19': 'O1', '20': 'Oz', '21': 'O2', '22': 'F4', '23': 'FC6', '24': 'T8', '25': 'C4', '26': 'CP1', '27': 'CP2', '28': 'CP6', '29': 'TP10', '30': 'PO7', '31': 'PO8', '32': 'stim'}
    print(mapping)
    mne.rename_channels(raw.info, mapping)

    #Reading events
    events = mne.find_events(raw, stim_channel='stim', verbose=True)
    if events[-1,-1] != 57: # For some unkwnown reason a wrong marker is added add the end
        events = events[0:events.shape[0]-1]

    event_id = {'Start': 49, 'Lie': 50, 'True' : 51} ########## Lie Detector 

    mne.viz.plot_events(events, raw.info['sfreq'], event_id=event_id)

    #Including montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    #Save file
    new_fname = str(fname)
    new_fname = new_fname.replace('.xdf', '.fif')
    raw.save(new_fname,overwrite=True)

    #Save events
    new_fname = new_fname.replace('_eeg', '_eve')
    mne.write_events(new_fname, events, overwrite=True)


# Print the tree of files
print_dir_tree(bids_root, max_depth=None, return_str=False)


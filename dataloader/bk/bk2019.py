"""
Motor Imagery data recorded at Ho Chi Minh University of Technology (VNU-HCM)


Nguyen, M. T. D., Pham, C. Q., Nguyen, H. N., Le, K. Q., & Huynh, L. Q. (2022). 
A Statistical Approach to Evaluate Beta Response in Motor Imagery-Based Brain-Computer Interface. 
In 8th International Conference on the Development of Biomedical Engineering in Vietnam: 
Proceedings of BME 8, 2020, Vietnam: Healthcare Technology for Smart City in Low-and Middle-Income Countries 
(pp. 203-217). Springer International Publishing.

======================
Authors: Cuong Pham
cuongquocpham151@gmail.com

"""
import os
import numpy as np
import pandas as pd
import mne
from mne.channels import make_standard_montage
from moabb.datasets.base import BaseDataset


#=========================#
## CONFIG
ROOT = "/home/pham/bci/DATASET/BACHKHOA"
FS = 500
LIST_SUBJECTS = list(range(1,13))
ALL_EVENTS = dict(right_hand=2, left_hand=1, feet=3)
EEG_CH_NAMES = ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']


#=========================#
def structurize_folder():
    """
    Return dictionary file of path each subject session
        d = {
            1: {
                0: ../[01]DucMinh/10_08
            }
        }
    """
    def sortDates(datesList):
        split_up = datesList.split('_')
        return split_up[0], split_up[1]

    d = {}
    for fol_sb in os.listdir(ROOT):
        list_ss = [i for i in os.listdir(ROOT+"/"+fol_sb) if i!="all"]
        list_ss_sort = sorted(list_ss, key=sortDates)

        d[int(fol_sb[1:3])] = {
            str(i): os.path.join(ROOT, fol_sb, v) \
                for i,v in enumerate(list_ss_sort)
        }
    return d


#=========================#
def extract_session(path_session:str = ""):
    """ extract eeg and events for each session """

    list_runs = []
    for root, dirs, files in os.walk(path_session):
        for file in files:
            if "I_event" in file:
                prefix = file.split("_event")[0] # BCI_Minh_023I
                list_runs.append(prefix)
    list_runs.sort(reverse=False)
    # print(list_runs)

    d_eeg = {}
    d_stim = {}

    for run, fn in enumerate(list_runs):

        ## check delay
        try:
            file_delay = os.path.join(path_session, "Files", f"{fn}_event.txt")
            with open(file_delay, 'r') as fid:
                txt = fid.readlines()
            mins = txt[22][18:20]
            secs = txt[22][21:23]
            delay = int(mins) * 60 + int(secs)
            # print(mins, secs, delay)
        except:
            print(f"[ERROR] file_delay | {file_delay}")
            continue

        ## load file
        try:
            file_data = os.path.join(path_session, "Files", f"{fn}.txt")
            s = np.loadtxt(file_data, skiprows=1) # (N, 7)
            eeg = s[FS*delay:, :6] # exclude ECG
            # print(eeg.shape)
        except:
            print(f"[ERROR] file_data | {file_data}")
            continue

        ## labels & create events
        try:
            file_trigger = os.path.join(path_session, "trigger", f"{fn}_trigger.csv")
            check = pd.read_csv(file_trigger, header=None).to_numpy()

            stim = [0]*eeg.shape[0]
            for j in range(check.shape[0]):
                if check.shape[1] > 3 and check[j,3] == 13:
                    continue
                else:
                    mi_start = int(check[j, 1] * FS)
                    lb = check[j, 0]
                    stim[mi_start] = lb
        except:
            print(f"[ERROR] events | {file_trigger}")
            continue

        d_eeg[str(run)] = eeg
        d_stim[str(run)] = stim

    
    return d_eeg, d_stim
        


#=========================#
class Bk2019_moabb(BaseDataset):
    """Motor Imagery dataset
    """

    def __init__(self):
        super().__init__(
            subjects=LIST_SUBJECTS,
            sessions_per_subject=5,
            events=ALL_EVENTS,
            code="bk2019",
            interval=[0, 6],
            paradigm="imagery",
            doi="",
        )
        
        self.sessions = 0
        self.runs = -1

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
    
        # fmt: off
        ch_types = ["eeg"]*6 + ["stim"]
        ch_names = EEG_CH_NAMES + ["Stim"]
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=FS)
        montage = make_standard_montage("standard_1020")

        ## read raw
        d_ss_path = self.data_path(subject)
        # print(d_ss_path)

        ## check chosen session
        if self.sessions != -1: 
            list_sessions = [str(self.sessions)]
        else:
            list_sessions = d_ss_path.keys()

        ## check chosen run
        if self.runs != -1: 
            list_runs = [str(self.runs)]
        else:
            list_runs = d_stim.keys()


        sessions = {}
        for session_idx in list_sessions:
            d_eeg, d_stim = extract_session(d_ss_path[session_idx])
  
            sessions[session_idx] = {}
            for run in list_runs:
                eeg = d_eeg[run].T # (6, N)
                stim = np.array(d_stim[run]).reshape(1,-1) # (1, N)
                data = np.vstack([eeg, stim]) # (7,N)

                # a,b=np.unique(stim, return_counts=True)
                # print([(i,v) for i,v in zip(a, b)])

                raw = mne.io.RawArray(data=data, info=info, verbose=False)
                raw.set_montage(montage)

                sessions[session_idx][run] = raw
        
        # a = sessions["0"]["0"]
        # z = a.get_data(picks=["C3"], units='uV')
        # print(z.shape)
        # event = 37233
        # print(z[0,event:event+6*500])

        return sessions


    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        d = structurize_folder()
        return d[subject]





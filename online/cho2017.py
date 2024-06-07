"""
Motor Imagery dataset from Cho et al 2017.

======================
Authors: Cuong Pham
cuongquocpham151@gmail.com

"""
import os
import numpy as np
from scipy.io import loadmat
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from moabb.datasets.base import BaseDataset



#=========================#
## CONFIG
ROOT = "/home/pham/bci/DATASET/CHO2017"
FS = 512
LIST_SUBJECTS = list(range(1,52))
ALL_EVENTS = dict(left_hand=1, right_hand=2)
EEG_CH_NAMES = [
    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
]

#=========================#
class Cho2017_moabb(BaseDataset):
    """
    >> replace moabb.datasets.Cho2017
    Modified from 
    https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/gigadb.py

    Download
    http://gigadb.org/dataset/view/id/100295


    Motor Imagery dataset from Cho et al 2017.
    =======  =======  =======  ==========  =================  ============  ===============  ===========
    Name       #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
    =======  =======  =======  ==========  =================  ============  ===============  ===========
    Cho2017       52       64           2                100  3s            512Hz                      1
    =======  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset Description**
    We conducted a BCI experiment for motor imagery movement (MI movement)
    of the left and right hands with 52 subjects (19 females, mean age ± SD
    age = 24.8 ± 3.86 years); Each subject took part in the same experiment,
    and subject ID was denoted and indexed as s1, s2, …, s52.
    Subjects s20 and s33 were both-handed, and the other 50 subjects
    were right-handed.
    EEG data were collected using 64 Ag/AgCl active electrodes.
    A 64-channel montage based on the international 10-10 system was used to
    record the EEG signals with 512 Hz sampling rates.
    The EEG device used in this experiment was the Biosemi ActiveTwo system.
    The BCI2000 system 3.0.2 was used to collect EEG data and present
    instructions (left hand or right hand MI). Furthermore, we recorded
    EMG as well as EEG simultaneously with the same system and sampling rate
    to check actual hand movements. Two EMG electrodes were attached to the
    flexor digitorum profundus and extensor digitorum on each arm.

    """

    def __init__(self):
        super().__init__(
            subjects=LIST_SUBJECTS,
            sessions_per_subject=1,
            events=ALL_EVENTS,
            code="Cho2017",
            interval=[0, 3],  # full trial is 0-3s, but edge effects
            paradigm="imagery",
            doi="10.5524/100295",
        )
    
    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""

        fname = self.data_path(subject)
        print(fname)

        data = loadmat(
            fname,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )["eeg"]



        # fmt: on
        emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
        ch_names = EEG_CH_NAMES + emg_ch_names + ["Stim"]
        ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]
        montage = make_standard_montage("standard_1005")
        imagery_left = data.imagery_left - data.imagery_left.mean(
            axis=1, keepdims=True)
        imagery_right = data.imagery_right - data.imagery_right.mean(
            axis=1, keepdims=True
        )

        eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
        eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])

        # trials are already non continuous. edge artifact can appears but
        # are likely to be present during rest / inter-trial activity
        eeg_data = np.hstack(
            [eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r]
        )
        # log.warning(
        #     "Trials demeaned and stacked with zero buffer to create "
        #     "continuous data -- edge effects present"
        # )


        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=data.srate)
        raw = RawArray(data=eeg_data, info=info, verbose=False)
        raw.set_montage(montage)

        print(raw)
        
        # ## CUONG
        # raw1 = raw.copy()
        # raw1 = raw1.pick_channels(EEG_FLEX)
        # raw1 = raw1.resample(sfreq=128)
        # data = raw1.get_data()
        # print(data.shape)
        # return raw1
        return {"0": {"0": raw}}



    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """
        Modified the data_path function, otherwise it will automatically download files again
        """
        if subject < 10: 
            fname = os.path.join(ROOT, f"s0{subject}.mat")
        else:
            fname = os.path.join(ROOT, f"s{subject}.mat")
        return fname



# #=========================#
# class Cho2017_torcheeg():
#     """

#     """
#     def __init__(self):

#         self.fs_resample = 512 # 512->128
#         self.n_chans = n_chans
#         self.n_times = int(window*self.fs_resample)

#         #
#         data = Cho2017_moabb()
#         paradigm = LeftRightImagery()
#         x, y, metadata = paradigm.get_data(dataset=data, subjects=[1])
#         print(x.shape)

#         self.dataset = moabb_dataset.MOABBDataset(dataset=data,
#                                                 paradigm=paradigm,
#                                                 # io_path="./io/moabb",
#                                                 # offline_transform=transforms.Compose([
#                                                 #     transforms.BandDifferentialEntropy()
#                                                 # ]),
#                                                 online_transform=transforms.Compose([
#                                                     # transforms.To2d(),
#                                                     transforms.ToTensor(),
#                                                 ]),
#                                                 label_transform=transforms.Compose([
#                                                     transforms.Select('label')
#                                                 ]),
#                                                 chunk_size=self.n_times,
#                                                 num_worker=4,
#         )
#         # self.cv = KFoldCrossSubject(n_splits=5, 
#         #                             shuffle=True, 
#         #                             random_state=SEEDS,
#         #                             split_path=".torcheeg/split_bciiv2a_5foldcross")


#     def get_dataset(self, fold=0):

#         for i, (train_dataset, test_dataset) in enumerate(self.cv.split(self.dataset)):
#             if i != fold:
#                 continue

#             print("\n====Data loading--> (len) train_dataset: {}".format(len(train_dataset)))
#             print("\n====Data loading--> (len) test_dataset: {}".format(len(test_dataset)))
#             for _ in train_dataset:
#                 x, y = iter(_)
#                 print(f"[train_dataset] idx 0 | x.shape: {x.shape}") # 2s => (1,22,500)
#                 print("y: ", y)
#                 break
        
#         return train_dataset, test_dataset
        


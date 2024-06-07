"""
BNCI 2014-001 Motor Imagery dataset.

======================
Authors: Cuong Pham
cuongquocpham151@gmail.com

"""

from moabb.datasets.base import BaseDataset
from moabb.datasets.bnci import _convert_mi
# from torcheeg.datasets import BCICIV2aDataset
# from torcheeg import transforms
# from torcheeg.model_selection import KFoldCrossSubject


#=========================#
## CONFIG
ROOT = "/home/pham/bci/DATASET/BCICIV_2a_mat"
LIST_SUBJECTS = list(range(1, 10))
ALL_EVENTS = {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4}
EEG_CH_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    "EOG1", "EOG2", "EOG3",
]



#=========================#
class BCIIV2a_moabb(BaseDataset):
    """
    >> replace moabb.datasets.BNCI2014_001
    Modified from
    https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci.py


    BNCI 2014-001 Motor Imagery dataset.

    ============  =======  =======  ==========  =================  ============  ===============  ===========
    Name           #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
    ============  =======  =======  ==========  =================  ============  ===============  ===========
    BNCI2014_001       9       22           4                144  4s            250Hz                      2
    ============  =======  =======  ==========  =================  ============  ===============  ===========
    Dataset IIa from BCI Competition 4 [1]_.

    **Dataset Description**
    This data set consists of EEG data from 9 subjects.  The cue-based BCI
    paradigm consisted of four different motor imagery tasks, namely the imag-
    ination of movement of the left hand (class 1), right hand (class 2), both
    feet (class 3), and tongue (class 4).  Two sessions on different days were
    recorded for each subject.  Each session is comprised of 6 runs separated
    by short breaks.  One run consists of 48 trials (12 for each of the four
    possible classes), yielding a total of 288 trials per session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    """

    def __init__(self):
        super().__init__(
            subjects=LIST_SUBJECTS,
            sessions_per_subject=2,
            events=ALL_EVENTS,
            code="BNCI2014-001",
            interval=[2, 6], # events start at 2s
            paradigm="imagery",
            doi="10.3389/fnins.2012.00055",
        )
    
    def _get_single_subject_data(self, subject):
        """
        Return data for a single subject.
        Load data for 001-2014 dataset.
        (Each session has 72-trial x 4-class)
        """

        base_url = "/home/pham/bci/DATASET/BCICIV_2a_mat"
        _map = {"T": "train", "E": "test"}

        # fmt: on
        ch_types = ["eeg"] * 22 + ["eog"] * 3

        sessions = {}
        filenames = []
        # for session_idx, r in enumerate(["T", "E"]):
        for session_idx, r in enumerate(["T"]):

            filename = "{u}/A{s:02d}{r}.mat".format(u=ROOT, s=subject, r=r)
            
            runs, ev = _convert_mi(filename, EEG_CH_NAMES, ch_types)

            sessions[f"{session_idx}{_map[r]}"] = {
                str(ii): run for ii, run in enumerate(runs)
            }
            
        # a = sessions["0train"]["0"]
        # z = a.get_data(picks=["Fz"], units='uV')
        # print(z.shape)
        # event = 4171
        # # print(z[:, event:event+4*250])

        return sessions
    
    def data_path(self):
        pass





#=========================#
# class BCIIV2a_torcheeg():
#     """ torcheeg pipeline for BCIIV2A dataset """

#     def __init__(
#         self, 
#         root = "/home/pham/bci/DATASET/BCICIV_2a_mat",
#         fs = 250,
#         window = 2.0,
#         overlap = 0.0,
#         n_chans = 22,
#         n_classes = 4,
#         ):

#         self.n_chans = n_chans
#         self.n_times = int(window*fs)
#         self.dataset = BCICIV2aDataset(root_path=root,
#                                         io_path=".torcheeg/datasets_bciiv2a_2s",
#                                         online_transform=transforms.Compose([
#                                             transforms.To2d(),
#                                             transforms.ToTensor()
#                                         ]),
#                                         label_transform=transforms.Compose([
#                                             transforms.Select('label'),
#                                             transforms.Lambda(lambda x: x - 1)
#                                         ]),
#                                         chunk_size=self.n_times,
#                                         overlap=overlap,
#                                         num_worker=4,
#                                         )

#         self.cv = KFoldCrossSubject(n_splits=5, 
#                                     shuffle=True, 
#                                     random_state=42,
#                                     split_path=".torcheeg/split_bciiv2a_5foldcross")


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
        


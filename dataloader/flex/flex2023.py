"""
Motor Imagery dataset conducted at International University (VNU-HCM)

=================
Author: Cuong Pham
cuongquocpham151@gmail.com

"""
import os
import numpy as np
import pandas as pd
import mne
from moabb.datasets.base import BaseDataset
from config import *

################################
class Flex2023_moabb(BaseDataset):
    """
    Motor Imagery moabb dataset
    Args:
        protocol (str): Protocol name. Defaults to "8c".
        session (str): session name. Defaults to "ss1".
        run (str): run name. Defaults to "run1".
    
    """
    def __init__(
        self, 
        dir_raw_data:str = "",
        protocol:str= "8c", 
        session:str= "ss1", 
        run:str= "run1"
    ):

        if "4c" in protocol:
            events = EVENT_IDX_4CLASS
        elif "8c" in protocol:
            events = EVENT_IDX_8CLASS

        super().__init__(
            subjects=LIST_SUBJECTS,
            sessions_per_subject=1,
            events= events,
            code="Flex2023",
            interval=[4, 8], # events at 4s
            paradigm="imagery",
            doi="",
        )
        self.dir_raw_data = dir_raw_data
        self.protocol = protocol
        self.session = session
        self.run = run

        print(self.dir_raw_data)

    def _flow(self, raw0, stim):
        """Single flow of raw processing"""

        ## get eeg (32,N)
        data = raw0.get_data(picks=EEG_CH_NAMES)

        # stack eeg (32,N) with stim (1,N) => (32, N)
        data = np.vstack([data, stim.reshape(1,-1)])

        ch_types = ["eeg"]*32 + ["stim"]
        ch_names = EEG_CH_NAMES + ["Stim"]
        info = mne.create_info(ch_names=ch_names, 
                                ch_types=ch_types, 
                                sfreq=FS)
        raw = mne.io.RawArray(data=data,
                              info=info, 
                              verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        # print(raw0.info)
        # print(raw.info)

        raw.filter(l_freq=1.0, h_freq=None, method='iir') \
            .notch_filter(freqs=[50])
        #     .set_eeg_reference(ref_channels='average')
        return raw


    def _get_stim_data(self, edf_raw, subject):
        assert int(subject) >= 12
        return edf_raw.get_data(picks=["MarkerValueInt"], units='uV')[0]


    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""

        # path
        list_edf = self.data_path(subject)

        if (subject in [10,11] and self.protocol=="4c*") or (self.run == "-1"):  
            list_edf_select = list_edf
        else:
            list_edf_select = [p for p in list_edf if self.run in p]

        # concat runs 
        list_raw = []
        for _edf in list_edf_select:
            raw0 = mne.io.read_raw_edf(_edf, preload=False)
            stim = self._get_stim_data(raw0, subject)
            raw_run = self._flow(raw0, stim)
            list_raw.append(raw_run)
        raw = mne.concatenate_raws(list_raw)

        return {"0": {"0": raw}}


    def data_path(self, subject, **kwargs) -> None:
        """Return list of path of edf files for predefined protocols"""

        list_edf = []
        subkey = f"F{subject}_{self.protocol}_{self.session}"
        for root, dirs, files in os.walk(self.dir_raw_data):
            for file in files:
                if file.endswith(".edf") and \
                    (subkey in file) and (".md" not in file):
                        list_edf.append(os.path.join(root, file))
        if list_edf:
            return list_edf
        else:
            raise FileNotFoundError(f"Can not find filename with <{subkey}>")

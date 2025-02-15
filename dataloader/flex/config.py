################################
## CONFIG
# ROOT = "/home/pham/bci/DATASET/FLEX"
LIST_SUBJECTS =  list(range(1, 100))
EEG_CH_NAMES = [
    'Cz', 'Fz', 'Fp1', 'F7', 'F3', 
    'FC1', 'C3', 'FC5', 'FT9', 'T7', 
    'CP5', 'CP1', 'P3', 'P7', 'PO9', 
    'O1', 'Pz', 'Oz', 'O2', 'PO10', 
    'P8', 'P4', 'CP2', 'CP6', 'T8', 
    'FT10', 'FC6', 'C4', 'FC2', 'F4', 
    'F8', 'Fp2'
]
FS = 128

EVENT_IDX_4CLASS = dict(
    right_hand=1, 
    left_hand=2,
    right_foot=3, 
    left_foot=4
)

EVENT_IDX_8CLASS = dict(
    right_hand=1, 
    left_hand=2, 
    right_foot=3, 
    left_foot=4,
    right_hand_r=5, 
    left_hand_r=6, 
    right_foot_r=7, 
    left_foot_r=8,
)
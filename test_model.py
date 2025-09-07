from train_fusion_model import Test_UnifiedFusionModel
from Config import TEST_TRAIN_SPLIT_ANNOTATION_PATH
from Config import *

seq_path=r"C:\Users\billy\Desktop\VIP\Organoid-Analyzer\Generated\CARTtrajectory_dataset_100.npz"
track_path=r"C:\Users\billy\Desktop\VIP\Organoid-Analyzer\Generated\CARTtrajectory_dataset_100.npz"
model_path = r"C:\Users\billy\Desktop\VIP\Organoid-Analyzer\Results\ablation_Specify\graphs\hidden_32\fusion_64\train results\hidden32_fusion64.pth"

Test_UnifiedFusionModel(seq_path, track_path, model_path, TEST_TRAIN_SPLIT_ANNOTATION_PATH, results_dir="test", 
                        seq_input_size=FEATURE_LEN, track_input_size=TRACK_LEN, hidden_size=32, fusion_size=64, dropout=0.3)
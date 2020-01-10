import json    
import numpy as np
import os
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        elif isinstance(obj,(list,)):
            if isinstance(obj[0], (np.float_, np.float16, np.float32, np.float64)):
                return [float(elem) for elem in obj]
            elif isinstance(obj[0], (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
                return [int(elem) for elem in obj]
        return json.JSONEncoder.default(self, obj)


class DataCleaner ():
    def __init__(self, _archive_folder):
        self.archive_folder = _archive_folder
    
    def archive_sample(self, retraining_text_seq_path, retraining_tags_seq_path):
        text_sequence_filename = os.path.basename(retraining_text_seq_path)
        tags_sequence_filename = os.path.basename(retraining_tags_seq_path)
        archive_text_seq_file = os.path.join(self.archive_folder, text_sequence_filename)
        archive_tags_seq_file = os.path.join(self.archive_folder, tags_sequence_filename)
        
        hot_file_text = open(retraining_text_seq_path, "r", encoding="utf-8")
        hot_file_tags = open(retraining_tags_seq_path, "r")
        sample_text_seqs = hot_file_text.readlines()
        sample_tags_seqs = hot_file_tags.readlines()

        with open(archive_text_seq_file, "r", encoding="utf-8") as f:
            archive_text_seqs = f.readlines()
        
        archive_file_text = open(archive_text_seq_file, "a+", encoding="utf-8")
        archive_file_tags = open(archive_tags_seq_file, "a+")
        

        for text_seq, tags_seq in zip(sample_text_seqs, sample_tags_seqs):
            if text_seq not in archive_text_seqs:
                archive_file_text.write(text_seq)
                archive_file_tags.write(tags_seq)
        
        hot_file_text.close()
        hot_file_tags.close()
        archive_file_text.close()
        archive_file_tags.close()

        open(retraining_text_seq_path, "w", encoding="utf-8").close()
        open(retraining_tags_seq_path, "w").close()



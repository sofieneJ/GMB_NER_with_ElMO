import json
import python-Levenshtein

def find_tagged_seq (seq, tagged_seq):
    ret_index = -1
    # if len(tagged_seq)==1:
    #     print (tagged_seq)
    for i in range (0, len(seq)-len(tagged_seq)+1):
        temp_list = seq[i:i+len(tagged_seq)]
        if tagged_seq == temp_list:
            ret_index = i
            break
            # print (f'found sublist at {i} !!! ')
    if ret_index == -1:
        distance_dict = {}
        for i in range (0, len(seq)-len(tagged_seq)+1):
            temp_list = seq[i:i+len(tagged_seq)]
            levenstein_dist = ' '.joint(tagged_seq), ' '.join(temp_list)
            distance_dict[i]=levenstein_dist
        
    return ret_index

def preprocess_json():
    punctuation_list = [',','•',':','-','❖']
    raw_data_path = "../datasets/Resumes.json"
    seq_list = []
    tag_seq_list =[]
    with open (raw_data_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        resumes_raw = [json.loads(l) for l in lines]
        for resume_raw in resumes_raw:
            seq = resume_raw['content'].split()
            tag_seq = ["O" for i,_ in enumerate(seq)]
            for annotation in resume_raw['annotation']:
                tagged_seq = annotation['points'][0]['text'].split()
                try:
                    tag_label = annotation['label'][0]
                except:
                    print(f"no label for annotation {annotation['label']}")
                    tag_label = None
                if tag_label != None:
                    start_index = find_tagged_seq(seq, tagged_seq)
                    try:
                        assert (start_index >= 0)
                        for i in range (start_index, start_index+len(tagged_seq)):
                            if tagged_seq[i-start_index] not in punctuation_list:
                                tag_seq[i] = tag_label 
                    except:
                        print (f'did not find seq {tagged_seq}')

            seq_list.append(seq)
            tag_seq_list.append(tag_seq)

if __name__=='__main__':
    preprocess_json()


def my_tests():
    raw_data_path = "../datasets/Resumes.json"
    with open (raw_data_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        resumes_raw = [json.loads(l) for l in lines]
    seq = resumes_raw[0]['content'].split()
    print (seq)
    tagged_seq = resumes_raw[0]['annotation'][0]['points'][0]['text'].split()
    print (tagged_seq)
    print (tagged_seq in seq)
    tag_label = resumes_raw[0]['annotation'][0]['label'][0]
    print (tag_label)
    print (' '.join(seq).index(' '.join(tagged_seq)))
    print (seq[-len(tagged_seq)+1])
    for i in range (len(seq)-len(tagged_seq), len(seq)):
        if seq [i] != tagged_seq[i-len(seq)+len(tagged_seq)]:
            print (f'DIFFFFF! {i} \n')


    tag_seq = ["O" for i,_ in enumerate(seq)]






import pandas as pd
import glob
import os
import random
import json
if __name__=='__main__':
    input_dir = '../data'
    output_dir = '../data_for_train'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_list = glob.glob(input_dir+'/Youtube01-Psy.csv')
    all_data = []
    all_label = []
    for file in file_list:
        data= pd.read_csv(file)
        data = data.values
        print(data[:,4])
        all_data.extend(data[:,3])
        all_label.extend(data[:,4])
        pass

    data = [ [d,all_label[idx]] for idx,d in enumerate(all_data)]
    random.shuffle(data)
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):int(0.9*len(data))]
    test_data = data[int(0.9*len(data)):]
    json.dump(train_data,open(os.path.join(output_dir,'train.json'),'w'))
    json.dump(val_data,open(os.path.join(output_dir,'val.json'),'w'))
    json.dump(test_data,open(os.path.join(output_dir,'test.json'),'w'))
    
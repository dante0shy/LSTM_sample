
import json
import nltk

class Data_loader():
    def __init__(self,input_dir,word_list_file,max_len = 700):
        self.pos= 0
        self.max_len = max_len
        self.data_list = json.load(open(input_dir,'r'))
        self.stemmer = nltk.stem.PorterStemmer()
        self.word_list = tuple(json.load(open(word_list_file,'r')))
        self.rebuild_data_list =[]
        self.rebuild_data_mask =[]
        self.rebuild_data_label =[]
        #transform the text to max_len dimension vector and the mask for vector
        for  d,l in self.data_list:
            data,mask,label=self._rebuild_data(d,l)
            self.rebuild_data_list.append(data)
            self.rebuild_data_mask.append(mask)
            self.rebuild_data_label.append(label)


    def _rebuild_data(self,text,l):
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens]
        tokens = [self.word_list.index(token) for token in tokens if token in self.word_list]
        mask = len(tokens)
        if len(tokens) >= self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend([0] * (self.max_len-len(tokens)))

        tmp =[0,0]
        tmp[l] = 1
        return tokens ,mask ,tmp


    def get_data(self,batch_size):
        tims = len(self.data_list)/batch_size+1
        pos =0
        #generate one epoch batch data
        for _ in range(tims):
            start = pos
            if start+batch_size >len(self.data_list):
                yield self.rebuild_data_list[start:],self.rebuild_data_mask[start:],self.rebuild_data_label[start:]
                pos = len(self.data_list)
            else:
                end = start+batch_size
                yield self.rebuild_data_list[start:end],self.rebuild_data_mask[start:end],self.rebuild_data_label[start:end]
                pos = end
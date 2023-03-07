import random
import os
import csv

from transformers import AutoTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import copy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def np_unranked_unique(nparray):
    n_unique = len(np.unique(nparray))
    ranked_unique = []
    np.empty((n_unique),dtype=str)
    i = 0
    for x in nparray:
        if x not in ranked_unique:
            ranked_unique.append(x)
            i += 1
    ranked_unique=np.asarray(ranked_unique)
    return ranked_unique

class Data:
    def __init__(self, args):
        set_seed(args.seed)
        self.args = args
        MAX_SEQ_LEN = {'pdtb3':128} 
        self.max_seq_length = MAX_SEQ_LEN[args.dataset]
        self.data_dir = os.path.join(args.data_dir)
        self.label_list_level_1, self.label_list_label_level_2_or_level_3, self.label_list_label_level_2 = self.get_label_list()
        self.number_label_level_2 = len(self.label_list_label_level_2)-1
        self.number_label_level_1 = len(self.label_list_level_1)-1
        print('level_1 label:', self.number_label_level_1)
        print('level_2 label:', self.number_label_level_2)
        self.train_examples = self.get_examples('train')
        self.dev_examples = self.get_examples('dev')
        self.test_examples = self.get_examples('test')
        self.train_dataloader = self.get_data_loader(self.train_examples, 'train')
        self.eval_dataloader = self.get_data_loader(self.dev_examples, 'dev')
        self.test_dataloader = self.get_data_loader(self.test_examples, 'test')
        print('num_train_samples', len(self.train_examples))
        print('num_dev_samples', len(self.dev_examples))
        print('num_test_samples', len(self.test_examples))

    def get_label_list(self):
        data_label_level_2_or_level_3=[]
        data_label_level_1=[]
        data_label_level_2=[]
        for key,value in self.level_2_to_level_1().items():
            data_label_level_2.append(key)
            data_label_level_1.append(value)
        for key,value in self.level_2_to_level_3().items():
            data_label_level_2_or_level_3.append(key)

        label_list_label_level_2 = np.array([ i for i in data_label_level_2], dtype=str)
        label_list_label_level_2_or_level_3= np.array([ i for i in data_label_level_2_or_level_3], dtype=str)
        label_list_level_1 = np_unranked_unique(np.array([ i for i in data_label_level_1], dtype=str))
        return label_list_level_1, label_list_label_level_2_or_level_3, label_list_label_level_2

        # load data from files

    def get_examples(self, mode):
        lines = self.read_tsv(os.path.join(self.data_dir, mode + '.tsv'))
        examples = []
        for (i, line) in enumerate(lines):
            text_a = line[0]
            text_b = line[1]
            if(line[4]=='None'):
                label_level_2_or_level_3 = line[3]
            elif(line[4] not in self.label_list_label_level_2_or_level_3):
                label_level_2_or_level_3 = line[3]
            else:
                label_level_2_or_level_3 = line[4]#merge level-2 and level-3
            label_level_2=line[3]# for the level-2 target
            label_level_1 = self.level_2_to_level_1()[label_level_2]

#get the second label
            if(line[7]=='None'):
                label_level_2_or_level_3_label2 = line[6] 
            elif(line[7] not in self.label_list_label_level_2_or_level_3):
                label_level_2_or_level_3_label2= line[6]

            else:
                label_level_2_or_level_3_label2 = line[7]
            label_level_1_label2 = line[5]
            label_level_2_label2=line[6]   
            conn=line[8]
            conn2=line[9]
            examples.append(
                InputExample(text_a=text_a,text_b=text_b, 
                label_level_1=label_level_1, label_level_2=label_level_2, label_level_2_or_level_3=label_level_2_or_level_3,
                label_level_1_label2=label_level_1_label2, label_level_2_label2=label_level_2_label2,label_level_2_or_level_3_label2=label_level_2_or_level_3_label2,
                conn=conn,conn2=conn2))
        return examples

    def get_data_loader(self, examples, mode):
        tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model, do_lower_case=True)    
        
        features = self.convert_examples_to_features(examples, self.max_seq_length, tokenizer, mode)

        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        
        label_level_1 = torch.tensor([f.label_id_level_1 for f in features], dtype=torch.long)
        label_level_2_or_level_3 = torch.tensor([f.label_id_level_2_or_level_3 for f in features], dtype=torch.long)
        label_level_2= torch.tensor([f.label_id_level_2 for f in features], dtype=torch.long)

        label_level_1_label2 = torch.tensor([f.label_id_level_1_label2 for f in features], dtype=torch.long)
        label_level_2_or_level_3_label2 = torch.tensor([f.label_id_level_2_or_level_3_label2 for f in features], dtype=torch.long)
        label_level_2_label2  = torch.tensor([f.label_id_level_2_label2  for f in features], dtype=torch.long)

        input_ids_explicit = torch.tensor([f.input_ids_explicit for f in features], dtype=torch.long)
        input_mask_explicit = torch.tensor([f.input_mask_explicit for f in features], dtype=torch.long)
        segment_ids_explicit = torch.tensor([f.segment_ids_explicit for f in features], dtype=torch.long)

        data = TensorDataset(input_ids, input_mask, segment_ids, label_level_1, label_level_2, label_level_2_or_level_3,
        label_level_1_label2, label_level_2_label2, label_level_2_or_level_3_label2,input_ids_explicit,input_mask_explicit,segment_ids_explicit)
        
        if mode == 'train':
            sampler = RandomSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = self.args.train_batch_size)    
        elif mode == 'dev' or mode == 'test':
            sampler = SequentialSampler(data)
            dataloader = DataLoader(data, sampler=sampler, batch_size = self.args.eval_batch_size) 
        return dataloader

    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer, mode):
        if('roberta' in self.args.bert_model):
            label_map_level_1 = {}
            label_map_level_2_or_level_3 = {}
            label_map_level_2 = {}
            for i, label in enumerate(self.label_list_level_1):
                label_map_level_1[label] = i

            for i, label in enumerate(self.label_list_label_level_2_or_level_3):
                label_map_level_2_or_level_3[label] = i
            
            for i, label in enumerate(self.label_list_label_level_2):
                label_map_level_2[label] = i
            
            print(label_map_level_1, label_map_level_2_or_level_3,label_map_level_2)

            features = []
            for _, example in enumerate(examples):
                tokens_a = tokenizer.tokenize(example.text_a)
                tokens_b = None
                if example.text_b:
                    tokens_b = tokenizer.tokenize(example.text_b)
                if tokens_b: 
                    self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[0:(max_seq_length - 2)]
                tokens = []
                segment_ids = []
                tokens.append("<s>")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("</s>")
                segment_ids.append(0)
                if tokens_b:
                    tokens.append("</s>")
                    segment_ids.append(0)
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(0)
                    tokens.append("</s>")
                    segment_ids.append(0)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

            # data augmentation: add inserted connectives to the examples
                tokens_a = tokenizer.tokenize(example.text_a)
                tokens_b = None
                if example.text_b:
                    tokens_b = tokenizer.tokenize(example.conn + ', ' + example.text_b)

                if tokens_b:
                    self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[0:(max_seq_length - 2)]
                
                tokens = []
                segment_ids_explicit = []
                tokens.append("<s>")
                segment_ids_explicit.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids_explicit.append(0)
                tokens.append("</s>")
                segment_ids_explicit.append(0)

                if tokens_b:
                    tokens.append("</s>")
                    segment_ids_explicit.append(0)

                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids_explicit.append(0)
                    tokens.append("</s>")
                    segment_ids_explicit.append(0)
                input_ids_explicit = tokenizer.convert_tokens_to_ids(tokens)

                input_mask_explicit= [1] * len(input_ids_explicit)

                while len(input_ids_explicit) < max_seq_length:
                    input_ids_explicit.append(0)
                    input_mask_explicit.append(0)
                    segment_ids_explicit.append(0)
                assert len(input_ids_explicit) == max_seq_length
                assert len(input_mask_explicit) == max_seq_length
                assert len(segment_ids_explicit) == max_seq_length

               
                label_id_level_1 = label_map_level_1[example.label_level_1]
                label_id_level_2_or_level_3 = label_map_level_2_or_level_3[example.label_level_2_or_level_3]
                label_id_level_2 = label_map_level_2[example.label_level_2]
                label_id_level_1_label2 = label_map_level_1[example.label_level_1_label2]
                label_id_level_2_or_level_3_label2 = label_map_level_2_or_level_3[example.label_level_2_or_level_3_label2]
                label_id_level_2_label2  = label_map_level_2[example.label_level_2_label2]


                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id_level_1=label_id_level_1,
                                  label_id_level_2_or_level_3=label_id_level_2_or_level_3,
                                  label_id_level_2=label_id_level_2,
                                  label_id_level_1_label2=label_id_level_1_label2,
                                  label_id_level_2_or_level_3_label2=label_id_level_2_or_level_3_label2,
                                  label_id_level_2_label2=label_id_level_2_label2,
                                  input_ids_explicit=input_ids_explicit,
                                  input_mask_explicit=input_mask_explicit,
                                  segment_ids_explicit=segment_ids_explicit
                                  ))
            return features

        else:
            label_map_level_1 = {}
            label_map_level_2 = {}
            for i, label in enumerate(self.label_list_level_1):
                label_map_level_1[label] = i

            for i, label in enumerate(self.label_list_level_2):
                label_map_level_2[label] = i

            features = []
            for _, example in enumerate(examples):
                tokens_a = tokenizer.tokenize(example.text_a)
                tokens_b = None
                if example.text_b:
                    tokens_b = tokenizer.tokenize(example.text_b)

                if tokens_b:
                    self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[0:(max_seq_length - 2)]
                    
                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                if tokens_b:
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids.append(1)
                    tokens.append("[SEP]")
                    segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                tokens_a = tokenizer.tokenize(example.text_a)

                tokens_b = None
                if example.text_b:
                    tokens_b = tokenizer.tokenize(example.conn+', '+example.text_b)

                if tokens_b:
                    self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
                else:
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[0:(max_seq_length - 2)]
                tokens = []
                segment_ids_explicit = []
                tokens.append("[CLS]")
                segment_ids_explicit.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids_explicit.append(0)
                tokens.append("[SEP]")
                segment_ids_explicit.append(0)

                if tokens_b:
                    for token in tokens_b:
                        tokens.append(token)
                        segment_ids_explicit.append(1)
                    tokens.append("[SEP]")
                    segment_ids_explicit.append(1)
                input_ids_2 = tokenizer.convert_tokens_to_ids(tokens)

                input_mask_2 = [1] * len(input_ids_2)

                while len(input_ids_2) < max_seq_length:
                    input_ids_explicit.append(0)
                    input_mask_explicit.append(0)
                    segment_ids_explicit.append(0)
        
                assert len(input_ids_explicit) == max_seq_length
                assert len(input_mask_explicit) == max_seq_length
                assert len(segment_ids_explicit) == max_seq_length


                label_id_level_1 = label_map_level_1[example.label_level_1]
                label_id_level_2 = label_map_level_2[example.label_level_2]
                label_id_level_1_label2 = label_map_level_1[example.label_level_1_label2]
                label_id_level_2_label2 = label_map_level_2[example.label_level_2_label2]

                label_id_level_1 = label_map_level_1[example.label_level_1]
                label_id_level_2_or_level_3 = label_map_level_2_or_level_3[example.label_level_2_or_level_3]
                label_id_level_1_label2 = label_map_level_1[example.label_level_1_label2]
                label_id_level_2_or_level_3_label2 = label_map_level_2_or_level_3[example.label_level_2_or_level_3_label2]
            

            
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id_level_1=label_id_level_1,
                                  label_id_level_2_or_level_3=label_id_level_2_or_level_3,
                                  label_id_level_1_label2=label_id_level_1_label2,
                                  label_id_level_2_or_level_3_label2=label_id_level_2_or_level_3_label2,
                                  input_ids_explicit=input_ids_explicit,
                                  input_mask_explicit=input_mask_explicit,
                                  segment_ids_explicit=segment_ids_explicit
                                  ))
            return features

    def read_tsv(self, file):
        """Reads a tab separated value file."""
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            a=0
            for line in reader:
                if(a==0):               
                    a=-1
                    continue     
                if('train' not in file):
                    if(line[5]!='None'):
                        if(len(line[10].split('.'))==2 and len(line[12].split('.'))==2):
                                lines.append([line[7].replace('\n',''),line[8].replace('\n',''),
                                    line[10].replace('\n','').split('.')[0],line[10].replace('\n','').split('.')[1],'None',
                                    line[12].replace('\n','').split('.')[0],line[12].replace('\n','').split('.')[1],'None',
                                    line[9].replace('\n', ''),line[11].replace('\n', '')])
                        elif(len(line[10].split('.'))==2 and len(line[12].split('.'))==3):
                                    lines.append([line[7].replace('\n',''),line[8].replace('\n',''),
                                    line[10].replace('\n','').split('.')[0],line[10].replace('\n','').split('.')[1],'None',
                                    line[12].replace('\n','').split('.')[0],line[12].replace('\n','').split('.')[1],line[12].replace('\n','').split('.')[2],
                                    line[9].replace('\n', ''),line[11].replace('\n', '')])
                        elif(len(line[10].split('.'))==3 and len(line[12].split('.'))==2):
                                    lines.append([line[7].replace('\n',''),line[8].replace('\n',''),
                                    line[10].replace('\n','').split('.')[0],line[10].replace('\n','').split('.')[1],line[10].replace('\n','').split('.')[2],
                                    line[12].replace('\n','').split('.')[0],line[12].replace('\n','').split('.')[1],'None',
                                    line[9].replace('\n', ''),line[11].replace('\n', '')])
                        else:
                                    lines.append([line[7].replace('\n',''),line[8].replace('\n',''),
                                    line[10].replace('\n','').split('.')[0],line[10].replace('\n','').split('.')[1],line[10].replace('\n','').split('.')[2],
                                    line[12].replace('\n','').split('.')[0],line[12].replace('\n','').split('.')[1],line[12].replace('\n','').split('.')[2],
                                    line[9].replace('\n', ''),line[11].replace('\n', '')])


                    else:
                        if(len(line[10].split('.'))==2):
                            lines.append(
                                [line[7].replace('\n', ''), line[8].replace('\n', ''), 
                                line[10].replace('\n', '').split('.')[0], line[10].replace('\n', '').split('.')[1],'None',
                                'None', 'None','None',line[9].replace('\n', ''),'None'])
                        else:
                            lines.append(
                                [line[7].replace('\n', ''), line[8].replace('\n', ''), 
                                line[10].replace('\n', '').split('.')[0], line[10].replace('\n', '').split('.')[1],line[10].replace('\n', '').split('.')[2],
                                'None', 'None','None',line[9].replace('\n', ''),'None'])

                else:
                    #print(line)
                    if(len(line[9].split('.'))==2):

                        lines.append(
                            [line[6].replace('\n', ''), line[7].replace('\n', ''), 
                            line[9].replace('\n', '').split('.')[0],
                            line[9].replace('\n', '').split('.')[1],'None',
                            'None', 'None','None',line[8].replace('\n', ''),'None'])
                    else:
                        if(len(line[9].split('.'))!=3):
                            print(line[6],len(line[9].split('.')),line[9])
                        lines.append(
                            [line[6].replace('\n', ''), line[7].replace('\n', ''), 
                            line[9].replace('\n', '').split('.')[0],
                            line[9].replace('\n', '').split('.')[1],line[9].replace('\n', '').split('.')[2],
                            'None', 'None','None',line[8].replace('\n', ''),'None'])
      
            return lines


    def level_2_to_level_3(self):#include the third level
            label_dict={'Concession':'Comparison','Contrast':'Comparison',
                        'Condition':'Contingency','Purpose':'Contingency',
                        'Conjunction':'Expansion','Equivalence':'Expansion','Instantiation':'Expansion','Level-of-detail':'Expansion',
                        'Manner':'Expansion','Substitution':'Expansion',
                        'Synchronous':'Temporal',
                        'Precedence':'Asynchronous','Succession':'Asynchronous','Reason':'Cause',
                        'Result':'Cause','Reason+Belief':'Cause+Belief','Result+Belief':'Cause+Belief',
                        'None':'None'}
                        # 'Concession':'Comparison','Contrast':'Comparison','Cause':'Contingency','Cause+Belief':'Contingency',
                        # 'Condition':'Contingency','Purpose':'Contingency',
                        # 'Conjunction':'Expansion','Equivalence':'Expansion','Instantiation':'Expansion','Level-of-detail':'Expansion',
                        # 'Manner':'Expansion','Substitution':'Expansion',
                        # 'Asynchronous':'Temporal','Synchronous':'Temporal'}
            return label_dict
    def level_2_to_level_1(self):#only include level_1 and level_2
        label_dict={'Concession':'Comparison','Contrast':'Comparison','Cause':'Contingency','Cause+Belief':'Contingency',
                    'Condition':'Contingency','Purpose':'Contingency',
                    'Conjunction':'Expansion','Equivalence':'Expansion','Instantiation':'Expansion','Level-of-detail':'Expansion',
                    'Manner':'Expansion','Substitution':'Expansion',
                    'Asynchronous':'Temporal','Synchronous':'Temporal',
                    'None':'None'}
        return label_dict


class InputExample(object):
    """Convert data to inputs for bert"""

    def __init__(self, text_a,text_b, label_level_1=None, label_level_2=None,label_level_2_or_level_3=None,label_level_1_label2=None, label_level_2_label2=None,label_level_2_or_level_3_label2=None,conn=None,conn2=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label_level_1 = label_level_1
        self.label_level_2_or_level_3 = label_level_2_or_level_3
        self.label_level_2 = label_level_2
        self.label_level_1_label2 = label_level_1_label2
        self.label_level_2_or_level_3_label2 = label_level_2_or_level_3_label2
        self.label_level_2_label2 = label_level_2_label2
        self.conn=conn
        self.conn2=conn2
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id_level_1, label_id_level_2,label_id_level_2_or_level_3,label_id_level_2_label2, label_id_level_1_label2, label_id_level_2_or_level_3_label2, input_ids_explicit, input_mask_explicit, segment_ids_explicit):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id_level_1 = label_id_level_1
        self.label_id_level_2_or_level_3 = label_id_level_2_or_level_3
        self.label_id_level_2 = label_id_level_2
        self.label_id_level_1_label2 = label_id_level_1_label2
        self.label_id_level_2_or_level_3_label2 = label_id_level_2_or_level_3_label2
        self.label_id_level_2_label2 = label_id_level_2_label2
        self.input_ids_explicit = input_ids_explicit
        self.input_mask_explicit = input_mask_explicit 
        self.segment_ids_explicit = segment_ids_explicit
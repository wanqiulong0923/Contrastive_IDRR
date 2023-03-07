import os
import copy
import random
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, BertModel, AutoModel
import torch.nn.functional as F
from transformers.modeling_utils import WEIGHTS_NAME
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support,f1_score#, accuracy_score,



from tensorboardX import SummaryWriter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def accuracy_score(target,pred):
    num_example=0
    correct_example=0
    for i in range(len(target)):
        if(target[i]==1):
            num_example+=1
            if(pred[i]==1):
                correct_example+=1
    return correct_example/num_example*100

def evaluate_accuracy(pred, target, second_target=None): 
    num_classes = len(list(set(target)))  
    tem_target = np.zeros((len(target), num_classes), dtype=int)
    # print(tem_target.shape)
    for num, i in enumerate(target):
        tem_target[num][i] = 1
    target=tem_target
    num_examples, num_classes = target.shape

    correct = 0
    pred_matrix = np.zeros_like(target)
    target_matrix = target

    for i in range(num_examples):
        j = pred[i]
        pred_matrix[i, j] = 1 
        if second_target is not None:
            if (second_target[i] != num_classes):
                j = second_target[i]
                target_matrix[i, j] = 1
    target_number = target_matrix.sum(axis=0)
    real_labels = target_number > 0
    result = {}
    for i in range(len(pred_matrix)):
        for j in range(len(pred_matrix[i])):
            if (pred_matrix[i][j] != 1):
                pred_matrix[i][j] = -2
    for i in range(len(target_matrix)):
        for j in range(len(target_matrix[i])):

            if (target_matrix[i][j] != 1):
                target_matrix[i][j] = -1
    for i in range(num_examples):
            for j in range(num_classes):
                if (target_matrix[i, j] == pred_matrix[i, j] and target_matrix[i, j] == 1):
                    pred_matrix[i] = target_matrix[i]
                    break
    for i in range(len(pred_matrix)):
        for j in range(len(pred_matrix[i])):
            if (pred_matrix[i][j] == target_matrix[i][j]):
                correct += 1
                break

    for c in range(num_classes):
        if real_labels[c]:
            result[c] = accuracy_score(target_matrix[:, c], pred_matrix[:, c])
        else:
            result[c] = 1.0  
    result["overall"] = correct / num_examples
    return result

  



def evaluate_precision_recall_f1(pred, target, second_target=None,average="macro",mode='eva'):  
    num_classes = len(list(set(target))) 
    copy_pred=copy.deepcopy(pred)
    copy_target=copy.deepcopy(target)

    if(mode!='tra'):
        for i in range(len(copy_pred)):
            if(copy_pred[i]==second_target[i]):
                copy_pred[i]=target[i]
    target_matrix = np.zeros((len(target), num_classes), dtype=int)
    for num, i in enumerate(target):
        target_matrix[num][i] = 1
    num_examples, num_classes = target_matrix.shape
    pred_matrix = np.zeros_like(target_matrix)
    for i in range(num_examples):
        j = pred[i]
        pred_matrix[i, j] = 1
        if second_target is not None:
            if (second_target[i] != num_classes):
                j = second_target[i]
                target_matrix[i, j] = 1
    for i in range(num_examples):
        for j in range(num_classes):
            if (target_matrix[i, j] == pred_matrix[i, j] and target_matrix[i, j] == 1):
                pred_matrix[i] = target_matrix[i]
                break

    target_number = target_matrix.sum(axis=0)
    real_labels = target_number > 0
    result = {}
    for c in range(num_classes):
        if real_labels[c]:
            result[c] = tuple(
                precision_recall_fscore_support(target_matrix[:, c], pred_matrix[:, c], average="binary")[0:3])
        else:
            result[c] = (0.0, 0.0, 0.0)
    result["overall"] = tuple(
        precision_recall_fscore_support(target_matrix[:, real_labels], pred_matrix[:, real_labels],
                                        average=average)[0:4])

    result['f1_score_all']=f1_score(copy_target,copy_pred,average=average)
    return result



class ModelManager:
    def __init__(self, args, data):
        set_seed(args.seed)
        self.args=args
        self.data=data
        self.model=BertForModel(args,data.number_label_level_2, data.number_label_level_1)
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
        self.device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = self.get_optimizer(args)
        self.best_eval_score=0
        self.num_training_steps = int(len(data.train_examples)/args.train_batch_size)*args.num_train_epochs
        self.num_warmup_steps=int(self.args.warmup_proportion*self.num_training_steps)
        self.scheduler=get_linear_schedule_with_warmup(optimizer=self.optimizer,
        num_warmup_steps=self.num_warmup_steps,num_training_steps=self.num_training_steps)

    def get_optimizer(self,args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer=AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias= False)
        return optimizer

    def save_model(self):
        if not os.path.exists("new_log/"+self.args.log_name):
            os.makedirs("new_log/"+self.args.log_name)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join('new_log/'+self.args.log_name, WEIGHTS_NAME)
        model_config_file = os.path.join('new_log/'+self.args.log_name, 'config')
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())
    
    def load_model(self, model,saved_path=None):
        if(saved_path!=None):
            saved_path = os.path.join("new_log/"+saved_path, WEIGHTS_NAME)
        else:
            saved_path=os.path.join("new_log/"+self.args.save_model_path, WEIGHTS_Name)
        print(f'load checkpoint from{saved_path} model.')
        checkpoint= torch.load(saved_path)
        model.load_state_dict(checkpoint)

    def train(self):
        wait = 0
        best_model = None
        eval_score = self.eval()
        print('Epoch{} eval_score:{}'.format(-1,eval_score))
        total_step = 0
        for epoch in range(1, int(self.args.num_train_epochs)+1):
            All_labels_level_1 = torch.empty(0, dtype=torch.long).to(self.device)
            All_logits_level_1 = torch.empty((0, self.data.number_label_level_1)).to(self.device)
            All_labels_level_2 = torch.empty(0,dtype=torch.long).to(self.device)
            All_logits_level_2 = torch.empty((0,self.data.number_label_level_2)).to(self.device)
            
            self.model.train()
            train_loss=0
            number_train_examples, number_train_steps = 0, 0
            for step, batch in enumerate(self.data.train_dataloader):
                total_step +=1
                batch = tuple(t.to(self.device)for t in batch)
                input_ids, input_mask, segment_ids, label_id_level_1, label_id_level_2, label_id_level_2_or_level_3, \
                                label_level_1_label2, label_id_level_2_label2,label_id_level_2_or_level_3_label2, input_ids_explicit, input_mask_explicit, segment_ids_explicit = batch   

                with torch.set_grad_enabled(True):
                            loss, logits_level_1, logits_level_2, loss_level_1, loss_level_2, contrastive_loss, explicit_loss_level_1, explicit_loss_level_2 = \
                                self.model(input_ids, segment_ids, input_mask, labels_level_1=label_id_level_1, labels_level_2_or_level_3=label_id_level_2_or_level_3, label_level_2=label_id_level_2,
                                            mode="train", input_ids_explicit=input_ids_explicit
                                            , segment_ids_explicit=segment_ids_explicit, input_mask_explicit=input_mask_explicit, weight=None,
                                            )  
                            All_labels_level_1 = torch.cat((All_labels_level_1, label_id_level_1)) 
                            All_logits_level_1 = torch.cat((All_logits_level_1, logits_level_1)) 
                            All_labels_level_2 = torch.cat((All_labels_level_2, label_id_level_2))  
                            All_logits_level_2 = torch.cat((All_logits_level_2, logits_level_2)) 
                            loss.backward()
                            train_loss += loss.item() 
                            nn.utils.clip_grad_norm_(self.model.parameters(),2.0)
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                            number_train_examples += input_ids.size(0)
                            number_train_steps +=1

            total_probs, total_preds = F.softmax(All_logits_level_1.detach(),dim=1).max(dim=1)
            y_pred = total_preds.cpu().numpy()
            y_label = All_labels_level_1.cpu().numpy()

            acc_level_1 = evaluate_accuracy(target = y_label, pred=y_pred)['overall']*100
            micrio_f1_level_1 = evaluate_precision_recall_f1(target=y_label, pred=y_pred, average='micro',mode='tra')['overall'][2]
            macro_f1_level_1 = evaluate_precision_recall_f1(target=y_label, pred=y_pred, average='macro',mode='tra')['overall'][2]
                    

            total_probs, total_preds = F.softmax(All_logits_level_2.detach(), dim=1).max(dim=1)
            y_pred = total_preds.cpu().numpy()
            y_label = All_labels_level_2.cpu().numpy()
            acc_level_2 = evaluate_accuracy(target=y_label, pred=y_pred)['overall'] * 100
            micro_f1_level_2 = evaluate_precision_recall_f1(target=y_label, pred=y_pred, average='micro',mode='tra')['overall'][2]
            macro_f1_level_2 = evaluate_precision_recall_f1(target=y_label, pred=y_pred, average='macro',mode='tra')['overall'][2]

            loss = train_loss / number_train_steps
            test_score = self.test(epoch)
            print('Epoch {} train_loss: {}'.format(epoch, loss))
            eval_score = self.eval()
            print('Epoch {} eval_score: {}'.format(epoch, eval_score))
            print('Epoch {} test_score: {}'.format(epoch, test_score))

            if eval_score['level_1_acc']+eval_score['level_2_acc'] > self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score['level_1_acc'] + eval_score['level_2_acc']
            else:
                wait +=1
                if wait >= self.args.wait_patient:
                    break
                    
        self.model= best_model
        if self.args.save_model:
            self.save_model()
        test_score = self.test(epoch,write_file=False)
        print(test_score)

    def eval(self):
        with torch.no_grad():
            self.model.eval()
            All_labels_level_1 = torch.empty(0, dtype=torch.long).to(self.device)
            All_logits_level_1 = torch.empty((0, self.data.number_label_level_1)).to(self.device)
            All_labels_level_2 = torch.empty(0, dtype=torch.long).to(self.device)
            All_logits_level_2 = torch.empty((0, self.data.number_label_level_2)).to(self.device)
            All_labels_level_2_label2 = torch.empty(0, dtype=torch.long).to(self.device)
            All_labels_level_1_label2 = torch.empty(0, dtype=torch.long).to(self.device)

            for step, batch in enumerate(self.data.eval_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_id_level_1, label_id_level_2, label_id_level_2_or_level_3, \
                        label_level_1_label2, label_level_2_label2, label_id_level_2_or_level_3_label2, input_ids_explicit, input_mask_explicit, segment_ids_explicit = batch  
                 
                loss, logits_level_1, logits_level_2 = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels_level_1=label_id_level_1,label_level_2=label_id_level_2, labels_level_2_or_level_3=label_id_level_2_or_level_3, 
                                                                 input_ids_explicit=input_ids_explicit,
                                                              segment_ids_explicit=segment_ids_explicit, input_mask_explicit=input_mask_explicit
                                                              )  
                All_labels_level_1 = torch.cat((All_labels_level_1, label_id_level_1)) 
                All_logits_level_1 = torch.cat((All_logits_level_1, logits_level_1)) 
                All_labels_level_2= torch.cat((All_labels_level_2, label_id_level_2))  
                All_logits_level_2 = torch.cat((All_logits_level_2, logits_level_2))  
                All_labels_level_2_label2 = torch.cat((All_labels_level_2_label2,label_level_2_label2))  
                All_labels_level_1_label2 = torch.cat((All_labels_level_1_label2, label_level_1_label2))

            total_probs, total_preds = F.softmax(All_logits_level_1.detach(), dim=1).max(dim=1)
            y_pred = total_preds.cpu().numpy()
            y_label = All_labels_level_1.cpu().numpy()
            y_label_2 = All_labels_level_1_label2.cpu().numpy()
            acc_level_1 = evaluate_accuracy(target=y_label, second_target=y_label_2, pred=y_pred)['overall'] * 100
            micro_f1_level_1 = \
            evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='micro')[
                'overall'][2]
            macro_f1_level_1 = \
            evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='macro')[
                'overall'][2]
            all_micro_f1_level_1=evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='micro')
            all_macro_f1_level_1=evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='macro')


            total_probs, total_preds = F.softmax(All_logits_level_2.detach(), dim=1).max(dim=1)
            y_pred = total_preds.cpu().numpy()
            y_label = All_labels_level_2.cpu().numpy()
            y_label_2 = All_labels_level_2_label2.cpu().numpy()

            acc_level_2 = evaluate_accuracy(target=y_label, second_target=y_label_2, pred=y_pred)['overall'] * 100
            micro_f1_level_2 = \
            evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='micro')[
                'overall'][2]
            macro_f1_level_2 = \
            evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='macro')[
                'overall'][2]
            all_micro_f1_level_2=evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='micro')
            all_macro_f1_level_2=evaluate_precision_recall_f1(target=y_label, second_target=y_label_2, pred=y_pred, average='macro')

            results_all = {'level_2_acc': acc_level_2, 'level_1_acc': acc_level_1, 'micro_f1_level_1': micro_f1_level_1,
                           'micro_f1_level_2': micro_f1_level_2,
                           'macro_f1_level_1': macro_f1_level_1, 'macro_f1_level_2': macro_f1_level_2,
                           'all_micro_f1_level_2':all_micro_f1_level_2,
                           'all_macro_f1_level_1':all_macro_f1_level_1,
                           'all_macro_f2_level_2':all_macro_f1_level_2,
                           'all_micro_f1_level_1':all_micro_f1_level_1}
            return results_all
        

    def test(self, epoch,write_file=False,entropy=False,plot_figure=False):
        with torch.no_grad():
            self.model.eval()

            All_labels_level_1 = torch.empty(0, dtype=torch.long).to(self.device)
            All_logits_level_1 = torch.empty((0, self.data.number_label_level_1)).to(self.device)
            All_labels_level_2 = torch.empty(0, dtype=torch.long).to(self.device)
            All_logits_level_2 = torch.empty((0, self.data.number_label_level_2)).to(self.device)
            All_labels_level_2_label2 = torch.empty(0, dtype=torch.long).to(self.device)
            All_labels_level_1_label2 = torch.empty(0, dtype=torch.long).to(self.device)


            for step, batch in enumerate(self.data.test_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_id_level_1, label_id_level_2, label_id_level_2_or_level_3, \
                    label_level_1_label2, label_level_2_label2, label_level_2_or_level3_label2, input_ids_explicit, input_mask_explicit, segment_ids_explicit = batch
                print(input_ids_explicit.shape)
                feature, logits_level_1, logits_level_2 = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels_level_1=label_id_level_1,label_level_2=label_id_level_2, labels_level_2_or_level_3=label_id_level_2_or_level_3, 
                                                                 input_ids_explicit=input_ids_explicit,
                                                              segment_ids_explicit=segment_ids_explicit, input_mask_explicit=input_mask_explicit
                                                              )  


                All_labels_level_1 = torch.cat((All_labels_level_1, label_id_level_1)) 
                All_logits_level_1 = torch.cat((All_logits_level_1, logits_level_1)) 
                All_labels_level_2= torch.cat((All_labels_level_2, label_id_level_2))  
                All_logits_level_2 = torch.cat((All_logits_level_2, logits_level_2))  
                All_labels_level_2_label2 = torch.cat((All_labels_level_2_label2,label_level_2_label2))  
                All_labels_level_1_label2 = torch.cat((All_labels_level_1_label2,label_level_1_label2))

           


            total_probs, total_preds = F.softmax(All_logits_level_1.detach(), dim=1).max(dim=1)
            y_pred = total_preds.cpu().numpy()
            y_true = All_labels_level_1.cpu().numpy()
            y_true_2 = All_labels_level_1_label2.cpu().numpy()
            


            acc_level_1 = evaluate_accuracy(target=y_true, second_target=y_true_2, pred=y_pred)['overall'] * 100
            micro_f1_level_1 = \
            evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred, average='micro')[
                'overall'][2]
            macro_f1_level_1 = \
            evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred, average='macro')[
                'overall'][2]

            all_acc_level_1 = evaluate_accuracy(target=y_true, second_target=y_true_2, pred=y_pred)
            all_micro_f1_level_1 = evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred,
                                                               average='micro')
            all_macro_f1_level_1 = evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred,
                                                               average='macro')
            total_probs, total_preds = F.softmax(All_logits_level_2.detach(), dim=1).max(dim=1)
            y_pred = total_preds.cpu().numpy()
            y_true = All_labels_level_2.cpu().numpy()
            y_true_2 = All_labels_level_2_label2.cpu().numpy()

            acc_level_2 = evaluate_accuracy(target=y_true, second_target=y_true_2, pred=y_pred)['overall'] * 100

            micro_f1_level_2 = evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred, average='micro')[
                'overall'][2]
            macro_f1_level_2 = evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred, average='macro')[
                'overall'][2]
            all_acc_level_2 = evaluate_accuracy(target=y_true, second_target=y_true_2, pred=y_pred)
            all_micro_f1_level_2 = evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred,
                                                             average='micro')
            all_macro_f1_level_2 = evaluate_precision_recall_f1(target=y_true, second_target=y_true_2, pred=y_pred,
                                                             average='macro')

            results_all = {'level_2_acc': acc_level_2, 'level_1_acc': acc_level_1, 'micro_f1_level_1': micro_f1_level_1,
                           'micro_f1_level_2': micro_f1_level_2,
                           'macro_f1_level_1': macro_f1_level_1, 'macro_f1_level_2': macro_f1_level_2, 
                          
                           'level_1_accuracy': all_acc_level_1, 'level_1 macro f1': all_macro_f1_level_1,
                           'level_1 micro f1': all_micro_f1_level_1,
                           'level_2_accuracy': all_acc_level_2, 'level_2 macro f1': all_macro_f1_level_2,
                           'level_2 micro f1': all_micro_f1_level_2}
            return results_all

class BertForModel(nn.Module):
    def __init__(self, args, num_level_2, num_level_1):
        super(BertForModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_level_2
        self.args = args
        print("start loading model")      
        s_ = time.time()
        self.bert = AutoModel.from_pretrained(args.bert_model)
        print("end loading model")
        print(f"time:{time.time()- s_}")
        self.config = self.bert.config
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.temperature = args.temperature 
        self.classifier_level_2 = nn.Linear(self.config.hidden_size, num_level_2)
        self.classifier_level_1 = nn.Linear(self.config.hidden_size, num_level_1)  

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels_level_1=None, label_level_2=None, labels_level_2_or_level_3=None,
                mode=None, input_ids_explicit=None, segment_ids_explicit=None, input_mask_explicit=None,
                weight=None, warmup_step=0, current_step=0): 
        
        encoded_layer_12, pool, encoded_layer = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                                          output_hidden_states=True,
                                                          return_dict=False)

        encoded_layer_12_explicit, explicit_pool, encoded_layer_explicit = self.bert(input_ids=input_ids_explicit, attention_mask=input_mask_explicit,
                                                                   token_type_ids=segment_ids_explicit,
                                                                   output_hidden_states=True, return_dict=False)

        pooled_layer_12 = self.dense(encoded_layer_12[:, 0, :])
        pooled_layer_12_explicit = self.dense(encoded_layer_12_explicit[:, 0, :])

        pooled_output_12 = self.activation(pooled_layer_12)
        pooled_output_12_explicit = self.activation(pooled_layer_12_explicit)
        
        pooled_output_12 = self.dropout(pooled_output_12)
        pooled_output_12_explicit = self.dropout(pooled_output_12_explicit)
        logits_level_1 = self.classifier_level_1(pooled_output_12)
        logits_level_2 = self.classifier_level_2(pooled_output_12)

        explicit_logits_level_1 = self.classifier_level_1(pooled_output_12_explicit)
        explicit_logits_level_2 = self.classifier_level_2(pooled_output_12_explicit)

        if mode == 'train':
            contrastiveLoss = self.contrastive_setting(pooled_output_12, pooled_output_12_explicit, labels_level_1, labels_level_2_or_level_3)
            loss = nn.CrossEntropyLoss()(logits_level_1, labels_level_1)
            loss_2 = nn.CrossEntropyLoss()(logits_level_2, label_level_2)
            explicit_loss = nn.CrossEntropyLoss()(explicit_logits_level_1, labels_level_1)
            explicit_loss_2 = nn.CrossEntropyLoss()(explicit_logits_level_2, label_level_2)
            # print(loss_2,contrastiveLoss)
            return  self.args.b1*contrastiveLoss + loss+ loss_2+ 0.1*(explicit_loss + explicit_loss_2), logits_level_1, logits_level_2, loss, loss_2, contrastiveLoss, explicit_loss, explicit_loss_2  
        else:
            return pooled_output_12, logits_level_1, logits_level_2 

    def pair_cosine_similarity(self, x, x_explicit, eps=1e-8):
            n = x.norm(p=2, dim=1,keepdim=True)
            n_explicit = x_explicit.norm(p=2,dim=1,keepdim=True)
            return (x @ x.t()) / (n * n.t()).clamp(min=eps), (x_explicit @ x_explicit.t()) / (n_explicit * n_explicit.t()).clamp(min=eps), (
                x @ x_explicit.t()) / (n * n_explicit.t()).clamp(min=eps)

    def contrastiveLoss(self,feature_implicit, feature_explicit, positive_position, mask_all, mask_positive, cuda = True, t=0.1):
            sim_implicit, sim_explicit, sim_implicit_explicit = self.pair_cosine_similarity(feature_implicit, feature_explicit)
            sim_implicit = torch.exp(sim_implicit / t)
            sim_explicit = torch.exp(sim_explicit / t)
            sim_implicit_explicit = torch.exp(sim_implicit_explicit / t)

            positive_count = positive_position.sum(1)
            negative_position = (~(positive_position.bool())).float()
            dis = torch.div((sim_implicit * (mask_positive - self.args.con1* torch.eye(sim_implicit.size(0)).float().cuda()) + sim_implicit_explicit * mask_positive), ( \
                (sim_implicit * (mask_all - self.args.con1* torch.eye(sim_implicit.size(0)).float().cuda())).sum(1, keepdim=True).repeat([1, sim_implicit.size(0)]) \
                 + (sim_implicit_explicit * mask_all).sum(1,keepdim=True).repeat([1, sim_implicit_explicit.size(0)]))) + negative_position
                  
            dis_explicit = torch.div(
            (sim_explicit * (mask_positive - self.args.con1 * torch.eye(sim_explicit.size(0)).float().cuda()) + sim_implicit_explicit.T * mask_positive), ( \
                    (sim_explicit * (mask_all - self.args.con1* torch.eye(sim_implicit.size(0)).float().cuda())).sum(1, keepdim=True).repeat([1, sim_explicit.size(0)])  \
                    + (sim_implicit_explicit * mask_all).sum(0).unsqueeze(1).repeat([1, sim_implicit_explicit.size(0)]))) + negative_position 
            loss = (torch.log(dis).sum(1) + torch.log(dis_explicit).sum(1)) / positive_count
            return -loss.mean()

    def contrastive_setting(self, feature_implicit, feature_explicit,labels_level_1, labels_level_2):
            batch_size = feature_implicit.shape[0]
            if labels_level_2.shape[0]!= batch_size:
                raise ValueError("Num of labels does not match sum of features")
            if(self.args.task=='pdtb3'):
                level_2_level_1 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 1, 14:1, 15:1, 16:1}
            else:
                level_2_level_1 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 11:2 , 12:2 , 13: 2, 14:2, 15:2, 16:2,
            17:2,18:2,19:2,20:3,21:3,22:3}
            mask_all = torch.zeros((batch_size, batch_size))
            mask_positive = torch.zeros((batch_size, batch_size))  
        
            for i in range(batch_size):
                for j in range(batch_size):
                    if (i == j):
                        mask_all[i][j] = self.args.con1
                    elif (labels_level_2[i].item() == labels_level_2[j].item()):
                        mask_all[i][j] = self.args.con1
                    elif (level_2_level_1[labels_level_2[i].item()] == level_2_level_1[labels_level_2[j].item()]):
                        mask_all[i][j] = 1
                    # elif (labels_level_1[i].item() == labels_level_1[j].item()):
                    #     mask_all[i][j] = self.args.con2
                    else:
                        mask_all[i][j] = 0
            for i in range(batch_size):
                for j in range(batch_size):
                    if (i == j):
                        mask_positive[i][j] = self.args.con1
                    elif (labels_level_2[i].item() == labels_level_2[j].item()):
                        mask_positive[i][j] = self.args.con1
                    elif (level_2_level_1[labels_level_2[i].item()] == level_2_level_1[labels_level_2[j].item()]):
                        mask_positive[i][j] = 0
                    else:
                        mask_positive[i][j] = 0

            positive_position = torch.eq(labels_level_2.unsqueeze(1), labels_level_2.unsqueeze(1).T).float().to(self.device)
            mask_all = mask_all.to(self.device)
            mask_positive = mask_positive.to(self.device)

            loss = self.contrastiveLoss(feature_implicit, feature_explicit, positive_position, mask_all, mask_positive, t=self.temperature)

            return loss






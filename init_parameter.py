from argparse import ArgumentParser

def init_model():
    parser = ArgumentParser()
    
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    
    parser.add_argument("--dstore_mmap", default='knn_explicit/datastore', type=str,
                    help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--dstore_size", default=20244, type=int,
                    help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--k", default=25, type=int,
                    help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--knn_lambda", default=0.4, type=float,
                    help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--save_results_path", type=str, default='outputs', help="The path to save results.")
    
    parser.add_argument("--pretrain_dir", default='pretrain_models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 
    
    parser.add_argument("--bert_model", default="../bert_base_uncased", type=str, help="The path for the pre-trained bert model.")
    
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--freeze_bert_parameters", action="store_true", help="Freeze the last parameters of BERT.")

    parser.add_argument("--save_model", action="store_true", help="Save trained model.")
    parser.add_argument("--save_model_path", type=str, default='outputs', help="The path to save results.")

    parser.add_argument("--pretrain", action="store_true", help="Pre-train the model with labeled data.")

    parser.add_argument("--ce_continue_cl", action="store_true", help="先进行对比学习在进行ce学习")
    parser.add_argument("--log_name", default=None, type=str, required=True,
                        help="The name of log.")

    parser.add_argument("--dataset", default=None, type=str, required=True, 
                        help="The name of the dataset to train selected.")
    
    #parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True, help="The number of known classes.")
    
    #parser.add_argument("--cluster_num_factor", default=1.0, type=float, required=True, help="The factor (magnification) of the number of clusters K.")

    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization.")

    parser.add_argument("--method", type=str, default='DeepAligned',help="Which method to use.")

    parser.add_argument("--labeled_ratio", default=0.1, type=float, help="The ratio of labeled samples in the training set.")
    
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id.")

    parser.add_argument("--train_batch_size", default=128, type=int,#初始是128
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop.") #early stop的步数

    parser.add_argument("--num_pretrain_epochs", default=100, type=float,
                        help="The pre-training epochs.")
    parser.add_argument("--ce_pretrain_epochs", default=40, type=float,
                        help="The pre-training epochs.")

    parser.add_argument("--num_train_epochs", default=100, type=float,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="The temperature for dot product.")
    
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The learning rate for training.")  
    parser.add_argument("--alpha", default=0.1, type=float,
                        help="portion of hard samples")
    parser.add_argument("--a1", default=0.7, type=float,
                        help="hyper of thres")
    parser.add_argument("--a2", default=0.2, type=float,
                        help="hyper2 of thres")
    parser.add_argument("--positive_l1", default=1.2, type=float,
                        help="portion of hard samples")
    parser.add_argument("--positive_l2", default=1.4, type=float,
                        help="hyper of thres")
    parser.add_argument("--negative", default=0.9, type=float,
                        help="hyper2 of thres")
    parser.add_argument("--con1", default=1.8, type=float,
                            help="最后一层正例的比例大小")
    parser.add_argument("--con2", default=1.4, type=float,
                            help="第二层正例的比例大小")

    parser.add_argument("--b1", default=1, type=float,
                            help="loss计算中contrastive")
    parser.add_argument("--task", default='pdtb2', type=str,
                            help="loss计算中contrastive")


    return parser

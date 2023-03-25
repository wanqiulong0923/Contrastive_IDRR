from init_parameter import init_model
import contrastive_data
from contrastive_model import ModelManager
from transformers import logging
import time
import os

if __name__ == '__main__':
    logging.set_verbosity_error()
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    args.log_name=args.log_name+'_warmup_proportion_{}_seed_{}_train_batch_size_{}_lr_{}_temperatue_{}_alpha_{}_a1_{}_a2_{}' \
                                '_neg_{}_b1_{}_con1_{}_con2_{}'.format(args.warmup_proportion,args.ce_continue_cl
                                                                                                    ,args.seed,args.train_batch_size,args.lr,args.temperature
                                                                                                    ,args.alpha,args.a1,args.a2,args.negative,
                                                                                                    args.b1,args.con1,args.con2)
    data = contrastive_data.Data(args)
    f1 = '/rds-d5/user/co-long1/hpc-work/python/hiera/contrastive/contrastive_data.py'
    f2 = '/rds-d5/user/co-long1/hpc-work/python/hiera/contrastive/contrastive_model.py'
    command='rm -rf '+'new_log/'+args.log_name 
    os.system(command)  

    if not os.path.exists('new_log/'+args.log_name):
        os.makedirs('new_log/'+args.log_name)

  
    command = "cp -r " + f1 + " " + 'new_log/'+args.log_name+'/implicit.py'
    os.system(command) 
    command = "cp -r " + f2 + " " + 'new_log/'+args.log_name+'implicit_model.py' 
    os.system(command) 

    args.pretrain_dir= args.pretrain_dir+'/{}'.format(args.log_name)

    start_time = time.time()

    manager = ModelManager(args, data)
    # manager.load_model(manager.model,saved_path='pdtb3-robert-base-explicit_warmup_proportion_0.1_seed_42_train_batch_size_256_lr_5e-05_temperatue_0.2_neg_0.9_b1_1_con1_1.6_con2_1.3')
   
    end_time = time.time()
    print(f"time: {end_time - start_time}")

    print('Training begin...')
    start = time.time()
    manager.train()
   
    metric = manager.test(1)

    end = time.time()
    print(end - start)
    print('Training finished!')

    print('Evaluation begin...')
    #metric = manager.test(write_file=True)
    print('Evaluation finished!')
    print(metric)

dst_vocab = 'cdistnet/utils/dict_36.txt'   
dst_vocab_size = 40
rgb2gray =False
keep_aspect_ratio = False
width = 144 #100 144
height = 36 #32 36
max_width = 180
is_lower = True 
cnn_num = 2
leakyRelu = False
hidden_units = 384#768
ff_units = 1024      #ff
scale_embedding = True
attention_dropout_rate = 0.0
residual_dropout_rate = 0.1
num_encoder_blocks = 3
num_decoder_blocks = 3
num_heads = 8
beam_size = 1
n_best = 1
data_aug = True
num_fiducial = 20           #number of fiducial points of TPS-STN
train_method = 'origin'     #dist:  use distributed train method origin
optim = 'origin'

# method choice
tps_block = 'TPS'      # TPS None
feature_block = 'Resnet45'    # Resnet45 Resnet31 MTB

train = dict(
    grads_clip=5,
    optimizer='adam_decay',  # not used
    learning_rate_warmup_steps=10000,
    label_smoothing=True,  # fixed in code
    shared_embedding=False,  # not used
    device='cuda',
    # gt_file=['../dataset/MJ/MJ_train/',
    #         '../dataset/MJ/MJ_test/',
    #         '../dataset/MJ/MJ_valid/',
    #         '../dataset/ST'],
    gt_file=['/data/xk/datasets/training/MJ/MJ_valid','/data/xk/datasets/training/MJ/MJ_test','/data/xk/datasets/training/MJ/MJ_train','/data/xk/datasets/training/ST'],#/media/dell/A0BA0D9FBA0D72D8/xk/training/MJ/MJ_valid
    #gt_file=['/root/autodl-tmp/datasets/clean_lmdb/IIIT5k'],
    num_worker=0,
    # model_dir ='model/test',
    model_dir='models/11_14_4', 
    num_epochs=20,
    # gpu_device_ids=[1,2,3,4,5,6,7],
    batch_size=900,  # 4gpu 1800 256384 512  384 768 1024
    model=None,
    # model ='models/new_baseline_sem_pos_pos_vis_3_32*128_tps_resnet45_epoch_6/model_epoch_5.pth',
    # current_epoch=6,  # epoch start
    save_iter=10000,
    display_iter=100,
    tfboard_iter=100,
    eval_iter=1000,#1568 2091  3136,1568,1046
)


val = dict(
    model='',  # abandon
    device='cuda',
    # is_val_gt=True,
    # image_dir='datasets/NewVersion/val_data',
    # gt_file= [
    #            './dataset/eval/IC13_857',
    #            './dataset/eval/SVT',
    #             './dataset/eval/IIIT5k_3000',
    #            './dataset/eval/IC15_1811',
    #             './dataset/eval/SVTP',
    #            './dataset/eval/CUTE80'],

    gt_file=['/data/xk/datasets/evaluation/clean_lmdb/IIIT5k',
            '/data/xk/datasets/evaluation/clean_lmdb/SVT',
            '/data/xk/datasets/evaluation/clean_lmdb/IC13_857',
            '/data/xk/datasets/evaluation/clean_lmdb/IC13_1015',
            '/data/xk/datasets/evaluation/clean_lmdb/IC15_1811',
            '/data/xk/datasets/evaluation/clean_lmdb/IC15_2077',
            '/data/xk/datasets/evaluation/clean_lmdb/SVTP',
            '/data/xk/datasets/evaluation/clean_lmdb/CUTE80',
            #'/data/xk/datasets/evaluation/wordart',
            #'/data/xk/datasets/evaluation/COCOv1.4',
            #'/data/xk/datasets/evaluation/ArT',
            #'/data/xk/datasets/evaluation/Uber',
            #'/data/xk/datasets/evaluation/host',
            #'/data/xk/datasets/evaluation/wost',
            #'/data/xk/datasets/lmdb_format/artistic',
            #'/data/xk/datasets/lmdb_format/contextless',
            #'/data/xk/datasets/lmdb_format/curve',
            #'/data/xk/datasets/lmdb_format/general',
            #'/data/xk/datasets/lmdb_format/multi_oriented',
            #'/data/xk/datasets/lmdb_format/multi_words',
            #'/data/xk/datasets/lmdb_format/salient',
            
    ],

    
    # gt_file=['datasets/NewVersion/val_data/val_data.txt'],
    # gt_file='../dataset/MJ/MJ_valid/',
    batch_size=64,  # 4gpu 1800 64
    num_worker=0,
)


test = dict(
    test_one=False,
    device='cuda',
    rotate=False,  
    best_acc_test=True,  # test best_acc
    eval_all=False,  # test all model_epoch_9_iter_4080.pth
    s_epoch=7,  # start_epoch
    e_epoch=10,  
    avg_s=-1,  
    avg_e=9,
    avg_all=False,  
    is_test_gt=False,
    image_dir= None,     #if is_test_gt == False,needn't use image_dir
    test_list=[
 '/data/xk/datasets/evaluation/clean_lmdb/IIIT5k'
    ],

    batch_size=20,
    num_worker=0,
    model_dir='/home/yzh/xk/cvpr/PD/models/11_14_4/',  # load test model
    script_path='utils/Evaluation_TextRecog/script.py',
    python_path='/data1/zs/anaconda3/envs/py2/bin/python' #abandon
)

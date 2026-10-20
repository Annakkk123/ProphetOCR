import os
import argparse
import codecs
import subprocess
import csv
import glob
# os.environ['CUDA_VISIBLE_DEVICES']="6"
from tqdm import tqdm
from mmcv import Config
import torch
import torch.nn as nn
import torch.distributed as dist
import time
from cdistnet.optim.loss import cal_performance
from cdistnet.data.data import make_data_loader_test, make_lmdb_data_loader_test,make_data_loader
# from cdistnet.hdf5loader import make_data_loader
from cdistnet.model.translator import Translator

from cdistnet.model.model import build_CDistNet
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train CDistNet')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,help='node rank for distributed training')
    args = parser.parse_args()
    return args


def load_vocab(vocab=None, vocab_size=None):
    """
    Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    # print('Load set vocabularies as %s.' % vocab)
    vocab = [' ' if len(line.split()) == 0 else line.split()[0] for line in codecs.open(vocab, 'r', 'utf-8')]
    vocab = vocab[:vocab_size]
    assert len(vocab) == vocab_size
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def get_alphabet(dict_path):
    with open(dict_path, "r") as f:
        data = f.readlines()
        data = list(map(lambda x: x.strip(), data))
        data = data[4:]
    return data


def get_pred_gt_name(translator, idx2word, b_image, b_gt, b_name, num, dict_path, rotate,rgb2gray,is_test_gt=True):
    # rgb2gray=False
    gt_list, name_list, pred_list = [], [], []
    alphabet = get_alphabet(dict_path)  # not used
    if rotate:
        batch_hyp, batch_scores = translator.translate_batch(
            images=b_image.view(-1, b_image.shape[-2], b_image.shape[-1]).unsqueeze(dim=1)
        )
        batch_scores = torch.cat(batch_scores, dim=0).view(-1, 3)
        _, idx = torch.max(batch_scores, 1)
        idx = torch.arange(0, idx.shape[0], dtype=torch.long) * 3 + idx.cpu()
        batch_hyp_ = []
        for id, v in enumerate(batch_hyp):
            if id in idx:
                batch_hyp_.append(v)
        batch_hyp = batch_hyp_
    else:
        if rgb2gray == False:
            batch_hyp, batch_scores = translator.translate_batch(images=b_image[:, :3, :, :])
        else:
            batch_hyp, batch_scores = translator.translate_batch(images=b_image[:, 0:1, :, :])
    for idx, seqs in enumerate(batch_hyp):
        for seq in seqs:
            seq = [x for x in seq if x != 3]
            pred = [idx2word[x] for x in seq]
            pred = ''.join(pred)
        flag = False
        if is_test_gt==False:
            num += 1
            pred_list.append('word_{}.png'.format(num) + ', "' + pred + '"\n')
            gt_list.append('word_{}.png'.format(num) + ', "' + b_gt[idx] + '"\n')
            name_list.append(b_name[idx] + '\n')
        else:
            num += 1
            pred_list.append('{}'.format(b_name[idx]) + ', "' + pred + '"\n')
            gt_list.append('{}'.format(b_name[idx]) + ', "' + b_gt[idx] + '"\n')
            name_list.append(b_name[idx] + '\n')
    return gt_list, name_list, pred_list, num


def write_to_file(file_name, datas):
    with open(file_name, "w") as f:
        f.writelines(datas)


def eval_and_save(script_path, gt_file, submit_file, python_path):
    cmd = "%s %s -g=%s -s=%s" % (python_path, script_path, gt_file, submit_file)
    print("cmd:{}".format(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    cmd_out = p.stdout.read().decode('utf-8')
    # Calculated!{"crwup": 0.9443155452436195, "tedupL": 0.9704312114989733, "tedL": 0.9704312114989733, "totalWords": 862, "crwN": 814.0, "crwupN": 814.0, "ted": 144.0, "tedup": 144.0, "detWords": 862, "crw": 0.9443155452436195}
    print("cmd_out: {}".format(cmd_out))
    crwup = cmd_out[cmd_out.index('{') + 1:cmd_out.index('}')].split(', ')[0]
    res = crwup.split(': ')[1]
    res = float(res) * 100
    res = '%.2f' % float(res)
    return str(res)


def start_eval(script_path, data_name, gt_file, pred_file, name_file, lexdir, python_path):
    submit_file_list = []
    res = [eval_and_save(script_path, gt_file, pred_file, python_path)]
    if data_name == 'icdar2003':
        submit_file_list.append(ic03_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
        submit_file_list.append(ic03_lex(os.path.join(lexdir, data_name), 'full', gt_file, pred_file, name_file))
    elif data_name == 'svt':
        submit_file_list.append(svt_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
    elif data_name == 'svt-p':
        submit_file_list.append(svt_p_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
    elif data_name == 'iiit5k':
        submit_file_list.append(iiit5k_lex(os.path.join(lexdir, data_name), '50', gt_file, pred_file, name_file))
        submit_file_list.append(iiit5k_lex(os.path.join(lexdir, data_name), '1k', gt_file, pred_file, name_file))
    for submit_file in submit_file_list:
        res.append(eval_and_save(script_path, gt_file, submit_file, python_path))
    return res

def start_eval_simple(submit_list,gt_list,name_list):
    i = 0
    total = 0
    num = len(gt_list)
    err_list = []
    for pred in submit_list:
        if pred.lower() == gt_list[i].lower():
            total+=1
        # else:
            # err_list.append("{} image is diff:{} ---- {}\n".format(name_list[i],pred,gt_list[i].lower()))
        i +=1
#     print(err_list)
#     with open(err_dir, "w") as f:
#         f.writelines(err_list)
    return total / num *100.0

def eval(cfg, args,model_path):
    # init dist_train
    if cfg.train_method=='dist':
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    model = build_CDistNet(cfg)
    model.load_state_dict(torch.load(model_path))
    if cfg.train_method=='dist':
        model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        device = torch.device(cfg.test.device)
        model.to(device)
    model.eval()

    translator = Translator(cfg, model=model)
    word2idx, idx2word = load_vocab(cfg.dst_vocab, cfg.dst_vocab_size)
    lexdir = cfg.test.image_dir
    result_line = []
    val_dataloader=[]
    for val_gt_file in cfg.val.gt_file:
        val_dataloader.append(make_data_loader(cfg, is_train=False,val_gt_file=val_gt_file))
    i = 0
    avg_acc = .0
    total_loss = 0
    data_len=len(val_dataloader)
    list_accuracy = 'accuracy: '
    list_accuracy11 = 'acc: '
    #n_word_total = 0
    #n_word_correct = 0
    #n_word_correct1 = 0
    #len_all = 0
    #f = 0
    #all = 0

    #datasetlist=['III5K','SVT','IC13_857','IC13_1015','IC15_1811','IC15_2077','SVTP','CUTE80','wordart','coco','ArT','Uber','host','wost','artistic','contextless','curve','general','multi_oriented','multi_words','salient']#
    datasetlist=['III5K','SVT','IC13_857','IC13_1015','IC15_1811','IC15_2077','SVTP','CUTE80']
    with torch.no_grad():
        for dataset in val_dataloader:
            n_correct = 0
            infer_time = 0
            batch_a = 0
            n_word_total = 0
            n_word_correct = 0
            n_word_correct1 = 0
            len_all = 0
            f = 0
            all = 0
            for iteration, batch in enumerate(dataset, 0):
                if cfg.train_method=='dist':
                    images = batch[0].cuda(device,non_blocking=True)
                    tgt = batch[1].cuda(device,non_blocking=True)
                else:
                    #if batch == None:
                       #continue
                    images = batch[0].to(device)
                    tgt = batch[1].to(device)
                    
                     
                
                start_time = time.time()
                batch_a = batch_a+ tgt.shape[0]
                pred,pred1,pred2 = model(images,tgt)
                forward_time = time.time() - start_time
                #print("pred:",pred.shape)
                #aaaa = pred.max(2)[1]
                a = pred[:, :,:].reshape(-1,40)
                #b = pred1[:, :,:].reshape(-1,40)
                a = a.max(1)[1]
                #b = b.max(1)[1]
                non = tgt.contiguous().view(-1).ne(0)
                for ff in range(non.shape[0]):
                    if non[ff] == True:
                       len_all+=1
                all = all + len(non)
                #haha = a.eq(b)
                
               # print("haha:",haha)
                #n_correct = n_correct.masked_select(non_pad_mask).sum().item()
                #for t in range(haha.shape[0]):
                   #if haha[t] == True:
                      # f = f  + 1
               
                #pred = F.log_softmax(pred, dim=-1)
               # _,index = pred.max(2)
               
                pred = pred[:, :,:].reshape(-1,40)
                #pred1 = pred1[:, 1:,:].reshape(-1,40)
                
                tgt = tgt[:, 1:]
                
                #labels = decode(tgt,idx2word)
                #preds_str = decode(index,idx2word)
                
                
                #for pred, gt in zip(preds_str, labels):
                    #pred = pred[:pred.find('</s>')]  # prune after "end of sentence" token ([s])
                    #gt = gt[:gt.find('</s>')]
                    #with open(f'./fuck.txt', 'a') as log:
                       #log.write(pred+'  ')
                       #log.write(gt+'\n')
                    #if pred == gt:
                       #n_correct += 1
                #tgt1 = tgt[:, 1:]
                loss, n_correct = cal_performance(pred, tgt, smoothing=True,local_rank=device)
                
                #loss1, n_correct1 = cal_performance(pred1, tgt1, smoothing=True,local_rank=device)
                total_loss = total_loss + loss.item()#+ loss1.item()
                non_pad_mask = tgt.ne(0)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct
                #n_word_correct1 += n_correct1
                infer_time += forward_time
            fps = infer_time / batch_a * 1000
            loss_per_word = total_loss / data_len
            accuracy = n_word_correct / float(n_word_total)*100
            #accuracy = n_correct / float(batch_a)*100
            
            #accuracy1 = n_word_correct1 / float(n_word_total)*100
            #aaaaa = (f -(all - len_all))/ float(len_all)*100
            
            list_accuracy += f'{datasetlist[i]}: {accuracy:0.4f}\t'
            list_accuracy11 +=f'{datasetlist[i]}: {fps:0.4f}\t'
            # print("accuracy:{}".format(accuracy))
            avg_acc +=accuracy
            i = i + 1

    return loss_per_word, avg_acc/data_len,list_accuracy,list_accuracy11
haha=['<blank>','<unk>','<s>','</s>','0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def decode(text_index,idx2word):
    
    texts = []
    for p in range(text_index.shape[0]):
       text = ''.join([haha[i] for i in text_index[p, :]])
       
       texts.append(text)
    #for index, l in enumerate(length):
            #text = ''.join([haha[i] for i in text_index[index, :]])
            #texts.append(text)
    
    return texts
def average(model, models):
    """Average models into model"""
    # with torch.no_grad():
    #     for ps in zip(*[m.parameters() for m in [model] + models]):
    #         ps[0].copy_(torch.sum(torch.stack(ps[1:]), dim=0) / len(ps[1:]))

    with torch.no_grad():
        for key in model.state_dict().keys():
            v = []
            for m in models:
                v.append(m.state_dict()[key])
            v = torch.sum(torch.stack(v), dim=0) / len(v)
            model.state_dict()[key].copy_(v)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    headers = cfg.test.test_list
    
    if cfg.test.best_acc_test:
        path2 = glob.glob(cfg.test.model_dir + '/epoch9_*.pth')
        path = glob.glob(cfg.test.model_dir + 'epoch2_best_acc.pth')
        print("cfg.test.model_dir:",cfg.test.model_dir)
        for model_path in path2:
            print("model: {}".format(model_path))
            # eval(cfg, args,os.path.join(cfg.test.model_dir, model_path))
            a, b,list_accuracy,list_accuracy11 = eval(cfg, args, model_path)
            print("aaaaaa")
            with open(f'./saved_models/test_1_12_patch.txt', 'a') as log:
                 print("bbbbbb")
                 log.write(list_accuracy+'\n')
            with open(f'./saved_models/8.4_test_1_12_patch_a.txt', 'a') as log:
                   
                 log.write(list_accuracy11+'\n')
        for model_path in path:
            print("model: {}".format(model_path))
            a, b,list_accuracy,list_accuracy11 = eval(cfg, args, model_path)
            # eval(cfg, args,os.path.join(cfg.test.model_dir, model_path))
            with open(f'./11.17.txt', 'a') as log:
                 log.write(model_path+'\n')
                 log.write(list_accuracy+'\n')
                 log.write(list_accuracy11+'\n')
            #with open(f'./test_1_12_PD1_a.txt', 'a') as log:
                   
                 #log.write(list_accuracy11+'\n')
            #eval(cfg, args, model_path)
        # return




if __name__ == '__main__':
    main()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import sys
# sys.path.append(os.path.join(sys.path[0], '../'))
# sys.path.insert(1, '/home/songmeis/tianhao_byol/yth/lib/python3.7/site-packages')
import torch
import torch.nn as nn
# from dataset_emo import Dataiemocap,IEMOCAP_fold_datasets,IEMOCAP_fold_datasets_test
from datast_MSP_wav2vec import Dataiemocap, IEMOCAP_fold_datasets, IEMOCAP_fold_datasets_test
# from dataset_fused import Datafused, Fused_fold_datasets
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from model_MSP_RMSCNN_w2v2_test import EmotionModel, EmotionModel_w2v2rmsc_1D, EmotionModel_w2v2rmsc_1D_down
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from common import load_yaml_config
from utilities import create_folder, create_logging
import logging
import time
import random
import itertools
import torch.optim as optim
from sklearn.metrics import recall_score as recall
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.amp import GradScaler, autocast
import tensorflow as tf
# from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Wav2Vec2Config

tf.get_logger().setLevel(logging.ERROR)

logging.basicConfig(level=logging.DEBUG)


# device = torch.device('cuda')

# def loss_fnc(predictions, targets,device):
#     return nn.CrossEntropyLoss(weight=torch.tensor([10.6,17.0,3.1,1.8]).to(device))(input=predictions, target=targets)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_fnc(predictions, targets):
    return nn.MSELoss()(input=predictions, target=targets)


def cc_coef(output, target):
    mu_y_true = torch.mean(target)
    mu_y_pred = torch.mean(output)
    return 1 - 2 * torch.mean((target - mu_y_true) * (output - mu_y_pred)) / (
                torch.var(target) + torch.var(output) + torch.mean((mu_y_pred - mu_y_true) ** 2))


def evaluation_metrics(true_value, predicted_value):
    corr_coeff = np.corrcoef(true_value, predicted_value)
    ccc = 2 * predicted_value.std() * true_value.std() * corr_coeff[0, 1] / (
                predicted_value.var() + true_value.var() + (predicted_value.mean() - true_value.mean()) ** 2)
    return (ccc, corr_coeff)


def compute_uar(pred, gold):
    # reca = recall(gold, pred, 'macro')
    pred_Rsl_CCC = evaluation_metrics(gold, pred)[0]
    # return reca
    return pred_Rsl_CCC


def get_CI(data, bstrap):
    """

    :param data: [pred, groundtruth]
    :param bstrap:
    :return:
    """

    uars = []
    for _ in range(bstrap):
        idx = np.random.choice(range(len(data)), len(data), replace=True)
        samples = [data[i] for i in idx]
        sample_pred = [x[0] for x in samples]
        sample_groundtruth = [x[1] for x in samples]
        # sample_pred, sample_groundtruth = [data[i] for i in idx]
        sample_pred = np.array(sample_pred)
        sample_groundtruth = np.array(sample_groundtruth)
        uar = compute_uar(sample_pred, sample_groundtruth)
        uars.append(uar)

    lower_boundary_uar = pd.DataFrame(np.array(uars)).quantile(0.025)[0]
    higher_boundary_uar = pd.DataFrame(np.array(uars)).quantile(1 - 0.025)[0]

    return (higher_boundary_uar - lower_boundary_uar) / 2


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train():
    cfg = load_yaml_config('/home/yanlecun/w2v2-how-to/config_MSP_wav2vec.yaml')
    workspace = '/home/yanlecun/w2v2-how-to/workspace_emo'

    filename = cfg.filename
    learning_rate = cfg.lr
    batch_size = cfg.batch_size
    num_epochs = cfg.epochs
    sample_rate = cfg.sample_rate

    seed_everything(38)
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 'batch_size={}'.format(cfg.batch_size))

    create_folder(checkpoints_dir)

    logs_dir = os.path.join(workspace, 'checkpoints', filename, 'batch_size={}'.format(cfg.batch_size))

    create_logging(logs_dir, 'w')
    logging.info(cfg)

    df_files = pd.read_csv(cfg.root_files)
    emo_files_path = df_files['FileName']
    # asc_files_path = df_files['asc_file_path']
    # emo_files_label = df_files['Emo_onehot']
    emo_arouse_label = df_files['EmoAct']
    # emo_arouse_label = list(df_files['EmoAct'])
    emo_valence_label = df_files['EmoVal']
    emo_domaince_label = df_files['EmoDom']

    # Dataset
    # tfms = AugmentationModule((64, 300),len(files_path))
    tfms = None
    dataset_train = Dataiemocap(cfg, emo_files_path, emo_arouse_label, tfms=tfms, )
    dataset_test = Dataiemocap(cfg, emo_files_path, emo_arouse_label, tfms=tfms, )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # for fold in range(1,2):
    time_begin = time.time()
    logging.info('-' * 30)
    best_uar = 0

    train_dataset, valid_dataset = IEMOCAP_fold_datasets(dataset_train)
    test_dataset = IEMOCAP_fold_datasets_test(dataset_test)

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4,pin_memory=True,collate_fn=dataset_train.collator),
        'valid': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset_train.collator),
        'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=dataset_test.collator),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # model = ResNet50(emo_num_classes, channels=1).to(device)
    # model = TDRG(num_classes = emo_num_classes).to(device)
    # model = GLAM(shape=(62,40),).to(device)
    scaler = GradScaler()
    #  model_name = '/home/yanlecun/w2v2-how-to/w2v2_model'
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    # processor = Wav2Vec2Processor.from_pretrained(model_name)
    #  config = Wav2Vec2Config.from_pretrained(model_name)
    # config.num_labels = 1
    # config.final_dropout = 0.1
    # config.hidden_size = 1024
    model = EmotionModel_w2v2rmsc_1D_down.from_pretrained(model_name).to(device)
    num_params = count_parameters(model)
    print(f'The model has {num_params:,} trainable parameters')
    for name, param in model.named_parameters():
        if name.split('.')[0] == 'wav2vec2':
            param.requires_grad = False
        print(f"{name}: {param.requires_grad}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    logging.info('GPU number: {}'.format(torch.cuda.device_count()))

    test_emo_acc = 0
    for epoch in range(num_epochs):
        logging.info('*' * 10)
        logging.info('Epoch {}/{}, lr:{}'.format(epoch, num_epochs - 1, optimizer.param_groups[0]['lr']))
        logging.info('*' * 10)
        running_loss = 0.0
        epoch_emo_acc = 0
        epoch_loss = 0
        val_loss = 0
        val_emo_acc = 0
        val_emo_predictions = []
        val_emo_labels = []
        test_emo_predictions = []
        test_emo_labels = []
        # Each epoch is composed of training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # Iterate over data
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                if phase == 'train':
                    # sample_batched = sample_batched.cuda(rank)
                    inputs, emo_labels = sample_batched
                    # forward
                    optimizer.zero_grad()
                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        with autocast('cuda'):
                            emo_output_logits = model(inputs["input_values"].to(device), inputs["attention_mask"].to(device))[1].squeeze(1)
                            # emo_loss = cc_coef(emo_output_logits, emo_labels.to(device))
                            emo_loss = loss_fnc(emo_output_logits, emo_labels.float().to(device))
                        scaler.scale(emo_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        epoch_loss += emo_loss.item()

                        del inputs, emo_labels, emo_output_logits, emo_loss
                        torch.cuda.empty_cache()

                else:
                    with torch.no_grad():
                        inputs, emo_labels = sample_batched
                        emo_output_logits = model(inputs["input_values"].to(device), inputs["attention_mask"].to(device))[1].squeeze(1)
                        # emo_loss = cc_coef(emo_output_logits, emo_labels.to(device))
                        emo_loss = loss_fnc(emo_output_logits, emo_labels.float().to(device))

                        # emo_loss = np.mean(emo_loss)
                        val_loss += emo_loss.item()  # * inputs.size(0)
                        emo_output_logits = emo_output_logits.float().detach().cpu().tolist()
                        emo_labels = emo_labels.float().detach().cpu().tolist()

                        val_emo_predictions.append(emo_output_logits)
                        val_emo_labels.append(emo_labels)

                        del inputs, emo_labels, emo_output_logits, emo_loss
                        torch.cuda.empty_cache()

        val_emo_predictions_list = np.array(list(itertools.chain(*val_emo_predictions)))

        val_emo_labels_list = np.array(list(itertools.chain(*val_emo_labels)))

        pred_Rsl_CCC = evaluation_metrics(val_emo_labels_list, val_emo_predictions_list)[0]

        logging.info('Train_Emo_Loss: {:.4f}'.format(epoch_loss / len(dataloaders['train'])))
        logging.info('Valid_Emo_Loss: {:.4f},'.format(val_loss / len(dataloaders['valid'])))
        logging.info('Valid_Emo_CCC: {:.4f}'.format(pred_Rsl_CCC))

        if pred_Rsl_CCC > best_uar and pred_Rsl_CCC > 0.5:
            best_uar = pred_Rsl_CCC
            checkpoint_path = os.path.join(
                checkpoints_dir, 'SER_arouse_wav2vec2+RMSCNN_freeze_down_pretrain_bs8_model' + '_' + str(epoch) + '.pt')

            torch.save(model.state_dict(), checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            best_model = model

            best_model.eval()
            for i_batch, sample_batched in enumerate(dataloaders['test']):
                with torch.no_grad():
                    test_inputs, test_emo_label = sample_batched
                    test_emo_output_logits = best_model(test_inputs["input_values"].to(device), test_inputs["attention_mask"].to(device))[1].squeeze(1)

                    test_emo_prediction = test_emo_output_logits.float().detach().cpu().tolist()
                    test_emo_label = test_emo_label.float().detach().cpu().tolist()

                    test_emo_predictions.append(test_emo_prediction)
                    test_emo_labels.append(test_emo_label)
                    del test_inputs, test_emo_label, test_emo_output_logits
                    torch.cuda.empty_cache()

            test_emo_predictions_list = np.array(list(itertools.chain(*test_emo_predictions)))

            test_emo_labels_list = np.array(list(itertools.chain(*test_emo_labels)))

            pred_Rsl_CCC = evaluation_metrics(test_emo_labels_list, test_emo_predictions_list)[0]

            data = [[p, g] for p, g in zip(test_emo_predictions_list, test_emo_labels_list)]
            # a, b , c ,d = mean_confidence_interval(prediction, confidence=0.95)
            cis = get_CI(data, 1000)
            logging.info('confidence_interval:{}'.format(cis))

            # logging.info('Test_Emo_Acc: {:.4f}'.format(test_emo_acc / len(dataloaders['test'])))
            logging.info('-' * 10)
            logging.info('Test_Emo_arouse_wav2vec2+RMSCNN_freeze_down_pretrain_bs8_CCC: {:.4f}'.format(pred_Rsl_CCC))
            logging.info('-' * 10)

    # cleanup()
    time_elapsed = time.time() - time_begin
    logging.info('*' * 20)
    logging.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# def main():
#     world_size = 4
#     mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    train()
    print('_____________')
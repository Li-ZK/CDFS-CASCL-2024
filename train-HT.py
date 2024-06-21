import numpy as np
import os
import argparse
import pickle
import time
import imp
import logging
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from model.mapping import Mapping
from model.encoder import Encoder
from utils.dataloader import get_HBKC_data_loader, Task, get_target_dataset_houston, tagetSSLDataset
from utils import utils, loss_function, data_augment

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--config', type=str, default=os.path.join( './config', 'houston.py'))
args = parser.parse_args()

# load hyperparameters
config = imp.load_source("", args.config).config
train_opt = config['train_config']
data_path = config['data_path']
source_data = config['source_data']
target_data = config['target_data']
# houston
target_data_gt_train = config['target_data_gt_train']
target_data_gt_test = config['target_data_gt_test']
log_dir = config['log_dir']
patch_size = train_opt['patch_size']
batch_task = train_opt['batch_task']
emb_size = train_opt['d_emb']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']
N_DIMENSION = train_opt['n_dim']
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
EPISODE = train_opt['episode']
LEARNING_RATE = train_opt['lr']
GPU = config['gpu']
TAR_CLASS_NUM = train_opt['tar_class_num'] # the number of class
TAR_LSAMPLE_NUM_PER_CLASS = train_opt['tar_lsample_num_per_class'] # the number of labeled samples per class
WEIGHT_DECAY = train_opt['weight_decay']

utils.same_seeds(0)

# get src/tar class number -> label semantic vector
labels_src = ["water", "bare soil school", "bare soil park", "bare soil farmland", "natural plants", "weeds in farmland", "forest", "grass", "rice field grown", "rice field first stage", "row crops", "plastic house", "manmade non dark", "manmade dark", "manmade blue", "manmade red", "manmade grass", "asphalt"]

# IP
# labels_tar = ["Alfalfa", "Corn notill", "Corn mintill", "Corn", "Grass pasture", "Grass trees", "Grass pasture mowed", "Hay windrowed", "Oats", "Soybean notill", "Soybean mintill", "Soybean clean", "Wheat", "Woods", "Buildings Grass Trees Drives", "Stone Steel Towers"]

# houston
labels_tar = ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential", "Commercial", "Road", "Highway", "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"]

# salinas
# labels_tar = ["Brocoli green weeds 1", "Brocoli green weeds 2", "Fallow", "Fallow rough plow", "Fallow smooth", "Stubble", "Celery", "Grapes untrained", "Soil vinyard develop", "Corn senesced green weeds","Lettuce romaine 4wk", "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk" , "Vinyard untrained", "Vinyard vertical trellis"]

# longkou
# labels_tar = ["Corn", "Cotton", "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean", "Rice", "Water", "Roads and houses", "Mixed weed"]

from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('pretrain-model/bert-base-uncased')
model.eval()
tokenizer = BertTokenizer.from_pretrained('pretrain-model/bert-base-uncased')

encoded_inputs_src = tokenizer(labels_src, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs_src = model(**encoded_inputs_src)
semantic_mapping_src = outputs_src.last_hidden_state[:, 0, :]  # (num_classess, 768)

encoded_inputs_tar = tokenizer(labels_tar, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    outputs_tar = model(**encoded_inputs_tar)
semantic_mapping_tar = outputs_tar.last_hidden_state[:, 0, :]  # (num_classess, 768)

semantic_mapping_src = semantic_mapping_src.cpu().numpy()
semantic_mapping_tar = semantic_mapping_tar.cpu().numpy()

# load source domain data
with open(os.path.join(data_path, source_data), 'rb') as handle:
    source_imdb = pickle.load(handle)

data_train = source_imdb['data']
labels_train = source_imdb['Labels']

keys_all_train = sorted(list(set(labels_train)))
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
train_dict = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_dict:
        train_dict[label_encoder_train[class_]] = []
    train_dict[label_encoder_train[class_]].append(path)
del keys_all_train
del label_encoder_train

metatrain_data = utils.sanity_check(train_dict)

for class_ in metatrain_data:
    for i in range(len(metatrain_data[class_])):
        metatrain_data[class_][i] = np.transpose(metatrain_data[class_][i], (2, 0, 1))

# load target data
test_data = os.path.join(data_path,target_data)
test_label_train = os.path.join(data_path,target_data_gt_train)
test_label_test = os.path.join(data_path,target_data_gt_test)
# houston
Data_Band_Scaler, GroundTruth_train,  GroundTruth_test = utils.load_data_houston(test_data, test_label_train, test_label_test)

# loss init
crossEntropy = nn.CrossEntropyLoss().to(GPU)
cos_criterion = nn.CosineSimilarity(dim=1).to(GPU)

infoNCE_Loss = loss_function.ContrastiveLoss(batch_size = TAR_CLASS_NUM).to(GPU)
infoNCE_Loss_SSL = loss_function.ContrastiveLoss(batch_size = 128).to(GPU)
SupConLoss_t = loss_function.SupConLoss(temperature=0.1).to(GPU)

# experimental result index
nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TAR_CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None,None,None,None,None

seeds = [1222, 1230, 1333, 1337, 1339, 1231, 1233, 1234, 1212, 1213]

# log setting
experimentSetting = '{}way_{}shot_{}'.format(TAR_CLASS_NUM, TAR_LSAMPLE_NUM_PER_CLASS, target_data.split('/')[0])
utils.set_logging_config(os.path.join(log_dir, experimentSetting), nDataSet)
logger = logging.getLogger('main')
logger.info('seeds_list:{}'.format(seeds))

for iDataSet in range(nDataSet) :
    logger.info('emb_size:{}'.format(emb_size))
    logger.info('patch_size:{}'.format(patch_size))
    logger.info('seeds:{}'.format(seeds[iDataSet]))

    utils.same_seeds(seeds[iDataSet])

    #  load target domain data for training and testing
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain, target_aug_data_ssl, target_aug_label_ssl = get_target_dataset_houston(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth_train=GroundTruth_train,
        GroundTruth_test=GroundTruth_test,
        class_num=TAR_CLASS_NUM,
        tar_lsample_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        shot_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        patch_size=patch_size)

    target_ssl_dataset = tagetSSLDataset(target_aug_data_ssl, target_aug_label_ssl)
    target_ssl_dataloader = torch.utils.data.DataLoader(target_ssl_dataset, batch_size=64,shuffle=True, drop_last=True)

    num_supports, num_samples, query_edge_mask, evaluation_mask = utils.preprocess(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, batch_task, GPU)

    mapping_src = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    mapping_tar = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    encoder = Encoder(n_dimension=N_DIMENSION, patch_size=patch_size, emb_size=emb_size, dropout=0.7).to(GPU)

    mapping_src_optim = torch.optim.SGD(mapping_src.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    mapping_tar_optim = torch.optim.SGD(mapping_tar.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    encoder_optim = torch.optim.SGD(encoder.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    mapping_src.apply(utils.weights_init)
    mapping_tar.apply(utils.weights_init)
    encoder.apply(utils.weights_init)

    mapping_src.to(GPU)
    mapping_tar.to(GPU)
    encoder.to(GPU)

    mapping_src.train()
    mapping_tar.train()
    encoder.train()

    logger.info("Training...")
    last_accuracy = 0.0
    best_episode = 0
    total_hit_src, total_num_src, total_hit_tar, total_num_tar, acc_src, acc_tar = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    train_start = time.time()
    writer = SummaryWriter()

    target_ssl_iter = iter(target_ssl_dataloader)

    for episode in range(EPISODE) :
        # source and target few-shot learning
        task_src = Task(metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=False)

        task_tar = Task(target_da_metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=False)

        support_src, support_label_src = support_dataloader_src.__iter__().next()
        query_src, query_label_src = query_dataloader_src.__iter__().next()
        support_real_labels_src = task_src.support_real_labels
        support_real_labels_tar = task_tar.support_real_labels

        semantic_support_src = torch.zeros(TAR_CLASS_NUM, 768)
        for i, class_id in enumerate(support_real_labels_src) :
            semantic_support_src[i] = torch.from_numpy(semantic_mapping_src[class_id])

        semantic_support_tar = torch.zeros(TAR_CLASS_NUM, 768)
        for i, class_id in enumerate(support_real_labels_tar):
            semantic_support_tar[i] = torch.from_numpy(semantic_mapping_tar[class_id])

        support_tar, support_label_tar = support_dataloader_tar.__iter__().next()
        query_tar, query_label_tar = query_dataloader_tar.__iter__().next()

        support_features_src, semantic_feature_src = encoder(mapping_src(support_src.to(GPU)), semantic_feature=semantic_support_src.to(GPU), s_or_q = "support")
        query_features_src = encoder(mapping_src(query_src.to(GPU)))

        support_features_tar, semantic_feature_tar = encoder(mapping_tar(support_tar.to(GPU)), semantic_feature=semantic_support_tar.to(GPU), s_or_q = "support")
        query_features_tar = encoder(mapping_tar(query_tar.to(GPU)))

        if SHOT_NUM_PER_CLASS > 1:
            support_proto_src = support_features_src.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            support_proto_tar = support_features_tar.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
        else:
            support_proto_src = support_features_src
            support_proto_tar = support_features_tar

        logits_src = utils.euclidean_metric(query_features_src, support_proto_src)
        f_loss_src = crossEntropy(logits_src, query_label_src.long().to(GPU))

        logits_tar = utils.euclidean_metric(query_features_tar, support_proto_tar)
        f_loss_tar = crossEntropy(logits_tar, query_label_tar.long().to(GPU))

        f_loss = f_loss_src + f_loss_tar

        # cross-modal alignment loss
        text_align_loss = infoNCE_Loss(semantic_feature_src, support_features_src) + infoNCE_Loss(semantic_feature_tar, support_features_tar)

        # target domain supervised contrastive learning
        try:
            target_ssl_data, target_ssl_label = target_ssl_iter.next()
        except Exception as err:
            target_ssl_iter = iter(target_ssl_dataloader)
            target_ssl_data, target_ssl_label = target_ssl_iter.next()

        augment1_target_ssl_data = torch.FloatTensor(data_augment.random_mask_batch_image(target_ssl_data.data.cpu(), 0.2))
        augment2_target_ssl_data = torch.FloatTensor(data_augment.random_mask_batch_image(target_ssl_data.data.cpu(), 0.2))
        augment_target_ssl_data = torch.cat((augment1_target_ssl_data, augment2_target_ssl_data), dim=0)
        features_augment = encoder(mapping_tar(augment_target_ssl_data.to(GPU)))

        augment1_target_ssl_feature = F.normalize(features_augment[:len(target_ssl_data), :], dim = 1)
        augment2_target_ssl_feature = F.normalize(features_augment[len(target_ssl_data):, :], dim = 1)
        augment_target_ssl_feature = torch.cat([augment1_target_ssl_feature.unsqueeze(1), augment2_target_ssl_feature.unsqueeze(1)], dim=1)
        scl_loss_tar = SupConLoss_t(augment_target_ssl_feature, target_ssl_label)

        loss = f_loss + 2.0 * text_align_loss + scl_loss_tar

        mapping_src.zero_grad()
        mapping_tar.zero_grad()
        encoder.zero_grad()

        loss.backward()

        mapping_src_optim.step()
        mapping_tar_optim.step()
        encoder_optim.step()

        total_hit_src += torch.sum(torch.argmax(logits_src, dim=1).cpu() == query_label_src).item()
        total_num_src += query_src.shape[0]
        acc_src = total_hit_src / total_num_src

        total_hit_tar += torch.sum(torch.argmax(logits_tar, dim=1).cpu() == query_label_tar).item()
        total_num_tar += query_tar.shape[0]
        acc_tar = total_hit_tar / total_num_tar

        if (episode + 1) % 100 == 0:
            logger.info('episode: {:>3d}, f_loss: {:6.4f}, scl_loss_tar: {:6.4f}, text_align_loss: {:6.4f}, loss: {:6.4f}, acc_src: {:6.4f}, acc_tar: {:6.4f}'.format(
                episode + 1,
                f_loss.item(),
                scl_loss_tar.item(),
                text_align_loss.item(),
                loss.item(),
                acc_src,
                acc_tar))

            writer.add_scalar('Loss/f_loss', f_loss.item(), episode + 1)
            writer.add_scalar('Loss/scl_loss_tar', scl_loss_tar.item(), episode + 1)
            writer.add_scalar('Loss/text_align_loss', text_align_loss.item(), episode + 1)
            writer.add_scalar('Loss/loss', loss.item(), episode + 1)

            writer.add_scalar('Acc/acc_src', acc_src, episode + 1)
            writer.add_scalar('Acc/acc_tar', acc_tar, episode + 1)

        if (episode + 1) % 500 == 0 or episode == 0:
            with torch.no_grad():
                # test
                logger.info("Testing ...")
                train_end = time.time()
                mapping_tar.eval()
                encoder.eval()
                total_rewards = 0
                counter = 0
                accuracies = []
                predict = np.array([], dtype=np.int64)
                predict_gnn = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)

                train_datas, train_labels = train_loader.__iter__().next()

                support_real_labels = train_labels

                semantic_support = torch.zeros(TAR_CLASS_NUM*TAR_LSAMPLE_NUM_PER_CLASS, 768)
                for i, class_id in enumerate(support_real_labels):
                    semantic_support[i] = torch.from_numpy(semantic_mapping_tar[class_id])

                train_features, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)), semantic_feature = semantic_support.to(GPU),  s_or_q = "support")

                max_value = train_features.max()
                min_value = train_features.min()
                print(max_value.item())
                print(min_value.item())
                train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

                KNN_classifier = KNeighborsClassifier(n_neighbors=1)
                KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_features = encoder(mapping_tar((Variable(test_datas).to(GPU))))

                    test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                    predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                    test_labels = test_labels.numpy()
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                test_accuracy = 100. * total_rewards / len(test_loader.dataset)
                writer.add_scalar('Acc/acc_test', test_accuracy, episode + 1)

                logger.info('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset), 100. * total_rewards / len(test_loader.dataset)))
                test_end = time.time()

                # Training mode
                mapping_tar.train()
                encoder.train()
                if test_accuracy > last_accuracy:
                    last_accuracy = test_accuracy
                    best_episode = episode
                    acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                    OA = acc
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
                    best_predict_all = predict
                    best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                logger.info('best episode:[{}], best accuracy={}'.format(best_episode + 1, last_accuracy))

    logger.info('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episode + 1, last_accuracy))
    logger.info ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
    logger.info("accuracy list: {}".format(acc))
    logger.info('***********************************************************************************')

OAMean = np.mean(acc)
OAStd = np.std(acc)

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)

kMean = np.mean(k)
kStd = np.std(k)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

logger.info ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
logger.info ("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
logger.info ("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format( OAStd))
logger.info ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
logger.info ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
logger.info ("accuracy list: {}".format(acc))
logger.info ("accuracy for each class: ")
for i in range(TAR_CLASS_NUM):
    logger.info ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


#################classification map################################
for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[i]]][best_Column[best_RandPerm[i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0.77, 0.87, 0.7]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0.43, 0.67, 0.27]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0.32, 0.50, 0.2]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [0.21, 0.34, 0.13]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [0.77, 0.35, 0.06]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [0, 0.69, 0.94]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.75, 0, 0]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.7, 0.78, 0.9]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.48, 0.48, 0.48]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.95, 0.69, 0.51]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.97, 0.79, 0.67]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.34, 0.34, 0.34]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.8, 0.8, 0]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0.8, 0.4]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [1, 0, 0]

halfwidth = patch_size // 2
utils.classification_map(hsi_pic[halfwidth:-halfwidth, halfwidth:-halfwidth, :], best_G[halfwidth:-halfwidth, halfwidth:-halfwidth], 24,  "classificationMap/houston_{}shot.png".format(TAR_LSAMPLE_NUM_PER_CLASS))







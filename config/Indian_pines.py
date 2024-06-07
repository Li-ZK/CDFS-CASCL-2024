from collections import OrderedDict

config = OrderedDict()
config['data_path'] = 'datasets'
config['source_data'] = 'Chikusei_imdb_128_7_7.pickle'
config['target_data'] = 'IP/indian_pines_corrected.mat'
config['target_data_gt'] = 'IP/indian_pines_gt.mat'
config['gpu'] = 0

config['log_dir'] = './logs'

train_opt = OrderedDict()
train_opt['patch_size'] = 7
train_opt['batch_task'] = 1
train_opt['num_ways'] = 5
train_opt['num_shots'] = 1
train_opt['episode'] = 5000
train_opt['lr'] = 1e-3
train_opt['weight_decay'] = 1e-4

train_opt['d_emb'] = 128
train_opt['src_input_dim'] = 128
train_opt['tar_input_dim'] = 200
train_opt['n_dim'] = 100

train_opt['shot_num_per_class'] = 1
train_opt['query_num_per_class'] = 19

train_opt['tar_class_num'] = 16
train_opt['tar_lsample_num_per_class'] = 5

config['train_config'] = train_opt


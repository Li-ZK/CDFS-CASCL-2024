# CDFS-CASCL-2024
This is a code demo for the paper "Cross-Domain Few-shot Hyperspectral Image Classification with Cross-Modal Alignment and Supervised Contrastive Learning".


## Requirements

- CUDA = 12.2

- python = 3.9.18 

- torch = 1.11.0+cu113 

- transformers = 4.30.2

- sklearn = 0.0.post9

- numpy = 1.26.0

## Datasets

- source domain dataset
  - Chikusei

- target domain datasets
  - Indian Pines
  - Houston
  - Salinas
  - WHU-Hi-LongKou

You can download the source and target datasets mentioned above at https://pan.baidu.com/s/1wo9xj85YaT3JGogVyJKZTQ?pwd=5lkl, and move to folder `datasets`.  In particular, for the source dataset Chikusei, you can choose to download it in mat format, and then use the utils/chikusei_imdb_128.py file to process it to get the patch size you want, or directly use the preprocessed source dataset Chikusei_imdb_128_7_7.pickle with a patch size of 7 $\times$ 7. 

An example datasets folder has the following structure:

```
datasets
├── Chikusei_imdb_128_7_7.pickle
├── Chikusei_raw_mat
│   ├── HyperspecVNIR_Chikusei_20140729.mat
│   └── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
├── IP
│   ├── indian_pines_corrected.mat
│   └── indian_pines_gt.mat
├── Houston
│   ├── data.mat
│	├── mask_train.mat
│   └── mask_test.mat
├── salinas
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
└── WHU-Hi-LongKou
    ├── WHU_Hi_LongKou.mat
    └── WHU_Hi_LongKou_gt.mat
```

## Pretrain model

You can download the pre-trained model of Base Bert, bert-base-uncased, at https://pan.baidu.com/s/1C6qExEcVd3foNtLcn7PKFw?pwd=enda, and move to folder `pretrain-model`.

An example pretrain-model folder has the following structure:

```
pretrain-model
└── bert-base-uncased
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

## Usage

1. Download the required source and target datasets and move to folder `datasets`.

- If you down the source domain dataset (Chikusei) in mat format, you need to run the script `Chikusei_imdb_128.py` to generate preprocessed source domain data. 
- If you downloaded Chikusei_imdb_128_7_7.pickle, move it directly to the corresponding dataset directory.

2. Download the required Base Bert pre-trained model and move to folder `pretrain-model`.
3. Run `CDFS-CASCL.py`. 

# Bone Suppression from Chest Radiographs

The project is a tool to build **Bone Suppression** model, written in tensorflow

<img src="description.png" alt="CAM example image"/>

##AutoEncoding
[paper](https://www.researchgate.net/publication/320252756_Deep_learning_models_for_bone_suppression_in_chest_radiographs?enrichId=rgreq-7b19be48d9763ea61b22252eaf96edca-XXX&enrichSource=Y292ZXJQYWdlOzMyMDI1Mjc1NjtBUzo1ODQ1MzY0NDY0ODAzODRAMTUxNjM3NTc1NzU5Nw%3D%3D&el=1_x_3&_esc=publicationCoverPdf)

This code is based off of two models described in this paper. The first is an autoencoder-like model of CNNs that shrinks the image down before deconding it with mirrored weights. The second is a family of
CNN's that keeps the image size the same.

##Data Set
I trained the model on the JRT dataset and the associated bone free images in the BJRT data set.

## In this project you can
1. Preprocessing data, including registration and augmentation.
2. Train/test by following the quickstart. You can get a model with performance close to the paper.
3. Visualize your training result with tensorboard

## Requirements
The project requires `Python>=3.5`.


### [TRAIN](config/train.cfg)
1. `source_folder` and `target_folder` are folders to load training images.
4. If you want to continue training from your last model, set `use_trained_model` to true and `trained_model` to your model path.
5. `output_model` is where you save your model during training and `output_log` is where you save the tensorboard checkpoints.
6. The other parameters is set following the published [paper](https://www.researchgate.net/publication/320252756_Deep_learning_models_for_bone_suppression_in_chest_radiographs?enrichId=rgreq-7b19be48d9763ea61b22252eaf96edca-XXX&enrichSource=Y292ZXJQYWdlOzMyMDI1Mjc1NjtBUzo1ODQ1MzY0NDY0ODAzODRAMTUxNjM3NTc1NzU5Nw%3D%3D&el=1_x_3&_esc=publicationCoverPdf)

## Pretrained model
If you want to start testing without training from scratch, you can use the [model](/model) I have trained. The model has loss value: 0.01409, MSE: 7.1687e-4, MS-SSIM: 0.01517

## Quickstart
**Note that currently this project can only be executed in Linux and macOS. You might run into some issues in Windows.**
1. Create & activate a new python3 virtualenv. (optional)
2. Install dependencies by running `pip install -r requirements.txt`.
3. Run `python preprocessing.py` to preprocess dataset. If you want to change your config path:
```
python preprocessing.py --config <config path>
```
4. Run `python train.py` to train a new model. If you want to change your config path:
```
python train.py --config <config path>
```
During training, you can use Tensorboard to visualize the results:
```
tensorboard --logdir=<output_log in train.cfg>
```
5. Run `python master.py` to augment the data set, split the set into test and training sets, train the model and test the model.
To change default parameters, you can use:
```
```

## Acknowledgement
I would like to thank [LoudeNOUGH](https://github.com/LoudeNOUGH/bone-suppression) for scratch training script and Hussam Habbreeh (حسام هب الريح) for sharing his experiences on this task.

## Author
Chuong M. Huynh (minhchuong.itus@gmail.com)

## License
MIT

from __future__ import division
from configparser import ConfigParser
import argparse
from model import AELikeModel
from preprocessing import split_train

def main(args):
    # parser config
    # see train.cfg for parameters


    cp = ConfigParser()
    cp.read(args.train)
    cp.read(args.augmentation)
    cp.read(args.test)
    image_size = cp["TRAIN"].getint("image_size")
    alpha = cp["TRAIN"].getfloat("alpha")

    # takes unified x at "source folder", y "target folder" and splits into training/ testing batches at "split output dir/test /train" 
    if split:
        
        split_train(cp["SPLIT"].get("source_folder"), cp["SPLIT"].get("target_folder"), cp["SPLIT"].get("split_output_dir") )
        


    if train:
        cp.read(args.train)
        x_train_folder= cp["TRAIN"].get("source_folder")
        y_train_folder = cp["TRAIN"].get("target_folder")
        
        # use_trained_model = cp["TRAIN"].getboolean("use_trained_model")
        epochs = cp["TRAIN"].getint("epochs")
        train_steps = cp["TRAIN"].getint("train_steps")
        learning_rate = cp["TRAIN"].getfloat("learning_rate")
        epochs_to_reduce_lr = cp["TRAIN"].getint("epochs_to_reduce_lr")
        reduce_lr = cp["TRAIN"].getfloat("reduce_lr")
        output_model = cp["TRAIN"].get("output_model")
        output_log = cp["TRAIN"].get("output_log")
        batch_size = cp["TRAIN"].getint("batch_size")
        verbose = cp["TRAIN"].getboolean("verbose")

        model = AELikeModel(image_size, alpha, AE = True, verbose=True, trained_model = None)
        model.train(x_train_folder, y_train_folder, epochs, train_steps, learning_rate, epochs_to_reduce_lr, reduce_lr, output_model, output_log, batch_size)

        # loads previously trained model
        # AE = True uses Autoencoder architecuture
    elif test:
        cp.read(args.test)
        ModelPath = cp["TEST"].get("output_model")
        model = AELikeModel(image_size, alpha, AE = True, verbose=True, trained_model = ModelPath)

    if test:
        x_test_folder = cp["TEST"].get("source_folder")
        y_test_folder = cp["TEST"].get("target_folder")
        cp.read(args.test)
        print(y_test_folder)
        results = model.test(x_test_folder,y_test_folder,y_test_folder+"/results/")
        print("Mean Squared Error"+ str(results["MSE"]))       
        # print("Average SSIM",results["SSIM"])
        # print("combined", results["cost"])



    # Parse arguments


    # # Training
    # trained_model = None
    # if use_trained_model:
    #     trained_model = cp["TRAIN"].get("trained_model")
    # print('into the model')
    # model = AELikeModel(image_size, alpha, AE = False, verbose, trained_model)
    # model.train(source_folder, target_folder, epochs, train_steps, learning_rate, epochs_to_reduce_lr, reduce_lr, output_model, output_log, batch_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bone Suppression Testing')

    split = False
    augment = False
    train = False
    test = True

    parser.add_argument('--train', default='config/train.cfg', type=str, help='train config file')
    parser.add_argument('--augmentation', default='config/data_preprocessing.cfg', type=str, help='augment config file')
    parser.add_argument('--test', default='config/test.cfg', type=str, help='test config file')

    # if  split:

    #     parser.add_argument('--train', default='config/train.cfg', type=str, help='train config file')

    # if augment:
    #     parser = argparse.ArgumentParser(description='bakerco - BoneSuppression v1 - Training')
    #     parser.add_argument('--augmentation', default='config/data_preprocessing.cfg', type=str, help='augment config file')

    # if train: 
    #     parser = argparse.ArgumentParser(description='bakerco - BoneSuppression v1 - Training')
    #     parser.add_argument('--train', default='config/train.cfg', type=str, help='train config file')

    # if test:
    #     parser = argparse.ArgumentParser(description='bakerco - BoneSuppression v1 - Training')
    #     parser.add_argument('--test', default='config/test.cfg', type=str, help='test config file')


    args = parser.parse_args()
    main(args)
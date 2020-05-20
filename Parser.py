import argparse

def parse_all_args():
    # Parses commandline arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("-tr",type=str,
                help="The root for the training dataset (str) [default: ../train", default="../train")
    parser.add_argument("-va",type=str,
                help="The root for the validation dataset (str) [default: ../valid", default="../valid")
    parser.add_argument("-te",type=str,
                help="The root for the testing dataset (str) [default: ../test", default="../test")
    parser.add_argument("-exp",type=str,
                help="The root to store models (str) [default: ../models", default="../models")
    parser.add_argument("-lr",type=float,
                help="The learning rate (float) [default: 0.001]", default=0.001)
    parser.add_argument('-lrS',type=int,
                help='StepLR learning rate scheduler step, default=20', default=20)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (int) [default: 32]",default=32)
    parser.add_argument("-epochs",type=int,
                help="The number of training epochs (int) [default: 100]", default=100)
    parser.add_argument('-its', type=int,
                help='number of episodes per epoch, default=100', default=100)
    parser.add_argument('-cTr', type=int,
                help='number of random classes per episode for training, default=60', default=60)
    parser.add_argument('-nsTr', type=int,
                help='number of samples per class to use as support for training, default=5', default=5)
    parser.add_argument('-nqTr', type=int,
                help='number of samples per class to use as query for training, default=5', default=5)
    parser.add_argument('-cVa', type=int,
                help='number of random classes per episode for validation, default=5', default=5)
    parser.add_argument('-nsVa', type=int,
                help='number of samples per class to use as support for validation, default=5', default=5)
    parser.add_argument('-nqVa', type=int,
                help='number of samples per class to use as query for validation, default=15', default=15)

    return parser.parse_args()
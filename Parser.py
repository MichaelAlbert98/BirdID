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
    parser.add_argument("-its",type=int,
                help="The number of training iterations (int) [default: 100]", default=100)
    parser.add_argument('-eps', type=int,
                help='number of episodes per iteration, default=100', default=100)
    parser.add_argument('-n', type=int,
                help='number of classes to use, default=5', default=5)
    parser.add_argument('-k', type=int,
                help='number of examples per class, default=5', default=5)
    return parser.parse_args()
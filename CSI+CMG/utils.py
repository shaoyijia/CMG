import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device for training')
    parser.add_argument('--params-dict-name', type=str,
                        help='name of the classifier checkpoint file')
    parser.add_argument('--params-dict-name2', type=str,
                        help='name of the CVAE checkpoint file')
    return parser.parse_args()

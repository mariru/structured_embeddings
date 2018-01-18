import argparse

def parse_args():
        parser = argparse.ArgumentParser(description="run exponential family embeddings on text")

        parser.add_argument('--K', type=int, default=100,
                            help='Number of dimensions. Default is 100.')

        parser.add_argument('--H0', type=int, default=5,
                            help='Number of units in modulation network. Default is 5.')

        parser.add_argument('--sig', type=float, default = 10.0,
                            help='Regularization on global embeddings. Default is 10.')

        parser.add_argument('--n_iter', type=int, default = 1,
                            help='Number of passes over the data. Default is 1.')

        parser.add_argument('--n_epochs', type=int, default=100,
                            help='Number of epochs. Default is 100.')

        parser.add_argument('--cs', type=int, default=4,
                            help='Context size. Default is 4.')

        parser.add_argument('--ns', type=int, default=10,
                            help='Number of negative samples. Default is 10. You will get a difficult to read tensorflow error if you use more negative samples than you generated in step 4 of the data preparation.')

        parser.add_argument('--hierarchical', type=bool, default=False,
                            help='hierarchical embedding model. Default is False.')

        parser.add_argument('--amortized', type=bool, default=False,
                            help='amortized embedding model. Default is False.')

        parser.add_argument('--resnet', type=bool, default=False,
                            help='Use resnet architecture for amortization. Default is False.')

        parser.add_argument('--debug', type=bool, default=False,
                            help='Debug mode (10th of data). Default is False.')

        parser.add_argument('--init', type=str, default='',
                            help='Folder name to load variational.dat for initialization. Default is \'\' for no initialization')

        parser.add_argument('--fpath', type=str, default='../dat/lorem_ipsum/',
                            help='path to data')

        args =  parser.parse_args()
        print(args)
        return args

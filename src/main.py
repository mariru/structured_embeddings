from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from args import *
from data import *
from models import *
from utils import *

args = parse_args()

d = bern_emb_data(args.cs, args.ns, args.fpath, 
                 (args.hierarchical or args.amortized ), 
                 args.n_epochs, args.debug)

dir_name = make_dir(d.name)

m = define_model(args, d, dir_name)

m.initialize_training()

m.train_embeddings()

m.evaluate_embeddings()

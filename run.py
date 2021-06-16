#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    run.py train [options]
    run.py predict [options]

Options:
    -h --help                           show this screen.
    
    --seed=<int>                        seed
"""

from docopt import docopt
import numpy as np

def main():
    args = docopt(__doc__)
    print(args,'\n')

    # seed the random number generators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = int(args['--seed'])
    
#    if args['train']:
#        train(args, device)
#    elif args['predict']:
#        predict(args)
#    else:
#        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()


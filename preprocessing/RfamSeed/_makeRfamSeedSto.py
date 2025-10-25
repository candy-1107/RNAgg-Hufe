# -*- coding: utf-8 -*-

import os
import sys
import argparse
sys.path.append(os.environ['HOME'] + "/pyscript")
#sys.path.append("/home/terai/pyscript")
#import basic
#from Bio import SeqIO
#from Bio.SeqRecord import SeqRecord
#import re
#import csv
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn import svm
#from scipy.stats import pearsonr
#import RNA

def main(args: dict):
    with open(args.seed, encoding='iso-8859-1') as f:
        data = []
        fam_id = None
        for line in f:
            line = line.replace('\n','') 
            if line.startswith('#=GF AC'):
                fam_id = line.split()[2].strip()
            if line.startswith('//'):
                data.append(line)
                if fam_id == args.fam_id:
                    if args.out_stk:
                        with open(args.out_stk, 'w', encoding='utf-8') as fout:
                            for l in data:
                                fout.write(l + '\n')
                    else:
                        for l in data:
                            print(l)
                data = []
            else:
                data.append(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fam_id', help='family id')
    parser.add_argument('seed', help='Rfam.seed file')
    parser.add_argument('--out_stk', help='output stockholm file', default=None)
    args = parser.parse_args()
    main(args)

from warnings import filterwarnings
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
filterwarnings(action='ignore')

import pretty_errors

from main_dir.mask_script import Test

plot_matrix = input("\nDisplay Confusion Matrix? [Y/n] : ").strip().lower()
if plot_matrix in {'n', 'no'}:
    plot_matrix = False
else:
    plot_matrix = True
   
tester = Test(batch_size=16)
tester.test(con_matrix=plot_matrix)
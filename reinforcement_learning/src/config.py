
import numpy as np 

config = {'discount':  np.linspace(0.1, 0.99, 10, endpoint=True),
          'ql_iters': [30000],
          'ql_alpha':  np.linspace(0.1, 0.9, 3, True),       #just one learning rate
          'ql_epsilon': np.linspace(0.1, 1, 3, True),
          'ql_discount': [0.99],
          'ql_params': {'forest': {'S':32},
                        'random': {'S':2048, 'A': 16}
          },
          'ql_hyper_params': {'forest': {'alpha': 0.9, 'epsilon':0.1},
                              'random': {'alpha': 0.9, 'epsilon':1.0}
                              },
          
          's_ranges': {'forest': [2**i for i in range(1,7)],
                       'random': [2**i for i in range(6,12)]},
          'problems': {'forest': None,
                      'random': None
                     },

          }

from pylearn2.config import yaml_parse

train = open('conv.yaml', 'r').read()
train_params = {'train_stop': 50000,
                    'valid_stop': 60000,
                    'test_stop': 10000,
                    'batch_size': 100,
                    'output_channels_h2': 16, 
                    'output_channels_h3': 64,  
                    'max_epochs': 500,
                    'save_path': '.'}
train = train % (train_params)
train = yaml_parse.load(train)
train.main_loop()

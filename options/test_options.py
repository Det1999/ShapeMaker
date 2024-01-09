import argparse

import yaml


class TestOptions:
    def __init__(self):
        self.initialized = False
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        ###############################################################################################
        #                                   common settings                                           #
        ###############################################################################################
        self.parser.add_argument("--full_check_name", type=str, default='', help="*****.pth")
        self.parser.add_argument("--partial_check_name", type=str, default='', help="*****.pth")
        self.parser.add_argument("--rd_check_name", type=str, default='', help="*****.pth")
        self.parser.add_argument("--test_batch", type=int, default=3, help="batch_size for testing")
        self.parser.add_argument("--save_dir", type=str, default='test', help="the path ave test result")
        self.parser.add_argument("--top_k", type=int, default=10, help="topk nearst source shape")
        self.parser.add_argument("--use_partial", type=bool, default=False, help="whecher use partial")
        self.parser.add_argument("--load_dir", type=bool, default=False, help="root path of model's weight file")
        self.initialized = True

    def combine_configs(self, combine=True, file_path=None):
        self.initialize()
        opt = self.parser.parse_args()
        if combine is False:
            return opt
        with open(file_path, 'r') as yaml_file:
            cfg = yaml.safe_load(yaml_file)
        
        opt.full_check_name = cfg['full_check_name']
        opt.partial_check_name = cfg['partial_check_name']
        opt.rd_check_name = cfg['rd_check_name']
        opt.test_batch = cfg['test_batch']
        opt.save_dir = cfg['save_dir']
        opt.top_k = cfg['top_k']
        opt.use_partial = cfg['use_partial']
        opt.load_dir = cfg['load_dir']
        
        return opt

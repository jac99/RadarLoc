import os
import configparser
import time

from datasets.quantization import PolarQuantizer, CartesianQuantizer


class ModelParams:
    def __init__(self, model_params_path):
        assert os.path.exists(model_params_path), f"Cannot access model config file: {model_params_path}"
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.output_dim = params.getint('output_dim', 256)      # Size of the final descriptor

        self.lidar_model = params.get('lidar_model', None)
        if self.lidar_model is not None:
            self.lidar_coordinates = params.get('lidar_coordinates', 'polar')
            assert self.lidar_coordinates in ['polar', 'cartesian'], f'Unsupported coordinates: {self.lidar_coordinates}'

            if 'polar' in self.lidar_coordinates:
                # 3 quantization steps for polar coordinates: for sectors (in degrees), rings (in meters) and z coordinate (in meters)
                self.lidar_quantization_step = [float(e) for e in params['lidar_quantization_step'].split(',')]
                assert len(self.lidar_quantization_step) == 3, f'Expected 3 quantization steps: for sectors (degrees), rings (meters) and z coordinate (meters)'
                self.lidar_quantizer = PolarQuantizer(quant_step=self.lidar_quantization_step)
            elif 'cartesian' in self.lidar_coordinates:
                # Single quantization step for cartesian coordinates
                self.lidar_quantization_step = params.getfloat('lidar_quantization_step')
                self.lidar_quantizer = CartesianQuantizer(quant_step=self.lidar_quantization_step)
            else:
                raise NotImplementedError(f"Unsupported coordinates: {self.lidar_coordinates}")
        self.radar_model = params.get('radar_model', None)

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e == 'quantization_step':
                s = param_dict[e]
                if self.lidar_coordinates == 'polar':
                    print(f'quantization_step - sector: {s[0]} [deg] / ring: {s[1]} [m] / z: {s[2]} [m]')
                else:
                    print(f'quantization_step: {s} [m]')
            else:
                print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class TrainingParams:
    """
    Parameters for model training
    """
    def __init__(self, params_path, model_params_path):
        """
        Configuration files
        :param path: Training configuration file
        :param model_params: Model-specific configuration file
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.dataset = params.get('dataset', 'mulran').lower()
        self.dataset_folder = params.get('dataset_folder')

        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        # Initial batch size for global descriptors (for both main and secondary dataset)
        self.batch_size = params.getint('batch_size', 64)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)

        scheduler_milestones = params.get('scheduler_milestones')
        self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.loss = params.get('loss')

        if 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.2)    # Margin used in loss function
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)
        self.test_file = params.get('test_file')

        # Read model parameters
        self.model_params = ModelParams(self.model_params_path)
        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e != 'model_params':
                print('{}: {}'.format(e, param_dict[e]))

        self.model_params.print()
        print('')


from layers.lidar.eca import ECABasicBlock
from models.lidar.minkg import MinkHead, MinkTrunk, MinkG
from misc.utils import ModelParams
from models.radar.radar_net import RadarNet
import layers.radar.pooling as image_pooling
from models.radar.cylindrical_resnet_FPN import FPNFeatureExtractor
from models.radar.baseline_nets import BaselineModel1


def model_factory(model_params: ModelParams):
    assert model_params.radar_model is None or model_params.lidar_model is None, "radar_model and lidar_model cannot be used at the same time"

    if model_params.radar_model is not None:
        net = create_radar_model(model_params)
    elif model_params.lidar_model is not None:
        net = create_lidar_model(model_params)
    else:
        raise NotImplementedError("No model defined")

    return net


def create_lidar_model(model_params: ModelParams):
    model_name = model_params.lidar_model

    if model_name == 'minkloc':
        block = ECABasicBlock
        planes = [32, 64, 64, 128, 128, 128, 128]
        layers = [1, 1, 1, 1, 1, 1, 1]

        global_in_levels = [5, 6, 7]
        global_map_channels = 256
        global_normalize = False
    else:
        raise NotImplementedError(f'Unknown lidar-based model: {model_params.lidar_model}')

    # CREATE LIDAR NET
    # Planes list number of channels for level 1 and above
    global_in_channels = [planes[i-1] for i in global_in_levels]
    head_global = MinkHead(global_in_levels, global_in_channels, global_map_channels)
    min_out_level = min(global_in_levels)
    trunk = MinkTrunk(in_channels=1, planes=planes, layers=layers, conv0_kernel_size=5, block=block,
                      min_out_level=min_out_level)
    lidar_net = MinkG(trunk, global_head=head_global, global_pool_method='GeM', global_normalize=global_normalize)

    return lidar_net


def create_radar_model(model_params: ModelParams):
    model_name = model_params.radar_model

    if model_name == 'vgg16netvlad':
        radar_net = BaselineModel1()
    elif model_name == 'radarloc':
        # OUR BASE MODEL
        radar_fe = FPNFeatureExtractor(planes=[32, 64, 128, 256], layers=[1, 1, 1, 1], min_out_level=3,
                                       out_channels=256, cylindrical_padding=True)
        radar_pooling = image_pooling.GeM()
        radar_net = RadarNet(radar_fe, pooling=radar_pooling, normalize=False)
    else:
        raise NotImplementedError(f'Unknown radar-based model: {model_params.radar_model}')

    return radar_net


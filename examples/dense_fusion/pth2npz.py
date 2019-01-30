import argparse
import numpy as np

import chainer
import torch

from chainer_dense_fusion.links.model import PoseNet


def copy_layer(layer, params, param_name, param_type):
    if param_type == 'weight':
        layer.W.array[:] = np.asarray(params[param_name])
    elif param_type == 'bias':
        layer.b.array[:] = np.asarray(params[param_name])
    else:
        raise ValueError(
            'param: {} is not supported'.format(param_name))


def torch2chainer(model, params):
    rtc_dict = {'r': 'rot', 't': 'trans', 'c': 'conf'}
    emb_dict = {'e_conv': 'img', 'conv': 'pcd'}
    param_names = list(params.keys())
    finished_param_names = []
    for param_name in param_names:
        # pose net conv
        if param_name.startswith('conv'):
            param_layer_name, param_type = param_name.split('.')
            layer_num = int(param_layer_name[4])
            layer_type = rtc_dict[param_layer_name[-1]]
            layer = getattr(model, 'conv{}_{}'.format(layer_num, layer_type))
            copy_layer(layer, params, param_name, param_type)
            finished_param_names.append(param_name)
        # pose net extractor
        elif param_name.startswith('feat'):
            param_layer_name, param_type = param_name.split('.')[1:]
            layer_num = int(param_layer_name[-1])
            if layer_num in [1, 2]:
                layer_type = emb_dict[param_layer_name[:-1]]
                layer_name = 'conv{}_{}'.format(layer_num, layer_type)
            elif layer_num in [5, 6]:
                layer_name = 'conv{}'.format(layer_num - 2)
            else:
                raise ValueError(
                    'param: {} is not supported'.format(param_name))
            layer = getattr(model.posenet_extractor, layer_name)
            copy_layer(layer, params, param_name, param_type)
            finished_param_names.append(param_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', type=str, default=None)
    parser.add_argument('pthfile', type=str)
    args = parser.parse_args()

    chainer_model = PoseNet(
        n_fg_class=21, n_point=1000)
    torch_params = torch.load(args.pthfile)
    torch2chainer(chainer_model, torch_params)
    if args.out is None:
        outpath = 'pose_net_ycb_converted.npz'
    else:
        outpath = args.out
    chainer.serializers.save_npz(outpath, chainer_model)


if __name__ == '__main__':
    main()

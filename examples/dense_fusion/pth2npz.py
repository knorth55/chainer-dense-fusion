import argparse
import warnings

import chainer
import torch

from chainer_dense_fusion.links.model import DenseFusion

from pth2npz_posenet import copy_layer
from pth2npz_posenet import posenet_torch2chainer


def refiner_torch2chainer(model, params):
    rtc_dict = {'r': 'rot', 't': 'trans', 'c': 'conf'}
    emb_dict = {'e_conv': 'img', 'conv': 'pcd'}
    param_names = list(params.keys())
    uncopied_param_names = param_names.copy()
    for param_name in param_names:
        # pose net conv
        if param_name.startswith('conv'):
            param_layer_name, param_type = param_name.split('.')
            layer_num = int(param_layer_name[4])
            layer_type = rtc_dict[param_layer_name[-1]]
            layer = getattr(model, 'conv{}_{}'.format(layer_num, layer_type))
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)
        # pose refine net extractor
        if param_name.startswith('feat'):
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
            layer = getattr(model.extractor, layer_name)
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)

    if len(uncopied_param_names) > 0:
        warnings.warn('Some params are not copied:')
        for param_name in uncopied_param_names:
            print('param_name: {} not copied'.format(param_name))
    else:
        print('All valid parameter are copied.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', type=str, default=None)
    parser.add_argument('estimator_pthfile', type=str)
    parser.add_argument('refiner_pthfile', type=str)
    args = parser.parse_args()

    chainer_model = DenseFusion(
        n_fg_class=21, n_point=1000)
    print('start loading estimator params')
    estimator_torch_params = torch.load(args.estimator_pthfile)
    print('finish loading estimator params')
    print('start copying estimator params')
    posenet_torch2chainer(
        chainer_model.posenet, estimator_torch_params)
    print('finish copying estimator params')
    print('start loading estimator params')
    refiner_torch_params = torch.load(args.refiner_pthfile)
    print('finish loading refiner params')
    print('start copying refiner params')
    refiner_torch2chainer(
        chainer_model.refiner, refiner_torch_params)
    print('finish copying refiner params')
    if args.out is None:
        args.out = 'dense_fusion_ycb_converted.npz'

    print('start saving npz')
    chainer.serializers.save_npz(args.out, chainer_model)
    print('finish saving npz')


if __name__ == '__main__':
    main()

import argparse
import numpy as np

import chainer
import torch

from chainer_dense_fusion.links.model import PoseNet


def copy_layer(layer, params, param_name, param_type):
    print('param_name: {} copied'.format(param_name))
    param = np.asarray(params[param_name])
    if param_type == 'weight':
        assert layer.W.array.shape == param.shape
        layer.W.array[:] = param
    elif param_type == 'bias':
        assert layer.b.array.shape == param.shape
        layer.b.array[:] = param
    else:
        raise ValueError(
            'param_name: {} is not supported'.format(param_name))


def torch2chainer(model, params):
    rtc_dict = {'r': 'rot', 't': 'trans', 'c': 'conf'}
    emb_dict = {'e_conv': 'img', 'conv': 'pcd'}
    param_names = list(params.keys())
    uncopied_param_names = param_names.copy()
    print('start copying params')
    for param_name in param_names:
        # pose net conv
        if param_name.startswith('conv'):
            param_layer_name, param_type = param_name.split('.')
            layer_num = int(param_layer_name[4])
            layer_type = rtc_dict[param_layer_name[-1]]
            layer = getattr(model, 'conv{}_{}'.format(layer_num, layer_type))
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)
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
            uncopied_param_names.remove(param_name)
        # pspnet classifier
        elif param_name.startswith('cnn.model.module.classifier'):
            uncopied_param_names.remove(param_name)
        # pspnet conv1
        elif param_name.startswith('cnn.model.module.final'):
            layer = model.pspnet_extractor.conv1
            param_type = param_name.split('.')[-1]
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)
        # pspnet upsample
        elif param_name.startswith('cnn.model.module.up'):
            param_module_name, _, layer_num, param_type = \
                param_name.split('.')[3:]
            module_num = int(param_module_name[-1])
            module_name = 'up{}'.format(module_num)
            layer_num = int(layer_num)
            if layer_num == 1:
                layer_name = 'conv'
            elif layer_num == 2:
                layer_name = 'prelu'
            else:
                raise ValueError(
                    'param: {} is not supported'.format(param_name))
            layer = getattr(
                getattr(model.pspnet_extractor, module_name), layer_name)
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)
        # pspnet bottleneck
        elif param_name.startswith('cnn.model.module.psp.bottleneck'):
            param_type = param_name.split('.')[-1]
            layer = model.pspnet_extractor.psp.bottleneck
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)
        # pspnet stages
        elif param_name.startswith('cnn.model.module.psp.stages'):
            layer_num, _, param_type = param_name.split('.')[5:]
            layer_num = int(layer_num) + 1
            layer = getattr(
                model.pspnet_extractor.psp, 'conv{}'.format(layer_num))
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)
        # pspnet resnet conv1
        elif param_name.startswith('cnn.model.module.feats.conv1'):
            param_type = param_name.split('.')[-1]
            layer = model.pspnet_extractor.extractor.conv1
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)
        # pspnet resnet2-5
        elif param_name.startswith('cnn.model.module.feats.layer'):
            if param_name.split('.')[6].startswith('conv'):
                param_resblock_name, block_num, layer_name, param_type = \
                    param_name.split('.')[4:]
            elif param_name.split('.')[6].startswith('downsample'):
                param_resblock_name, block_num, _, _, param_type = \
                    param_name.split('.')[4:]
                layer_name = 'residual_conv'
            else:
                raise ValueError(
                    'param: {} is not supported'.format(param_name))
            resblock_num = int(param_resblock_name[-1]) + 1
            resblock = getattr(
                model.pspnet_extractor.extractor, 'res{}'.format(resblock_num))
            block_num = int(block_num)
            if block_num == 0:
                block_name = 'a'
            else:
                block_name = 'b{}'.format(block_num)
            block = getattr(resblock, block_name)
            layer = getattr(block, layer_name)
            copy_layer(layer, params, param_name, param_type)
            uncopied_param_names.remove(param_name)

    print('')
    if len(uncopied_param_names) > 0:
        print('Some params are not copied:')
        for param_name in uncopied_param_names:
            print('param_name: {} not copied'.format(param_name))
    else:
        print('All valid parameter are copied.')


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
        outpath = 'posenet_ycb_converted.npz'
    else:
        outpath = args.out
    chainer.serializers.save_npz(outpath, chainer_model)


if __name__ == '__main__':
    main()

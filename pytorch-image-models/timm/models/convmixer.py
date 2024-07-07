import sys
sys.path.append('../../../')
from convmixer import ConvMixer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}


@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model


@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_8_binary(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=5, patch_size=2, n_classes=2)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_8(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=5, patch_size=2, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_128_16_5_2(pretrained=False, **kwargs):
    model = ConvMixer(128, 16, kernel_size=5, patch_size=2, n_classes=2)
    model.default_cfg = _cfg
    return model


@register_model
def convmixer_128_32_5_2(pretrained=False, **kwargs):
    model = ConvMixer(128, 32, kernel_size=5, patch_size=2, n_classes=2)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_128_8_5_2(pretrained=False, **kwargs):
    model = ConvMixer(128, 8, kernel_size=5, patch_size=2, n_classes=2)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_128_4_5_2(pretrained=False, **kwargs):
    model = ConvMixer(128, 4, kernel_size=5, patch_size=2, n_classes=2)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_128_16_7_1(pretrained=False, **kwargs):
    model = ConvMixer(128, 16, kernel_size=7, patch_size=1, n_classes=2)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_8_7_1(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=7, patch_size=1, n_classes=2)
    model.default_cfg = _cfg
    return model
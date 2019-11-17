import paddle.fluid as fluid
from paddle.fluid.layers import fc, conv2d, conv2d_transpose, reshape, pixel_shuffle, batch_norm, dropout, flatten


# 上采样
def up_sampling_2(x, num_filters, name, act='relu'):
    x = pixel_shuffle(x, 2)
    x = conv2d(x, num_filters, 3, padding=1, name=name + "_conv2d_1")
    x = batch_norm(x, act=act, name=name + "_bn")
    return x


# 下采样
def down_sampling_2(x, num_filters, name, act='leaky_relu'):
    x = conv2d(x, num_filters, 3, stride=2, padding=1, name=name + "_conv2d")
    x = batch_norm(x, act=act, name=name + "_bn")
    x = dropout(x, 0.25, name=name + "_dropout")
    return x


class Generator:
    def __init__(self, name=None, image_size=None, z_dim=100):
        self.name = name or "G"
        self.image_size = image_size or [3, 128, 128]
        self.inputs = None
        self.z_dim = z_dim

    def net(self, inputs=None):
        if self.inputs is None:
            self.inputs = inputs or fluid.layers.data(name=self.name + "_z", shape=[self.z_dim], dtype='float32')

        act = 'relu'
        x = fc(self.inputs, 1024 * 4 * 4, act=act, name=self.name + "_fc_1")
        x = reshape(x, [-1, 1024, 4, 4], name=self.name + "_reshape")

        x = up_sampling_2(x, 512, name=self.name + "_512×8×8")
        x = up_sampling_2(x, 256, name=self.name + "_256×16×16")
        x = up_sampling_2(x, 128, name=self.name + "_128×32×32")
        x = up_sampling_2(x, 64, name=self.name + "_64×64×64")
        x = up_sampling_2(x, 32, name=self.name + "_32×128×128")

        out = conv2d(x, 3, 3, padding=1, act='tanh', name=self.name + "_conv2d_out")
        return out


class Discriminator:
    def __init__(self, name=None, image_size=None):
        self.name = name or "D"
        self.image_size = image_size or [3, 128, 128]
        self.inputs = None

    def net(self, inputs=None):
        if self.inputs is None:
            self.inputs = inputs or fluid.layers.data(name=self.name + "_image", shape=self.image_size, dtype='float32')

        x = conv2d(self.inputs, 32, 3, stride=2, padding=1, act='leaky_relu', name=self.name + "_conv2d_1")
        x = dropout(x, 0.25)

        x = down_sampling_2(x, 64, name=self.name + "_64×32*32")
        x = down_sampling_2(x, 128, name=self.name + "_128×16*16")
        x = down_sampling_2(x, 256, name=self.name + "_256×8*8")

        x = flatten(x, name=self.name + "_fc")
        x = fc(x, 1, act="sigmoid")

        return x


if __name__ == '__main__':
    G = Generator().net()
    D = Discriminator().net()

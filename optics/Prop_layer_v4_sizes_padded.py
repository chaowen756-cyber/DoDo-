import math as m

import numpy as np
import tensorflow as K
from keras.constraints import NonNeg
from keras.layers import Layer


class PropagationPadded(Layer):
    """
    A drop-in propagation layer with optional zero-padding + center-crop.
    pad_factor=1 keeps original behavior.
    """

    def __init__(self, Mp=300, L=1, wave_lengths=None, zi=2, Trai=True, pad_factor=1, **kwargs):
        self.Mpi = Mp
        self.Li = L
        self.zi = zi
        self.Trai = Trai
        self.pad_factor = int(pad_factor)
        if self.pad_factor < 1:
            raise ValueError("pad_factor must be >= 1")
        if wave_lengths is not None:
            self.wave_lengths = wave_lengths
        else:
            self.wave_lengths = np.linspace(420, 660, 25) * 1e-9
        super(PropagationPadded, self).__init__(**kwargs)

    def build(self, input_shape):
        initializer_c = K.constant_initializer(self.zi)
        self.z = self.add_weight(
            name="Distance",
            shape=[1],
            constraint=NonNeg(),
            initializer=initializer_c,
            trainable=self.Trai,
        )
        super(PropagationPadded, self).build(input_shape)

    def call(self, input_tensor, **kwargs):
        mp = self.Mpi
        l_phys = self.Li
        lambdas = self.wave_lengths

        if self.pad_factor == 1:
            work = input_tensor
            ns = mp
            pad_before = 0
            dx = l_phys / mp
            l_work = l_phys
        else:
            mpad = mp * self.pad_factor
            pad_total = mpad - mp
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            work = K.pad(input_tensor, [[0, 0], [pad_before, pad_after], [pad_before, pad_after], [0, 0]])
            ns = mpad
            dx = l_phys / mp
            l_work = dx * ns

        fx = K.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / l_work, ns)
        ff_x, ff_y = K.meshgrid(fx, fx)

        for n_lam in range(25):
            aux = -1j * m.pi * lambdas[n_lam] * K.cast(self.z, K.complex64)
            aux2 = K.cast(ff_x ** 2 + ff_y ** 2, K.complex64)
            h_slice = K.math.exp(aux * aux2)
            h_slice = K.expand_dims(K.signal.fftshift(h_slice, axes=[0, 1]), 2)
            if n_lam > 0:
                h = K.concat([h, h_slice], axis=2, name="stack")
            else:
                h = h_slice

        aux3 = K.signal.fftshift(K.cast(work, K.complex64), axes=[1, 2])
        u1f = K.signal.fft2d(K.transpose(aux3, (0, 3, 1, 2)))
        u1f = K.transpose(u1f, (0, 2, 3, 1))
        h = K.expand_dims(h, 0)
        u2f = K.math.multiply(u1f, h)
        u2 = K.transpose(
            K.signal.ifftshift(K.signal.ifft2d(K.transpose(u2f, (0, 3, 1, 2))), axes=[2, 3]),
            (0, 2, 3, 1),
        )

        if self.pad_factor > 1:
            u2 = u2[:, pad_before : pad_before + mp, pad_before : pad_before + mp, :]

        return u2

    def get_config(self):
        config = super(PropagationPadded, self).get_config()
        config.update(
            {
                "Mp": self.Mpi,
                "L": self.Li,
                "zi": self.zi,
                "Trai": self.Trai,
                "pad_factor": self.pad_factor,
            }
        )
        return config

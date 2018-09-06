#https://raw.githubusercontent.com/davidtvs/Keras-LinkNet/master/models/linknet.py

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, \
    Activation, Add, Input, Softmax
from keras.models import Model
from keras.backend import int_shape, is_keras_tensor

import requests
import math
from tqdm import tqdm
def _downloadFile(url, filename):
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0)); 
    block_size = 1024
    wrote = 0 
    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , 
                         unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")  
        
        

from keras.engine import InputSpec
from keras.layers import Conv2D
from keras.utils.conv_utils import deconv_length
import keras.backend as K


class Conv2DTranspose(Conv2D):
    """Transposed convolution layer (sometimes called Deconvolution).

    The need for transposed convolutions generally arises
    from the desire to use a transformation going in the opposite direction
    of a normal convolution, i.e., from something that has the shape of the
    output of some convolution to something that has the shape of its input
    while maintaining a connectivity pattern that is compatible with
    said convolution.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_shape: A tuple of integers specifying the shape of the output
            without the batch size. When not specified, the output shape is
            inferred from the input shape. For some combinations of input
            shape and layer parameters, the output shape is ambigous which
            can result in an undesired output shape.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 2 integers, specifying
            the dilation rate to use for dilated convolution.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    # References
        - [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        output_shape=None,
        data_format=None,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Conv2DTranspose, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.input_spec = InputSpec(ndim=4)
        if output_shape is not None:
            try:
                self._output_shape = tuple(output_shape)
            except TypeError:
                raise ValueError('`output_shape` argument must be a ' +
                                 'tuple. Received: ' + str(output_shape))
            if len(self._output_shape) != 3:
                raise ValueError('`output_shape` argument should have ' +
                                 'rank ' + str(3) + '; Received:', str(output_shape))
        else:
            self._output_shape = output_shape

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                'Inputs should have rank ' + str(4) +
                '; Received input shape:', str(input_shape)
            )
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError(
                'The channel dimension of the inputs '
                'should be defined. Found `None`.'
            )
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters, ),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        # Infer the dynamic output shape:
        if self._output_shape is None:
            out_height = deconv_length(height, stride_h, kernel_h, self.padding)
            out_width = deconv_length(width, stride_w, kernel_w, self.padding)
            if self.data_format == 'channels_first':
                output_shape = (
                    batch_size, self.filters, out_height, out_width
                )
            else:
                output_shape = (
                    batch_size, out_height, out_width, self.filters
                )
        else:
            output_shape = (batch_size,) + self._output_shape

        outputs = K.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format
        )

        if self.bias:
            outputs = K.bias_add(
                outputs, self.bias, data_format=self.data_format
            )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self._output_shape is None:
            output_shape[c_axis] = self.filters
            output_shape[h_axis] = deconv_length(
                output_shape[h_axis], stride_h, kernel_h, self.padding
            )
            output_shape[w_axis] = deconv_length(
                output_shape[w_axis], stride_w, kernel_w, self.padding
            )
        else:
            output_shape[1:] = self._output_shape

        return tuple(output_shape)

    def get_config(self):
        config = super(Conv2DTranspose, self).get_config()
        config.pop('dilation_rate')
        config['output_shape'] = self._output_shape
        return config        
        
        
class LinkNet():
    """LinkNet architecture.

    The model follows the architecture presented in: https://arxiv.org/abs/1707.03718

    Args:
        num_classes (int): the number of classes to segment.
        input_tensor (tensor, optional): Keras tensor
            (i.e. output of `layers.Input()`) to use as image input for
            the model. Default: None.
        input_shape (tuple, optional): Shape tuple of the model input.
            Default: None.
        initial_block_filters (int, optional): The number of filters after
            the initial block (see the paper for details on the initial
            block). Default: None.
        bias (bool, optional): If ``True``, adds a learnable bias.
            Default: ``False``.

    """

    def __init__(
        self,
        num_classes,
        input_tensor=None,
        input_shape=None,
        initial_block_filters=64,
        bias=False,
        name='linknet'
    ):
        self.num_classes = num_classes
        self.initial_block_filters = initial_block_filters
        self.bias = bias
        self.output_shape = input_shape[:-1] + (num_classes, )

        # Create a Keras tensor from the input_shape/input_tensor
        if input_tensor is None:
            self.input = Input(shape=input_shape, name='input_img')
        elif is_keras_tensor(input_tensor):
            self.input = input_tensor
        else:
            # input_tensor is a tensor but not one from Keras
            self.input = Input(
                tensor=input_tensor, shape=input_shape, name='input_img'
            )

        self.name = name

    def get_model(
        self,
        pretrained_encoder=True,
        weights_path='./checkpoints/linknet_encoder_weights.h5'
    ):
        """Initializes a LinkNet model.

        Returns:
            A Keras model instance.

        """
        # Build encoder
        encoder_model = self.get_encoder()
        if pretrained_encoder:
            encoder_model.load_weights(weights_path)
        encoder_out = encoder_model(self.input)

        # Build decoder
        decoder_model = self.get_decoder(encoder_out)
        decoder_out = decoder_model(encoder_out[:-1])

        return Model(inputs=self.input, outputs=decoder_out, name=self.name)

    def get_encoder(self, name='encoder'):
        """Builds the encoder of a LinkNet architecture.

        Args:
            name (string, optional): The encoder model name.
                Default: 'encoder'.

        Returns:
            The encoder as a Keras model instance.

        """
        # Initial block
        initial1 = Conv2D(
            self.initial_block_filters,
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/conv2d_1'
        )(self.input)
        initial1 = BatchNormalization(name=name + '/0/bn_1')(initial1)
        initial1 = Activation('relu', name=name + '/0/relu_1')(initial1)
        initial2 = MaxPooling2D(pool_size=2, name=name + '/0/maxpool_1')(initial1)  # yapf: disable

        # Encoder blocks
        encoder1 = self.encoder_block(
            initial2,
            self.initial_block_filters,
            strides=1,
            bias=self.bias,
            name=name + '/1'
        )
        encoder2 = self.encoder_block(
            encoder1,
            self.initial_block_filters * 2,
            strides=(2, 1),
            bias=self.bias,
            name=name + '/2'
        )
        encoder3 = self.encoder_block(
            encoder2,
            self.initial_block_filters * 4,
            strides=(2, 1),
            bias=self.bias,
            name=name + '/3'
        )
        encoder4 = self.encoder_block(
            encoder3,
            self.initial_block_filters * 8,
            strides=(2, 1),
            bias=self.bias,
            name=name + '/4'
        )

        return Model(
            inputs=self.input,
            outputs=[
                encoder4, encoder3, encoder2, encoder1, initial2, initial1
            ],
            name=name
        )

    def encoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        bias=False,
        name=''
    ):
        """Creates an encoder block.

        The encoder block is a combination of two basic encoder blocks
        (see ``encoder_basic_block``). The first with stride 2 and the
        the second with stride 1.

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, or list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, or list, optional): A tuple/list of two
                integers, specifying the stride for each basic block. A
                single integer can also be specified, in which case both
                basic blocks use the same stride. Default: 1.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.
            name (string, optional): A string to identify this block.
                Default: Empty string.

        Returns:
            The output tensor of the block.

        """
        assert isinstance(strides, (int, tuple, list)), (
            "expected int, tuple, or list for strides"
        )  # yapf: disable
        if (isinstance(strides, (tuple, list))):
            if len(strides) == 2:
                stride_1, stride_2 = strides
            else:
                raise ValueError("expected a list or tuple on length 2")
        else:
            stride_1 = strides
            stride_2 = strides

        x = self.encoder_basic_block(
            input,
            out_filters,
            kernel_size=kernel_size,
            strides=stride_1,
            padding=padding,
            bias=bias,
            name=name + '/1'
        )

        x = self.encoder_basic_block(
            x,
            out_filters,
            kernel_size=kernel_size,
            strides=stride_2,
            padding=padding,
            bias=bias,
            name=name + '/2'
        )

        return x

    def encoder_basic_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        bias=False,
        name=''
    ):
        """Creates a basic encoder block.

        Main brach architecture:
        1. Conv2D
        2. BatchNormalization
        3. ReLU
        4. Conv2D
        5. BatchNormalization
        Residual branch architecture:
        1. Conv2D, if `strides` is greater than 1
        The output of the main and residual branches are then added together
        with ReLU activation.

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the strides along the height and width
                of the 2D input. In case it's a single integer, it's value
                is used for all spatial dimensions. Default: 1.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.
            name (string, optional): A string to identify this block.
                Default: Empty string.

        Returns:
            The output tensor of the block.

        """
        residual = input

        x = Conv2D(
            out_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=bias,
            name=name + '/main/conv2d_1'
        )(input)
        x = BatchNormalization(name=name + '/main/bn_1')(x)
        x = Activation('relu', name=name + '/main/relu_1')(x)

        x = Conv2D(
            out_filters,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/main/conv2d_2'
        )(x)
        x = BatchNormalization(name=name + '/main/bn_2')(x)

        if strides > 1:
            residual = Conv2D(
                out_filters,
                kernel_size=1,
                strides=strides,
                padding=padding,
                use_bias=bias,
                name=name + '/res/conv2d_1'
            )(residual)
            residual = BatchNormalization(name=name + '/res/bn_1')(residual)

        x = Add(name=name + '/add')([x, residual])
        x = Activation('relu', name=name + '/relu_1')(x)

        return x

    def get_decoder(self, inputs, name='decoder'):
        """Builds the decoder of a LinkNet architecture.

        Args:
            name (string, optional): The encoder model name.
                Default: 'decoder'.

        Returns:
            The decoder as a Keras model instance.

        """
        # Decoder inputs
        encoder4 = Input(shape=int_shape(inputs[0])[1:], name='encoder4')
        encoder3 = Input(shape=int_shape(inputs[1])[1:], name='encoder3')
        encoder2 = Input(shape=int_shape(inputs[2])[1:], name='encoder2')
        encoder1 = Input(shape=int_shape(inputs[3])[1:], name='encoder1')
        initial2 = Input(shape=int_shape(inputs[4])[1:], name='initial2')
        initial1 = inputs[5]

        # Decoder blocks
        decoder4 = self.decoder_block(
            encoder4,
            self.initial_block_filters * 4,
            strides=2,
            output_shape=int_shape(encoder3)[1:],
            bias=self.bias,
            name=name + '/4'
        )
        decoder4 = Add(name=name + '/shortcut_e3_d4')([encoder3, decoder4])

        decoder3 = self.decoder_block(
            decoder4,
            self.initial_block_filters * 2,
            strides=2,
            output_shape=int_shape(encoder2)[1:],
            bias=self.bias,
            name=name + '/3'
        )
        decoder3 = Add(name=name + '/shortcut_e2_d3')([encoder2, decoder3])

        decoder2 = self.decoder_block(
            decoder3,
            self.initial_block_filters,
            strides=2,
            output_shape=int_shape(encoder1)[1:],
            bias=self.bias,
            name=name + '/2'
        )
        decoder2 = Add(name=name + '/shortcut_e1_d2')([encoder1, decoder2])

        decoder1 = self.decoder_block(
            decoder2,
            self.initial_block_filters,
            strides=1,
            output_shape=int_shape(initial2)[1:],
            bias=self.bias,
            name=name + '/1'
        )
        decoder1 = Add(name=name + '/shortcut_init_d1')([initial2, decoder1])

        # Final block
        # Build the output shape of the next layer - same width and height
        # as initial1
        shape = (
            int_shape(initial1)[1],
            int_shape(initial1)[2],
            self.initial_block_filters // 2,
        )
        final = Conv2DTranspose(
            self.initial_block_filters // 2,
            kernel_size=3,
            strides=2,
            padding='same',
            output_shape=shape,
            use_bias=self.bias,
            name=name + '/0/transposed2d_1'
        )(decoder1)
        final = BatchNormalization(name=name + '/0/bn_1')(final)
        final = Activation('relu', name=name + '/0/relu_1')(final)

        final = Conv2D(
            self.initial_block_filters // 2,
            kernel_size=3,
            padding='same',
            use_bias=self.bias,
            name=name + '/0/conv2d_1'
        )(final)
        final = BatchNormalization(name=name + '/0/bn_2')(final)
        final = Activation('relu', name=name + '/0/relu_2')(final)

        logits = Conv2DTranspose(
            self.num_classes,
            kernel_size=2,
            strides=2,
            padding='same',
            output_shape=self.output_shape,
            use_bias=self.bias,
            name=name + '/0/transposed2d_2'
        )(final)

        prediction = Softmax(name=name + '/0/softmax')(logits)

        return Model(
            inputs=[
                encoder4, encoder3, encoder2, encoder1, initial2
            ],
            outputs=prediction,
            name=name
        )

    def decoder_block(
        self,
        input,
        out_filters,
        kernel_size=3,
        strides=2,
        projection_ratio=4,
        padding='same',
        output_shape=None,
        bias=False,
        name=''
    ):
        """Creates a decoder block.

        Decoder block architecture:
        1. Conv2D
        2. BatchNormalization
        3. ReLU
        4. Conv2DTranspose
        5. BatchNormalization
        6. ReLU
        7. Conv2D
        8. BatchNormalization
        9. ReLU

        Args:
            input (tensor): A tensor or variable.
            out_filters (int): The number of filters in the block output.
            kernel_size (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the height and width of the 2D kernel
                window. In case it's a single integer, it's value is used
                for all spatial dimensions. Default: 3.
            strides (int, tuple, list, optional): A tuple/list of 2
                integers, specifying the strides along the height and width
                of the 2D input. In case it's a single integer, it's value
                is used for all spatial dimensions. Default: 1.
            projection_ratio (int, optional): A scale factor applied to
                the number of input channels. The output of the first
                convolution will have ``input_channels // projection_ratio``.
                The goal is to decrease the number of parameters in the
                transposed convolution layer. Default: 4.
            padding (str, optional): One of "valid" or "same" (case-insensitive).
                Default: "same".
            output_shape: A tuple of integers specifying the shape of the output
                without the batch size. Default: None.
            bias (bool, optional): If ``True``, adds a learnable bias.
                Default: ``False``.
            name (string, optional): A string to identify this block.
                Default: Empty string.

        Returns:
            The output tensor of the block.

        """
        internal_filters = int_shape(input)[-1] // projection_ratio
        x = Conv2D(
            internal_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/conv2d_1'
        )(input)
        x = BatchNormalization(name=name + '/bn_1')(x)
        x = Activation('relu', name=name + '/relu_1')(x)

        # The shape of the following trasposed convolution is the output
        # shape of the block with 'internal_filters' channels
        shape = output_shape[:-1] + (internal_filters, )
        x = Conv2DTranspose(
            internal_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_shape=shape,
            use_bias=bias,
            name=name + '/transposed2d_1'
        )(x)
        x = BatchNormalization(name=name + '/bn_2')(x)
        x = Activation('relu', name=name + '/relu_2')(x)

        x = Conv2D(
            out_filters,
            kernel_size=1,
            strides=1,
            padding=padding,
            use_bias=bias,
            name=name + '/conv2d_2'
        )(x)
        x = BatchNormalization(name=name + '/bn_3')(x)
        x = Activation('relu', name=name + '/relu_3')(x)

        return x
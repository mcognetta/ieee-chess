from enum import Enum
from onnx2torch import convert
import torch


class MAIA_TYPE(Enum):
    MAIA_1100 = 1100
    MAIA_1200 = 1200
    MAIA_1300 = 1300
    MAIA_1400 = 1400
    MAIA_1500 = 1500
    MAIA_1600 = 1600
    MAIA_1700 = 1700
    MAIA_1800 = 1800
    MAIA_1900 = 1900


_MAIA_EMBED_SIZE = 4096


def _maia_embed(self, input_1):

    bsz = input_1.shape[0]
    inputconv = self.inputconv(input_1)
    input_1 = None
    inputconv_relu = getattr(self, "inputconv/relu")(inputconv)
    inputconv = None
    block0_conv1 = getattr(self, "block0/conv1")(inputconv_relu)
    block0_conv1_relu = getattr(self, "block0/conv1/relu")(block0_conv1)
    block0_conv1 = None
    block0_conv2 = getattr(self, "block0/conv2")(block0_conv1_relu)
    block0_conv1_relu = None
    block0_conv2_se_reduce_mean = getattr(self, "block0/conv2/se/reduce_mean")(
        block0_conv2
    )
    initializers_onnx_initializer_0 = self.initializers.onnx_initializer_0
    block0_conv2_se_matmul1 = getattr(self, "block0/conv2/se/matmul1")(
        block0_conv2_se_reduce_mean, initializers_onnx_initializer_0
    )
    block0_conv2_se_reduce_mean = initializers_onnx_initializer_0 = None
    initializers_onnx_initializer_1 = self.initializers.onnx_initializer_1
    block0_conv2_se_add1 = getattr(self, "block0/conv2/se/add1")(
        block0_conv2_se_matmul1, initializers_onnx_initializer_1
    )
    block0_conv2_se_matmul1 = initializers_onnx_initializer_1 = None
    block0_conv2_se_relu = getattr(self, "block0/conv2/se/relu")(block0_conv2_se_add1)
    block0_conv2_se_add1 = None
    initializers_onnx_initializer_2 = self.initializers.onnx_initializer_2
    block0_conv2_se_matmul2 = getattr(self, "block0/conv2/se/matmul2")(
        block0_conv2_se_relu, initializers_onnx_initializer_2
    )
    block0_conv2_se_relu = initializers_onnx_initializer_2 = None
    initializers_onnx_initializer_3 = self.initializers.onnx_initializer_3
    block0_conv2_se_add2 = getattr(self, "block0/conv2/se/add2")(
        block0_conv2_se_matmul2, initializers_onnx_initializer_3
    )
    block0_conv2_se_matmul2 = initializers_onnx_initializer_3 = None
    initializers_onnx_initializer_4 = self.initializers.onnx_initializer_4
    block0_conv2_se_reshape = getattr(self, "block0/conv2/se/reshape")(
        block0_conv2_se_add2, initializers_onnx_initializer_4
    )
    block0_conv2_se_add2 = initializers_onnx_initializer_4 = None
    block0_conv2_se_split = getattr(self, "block0/conv2/se/split")(
        block0_conv2_se_reshape
    )
    block0_conv2_se_reshape = None
    getitem = block0_conv2_se_split[0]
    block0_conv2_se_sigmoid = getattr(self, "block0/conv2/se/sigmoid")(getitem)
    getitem = None
    block0_conv2_se_mul = getattr(self, "block0/conv2/se/mul")(
        block0_conv2_se_sigmoid, block0_conv2
    )
    block0_conv2_se_sigmoid = block0_conv2 = None
    getitem_1 = block0_conv2_se_split[1]
    block0_conv2_se_split = None
    block0_conv2_se_add3 = getattr(self, "block0/conv2/se/add3")(
        block0_conv2_se_mul, getitem_1
    )
    block0_conv2_se_mul = getitem_1 = None
    block0_conv2_mixin = getattr(self, "block0/conv2/mixin")(
        block0_conv2_se_add3, inputconv_relu
    )
    block0_conv2_se_add3 = inputconv_relu = None
    block0_conv2_relu = getattr(self, "block0/conv2/relu")(block0_conv2_mixin)
    block0_conv2_mixin = None
    block1_conv1 = getattr(self, "block1/conv1")(block0_conv2_relu)
    block1_conv1_relu = getattr(self, "block1/conv1/relu")(block1_conv1)
    block1_conv1 = None
    block1_conv2 = getattr(self, "block1/conv2")(block1_conv1_relu)
    block1_conv1_relu = None
    block1_conv2_se_reduce_mean = getattr(self, "block1/conv2/se/reduce_mean")(
        block1_conv2
    )
    initializers_onnx_initializer_5 = self.initializers.onnx_initializer_5
    block1_conv2_se_matmul1 = getattr(self, "block1/conv2/se/matmul1")(
        block1_conv2_se_reduce_mean, initializers_onnx_initializer_5
    )
    block1_conv2_se_reduce_mean = initializers_onnx_initializer_5 = None
    initializers_onnx_initializer_6 = self.initializers.onnx_initializer_6
    block1_conv2_se_add1 = getattr(self, "block1/conv2/se/add1")(
        block1_conv2_se_matmul1, initializers_onnx_initializer_6
    )
    block1_conv2_se_matmul1 = initializers_onnx_initializer_6 = None
    block1_conv2_se_relu = getattr(self, "block1/conv2/se/relu")(block1_conv2_se_add1)
    block1_conv2_se_add1 = None
    initializers_onnx_initializer_7 = self.initializers.onnx_initializer_7
    block1_conv2_se_matmul2 = getattr(self, "block1/conv2/se/matmul2")(
        block1_conv2_se_relu, initializers_onnx_initializer_7
    )
    block1_conv2_se_relu = initializers_onnx_initializer_7 = None
    initializers_onnx_initializer_8 = self.initializers.onnx_initializer_8
    block1_conv2_se_add2 = getattr(self, "block1/conv2/se/add2")(
        block1_conv2_se_matmul2, initializers_onnx_initializer_8
    )
    block1_conv2_se_matmul2 = initializers_onnx_initializer_8 = None
    initializers_onnx_initializer_9 = self.initializers.onnx_initializer_9
    block1_conv2_se_reshape = getattr(self, "block1/conv2/se/reshape")(
        block1_conv2_se_add2, initializers_onnx_initializer_9
    )
    block1_conv2_se_add2 = initializers_onnx_initializer_9 = None
    block1_conv2_se_split = getattr(self, "block1/conv2/se/split")(
        block1_conv2_se_reshape
    )
    block1_conv2_se_reshape = None
    getitem_2 = block1_conv2_se_split[0]
    block1_conv2_se_sigmoid = getattr(self, "block1/conv2/se/sigmoid")(getitem_2)
    getitem_2 = None
    block1_conv2_se_mul = getattr(self, "block1/conv2/se/mul")(
        block1_conv2_se_sigmoid, block1_conv2
    )
    block1_conv2_se_sigmoid = block1_conv2 = None
    getitem_3 = block1_conv2_se_split[1]
    block1_conv2_se_split = None
    block1_conv2_se_add3 = getattr(self, "block1/conv2/se/add3")(
        block1_conv2_se_mul, getitem_3
    )
    block1_conv2_se_mul = getitem_3 = None
    block1_conv2_mixin = getattr(self, "block1/conv2/mixin")(
        block1_conv2_se_add3, block0_conv2_relu
    )
    block1_conv2_se_add3 = block0_conv2_relu = None
    block1_conv2_relu = getattr(self, "block1/conv2/relu")(block1_conv2_mixin)
    block1_conv2_mixin = None
    block2_conv1 = getattr(self, "block2/conv1")(block1_conv2_relu)
    block2_conv1_relu = getattr(self, "block2/conv1/relu")(block2_conv1)
    block2_conv1 = None
    block2_conv2 = getattr(self, "block2/conv2")(block2_conv1_relu)
    block2_conv1_relu = None
    block2_conv2_se_reduce_mean = getattr(self, "block2/conv2/se/reduce_mean")(
        block2_conv2
    )
    initializers_onnx_initializer_10 = self.initializers.onnx_initializer_10
    block2_conv2_se_matmul1 = getattr(self, "block2/conv2/se/matmul1")(
        block2_conv2_se_reduce_mean, initializers_onnx_initializer_10
    )
    block2_conv2_se_reduce_mean = initializers_onnx_initializer_10 = None
    initializers_onnx_initializer_11 = self.initializers.onnx_initializer_11
    block2_conv2_se_add1 = getattr(self, "block2/conv2/se/add1")(
        block2_conv2_se_matmul1, initializers_onnx_initializer_11
    )
    block2_conv2_se_matmul1 = initializers_onnx_initializer_11 = None
    block2_conv2_se_relu = getattr(self, "block2/conv2/se/relu")(block2_conv2_se_add1)
    block2_conv2_se_add1 = None
    initializers_onnx_initializer_12 = self.initializers.onnx_initializer_12
    block2_conv2_se_matmul2 = getattr(self, "block2/conv2/se/matmul2")(
        block2_conv2_se_relu, initializers_onnx_initializer_12
    )
    block2_conv2_se_relu = initializers_onnx_initializer_12 = None
    initializers_onnx_initializer_13 = self.initializers.onnx_initializer_13
    block2_conv2_se_add2 = getattr(self, "block2/conv2/se/add2")(
        block2_conv2_se_matmul2, initializers_onnx_initializer_13
    )
    block2_conv2_se_matmul2 = initializers_onnx_initializer_13 = None
    initializers_onnx_initializer_14 = self.initializers.onnx_initializer_14
    block2_conv2_se_reshape = getattr(self, "block2/conv2/se/reshape")(
        block2_conv2_se_add2, initializers_onnx_initializer_14
    )
    block2_conv2_se_add2 = initializers_onnx_initializer_14 = None
    block2_conv2_se_split = getattr(self, "block2/conv2/se/split")(
        block2_conv2_se_reshape
    )
    block2_conv2_se_reshape = None
    getitem_4 = block2_conv2_se_split[0]
    block2_conv2_se_sigmoid = getattr(self, "block2/conv2/se/sigmoid")(getitem_4)
    getitem_4 = None
    block2_conv2_se_mul = getattr(self, "block2/conv2/se/mul")(
        block2_conv2_se_sigmoid, block2_conv2
    )
    block2_conv2_se_sigmoid = block2_conv2 = None
    getitem_5 = block2_conv2_se_split[1]
    block2_conv2_se_split = None
    block2_conv2_se_add3 = getattr(self, "block2/conv2/se/add3")(
        block2_conv2_se_mul, getitem_5
    )
    block2_conv2_se_mul = getitem_5 = None
    block2_conv2_mixin = getattr(self, "block2/conv2/mixin")(
        block2_conv2_se_add3, block1_conv2_relu
    )
    block2_conv2_se_add3 = block1_conv2_relu = None
    block2_conv2_relu = getattr(self, "block2/conv2/relu")(block2_conv2_mixin)
    block2_conv2_mixin = None
    block3_conv1 = getattr(self, "block3/conv1")(block2_conv2_relu)
    block3_conv1_relu = getattr(self, "block3/conv1/relu")(block3_conv1)
    block3_conv1 = None
    block3_conv2 = getattr(self, "block3/conv2")(block3_conv1_relu)
    block3_conv1_relu = None
    block3_conv2_se_reduce_mean = getattr(self, "block3/conv2/se/reduce_mean")(
        block3_conv2
    )
    initializers_onnx_initializer_15 = self.initializers.onnx_initializer_15
    block3_conv2_se_matmul1 = getattr(self, "block3/conv2/se/matmul1")(
        block3_conv2_se_reduce_mean, initializers_onnx_initializer_15
    )
    block3_conv2_se_reduce_mean = initializers_onnx_initializer_15 = None
    initializers_onnx_initializer_16 = self.initializers.onnx_initializer_16
    block3_conv2_se_add1 = getattr(self, "block3/conv2/se/add1")(
        block3_conv2_se_matmul1, initializers_onnx_initializer_16
    )
    block3_conv2_se_matmul1 = initializers_onnx_initializer_16 = None
    block3_conv2_se_relu = getattr(self, "block3/conv2/se/relu")(block3_conv2_se_add1)
    block3_conv2_se_add1 = None
    initializers_onnx_initializer_17 = self.initializers.onnx_initializer_17
    block3_conv2_se_matmul2 = getattr(self, "block3/conv2/se/matmul2")(
        block3_conv2_se_relu, initializers_onnx_initializer_17
    )
    block3_conv2_se_relu = initializers_onnx_initializer_17 = None
    initializers_onnx_initializer_18 = self.initializers.onnx_initializer_18
    block3_conv2_se_add2 = getattr(self, "block3/conv2/se/add2")(
        block3_conv2_se_matmul2, initializers_onnx_initializer_18
    )
    block3_conv2_se_matmul2 = initializers_onnx_initializer_18 = None
    initializers_onnx_initializer_19 = self.initializers.onnx_initializer_19
    block3_conv2_se_reshape = getattr(self, "block3/conv2/se/reshape")(
        block3_conv2_se_add2, initializers_onnx_initializer_19
    )
    block3_conv2_se_add2 = initializers_onnx_initializer_19 = None
    block3_conv2_se_split = getattr(self, "block3/conv2/se/split")(
        block3_conv2_se_reshape
    )
    block3_conv2_se_reshape = None
    getitem_6 = block3_conv2_se_split[0]
    block3_conv2_se_sigmoid = getattr(self, "block3/conv2/se/sigmoid")(getitem_6)
    getitem_6 = None
    block3_conv2_se_mul = getattr(self, "block3/conv2/se/mul")(
        block3_conv2_se_sigmoid, block3_conv2
    )
    block3_conv2_se_sigmoid = block3_conv2 = None
    getitem_7 = block3_conv2_se_split[1]
    block3_conv2_se_split = None
    block3_conv2_se_add3 = getattr(self, "block3/conv2/se/add3")(
        block3_conv2_se_mul, getitem_7
    )
    block3_conv2_se_mul = getitem_7 = None
    block3_conv2_mixin = getattr(self, "block3/conv2/mixin")(
        block3_conv2_se_add3, block2_conv2_relu
    )
    block3_conv2_se_add3 = block2_conv2_relu = None
    block3_conv2_relu = getattr(self, "block3/conv2/relu")(block3_conv2_mixin)
    block3_conv2_mixin = None
    block4_conv1 = getattr(self, "block4/conv1")(block3_conv2_relu)
    block4_conv1_relu = getattr(self, "block4/conv1/relu")(block4_conv1)
    block4_conv1 = None
    block4_conv2 = getattr(self, "block4/conv2")(block4_conv1_relu)
    block4_conv1_relu = None
    block4_conv2_se_reduce_mean = getattr(self, "block4/conv2/se/reduce_mean")(
        block4_conv2
    )
    initializers_onnx_initializer_20 = self.initializers.onnx_initializer_20
    block4_conv2_se_matmul1 = getattr(self, "block4/conv2/se/matmul1")(
        block4_conv2_se_reduce_mean, initializers_onnx_initializer_20
    )
    block4_conv2_se_reduce_mean = initializers_onnx_initializer_20 = None
    initializers_onnx_initializer_21 = self.initializers.onnx_initializer_21
    block4_conv2_se_add1 = getattr(self, "block4/conv2/se/add1")(
        block4_conv2_se_matmul1, initializers_onnx_initializer_21
    )
    block4_conv2_se_matmul1 = initializers_onnx_initializer_21 = None
    block4_conv2_se_relu = getattr(self, "block4/conv2/se/relu")(block4_conv2_se_add1)
    block4_conv2_se_add1 = None
    initializers_onnx_initializer_22 = self.initializers.onnx_initializer_22
    block4_conv2_se_matmul2 = getattr(self, "block4/conv2/se/matmul2")(
        block4_conv2_se_relu, initializers_onnx_initializer_22
    )
    block4_conv2_se_relu = initializers_onnx_initializer_22 = None
    initializers_onnx_initializer_23 = self.initializers.onnx_initializer_23
    block4_conv2_se_add2 = getattr(self, "block4/conv2/se/add2")(
        block4_conv2_se_matmul2, initializers_onnx_initializer_23
    )
    block4_conv2_se_matmul2 = initializers_onnx_initializer_23 = None
    initializers_onnx_initializer_24 = self.initializers.onnx_initializer_24
    block4_conv2_se_reshape = getattr(self, "block4/conv2/se/reshape")(
        block4_conv2_se_add2, initializers_onnx_initializer_24
    )
    block4_conv2_se_add2 = initializers_onnx_initializer_24 = None
    block4_conv2_se_split = getattr(self, "block4/conv2/se/split")(
        block4_conv2_se_reshape
    )
    block4_conv2_se_reshape = None
    getitem_8 = block4_conv2_se_split[0]
    block4_conv2_se_sigmoid = getattr(self, "block4/conv2/se/sigmoid")(getitem_8)
    getitem_8 = None
    block4_conv2_se_mul = getattr(self, "block4/conv2/se/mul")(
        block4_conv2_se_sigmoid, block4_conv2
    )
    block4_conv2_se_sigmoid = block4_conv2 = None
    getitem_9 = block4_conv2_se_split[1]
    block4_conv2_se_split = None
    block4_conv2_se_add3 = getattr(self, "block4/conv2/se/add3")(
        block4_conv2_se_mul, getitem_9
    )
    block4_conv2_se_mul = getitem_9 = None
    block4_conv2_mixin = getattr(self, "block4/conv2/mixin")(
        block4_conv2_se_add3, block3_conv2_relu
    )
    block4_conv2_se_add3 = block3_conv2_relu = None
    block4_conv2_relu = getattr(self, "block4/conv2/relu")(block4_conv2_mixin)
    block4_conv2_mixin = None
    block5_conv1 = getattr(self, "block5/conv1")(block4_conv2_relu)
    block5_conv1_relu = getattr(self, "block5/conv1/relu")(block5_conv1)
    block5_conv1 = None
    block5_conv2 = getattr(self, "block5/conv2")(block5_conv1_relu)
    block5_conv1_relu = None
    block5_conv2_se_reduce_mean = getattr(self, "block5/conv2/se/reduce_mean")(
        block5_conv2
    )
    initializers_onnx_initializer_25 = self.initializers.onnx_initializer_25
    block5_conv2_se_matmul1 = getattr(self, "block5/conv2/se/matmul1")(
        block5_conv2_se_reduce_mean, initializers_onnx_initializer_25
    )
    block5_conv2_se_reduce_mean = initializers_onnx_initializer_25 = None
    initializers_onnx_initializer_26 = self.initializers.onnx_initializer_26
    block5_conv2_se_add1 = getattr(self, "block5/conv2/se/add1")(
        block5_conv2_se_matmul1, initializers_onnx_initializer_26
    )
    block5_conv2_se_matmul1 = initializers_onnx_initializer_26 = None
    block5_conv2_se_relu = getattr(self, "block5/conv2/se/relu")(block5_conv2_se_add1)
    block5_conv2_se_add1 = None
    initializers_onnx_initializer_27 = self.initializers.onnx_initializer_27
    block5_conv2_se_matmul2 = getattr(self, "block5/conv2/se/matmul2")(
        block5_conv2_se_relu, initializers_onnx_initializer_27
    )
    block5_conv2_se_relu = initializers_onnx_initializer_27 = None
    initializers_onnx_initializer_28 = self.initializers.onnx_initializer_28
    block5_conv2_se_add2 = getattr(self, "block5/conv2/se/add2")(
        block5_conv2_se_matmul2, initializers_onnx_initializer_28
    )
    block5_conv2_se_matmul2 = initializers_onnx_initializer_28 = None
    initializers_onnx_initializer_29 = self.initializers.onnx_initializer_29
    block5_conv2_se_reshape = getattr(self, "block5/conv2/se/reshape")(
        block5_conv2_se_add2, initializers_onnx_initializer_29
    )
    block5_conv2_se_add2 = initializers_onnx_initializer_29 = None
    block5_conv2_se_split = getattr(self, "block5/conv2/se/split")(
        block5_conv2_se_reshape
    )
    block5_conv2_se_reshape = None
    getitem_10 = block5_conv2_se_split[0]
    block5_conv2_se_sigmoid = getattr(self, "block5/conv2/se/sigmoid")(getitem_10)
    getitem_10 = None
    block5_conv2_se_mul = getattr(self, "block5/conv2/se/mul")(
        block5_conv2_se_sigmoid, block5_conv2
    )
    block5_conv2_se_sigmoid = block5_conv2 = None
    getitem_11 = block5_conv2_se_split[1]
    block5_conv2_se_split = None
    block5_conv2_se_add3 = getattr(self, "block5/conv2/se/add3")(
        block5_conv2_se_mul, getitem_11
    )
    block5_conv2_se_mul = getitem_11 = None
    block5_conv2_mixin = getattr(self, "block5/conv2/mixin")(
        block5_conv2_se_add3, block4_conv2_relu
    )
    block5_conv2_se_add3 = block4_conv2_relu = None
    # block5_conv2_relu = getattr(self, "block5/conv2/relu")(block5_conv2_mixin);  block5_conv2_mixin = None

    # policy_conv1 = getattr(self, "policy/conv1")(block5_conv2_relu)
    # policy_conv1_relu = getattr(self, "policy/conv1/relu")(policy_conv1);  policy_conv1 = None
    # policy_conv2 = getattr(self, "policy/conv2")(policy_conv1_relu);  policy_conv1_relu = None
    # initializers_onnx_initializer_30 = self.initializers.onnx_initializer_30
    # policy_flatten = getattr(self, "policy/flatten")(policy_conv2, initializers_onnx_initializer_30);  policy_conv2 = initializers_onnx_initializer_30 = None
    # initializers_onnx_initializer_31 = self.initializers.onnx_initializer_31
    # output_policy = getattr(self, "output/policy")(policy_flatten, initializers_onnx_initializer_31);  policy_flatten = initializers_onnx_initializer_31 = None

    # value_conv = getattr(self, "value/conv")(block5_conv2_relu);  block5_conv2_relu = None
    # value_conv_relu = getattr(self, "value/conv/relu")(value_conv);  value_conv = None
    # initializers_onnx_initializer_32 = self.initializers.onnx_initializer_32
    # value_reshape = getattr(self, "value/reshape")(value_conv_relu, initializers_onnx_initializer_32);  value_conv_relu = initializers_onnx_initializer_32 = None
    # initializers_onnx_initializer_33 = self.initializers.onnx_initializer_33
    # value_dense1_matmul = getattr(self, "value/dense1/matmul")(value_reshape, initializers_onnx_initializer_33);  value_reshape = initializers_onnx_initializer_33 = None
    # initializers_onnx_initializer_34 = self.initializers.onnx_initializer_34
    # value_dense1_add = getattr(self, "value/dense1/add")(value_dense1_matmul, initializers_onnx_initializer_34);  value_dense1_matmul = initializers_onnx_initializer_34 = None
    # value_dense1_relu = getattr(self, "value/dense1/relu")(value_dense1_add);  value_dense1_add = None
    # initializers_onnx_initializer_35 = self.initializers.onnx_initializer_35
    # value_dense2_matmul = getattr(self, "value/dense2/matmul")(value_dense1_relu, initializers_onnx_initializer_35);  value_dense1_relu = initializers_onnx_initializer_35 = None
    # initializers_onnx_initializer_36 = self.initializers.onnx_initializer_36
    # value_dense2_add = getattr(self, "value/dense2/add")(value_dense2_matmul, initializers_onnx_initializer_36);  value_dense2_matmul = initializers_onnx_initializer_36 = None
    # output_wdl = getattr(self, "output/wdl")(value_dense2_add);  value_dense2_add = None

    # NOT FINALIZED
    return block5_conv2_mixin.reshape(bsz, -1)


class MaiaEmbedder(torch.nn.Module):

    def __init__(self, model_type=MAIA_TYPE.MAIA_1100):
        super(MaiaEmbedder, self).__init__()
        self._model_type = model_type
        self._embed_fn = _maia_embed
        self._embed_size = _MAIA_EMBED_SIZE
        self._model_path = f"models/maia-{model_type.value}.onnx"
        self._model = convert(self._model_path)

    def __call__(self, input):
        return self._embed_fn(self._model, input)

    def serialize_to_dict(self):
        return {"model_type": self._model_type}

    def embed_size(self):
        return self._embed_size

    @classmethod
    def load_from_dict(cls, data):
        return cls(model_type=data["model_type"])


if __name__ == "__main__":
    from leela_board import LeelaBoard

    fen = "5qk1/p1pR1p2/1b3Rp1/4r3/1r6/7P/PP4P1/1B1Q3K b - - 0 32"
    leela_board = LeelaBoard(fen=fen)

    model = convert("models/maia-1500.onnx")

    out = _maia_embed(
        model, torch.from_numpy(leela_board.lcz_features()).float().unsqueeze(0)
    )
    print(torch.from_numpy(leela_board.lcz_features()).float().unsqueeze(0).shape)
    print(out.shape)

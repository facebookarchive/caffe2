from caffe2.python import cnn, core, workspace, dyndep, net_drawer
from caffe2.proto import caffe2_pb2

# basic building block
# Convolution + BatchNorm
def ConvBN(model, input_blob, output_name, num_input, num_output, filter_size, stride, pad, is_test=False):
    conv = model.Conv(
        input_blob,
        output_name + "_conv",
        num_input,
        num_output,
        filter_size,
        ('XavierFill', {}),
        ('ConstantFill', {}),
        stride=stride,
        pad=pad
    )
    keywords = {'is_test' : 1} if is_test else {'is_test' : 0}
    bn = model.SpatialBN(
        conv,
        output_name + "_bn",
        num_output,
        **keywords)
    return bn

def ConvBNRelu(model, input_blob, output_name, num_input, num_output, filter_size, stride, pad, is_test=False):
    bn = ConvBN(model, input_blob, output_name, num_input, num_output, filter_size, stride, pad, is_test)

    bn = model.Relu(bn, bn)

    return bn

def ResNetStem(model, data, is_test=False):
    conv1 = ConvBNRelu(
        model,
        data,
        "conv1",
        num_input=3,
        num_output=64,
        filter_size=7,
        stride=2,
        pad=3,
        is_test=is_test
    )
    pool1 = model.MaxPool(
        conv1,
        "pool1",
        kernel=3,
        stride=2,
        pad=1
    )
    return pool1


# Block for ResNet_{18,34}
def BasicBlock(model, block, num, input_blob, input_filters, intermediate_filters, output_filters, prefix, is_test=False):

    # Use convolution-shortcut (option C) for now
    stride = 2 if (num == 1) and (block != 2) else 1
    conv_shortcut = model.Conv(
        input_blob,
        prefix + "_{}_shortcut".format(num),
        input_filters,
        output_filters,
        1,
        ('XavierFill', {}),
        ('ConstantFill',{}),
        pad=0,
        stride=stride
    )

    conv1 = ConvBNRelu(
        model,
        input_blob,
        prefix + "_{}_1".format(num),
        num_input=input_filters,
        num_output=intermediate_filters,
        filter_size=3,
        stride=stride,
        pad=1,
        is_test=is_test
    )
    conv2 = ConvBN(
        model,
        conv1,
        prefix + "_{}_2".format(num),
        num_input=intermediate_filters,
        num_output=output_filters,
        filter_size=3,
        stride=1,
        pad=1,
        is_test=is_test
    )
    # Bring in shortcut layer
    # Add shortcut (identity) and conv3 
    add = model.Add([conv2, conv_shortcut], prefix+"_{}_sum".format(num))
    add = model.Relu(add, add)

    return add

# Block for ResNet_{50,101,152}
def Bottleneck(model, block, num, input_blob, input_filters, intermediate_filters, output_filters, prefix, is_test=False):

    stride = 2 if (num == 1) and (block != 2) else 1
    # Use convolution-shortcut (option C) for now
    conv_shortcut = model.Conv(
        input_blob,
        prefix + "_{}_shortcut".format(num),
        input_filters,
        output_filters,
        1,
        ('XavierFill', {}),
        ('ConstantFill',{}),
        stride=stride,
        pad=0
    )

    conv1 = ConvBNRelu(
        model,
        input_blob,
        prefix + "_{}_1".format(num),
        num_input=input_filters,
        num_output=intermediate_filters,
        filter_size=1,
        stride=stride,
        pad=0,
        is_test=is_test
    )
    conv2 = ConvBNRelu(
        model,
        conv1,
        prefix + "_{}_2".format(num),
        num_input=intermediate_filters,
        num_output=intermediate_filters,
        filter_size=3,
        stride=1,
        pad=1,
        is_test=is_test
    )
    conv3 = ConvBN(
        model,
        conv2,
        prefix + "_{}_3".format(num),
        num_input=intermediate_filters,
        num_output=output_filters,
        filter_size=1,
        stride=1,
        pad=0,
        is_test=is_test
    )
    # Bring in shortcut layer
    # Add shortcut (identity) and conv3 
    add = model.Add([conv3, conv_shortcut], prefix+"_{}_sum".format(num))
    add = model.Relu(add, add)

    return add

# Resnet 18 + 34
class ResNetSmall():
    @staticmethod
    def CropSize():
        return 224

# def BasicBlock(model, num, input_blob, input_filters, intermediate_filters, output_filters, prefix):
    # default to 34-layer resnet
    @staticmethod
    def Net(model, data, is_test=False, layer_numbers=[3,4,6,3]):
        stem = ResNetStem(model, data, is_test)

        current_input = stem
        current_input_filters = 64

        # ConvBNRelu + ConvBNRelu + ConvBN + EltWise + Relu
        for l in xrange(1, layer_numbers[0]+1):
            conv2 = BasicBlock(
                model,
                2,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=64,
                output_filters=64,
                prefix="conv2",
                is_test=is_test
            )
            current_input_filters = 64
            current_input = conv2
            
        # Next set: 512 filters in/out, 128 intermediate
        for l in xrange(1, layer_numbers[1]+1):
            conv3 = BasicBlock(
                model,
                3,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=128,
                output_filters=128,
                prefix="conv3",
                is_test=is_test
            )
            current_input_filters=128
            current_input=conv3

        # 256 intermediate, 1024 in/out
        for l in xrange(1, layer_numbers[2]+1):
            conv4 = BasicBlock(
                model,
                4,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=256,
                output_filters=256,
                prefix="conv4",
                is_test=is_test
            )
            current_input_filters=256
            current_input=conv4

        # 512 intermediate, 2048 in/out
        for l in xrange(1,layer_numbers[3]+1):
            conv5 = BasicBlock(
                model,
                5,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=512,
                output_filters=512,
                prefix="conv5",
                is_test=is_test
            )
            current_input_filters=512
            current_input=conv5

        # average pool
        pool2 = model.AveragePool(
            current_input,
            "pool2",
            kernel=7,
            stride=1,
            pad=0
        )
        fc = model.FC(
            pool2,
            "fc",
            512,
            1000,
            ('XavierFill', {}),
            ('ConstantFill', {})
        )
        pred = model.Softmax(fc, "pred")

        return pred


class ResNet():
    @staticmethod
    def CropSize():
        return 224

    # default to 50-layer resnet
    @staticmethod
    def Net(model, data, is_test=False, layer_numbers=[3,4,6,3]):
        stem = ResNetStem(model, data, is_test)

        current_input = stem
        current_input_filters = 64
        # ConvBNRelu + ConvBNRelu + ConvBN + EltWise + Relu
        for l in xrange(1, layer_numbers[0]+1):
            conv2 = Bottleneck(
                model,
                2,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=64,
                output_filters=256,
                prefix="conv2",
                is_test=is_test
            )
            current_input_filters = 256
            current_input = conv2
            
        # Next set: 512 filters in/out, 128 intermediate

        for l in xrange(1, layer_numbers[1]+1):
            conv3 = Bottleneck(
                model,
                3,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=128,
                output_filters=512,
                prefix="conv3",
                is_test=is_test
            )
            current_input_filters=512
            current_input=conv3

        # 256 intermediate, 1024 in/out
        for l in xrange(1, layer_numbers[2]+1):
            conv4 = Bottleneck(
                model,
                4,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=256,
                output_filters=1024,
                prefix="conv4",
                is_test=is_test
            )
            current_input_filters=1024
            current_input=conv4

        # 512 intermediate, 2048 in/out
        for l in xrange(1,layer_numbers[3]+1):
            conv5 = Bottleneck(
                model,
                5,
                l,
                current_input,
                input_filters=current_input_filters,
                intermediate_filters=512,
                output_filters=2048,
                prefix="conv5",
                is_test=is_test
            )
            current_input_filters=2048
            current_input=conv5

        # average pool
        pool2 = model.AveragePool(
            current_input,
            "pool2",
            kernel=7,
            stride=1,
            pad=0
        )
        fc = model.FC(
            pool2,
            "fc",
            2048,
            1000,
            ('XavierFill', {}),
            ('ConstantFill', {})
        )
        pred = model.Softmax(fc, "pred")

        return pred

class ResNetHelper():
    def __init__(self, depth):
        self.depth = depth

    def CropSize(self):
        return 224

    def Net(self, model, data, test_phase=False):
        if self.depth == 18:
            return ResNetSmall.Net(model, data, test_phase, [2,2,2,2])
        elif self.depth == 34:
            return ResNetSmall.Net(model, data, test_phase, [3,4,6,3])
        elif self.depth == 50:
            return ResNet.Net(model, data, test_phase, [3,4,6,3])
        elif self.depth == 101:
            return ResNet.Net(model, data, test_phase, [3,4,23,3])
        elif self.depth == 152:
            return ResNet.Net(model, data, test_phase, [3,8,36,3])
        else:
            print('Error - invalid resnet depth specified')



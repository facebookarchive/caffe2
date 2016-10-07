from caffe2.python import cnn, core, workspace, dyndep, net_drawer
from caffe2.proto import caffe2_pb2

# Rethinking the Inception Architecture for Computer Vision
# Port of MXNet definition (https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol_inception-v3.py)

def ConvBNRelu(model, input_blob, output_name, num_input, num_output, filter_size, stride=1, pad=0, is_test=False):
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
    relu = model.Relu(
        output_name + "_bn",
        output_name + "_bn"
    )
    return relu

def Inception7A(model,
                data,
                input_filters,
                num_1x1,
                num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5,
                proj,
                prefix, is_test=False):

    # 1x1 tower
    tower_1x1 = ConvBNRelu(model,
                           data,
                           prefix+"_conv",
                           num_input=input_filters,
                           num_output=num_1x1,
                           filter_size=1,
                           is_test=is_test)
    # 5x5 tower
    tower_5x5 = ConvBNRelu(model,
                           data,
                           prefix+"_tower_conv",
                           num_input=input_filters,
                           num_output=num_5x5_red,
                           filter_size=5,
                           pad=2,
                           is_test=is_test)
    tower_5x5 = ConvBNRelu(model,
                           tower_5x5,
                           prefix+"_tower_conv_1",
                           num_input=num_5x5_red,
                           num_output=num_5x5,
                           filter_size=5,
                           pad=2,
                           is_test=is_test)

    # 3x3 tower
    tower_3x3 = ConvBNRelu(model,
                           data,
                           prefix+"_tower_1_conv",
                           num_input=input_filters,
                           num_output=num_3x3_red,
                           filter_size=3,
                           pad=1,
                           is_test=is_test)
    tower_3x3 = ConvBNRelu(model,
                           tower_3x3,
                           prefix+"_tower_1_conv_1",
                           num_input=num_3x3_red,
                           num_output=num_3x3_1,
                           filter_size=3,
                           pad=1,
                           is_test=is_test)
    tower_3x3 = ConvBNRelu(model,
                           tower_3x3,
                           prefix+"_tower_1_conv_2",
                           num_input=num_3x3_1,
                           num_output=num_3x3_2,
                           filter_size=3,
                           pad=1,
                           is_test=is_test)

    # pooling tower
    pooling = model.AveragePool(data, prefix+"_pool", kernel=3, stride=1, pad=1)
    cproj = ConvBNRelu(model,
                       pooling,
                       prefix+"_tower_2_conv",
                       num_input=input_filters,
                       num_output=proj,
                       filter_size=1,
                       stride=1,
                       is_test=is_test)

    concat = model.Concat([tower_1x1, tower_3x3, tower_5x5, cproj], prefix+"_concat")

    # model, output_filters
    return concat, (num_1x1 + num_5x5 + num_3x3_2 + proj)

# First Downsample
def Inception7B(model,
                data,
                input_filters,
                num_3x3,
                num_d3x3_red, num_d3x3_1, num_d3x3_2,
                prefix,
                is_test=False):
    tower_3x3 = ConvBNRelu(model,
                           data,
                           prefix+"_conv",
                           num_input=input_filters,
                           num_output=num_3x3,
                           filter_size=3,
                           stride=2,
                           pad=0,
                           is_test=is_test)

    tower_d3x3 = ConvBNRelu(model,
                            data,
                            prefix+"_tower_conv",
                            num_input=input_filters,
                            num_output=num_d3x3_red,
                            filter_size=1,
                            pad=0,
                            is_test=is_test)
    tower_d3x3 = ConvBNRelu(model,
                            tower_d3x3,
                            prefix+"_tower_conv_1",
                            num_input=num_d3x3_red,
                            num_output=num_d3x3_1,
                            filter_size=3,
                            pad=1,
                            is_test=is_test)
    tower_d3x3 = ConvBNRelu(model,
                            tower_d3x3,
                            prefix+"_tower_conv_2",
                            num_input=num_d3x3_1,
                            num_output=num_d3x3_2,
                            filter_size=3,
                            pad=0,
                            stride=2,
                            is_test=is_test)

    pooling = model.MaxPool(data, prefix+"_pool", kernel=3, stride=2, pad=0)

    concat = model.Concat([tower_3x3, tower_d3x3, pooling], prefix+"_concat")

    return concat, (num_3x3+num_d3x3_2+input_filters)

def Inception7C(model,
                data,
                input_filters,
                num_1x1,
                num_d7_red, num_d7_1, num_d7_2, # d7_{1,2} correspond to factorized [7,1] [1,7]
                num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4, # q7_{{1,2},{3,4}} are factorized pairs
                proj,
                prefix, is_test=False):
    # 1x1 tower
    tower_1x1 = ConvBNRelu(model,
                           data,
                           prefix+"_conv",
                           num_input=input_filters,
                           num_output=num_1x1,
                           filter_size=1,
                           is_test=is_test)
                       
    # 7x7 tower (1)
    tower_d7 = ConvBNRelu(model,
                          data,
                          prefix+"_tower_conv",
                          num_input=input_filters,
                          num_output=num_d7_red,
                          filter_size=1,
                          pad=0,
                          is_test=is_test)
    # [7x1] [1x7] -> [7x7]
    tower_d7 = ConvBNRelu(model,
                          tower_d7,
                          prefix+"_tower_conv_1",
                          num_input=num_d7_red,
                          num_output=num_d7_2,
                          filter_size=7,
                          pad=3,
                          is_test=is_test)
    # Pair of [7x1][1x7] conv
    tower_q7 = ConvBNRelu(model,
                          data,
                          prefix+"_tower_1_conv",
                          num_input=input_filters,
                          num_output=num_q7_red,
                          filter_size=1,
                          pad=0,
                          is_test=is_test)
    tower_q7 = ConvBNRelu(model,
                          tower_q7,
                          prefix+"_tower_1_conv_1",
                          num_input=num_q7_red,
                          num_output=num_q7_2,
                          filter_size=7,
                          pad=3,
                          is_test=is_test)
    tower_q7 = ConvBNRelu(model,
                          tower_q7,
                          prefix+"_tower_1_conv_3",
                          num_input=num_q7_2,
                          num_output=num_q7_4,
                          filter_size=7,
                          pad=3,
                          is_test=is_test)

    # pooling
    pooling = model.AveragePool(data, prefix+"_pool", kernel=3, stride=1, pad=1)
    # projection
    cproj = ConvBNRelu(model,
                      pooling,
                      prefix+"_tower_2_conv",
                      num_input=input_filters,
                      num_output=proj,
                      filter_size=1,
                      is_test=is_test)

    concat = model.Concat([tower_1x1, tower_d7, tower_q7, cproj], prefix+"_concat")

    return concat, (num_1x1+num_d7_2+num_q7_4+proj)

def Inception7D(model,
                data,
                input_filters,
                num_3x3_red, num_3x3,
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3,
                prefix, is_test=False):
    # tower 3x3
    tower_3x3 = ConvBNRelu(model,
                           data,
                           prefix+"_conv",
                           num_input=input_filters,
                           num_output=num_3x3_red,
                           filter_size=1,
                           is_test=is_test)
    tower_3x3 = ConvBNRelu(model,
                           tower_3x3,
                           prefix+"_conv_1",
                           num_input=num_3x3_red,
                           num_output=num_3x3,
                           filter_size=3,
                           pad=0,
                           stride=2,
                           is_test=is_test)

    # tower d7
    tower_d7_3x3 = ConvBNRelu(model,
                              data,
                              prefix+"_tower_1_conv",
                              num_input=input_filters,
                              num_output=num_d7_3x3_red,
                              filter_size=1,
                              is_test=is_test)
    tower_d7_3x3 = ConvBNRelu(model,
                              tower_d7_3x3,
                              prefix+"_tower_1_conv_1",
                              num_input=num_d7_3x3_red,
                              num_output=num_d7_2,
                              filter_size=7,
                              pad=3,
                              is_test=is_test)
    tower_d7_3x3 = ConvBNRelu(model,
                              tower_d7_3x3,
                              prefix+"_tower_1_conv_3",
                              num_input=num_d7_2,
                              num_output=num_d7_3,
                              filter_size=3,
                              pad=0,
                              stride=2,
                              is_test=is_test)

    pooling = model.MaxPool(data, prefix+"_pool", kernel=3, stride=2)

    concat = model.Concat([tower_3x3, tower_d7_3x3, pooling], prefix+"_concat")

    return concat, (num_3x3+num_d7_3+input_filters)

def Inception7E(model,
                data,
                input_filters,
                num_1x1,
                num_d3_red, num_d3_1, num_d3_2,
                num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2,
                pool, proj,
                prefix, is_test = False):
    # 1x1 tower
    tower_1x1 = ConvBNRelu(model,
                           data,
                           prefix+"_conv",
                           num_input=input_filters,
                           num_output=num_1x1,
                           filter_size=1,
                           is_test=is_test)
    # d3 tower
    tower_d3 = ConvBNRelu(model,
                          data,
                          prefix+"_tower_conv",
                          num_input=input_filters,
                          num_output=num_d3_red,
                          filter_size=1,
                          is_test=is_test)

    tower_d3_ab = ConvBNRelu(model,
                             tower_d3,
                             prefix+"_tower_mixed_conv",
                             num_input=num_d3_red,
                             num_output=num_d3_2,
                             filter_size=3,
                             pad=1,
                             is_test=is_test)

    tower_3x3_d3 = ConvBNRelu(model,
                               data,
                               prefix+"_tower_1_conv",
                               num_input=input_filters,
                               num_output=num_3x3_d3_red,
                               filter_size=1,
                               is_test=is_test)
    tower_3x3_d3 = ConvBNRelu(model,
                               tower_3x3_d3,
                               prefix+"_tower_1_conv_1",
                               num_input=num_3x3_d3_red,
                               num_output=num_3x3,
                               filter_size=3,
                               pad=1,
                               is_test=is_test)
    tower_3x3_d3_ab = ConvBNRelu(model,
                                  tower_3x3_d3,
                                  prefix+"_tower_1_mixed_conv_12",
                                  num_input=num_3x3,
                                  num_output=num_3x3_d3_2,
                                  filter_size=3,
                                  pad=1,
                                  is_test=is_test)

    pool_fn = model.MaxPool if (pool=="MAX") else model.AveragePool

    pooling = pool_fn(data, prefix+"_pool", kernel=3, pad=1, stride=1)
    cproj = ConvBNRelu(model,
                      pooling,
                      prefix+"_tower_2_conv",
                      num_input=input_filters,
                      num_output=proj,
                      filter_size=1,
                      is_test=is_test)

    concat = model.Concat([tower_1x1, tower_d3_ab, tower_3x3_d3_ab, cproj], prefix+"_concat")

    return concat, (num_1x1+num_d3_2+num_3x3_d3_2+proj)

class Inception_v3():
    @staticmethod
    def CropSize():
        return 299

    @staticmethod
    def Net(model, data, test_phase=False):

        # stage1
        conv = ConvBNRelu(model,
                          data,
                          "conv",
                          num_input=3,
                          num_output=32,
                          filter_size=3,
                          stride=2,
                          is_test=test_phase)
        conv_1 = ConvBNRelu(model,
                            conv,
                            "conv_1",
                            num_input=32,
                            num_output=32,
                            filter_size=3,
                            stride=1,
                            is_test=test_phase)
        conv_2 = ConvBNRelu(model,
                            conv_1,
                            "conv_2",
                            num_input=32,
                            num_output=64,
                            filter_size=3,
                            stride=1,
                            pad=1,
                            is_test=test_phase)
        pool = model.MaxPool(conv_2,
                                "pool",
                                kernel=3,
                                stride=2)

        # stage 2
        conv_3 = ConvBNRelu(model,
                            pool,
                            "conv_3",
                            num_input=64,
                            num_output=80,
                            filter_size=1,
                            stride=1,
                            is_test=test_phase)
        conv_4 = ConvBNRelu(model,
                            conv_3,
                            "conv_4",
                            num_input=80,
                            num_output=192,
                            filter_size=3,
                            stride=1,
                            pad=1,
                            is_test=test_phase)
        pool1 = model.MaxPool(conv_4,
                              "pool1",
                              kernel=3,
                              stride=2)

        # stage 3
        in3a, outputs = Inception7A(model,
                           pool1,
                           192,
                           64,
                           64, 96, 96,
                           48, 64,
                           32,
                           prefix="mixed",
                           is_test=test_phase)

        in3b, outputs = Inception7A(model,
                           in3a,
                           outputs,
                           64,
                           64, 96, 96,
                           48, 64,
                           64,
                           prefix="mixed_1",
                           is_test=test_phase)

        in3c, outputs = Inception7A(model,
                           in3b,
                           outputs,
                           64,
                           64, 96, 96,
                           48, 64,
                           64,
                           prefix="mixed_2",
                           is_test=test_phase)

        in3d, outputs = Inception7B(model,
                           in3c,
                           outputs,
                           384,
                           64, 96, 96,
                           prefix="mixed_3",
                           is_test=test_phase)

        # stage 4
        in4a, outputs = Inception7C(model,
                           in3d,
                           outputs,
                           192,
                           128, 128, 192,
                           128, 128, 128, 128, 192,
                           192,
                           prefix="mixed_4",
                           is_test=test_phase)

        in4b, outputs = Inception7C(model,
                           in4a,
                           outputs,
                           192,
                           160, 160, 192,
                           160, 160, 160, 160, 192,
                           192,
                           prefix="mixed_5",
                           is_test=test_phase)

        in4c, outputs = Inception7C(model,
                           in4b,
                           outputs,
                           192,
                           160, 160, 192,
                           160, 160, 160, 160, 192,
                           192,
                           prefix="mixed_6",
                           is_test=test_phase)

        in4d, outputs = Inception7C(model,
                           in4c,
                           outputs,
                           192,
                           192, 192, 192,
                           192, 192, 192, 192, 192,
                           192,
                           prefix="mixed_7",
                           is_test=test_phase)

        in4e, outputs = Inception7D(model,
                           in4d,
                           outputs,
                           192, 320,
                           192, 192, 192, 192,
                           prefix="mixed_8",
                           is_test=test_phase)

        # Stage 5
        in5a, outputs = Inception7E(model,
                           in4e,
                           outputs,
                           320,
                           384, 384, 384,
                           448, 384, 384, 384,
                           "AVG",
                           192,
                           prefix="mixed_9",
                           is_test=test_phase)

        in5b, outputs = Inception7E(model,
                           in5a,
                           outputs,
                           320,
                           384, 384, 384,
                           448, 384, 384, 384,
                           "MAX",
                           192,
                           prefix="mixed_10",
                           is_test=test_phase)

        pool = model.AveragePool(in5b, "global_pool", kernel=8, stride=1)

        fc = model.FC(pool, "fc", outputs, 1000)

        pred = model.Softmax(fc, "pred")

        return pred

## @package resnet50_trainer
# Module caffe2.python.examples.resnet50_trainer
import logging

from caffe2.python import trainer, workspace

import caffe2.python.models.resnet as resnet

logging.basicConfig()
log = logging.getLogger("resnet50_trainer")
log.setLevel(logging.DEBUG)

# Model building functions
def create_resnet50_model_ops(model, loss_scale):
    [softmax, loss] = resnet.create_resnet50(
        model,
        "data",
        num_input_channels=args.num_channels,
        num_labels=args.num_labels,
        label="label",
        no_bias=True,
    )
    loss = model.Scale(loss, scale=loss_scale)
    model.Accuracy([softmax, "label"], "accuracy")
    return [loss]


def main():
    parser = trainer.get_args_parser()

    args = parser.parse_args()

    Train(args, create_resnet50_model_ops)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()

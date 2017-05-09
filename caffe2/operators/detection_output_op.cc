#include "caffe2/operators/detection_output_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(DetectionOutput, DetectionOutputOp<float, CPUContext>);
NO_GRADIENT(DetectionOutput);

OPERATOR_SCHEMA(DetectionOutput)
	.NumInputs(3)
	.NumOutputs(1)
	.SetDoc(R"DOC(DetectionOutputLayer in SSD)DOC")
	.Arg("num_classes", "num_classes")
	.Arg("share_location", "share_location")
	.Arg("background_label_id", "background_label_id")
	.Arg("nms_threshold", "nms_threshold")
	.Arg("top_k", "top_k")
	.Arg("eta", "eta")
	.Arg("code_type", "code_type")
	.Arg("variance_encoded_in_target", "variance_encoded_in_target")
	.Arg("keep_top_k", "keep_top_k")
	.Arg("confidence_threshold", "confidence_threshold")
	.Input(0, "loc", "loc")
	.Input(1, "conf", "conf")
	.Input(2, "prior", "prior")
	.Output(0, "Y", "detection results");
} // namespace 

} // namespace caffe2

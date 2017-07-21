# Docker images

## CUDA Image

Build like: `docker build -f docker/Dockerfile.cuda -t caffe2:cuda .`

Run like: `nvidia-docker run --rm -it caffe2:cuda python -m caffe2.python.operator_test.relu_op_test`

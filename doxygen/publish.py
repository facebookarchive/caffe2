## @package publish
# Module doxygen.publish
import os, shutil
if os.path.exists("/Users/aaronmarkham/caffe2/doxygen-c"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("/Users/aaronmarkham/caffe2/doxygen-c")
if os.path.exists("/Users/aaronmarkham/caffe2/doxygen-python"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("/Users/aaronmarkham/caffe2/doxygen-python")


os.system("cp -rf doxygen-c /Users/aaronmarkham/caffe2/")
os.system("cp -rf doxygen-python /Users/aaronmarkham/caffe2/")

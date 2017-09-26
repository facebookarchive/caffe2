## @package process
# Module docs.process
# Script to insert preamble for doxygen and regen API docs
# Assumes you are in the master branch
# Usage: python process.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import glob, os, shutil

# Module caffe2...caffe2.python.control_test
# If you rename a module, be sure to update the preamble!
# Otherwise this script will add a new one
def insert(originalfile, first_line, description):
    with open(originalfile,'r') as f:
        f1 = f.readline()
        if(f1.find(first_line)<0):
            docs = first_line + description + f1
            with open('newfile.txt','w') as f2:
                f2.write(docs)
                f2.write(f.read())
            os.rename('newfile.txt',originalfile)
        else:
            print('already inserted')

# move up from /caffe2_root/docs
os.chdir("..")

# Insert the doxygen preamble where needed
for root, dirs, files in os.walk("."):
    for file in files:
        if (file.endswith(".py") and not file.endswith("_test.py") and not file.endswith("__.py")):
            filepath = os.path.join(root, file)
            print("filepath: {}".format(filepath))
            directory = os.path.dirname(filepath)[2:]
            directory = directory.replace("/",".")
            print("directory: {}".format(directory))
            name = os.path.splitext(file)[0]
            first_line = "## @package " + name
            description = "\n# Module " + directory + "." + name + "\n"
            print(first_line, description)
            insert(filepath, first_line, description)

if os.path.exists("build/docs/doxygen-python"):
    print("Looks like you ran this before, so we need to cleanup those old Python API files...")
    shutil.rmtree("build/docs/doxygen-python")

os.makedirs("build/docs/doxygen-python")

if os.path.exists("build/docs/doxygen-c"):
    print("Looks like you ran this before, so we need to cleanup those old C++ API files...")
    shutil.rmtree("build/docs/doxygen-c")

os.makedirs("build/docs/doxygen-c")

# Generate the docs
print("Generating Python API Docs...")
os.system("doxygen docs/Doxyfile-python")
print("Generationg C++ API Docs...")
os.system("doxygen docs/Doxyfile-c")

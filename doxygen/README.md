# Doxygen Notes

In the testing of Doxygen it was discovered that it behaved better if the processing between C++ and Python was split up. This is why there are two different links to cover each API.

C++ API docs work out of the box with the Caffe2 source code. Python docs require “python blocks (http://www.stack.nl/~dimitri/doxygen/manual/docblocks.html#pythonblocks)” which are currently missing in the Python code.

The Python script called “process.py” that resides in the /doxygen folder is to prepare the docs by looking for the block and if it doesn't exist prepend and customize the python blocks section with the module's path (e.g. Module caffe2.python.examples.char_rnn). It was noted that you need to delete the previous version of docs when you regenerate the docs or else things get messy, so the script deals with that as well.

The doxygen customization includes these files in the doxygen folder:

* header.html - logo links back to the main docs page
* footer.html - includes the Facebook OSS footer
* stylesheet.css - doxygen's default CSS; tweaked to fix formatting problems with the custom logo, header, and footer
* main.css - copied from the caffe2ai CSS, so this should be refreshed after the design changes (this overrides/extends stylesheet.css)

It also extracts info from markdown files found in the source tree. A legacy installation file was in the /docs folder and this was removed. These file show up in the top navigation under “Related Pages”.

The flow to create the API documents is simple now:

1. Run /caffe2_root/doxygen/process.py
2. Copy the doxygen-c and doxygen-python folders created by the script to the gh-pages branch.

Settings that were customized:

OPTIMIZE_OUTPUT_JAVA - turned on for Python config, off for C++ config
USE_MDFILE_AS_MAINPAGE  - use to flag a markdown file for the mainpage
EXTRACT_ALL
QUIET
WARN_IF_UNDOCUMENTED
FILE_PATTERNS
DOT_MULTI_TARGETS = YES
JAVADOC_AUTOBRIEF = YES
QUIET = YES
SOURCE_BROWSER = YES
VERBATIM_HEADERS = NO
SHOW_NAMESPACES = NO for C++ config

Not using this (was in old config file, but seems to be for Latex):
EXTRA_PACKAGES = amsmath \
amsfonts \
xr

### NOTE / TODO:

useful for xcode, currently off
GENERATE_DOCSET = NO

Look at search engine integration, xml output, etc
EXTERNAL_SEARCH = YES


### process.py

```
## @package process
# Module doxygen.process
# Script to insert preamble for doxygen and regen API docs

import glob, os, shutil

# Module caffe2...caffe2.python.control_test
def insert(originalfile,first_line,description):
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

# move up from /caffe2_root/doxygen
os.chdir("..")
os.system("git checkout caffe2/python/.")

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            print("filepath: " + filepath)
            directory = os.path.dirname(filepath)[2:]
            directory = directory.replace("/",".")
            print "directory: " + directory
            name = os.path.splitext(file)[0]
            first_line = "## @package " + name
            description = "\n# Module " + directory + "." + name + "\n"
            print first_line,description
            insert(filepath,first_line,description)

if os.path.exists("doxygen/doxygen-python"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("doxygen/doxygen-python")
else:
    os.makedirs("doxygen/doxygen-python")

if os.path.exists("doxygen/doxygen-c"):
    print("Looks like you ran this before, so we need to cleanup those old files...")
    shutil.rmtree("doxygen/doxygen-c")
else:
    os.makedirs("doxygen/doxygen-c")

os.system("doxygen .Doxyfile-python")
os.system("doxygen .Doxyfile-c")
```

### Other Notes

To achieve better output in the Python docs:
http://stackoverflow.com/questions/7690220/how-to-document-python-function-parameter-types

Swap this kind of formatting into py files:

```
def my_method(x, y):"""
    my_method description

    @type x: int
    @param x: An integer

    @type y: int|string
    @param y: An integer or string

    @rtype: string
    @return: Returns a sentence with your variables in it
    """return "Hello World! %s, %s" % (x,y)
```

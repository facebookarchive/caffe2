# Caffe2 Benchmark Framework

The Caffe2 benchmark framework is used to provide Caffe2 runtime inferencing speed information on the host and Android platforms for various models.

## Stand alone benchmark run
The `harness.py` is the entry point for one benchmark run. It collects the runtime for an entire net and/or individual operator, and saves the data locally or push to remote server. The usage of the script is as follows:

<pre>
usage: harness.py [-h] [--android] [--host] [--local_reporter LOCAL_REPORTER]
                  [--remote_reporter REMOTE_REPORTER]
                  [--remote_access_token REMOTE_ACCESS_TOKEN] --net NET
                  --init_net INIT_NET [--input INPUT]
                  [--input_file INPUT_FILE] [--input_dims INPUT_DIMS]
                  [--output OUTPUT] [--output_folder OUTPUT_FOLDER]
                  [--warmup WARMUP] [--iter ITER] [--run_individual]
                  [--program PROGRAM] [--git_commit GIT_COMMIT]
                  [--exec_base_dir EXEC_BASE_DIR]

Perform one benchmark run

optional arguments:
  -h, --help            show this help message and exit
  --android             Run the benchmark on all collected android devices.
  --host                Run the benchmark on the host.
  --local_reporter LOCAL_REPORTER
                        Save the result to a directory specified by this
                        argument.
  --remote_reporter REMOTE_REPORTER
                        Save the result to a remote server. The style is
                        <domain_name>/<endpoint>|<category>
  --remote_access_token REMOTE_ACCESS_TOKEN
                        The access token to access the remote server
  --net NET             The given predict net to benchmark.
  --init_net INIT_NET   The given net to initialize any parameters.
  --input INPUT         Input that is needed for running the network. If
                        multiple input needed, use comma separated string.
  --input_file INPUT_FILE
                        Input file that contain the serialized protobuf for
                        the input blobs. If multiple input needed, use comma
                        separated string. Must have the same number of items
                        as input does.
  --input_dims INPUT_DIMS
                        Alternate to input_files, if all inputs are simple
                        float TensorCPUs, specify the dimension using comma
                        separated numbers. If multiple input needed, use
                        semicolon to separate the dimension of different
                        tensors.
  --output OUTPUT       Output that should be dumped after the execution
                        finishes. If multiple outputs are needed, use comma
                        separated string. If you want to dump everything, pass
                        "\*" as the output value.
  --output_folder OUTPUT_FOLDER
                        The folder that the output should be written to. This
                        folder must already exist in the file system.
  --warmup WARMUP       The number of iterations to warm up.
  --iter ITER           The number of iterations to run.
  --run_individual      Whether to benchmark individual operators.
  --program PROGRAM     The program to run on the platform.
  --git_commit GIT_COMMIT
                        The git commit on this benchmark run.
  --exec_base_dir EXEC_BASE_DIR
                        The base directory of the commit that the program is
                        built from
</pre>

## Continuous benchmark run
The `git_driver.py` is the entry point to run the benchmark continuously. It repeatedly pulls the Caffe2 from github, builds the Caffe2, and launches the `harness.py` for every line specified in the config file. In the config file, each line contains the command line arguments passed to `harness.py`. An example of config file is as follows:

<pre>
--init_net <models_dir>/squeezenet/squeeze_init_net.pb --net <models_dir>/squeezenet/squeeze_predict_net.pb --input_dims 64,3,30,30 --input data --run_individual --warmup 10 --iter 10
</pre>

The `<models_dir>` is a placeholder that is replaced by the actual model directory specified in the command line `--models_dir`.

The accepted arguments are as follows:

<pre>
usage: git_driver.py [-h] --config CONFIG --models_dir MODELS_DIR --git_dir
                     GIT_DIR [--git_commit GIT_COMMIT] [--host] [--android]
                     [--interval INTERVAL] [--status_file STATUS_FILE]
                     [--git_repository GIT_REPOSITORY]
                     [--git_branch GIT_BRANCH]

Perform one benchmark run

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The test config file containing all the tests to run
  --models_dir MODELS_DIR
                        The root directory that all models resides.
  --git_dir GIT_DIR     The base git directory.
  --git_commit GIT_COMMIT
                        The git commit this benchmark runs on. It can be a
                        branch.
  --host                Run the benchmark on the host.
  --android             Run the benchmark on all collected android devices.
  --interval INTERVAL   The minimum time interval in seconds between two
                        benchmark runs.
  --status_file STATUS_FILE
                        A file to inform the driver stops running when the
                        content of the file is 0.
  --git_repository GIT_REPOSITORY
                        The remote git repository.
  --git_branch GIT_BRANCH
                        The remote git repository branch.
</pre>

The `git_driver.py` can also take the arguments that are recognized by `harness.py`. It just passes those arguments over.

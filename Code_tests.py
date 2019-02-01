# /usr/bin/python
#
# Copyright 2019, Fluid Numerics, LLC
#
# All Rights Reserved.
#
# This code is meant to conduct build and run tests of IPE
# These tests verify that the code can be built and can be run
# to completion. Results are written to json file that include
# any error codes encountered in either step of the process.
#
# This code takes a yaml file, in the format shown in conf/consistency-tests.yaml
# This file specifies flavors of IPE to build and the details necessary
# to build them on an HPC cluster that uses environment modules.
#


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import json
import os
import subprocess
import shlex
import pprint
import time


#[ START get_configurations ]
def get_configurations():

  parser = ArgumentParser( description='Conduct build and run tests of IPE and provide summary reports',
                           formatter_class=ArgumentDefaultsHelpFormatter )
  parser.add_argument('-c', '--config', 
                      help='.yaml file containing the configurations to build test and the examples to execute for run tests and benchmarking',
                      type=str,
                      required=True )
  args=parser.parse_args()

  with open(args.config, 'r' ) as stream:
    configs = yaml.load(stream)

  return configs

#[ END get_configurations ]

#[ START download_and_setup_repo ]
def download_and_setup_repo( configs ):

  setup_path = configs["home"]+"/"+configs["setup_path"]
  repo_path  = setup_path + "/"+configs["ipe_repo"]

  # Make the path to the directory where the repo will be cloned
  subprocess.call(['mkdir','-p', setup_path])

  # If the repo_path exists, we'll remove it to prepare for a clean download
  if os.path.isdir( repo_path ):
    subprocess.call(['rm','-r', repo_path])

  # Clone the repo and only the specified branch
  subprocess.call(['git','clone', configs["ipe_url"], repo_path, '--single-branch','--branch', configs["ipe_branch"]])

#[ END download_and_setup_repo ]

#[ START build_flavors ]
def build_flavors( configs ):

  setup_path   = configs["home"]+"/"+configs["setup_path"]
  install_base = setup_path+"/"+configs["ipe_install"]
  repo_path    = setup_path+"/"+configs["ipe_repo"]


  # Setup the install directory.
  # Underneath, each build flavor will have a subdirectory
  if os.path.isdir( install_base ):
    subprocess.call(['rm','-r', install_base])

  subprocess.call(['mkdir','-p', install_base])


  build_success = True
  build_results = []
  k = 0
  for config in configs["configs"]:
    k += 1
    print( "Building {0} [{1}/{2}]".format(config["config"]["name"], k, len(configs["configs"])) )   
    
    # Make a directory for this particular config
    subprocess.call(['mkdir','-p', install_base+"/"+config["config"]["name"]])

    # Place a build script in this directory
    build_instructions = """
#/bin/bash

module load {0}
cd {1}
autoreconf --install
make distclean
./configure {2} \\
             --prefix={3}
make
make install

""".format( config["config"]["dep_modules"],
            repo_path,
            configs["config_flags"]+" "+config["config"]["build_flags"],
            install_base+"/"+config["config"]["name"] )

    f = open( install_base+"/"+config["config"]["name"]+"/build.sh", 'w')
    f.write( build_instructions )
    f.close()
    os.chmod( install_base+"/"+config["config"]["name"]+"/build.sh", 0o755)

    # Execute the build script, and wait for it to complete.
    # stderr and stdout are directed in build.log underneath this install directory
    logfile = open( install_base+"/"+config["config"]["name"]+"/build.log", 'w')
    build_process = subprocess.Popen( ["sh", install_base+"/"+config["config"]["name"]+"/build.sh"],
                                      stdout=logfile,stderr=subprocess.STDOUT )
    streamdata = build_process.communicate()[0]    
    build_results.append( {config["config"]["name"]:[{'build':{'error_code': build_process.returncode, 
                                                               'log_file':install_base+"/"+config["config"]["name"]+'/build.log'} }] } )
    if build_process.returncode != 0:
      build_success = False

    logfile.close()

  return build_results, build_success

#[ END build_flavors ]

#[ START submit_jobs ]
def submit_jobs( configs ):

  cwd = os.getcwd()
  setup_path   = configs["home"]+"/"+configs["setup_path"]
  ipe_path    = setup_path +"/"+configs["ipe_repo"]
  install_base = setup_path+"/"+configs["ipe_install"]
  repo_path    = setup_path+"/"+configs["ipe_repo"]

  for config in configs["configs"]:
    for test in configs["tests"]:

      # Determine if mpi is used in this configuration
      if "openmpi" in config["config"]["dep_modules"] or "mvapich" in config["config"]["dep_modules"]:
        have_mpi="yes"
      else:
        have_mpi="no"
  
      # Make a directory for the model output
      run_path = install_base+"/"+config["config"]["name"]+"/"+test["test"]["name"]
      print( 'Running test under ' + run_path )

      subprocess.call(['mkdir','-p', run_path])

      # Copy data into this directory #
      subprocess.call(['cp',test["test"]["data_directory"]+"/*",run_path])
  
      if configs["scheduler"] == "sbatch":
        run_instructions = """
#/bin/bash

export OMP_NUM_THREADS={0}
export INS_DIR="{1}"
export RUN_DIR="{2}"
export MODULES="{3}"
export IPEDIR="{4}"
export HAVE_MPI="{5}"
export TEST_CASE="{6}"
export N_MPI_RANKS={15}

sbatch --export=ALL \\
       --nodes={7} \\
       --ntasks={8} \\
       --ntasks-per-node={9} \\
       --cpus-per-task={10} \\
       --partition={11} \\
       --time={12} \\
       --job-name="{13}" \\
       --output="{14}" \\
       {16}/ipe.modelrun

""".format( config["config"]["omp_num_threads"],
            install_base+"/"+config["config"]["name"],
            run_path,
            config["config"]["dep_modules"],
            ipe_path,
            have_mpi,
            test["test"]["name"],
            config["config"]["num_nodes"],
            config["config"]["num_tasks"],
            config["config"]["num_tasks_per_node"],
            config["config"]["cpus_per_task"],
            config["config"]["partition"],
            config["config"]["job_wall_time"],
            "ipe-"+configs["ipe_branch"]+"-"+config["config"]["name"]+"-"+test["test"]["name"],
            run_path +"/run.log", 
            config["config"]["num_mpi_ranks"],
            cwd )
      elif configs["scheduler"] == "qsub":
        run_instructions = """
#/bin/bash

qsub -v OMP_NUM_THREADS={0},INS_DIR="{1}",RUN_DIR="{2}",MODULES="{3}",IPEDIR="{4}",HAVE_MPI="{5}",TEST_CASE="{6}",N_MPI_RANKS={7}  \\
     -l nodes={7}:ppn={8},walltime={9} \\
     -q {10} \\
     -N {11} \\
     -o {12} \\
     {13}/ipe.modelrun

""".format( config["config"]["omp_num_threads"],
            install_base+"/"+config["config"]["name"],
            run_path,
            config["config"]["dep_modules"],
            ipe_path,
            have_mpi,
            test["test"]["name"],
            config["config"]["num_mpi_ranks"],
            config["config"]["num_nodes"],
            config["config"]["num_tasks_per_node"],
            config["config"]["job_wall_time"],
            config["config"]["partition"],
            "ipe-"+configs["ipe_branch"]+"-"+config["config"]["name"]+"-"+test["test"]["name"],
            run_path +"/run.log", 
            cwd )

      else if configs["scheduler"] == "none"
        run_instructions = """
#/bin/bash

export OMP_NUM_THREADS={0}
export INS_DIR="{1}"
export RUN_DIR="{2}"
export MODULES="{3}"
export IPEDIR="{4}"
export HAVE_MPI="{5}"
export TEST_CASE="{6}"
export N_NODES={7}
export N_TASKS={8}
export N_TASKS_PER_NODE={9}
export CPUS_PER_TASK={10}
export N_MPI_RANKS={11}

bash {12}/ipe.modelrun.run > {13}

""".format( config["config"]["omp_num_threads"],
            install_base+"/"+config["config"]["name"],
            run_path,
            config["config"]["dep_modules"],
            ipe_path,
            have_mpi,
            test["test"]["name"],
            config["config"]["num_nodes"],
            config["config"]["num_tasks"],
            config["config"]["num_tasks_per_node"],
            config["config"]["cpus_per_task"],
            config["config"]["num_mpi_ranks"],
            cwd, run_path +"/run.log") 

      f = open( run_path+"/exe_job.sh", 'w')
      f.write( run_instructions )
      f.close()
      os.chmod( run_path+"/exe_job.sh", 0o755)

      # Submit the job
      subprocess.call( ["sh",run_path+"/exe_job.sh"] )

 
#[ END submit_jobs ]

#[ START check_job_status ]
def check_job_status( configs, results ):

  setup_path   = configs["home"]+"/"+configs["setup_path"]
  install_base = setup_path+"/"+configs["ipe_install"]

  for config in configs["configs"]:
    for test in configs["tests"]:

      # Make a directory for the model output
      run_path = install_base+"/"+config["config"]["name"]+"/"+test["test"]["name"]
      print( "Check for "+run_path +"/run.check" )

      while True:
     
        if os.path.isfile( run_path + "/run.check" ):

          f = open( run_path + "/run.check" )
          stat = int( f.readline().strip('\n') )
          results.append( { config["config"]["name"]: [{ 'test':[{'name':test["test"]["name"],'error_code': stat}] }] } )
          f.close()
          break 

        else:

          time.sleep( 10 )


#[ END check_job_status ]


# main #


configs = get_configurations()
test_path = configs["home"]+"/"+configs["setup_path"]

if not os.path.exists(test_path):
  download_and_setup_repo( configs )

results, build_success = build_flavors( configs )

if build_success:
  submit_jobs( configs )
  check_job_status( configs, results )


with open(test_path + '/results.json', 'w') as outfile:
    json.dump(results, outfile)


#!/bin/bash
# Configure the resources required
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -n 1              	                                # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 1              	                                # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=0:3:0                                        # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                      	# generic resource required (here requires 4 GPUs)
#SBATCH --mem=16GB                                              # specify memory required per node (here set to 16 GB)
# Configure notifications 
#SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=sunlibocs@gmail.com@gmail.com                    		# Email to which notification will be sent


python /fast/users/a1746546/code/cspnTest/main.py




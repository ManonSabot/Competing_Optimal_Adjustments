no_conda_msg1="Anaconda is not installed. Install it and try again!"
no_conda_msg2="To install Anaconda, head to: https://docs.anaconda.com/anaconda/install/"

all: check_conda create_env convert_files_format

check_conda:
	@conda info --envs || (echo ${no_conda_msg1}; echo ${no_conda_msg2}; exit 1)

create_env:
	@conda env create -f src/extra/competing_opts.yml
	@conda activate competing_opts || @source activate competing_opts

convert_files_format:
	- src/preparation/convert_npy_2_csv.py
	rm -f input/obs_test/*.npy
	rm -f input/canopy_test/*.npy

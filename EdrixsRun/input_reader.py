__all__ = ['readf', 'arg_input']
# Empty imports
import argparse


def readf(fname='RIXS.input'):
	"""

	Return atomic spin-orbit coupling matrix :math:`\\vec{l}\\cdot\\vec{s}`
	in complex spherical harmonics basis.

	Parameters
	----------
	fname: str
		Filename of the input-file

	Returns
	-------
	inpu_params: dict
		Dictionary with all input parameter as a str-list
	"""
	input_params = {}
	with open(fname) as file:
		for line in file:
			if line.find(":") == 0:  # Checking for input-command
				tmp_line = line[1:].split()  # Splitting the input-command
				# Adding the formally empty dictionary
				input_params.update({tmp_line[0].lower(): tmp_line[1:]})
	return input_params


def arg_input():
	# Reading the input of the command-line, respectively, the terminal
	despription_init = "Reading the input-file for edrixs-run"
	parser = argparse.ArgumentParser(description=despription_init)
	parser.add_argument('-i', '--inflile', required=True,
	                    help="The name of the inputfile", dest='fname')
	# parser.add_argument('-d', '--debug', help="Activating debug-mode including wave-function-printing",
	#                    default=False, action='store_true', dest='debug')
	parser.add_argument('-p', '--plot', help="Deactivating the plot-mode",
	                    default=True, action='store_false', dest='plot')
	parser.add_argument('-s', '--show', help="Deactivating the plot-showing-mode",
	                    default=True, action='store_false', dest='show')
	parser.add_argument('-t', '--txt', help="Deactivating the txt-file-saving",
	                    default=True, action='store_false', dest='txtfile')
	parser.add_argument('-w', '--wavefunction', help="Activating the wave-function-printing",
	                    default=False, action='store_true', dest='wave')
	parser.add_argument('-f', '--fname', type=str, default='output', help="Filename of the outputs (default: output)"
	                    , dest='fname')
	args = parser.parse_args()
	input_params = readf(fname=args.fname)
	return input_params, args


# For later for transforming string in real boolean-expressions
def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
	arg_input()

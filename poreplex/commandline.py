#
# Copyright (c) 2018-2019 Institute for Basic Science
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

import argparse
import sys
import os
import yaml
import shutil
import logging
from functools import partial
from . import *
from .pipeline import ProcessingSession
from .utils import *

VERSION_STRING = """\
poreplex version {version}
Written by Hyeshik Chang <hyeshik@snu.ac.kr>.

Copyright (c) 2018-2019 Institute for Basic Science""".format(version=__version__)

def show_banner():
    print("""
\x1b[1mPoreplex\x1b[0m version {version} by Hyeshik Chang <hyeshik@snu.ac.kr>
- Cuts nanopore direct RNA sequencing data into bite-size pieces for RNA Biology
""".format(version=__version__))

class VersionAction(argparse.Action):

    def __init__(self, option_strings, dest, default=None, required=False,
                 help=None, metavar=None):
        super(VersionAction, self).__init__(
            option_strings=option_strings, dest=dest, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(VERSION_STRING)
        parser.exit()

def load_config(args):
    presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
    if not args.config:
        config_path = os.path.join(presets_dir, 'rna-r941.cfg')
    elif os.path.isfile(args.config):
        config_path = args.config
    elif os.path.isfile(os.path.join(presets_dir, args.config + '.cfg')):
        config_path = os.path.join(presets_dir, args.config + '.cfg')
    else:
        errx('ERROR: Cannot find a configuration in {}.'.format(args.config))

    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    kmer_models_dir = os.path.join(os.path.dirname(__file__), 'kmer_models')
    if not os.path.isabs(config['kmer_model']):
        config['kmer_model'] = os.path.join(kmer_models_dir, config['kmer_model'])

    return config

def init_logging(config):
    logfile = os.path.join(config['outputdir'], 'poreplex.log')
    logger = logging.getLogger('poreplex')
    logger.propagate = False
    handler = logging.FileHandler(logfile, 'w')

    logger.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)-15s %(message)s'))
    logger.addHandler(handler)

    return logger

def create_output_directories(config):
    outputdir = config['outputdir']
    existing = os.listdir(outputdir)
    if existing:
        while config['interactive']:
            try:
                answer = input('Output directory {} is not empty. Clear it? (y/N) '
                                .format(outputdir))
            except KeyboardInterrupt:
                raise SystemExit
            answer = answer.lower()[:1]
            if answer in ('', 'n'):
                sys.exit(1)
            elif answer == 'y':
                print()
                break

        for ent in existing:
            fpath = os.path.join(outputdir, ent)
            if os.path.isdir(fpath):
                shutil.rmtree(fpath)
            else:
                os.unlink(fpath)

    subdirs = []
    conditional_subdirs = [
        ('fast5_output', 'fast5'),
    ]
    for condition, subdir in conditional_subdirs:
        if config[condition]:
            subdirs.append(subdir)

    for subdir in subdirs:
        fullpath = os.path.join(outputdir, subdir)
        if not os.path.isdir(fullpath):
            os.makedirs(fullpath)

    if not os.path.isdir(config['tmpdir']):
        os.makedirs(config['tmpdir'])
        config['cleanup_tmpdir'] = True


def setup_output_name_mapping(config):
    label_names = {'fail': OUTPUT_NAME_FAILED, 'pass': OUTPUT_NAME_PASSED}

    barcode_names = {None: OUTPUT_NAME_BARCODING_OFF}
    layout_maps = {
        (label, None): labelname for label, labelname in label_names.items()}

    return label_names, barcode_names, layout_maps


def show_configuration(config, output):
    if hasattr(output, 'write'): # file-like object
        _ = partial(print, sep='\t', file=output)
    else: # logger object
        _ = lambda *args: output.info(' '.join(map(str, args)))

    bool2yn = lambda b: 'Yes' if b else 'No'

    _("== Analysis settings ======================================")
    _(" * Input:", config['inputdir'],
      '(live, {} sec delay)'.format(config['analysis_start_delay'])
      if config['live'] else '')
    _(" * Output:", config['outputdir'])
    _(" * Presets:", config['preset_name'])
    _(" * FAST5 in output:\t", bool2yn(config['fast5_output']))
    _("===========================================================")
    _("")

def test_prerequisite_compatibility(config):
    from distutils.version import LooseVersion
    from pomegranate import __version__ as pomegranate_version
    if LooseVersion(pomegranate_version) <= LooseVersion('0.9.0'):
        errprint('''
WARNING: You have pomegranate {} installed, which has a known
problem that the memory consumption indefinitely grow. The processing
may stop after processing few thousands of reads due to the out of memory
(OOM) errors. Use this command to install until the new release comes out
with the fix:

  pip install cython
  pip install git+https://github.com/jmschrei/pomegranate.git\n'''.format(pomegranate_version))


def test_inputs_and_outputs(config):
    if not os.path.isdir(config['inputdir']):
        errx('ERROR: Cannot open the input directory {}.'.format(config['inputdir']))

    if not os.path.isdir(config['outputdir']):
        try:
            os.makedirs(config['outputdir'])
        except:
            errx('ERROR: Failed to create the output directory {}.'.format(config['outputdir']))

def fix_options(config):
    printed_any = False

    if printed_any:
        errprint('')

def main(args):
    if not args.quiet:
        show_banner()

    config = load_config(args)
    config['quiet'] = args.quiet
    config['interactive'] = not args.yes
    config['inputdir'] = args.input
    config['outputdir'] = args.output
    config['live'] = args.live
    config['tmpdir'] = args.tmpdir if args.tmpdir else os.path.join(args.output, 'tmp')
    config['barcoding'] = args.barcoding
    config['measure_polya'] = args.polya
    config['batch_chunk_size'] = args.batch_size
    config['fast5_output'] = args.fast5 or args.nanopolish
    config['fast5_batch_size'] = args.fast5_batch_size
    config['label_names'], config['barcode_names'], config['output_layout'] = \
        setup_output_name_mapping(config)
    config['nobasecall_stop_trigger'] = 1000

    test_inputs_and_outputs(config)
    create_output_directories(config)

    logger = init_logging(config)
    test_prerequisite_compatibility(config)

    logger.info('Starting poreplex version {}'.format(__version__))
    logger.info('Command line: ' + ' '.join(sys.argv))

    show_configuration(config, output=logger)
    if not config['quiet']:
        show_configuration(config, output=sys.stdout)

    procresult = ProcessingSession.run(config, logger)

    if procresult is not None:
        if not config['quiet']:
            procresult(sys.stdout)
        procresult(logger)

    logger.info('Finished.')

    if config['cleanup_tmpdir']:
        try:
            shutil.rmtree(config['tmpdir'])
        except:
            pass

def __main__():
    parser = argparse.ArgumentParser(
        prog='poreplex', add_help=False,
        description='Cuts nanopore direct RNA sequencing data '
                    'into bite-size pieces for RNA Biology')

    group = parser.add_argument_group('Data Settings')
    group.add_argument('-i', '--input', required=True, metavar='DIR',
                       help='path to the directory with the input FAST5 files '
                            '(Required)')
    group.add_argument('-o', '--output', required=True, metavar='DIR',
                       help='output directory path (Required)')
    group.add_argument('-c', '--config', default='', metavar='NAME',
                       help='path to signal processing configuration')

    group = parser.add_argument_group('Optional Analyses')
    group.add_argument('--barcoding', default=False, action='store_true',
                       help='sort barcoded reads into separate outputs')

    group.add_argument('--polya', default=False, action='store_true',
                       help='output poly(A) tail length measurements')

    group = parser.add_argument_group('Live Mode')
    group.add_argument('--live', default=False, action='store_true',
                       help='monitor new files in the input directory')

    group = parser.add_argument_group('Output Options')
    group.add_argument('--fast5', default=False, action='store_true',
                       help='link or copy FAST5 files to separate output directories')
    group.add_argument('--fast5-batch-size', default=4000, type=int,
                       help='number of reads in a FAST5 for output')

    group = parser.add_argument_group('User Interface')
    group.add_argument('--contig-aliases', default=None, metavar='FILE', type=str,
                       help='path to a tab-separated text file for aliases to show '
                            'as a contig names in the dashboard (see README)')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='suppress non-error messages')
    group.add_argument('-y', '--yes', default=False, action='store_true',
                       help='suppress all questions')

    group = parser.add_argument_group('Pipeline Options')
    group.add_argument('--tmpdir', default='', type=str, metavar='DIR',
                       help='temporary directory for intermediate data')
    group.add_argument('--batch-size', default=128, type=int, metavar='SIZE',
                       help='number of files in a single batch (default: 128)')
    group.add_argument('--version', action=VersionAction,
                       help="show program's version number and exit")
    group.add_argument('-h', '--help', action='help',
                       help='show this help message and exit')

    args = parser.parse_args(sys.argv[1:])
    main(args)

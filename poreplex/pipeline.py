#
# Copyright (c) 2018 Institute for Basic Science
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


import sys
import os
from io import StringIO
from itertools import cycle
from collections import defaultdict
import time
import numpy as np
import pandas as pd
from . import *
from .io import (
    FASTQWriter, SequencingSummaryWriter, FinalSummaryTracker,
    NanopolishReadDBWriter, create_adapter_dumps_inventory,
    create_events_inventory, FAST5Writer)
from .signal_analyzer import process_batch
from .utils import *
from .fast5_file import get_read_ids

FAST5_SUFFIX = '.fast5'
TXT_SUFFIX = '.txt'

def scan_dir_recursive_worker(dirname, suffix=FAST5_SUFFIX):
    files, dirs, scale_files = [], [], []
    for entryname in os.listdir(dirname):
        if entryname.startswith('.'):
            continue

        fullpath = os.path.join(dirname, entryname)
        if os.path.isdir(fullpath):
            dirs.append(entryname)
        elif entryname.lower().endswith(suffix):    
            files.append(entryname)
        elif entryname.lower().endswith(TXT_SUFFIX):
            scale_files.append(entryname.split(TXT_SUFFIX)[0] + FAST5_SUFFIX)

    files = list(set(files) & set(scale_files))

    return dirs, files


def show_memory_usage():
    usages = open('/proc/self/statm').read().split()
    print('{:05d} MEMORY total={} RSS={} shared={} data={}'.format(
            batchid, usages[0], usages[1], usages[2], usages[4]))


class ProcessingSession:

    def __init__(self, config, logger):
        self.running = True
        self.scan_finished = False
        self.reads_queued = self.reads_found = 0
        self.reads_processed = 0
        self.next_batch_id = 0
        self.reads_done = set()
        self.active_batches = 0
        self.error_status_counts = defaultdict(int)
        self.jobstack = []

        self.config = config
        self.logger = logger

        self.loop = self.fastq_writer = self.fast5_writer = \
            self.alignment_writer = self.npreaddb_writer = None
        self.dashboard = self.pbar = None

    def __enter__(self):
        if self.config['fastq_output']:
            self.fastq_writer = FASTQWriter(
                self.config['outputdir'], self.config['output_layout'])
        if self.config['fast5_output']:
            self.fast5_writer = FAST5Writer(
                self.config['outputdir'], self.config['output_layout'],
                self.config['inputdir'], self.config['fast5_batch_size'])
        if self.config['nanopolish_output']:
            self.npreaddb_writer = NanopolishReadDBWriter(
                self.config['outputdir'], self.config['output_layout'])
        self.seqsummary_writer = SequencingSummaryWriter(
            self.config, self.config['outputdir'], self.config['label_names'],
            self.config['barcode_names'])
        self.finalsummary_tracker = FinalSummaryTracker(
            self.config['label_names'], self.config['barcode_names'])

        return self

    def __exit__(self, *args):
        if self.fastq_writer is not None:
            self.fastq_writer.close()
            self.fastq_writer = None

        if self.fast5_writer is not None:
            self.fast5_writer.close()
            self.fast5_writer = None

        if self.npreaddb_writer is not None:
            self.npreaddb_writer.close()
            self.npreaddb_writer = None

        if self.seqsummary_writer is not None:
            self.seqsummary_writer.close()
            self.seqsummary_writer = None

        if self.alignment_writer is not None:
            self.alignment_writer.close()
            self.alignment_writer = None

    def errx(self, message):
        if self.running:
            errprint(message, end='')
            # self.stop('ERROR')

    def show_message(self, message):
        if not self.config['quiet']:
            print(message)

    def run_in_executor_compute(self, *args):
        return self.loop.run_in_executor(self.executor_compute, *args)

    def run_in_executor_io(self, *args):
        return self.loop.run_in_executor(self.executor_io, *args)

    def run_in_executor_mon(self, *args):
        return self.loop.run_in_executor(self.executor_mon, *args)

    def run_process_batch(self, batchid, files):
        # Wait until the input files become ready if needed

        self.active_batches += 1
        try:

            start = self.config['batch_chunk_size'] * batchid
            end = (batchid + 1) * self.config['batch_chunk_size']
            self.config['fit_scaling_params'] = {}
            self.config['fit_scaling_params']['scale'] = self.fit_scaling_params.iloc[start:end, 2].to_list()
            self.config['fit_scaling_params']['shift'] = self.fit_scaling_params.iloc[start:end, 3].to_list()

            results = process_batch(batchid, files, self.config)

            if len(results) > 0 and results[0] == -1: # Unhandled exception occurred
                error_message = results[1]
                self.logger.error(error_message)
                for line in results[2].splitlines():
                    self.logger.error(line)
                self.errx("ERROR: " + error_message)
                return

            # Remove duplicated results that could be fed multiple times in live monitoring
            nd_results = []
            for result in results:
                readpath = result['filename'], result['read_id']
                if readpath not in self.reads_done:
                    if result['status'] == 'okay':
                        self.reads_done.add(readpath)
                    elif 'error_message' in result:
                        self.logger.error(result['error_message'])
                    nd_results.append(result)
                else: # Cancel the duplicated result
                    self.reads_queued -= 1
                    self.reads_found -= 1

                self.error_status_counts[result['status']] += 1

            if nd_results:
                if self.config['fastq_output']:
                    self.fastq_writer.write_sequences(nd_results)

                if self.config['fast5_output']:
                    self.fast5_writer.transfer_reads(nd_results)

                if self.config['nanopolish_output']:
                    self.npreaddb_writer.write_sequences(nd_results)

                if self.config['minimap2_index']:
                    rescounts = self.alignment_writer.process(nd_results)
                    if self.dashboard is not None:
                        self.dashboard.feed_mapped(rescounts)

                self.seqsummary_writer.write_results(nd_results)

                self.finalsummary_tracker.feed_results(nd_results)

            if (self.error_status_counts['okay'] == 0 and self.running and
                    self.error_status_counts['not_basecalled'] >=
                        self.config['nobasecall_stop_trigger']):

                stopmsg = ('Early stopping: {} out of {} reads are not basecalled. '
                           'Please check if the files are correctly analyzed, or '
                           'add `--basecall\' to the command line.'.format(
                                self.error_status_counts['not_basecalled'],
                                sum(self.error_status_counts.values())))
                self.logger.error(stopmsg)
                self.errx(stopmsg)

        except Exception as exc:
            self.logger.error('Unhandled error during processing reads', exc_info=exc)
            return self.errx('ERROR: Unhandled error ' + str(exc))
        finally:
            self.active_batches -= 1

        self.reads_processed += len(nd_results)
        self.reads_queued -= len(nd_results)

    def queue_processing(self, readpath):
        self.jobstack.append(readpath)
        self.reads_queued += 1
        self.reads_found += 1
        if len(self.jobstack) >= self.config['batch_chunk_size']:
            self.flush_jobstack()

    def flush_jobstack(self):
        if self.running and self.jobstack:
            batch_id = self.next_batch_id
            self.next_batch_id += 1

            # Remove files already processed successfully. The same file can be
            # fed into the stack while making transition from the existing
            # files to newly updated files from the live monitoring.
            reads_to_submit = [
                readpath for readpath in self.jobstack
                if readpath not in self.reads_done]
            num_canceled = len(self.jobstack) - len(reads_to_submit)
            if num_canceled:
                self.reads_queued -= num_canceled
                self.reads_found -= num_canceled
            del self.jobstack[:]

            if reads_to_submit:
                start = time.time()
                self.run_process_batch(batch_id, reads_to_submit)
                end = time.time()

                print(f"{batch_id} : {end - start:5f} sec")

    def scan_dir_recursive(self, topdir, dirname=''):
        if not self.running:
            return

        is_topdir = (dirname == '')

        try:
            errormsg = None
            dirs, files = scan_dir_recursive_worker(os.path.join(topdir, dirname))

        except Exception as exc:
            errormsg = str(exc)

        if errormsg is not None:
            return self.errx('ERROR: ' + str(errormsg))

        for filename in files:
            filepath = os.path.join(dirname, filename)
            
            scale_path = os.path.join(self.config['inputdir'], filename.split('.')[0] + '.txt')
            scaling_param = pd.read_csv(scale_path, sep='\t', usecols=[0, 1, 2], names=['read_id', 'scale', 'shift'])

            readpaths = get_read_ids(filepath, topdir)
            test = pd.DataFrame(readpaths, columns=['file', 'read_id'])
            self.fit_scaling_params = test.merge(scaling_param, how='left', on=['read_id'])
            for readpath in readpaths:
                self.queue_processing(readpath)

        try:
            for subdir in dirs:
                subdirpath = os.path.join(dirname, subdir)
                self.scan_dir_recursive(topdir, subdirpath)
        except Exception as exc:
            if is_topdir: return
            else: raise exc

        if is_topdir:
            self.flush_jobstack()
            self.scan_finished = True

    @classmethod
    def run(kls, config, logging):
        with kls(config, logging) as sess:
            sess.show_message("==> Processing FAST5 files")

            # Start the directory scanner
            sess.scan_dir_recursive(config['inputdir'])
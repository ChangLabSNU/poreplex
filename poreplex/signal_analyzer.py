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

from weakref import proxy
from itertools import groupby
import numpy as np
from io import StringIO
import traceback
import sys
import os
from .worker_persistence import WorkerPersistenceStorage

__all__ = ['SignalAnalyzer', 'SignalAnalysis', 'process_batch']


class SignalAnalysisError(Exception):
    pass


# This function must be picklable.
def process_batch(batchid, reads, config):
    try:
        with SignalAnalyzer(config, batchid) as analyzer:
            return analyzer.process(reads)
    except Exception as exc:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[-1]
        errorf = StringIO()
        traceback.print_exc(file=errorf)

        return (-1, '[{filename}:{lineno}] Unhandled exception {name}: {msg}'.format(
                        filename=filename, lineno=exc_tb.tb_lineno,
                        name=type(exc).__name__, msg=str(exc)), errorf.getvalue())


class SignalAnalyzer:

    _EVENT_DUMP_FIELD_NAMES = [
        'mean', 'start', 'stdv', 'length', 'model_state',
        'move', 'pos', 'end', 'scaled_mean']
    _EVENT_DUMP_FIELD_DTYPES = [
        '<f4', '<u8', '<f4', '<u8', None,
        '<i4', '<u8', '<u8', '<f8']
    EVENT_DUMP_FIELDS = list(zip(_EVENT_DUMP_FIELD_NAMES, _EVENT_DUMP_FIELD_DTYPES))

    def __init__(self, config, batchid):
        WorkerPersistenceStorage(config).retrieve_objects(self)

        self.config = config
        self.inputdir = config['inputdir']
        self.outputdir = config['outputdir']
        self.batchid = batchid
        self.formatted_batchid = format(batchid, '08d')
        self.open_dumps()

    def process(self, reads):
        inputdir = self.config['inputdir']
        results, loaded = [], []

        # Initialize processors and preload signals from fast5
        nextprocs = []
        prepare_loading = self.loader.prepare_loading
        for f5file, read_id in reads:
            if not os.path.exists(os.path.join(inputdir, f5file)):
                results.append({'filename': f5file, 'status': 'disappeared'})
                continue

            try:
                npread = prepare_loading(f5file, read_id)
                if npread.is_stopped():
                    results.append(npread.report())
                else:
                    siganal = SignalAnalysis(npread, self)
                    nextprocs.append(siganal)
                    loaded.append(npread)
            except Exception as exc:
                error = self.pack_unhandled_exception(f5file, read_id, exc, sys.exc_info())
                results.append(error)

        # Determine scaling parameters
        self.loader.fit_scalers(self.config['fit_scaling_params'])

        # Perform the main analysis procedures
        procs, nextprocs = nextprocs, []
        for siganal in procs:
            try:
                if not siganal.is_stopped():
                    siganal.process()
                    nextprocs.append(siganal)
                else:
                    sys.stdout.flush()
            except Exception as exc:
                f5file = siganal.npread.filename
                read_id = siganal.npread.read_id
                error = self.pack_unhandled_exception(f5file, read_id, exc, sys.exc_info())
                siganal.set_error(error)
            finally:
                siganal.clear_cache()

        # Copy the final results
        for npread in loaded:
            results.append(npread.report())

        return results

    def pack_unhandled_exception(self, f5filename, read_id, exc, excinfo):
        exc_type, exc_obj, exc_tb = excinfo
        srcfilename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[-1]
        errorf = StringIO()
        traceback.print_exc(file=errorf)

        errmsg = ('[{srcfilename}:{lineno}] ({f5filename}#{read_id}) Unhandled '
                  'exception {name}: {msg}\n{exc}'.format(
            srcfilename=srcfilename, lineno=exc_tb.tb_lineno,
            f5filename=f5filename, read_id=read_id, name=type(exc).__name__, msg=str(exc),
            exc=errorf.getvalue()))

        return {
            'filename': f5filename,
            'read_id': read_id,
            'status': 'unknown_error',
            'error_message': errmsg,
        }

    def open_dumps(self):
        self.EVENT_DUMP_FIELDS[4] = (self.EVENT_DUMP_FIELDS[4][0], 'S{}'.format(self.kmersize))
        self.adapter_dump_file = self.adapter_dump_group = None
        self.basecall_dump_file = self.basecall_dump_group = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.adapter_dump_file is not None:
            catgrp = self.adapter_dump_file.require_group('catalog/adapter')
            encodedarray = np.array(self.adapter_dump_list,
                dtype=[('read_id', 'S36'), ('start', 'i8'), ('end', 'i8')])
            catgrp.create_dataset(self.formatted_batchid, shape=encodedarray.shape,
                                  data=encodedarray)
            self.adapter_dump_file.close()

        if self.basecall_dump_file is not None:
            self.basecall_dump_file.close()


class SignalAnalysis:

    def __init__(self, npread, analyzer):
        self.npread = npread
        self.config = analyzer.config
        self.analyzer = proxy(analyzer)

    def set_error(self, error):
        self.npread.set_error(error['status'], error['error_message'])

    def is_stopped(self):
        return self.npread.is_stopped()

    def clear_cache(self):
        self.npread.close()

    def process(self):
        stride = self.config['signal_processing']['rough_signal_stride']

        try:
            # Load the raw signals for segmentation and in-read adapter signals
            signal = self.npread.load_signal(pool=stride)

            # Rough segmentation of signal for extracting adapter signals
            segments = self.detect_segments(signal, stride)
            if 'adapter' not in segments:
                raise SignalAnalysisError('adapter_not_detected')

            # Measure poly(A) tail signals
            if self.config['measure_polya']:
                if 'polya-tail' in segments:
                    rough_range = segments['polya-tail']
                else:
                    rough_range = segments['adapter'][1] + 1, None
                self.analyzer.polyaanalyzer(self.npread, rough_range, stride)

            # Discard short sequences
            if self.npread.sequence is not None:
                readlength = len(self.npread.sequence[0]) - self.npread.sequence[2]
                if readlength < self.config['minimum_sequence_length']:
                    raise SignalAnalysisError('sequence_too_short')

        except SignalAnalysisError as exc:
            outname = 'artifact' if exc.args[0] in ('unsplit_read',) else 'fail'
            self.npread.set_status(exc.args[0], stop=True)
            self.npread.set_label(outname)
        else:
            self.npread.set_label('pass')

    def detect_segments(self, signal, elspan):
        scan_limit = self.config['segmentation']['segmentation_scan_limit'] // elspan
        if len(signal) > scan_limit:
            signal = signal[:scan_limit]

        # Run Viterbi fitting to signal model
        plogsum, statecalls = self.analyzer.segmodel.viterbi(signal)

        # Summarize state transitions
        sigparts = {}
        for _, positions in groupby(enumerate(statecalls[1:]),
                                    lambda st: id(st[1][1])):
            first, state = last, _ = next(positions)
            statename = state[1].name
            for last, _ in positions:
                pass
            sigparts[statename] = (first, last) # right-inclusive

        return sigparts
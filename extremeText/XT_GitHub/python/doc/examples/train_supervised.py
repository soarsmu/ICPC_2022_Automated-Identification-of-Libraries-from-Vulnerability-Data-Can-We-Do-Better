#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function

import os
from extremeText import train_supervised


def print_results(N, p, r, c):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    print("C@{}\t{:.3f}".format(1, c))

if __name__ == "__main__":
    train_data = os.path.join(os.getenv("DATADIR", ''), 'dbpedia.train')
    valid_data = os.path.join(os.getenv("DATADIR", ''), 'dbpedia.test')

    #train_supervised uses the same arguments and defaults as the fastText/extremeText cli
    model = train_supervised(
        input=train_data, epoch=25, lr=1.0, wordNgrams=2, verbose=3, minCount=1,
    )
    print_results(*model.test(valid_data))
    model.save_model("dbpedia.bin")

    model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
    print_results(*model.test(valid_data))
    model.save_model("dbpedia.ftz")

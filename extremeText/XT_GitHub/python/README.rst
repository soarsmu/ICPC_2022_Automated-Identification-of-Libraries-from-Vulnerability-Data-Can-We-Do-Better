extremeText
===========

`extremeText <https://github.com/mwydmuch/extremeText>`__ is an
extension of `fastText <https://github.com/facebookresearch/fastText>`__
library for multi-label classification including extreme cases with
hundreds of thousands and millions of labels.

`extremeText <https://github.com/mwydmuch/extremeText>`__ implements:

-  Probabilistic Labels Tree (PLT) loss for extreme multi-Label
   classification with top-down hierarchical clustering (k-means) for
   tree building,
-  sigmoid loss for multi-label classification,
-  L2 regularization update for all losses,
-  ensemble of loss layers with bagging,
-  calculation of hidden (document) vector as a weighted average of the
   word vectors,
-  calculation of TF-IDF weights for words.

Requirements
------------

`extremeText <https://github.com/mwydmuch/extremeText>`__ builds on
modern Mac OS and Linux distributions. Since it uses C++11 features, it
requires a compiler with good C++11 support. These include:

-  (gcc-4.8 or newer) or (clang-3.3 or newer)

You will need:

-  `Python <https://www.python.org/>`__ version 2.7 or >=3.4
-  `NumPy <http://www.numpy.org/>`__ &
   `SciPy <https://www.scipy.org/>`__
-  `pybind11 <https://github.com/pybind/pybind11>`__

Installing extremeText
----------------------

The easiest way to get
`extremeText <https://github.com/mwydmuch/extremeText>`__ is to use
`pip <https://pip.pypa.io/en/stable/>`__.

::

    $ pip install extremetext

Installing on MacOS may require setting
``MACOSX_DEPLOYMENT_TARGET=10.9`` first:

::

    $ export MACOSX_DEPLOYMENT_TARGET=10.9
    $ pip install extremetext

The latest version of
`extremeText <https://github.com/mwydmuch/extremeText>`__ can be build
from sources using pip or alternatively setuptools.

::

    $ git clone https://github.com/mwydmuch/extremeText.git
    $ cd extremeText
    $ pip install .
    (or) $ python setup.py install

Now you can import this library with:

::

    import extremeText

Examples
--------

In general it is assumed that the reader already has good knowledge of
fastText/extremeText. For this consider the main
`README <https://github.com/mwydmuch/extremeText/blob/master/README.md>`__
and `the tutorials on fastText
website <https://fasttext.cc/docs/en/supervised-tutorial.html>`__.

We recommend you look at the `examples within the doc
folder <https://github.com/mwydmuch/extremeText/tree/master/python/doc/examples>`__.

As with any package you can get help on any Python function using the
help function.

For example:

::

    +>>> import extremeText
    +>>> help(extremeText.ExtremeText)

    Help on module extremeText.ExtremeText in extremeText:

    NAME
        extremeText.ExtremeText

    DESCRIPTION
        # Copyright (c) 2017-present, Facebook, Inc.
        # All rights reserved.
        #
        # This source code is licensed under the BSD-style license found in the
        # LICENSE file in the root directory of this source tree. An additional grant
        # of patent rights can be found in the PATENTS file in the same directory.

    FUNCTIONS
        load_model(path)
            Load a model given a filepath and return a model object.

        tokenize(text)
            Given a string of text, tokenize it and return a list of tokens
    [...]

IMPORTANT: Preprocessing data / enconding conventions
-----------------------------------------------------

In general it is important to properly preprocess your data. Example
scripts in the `root
folder <https://github.com/mwydmuch/extremeText/extremeText>`__ do this.

extremeText like fastText assumes UTF-8 encoded text. All text must be
`unicode for
Python2 <https://docs.python.org/2/library/functions.html#unicode>`__
and `str for
Python3 <https://docs.python.org/3.5/library/stdtypes.html#textseq>`__.
The passed text will be `encoded as UTF-8 by
pybind11 <https://pybind11.readthedocs.io/en/master/advanced/cast/strings.html?highlight=utf-8#strings-bytes-and-unicode-conversions>`__
before passed to the extremeText C++ library. This means it is important
to use UTF-8 encoded text when building a model. On Unix-like systems
you can convert text using
`iconv <https://en.wikipedia.org/wiki/Iconv>`__.

extremeText will tokenize (split text into pieces) based on the
following ASCII characters (bytes). In particular, it is not aware of
UTF-8 whitespace. We advice the user to convert UTF-8 whitespace / word
boundaries into one of the following symbols as appropiate.

-  space
-  tab
-  vertical tab
-  carriage return
-  formfeed
-  the null character

The newline character is used to delimit lines of text. In particular,
the EOS token is appended to a line of text if a newline character is
encountered. The only exception is if the number of tokens exceeds the
MAX\_LINE\_SIZE constant as defined in the `Dictionary
header <https://github.com/mwydmuch/extremeText/blob/master/src/dictionary.h>`__.
This means if you have text that is not separate by newlines, such as
the `fil9 dataset <http://mattmahoney.net/dc/textdata>`__, it will be
broken into chunks with MAX\_LINE\_SIZE of tokens and the EOS token is
not appended.

The length of a token is the number of UTF-8 characters by considering
the `leading two bits of a
byte <https://en.wikipedia.org/wiki/UTF-8#Description>`__ to identify
`subsequent bytes of a multi-byte
sequence <https://github.com/mwydmuch/extremeText/blob/master/src/dictionary.cc>`__.
Knowing this is especially important when choosing the minimum and
maximum length of subwords. Further, the EOS token (as specified in the
`Dictionary
header <https://github.com/mwydmuch/extremeText/blob/master/src/dictionary.h>`__)
is considered a character and will not be broken into subwords.

Reference
---------

Please cite below work if using this package for extreme classification.

M. Wydmuch, K. Jasinska, M. Kuznetsov, R. Busa-Fekete, K. Dembczyński.
`*A no-regret generalization of hierarchical softmax to extreme
multi-label
classification* <http://papers.nips.cc/paper/7872-a-no-regret-generalization-of-hierarchical-softmax-to-extreme-multi-label-classification>`__.
Advances in Neural Information Processing Systems 31, 2018.

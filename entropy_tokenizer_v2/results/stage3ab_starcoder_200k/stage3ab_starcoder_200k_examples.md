# Stage3 hybrid_ab examples (first 5 files)

## Sample 0

- B 前后相同（B 未改写文本）。

### 原文片段
```
##########################################################################
#
#  Copyright (c) 2010-2012, Image Engine Design Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in t...
```

### Stage1 后
```
##########################################################################
#
#  Copyright (c) 2010-2012, Image Engine Design Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in t...
```

### Stage2 后
```

<SYN_3> __future__ with_statement
import os
import sys
import shutil
import unittest
import IECore
class TestBasicPreset( unittest.TestCase ) :
<SYN_1> testCopy self
testObj = IECore.Parameterised( "testParameterised1" )
testObj.parameters().addParameters(
[
IECore.BoolParameter( "a", "", True ),
IECore.FloatParameter( "b", "", 1.0 ),
]
)
testObj2 = IECore.Parameterised( "testParameterised2" )
testObj2.parameters().addParameters(
[
IECore.BoolParameter( "a", "", False ),
IECore.FloatParameter( "c", "", 0.0	),
]
)
p = IECore.BasicPreset( testObj, testObj.parameters() )
self.assertTrue( p.ap...
```

### Stage3 A 后（含 guardrail）
```

<SYN_3> __future__ with_statement
import os
import sys
import shutil
import unittest
import IECore
class TestBasicPreset( unittest.TestCase ) :
<SYN_1> testCopy self
testObj = IECore.Parameterised( "testParameterised1" )
testObj.parameters().addParameters(
[
IECore.BoolParameter( "a", "", True ),
IECore.FloatParameter( "b", "", 1.0 ),
]
)
k2 = IECore.Parameterised( "testParameterised2" )
k2.parameters().addParameters(
[
IECore.BoolParameter( "a", "", False ),
IECore.FloatParameter( "c", "", 0.0	),
]
)
p = IECore.BasicPreset( testObj, testObj.parameters() )
self.assertTrue( p.applicableTo( ...
```

### Stage3 最终（A+B，含 guardrail）
```

<SYN_3> __future__ with_statement
import os
import sys
import shutil
import unittest
import IECore
class TestBasicPreset( unittest.TestCase ) :
<SYN_1> testCopy self
testObj = IECore.Parameterised( "testParameterised1" )
testObj.parameters().addParameters(
[
IECore.BoolParameter( "a", "", True ),
IECore.FloatParameter( "b", "", 1.0 ),
]
)
k2 = IECore.Parameterised( "testParameterised2" )
k2.parameters().addParameters(
[
IECore.BoolParameter( "a", "", False ),
IECore.FloatParameter( "c", "", 0.0	),
]
)
p = IECore.BasicPreset( testObj, testObj.parameters() )
self.assertTrue( p.applicableTo( ...
```

## Sample 1

- B 前后相同（B 未改写文本）。

### 原文片段
```
<filename>src/biotite/copyable.py
# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite"
__author__ = "<NAME>"
__all__ = ["Copyable"]

import abc


class Copyable(metaclass=abc.ABCMeta):
    """
    Base class for all objects, that should be copyable.
    
    The public method `copy()` first creates a fresh instance of the
    class of the instance, that is copied via the `__copy_create__()`
    method. All variables, that could not be set via the constructor,
    are t...
```

### Stage1 后
```
<filename>src/biotite/copyable.py
# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite"
__author__ = "<NAME>"
__all__ = ["Copyable"]

import abc


class Copyable(metaclass=abc.ABCMeta):
    """
    Base class for all objects, that should be copyable.
    
    The public method `copy()` first creates a fresh instance of the
    class of the instance, that is copied via the `__copy_create__()`
    method. All variables, that could not be set via the constructor,
    are t...
```

### Stage2 后
```
<filename>src/biotite/copyable.py
__name__ = "biotite"
__author__ = "<NAME>"
__all__ = ["Copyable"]
import abc
class Copyable(metaclass=abc.ABCMeta):
"""
Base class for all objects, that should be copyable.

The public method `copy()` first creates a fresh instance of the
class of the instance, that is copied via the `__copy_create__()`
method. All variables, that could not be set via the constructor,
are then copied via `__copy_fill__()`, starting with the method in
    the uppermost base class and ending with the class of the instance
    to be copied.
    
    This approach solves the pr...
```

### Stage3 A 后（含 guardrail）
```
<filename>src/biotite/copyable.py
__name__ = "biotite"
__author__ = "<NAME>"
__all__ = ["Copyable"]
import abc
class Copyable(metaclass=abc.ABCMeta):
"""
Base class for all objects, that should be copyable.

The public method `copy()` first creates a fresh instance of the
class of the instance, that is copied via the `__copy_create__()`
method. All variables, that could not be set via the constructor,
are then copied via `__copy_fill__()`, starting with the method in
    the uppermost base class and ending with the class of the instance
    to be copied.
    
    This approach solves the pr...
```

### Stage3 最终（A+B，含 guardrail）
```
<filename>src/biotite/copyable.py
__name__ = "biotite"
__author__ = "<NAME>"
__all__ = ["Copyable"]
import abc
class Copyable(metaclass=abc.ABCMeta):
"""
Base class for all objects, that should be copyable.

The public method `copy()` first creates a fresh instance of the
class of the instance, that is copied via the `__copy_create__()`
method. All variables, that could not be set via the constructor,
are then copied via `__copy_fill__()`, starting with the method in
    the uppermost base class and ending with the class of the instance
    to be copied.
    
    This approach solves the pr...
```

## Sample 2

- B 前后相同（B 未改写文本）。

### 原文片段
```
<filename>tests/keras/layers/wrappers_test.py<gh_stars>100-1000
import pytest
import numpy as np
import copy
from numpy.testing import assert_allclose
from keras.utils import CustomObjectScope
from keras.layers import wrappers, Input, Layer
from keras.layers import RNN
from keras import layers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.utils.generic_utils import object_list_uid, to_list


@pytest.mark.skipif(K.backend() == 'mxnet',
                    reason='MXNet backend does not support TimeDistributed and RNN yet')
def test_Time...
```

### Stage1 后
```
<filename>tests/keras/layers/wrappers_test.py<gh_stars>100-1000
import pytest
import numpy as np
import copy
from numpy.testing import assert_allclose
from keras.utils import CustomObjectScope
from keras.layers import wrappers, Input, Layer
from keras.layers import RNN
from keras import layers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.utils.generic_utils import object_list_uid, to_list


@pytest.mark.skipif(K.backend() == 'mxnet',
                    reason='MXNet backend does not support TimeDistributed and RNN yet')
def test_Time...
```

### Stage2 后
```
<filename>tests/keras/layers/wrappers_test.py<gh_stars>100-1000
import pytest
import numpy as np
import copy
from numpy.testing import assert_allclose
from keras.utils import CustomObjectScope
from keras.layers import wrappers, Input, Layer
from keras.layers import RNN
from keras import layers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.utils.generic_utils import object_list_uid, to_list
@pytest.mark.skipif(K.backend() == 'mxnet',
reason='MXNet backend does not support TimeDistributed and RNN yet')
def test_TimeDistributed():
model =...
```

### Stage3 A 后（含 guardrail）
```
<filename>tests/keras/layers/wrappers_test.py<gh_stars>100-1000
import pytest
import numpy as np
import copy
from numpy.testing import assert_allclose
from keras.utils import CustomObjectScope
from keras.layers import wrappers, Input, Layer
from keras.layers import RNN
from keras import layers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.utils.generic_utils import object_list_uid, to_list
@pytest.mark.skipif(K.backend() == 'mxnet',
reason='MXNet backend does not support TimeDistributed and RNN yet')
def test_TimeDistributed():
model =...
```

### Stage3 最终（A+B，含 guardrail）
```
<filename>tests/keras/layers/wrappers_test.py<gh_stars>100-1000
import pytest
import numpy as np
import copy
from numpy.testing import assert_allclose
from keras.utils import CustomObjectScope
from keras.layers import wrappers, Input, Layer
from keras.layers import RNN
from keras import layers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.utils.generic_utils import object_list_uid, to_list
@pytest.mark.skipif(K.backend() == 'mxnet',
reason='MXNet backend does not support TimeDistributed and RNN yet')
def test_TimeDistributed():
model =...
```

## Sample 3

- B 前后相同（B 未改写文本）。

### 原文片段
```
<gh_stars>100-1000
"""Lowest-common-denominator implementations of platform functionality."""
from __future__ import absolute_import, division, print_function, with_statement

import errno
import socket

from tornado.platform import interface


class Waker(interface.Waker):
    """Create an OS independent asynchronous pipe.

    For use on platforms that don't have os.pipe() (or where pipes cannot
    be passed to select()), but do have sockets.  This includes Windows
    and Jython.
    """
    def __init__(self):
        # Based on Zope async.py: http://svn.zope.org/zc.ngi/trunk/src/zc/ng...
```

### Stage1 后
```
<gh_stars>100-1000
"""Lowest-common-denominator implementations of platform functionality."""
from __future__ import absolute_import, division, print_function, with_statement

import errno
import socket

from tornado.platform import interface


class Waker(interface.Waker):
    """Create an OS independent asynchronous pipe.

    For use on platforms that don't have os.pipe() (or where pipes cannot
    be passed to select()), but do have sockets.  This includes Windows
    and Jython.
    """
    def __init__(self):
        # Based on Zope async.py: http://svn.zope.org/zc.ngi/trunk/src/zc/ng...
```

### Stage2 后
```
<gh_stars>100-1000
"""Lowest-common-denominator implementations of platform functionality."""
from __future__ import absolute_import, division, print_function, with_statement
import errno
import socket
from tornado.platform import interface
class Waker(interface.Waker):
"""Create an OS independent asynchronous pipe.

For use on platforms that don't have os.pipe() (or where pipes cannot
be passed to select()), but do have sockets.  This includes Windows
    and Jython.
    """
    def __init__(self):
        self.writer = socket.socket()
        self.writer.setsockopt(socket.IPPROTO_TCP, soc...
```

### Stage3 A 后（含 guardrail）
```
<gh_stars>100-1000
"""Lowest-common-denominator implementations of platform functionality."""
from __future__ import absolute_import, division, print_function, with_statement
import errno
import socket
from tornado.platform import interface
class Waker(interface.Waker):
"""Create an OS independent asynchronous pipe.

For use on platforms that don't have os.pipe() (or where pipes cannot
be passed to select()), but do have sockets.  This includes Windows
    and Jython.
    """
    def __init__(self):
        self.writer = socket.socket()
        self.writer.setsockopt(socket.IPPROTO_TCP, soc...
```

### Stage3 最终（A+B，含 guardrail）
```
<gh_stars>100-1000
"""Lowest-common-denominator implementations of platform functionality."""
from __future__ import absolute_import, division, print_function, with_statement
import errno
import socket
from tornado.platform import interface
class Waker(interface.Waker):
"""Create an OS independent asynchronous pipe.

For use on platforms that don't have os.pipe() (or where pipes cannot
be passed to select()), but do have sockets.  This includes Windows
    and Jython.
    """
    def __init__(self):
        self.writer = socket.socket()
        self.writer.setsockopt(socket.IPPROTO_TCP, soc...
```

## Sample 4

- B 前后相同（B 未改写文本）。

### 原文片段
```
"""
Basic usage
===========

This example presents the basic usage of brokenaxes

"""


import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np

fig = plt.figure(figsize=(5,2))
bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05)
x = np.linspace(0, 1, 100)
bax.plot(x, np.sin(10 * x), label='sin')
bax.plot(x, np.cos(10 * x), label='cos')
bax.legend(loc=3)
bax.set_xlabel('time')
bax.set_ylabel('value')

```

### Stage1 后
```
"""
Basic usage
===========

This example presents the basic usage of brokenaxes

"""


import matplotlib.pyplot as plt
<SYN_3> brokenaxes brokenaxes
import numpy as np

fig = plt.figure(figsize=(5,2))
bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05)
x = np.linspace(0, 1, 100)
bax.plot(x, np.sin(10 * x), label='sin')
bax.plot(x, np.cos(10 * x), label='cos')
bax.legend(loc=3)
bax.set_xlabel('time')
bax.set_ylabel('value')
```

### Stage2 后
```
"""
Basic usage
===========

This example presents the basic usage of brokenaxes

"""
import matplotlib.pyplot as plt
<SYN_3> brokenaxes brokenaxes
import numpy as np
fig = plt.figure(figsize=(5,2))
bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05)
x = np.linspace(0, 1, 100)
bax.plot(x, np.sin(10 * x), label='sin')
bax.plot(x, np.cos(10 * x), label='cos')
bax.legend(loc=3)
bax.set_xlabel('time')
bax.set_ylabel('value')
```

### Stage3 A 后（含 guardrail）
```
"""
Basic usage
===========

This example presents the basic usage of brokenaxes

"""
import matplotlib.pyplot as plt
<SYN_3> brokenaxes brokenaxes
import numpy as np
fig = plt.figure(figsize=(5,2))
bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05)
x = np.linspace(0, 1, 100)
bax.plot(x, np.sin(10 * x), label='sin')
bax.plot(x, np.cos(10 * x), label='cos')
bax.legend(loc=3)
bax.set_xlabel('time')
bax.set_ylabel('value')
```

### Stage3 最终（A+B，含 guardrail）
```
"""
Basic usage
===========

This example presents the basic usage of brokenaxes

"""
import matplotlib.pyplot as plt
<SYN_3> brokenaxes brokenaxes
import numpy as np
fig = plt.figure(figsize=(5,2))
bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05)
x = np.linspace(0, 1, 100)
bax.plot(x, np.sin(10 * x), label='sin')
bax.plot(x, np.cos(10 * x), label='cos')
bax.legend(loc=3)
bax.set_xlabel('time')
bax.set_ylabel('value')
```

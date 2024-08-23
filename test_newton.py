import pytest
import numpy as np
import math

import newton

# # Important: structure of tests assumes a dictionary with an 'x'
# # key as the output. 

#def test_basic_function():
#    assert np.isclose(newton.newton(2.95), math.pi)

def test_bad_input():
    #with pytest.raises(TypeError):   
    #    newton.newton(2.95)
    ## Ideally, our function would raise the exception with a useful message.
    #with pytest.raises(TypeError, match='`x0` must be numeric'):
    #    newton.newton(2.95)
    with pytest.raises(ZeroDivisionError):
        newton.division(1)


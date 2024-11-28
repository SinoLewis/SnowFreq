# How to optimize the code?

In this notebook I want to show how to write efficient code and how cython and C code can help to improve the speed.  
I decided to consider as example the SOR algorithm. Here we can see how the algorithm presented in the Notebook **A.1 Solution of linear equations** can be modified for our specific needs (i.e. solving PDEs). 

Again, if you are curious about the SOR and want to know more, have a look at the wiki page [link](https://en.wikipedia.org/wiki/Successive_over-relaxation).

## Contents
   - [Python implelentation](#sec1)
   - [Cython](#sec2)
   - [C code](#sec3)
      - [BS python vs C](#sec3.1) 


```python
import os
import subprocess
import numpy as np
import scipy as scp
from scipy.linalg import norm
from FMNM.Solvers import SOR, SOR2

%load_ext cython
import cython
```


```python
N = 3000
aa = 2
bb = 10
cc = 5
A = np.diag(aa * np.ones(N - 1), -1) + np.diag(bb * np.ones(N), 0) + np.diag(cc * np.ones(N - 1), 1)
x = 2 * np.ones(N)
b = A @ x
```

Here we use a tridiagonal matrix A 

$$ \left(
\begin{array}{ccccc}
bb     & cc     & 0      & \cdots & 0 \\
aa      & bb     & cc     & 0      & 0  \\
0      & \ddots & \ddots & \ddots & 0  \\
\vdots & 0      & aa     & bb     & cc  \\
0      & 0      & 0      & aa     & bb \\
\end{array}
\right) $$

with equal elements in the three diagonals:   

$$ aa = 2, \quad bb = 10, \quad cc = 5 $$

This is the case of the Black-Scholes equation (in log-variables).   
The matrix A is quite big because we want to test the performances of the algorithms.

The linear system is always the same: 

$$ A x = b$$

For simplicity I chose $x = [2,...,2]$. 

<a id='sec1'></a>
## Python implementation

I wrote two functions to implement the SOR algorithm with the aim of solving PDEs. 
 - ```SOR``` uses matrix multiplications. The code is the same presented in the notebook **A1**: First it creates the matrices D,U,L (if A is sparse, it is converted into a numpy.array). Then it iterates the solutions until convergence.  
 - ```SOR2``` iterates over all components of $x$ . It does not perform matrix multiplications but it considers each component of $x$ for the computations.     
The algorithm is the following:   

```python
    x0 = np.ones_like(b, dtype=np.float64) # initial guess
    x_new = np.ones_like(x0)               # new solution
    
    for k in range(1,N_max+1):           # iteration until convergence
        for i in range(N):               # iteration over all the rows
            S = 0
            for j in range(N):           # iteration over the columns
                if j != i:
                    S += A[i,j]*x_new[j]
            x_new[i] = (1-w)*x_new[i] + (w/A[i,i]) * (b[i] - S)  
                   
        if norm(x_new - x0) < eps:       # check convergence
            return x_new
        x0 = x_new.copy()                # updates the solution 
        if k==N_max:
            print("Fail to converge in {} iterations".format(k))
```
This algorithm is taken from the SOR wiki [page](https://en.wikipedia.org/wiki/Successive_over-relaxation) and it is equivalent to the algorithm presented in the notebook **A1**.

Let us see how fast they are: (well... how **slow** they are... be ready to wait about 6 minutes)


```python
%%time
SOR(A, b)
```

    CPU times: user 979 ms, sys: 279 ms, total: 1.26 s
    Wall time: 869 ms





    array([2., 2., 2., ..., 2., 2., 2.])




```python
%%time
SOR2(A, b)
```

    CPU times: user 59.9 s, sys: 597 ms, total: 1min
    Wall time: 1min





    array([2., 2., 2., ..., 2., 2., 2.])



## TOO BAD!

The second algorithm is very bad. There is an immediate improvement to do:  
We are working with a **tridiagonal matrix**. It means that all the elements not on the three diagonals are zero. The first piece of code to modify is OBVIOUSLY this:
```python
for j in range(N):           # iteration over the columns
    if j != i:
        S += A[i,j]*x_new[j]
``` 
There is no need to sum zero elements.  
Let us consider the new function:


```python
def SOR3(A, b, w=1, eps=1e-10, N_max=100):
    N = len(b)
    x0 = np.ones_like(b, dtype=np.float64)  # initial guess
    x_new = np.ones_like(x0)  # new solution
    for k in range(1, N_max + 1):
        for i in range(N):
            if i == 0:  # new code start
                S = A[0, 1] * x_new[1]
            elif i == N - 1:
                S = A[N - 1, N - 2] * x_new[N - 2]
            else:
                S = A[i, i - 1] * x_new[i - 1] + A[i, i + 1] * x_new[i + 1]
                # new code end
            x_new[i] = (1 - w) * x_new[i] + (w / A[i, i]) * (b[i] - S)
        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new.copy()
        if k == N_max:
            print("Fail to converge in {} iterations".format(k))
```


```python
%%timeit
SOR3(A, b)
```

    83.5 ms ± 1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


### OK ... it was easy!

But wait a second... if all the elements in the three diagonals are equal, do we really need a matrix?   
Of course, we can use sparse matrices to save space in memory. But do we really need any kind of matrix?  
The same algorithm can be written considering just the three values $aa,bb,cc$.   

**In the following algorithm, even if the gain in speed is not so much, we save a lot of space in memory!!** 


```python
def SOR4(aa, bb, cc, b, w=1, eps=1e-10, N_max=100):
    N = len(b)
    x0 = np.ones_like(b, dtype=np.float64)  # initial guess
    x_new = np.ones_like(x0)  # new solution
    for k in range(1, N_max + 1):
        for i in range(N):
            if i == 0:
                S = cc * x_new[1]
            elif i == N - 1:
                S = aa * x_new[N - 2]
            else:
                S = aa * x_new[i - 1] + cc * x_new[i + 1]
            x_new[i] = (1 - w) * x_new[i] + (w / bb) * (b[i] - S)
        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new.copy()
        if k == N_max:
            print("Fail to converge in {} iterations".format(k))
            return x_new
```


```python
%%timeit
SOR4(aa, bb, cc, b)
```

    59 ms ± 242 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


<a id='sec2'></a>
## Cython


For those who are not familiar with Cython, I suggest to read this introduction [link](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html).

Cython, basically, consists in adding types to the python variables. 

Let's see what happens to the speed when we add types to the previous pure python function (SOR4)


```cython
%%cython --compile-args=-mcpu=apple-m2 --compile-args=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION --compile-args=-w --force
import numpy as np
from scipy.linalg import norm
cimport numpy as np  
cimport cython

@cython.boundscheck(False)    # turn off bounds-checking for entire function
@cython.wraparound(False)     # turn off negative index wrapping for entire function
def SOR_cy(np.float64_t aa, 
              np.float64_t bb, np.float64_t cc, 
              np.ndarray[np.float64_t , ndim=1] b, 
              double w=1, double eps=1e-10, int N_max = 100):
    
    cdef unsigned int N = b.size
    cdef np.ndarray[np.float64_t , ndim=1] x0 = np.ones(N, dtype=np.float64)     # initial guess
    cdef np.ndarray[np.float64_t , ndim=1] x_new = np.ones(N, dtype=np.float64)  # new solution
    cdef unsigned int i, k
    cdef np.float64_t S
    
    for k in range(1,N_max+1):
        for i in range(N):
            if (i==0):
                S = cc * x_new[1]
            elif (i==N-1):
                S = aa * x_new[N-2]
            else:
                S = aa * x_new[i-1] + cc * x_new[i+1]
            x_new[i] = (1-w)*x_new[i] + (w/bb) * (b[i] - S)  
        if norm(x_new - x0) < eps:
            return x_new
        x0 = x_new.copy()
        if k==N_max:
            print("Fail to converge in {} iterations".format(k))
            return x_new
```


```python
%%timeit
SOR_cy(aa, bb, cc, b)
```

    1.06 ms ± 1.09 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


### About 100 times faster!!!

That's good.

So... those who are not familiar with Cython maybe are confused about the new type `np.float64_t`. We wrote: 
```python
import numpy as np
cimport numpy as np  
```  
The first line imports numpy module in the python space. 
It only gives access to Numpy’s pure-Python API and it occurs at runtime.

The second line gives access to the Numpy’s C API defined in the `__init__.pxd` file ([link to the file](https://github.com/cython/cython/blob/master/Cython/Includes/numpy/__init__.pxd)) during compile time.  

Even if they are both named `np`, they are automatically recognized.
In `__init__.pdx` it is defined:
```
ctypedef double       npy_float64
ctypedef npy_float64    float64_t
``` 
The `np.float64_t` represents the type `double` in C.

### Memoryviews

Let us re-write the previous code using the faster [memoryviews](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html).   
I suggest to the reader to have a fast look at the memoryviews manual in the link. There are no difficult concepts and the notation is not so different from the notation used in the previous function. 

Memoryviews is another tool to help speed up the algorithm.

I have to admit that when I was writing the new code I realized that using the function `norm` is not the optimal way. (I got an error because `norm` only accepts ndarrays... so, thanks memoryviews :)  ).  
Well, the `norm` function computes a square root, which still requires some computations.  
We can define our own function `distance2` (which is the square of the distance) that is compared with the square of the tolerance parameter `eps * eps`. This is another improvement of the algorithm.


```cython
%%cython --compile-args=-O2 --compile-args=-mcpu=apple-m2 --compile-args=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION --compile-args=-w --force
import numpy as np
cimport numpy as np
cimport cython

cdef double distance2(double[:] a, double[:] b, unsigned int N):
    cdef double dist = 0
    cdef unsigned int i 
    for i in range(N):
        dist += (a[i] - b[i]) * (a[i] - b[i])
    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def SOR_cy2(double aa, 
              double bb, double cc, 
              double[:] b, 
              double w=1, double eps=1e-10, int N_max = 200):
    
    cdef unsigned int N = b.size    
    cdef double[:] x0 = np.ones(N, dtype=np.float64)          # initial guess
    cdef double[:] x_new = np.ones(N, dtype=np.float64)       # new solution
    cdef unsigned int i, k
    cdef double S
    
    for k in range(1,N_max+1):
        for i in range(N):
            if (i==0):
                S = cc * x_new[1]
            elif (i==N-1):
                S = aa * x_new[N-2]
            else:
                S = aa * x_new[i-1] + cc * x_new[i+1]
            x_new[i] = (1-w)*x_new[i] + (w/bb) * (b[i] - S)  
        if distance2(x_new, x0, N) < eps*eps:
            return np.asarray(x_new)
        x0[:] = x_new
        if k==N_max:
            print("Fail to converge in {} iterations".format(k))
            return np.asarray(x_new)
```


```python
%%timeit
SOR_cy2(aa, bb, cc, b)
```

    1.06 ms ± 5.02 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


### Good job!! Another improvement!

<a id='sec3'></a>
## C code

The last improvement is to write the function in C code and call it from python.  
Inside the folder `src/C` you can find the header file `SOR.h` and the implementation file `SOR.c` (you will find also the `mainSOR.c` if you want to test the SOR algorithm directly in C).    
I will call the function `SOR_abc` declared in the header `SOR.h`.  
First it is declared as extern, and then it is called inside `SOR_c` with a cast to `<double[:arr_memview.shape[0]]>`.


```cython
%%cython -I src/C --compile-args=-O2 --compile-args=-mcpu=apple-m2 --compile-args=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION --compile-args=-w --force
#
# The %%cython directive must be the first keyword in the cell

cdef extern from "SOR.c":
    pass
cdef extern from "SOR.h":
    double* SOR_abc(double, double, double, double *, int, double, double, int)

import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def SOR_c(double aa, double bb, double cc, B, double w=1, double eps=1e-10, int N_max = 200): 

    if not B.flags['C_CONTIGUOUS']:
        B = np.ascontiguousarray(B) # Makes a contiguous copy of the numpy array
        
    cdef double[::1] arr_memview = B    
    cdef double[::1] x = <double[:arr_memview.shape[0]]>SOR_abc(aa, bb, cc, 
                                            &arr_memview[0], arr_memview.shape[0], 
                                            w, eps, N_max)
    return np.asarray(x)
```


```python
%%timeit
SOR_c(aa, bb, cc, b)
```

    738 µs ± 3.53 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)


## Well... it looks like that using Cython with memoryviews has the same performances as wrapping a C function.

For this reason, I used the cython version as solver in the class `BS_pricer`.  
We already compared some performances in the notebook **1.2 - BS PDE**, and we saw that the SOR algorithm is slow compared to the LU or Thomas algorithms.  
Just for curiosity, let us compare the speed of the python PDE_price method implemented with cython SOR algorithm, and a pricer with same SOR algorithm fully implemented in C.


```python
from FMNM.Parameters import Option_param
from FMNM.Processes import Diffusion_process
from FMNM.BS_pricer import BS_pricer

opt_param = Option_param(S0=100, K=100, T=1, exercise="European", payoff="call")
diff_param = Diffusion_process(r=0.1, sig=0.2)
BS = BS_pricer(opt_param, diff_param)
```

<a id='sec3.1'></a>
## BS python vs C

Run the command `make` to compile the C [code](./FMNM/C/PDE_solver.c):


```python
os.system("cd ./src/C/ && make")
```

    clang  -Wall -Werror -O2 -c -o BS_SOR_main.o BS_SOR_main.c
    clang  -Wall -Werror -O2 -c -o SOR.o SOR.c
    clang  -Wall -Werror -O2 -c -o PDE_solver.o PDE_solver.c
    clang  -Wall -Werror -O2 -o BS_sor BS_SOR_main.o SOR.o PDE_solver.o        -lm                  
     
    Compilation completed!
     





    0



Python program with Cython SOR method:


```python
print("Price: {0:.6f} Time: {1:.6f}".format(*BS.PDE_price((3000, 2000), Time=True, solver="SOR")))
```

    Price: 13.269170 Time: 7.370949


Pure C program:


```python
%%time
result = subprocess.run("./src/C/BS_sor", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout.decode("utf-8"))
```

    The price is: 13.269139 
     
    CPU times: user 821 µs, sys: 5.9 ms, total: 6.72 ms
    Wall time: 5.56 s


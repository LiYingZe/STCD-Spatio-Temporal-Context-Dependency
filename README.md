# Spatio-Temporal Context Dependency(STCD) 

## What is STCD?
A novel data dependency notation, spatio-temporal context dependency (STCD), which extends the expressiveness of data quality constraints from dependence on attributes (column) and  instances (rows) for time series data. We address the compatibility of STCD with column-based constraints (i.e., SD) and row-based constraints (i.e., DC).

For example:
Sequential Dependency 

$$
Time \rightarrow_{(0.1,0.2)} \mathsf{I}
$$

Could be expressed by STCD:

$$
 \forall t_i: (t_i.\mathsf{I}\rightarrow_{-0.1,0.1}^f t_{i+1}.\mathsf{I_2} ,f(t_i.\mathsf{I_2})=t_i.\mathsf{I_2})
$$

And DC :

$$
\forall t_i: \neg(t_i.I_1>t_i.I_2)
$$

Is equal to STCD:

$$
\forall t_i:(t_i.I_1\rightarrow_{-\infty,0}^f t_i.I_2 ,f(t_i.I_1)=t_i.I_1) 
$$



## How to discover STCD? 
Next we will show how to use our code to explore STCD relationships in data


### NonLinear-STCD
First , we generate some multi-dimentional sequential data using numpy.
The Data we generated has a certain physical Nonlinear relation ship.

Here, we use the formula for generating power from physics we learned in high school: 

P=UI

That is,power equals current times voltage.
Considering that time is monotonically increasing, we use the first column as the time axis

```python
import numpy as np
T = np.linspace(1,2000,2000)# Time is Linearly in creasing
# The first and second cols have no relationships
I = np.random.randint(500,50000,2000)
U = np.random.randint(9000,10000,2000)
# The third col :According to the laws of physics, Power is equal to current times voltage
P = U*I
D = np.zeros((2000,4))
D[:,0] = T
D[:,1] = I
D[:,2] = U
D[:,3] = P
```

Then we use our nonlinear rule finder to explore the rules.

**And It only takes two lines of code!**

```python
from NonLinear import NonLinear
L = NonLinear(4, winSize=2, confidence=0.95, n_components=1, generations=20)
L.Train(D, demo=True, ColumnNames=["T", "X0","X1", "X2"])
```
The results are as follows:
<details>
  <summary>RESULTS:</summary>

```

============================
-----Rule-----
Y: <T_i0>  <-  ['<I_i0>', '<U_i0>', '<T_i1>']
Param: {'param': 'sub(<T_i1>, div(add(<T_i1>, <T_i1>), div(sub(<I_i0>, <I_i0>), <U_i0>)))'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <I_i0>  <-  ['<U_i0>', '<P_i0>']
Param: {'param': 'div(<P_i0>, <U_i0>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <U_i0>  <-  ['<I_i0>', '<P_i0>']
Param: {'param': 'div(<P_i0>, <I_i0>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <P_i0>  <-  ['<I_i0>', '<U_i0>']
Param: {'param': 'mul(<I_i0>, <U_i0>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <T_i1>  <-  ['<T_i0>', '<I_i0>', '<U_i0>', '<P_i0>', '<I_i1>', '<U_i1>']
Param: {'param': 'add(div(add(<I_i1>, <U_i0>), mul(div(div(<U_i1>, <P_i0>), <U_i1>), <I_i0>)), <T_i0>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <I_i1>  <-  ['<U_i1>', '<P_i1>']
Param: {'param': 'div(<P_i1>, <U_i1>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <U_i1>  <-  ['<I_i1>', '<P_i1>']
Param: {'param': 'div(<P_i1>, <I_i1>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <P_i1>  <-  ['<I_i1>', '<U_i1>']
Param: {'param': 'mul(<I_i1>, <U_i1>)'}
lb: -1e-07 	ub: 1e-07
============================
```

</details>




It is not difficult to find that the algorithm finds all the dependency knowledge behind the data, 
but there are many redundant rules.

Many rules shown above are equivalent. For example:
```
Y: <I_i0>  <-  ['<U_i0>', '<P_i0>']
Param: {'param': 'div(<P_i0>, <U_i0>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <U_i0>  <-  ['<I_i0>', '<P_i0>']
Param: {'param': 'div(<P_i0>, <I_i0>)'}
lb: -1e-07 	ub: 1e-07
```


This type of rule redundancy can be addressed using the implication testing mentioned in our article.

**Just adding a single argument, the problem is solved.**

```python
from NonLinear import NonLinear
L = NonLinear(4, winSize=2, confidence=0.95, n_components=1, generations=20, ImplicationCheck = True)
L.Train(D, demo=True, ColumnNames=["T", "X0","X1", "X2"])
```

And Here is the answer:

<details>
  <summary>RESULTS:Imp</summary>

```
============================
Rules Discovered: 8
Rules Implied: 3
Proportion of Redundancy Eliminated: 3 / 8  = 0.375
============================
-----Rule-----
Y: <T_i0>  <-  ['<I_i0>', '<U_i0>', '<T_i1>']
Param: {'param': 'sub(<T_i1>, div(add(<T_i1>, <T_i1>), div(sub(<I_i0>, <I_i0>), <U_i0>)))'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <I_i0>  <-  ['<U_i0>', '<P_i0>']
Param: {'param': 'div(<P_i0>, <U_i0>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <T_i1>  <-  ['<T_i0>', '<I_i0>']
Param: {'param': 'add(div(<I_i0>, <I_i0>), <T_i0>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <I_i1>  <-  ['<U_i1>', '<P_i1>']
Param: {'param': 'div(<P_i1>, <U_i1>)'}
lb: -1e-07 	ub: 1e-07
-----Rule-----
Y: <P_i1>  <-  ['<I_i1>', '<U_i1>']
Param: {'param': 'mul(<U_i1>, <I_i1>)'}
lb: -1e-07 	ub: 1e-07
============================
```
</details>
This simple demo demonstrates the effectiveness of our approach!
In this example, we find that our approach can not only discover interesting data-dependency knowledge, but also eliminate a lot of redundant rules.
But as shown in our paper, although our implication testing is theoretically 
complete, the complexity of this problem is NP for general-nonlinear functions.
Therefore, we can only eliminate part of the obvious nonlinear STCD redundancy in a limited time.

Luckily, linear STCD does not have the above problems.

### Linear-STCD

Similarly, we first construct the data.
```python
import numpy as np
A0 = np.linspace(1,2000,2000)# Linearly in creasing
A1 = np.linspace(10000,20000,2000)# Linearly in creasing
A2 = A0 * 9 + A1
D = np.zeros((2000,2))
D[:,0] = A1
D[:,1] = A2
```
Then we use our nonlinear rule finder to explore the rules.

```python
from Linear import Linear
L = Linear(3,winSize=2,confidence=0.95,n_components=1,MaxLen=3,ImplicationCheck =True)
L.Train(D,demo=True)
```
And the result is listed as below:



<details>
  <summary>RESULTS-LIN</summary>
```
Rules Discovered: 6
Rules Implied: 1
Proportion of Redundancy Eliminated: 1 / 6  = 0.16666666666666666
============================
Rule Discovered:
-----Rule-----
Y: <0-0>  <-  ['<0-1>', '<1-0>']
Param: {'Type': 'Linear', 'coef': [0.0, 1.0], 'intercept': -1.0}
lb: -1.0825797627565288e-06 	ub: 1.0825797627565288e-06
-----Rule-----
Y: <0-1>  <-  ['<0-0>', '<0-2>']
Param: {'Type': 'Linear', 'coef': [-1.0, 1.0], 'intercept': -0.0}
lb: -4.6620296436523785e-06 	ub: 4.6620296436523785e-06
-----Rule-----
Y: <0-2>  <-  ['<0-0>', '<0-1>']
Param: {'Type': 'Linear', 'coef': [1.0, 1.0], 'intercept': 0.0}
lb: -2.5561066336864683e-06 	ub: 2.5561066336864683e-06
-----Rule-----
Y: <1-1>  <-  ['<0-0>', '<1-2>']
Param: {'Type': 'Linear', 'coef': [-1.0, 1.0], 'intercept': -1.0}
lb: -4.661308252601545e-06 	ub: 4.661308252601545e-06
-----Rule-----
Y: <1-2>  <-  ['<0-0>', '<1-1>']
Param: {'Type': 'Linear', 'coef': [1.0, 1.0], 'intercept': 1.0}
lb: -2.558575209116962e-06 	ub: 2.558575209116962e-06
```
</details>
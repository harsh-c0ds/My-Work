#!/usr/bin/env python
# coding: utf-8

# PROBLEM 3 - SHEET 5 
# 
# We chose to solve numerically the following integrals
# \begin{equation}
#     \int^1_0\left(\int_0^2xy^2dx\right)dy=\frac{2}{3},
# \end{equation}
# \begin{equation}
#     \int^1_0\left(\int_{2y}^2xy^2dx\right)dy=\frac{4}{15},
# \end{equation}
# \begin{equation}
#     \int^2_0\left(\int_0^{\frac{x}{2}}xy^2dy\right)dx=\frac{4}{15}.
# \end{equation}
# 
# This is the function that we defined in the solution of PROBLEM II.

# In[85]:


import numpy as np
import math

def simpson(x,y_i,h):
    t1 = 0
    
    for i in range(len(x)):
        if i == 0 or i == n_x-1: # Necessary conditions for choosing the coefficients
            t1 += f(x[i], y_i)
        elif i % 2 == 0:
            t1 += 2*f(x[i], y_i)
        else:
            t1 += 4*f(x[i], y_i)
    
    return (h/3)*(t1)


# In this section we compute
# \begin{equation}
#     \int^1_0\left(\int_0^2xy^2dx\right)dy=\frac{2}{3}.
# \end{equation}
# 
# For each $y$ we apply the Simpson's rule for the x integration and subsequently we moltiply by the proper coefficient. At the end of the loop we have to multiply by $\frac{h_y}{3}$ to complete the procedure.
# 
# We obtain 0.6664448902222215, which is correct up to the third digit.

# In[87]:


# Defining the function
def f(x,y):
    return x * y**2

# Limits of integration
a_x = 0
b_x = 2
a_y = 0
b_y = 1

# Number of intervals
n_x = 10**3 
n_y = 10**3

# Step
h_x = abs(b_x - a_x) / n_x
h_y = abs(b_y - a_y) / n_y

x = np.linspace(a_x, b_x, n_x + 1) # n_x + 1, because this is the number of points
y = np.linspace(a_y, b_y, n_y + 1)

final_result = 0

# For each y I apply the Simpson's rule for the x integration
for i in range(np.size(y)):
    y_i = y[i]
    t1 = simpson(x,y_i,h_x)
    if i % 2 == 1:
        t1 = 4 * t1
    elif i % 2 == 0:
        t1 = 2 * t1
        
    final_result += t1

final_result *= h_y / 3

print(f"The result of the double integral is: {final_result}")


# Let's move on to the second integral
# \begin{equation}
#     \int^1_0\left(\int_{2y}^2xy^2dx\right)dy=\frac{4}{15}.
# \end{equation}
# 
# The procedure is the same as before with the only difference that the integration limit $a_x=2y$ must be dynamically evaluated for each y.
# We obtain 0.2664445777775108, which is correct up to the third digit.

# In[89]:


# Limits of integration
b_x = 2
a_y = 0
b_y = 1

# Number of intervals
n_x = 10**3 
n_y = 10**3

# Step
h_y = abs(b_y - a_y) / n_y

y = np.linspace(a_y, b_y, n_y + 1)

final_result = 0

# For each y I apply the Simpson's rule for the x integration
for i in range(np.size(y)):
    y_i = y[i]
    # I have to dynamically define the limit a_x and consequently x and h_x
    a_x = 2 * y_i
    x = np.linspace(a_x, b_x, n_x + 1) 
    h_x = abs(b_x - a_x) / n_x
    t1 = simpson(x,y_i,h_x)
    if i % 2 == 1:
        t1 = 4 * t1
    elif i % 2 == 0:
        t1 = 2 * t1
        
    final_result += t1

final_result *= h_y / 3

print(f"The result of the double integral is: {final_result}")


# We are left with
# \begin{equation}
#     \int^2_0\left(\int_0^{\frac{x}{2}}xy^2dy\right)dx=\frac{4}{15}.
# \end{equation}
# 
# The procedure is exactly the same as before.
# However, now we have to perform the integral over y first. Thus, we have to modify our function accordingly.

# In[108]:


def simpson2(y,x_i,h):
    t1 = 0
    
    for i in range(len(y)):
        if i == 0 or i == n_y-1: # Necessary conditions for choosing the coefficients
            t1 += f(x_i, y[i])
        elif i % 2 == 0:
            t1 += 2*f(x_i, y[i])
        else:
            t1 += 4*f(x_i, y[i])
    
    return (h/3)*(t1)


# In[110]:


# Limits of integration
a_x = 0
b_x = 2
a_y = 0

# Number of intervals
n_x = 10**3 
n_y = 10**3

# Step
h_x = abs(b_x - a_x) / n_x

x = np.linspace(a_x, b_x, n_x + 1)

final_result = 0

# For each x I apply the Simpson's rule for the y integration
for i in range(np.size(x)):
    x_i = x[i]
    # I have to dynamically define the limit b_y and consequently y and h_y
    b_y = x_i / 2
    y = np.linspace(a_y, b_y, n_y + 1) 
    h_y = abs(b_y - a_y) / n_y
    t1 = simpson2(y,x_i,h_y)
    if i % 2 == 1:
        t1 = 4 * t1
    elif i % 2 == 0:
        t1 = 2 * t1
        
    final_result += t1

final_result *= h_x / 3

print(f"The result of the double integral is: {final_result}")


# In[ ]:





#creating matrices using numpy
import numpy as np 
A = np.array([[1, 2], [3, 4]]) #represents row vectors in a matrix
B = np.array([[5, 6],[7, 8]])
A*B #elementwise product -> not defined in Study's Linear Algebra, a*b(ij) = aij * bij
print(A@B) #matrix operation ->defined in Multiplication of Transformation Matrix in Study's Linear Algebra 
import sympy as sy #calculating library
A = sy.Matrix([[3, 4], [7, 8]])
B = sy.Matrix([[5, 3], [2, 1]])
A@B != B@A #shown in Multiplication of Transformation Matrix in Study's Linear Algebra 
#using variables as symbols -> creating variable math not number results
x1, x2, y1, y2 = sy.symbols('x1, x2, y1, y2', real = True) 
#create column vectors using variables
x = sy.Matrix([x1, x2])
y = sy.Matrix([y1, y2])
#.T transposes the matrix, same as transpose() method
#Transpose in Study's Linear Algebra 
print(x.T@y)
print(y.T@x)
#prove (A dot B).T = B.T dot A.T
a, b, c, d, e, f, g, h, i, j, k, l = sy.symbols('a, b, c, d, e, f, g, h, i, j, k, l', real = True)
A = sy.Matrix([[a, b], [c, d], [e, f]])
B = sy.Matrix([[g, h, i], [j, k, l]])
AB = A@B
print(AB)
AB_tr = AB.transpose()
AB_tr #A dot B transpose
B_tr_A_tr = B.transpose()@A.transpose()
B_tr_A_tr #B.T dot A.T
AB_tr == B_tr_A_tr #check
import matplotlib.pyplot as plt
#Some examples about system of linear equation in Study's Linear Algebra
#linear equation solving, number of equations = 2
#forming f:I -> X
x1 = np.linspace(-5, 5, 100) #I
x2_1 = -x1 + 6 #equation for f1(i)
x2_2 = x1 + 4 #equation for f2(i)
#creating plots 
fig, ax = plt.subplots(figsize = (12, 7)) #creates a figure, window for a graph, and axe, a box where the graph is plotted
#plotting the solution with a red dot
ax.scatter(1, 5, s = 200, zorder=5, color = 'r', alpha = .8) 
#plotting (x_1, x_2) that are solutions of x_1 + x_2 = 6, f1: I -> x2_1
ax.plot(x1, x2_1, lw =3, label = '$x_1+x_2=6$') #dollar signs are used in matplot to show beginning and end of an expression
#plotting (x_1, x_2) that are solutions of x_1 - x_2 = 4, f2: I -> x2_2
ax.plot(x1, x2_2, lw =3, label = '$x_1-x_2=-4$')
#plotting blue lines that shows the solution 
ax.plot([1, 1], [0, 5], ls = '--', color = 'b', alpha = .5)
ax.plot([-5, 1], [5, 5], ls = '--', color = 'b', alpha = .5)
#setting the axis
ax.set_xlim([-5, 5])
ax.set_ylim([0, 12])
#choose to show plot legend
ax.legend()
#indicating the tuple for the solution
s = '$(1,5)$' #
ax.text(1, 5.5, s, fontsize = 20)
#setting the title
ax.set_title('Solution of $x_1+x_2=6$, $x_1-x_2=-4$', size = 22) 
#choose to show grid
ax.grid()
#shows created graphs so far
plt.show()
#linear equation solving, number of equations = 3 
#creating I with dim(I) = 2
x1 = np.linspace(-10, 10, 20)
x2 = np.linspace(-10, 10, 20)
#X1's elements represent the first element of I's tuple, X2 is the second
X1, X2 = np.meshgrid(x1, x2) 
#creating a figure
fig = plt.figure(figsize = (9, 9))
#makes an axe, places the axe, set it as a 3d graph
ax = fig.add_subplot(111, projection = '3d')
X3_1 = (6 - 2*X2 - X1) * (1/3) #f1(i) equation
#plotting a 2d figure 
ax.plot_surface(X1, X2, X3_1, cmap ='viridis', alpha = 1) #f1: I -> X3_1 plotting
X3_2 = (4 - 5*X2 - 2*X1) * (1/2) #f2(i) equation
ax.plot_surface(X1, X2, X3_2, cmap ='summer', alpha = 1) #f2: I -> X3_2 plotting
X3_3 = 2  + 3*X2 - 6*X1 #f3(i) equation
ax.plot_surface(X1, X2, X3_3, cmap ='spring', alpha = 1) #f3: I -> X3_3 plotting
#labelling the axis
ax.set_xlabel('$x_1$-axis')
ax.set_ylabel('$x_2$-axis')
ax.set_zlabel('$x_3$-axis')
#showing the figure
plt.show()
#represent Ax = b system
A = sy.Matrix([[1, 2, 3], [2, 5, 2], [6, -3, 1]]) #A
#use sympy to idnetify column vectors 
print('column vectors:', A.col(0), A.col(1), A.col(2))
#create 3d figure
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(111, projection = '3d')
#plotting b
ax.scatter(6, 4, 2, s = 200, color = 'red')
from util.plot_helpers import plot_vec, plot_vecs, autoscale_arrows
plot_vecs(A.col(0), A.col(1), A.col(2)) #util function to plot vectors
autoscale_arrows() #scaling the vectors
plt.show()
#determinant relation 
print(A.det()) #has a value meaning unique solution 
#the figure shows the combiation of these vectors should point at the red dot 
#use sympy to solve the system 
#make symbols
A = sy.Matrix(((1,2,3),(2,5,2),(6,-3,1)))
b = sy.Matrix((6,4,2))
system = A, b
from sympy.solvers.solveset import linsolve
#use linsolve function in sympy to solve the system -> column point of view, row point of view: finding each variables 
linsolve(system)
#no solution case 
A = sy.Matrix([[1, 1, 1], [1, -1, -2], [2, 0, -1]])
#plotting the case 
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(1, 2, 1, s = 200, color = 'red')
plot_vecs(A.col(0), A.col(1), A.col(2))
autoscale_arrows()
plt.show()
#solving the case
A = sy.Matrix(((1,1,1),(1,-1,-2),(2,0,-1)))
b = sy.Matrix((1,2,1))
system = A, b
print(linsolve(system))
#check the relationship of determinant and the system
print(A.det()) #0 #Theorem 4.9 and explanation after Theorem 3.10
#infinite solution case 
#plotting the case
A = sy.Matrix([[0, 1, -1], [2, 1, 2], [2, 2, 1]])
x1, x2, x3 = sy.symbols('x1 x2 x3') #setting the variables
x = sy.Matrix([x1,x2,x3]) #setting the variable vector
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(111, projection = '3d')
plot_vecs(A.col(0), A.col(1), A.col(2))
autoscale_arrows()
ax.scatter(4,4,8, s = 200, color = 'red')
ax.set(xlim = [0.,5.], ylim = [0.,5.,], zlim = [0., 10.]) #setting the axis
plt.show()
#solving the system
A = sy.Matrix(((0,1,-1),(2,1,2),(2,2,1)))
b = sy.Matrix((4,4,8))
system = A,b
#infinite answers
print(linsolve(system, x1, x2, x3)) #linsolve function returns answer with variables when variable objects are in
#determinant relation
print(A.det()) #0 #Theorem 4.9 and explanation after Theorem 3.10
#Inverse in Linear Algebra 
#Showing some logic in the proof of Theorem 3.10 
#making some matrices 
A = np.array([[1, 2, 3], [2, 5, 2], [6, -3, 1]])
b = np.array([6,4,2])
#Create inverse
A_inv = np.linalg.inv(A)
A_inv_b = A_inv @ b  # A^(-1)b
x = A_inv_b  # x = A^(-1)b
print(x)
#Checking the result
print(A@x)
print(b)
#Linear combination in Linear Algebra
fig, ax = plt.subplots(figsize=(8, 8))
#set of vectors needed to be shown a linear combination 
vec = np.array([[0,0,4,2],
                 [0,0,-2,2],
                 [0,0,2,10],
                 [0,0,8,4], 
                 [0,0,-6,6]]) #one 3d tensor, 5 matrices, one vector each in them, 4 elements each in it
#list created to set colors 
colors = ['b','b','r','b','b']
#tail이 origin, head가 (4,2), (-2,2), (2,10), (8,4), (-6,6)인 vector plot
for i in range(vec.shape[0]): #for the number of vectors in the set 
    X,Y,U,V = zip(vec[i,:]) #unpacking created tuple from the set 
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', color = colors[i], scale=1, alpha = .6) #plotting arrows
    ax.text(x = vec[i,2], y = vec[i,3], s = '(%.0d, %.0d)' %(vec[i,2],vec[i,3]), fontsize = 16) #labeling 
# tail이 (8,4), head가 (2,10)인 vector plot
points12 = np.array([[8,4],[2,10]])
ax.plot(points12[:,0], points12[:,1], c = 'b', lw = 3.5,alpha =0.5, ls = '--')
# tail이 (-6,6), head가 (2,10)인 vector plot
points34 = np.array([[-6, 6],[2,10]])
ax.plot(points34[:,0], points34[:,1], c = 'b', lw = 3.5,alpha =0.5, ls = '--')
#creating axis
ax.set_xlim([-10, 10])
ax.set_ylim([0, 10.5])
ax.set_xlabel('x-axis', fontsize =16)
ax.set_ylabel('y-axis', fontsize =16)
#show grid
ax.grid()
# 붉은색 격자 plot
a = np.arange(-11, 20, 1)
x = np.arange(-11, 20, 1)
for i in a:    
    y1 = i + 0.5*x  # 0.5(기울기) = 2/4
    ax.plot(x, y1, ls = '--', color = 'pink', lw = 2)
    y2 = i - x  # -1(기울기) = 2/(-2)
    ax.plot(x, y2, ls = '--', color = 'pink', lw = 2)    
#title the plot
ax.set_title('Linear Combination of Two Vectors in $\mathbf{R}^2$', size = 22, x =0.5, y = 1.01)
plt.show()
#Basis in Linear Algebra 
#create vectors 
from util.plot_helper import *
vectors = [(2,2)]
tails = [(-3,-2), (-3,1), (0,0), (1,-3)]
plot_vector(vectors, tails)
pyplot.title("The same vector, with its tail at four locations.")
plt.show()
#express the vectors as linear combinations
#basis vector
i = np.array((1,0))
j = np.array((0,1))
vec = 3*i + 2*j
vectors = [i, j, 3*i, 2*j, vec]
plot_vector(vectors)  
pyplot.title("The vector $(3,2)$ as a linear combination of the basis vectors.")
plt.show()
#Span in Linear Algebra 
#span
from numpy.random import randint
vectors = []
#basis 
i = np.array((1,0))
j = np.array((0,1))
#create 1000 random linear combinations of the vectors and plot
for _ in range(1000):
    m = np.random.randint(-10,10)
    n = np.random.randint(-10,10)
    vectors.append(m*i + n*j)  # i, j (basis vecor)의 linear combination
plot_vector(vectors) #plotting them 
pyplot.title("1000 random vectors, created from the basis vectors") #titling the plot
#changing the basis 
vectors = []
i = numpy.array((-2,1))
j = numpy.array((1,-3))
for _ in range(1000):
    m = np.random.randint(-10,10)
    n = np.random.randint(-10,10)
    vectors.append(m*i + n*j)  # i, j (basis vecor)의 linear combination
plot_vector(vectors)
pyplot.title("1000 random vectors, created from [-2,1], [1,-3]")
#span of none basis 
vectors = []
i = numpy.array((-2,1))
j = numpy.array((-1,0.5))
for _ in range(1000):
    m = randint(-10,10)
    n = randint(-10,10)
    vectors.append(m*i + n*j)  # i, j (basis vecor)의 linear combination
plot_vector(vectors)
pyplot.title("1000 random vectors, created from [-2, 1], [-1, 0.5]") #only 1 dimensional vector space is covered
#identifying a subspace from Linear Algebra 
#create a plot
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(projection='3d') #axe
#create meshgrid 
x2 = np.linspace(-2, 2, 10)
x3 = np.linspace(-2, 2, 10)
X2, X3 = np.meshgrid(x2, x3)
X1 = 3*X2 + 2*X3 #from this equation, vectors in the set for this span are (3,1,0) and (2,0,1)
#adds lattice like frame in the plot
ax.plot_wireframe(X1, X2, X3, linewidth = 1.5, color = 'g', alpha = .6)
#plot vectors 
vec = np.array([[[0, 0, 0, 3, 1, 0]],
               [[0, 0, 0, 2, 0, 1]],
               [[0, 0, 0, 10, 2, -2]]])
colors = ['r', 'b', 'purple']
for i in range(vec.shape[0]):
    X, Y, Z, U, V, W = zip(*vec[i,:,:])
    ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = colors[i],
              arrow_length_ratio = .08, pivot = 'tail',
              linestyles = 'solid',linewidths = 3, alpha = .6)
#plot grey parallel lines 
point12 = np.array([[2, 0, 1],[5, 1, 1]])
ax.plot(point12[:,0], point12[:,1], point12[:,2], lw =3, ls = '--', color = 'black', alpha=0.5)
point34 = np.array([[3, 1, 0], [5, 1, 1]])
ax.plot(point34[:,0], point34[:,1], point34[:,2], lw =3, ls = '--', color = 'black', alpha=0.5)
#labeling coordinates 
ax.text(x = 3, y = 1, z = 0, s='$(3, 1, 0)$', color = 'red', size = 16)
ax.text(x = 2, y = 0, z = 1, s='$(2, 0, 1)$', color = 'blue', size = 16)
ax.text(x = 10, y = 2, z = -2.3, s='$v (10,2,-2))$', color = 'purple', size = 16)
#labeling axis
ax.set_xlabel('x-axis', size = 18)
ax.set_ylabel('y-axis', size = 18)
ax.set_zlabel('z-axis', size = 18)
ax.view_init(elev=-29, azim=130)
plt.show()
#want to see the dimension of the subspace 
#by definition of rank(L^A) in Linear Algebra, let A = ((3,0,1),(2,1,0))
A = np.array([(3,0,1),(2,1,0)])
print(np.linalg.matrix_rank(A))
#rank A is 2 therefore dim(subspace) = 2 
#by Theorem 3.5 in Linear Algebra, one can see the set is a basis for the subspace 
#then one can see, the purple vector, let it be v, is not in the subspace
#then by the definition of linear system in Linear Algebra, such s does not exist for As = v
x1, x2 = sy.symbols('x1 x2') #setting symbols 
#idnetifying the system
A = sy.Matrix(((3,2),(1,0),(0,1)))
b = sy.Matrix((10,2,-2))
system = A,b
#solving it
print(linsolve(system, x1, x2))
#Linear transformation and change of coordinate basis in Linear Algebra 
#let basis r be = (v = (-2,1), w = (1,-3)) and A be and b be a standard basis for F^2
A = np.array([[-2,1],[1,-3]])
#let vectors, x and x_ be
x = np.array([-7,11])
x_ = np.array([2,-3])
#then by the definition of coordinate vector in Linear Algebra
#x = [x]^b and x_ = [x]^r
#And from the definition of Matrix Representation of Transformation in Linear Algebra 
#A = [I]^r,b. Then 
#[I(x)]^b is [x]^b by definition of identity transformation in Linear Algebra
#By Theorem 2.14 in Linear Algebra, [I(x)]^b = [x]^b = [I]^r,b[x]^r, 
#Then from above, x = Ax_
#one can check using python 
print(A@x_)
print(x)
#Then from above, A simply changes basis representation of a vector 
#let i and j be vectors in b 
i = np.array([1,0])
j = np.array([0,1])
v = np.array([-2,1])
w = np.array([1,-3])
#changing i to v, j to w
print(A@i, v)
print(A@j, w)
#visualize the change 
plot_linear_transformation(A)
plt.show()
#another example 
m = np.array([1,2])
n = np.array([2,1])
M = np.array([[1,2], [2,1]])
print(M@i, m)
print(M@j, n)
#visualize
plot_linear_transformation(M)
plt.show()
#visualizing vector representation change
x = np.array((0.5,1))
vectors = [x, M.dot(x)]  
plot_vector(vectors)
plt.show()
#do this for the first example 
y = numpy.array((2,-3))
vectors = [y, A.dot(y)]
plot_vector(vectors)
plt.show()
#Inverse Matrix in Linear Algebra 
#MM^-1 = I
#define M
M = numpy.array([[1,1], [0,-1]])
#find inverse using python 
M_inv = np.linalg.inv(M)
#use python package to visualize, then from logic above, this is just coming back from a different basis
plot_linear_transformations(M, M_inv) 
plt.show()
#The volume of a parallelogram in Linear Algebra and rientation in Linear Algebra 
#defining custom plotting function, plots 2d vectors 
def plotvectors(vecs, colors, alpha=1):
    """
    Determinant 섹션의 plot에 사용하기 위한 custom function입니다.

    Parameters
    ----------
    vecs : plot할 vector(numpy array)의 list  (e.g., [[1, 3], [2, 2]] )
    colors : 각 vector의 color  (e.g., ['red', 'blue'])
    alpha : 투명도

    Returns
    -------
    fig : figure
    """
    plt.axvline(x=0, color='#A9A9A9', zorder=0) #add vertical line in an axes
    plt.axhline(y=0, color='#A9A9A9', zorder=0) #add horizontal line in an axes

    for i in range(len(vecs)):
        if (isinstance(alpha, list)): #check the type of the alpha values 
            alpha_i = alpha[i] #if its list, then apply each alpha value to each vector
        else:
            alpha_i = alpha 
        x = np.concatenate([[0,0],vecs[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]], angles='xy', 
                   scale_units='xy', scale=1, color=colors[i],alpha=alpha_i) #plotting the vectors 
#determine the color of vectors 
orange = '#FF9A13' 
blue = '#1190FF'
i = [1, 0]
j = [0, 1]
#create a figure 
fig = plt.figure(figsize = (4,4))
#plotting the vectors
plotvectors([i, j], [[blue], [orange]], alpha=1)
#add parallel lines 
plt.plot([0, 1], [1, 1], ls = '--', color = 'black', alpha = .5)
plt.plot([1, 1], [0, 1], ls = '--', color = 'black', alpha = .5)
#scale the axis
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.show()
#define a matrix and find ran(L^A)
A = np.array([[2, 0], [0, 2]])
#standard vectors are changed 
new_i = A.dot(i)
new_j = A.dot(j)
#create a figure 
fig = plt.figure(figsize = (4,4))
#plotting vectors 
plotvectors([new_i, new_j], [[blue], [orange]])
#parallel lines 
plt.plot([0, 2], [2, 2], ls = '--', color = 'black', alpha = .5)
plt.plot([2, 2], [0, 2], ls = '--', color = 'black', alpha = .5)
#scaling 
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 3)
plt.show()
#use determinant to find the volume difference between two spaces 
I = np.array([[1, 0], [0, 1]]) #volume from two original vectors 
print(np.linalg.det(I)) #is 1 
print(np.linalg.det(A)) #volume from changed vectors is 4
#the change makes the volume 4 times bigger
#det(A)/det(I) -> amount of volume change 
#define a new space 
#define a new matrix
B = np.array([[-2, 0], [0, 2]])
#new vectors 
new_i_1 = B.dot(i)
new_j_1 = B.dot(j)
#create a figure 
fig = plt.figure(figsize = (4,4))
#plotting them 
plotvectors([new_i_1, new_j_1], [['#1190FF'], ['#FF9A13']])
#scaling the axis
plt.xlim(-3, 0.5)
plt.ylim(-0.5, 3)
plt.show()
print(np.linalg.det(B))
#determinant is 4 so volume increased 4 times greater, but the sign is different 
#by the definition of orientation in Linear Algebra, B has negative orientation
#In conclusion
#orientation -> the amount of rotation occured for vectors
#when determinant is zero, by the properties and theorems about a determinant in Linear Algebra,  
#the orientation and change of volume cannot be measured, since the dimension of volume has changed 
#Corollary 15 in Linear Algebra
#r basis matrix
Q = np.array([[2,1],[1,2]])
#transformation matrix 
R = np.array([[0,1],[-1,0]]) #L^A: V -> V, couterclockwise 90 degree 
#inverse
Q_inv = np.linalg.inv(Q)
#let x be [x]^b, and x_ = be [x]^r
x = np.array([-3,0])
#as Q is [I]^r,b
x_ = np.linalg.solve(Q, x)
#Then, D = Q_inv@R@Q is change of basis for transformation
a = np.array([2,1])
b = np.array([1,2])
plot_change_basis(a, b, x_)
plt.show()
plot_vector([x, x_]) #the same from above, so one can see this is the result of a basis change  
plt.show()
v = Q_inv@R@Q@x_ #change of basis for transformation applied 
plot_vector([x_, v]) #the amount change of a transformation is different when the basis change
plt.show()
#Least square approximation in R -> A* is A^T from the explanation about Unitary matrix in Li
#First, create a data set 
weight = np.array([60,65,55,70,45,50])
height = np.array([177,170, 175, 180, 155, 160])
smoking = np.array([1, 0, 0, 1, 1, 0])
#life span 
y = np.array([66, 74, 78, 72, 70, 80])
#using only weight to find parameter for least square, use Theorem 6.12
A = weight.reshape(6,1)
#the parameter is found 
x_1 = np.linalg.inv(A.T@A)@A.T@y
#the predicted life span would be from the proof of Theorem 6.12
y_1 = A@x_1
#least square error would be 
e = np.linalg.norm(y_1 - y) 
print(e)
#now use weight and height
A = np.vstack([weight.reshape(1,6), height.reshape(1,6)]).T
#parameter
x_2 = np.linalg.inv(A.T@A)@A.T@y
#prediction 
y_2 = A@x_2
#least square error
e = np.linalg.norm(y_2 - y) 
print(e)
#now use all 
A = np.vstack([A.reshape(2,6), smoking.reshape(1,6)]).T
#parameter
x_3 = np.linalg.inv(A.T@A)@A.T@y
#prediction 
y_3 = A@x_3
#least square error
e = np.linalg.norm(y_3 - y)
print(e)
#one can see from the result that smoking variables does not help 
#use built in library function to solve this 
x_3 = np.linalg.lstsq(A, y,rcond=None)
#using curve fitting, similar to numpy's lstsq
from scipy import optimize
#define function for y_4 
def func(data, x_4_1, x_4_2, x_4_3):
  return data[0]*x_4_1 + data[1]*x_4_2 + data[2]*x_4_3
#create python list data set from np.array
weight = []
height = []
smoking = []
for item in A:
  weight.append(item[0])
  height.append(item[1])
  smoking.append(item[2])
#method = 'lm' means the function will use least square method 
#returns parameter, covariance of the parameter
x_3 = optimize.curve_fit(func, xdata = [weight,height,smoking], ydata = list(y), method = 'lm')[0]
#Gram_Schmidts
#setting vector space 
s = np.linspace(-1, 1, 10)
t = np.linspace(-1, 1, 10)
S, T = np.meshgrid(s, t)
#three vectors as a basis 
vec = np.array([[[0,0,0,3, 6, 2]],
             [[0,0,0,1, 2, 4]],
             [[0,0,0,2, -2, 1]]])
#plotting span of the basis in the space 
X = vec[0,:,3] * S + vec[1,:,3] * T
Y = vec[0,:,4] * S + vec[1,:,4] * T
Z = vec[0,:,5] * S + vec[1,:,5] * T
#visualization
fig = plt.figure(figsize = (7, 7)) #creating a box
#adding a plot in the box
ax = fig.add_subplot(projection='3d')
#creating a frame of span in the plot
ax.plot_wireframe(X, Y, Z, linewidth = 1.5, alpha = .3)
#plotting each vector in the basis 
colors = ['r','b','g']
s = ['$x_1$', '$x_2$', '$x_3$']
for i in range(vec.shape[0]):
    X,Y,Z,U,V,W = zip(*vec[i,:,:])
    ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False,
              color = colors[i], alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
              linestyles = 'solid',linewidths = 3)
    ax.text(vec[i,:,3][0], vec[i,:,4][0], vec[i,:,5][0], s = s[i], size = 15)
#labeling the axis
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
plt.show()
#now find projections of the vectors 
#use the equation from Theorem 6.4 in Linear Algebra 
x1 = np.array([3, 6, 2])
x2 = np.array([1, 2, 4])
x3 = np.array([2, -2, 1])

v1 = x1 
u1 = v1 / np.linalg.norm(v1)
x2_hat = (((x2@u1) / (u1@u1)) * u1)
v2 = x2 - x2_hat
u2 = v2 / np.linalg.norm(v2)

print(u1 @ u2)
print(x2_hat @ u2)

#creating vector space
s = np.linspace(-1, 1, 10)
t = np.linspace(-1, 1, 10)
S, T = np.meshgrid(s, t)

#span of two vectors 
X = x1[0] * S + x2[0] * T
Y = x1[1] * S + x2[1] * T
Z = x1[2] * S + x2[2] * T
#creating a box
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(projection='3d') #adding plot 
#adding frame
ax.plot_wireframe(X, Y, Z, linewidth = 1.5, alpha = .3)
#plotting vectors 
vec = np.array([[0, 0, 0, x1[0], x1[1], x1[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'red', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, x2[0], x2[1], x2[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'blue', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, x3[0], x3[1], x3[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'green', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, x2_hat[0],x2_hat[1], x2_hat[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'blue', alpha = .6,arrow_length_ratio = .12, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, v2[0], v2[1], v2[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'purple', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

#adding labels to the vectors 
ax.text(x1[0], x1[1], x1[2], '$\mathbf{x}_1 = \mathbf{v}_1 $', size = 15)
ax.text(x2[0], x2[1], x2[2], '$\mathbf{x}_2$', size = 15)
ax.text(x3[0], x3[1], x3[2], '$\mathbf{x}_3$', size = 15)
ax.text(x2_hat[0], x2_hat[1], x2_hat[2], '$\hat{\mathbf{x_2}}$', size = 15)
ax.text(v2[0], v2[1], v2[2], '$\mathbf{v}_2$', size = 15)
#adding labels to the axis
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
#adding dashed line
point1 = [x2_hat[0], x2_hat[1], x2_hat[2]]
point2 = [x2[0], x2[1], x2[2]]
line1 = np.array([point1, point2])
ax.plot(line1[:,0], line1[:,1], line1[:, 2], c = 'b', lw = 3.5,alpha =0.5, ls = '--')

point1 = [v2[0], v2[1], v2[2]]
point2 = [x2[0], x2[1], x2[2]]
line1 = np.array([point1, point2])
ax.plot(line1[:,0], line1[:,1], line1[:, 2], c = 'b', lw = 3.5,alpha =0.5, ls = '--')
plt.show()
#use the equation to find the third vector 
x3_hat = (((x3 @ u1) / (u1 @ u1)) * u1) + (((x3 @ u2) / (u2 @ u2)) * u2)
v3 = x3 - x3_hat
u3 = v3 / np.linalg.norm(v3)

print(u1@ u3)
print(u2 @ u3)
print(x3_hat @ u3)

#creating a vector space
s = np.linspace(-1, 1, 10)
t = np.linspace(-1, 1, 10)
S, T = np.meshgrid(s, t)
#plotting span of x1 and x2
X = x1[0] * S + x2[0] * T
Y = x1[1] * S + x2[1] * T
Z = x1[2] * S + x2[2] * T
#making a box
fig = plt.figure(figsize = (9, 9))
ax = fig.add_subplot(projection='3d') #adding plot
#plot the space in as frame
ax.plot_wireframe(X, Y, Z, linewidth = 1.5, alpha = .3)
#plotting vectors
vec = np.array([[0, 0, 0, x1[0], x1[1], x1[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'red', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, x2[0], x2[1], x2[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'red', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, x2_hat[0],x2_hat[1], x2_hat[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'blue', alpha = .6,arrow_length_ratio = .12, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, v2[0], v2[1], v2[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'purple', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)

vec = np.array([[0, 0, 0, x3[0], x3[1], x3[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'red', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)


vec = np.array([[0, 0, 0, x3_hat[0], x3_hat[1], x3_hat[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'black', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)


vec = np.array([[0, 0, 0, v3[0], v3[1], v3[2]]])
X, Y, Z, U, V, W = zip(*vec)
ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, color = 'purple', alpha = .6,arrow_length_ratio = .08, pivot = 'tail',
          linestyles = 'solid',linewidths = 3)
#labeling the vectors 
ax.text(x1[0], x1[1], x1[2], '$\mathbf{x}_1 = \mathbf{v}_1 $', size = 15)
ax.text(x2[0], x2[1], x2[2], '$\mathbf{x}_2$', size = 15)
ax.text(x2_hat[0], x2_hat[1], x2_hat[2], '$\hat{\mathbf{x}}_2$', size = 15)
ax.text(v2[0], v2[1], v2[2], '$\mathbf{v}_2$', size = 15)
ax.text(x3[0], x3[1], x3[2], '$\mathbf{x}_3$', size = 15)
ax.text(x3_hat[0], x3_hat[1], x3_hat[2], '$\hat{\mathbf{x}}_3$', size = 15)
ax.text(v3[0], v3[1], v3[2], '$\mathbf{v}_3$', size = 15)
#labeling the axes
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
#adding dashed lines
point1 = [x2_hat[0], x2_hat[1], x2_hat[2]]
point2 = [x2[0], x2[1], x2[2]]
line1 = np.array([point1, point2])
ax.plot(line1[:,0], line1[:,1], line1[:, 2], c = 'b', lw = 3.5,alpha =0.5, ls = '--')

point1 = [v2[0], v2[1], v2[2]]
point2 = [x2[0], x2[1], x2[2]]
line1 = np.array([point1, point2])
ax.plot(line1[:,0], line1[:,1], line1[:, 2], c = 'b', lw = 3.5,alpha =0.5, ls = '--')


point1 = [x3_hat[0], x3_hat[1], x3_hat[2]]
point2 = [x3[0], x3[1], x3[2]]
line1 = np.array([point1, point2])
ax.plot(line1[:,0], line1[:,1], line1[:, 2], c = 'b', lw = 3.5,alpha =0.5, ls = '--')
#cutting the axes 
ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5, 5)
ax.set_zlim3d(-5, 5)
plt.show()
#from orthogonal property and definition of matrix multiplication 
#stacking u1, u2, u3 will give an orthogonal matrix 
#then from the definition of orthogonal matrices 
U = np.vstack((u1,u2,u3)).T
U.T@U #this is an indentity matrix
#QR Decomposition 
A = np.array([[3,1,2],[6,2,-2],[2,4,1]])
#use the information above to find QR Decomposition 
#also use equation from Linear Algebra Chapter 6
y1 = np.array([np.linalg.norm(v1), 0, 0])
y2 = ((x2 @ u1)/ (u1 @ u1)) * u1 + (np.linalg.norm(v2) * u2)
y3 = ((x3 @ u1) / (u1 @ u1)) * u1 + ((x3 @ u2) / (u2 @ u2)) * u2 + (np.linalg.norm(v3) * u3)
A = np.array([[3,1,2],[6,2,-2],[2,4,1]])
U = np.vstack((u1,u2,u3)).T
Q = U
R = np.vstack((y1,y2,y3)).T
print(Q,R)
#also numpy function will solve this
Q,R = np.linalg.qr(A)
#this code gives a negative value in the norm -> this cannot happen so change it 
Q = Q * np.sign(np.diag(R))
R = R * np.sign(np.diag(R))
print(Q,R)
print(A)
print(np.linalg.inv(A))
#Digonalization of Transformations in Chapter 5 of Linear Algebra 
#first see this left matrix transformation 
A = np.array([[2,1], [1,2]])
print(A)
#shows the change in basis 
plot_linear_transformation(A)
#plot a set of vectors in circle, their norms being 1 
alpha = np.linspace(0, 2*np.pi, 41)  #uniformly distributed angle value vector 
vectors = list(zip(np.cos(alpha), np.sin(alpha)))
vectors = np.array(vectors)
newvectors = A @ vectors.T
newvectors = newvectors.T
#vectors plotted in a normal space 
plot_vector(vectors)
#vectors plotted in a space whose basis has changed  
plot_vector(newvectors)
plt.show()
#one can see that the circle is turned into an ellipse 
#for ellipses the longest and shortest radius are important
#first get all the norm of the radius vectors
lengths = np.linalg.norm(newvectors, axis=1) 
semi_major_index = np.argmax(lengths)  #the longest radius vector 
semi_major_vector = newvectors[semi_major_index]
semi_major_length = lengths[semi_major_index]
print(semi_major_vector)
print(semi_major_length)
semi_minor_index = np.argmin(lengths)  #the shortest radius vector 
semi_minor_vector = newvectors[semi_minor_index]
semi_minor_length = lengths[semi_minor_index]
print(semi_minor_vector)
print(semi_minor_length)
#then one can check these vectors before the transformation 
A_inv = np.linalg.inv(A) #find the inverse of the left transformation 
v1 = A_inv @ semi_major_vector #the longest's inverse 
v2 = A_inv @ semi_minor_vector #the shortest's inverse 
print(v1)
print(v2)
#then one can see the directions of the vectors 
plot_vector(np.array([v1, v2]))
#they have identical values with opposite second element's signs
#this means they are 45 degrees apart
#also from Linear Algebra, one can see these vectors are eigenvectors
print(3 * v1, semi_major_vector)
print(1 * v2, semi_minor_vector)
plt.show()
#then one might intuitively think that scaling all x values three times and rotating the vector would result same thing
#scaling matrix
S = np.array([[3,0],[0,1]])
ellipse = S @ vectors.T
ellipse = ellipse.T 
plot_vector(ellipse)
plt.show()
#now try rotating all the vectors 45 degree
theta = np.pi/4
R = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])
rotated_ellipse = R @ ellipse.T
rotated_ellipse = rotated_ellipse.T
plot_vector(rotated_ellipse)
plt.show() #the figure looks very similar 
#but one can see this is not exactly the same with A transformation 
plot_linear_transformations(S,R)
#two transformations are not the same 
#from Theorem 2.16 and defintion of eigenvector, Ax = cx -> aAx = acx -> Aax = cax -> ax is also eigenvector
# -v1 is also an eigenvector 
v1 = -v1 #this will be better eigenvector since elements are positive 
print(v1)
scaled_v1 = S @ v1 
rotated_scaled_v1 = R @ scaled_v1
plot_vector([v1, rotated_scaled_v1])
print(rotated_scaled_v1) 
plt.show()
#To make exact replication two rotation and one scaling are needed 
#one can check that column vectors of R are eigenvectors and also orthogonal  
#then by the definition of orthogonal matrices R^-1 = R.T
theta_list = [np.pi/8, np.pi/4, np.pi/2, np.pi]

for theta in theta_list:
  R = np.array([[numpy.cos(theta), -numpy.sin(theta)],  # 반시계방향으로 45도 회전하는 matrix
                  [numpy.sin(theta), numpy.cos(theta)]])
  print(R)
  R_inv = np.linalg.inv(R)
  print(R_inv)
#checking inverses and transposes 
for theta in theta_list:
  R = np.array([[numpy.cos(theta), -numpy.sin(theta)],  # 반시계방향으로 45도 회전하는 matrix
                  [numpy.sin(theta), numpy.cos(theta)]])
  R_inv = np.linalg.inv(R)
  
  print(R.T)
  print(R_inv)
#now, recall Corollary 26
#rotating R  
plot_linear_transformation(R.T, v1, v2)

rotated_v1 = R.T @ v1
rotated_v2 = R.T @ v2
#scaling
scaled_rotated_v1 = S @ rotated_v1
scaled_rotated_v2 = S @ rotated_v2

plot_linear_transformation(S, rotated_v1, rotated_v2)
#rotating again
rotated_scaled_rotated_v1 = R@scaled_rotated_v1
rotated_scaled_rotated_v2 = R@scaled_rotated_v2

plot_linear_transformation(R, scaled_rotated_v1, scaled_rotated_v2)
print(rotated_scaled_rotated_v1, rotated_scaled_rotated_v2)
#the result is the same with A
plot_linear_transformation(R @ S @ R.T)
plt.show()
#proof of Theorem 6.17(symmetric when F = R) using numpy
F = np.random.rand(5,5)
F_symmetric = F + F.T 
F_symmetric
#use numpy to find eigenvalues and vectors 
eigenvalues, eigenvectors = np.linalg.eig(F_symmetric)
eigenvectors
from itertools import combinations
for i,j in combinations(range(5), 2):
  print(eigenvectors[:,i] @ eigenvectors[:,j])
#checking diagonalization one more time 
B = np.array([[1,0], [1,3]])
plot_eigen(B)
eigenvalues, eigenvectors = np.linalg.eig(B)
print(eigenvalues)
print(eigenvectors)
plt.show()
#Singular Value Decomposition 
#for the case above, Theorem 6.17 does not work, T is not symmetric 
#this means eigenvectors might not be orthogonal 
#one can check this 
plot_linear_transformation(B, eigenvectors_B[:,0], eigenvectors_B[:,1], unit_vector=False, unit_circle=True)
#this means that the longest and shortest vectors are not eigenvectors 
#creating a circle of vectors 
alpha = np.linspace(0, 2*np.pi, 201)  
circle = np.vstack((np.cos(alpha), np.sin(alpha)))
#apply B transformation 
ellipse = B @ circle
#collecting norms 
distance = np.linalg.norm(ellipse, axis=0)
#longest and shortest vectors 
semi_major_id = np.argmax(distance)
semi_minor_id = np.argmin(distance)
semi_major = ellipse[:, semi_major_id]  
semi_minor = ellipse[:, semi_minor_id]
print(semi_major, np.linalg.norm(semi_major))  
print(semi_minor, np.linalg.norm(semi_minor)) 
#reversing the transformation 
B_inv = np.linalg.inv(B)
semi_major_before = B_inv @ semi_major
semi_minor_before = B_inv @ semi_minor
plot_linear_transformation(B, semi_major_before, semi_minor_before, unit_vector=False, unit_circle=True)
plt.show()
#although they are not eigen, they are orthogonal 
print(semi_major_before @ semi_minor_before) #dot product is zero 
#also after transformation, the results are orthogonal 
print(semi_major @ semi_minor)
#they are orthogonal then by Theorem 
#then Theorem 6.25 can be applied to this transformation 
#finding positive scalars 
s1 = np.linalg.norm(semi_major)
s2 = np.linalg.norm(semi_minor)
S = np.diag([s1,s2])
print(S)
#finding u vectors 
u1 = semi_major / s1
u2 = semi_minor / s2
U = np.vstack((u1,u2)).T
print(U)
#original basis vectors 
v1 = semi_major_before
v2 = semi_minor_before
V = np.vstack((v1,v2)).T
VT = V.T
print(VT)
#comparing decomposition and original transformation 
U @ S @ V.T
#comparing Diagonalization and SVD 
#S and R in Diagonalization were scaling and rotation
#are U and S also those? 
#first of all, one must see orthogonal matrices gives reflection or rotations 
from scipy.stats import ortho_group

orthogonal_matrix = ortho_group.rvs(dim=2)
print('orthogonal matrix:\n', orthogonal_matrix)

#creating a random vector
x1 = np.random.randn(2)
x1 = x1 / np.linalg.norm(x1)
print('x1:',x1)

#multiplying an orthogonal matrix to the vector 
Qx1 = orthogonal_matrix@x1
print('Qx1:',Qx1)

#the next vector 
x2 = 3*np.random.randn(2)
x2 = x2 / np.linalg.norm(x2)
print('x2:',x2)

#Qx2
Qx2 = orthogonal_matrix@x2
print('Qx2:',Qx2)

#x3
x3= 3*np.random.randn(2)
x3 = x3 / np.linalg.norm(x3)
print('x3:',x3)

#Qx3
Qx3 = orthogonal_matrix@x3
print('Qx3:',Qx3)

#see the norm is preserved 
norm_random_vector_1 = np.linalg.norm(x1)  # x1's norm
norm_transformed_vector_1 = np.linalg.norm(Qx1)  # Qx1's norm
print(norm_random_vector_1)  
print(norm_transformed_vector_1)  

norm_random_vector_2 = np.linalg.norm(x2)  # x2's norm
norm_transformed_vector_2 = np.linalg.norm(Qx2)  # Qx2's norm
print(norm_random_vector_2)
print(norm_transformed_vector_2)

norm_random_vector_3 = np.linalg.norm(x3)  # x3's norm
norm_transformed_vector_3 = np.linalg.norm(Qx3)  # Qx3's norm
print(norm_random_vector_3)
print(norm_transformed_vector_3)
#one can check Q is rotation from vectors' dot product that represents relative angle relation 
plot_linear_transformation(orthogonal_matrix, 3*x1, 3*x2, 3*x3, unit_vector=False)
#checking dot products 
x1x2 = x1@x2
Qx1Qx2 = Qx1@Qx2
print(x1x2, Qx1Qx2)
x2x3 = x2@x3
Qx2Qx3 = Qx2@Qx3
print(x2x3, Qx2Qx3)
x3x1 = x3@x1
Qx3Qx1 = Qx3@Qx1
print(x3x1, Qx3Qx1)
#one can conclude that these matrices are rotations 
plot_linear_transformations(VT, S, U, unit_circle=True)
#SVD in higher dimension 
#three transformations: rotation, scaling and rotation that are equivalent to A
A = np.array([[1,2,3], [1,1,1], [-1,1,0]])
U, S, VT = np.linalg.svd(A)
plot_3d_linear_transformations(VT, np.diag(S), U, unit_sphere=True)
#this is an R^3 -> R^2 transformation
N = np.array([[1,2,7], [0,1,3], [-3,1,0]])
plot_3d_linear_transformation(N)
#this done as SVD
U, S, VT = numpy.linalg.svd(N)
print('S:',S)
plot_3d_linear_transformations(VT, numpy.diag(S), U, unit_sphere=True)
#outcome is R^3 -> R^2
#Like Theorem 6.25, R^n -> R^m transformation will omit n - m inputs 
#this has various applications 
#full rank case
A = np.array([[1,2,3], [1,1,1], [-1,1,0]])
U, S, VT = np.linalg.svd(A)
print(np.linalg.matrix_rank(A))
print(S)
#not a full rank
N = np.array([[1,2,7], [0,1,3], [-3,1,0]])
U, S, VT = np.linalg.svd(N)
print(np.linalg.matrix_rank(N))
print(S)
#when n x n is not full rank, omitting is possible just like stated above 
#compare
U @ np.diag(S) @ VT
U[:,:2] @ np.diag(S)[:2,:2] @ VT[:2,:]
#now although rank is full, if sigular value is close to zero, omitting can be tried 
A = numpy.array([[1, 2, 3, 6],
                 [2, 5, 7, 10],
                 [3, 9, 12, 14],
                 [4, 7, 9, 15]])
#stating rank 
print(numpy.linalg.matrix_rank(A))
#SVD done
U, S, VT = numpy.linalg.svd(A)
print(S)
#omitting the fourth rank 
rank_3_approximation = U[:,:3] @ np.diag(S[:3]) @ VT[:3,:]
print(A - rank_3_approximation)
#non square SVD, Theorem 6.25
#2x3 matrix
C = np.array([[1,0],[1,1],[2,3]])
U, S, VT = np.linalg.svd(C)
print('U:', U)
print('\nS:', S)
print('\nVT: ', VT)
#finding singular values 
#Cv1
Cv1 = C@VT[0]
# sigma1 * u1
sigma_u1 = 3.909 * U[:,0]
print(Cv1, sigma_u1)
#Cv2
Cv2 = C@VT[1]
#sigma2 * u2
sigma_u2 = 0.848 * U[:,1]
print(Cv2, sigma_u2)
#two vectors can be found just like n x n SVD 
#the last one must orthogonal, so QR decomposition is used to find the thrid column of SVD matrices 
#linearly independent vector from Q's left two vectors 
u3 = np.array([1,0,0])
# U[:,0], U[;,1], u3 matrix
new_U = np.hstack((U[:,:2], np.array([[1],[0],[0]])))
print(new_U)
# QR decomposition is used 
Q, R = np.linalg.qr(new_U)
print(Q)
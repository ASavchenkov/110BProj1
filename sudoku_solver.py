import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction 
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
import matplotlib.pylab as plt # for visualization

def fixed_constraints(N=9):
    rowC = np.zeros(N)
    rowC[0] =1
    rowR = np.zeros(N)
    rowR[0] =1
    row = scl.toeplitz(rowC, rowR)
    ROW = np.kron(row, np.kron(np.ones((1,N)), np.eye(N)))
    
    colR = np.kron(np.ones((1,N)), rowC)
    col  = scl.toeplitz(rowC, colR)
    COL  = np.kron(col, np.eye(N))
    
    M = int(np.sqrt(N))
    boxC = np.zeros(M)
    boxC[0]=1
    boxR = np.kron(np.ones((1, M)), boxC) 
    box = scl.toeplitz(boxC, boxR)
    box = np.kron(np.eye(M), box)
    BOX = np.kron(box, np.block([np.eye(N), np.eye(N) ,np.eye(N)]))
    
    cell = np.eye(N**2)
    CELL = np.kron(cell, np.ones((1,N)))
    
    return scs.csr_matrix(np.block([[ROW],[COL],[BOX],[CELL]]))


# For the constraint from clues, we extract the nonzeros from the quiz string.
def clue_constraint(input_quiz, N=9):
    m = np.reshape([int(c) for c in input_quiz], (N,N))
    r, c = np.where(m.T)
    v = np.array([m[c[d],r[d]] for d in range(len(r))])
    
    table = N * c + r
    table = np.block([[table],[v-1]])
    
    # it is faster to use lil_matrix when changing the sparse structure.
    CLUE = scs.lil_matrix((len(table.T), N**3))
    for i in range(len(table.T)):
        CLUE[i,table[0,i]*N + table[1,i]] = 1
    # change back to csr_matrix.
    CLUE = CLUE.tocsr() 
    
    return CLUE


from collections import defaultdict
def boxes():

    index = defaultdict(list)
    ind = []
    for i in range(9):
        for j in range(9):
            ind.append([i,j])
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    res5 = []
    res6 = []
    res7 = []
    res8 = []
    res9 = []
    for item in ind:
        i = item[0]
        j = item[1]
        if i%9>=0 and i%9<=2:
            if j%9>=0 and j%9<=2:
                res1.append(item)
        if i%9>=0 and i%9<=2:
            if j%9>=3 and j%9<=5:
                res2.append(item)
        if i%9>=0 and i%9<=2:
            if j%9>=6 and j%9<=8:
                res3.append(item)
        if i%9>=3 and i%9<=5:
            if j%9>=0 and j%9<=2:
                res4.append(item)

        if i%9>=3 and i%9<=5:
            if j%9>=3 and j%9<=5:
                res5.append(item)
        if i%9>=3 and i%9<=5:
            if j%9>=6 and j%9<=8:
                res6.append(item)
        if i%9>=6 and i%9<=8:
            if j%9>=0 and j%9<=2:
                res7.append(item)

        if i%9>=6 and i%9<=8:
            if j%9>=3 and j%9<=5:
                res8.append(item)
        if i%9>=6 and i%9<=8:
            if j%9>=6 and j%9<=8:
                res9.append(item)
    box = [res1, res2, res3, res4, res5, res6, res7, res8, res9]
    return boxfrom collections import defaultdict
def repeats(matrix, original):
    #print(matrix)
    temp = defaultdict(list)
    marked_matrix = np.ones((9,9))
    for i in range(9):
        for j in range(9):
            mark = False
            val = matrix[i][j]
            temp[val].append([i,j])
            #print(val)
            
            for l in range(9):
                if matrix[l][j] == val and i!= l:
                    mark = True

                    marked_matrix[l][j] = 0
                    temp[val]
        
    
                    #print(marked_matrix)
            for k in range(9):
                if matrix[i][k]== val and k!= j:
                    mark = True
                    marked_matrix[i][k] = 0
                    
            for res in boxes():
                if [i,j] in res:
                    for each_sq in res:
                        a = each_sq[0]
                        b = each_sq[1]
                        if matrix[a][b] == val and [a,b] != [i,j] :
                            mark = True
                            marked_matrix[a][b] = 0
                        
                    if mark == True:
                        marked_matrix[i][j] = 0
                    
    for i in range(9):
        for j in range(9):
            if marked_matrix[i][j] ==0:
                matrix[i][j] = 0
    for i in range(9):
        for j in range(9):
            if original[i][j] != 0 :
                matrix[i][j] = original[i][j]
                

    return matrix
            


def convert_stringtoarray(quiz):
    #can work for original and solution
    res = []
    count = 0
    for i in range(9):
        temp = []
        for j in range(9):
            temp.append(int(quiz[count]))
            count+=1
        res.append(temp)

    return res

def convert_matrixtolist(after_del):
    res = []
    for i in after_del:
        for j in i:
            res.append(j)
    
    return res

def solver(input_):
    
    quiz = input_
    constraint_ = input_
    iter_ = 0
    X_re = 0
    while(iter_<=5):
        
        A0 = fixed_constraints()
        A1 = clue_constraint(constraint_)
        # Formulate the matrix A and vector B (B is all ones).
        A = scs.vstack((A0,A1))
        A = A.toarray()
        B = np.ones(A.shape[0])
        # Because rank defficiency. We need to extract effective rank.
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        K = np.sum(s > 1e-12)
        S = np.block([np.diag(s[:K]), np.zeros((K, A.shape[0]-K))])
        A = S@vh
        B = u.T@B
        B = B[:K]
        c = np.block([ np.ones(A.shape[1]), np.ones(A.shape[1]) ])

        G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))],\
                             [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
        h = np.zeros(A.shape[1]*2)
        H = np.block([A, -A])
        b = B
        L = 25
        epsilon = 10**-10

        #x_new = x_ori   #? or below one?
        x_top = np.zeros(A.shape[1])
        x_bottom = np.zeros(A.shape[1])
        x_ori = x_top - x_bottom
        for j in range(L):
            Weight = 1/(abs(x_ori)+1)


            W = np.block([Weight,Weight])


            cW = np.matrix(c*W)

    
            ret = sco.linprog(cW, G, h, H, b, method='interior-point', options={'tol':1e-10})
            x_new = ret.x[:A.shape[1]] - ret.x[A.shape[1]:]


            #x_new = np.reshape(x, (81, 9))
            if LA.norm((x_new - x_ori)) <epsilon:
                break
            else:
                x_ori = x_new
    
        x_re = np.reshape(x_new, (81, 9))
        X_re = x_re
        u = np.array([np.argmax(d)+1 for d in x_re])
        after_del = repeats(convert_stringtoarray(u), convert_stringtoarray(quiz))  # starting x's    
        new_x_ori = np.array(convert_matrixtolist(after_del))
        constraint_ = new_x_ori
        iter_+=1
    ans =  np.array([np.argmax(d)+1 for d in X_re])
    return ''.join([str(c) for c in ans]) 
    



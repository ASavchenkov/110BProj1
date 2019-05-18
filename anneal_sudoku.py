from scipy.optimize import basinhopping
import numpy as np
import pandas as pd



def str_to_arr(sudokustr):
    return np.reshape([int(c) for c in sudokustr], (9,9))


#makes each block one row in a 9x9 array
def reshape_blocks(a):
    a = a.reshape((3,3,9))
    a = np.swapaxes(a,1,2) #now each chunk is transposed (3,9,3)
    a = a.reshape((3,3,3,3))
    a = a.reshape((9,9))
    return a


def single_loss(a):
    dupes = lambda x: 9-np.unique(x).shape[0]
    return np.sum(np.apply_along_axis(dupes,1,a))

#don't need gradients, just a function
#assumes that it comes in as a 9x9 array
    
def whole_loss(a):
    a = a.astype(int).reshape((9,9))
    rows = single_loss(a)
    cols = single_loss(a.T)
    blocks = single_loss(reshape_blocks(a))
    return rows + cols + blocks

#stepsize is the probability of an individual cell changing
class TakeStep(object):
    def __init__(self, quiz, stepsize = 0.1):
        print('initializing')
        self.quiz = quiz
        self.stepsize = stepsize

    def __call__(self,x):
        rand_mask = np.random.binomial(1,self.stepsize,(9,9))
        quiz_mask = np.where(self.quiz==0,1,0) #only apply to non taken spaces
        mask = rand_mask * quiz_mask

        n = np.sum(mask)
        np.random.randint(1,9,size=n)
        np.place(x,mask,np.random.randint(1,9,size=n))
        return x

    


if __name__ == '__main__':

    data = pd.read_csv('./small1.csv')

    sudoku0 = data['quizzes'][0]
    sol0 = data['solutions'][0]
    sudoku0 = str_to_arr(sudoku0)

    stepper = TakeStep(sudoku0)
    res = basinhopping(whole_loss, sudoku0, niter=200, T=0.5,take_step=stepper,stepsize=0.001)
    h = res.x.astype(int).reshape((9,9))
    print(sudoku0)
    print(h)
    print(whole_loss(h))

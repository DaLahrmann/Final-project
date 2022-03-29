# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:37:31 2022

@author: lahrm
"""

import pandas as pd
import numpy as np
from collections import Counter 
# found this library at https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item
#will be used to count values of a list fast 
import copy

def lmake(x):
    if x == '0':
        return [str(x) for x in range(1,10)]
    else:
        return []

def init(data):
    data['sol']=data.puz[data.puz != '0']
    data['pos']=data.puz.apply(lmake)
    return data

def solve(puz,level):
    data = pd.read_csv('sudoku_grid.csv')
    data['puz']=puz
    data=init(data)
    soln=True
    f=[possibles,hidden_single,naked_pairs]
    while soln and data.sol.isna().sum() !=0:
        for i in range(level):
            data, soln=f[i](data)
            if soln: 
                break #go out this loop if function updated data
        data=data.apply(pic_number,axis=1) #transform each pos-list of len 1 in a solution entry
    return data,soln

def pic_number(row):
    #transform each pos-list of len 1 in a solution entry
    if len(row.pos) == 1:
        row.sol=row.pos[0]
        row.pos=[]
    return row

 
def possibles(data):
    m=len(data.pos.sum())
    for i in data.index: #iterrows will be need to loop through rows
        if data.sol[i] != data.sol[i]:
            l=data.pos[i]
            l2=[y for y in l if y not in data[data.rw == data.rw[i]].sol.values]
            # no two identical numbers are allowed in a row
            l3=[y for y in l2 if y not in data[data.cl == data.cl[i]].sol.values]
            # no two identical numbers are allowed in a column
            data.pos[i]=[y for y in l3 if y not in data[data.box == data.box[i]].sol.values]
            # no two identical numbers are allowed in a box
    
    soln = len(data.pos.sum()) != m # a trigger which reffers if  algorithm excluded possibilities
    
    return data,soln

def hidden_single(data):
    location=['rw','cl','box']
    m=len(data.pos.sum())
    for i in range(9):
        for p in location: 
            #go through each box/column/row of the sudoku puzzle 
            subset=data[data[p] == i]
            l1=subset.pos.sum()
            
            co=Counter(l1)
            l2=[x for x in co.keys() if (co[x] == 1) and (x not in subset.sol.values)]
            # find not solved digits in row/column/box 
            # which have only one possible position  
            
            if len(l2) != 0:
                for n in l2:
                    entry=subset[subset.pos.apply(lambda x: n in x)]
                    data=data.apply(lambda x: sol_up(x,entry.ind.values[0],n),axis=1)
                    # put digits in solution
    soln = len(data.pos.sum()) != m # a trigger which reffers if  algorithm excluded possibilities
    return data,soln

def sol_up(row,ind,n):
    # help function of hidden single
    # will be used to replace the pos list by a pos list of len 1 with the right solution
    if row.ind == ind:
        row.pos = [n]
    return row

def naked_pairs(data):
    location=['rw','cl','box']
    m=len(data.pos.sum())
    for i in range(9):
        for p in location: 
            #go through each box/column/row of the sudoku puzzle 
            subset=data[data[p] == i]
            
            #find all pos-list of len 2 in p
            subset2=subset[subset.pos.map(len) == 2]
            for k in subset2.index:
                for j in subset2.index:
                    if (k < j) and (subset2.pos[k]==subset2.pos[j]):
                         l=subset2.pos[j]
                         data=data.apply(lambda row: remove_l(row,p,i,j,k,l),axis=1) 
                         #for data[p]==i remove all values from pos which are in l 
                         #exclude naked pair

    soln = len(data.pos.sum()) != m # a trigger which reffers if  algorithm excluded possibilities
    return data,soln

def  remove_l(row,p,i,j,k,l2):
    ''' 
    if cell is in the same location as the nakede pair (row[p]==i)
    and  not solved (row.sol!=row.sol)
    and not part of the naked pair itself (row.ind not in [j,k])
    remove all digits which are saved as the  digits of the naked pair (are in l2)
    '''
    if (row[p]==i) & (row.sol!=row.sol) & (row.ind not in [j,k]):
        row.pos=[x for x in row.pos if x not in l2]
    return row


def slow_nn_solver(puz,model):

    sol=copy.copy(puz) #create a copy of the puzzle which can be used for calculations
    n=1
    mask=np.array([x==0 for x in sol])

    while mask.sum()!=0: 
        n+=1
        out=model.predict(sol.reshape(1,81,1)) 
        # generate prediction
        # input must be reshaped so it can works
        pred=np.argmax(out, axis=2)+1  
        # prediction of the model
        prob=np.max(out, axis=2)
        # probability of the prediction
        prob*=mask
        #exclude solved digits
        ind=np.argmax(prob)
        #find digit with highest probability

        dig=pred[0][ind]
        sol[ind]=dig/9
        # include digit in updated puzzle
        mask=np.array([x==0 for x in sol])
        #create new mask
    return np.round(sol*9,0).astype(int)

def fast_nn_solver(puz,model):
    # this function use a nn-model directly to solve the puzzle

    out=model.predict(puz.reshape(1,81,1)) 
    sol=np.argmax(out, axis=2)+1 
    #take for each entry the value with the highest prediction and retransform the solution back 
    # where each cell is a value between 1 and 9 
    
    return sol[0]
    
        
def bingo_solve(puz):
    '''
    The init step of the recursive backtracking algorithm 
    input puz 
    '''
    dat = pd.read_csv('sudoku_grid.csv')
    dat['puz']=puz
    dat=init(dat)
    inds=dat[dat.sol.isna()].ind.values
    # inds is a list of all not solved cells
    solved, n=bingo(dat,inds,0)
    # bingo is recursive function which solve the puzzle
    print(n)
    #n count the numbers of tries to set a digit in a cell
    return dat, inds, solved

def bingo(dat,inds,n):
    entry = dat[dat.ind==inds[0]]
    #look at the first not solved digits
    r,c,b=entry.rw.values[0],entry.cl.values[0],entry.box.values[0]
    # r,c,b refers the row, column, box of the not solved digit
    posses=entry.pos.values[0]
    posses=[x for x in posses if x not in dat[dat.rw == r].sol.values]
    posses=[x for x in posses if x not in dat[dat.cl == c].sol.values]
    posses=[x for x in posses if x not in dat[dat.box == b].sol.values]
    #get a list of all possible digits of the distinct c

    for x in posses:
        #loop through all possibilities
        dat.sol.loc[inds[0]] = x
        # set digit
        n+=1
        if len(inds) == 1:
            # end function if the last digit is set
            return True, n
        solved, n = bingo(dat,inds[1:],n)
        # go recursively to the next empty digit
        if solved:
            # end function if sudoku is solved
            return  solved, n
    
    dat.sol.loc[inds[0]] = float('NaN')
    # if all possabilities does not end in a successful solution end function and go back to an 
    # earlier step
    return False, n 

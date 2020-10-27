import numpy as np
import pdb
import logging
import math
import argparse

""" 
 note: in ipython, use these commands:
 %load_ext autoreload
 %autoreload 2
"""

logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
ch.setFormatter(formatter)
logger.addHandler(ch)

class sudoku_puzzle:
    def __init__(self,xsize=3,ysize=3,difficulty=0):
      self.complete_board = sudoku_board(xsize,ysize)
      self.board = self.generate(self.complete_board,difficulty)
      #self.shape = self.board.data.shape

    def generate(self,board,difficulty=0):
      """ generate a puzzle from the board, based on difficulty. 
          Remove an element at a time, and ensure that it is solvable. """
      
      solvable = self.is_solvable(board) 
      if not solvable: raise ImpossibleGridError(board.data,"Impossible Initial Grid")

      #while solvable: 
      #  last_valid_board = np.copy(board)
      #  # tmp_board = self.remove_element(board)
      #  solvable = self.is_solvable(board) 
      #return last_valid_board


    def solve(self,board):
      """ Solve the given board """
      #tmp = sudoku_board(self.shape[0],self.shape[1])
      tmp.populate(board.data)
      return self.board

    def is_solvable(self,board):
      """ Return True if solvable, False if impossible to solve """  
       
      return True


class sudoku_board:
    
    def __init__(self,xsize=3,ysize=3,initial_data=None):
      self.xsize = xsize; self.ysize = ysize;
      self.size = xsize*ysize
      self.value_list = np.arange(self.size)+1
      self.data = None
      self.prev_data = None
      self.possible = None
      self.Npossible = None
      self.unknown_val = 0     # this seems restricted to be only 0.
      self.already_populated_val = self.size + 10
      self.zero_board() # initialize the board with zeros.

      while (self.data==self.unknown_val).sum()>0:
        try:
          self.populate(initial_data=initial_data)
        except ImpossibleGridError as err:
          logger.debug ("{}\n {}".format(err.message,err.board_data))
          pass

      self.rows = [sudoku_row(self,i) for i in range(self.size)]
      self.cols = [sudoku_col(self,i) for i in range(self.size)]
      self.squares = [sudoku_square(self,i) for i in range(self.size)]

    def __str__(self):
      return str(self.data)

    def zero_board(self):
      self.prev_data = np.copy(self.data)
      self.data = np.full([self.size,self.size],self.unknown_val,dtype=int)
      self.possible = np.empty([self.size,self.size,self.size],dtype=int)
      for level in range(self.size):
         self.possible[:,:,level]=level+1
      self.Npossible = (self.possible>0).sum(2)
      return 
       
    def same_square(self,row,col,i,j):
      """ is (row,col) in the same square as (i,j)?"""
      ysize = self.ysize; xsize = self.xsize; 
      return (math.floor(row/ysize)==math.floor(i/ysize)) and (math.floor(col/xsize)==math.floor(j/xsize))
 
    def update_possible(self,row,col):  # update possible,Npossible based on change to elem (row,col)
      value = self.data[row,col]
      if value == self.unknown_val: return self.possible # Unknown will not affect possibilities.

      for i in range(self.size):
        for j in range(self.size):
          if (i==row) and (j==col): 
            np.place(self.possible[i,j],self.possible[i,j]!=value,self.unknown_val)
          elif self.same_square(row,col,i,j):
            np.place(self.possible[i,j],self.possible[i,j]==value,self.unknown_val)
          elif (i==row) or (j==col):   # remove the possibility for other rows,cols
            np.place(self.possible[i,j],self.possible[i,j]==value,self.unknown_val)
      self.Npossible = (self.possible>self.unknown_val).sum(2)
      return self.possible

    def populate_from_initial(self,initial_data):
      self.zero_board()
      self.data = np.copy(initial_data)
      it = np.nditer(self.data,flags=['multi_index'])
      for elem in it: 
        if elem == self.unknown_val: self.possible[it.multi_index]=self.value_list
      it.reset()
      for elem in it: self.update_possible(it.multi_index[0],it.multi_index[1])
      return self.data

    def populate_cell(self,mask):
      Npossible = np.where(mask,self.Npossible,self.unknown_val)
      Npossible[self.data != self.unknown_val]=self.already_populated_val
      row,col = np.unravel_index(np.argmin(Npossible,axis=None),Npossible.shape)
      poss = self.possible[row,col]
      poss = np.unique(poss[poss != self.unknown_val])
      if(Npossible==self.unknown_val).sum() > 0: raise ImpossibleGridError(self.data,"Impossible Grid Failure")
      self.data[row,col] = np.random.choice(poss,1,False)
      self.update_possible(row,col)
      mask = self.data==self.unknown_val  # candidate spots, which have not yet been populated
      return mask
 
    def populate(self,initial_data=None):
      if not(initial_data is None): return self.populate_from_initial(initial_data)
      else: 
        self.zero_board()
        mask = self.data==self.unknown_val  # candidate spots, which have not yet been populated
        while mask.sum() > 0: mask = self.populate_cell(mask)
      return self.data

    def populate1(self,initial_data=None):
      self.zero_board()
      if not(initial_data is None): 
        self.data = np.copy(initial_data)
        it = np.nditer(self.data,flags=['multi_index'])
        for elem in it: 
          if elem == self.unknown_val: self.possible[it.multi_index]=self.value_list
        it.reset()
        pdb.set_trace()
        for elem in it: self.update_possible(it.multi_index[0],it.multi_index[1])
      else: 
        mask = self.data==self.unknown_val  # candidate spots, which have not yet been populated
        while mask.sum() > 0:
          Npossible = np.where(mask,self.Npossible,self.unknown_val)
          Npossible[self.data != self.unknown_val]=self.already_populated_val
          row,col = np.unravel_index(np.argmin(Npossible,axis=None),Npossible.shape)
          poss = self.possible[row,col]
          poss = np.unique(poss[poss != self.unknown_val])
          if(Npossible==self.unknown_val).sum() > 0:
            raise ImpossibleGridError(self.data,"Impossible Grid Failure")
          if initial_data == None: self.data[row,col] = np.random.choice(poss,1,False)
          self.update_possible(row,col)
          mask = self.data==self.unknown_val  # candidate spots, which have not yet been populated
      return self.data


    def display(self):
      print (self.data)
 
class sudoku_group:

    def __init__(self,board,i):
       self.board = board

    def is_valid(self):
       non_zeros = self.data[np.where(self.data!=self.board.unknown_val)[0]]
       return np.array_equal(np.unique(non_zeros),non_zeros)   

    def is_complete(self):
       size = np.size(self.data)
       return all([_ in self.board.value_list for _ in self.data.reshape(size,1)])

    def set_elem(self,i,val):
       if not val in np.concatenate((self.board.unknown_val,self.board.value_list)):
         raise ValueError
       if (val!=0) and (val in self.data):
         raise ValueError
       self.data[i] = val

    def reset_elem(self,i):
       self.board.prev_data = np.copy(self.board.data)
       self.data[i] = self.board.unknown_val

class sudoku_row(sudoku_group):
    def __init__(self,board,i):
       self.board = board
       self.data = board.data[i]
    
class sudoku_col(sudoku_group):
    def __init__(self,board,i):
       self.board = board
       self.data = board.data[:,i]

class sudoku_square(sudoku_group):
    def __init__(self,board,i):
       self.board = board;
       xsize = board.xsize; ysize = board.ysize;
       self.row_bound = ysize*math.floor(i/ysize)
       self.col_bound = xsize*(i%xsize)
       self.data = board.data[self.row_bound:self.row_bound+ysize,self.col_bound:self.col_bound+xsize]

    def set_elem(self,i,val):
       if not val in np.arange(10):
         raise ValueError
       if (val!=0) and (val in self.data):
         raise ValueError
       row = math.floor(i/self.board.ysize)
       col = (i%self.board.xsize)
       self.board.prev_data = np.copy(self.board.data)
       self.data[row,col] = val

class Error(Exception): 
   """ Base class for exceptions in this module."""
   pass

class ImpossibleGridError(Error):
   def __init__(self,board_data,message):
     self.board_data = board_data
     self.message = message 

def main(args):
  p = sudoku_puzzle(2,2)
  
  d = np.copy(p.complete_board.data) 
  d[0,1] = 0
  d[0,2] = 0
  d[1,0] = 0
  d[2,3] = 0
  d[3,1] = 0
  d[3,2] = 0

  b = sudoku_board(2,2)
  b.populate(d)
  print(b.Npossible)
  pdb.set_trace()

parser = argparse.ArgumentParser(description="Generate Sudoku Puzzles")
parser.add_argument('--size',type=int,nargs=2,dest='size',default=[2,3],help='x and y size of puzzle')
args = parser.parse_args()

if __name__ == "__main__":
  main(args)

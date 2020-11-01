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
      self.board = sudoku_board(xsize,ysize,self.complete_board.data)
      self.board_stack = []
      #self.generate(self.complete_board,difficulty)
      #self.shape = self.board.data.shape

    def generate(self,difficulty=0):
      """ generate a puzzle from the board, based on difficulty. 
          Remove an element at a time, and ensure that it is solvable. """

      solvable = True
      itry = 0
      while solvable and itry<5+difficulty:
        itry = itry + 1
        solvable = self.is_solvable(self.board) 
        self.board_stack.append(np.copy(self.board.data))
        populated_elements = np.argwhere(self.board.data!=self.board.unknown_val) 
        remove_elem = populated_elements[np.random.randint(0,len(populated_elements))]
        self.board.unset_cell(remove_elem[0],remove_elem[1])
         
      self.board_stack.pop()
      self.board.populate_from_initial(self.board_stack[-1])

      return self.board

    def solve(self,board):
      """ Solve the given board. Throw ImpossibleGridError if board is unsolvable """
      tmp = sudoku_board(board.xsize,board.ysize)
      tmp.populate(board.data,uniquely=True)
      return tmp.data

    def is_solvable(self,board):
      """ Return True if solvable, False if impossible to solve """  
      solvable = True
      mask = board.data==board.unknown_val  # spots which are unknown
      if mask.sum() == 0: return True
      min_possible = np.min(board.Npossible[mask])
      if min_possible == 0:
         solvable = False
      if min_possible > 1:
         solvable = False
      if min_possible == 1:
         try:
           self.solve(board)
           solvable = True
         except ImpossibleGridError as err:
           print (err.message)
           solvable = False
           pdb.set_trace()
         except:
           print ("Error Solving")
           pdb.set_trace()
      return solvable


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
      self.unset_board() # initialize the board with zeros.

      while (self.data==self.unknown_val).sum()>0:
        try:
          self.populate(initial_data=initial_data)
        except ImpossibleGridError as err:
          logger.debug ("{}\n {}".format(err.message,err.board_data))
          pass
        except:
          logger.warn("Error populating board")

    def __str__(self):
      return str(self.data)

    def unset_cell(self,row,col):
      initial_data = np.copy(self.data)
      initial_data[row,col] = self.unknown_val
      self.populate_from_initial(initial_data) 
      return self.data
   
    def unset_board(self):
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

    def populate_from_initial(self,initial_data=None):
      self.unset_board()
      if initial_data is None: return self.data
      self.data = np.copy(initial_data)
      it = np.nditer(self.data,flags=['multi_index'])
      for elem in it: 
        if elem == self.unknown_val: self.possible[it.multi_index]=self.value_list
      it.reset()
      for elem in it: self.update_possible(it.multi_index[0],it.multi_index[1])
      return self.data

    def populate_cell(self,mask,uniquely=False):
      Npossible = np.where(mask,self.Npossible,self.unknown_val)
      Npossible[self.data != self.unknown_val]=self.already_populated_val
      row,col = np.unravel_index(np.argmin(Npossible,axis=None),Npossible.shape)
      poss = self.possible[row,col]
      poss = np.unique(poss[poss != self.unknown_val])
      if(Npossible==self.unknown_val).sum() > 0: raise ImpossibleGridError(self.data,"Impossible Grid Failure")
      if(uniquely and self.Npossible[row,col]!=1): raise ImpossibleGridError(self.data,"No Unique Solution")
      self.data[row,col] = np.random.choice(poss,1,False)
      self.update_possible(row,col)
      mask = self.data==self.unknown_val  # candidate spots, which have not yet been populated
      return mask
 
    def populate(self,initial_data=None,uniquely=False):
      self.populate_from_initial(initial_data)
      if (initial_data is None) and uniquely: raise ImpossibleGridError(self.data,"Can't uniquely solve empty")
      mask = self.data==self.unknown_val  # candidate spots, which have not yet been populated
      while mask.sum() > 0: mask = self.populate_cell(mask,uniquely)
      return self.data

    def display_possible(self):
      result_string = ''
      for row in range(self.data.shape[0]):
        for col in range(self.data.shape[1]):
          poss = self.possible[row,col]
          known_b = self.data[row,col]!=self.unknown_val
          knowns = poss!=self.unknown_val
          if known_b:
            tmp = "("+str(self.data[row,col])+")"
          elif knowns.sum()==1:
            tmp = " " + str(poss[knowns][0])+" "
          elif knowns.sum()==0:
            tmp = "   "
          else:
            tmp = str(-1*knowns.sum()) + " "
          result_string = result_string + tmp
        result_string = result_string + "\n"
      print(result_string)


    def display(self):
      result_string = ''
      for row in range(self.data.shape[0]):
        tmp = ''.join([str(k)+" " for k in self.data[row]])
        tmp = tmp.replace(str(self.unknown_val)," ")
        result_string = result_string + tmp + "\n" 
      print (result_string)
      return None
 
class Error(Exception): 
   """ Base class for exceptions in this module."""
   pass

class ImpossibleGridError(Error):
   def __init__(self,board_data,message):
     self.board_data = board_data
     self.message = message 

def main(args):
  p = sudoku_puzzle(2,2)
  
  print(p.complete_board)
  print(p.board)

  populated_elements = np.argwhere(p.board.data!=p.board.unknown_val)  # candidate spots, which are populated
  #remove_elem = populated_elements[np.random.randint(0,len(populated_elements))]
  #d[tuple(remove_elem)] = 0

parser = argparse.ArgumentParser(description="Generate Sudoku Puzzles")
parser.add_argument('--size',type=int,nargs=2,dest='size',default=[2,3],help='x and y size of puzzle')
args = parser.parse_args()

if __name__ == "__main__":
  main(args)

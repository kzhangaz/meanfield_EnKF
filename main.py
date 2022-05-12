import torch
from src import test as winner
import multiprocessing as mp

def dummy1():
	return None


if __name__ == "__main__":
	print("Iason is a bad programmer")	
	with mp.Pool(3) as pool:
		pool.map(winner.KeIsTheWinner,[0]*3) 


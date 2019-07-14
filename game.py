import numpy as np

class Environment(object):
	"""docstring for Environment"""
	def __init__(self, arg):
		super(Environment, self).__init__()
		self.arg = arg
	
class RandomAgent(object):
		"""docstring for RandomAgent"""
		def __init__(self, arg):
			super(RandomAgent, self).__init__()
			self.arg = arg

def generateTiles():
	tiles = []
	for i in range(7):
		for j in range(i+1):
			tiles.append([i, j])
	print(tiles)
	print(len(tiles))
		
def play(env, agents, verbose=False):
	pass


generateTiles()
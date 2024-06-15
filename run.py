from random import random
from math import floor
import matplotlib.pyplot as plt


MAKE_PLOTS = False


def pick_random(A):
	return A[floor(random()*len(A))]


class Sarsa:
	def __init__(self, Av, eps=0.1, alp=0.5, gam=0.9):
		self.eps = eps
		self.alp = alp
		self.gam = gam
		self.Av = Av
		self.Qv = {} # state: {actions: value}
					# e.g. (R,R,R): {R:1, P:3, S:-1}

	def Q(self, s, a=None):
		if s not in self.Qv:
			self.Qv[s] = {a:0.0 for a in self.Av}
		return self.Qv[s][a] if a else self.Qv[s]

	def ra(self):
		"""Return a random action."""
		return self.Av[floor(random()*len(self.Av))]

	def learn(self, s, a, r, s_):
		"""Update Q to reflect the new reward"""
		self.Q(s)[a] += self.alp*(r + self.gam*self.Q(s,self.act(s_)) - self.Q(s,a))
		# print(self.Qv)

	def act(self, s):
		"""Return the next action under the policy"""
		if random() < self.eps:
			a_ = self.ra()
		else:
			max_value = max(self.Q(s).values())
			best_actions = [a for a,v in self.Q(s).items() if v == max_value]
			a_ = pick_random(best_actions)
		return a_


class RandomStrategy:
	name = 'Random Strategy'

	def act(self, *args, **kwargs):
		return pick_random(('R','P','S'))


class AlmostRandomStrategy:
	name = 'Almost Random Strategy'

	def act(self, *args, **kwargs):
		rn = random()
		if rn <= 0.33:		# [0, 0.33]		34%
			a_ = 'R'
		elif rn <= 0.66:	# [0.34, 0.66]	33%
			a_ = 'P'
		else:				# [0.67, 0.99]	33%
			a_ = 'S'
		return a_


class CustomRandomStrategy:
	def __init__(self, intervals):
		self.intervals = intervals

	def act(self, *args, **kwargs):
		rn = random()
		if rn <= self.intervals[0]:
			a_ = 'R'
		elif rn <= self.intervals[1]:
			a_ = 'P'
		elif rn <= self.intervals[2]:
			a_ = 'S'
		else:
			a_ = 'R'
		return a_


class MostlyRandomStrategy(CustomRandomStrategy):
	name = 'Mostly Random Strategy'

	def __init__(self):
		self.intervals = (0.3,0.6,1)


class NumberphileStrategy:
	name = 'Numberphile Strategy'

	def act(self, *args, **kwargs):
		if _r > 0:
			all_actions = ('R','P','S')
			a_ = [a for a in all_actions if a != _ba and a != _pa][0]
		elif _r < 0:
			a_ = _ba
		else:
			a_ = pick_random(('R','P','S','R'))
		return a_


class StrangeHumanStrategy:
	name = 'Strange Human Strategy'

	def act(self, *args, **kwargs):
		if _ba == 'R':
			a_ = 'P'
		elif _ba == 'P':
			a_ = 'S'
		else:
			a_ = 'R'
		return a_


class AlwaysRockStrategy:
	name = 'Always Rock Strategy'

	def act(self, *args, **kwargs):
		return 'R'


class Game:
	def resolve(self, ba, pa):
		s_ = pa
		match (ba,pa):
			case ('R','R')|('P','P')|('S','S'):
				r_ = 0
			case ('R','P')|('P','S')|('S','R'):
				r_ = -1
			case ('P','R')|('S','P')|('R','S'):
				r_ = 1
		return s_, r_


if __name__ == '__main__':
	env = Game()
	sar = Sarsa(['R','P','S'])
	strats = (StrangeHumanStrategy(),
			RandomStrategy(), 
		   AlmostRandomStrategy(), 
		   MostlyRandomStrategy(), 
		   NumberphileStrategy(),
		   AlwaysRockStrategy())

	for strat in strats:
		print(f'Running: {strat.name}')
		totals = []

		for _ in range(10):
			_ba = 'R'
			_pa = 'R'
			_r = 0
			total = 0

			for _ in range(100000):
				ba = sar.act(_pa)
				pa = strat.act(_ba=_ba, _pa=_pa, _r=_r)
				s_, r_ = env.resolve(ba, pa)
				total += r_
				sar.learn(_pa, ba, r_, s_)
				_ba = ba
				_pa = pa
				_r = r_

			totals.append(total)


		ys = totals
		xs = [i for i in range(len(ys))]
		mean = sum(ys)/len(ys)

		if MAKE_PLOTS:
			plt.plot(xs, ys)
			plt.axhline(mean)
			plt.title(strat.name)
			plt.xlabel('Game')
			plt.ylabel('Score')
			plt.show()

		print(f'Totals: {totals}, Mean: {mean}')

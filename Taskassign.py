from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import itertools

verbose = True
save = False
# range (xrange, yrange), N is number of tasks points
def get_random_tasks(search_range, N):
  x = np.random.random(N)*search_range[0]
  y = np.random.random(N)*search_range[1]
  xy = np.vstack((x,y))
  # if verbose:
  #   print "Tasks points: ", xy
  if save:
    name = 'task_points-'+str(N)+'.txt'
    np.savetxt(name,xy)
  return xy

def get_random_robots(search_range, N):
  x = np.random.random(N)*search_range[0]
  y = np.random.random(N)*search_range[1]
  xy = np.vstack((x,y))
  if verbose:
    print "Robots points: ", xy
  if save:
    name = 'task_points-'+ str(N) + '.txt'
    np.savetxt(name,xy)
  return xy

def distance(robots, tasks):
  d = robots - tasks  
  return np.sqrt((d**2).sum(axis=1)).sum()

def exhaustive_all(robots, tasks):
  tasks_copy = np.copy(tasks)
  assignments = range(robots.shape[1])
  best_dist = distance(robots,tasks)
  for assign in itertools.permutations(assignments):
    assign = np.array(list(assign))
    # print assign
    tasks_copy[0] = tasks_copy[1][assign]
    tasks_copy[1] = tasks_copy[1][assign]
    dist = distance(robots,tasks_copy)
    if dist < best_dist:
      best_dist = dist
      best_assign = assign

  return best_dist, best_assign

def task_assignment_solver(num_seeds,max_NFE,tasks,robots, best_dist):
  # ft = np.zeros((num_seeds, max_NFE))
  ft = []
  tasks_copy = np.copy(tasks)
  num_tasks = tasks_copy.shape[1]
  assignment = np.arange(num_tasks)
  assignt = np.zeros((num_seeds,num_tasks)).astype(int)  
  max_NFE *= num_tasks/50
  for seed in np.arange(num_seeds):
    np.random.seed(seed)
    # random initial tour
    assignment = np.random.permutation(assignment)
    bestf = distance(robots, tasks_copy)
    i = 1
    # for i in np.arange(max_NFE):
    while i <= max_NFE and bestf > best_dist:
      # mutate the tour using two random cities
      trial_assignment = np.copy(assignment) # do not operate on original list
      a,b = np.random.random_integers(num_tasks,size=(2,))
      # the entire route between two cities are reversed
      if a > b:
        a,b = b,a
        trial_assignment[a:b+1] = trial_assignment[a:b+1][::-1]
      tasks_copy[0] = tasks_copy[1][assignment]
      tasks_copy[1] = tasks_copy[1][assignment]
      trial_f = distance(robots, tasks_copy)
      if trial_f < bestf:
        assignment = trial_assignment
        bestf = trial_f
      # ft[seed,i] = bestf
      ft.append(bestf)
      i += 1
      assignt[seed] = assignment
    print assignment
    print bestf
  ft = np.array(ft)
  best_ind = np.unravel_index(np.argmin(np.nonzero(ft)), ft.shape)
  bestf = ft[best_ind]
  bestassign = assignt[best_ind[0]]
  return bestassign,bestf,assignt,ft # best result of all seeds

def assignment_show(robots,tasks,assignt,ft):
  plt.subplot(1,2,1)
  plt.scatter(robots[0,:], robots[1,:], marker='o', color="r",alpha=0.8, s=50)
  plt.scatter(tasks[0,:],  tasks[1,:],  marker='*', color="b",alpha=0.8, s=50)
  for i, assign in enumerate(assignt):
    plt.plot([robots[0][i],tasks[0][assign]],[robots[1][i],tasks[1][assign]], color='y', alpha=0.8)
  # plt.plot(robots[:,0], robots[:,1])
  # plt.plot(tasks[:,0],tasks[:,1], marker='*')
  plt.subplot(1,2,2)
  plt.semilogx(ft.T, color='steelblue', linewidth=1)
  plt.xlabel('Iterations')
  plt.ylabel('Length of Tour')
  plt.show()

# NFE to different cities
num = 8 # tasks equal to num of robots
search_range = [20,20]
num_seeds = 10
max_NFE = 100000

robots = get_random_robots(search_range,num)
tasks  =  get_random_tasks(search_range,num)

best_dist, best_assign_theory = exhaustive_all(robots, tasks)
print "The theory best value is", best_dist
print "The best assginment is", best_assign_theory

bestassign,bestf,assignt,ft = task_assignment_solver(num_seeds,max_NFE,tasks,robots,best_dist)

assignment_show(robots,tasks,bestassign,ft)
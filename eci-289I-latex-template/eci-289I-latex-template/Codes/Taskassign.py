from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import itertools

verbose = False
save = False
exhaustive = True
# range (xrange, yrange), N is number of tasks points
def get_random_tasks(search_range, N):
  x = np.random.random(N)*search_range[0]
  y = np.random.random(N)*search_range[1]
  xy = np.vstack((x,y))
  if verbose:
    print "Tasks points: ", xy
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
  return (np.sqrt((d**2).sum(axis=0))).sum()

def exhaustive_all(robots, tasks):
  # print tasks
  tasks_copy = np.copy(tasks)
  assignments = range(robots.shape[1])
  best_dist = distance(robots,tasks_copy)
  for assign in itertools.permutations(assignments):
    assign = np.array(list(assign))
    tasks_copy = tasks_copy[:,assign]
    # tasks_copy[0] = tasks_copy[1][assign]
    # tasks_copy[1] = tasks_copy[1][assign]
    dist = distance(robots,tasks_copy)
    if dist <= best_dist:
      best_dist = dist
      best_assign = assign
      # print best_dist
    tasks_copy = np.copy(tasks) # change to original one

  print "Best", best_dist
  print "Best task points", tasks_copy[:,best_assign]
  return best_dist

# def task_assignment_solver(num_seeds,max_NFE,tasks,robots, best_dist):
def task_assignment_solver(num_seeds,max_NFE,tasks,robots,exhaustive):
  if exhaustive: 
    best_dist = exhaustive_all(robots,tasks)
  else:
    best_dist = 0
  # ft = []
  num_tasks = tasks.shape[1]
  ass = np.arange(num_tasks)
  assignt = np.zeros((num_seeds,num_tasks)).astype(int)
  # assignt = dict()
  # assignt = {k: [] for k in range(num_seeds)}  
  max_NFE *= num_tasks
  ft = np.zeros((num_seeds, max_NFE))
  T0 = 2 # initial temperature
  alpha = 0.95 # cooling parameter
  for seed in np.arange(num_seeds):
    np.random.seed(seed)
    # random initial tour
    best_assign = np.random.permutation(ass)
    # assignment = ass
    initial_order = best_assign
    tasks_copy = np.copy(tasks)
    tasks_copy = tasks_copy[:,best_assign]
    bestf = distance(robots,tasks_copy)
    i = 0   
    ft[seed,i] = bestf
    # assignt[seed].append(assignment)
    T=T0
    # for i in np.arange(max_NFE):
    while i < max_NFE-1 and bestf > best_dist*1.005:
  # mutate the tour using two random cities
      trial_assignment = np.copy(best_assign) # do not operate on original list
      # assignt = assignt[seed].append(trial_assignment)
      # a,b = np.random.random_integers(num_tasks,size=(2,))
      a = np.random.randint(num_tasks)
      b = np.random.randint(num_tasks)

    # the entire route between two cities are reversed
      if a > b:
        a,b = b,a
      trial_assignment[a:b+1] = trial_assignment[a:b+1][::-1]
    # if np.abs(bestf - best_dist) > 0.05:
    #   trial_assignment[a:b+1] = trial_assignment[a:b+1][::-1]

      tasks_copy = tasks_copy[:,trial_assignment]
      trial_f = distance(robots, tasks_copy)

      r = np.random.rand()
      if T > 10**-3: # protect division by zero
        P = np.min([1.0, np.exp((bestf - trial_f)/T)])
      # print P
      else:
        P = 0

      if trial_f <= bestf or r < P:
        best_assign = trial_assignment
        best_tasks = np.copy(tasks_copy)
        bestf = trial_f

      ft[seed,i] = bestf
      # print assignment
      # print bestf
      # ft[seed].append(bestf)
      i += 1
    # assignt[seed] = assignment
      T = T0*alpha**i
      # print a, b, seed, "seed", i, "trail", "r",r, "P", P, assignment, bestf

    assignt[seed] = best_assign
    # best_ind = np.unravel_index(np.argmin(ft[seed]), ft.shape)
    # assignment = assignt[best_ind]
    print "seed", seed, "best value", bestf
    # print "best tasks points list ", best_tasks
  # ft = np.array(ft)
  ft[ft==0] = np.inf
  # best_ind = np.unravel_index(np.argmin(np.nonzero(ft)), ft.shape)
  best_ind = np.unravel_index(np.argmin(ft), ft.shape)
  bestf = ft[best_ind]
  print best_ind
  bestassign = assignt[best_ind[0]]
  return best_tasks, bestf, ft, best_dist # best result of all seeds

def assignment_show(robots,tasks,ft):
  plt.subplot(1,2,1)
  plt.scatter(robots[0,:], robots[1,:], marker='o', color="r",alpha=0.8, s=50)
  plt.scatter(tasks[0,:],  tasks[1,:],  marker='*', color="b",alpha=0.8, s=50)
  # assignt = range(tasks.shape[1])
  for i, assign in enumerate(range(tasks.shape[1])):
    plt.plot([robots[0][i],tasks[0][assign]],[robots[1][i],tasks[1][assign]], color='y', alpha=0.8)
  # plt.plot(robots[:,0], robots[:,1])
  # plt.plot(tasks[:,0],tasks[:,1], marker='*')
  plt.subplot(1,2,2)
  # plt.semilogx(ft.T, color='steelblue', linewidth=1)
  plt.semilogx(ft.T, color='steelblue', linewidth=1)
  plt.xlabel('Iterations')
  plt.ylabel('Length of Tour')
  plt.show()

# NFE to different cities
num = 10 # tasks equal to num of robots
search_range = [20,20]
num_seeds = 10
max_NFE = 5000

np.random.seed(1)
robots =  get_random_robots(search_range,num)
tasks  =  get_random_tasks(search_range,num)

# best_dist, best_assign_theory = exhaustive_all(robots, tasks)

if exhaustive:
  best_tasks, bestf, ft = task_assignment_solver(num_seeds,max_NFE,tasks,robots,exhaustive)
else:
  best_tasks, bestf, ft, best_dist = task_assignment_solver(num_seeds,max_NFE,tasks,robots,exhaustive)

print "The best value found is", bestf
# print "The best assginment found is", bestassign
if exhaustive:
  print "The theory best value is", best_dist
# print "The theory best assginment is", best_assign_theory
# tasks_show = np.copy(tasks)[:,initial_order]
# tasks_show = tasks_show[:,initial_order]
assignment_show(robots,best_tasks,ft)

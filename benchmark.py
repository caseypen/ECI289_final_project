from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
import itertools

verbose = False
save = False
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
  # print "Best task points", tasks_copy[:,best_assign]
  return best_dist

N = [3,5,7,9,11,13,15,20,25,30]
best_f = np.zeros(len(N))
search_range = [1,1]
np.random.seed(1)
for i,num in enumerate(N):
  robots = get_random_robots(search_range,num)
  tasks  = get_random_tasks(search_range,num)
  best_f[i] = exhaustive_all(robots,tasks)

print best_f

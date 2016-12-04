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

# def task_assignment_solver(num_seeds,max_NFE,tasks,robots, best_dist):
def task_assignment_solver(num_seeds,max_NFE,N,search_range):
  NFEt = np.zeros((len(N),num_seeds)).astype(int) # NFE for differnt seeds and different NO. of cities  
  T0 = 2 # initial temperature
  alpha = 0.95 # cooling parameter
  for j,num in enumerate(N):
    L_upper = (np.sqrt(num)+np.log(num))*0.92
    for seed in np.arange(num_seeds):
      np.random.seed(seed)
      tasks = get_random_tasks(search_range,num)
      robots = get_random_robots(search_range,num)      
      best_assign = range(N[j])
      bestf = distance(robots,tasks)
      i = 0   
      T = T0
      # for i in np.arange(max_NFE):
      while i < max_NFE-1 and bestf > L_upper:
    # mutate the tour using two random cities
        trial_assignment = np.copy(best_assign) # do not operate on original list
        a = np.random.randint(num)
        b = np.random.randint(num)
      # the entire route between two cities are reversed
        if a > b:
          a,b = b,a
        trial_assignment[a:b+1] = trial_assignment[a:b+1][::-1]
        tasks = tasks[:,trial_assignment]
        trial_f = distance(robots, tasks)
        r = np.random.rand()
        if T > 10**-3: # protect division by zero
          P = np.min([1.0, np.exp((bestf - trial_f)/T)])
        # print P
        else:
          P = 0
        if trial_f <= bestf or r < P:
          best_assign = trial_assignment
          bestf = trial_f
        i += 1
        T = T0*alpha**i
      print "seed", seed, N[j], "NFE", i
      print bestf, L_upper*1.2
      NFEt[j][seed] = i    
    name = "NFEt-"+"result"+".txt"
    np.savetxt(name,NFEt)
  return NFEt, name # best result of all seeds

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
# num = 8 # tasks equal to num of robots
N = [5,15,20,25,30,45]
search_range = [1,1]
num_seeds = 10
max_NFE = 25000

# np.random.seed(1)
# robots =  get_random_robots(search_range,num)
# tasks  =  get_random_tasks(search_range,num)

# best_dist, best_assign_theory = exhaustive_all(robots, tasks)

NFEt, name = task_assignment_solver(num_seeds,max_NFE,N,search_range)


# print "The theory best assginment is", best_assign_theory
# tasks_show = np.copy(tasks)[:,initial_order]
# tasks_show = tasks_show[:,initial_order]
# assignment_show(robots,best_tasks,ft)

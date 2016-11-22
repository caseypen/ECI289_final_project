from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
# from itertools import groupby


verbose = True
save = False
# range (xrange, yrange), N is number of tasks points
def get_random_tasks(range,N):
  x = np.random.random(N)*range[0]
  y = np.random.random(N)*range[1]
  xy = np.vstack((x,y))
  if verbose:
    print "Tasks points: ", xy
  if save:
    name = 'task_points-'+str(N)+'.txt'
    np.savetxt(name,xy)
  return xy

def get_random_robots(range,N):
  x = np.random.random(N)*range[0]
  y = np.random.random(N)*range[1]
  xy = np.vstack((x,y))
  if verbose:
    print "Robots points: ", xy
  if save:
    name = 'task_points-'+str(N)+'.txt'
    np.savetxt(name,xy)
  return xy
# do the local search:
def raw_allocation(robots,tasks):
  num_robots = robots.shape[1]
  num_tasks = tasks.shape[1]
  dist_c = np.zeros((num_robots,num_tasks))
  for i in np.arange(num_robots):
    for j in np.arange(num_tasks):
      dist_c[i,j] = np.sqrt(((robots[:,i]-tasks[:,j])**2).sum())
  if verbose:
    print dist_c.shape
  robot_chosen = np.argmin(dist_c,axis=0)
  if verbose:
    print "Each task choose robot:", robot_chosen
  # [len(list(group)) for key, group in groupby(a)]
  robots_tasks = dict()
  robot_tasks = robots_tasks.fromkeys(range(num_robots),[])
  for i , robot in enumerate(robot_chosen):
    for j in np.arange(num_robots):
      if robot == j:
        robots_tasks[j].append(i) # each elements means which number of tasks is chosen
  if verbose:
    print "robot choose tasks:", robots_tasks
  return robots_tasks
def distance(robots, tasks):
  d = robots - tasks  
  return np.sqrt((d**2).sum(axis=1)).sum()

def task_assignment_solver(num_seeds,max_NFE,tasks,robots):
  ft = np.zeros((num_seeds, max_NFE))
  num_tasks = tasks.shape[1]
  assignment = np.arange(num_tasks)
  assignt = np.zeros((num_seeds,num_tasks)).astype(int)  
  for seed in np.arange(num_seeds):
    np.random.seed(seed)
    # random initial tour
    assignment = np.random.permutation(assignment)
    bestf = distance(robots, tasks)
    for i in np.arange(max_NFE):
      # mutate the tour using two random cities
      trial_assignment = np.copy(assignment) # do not operate on original list
      a,b = np.random.random_integers(num_tasks,size=(2,))
      # the entire route between two cities are reversed
      if a > b:
        a,b = b,a
        trial_assignment[a:b+1] = trial_assignment[a:b+1][::-1]
      tasks[0] = tasks[1][assignment]
      tasks[1] = tasks[1][assignment]
      trial_f = distance(robots, tasks)
      if trial_f < bestf:
        assignment = trial_assignment
        bestf = trial_f
      ft[seed,i] = bestf
      assignt[seed] = assignment
    print assignment
    print bestf
  best_ind = np.unravel_index(ft.argmin(), ft.shape)
  bestf = ft[best_ind]
  bestassign = assignt[best_ind[0]]
  return bestassign,bestf,assignt,ft # best result of all seeds

def assignment_show(robots,tasks,assignt,ft):
  plt.subplot(1,2,1)
  # tour = np.append(tour, tour[0]) # for plotting
  # tasks = tasks[assignt]
  plt.scatter(robots[0,:], robots[1,:], marker='o', color="r",alpha=0.3)
  plt.scatter(tasks[1,:],tasks[1,:], marker='*',color="b",alpha=0.5)
  for i, assign in enumerate(assignt):
    plt.plot([robots[0][i],tasks[0][assign]],[robots[1][i],tasks[1][assign]])
  # plt.plot(robots[:,0], robots[:,1])
  # plt.plot(tasks[:,0],tasks[:,1], marker='*')
  plt.subplot(1,2,2)
  plt.semilogx(ft.T, color='steelblue', linewidth=1)
  plt.xlabel('Iterations')
  plt.ylabel('Length of Tour')
  plt.show()


# # def TSP_NFE_Show(savename,num_seeds,N):
# #   NFEt= np.loadtxt(savename)
# # # B = np.loadtxt('bfgs-scaling-results.txt')
  
# #   x = np.tile(N,(10,1)).reshape(50,)
# #   y = NFEt.reshape(50,)

# #   slope,intercept,rvalue,pvalue,stderr = stats.linregress(x,y)
# #   plt.subplot(1,2,1)
# #   plt.scatter(x,y)
# #   # print slope
# #   plt.plot([np.min(x), np.max(x)], [intercept + np.min(x)*slope, intercept + np.max(x)*slope], 
# #             color='indianred', linewidth=2)
# #   # plt.text(250,120000, 'slope = %0.2f' % slope, fontsize=16)
# #   plt.xlabel('# Decision Variables')
# #   plt.ylabel('NFE to converge')
# #   plt.title('TSP Problem (slope:' + str(np.round(slope,2))+')')


#   x_log = np.log10(x)
#   y_log = np.log10(y)
#   slope,intercept,rvalue,pvalue,stderr = stats.linregress(x_log,y_log)
#   plt.subplot(1,2,2)
#   plt.scatter(x_log,y_log)
#   plt.plot([np.min(x_log), np.max(x_log)], [intercept + np.min(x_log)*slope, intercept + np.max(x_log)*slope], 
#             color='indianred', linewidth=2)
#   # plt.text(250,120000, 'slope = %0.2f' % slope, fontsize=16)
#   plt.xlabel('log(# Decision Variables)')
#   plt.ylabel('log(NFE to converge)')
#   plt.title('TSP Loglog (slope'+ str(np.round(slope,2))+', intercept'+str(np.round(intercept,2))+')')

#   plt.show()

# for large number of evaluation
"""
num_seeds = 10
max_NFE = 10000
# num_cities = 47
# L_conv = np.round(np.sqrt(2*num_cities))
# xy = get_random_cities(num_cities)
xy = np.loadtxt('tsp-48.txt')
tour, bestf, ft = TSP_solver_max(num_seeds,max_NFE,xy)
print tour 
print bestf
TSP_show(xy,tour,ft)

"""
# NFE to different cities
num_robots = 10 # tasks equal to num of robots
num_tasks = 15
range = [10,10]
num_seeds = 10
max_NFE = 10000
robots = get_random_robots(range,num_robots)
tasks = get_random_tasks(range,num_tasks)

robots_tasks = raw_allocation(robots,tasks)

# bestassign,bestf,assignt,ft = task_assignment_solver(num_seeds,max_NFE,tasks,robots)
# print "Best assignment:", bestassign

# assignment_show(robots,tasks,bestassign,ft)
from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats

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

# def get_random_cities(N):
#   xy = np.random.random_sample((N,2))
#   name = 'tsp-'+str(N)+'.txt'
#   np.savetxt(name,xy)
#   return xy

# # def TSP_NFE_CITIES(num_seeds, N):
# #   # ft = np.zeros((num_seeds, max_NFE))
# #   NFEt = np.zeros((num_seeds,len(N))).astype(int) # NFE for differnt seeds and different NO. of cities  
# #   for seed in range(num_seeds):
# #     np.random.seed(seed)
# #     # random initial tour
# #     NFE = 0
# #     for i,num in enumerate(N):
# #       L_conv = np.sqrt(1*num)
# #       xy = get_random_cities(num)
# #       num_cities = len(xy)
# #       tour = np.arange(num_cities)
# #       tour = np.random.permutation(tour)
# #       bestf = 99999
# #       while bestf > L_conv:
# #         # mutate the tour using two random cities
# #         trial_tour = np.copy(tour) # do not operate on original list
# #         a = np.random.randint(num_cities)
# #         b = np.random.randint(num_cities)
# #         # the entire route between two cities are reversed
# #         if a > b:
# #           a,b = b,a
# #           trial_tour[a:b+1] = trial_tour[a:b+1][::-1]

# #         trial_f = distance(trial_tour, xy)
# #         NFE += 1
# #         r = np.random.rand()
# #         if trial_f < bestf:
# #           tour = trial_tour
# #           bestf = trial_f
# #       # print bestf
# #       NFEt[seed][i] = NFE
# #       print str(seed*len(N)+i+1)+"/"+str(num_seeds*len(N))
# #   name = "NFEt-"+str(len(N))+"result"+".txt"
# #   np.savetxt(name,NFEt)
# #   return NFEt,name # best result of all seeds

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
num = 100 # tasks equal to num of robots
range = [10,10]
num_seeds = 10
max_NFE = 10000
robots = get_random_robots(range,num)
tasks = get_random_tasks(range,num)

bestassign,bestf,assignt,ft = task_assignment_solver(num_seeds,max_NFE,tasks,robots)
print "Best assignment:", bestassign

assignment_show(robots,tasks,bestassign,ft)
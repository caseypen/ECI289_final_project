import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats

verbose = False
save = False

''' ################################################## 
  Generate task points for a certain area
  range:(xrange, yrange), N is number of tasks points
'''###################################################
def get_random_tasks(search_range,N):
  x = np.random.random(N)*search_range[0]
  y = np.random.random(N)*search_range[1]
  xy = np.vstack((x,y))
  if verbose:
    print "Tasks points: ", xy
  if save:
    name = 'task_points-'+str(N)+'.txt'
    np.savetxt(name,xy)
  return xy
''' ################################################## 
      Distribute the robots in the task range
'''###################################################
def lhs(N,D):
  grid = np.linspace(0,1,N+1)
  result = np.random.uniform(low=grid[:-1], high=grid[1:], size=(D,N))
  for c in result:
    np.random.shuffle(c)
  return result

def assign_robots(search_range,N):
  x = lhs(N,1)*search_range[0]
  y = lhs(N,1)*search_range[1]
  xy = np.vstack((x,y))
  if verbose:
    print "Robots points: ", xy
  if save:
    name = 'task_points-'+str(N)+'.txt'
    np.savetxt(name,xy)
  return xy

''' ################################################## 
      Local search: (O(n*logn*m)+O(n**2))
      Tasks asks help for local robots
      It can only calls the robots nearsets to them
'''###################################################
def raw_allocation(robots,tasks,search_range):
  num_robots = robots.shape[1]
  num_tasks  = tasks.shape[1]
  dist_c = np.zeros((num_robots,num_tasks)) # distance matrix

  for i in range(num_robots):
    for j in range(num_tasks):
      dist_c[i,j] = np.sqrt(((robots[:,i]-tasks[:,j])**2).sum())

  if verbose:
    print dist_c.shape
  # Each task only need one nearest robot to help
  robot_chosen = np.argmin(dist_c,axis=0)

  if verbose:
    print "Each task choose robot:", robot_chosen

  # Task assignment of robots
  robots_tasks = dict()
  robots_tasks = {k: [] for k in range(num_robots)}
  # robots_tasks = robots_tasks.fromkeys(robot_list,[])
  for i,robot in enumerate(robot_chosen):
    robots_tasks[robot].append(i) # each elements means which number of tasks is chosen

  if verbose:
    print "robot choose tasks:", robots_tasks
  allocation_show(robots,tasks,robots_tasks,search_range)

  return robots_tasks
''' ################################################## 
      Local TSP: (Go through robots+task points)
      Task execution of each robots
'''###################################################
def distance(tour, xy):
  # index with list, and repeat the first city
  tour = np.append(tour, tour[0])
  d = np.diff(xy[tour], axis=0) 
  return np.sqrt((d**2).sum(axis=1)).sum()
 
def OneRobotTask_solver(num_seeds, max_NFE, xy, search_range):
  # calculate the theoritical minimum of search length
  num_rtsk = xy.shape[0]
  search_length = min(search_range)
  L_star = np.sqrt(num_rtsk)*0.25*search_length

  # time mark of each calculation
  ft = np.zeros((num_seeds, max_NFE)) 
  exect = np.arange(num_rtsk)  
  exectt = np.zeros((num_seeds,num_rtsk)).astype(int)

  # task execution
  for seed in range(num_seeds):
    np.random.seed(seed)
    # random initial tour
    exect = np.random.permutation(exect)
    bestf = 99999
    n = 0
    while n < max_NFE and bestf > L_star:
    # mutate the tour using two random cities
      trial_exect = np.copy(exect) # do not operate on original list
      a = np.random.randint(num_rtsk)
      b = np.random.randint(num_rtsk)
    # the entire route between two cities are reversed
      if a > b:
        a,b = b,a
        trial_exect[a:b+1] = trial_exect[a:b+1][::-1]

      trial_f = distance(trial_exect, xy)

      if trial_f < bestf:
        exect = trial_exect
        bestf = trial_f
      ft[seed,n] = bestf
      n += 1
      exectt[seed] = exect
  # print exect
  # print bestf
  best_ind = np.unravel_index(np.argmin(np.nonzero(ft)), ft.shape)
  bestf = ft[best_ind]
  bestexect = exectt[best_ind[0]]
  
  return bestexect,bestf,ft # best result of all seeds

''' ################################################## 
      Global TSP: summary of all robots execution
      Task execution of all robots
'''###################################################
def MRTA_Solver(robots,tasks,num_seeds,max_NFE,search_range):
  robots_tasks = raw_allocation(robots,tasks,search_range) # robot task assignment
  
  # change task assignment to points coordinates
  MRTA_EXE = np.zeros(robots.shape[1])
  MRTA_tour = dict()
  MRTA_tour = {k: [] for k in range(robots.shape[1])}
  for i in range(robots.shape[1]):
    if verbose:
      print robots[:,i].shape
      print tasks[:,robots_tasks[i]].shape
    OneRobotTask = np.vstack((robots[:,i].T,tasks[:,robots_tasks[i]].T))
    # print OneRobotTask
  # find best tour and shortest distance for each robot
    besttour, bestf, ft = OneRobotTask_solver(num_seeds,max_NFE,OneRobotTask,search_range)
  # Mark all the robots execution
    MRTA_tour[i] = OneRobotTask[besttour,:]
    MRTA_EXE[i] = bestf
    
    print "%d of %d robot executing tasks" %(i+1 , robots.shape[1])
    print "Task assignments", robots_tasks[i]
    print "Running distances", MRTA_EXE[i]
    
  # print MRTA_tour
  if save:
    filename = "MRTA_tour.npy"
    np.save(filename,MRTA_tour)
    print MRTA_tour

  return MRTA_tour, MRTA_EXE
''' #################################################### 
      Visualizations of task assignment and execution
'''#####################################################

def exect_map_show(MRTA_tour,robots,tasks,search_range):
  # plt.subplot(1,1,1)
  color_arr = color_produce(robots)
  for i in range(robots.shape[1]):
    MRTA_tour[i] = np.vstack((MRTA_tour[i], MRTA_tour[i][0])) # for plotting
    # print MRTA_tour[i]
    plt.plot(MRTA_tour[i][:,0], MRTA_tour[i][:,1], marker='o', color=color_arr[i,:])
  plt.show()

def color_produce(robots):
  np.random.seed(0)
  robot_num = robots.shape[1]
  color_arr = np.random.rand(robot_num,3)

  return color_arr
def allocation_show(robots,tasks,robots_tasks,search_range):
	# plt.scatter(robots[0,:], robots[1,:], marker='o', color="r",alpha=0.3)
  color_arr = color_produce(robots)
  plt.subplot(1,1,1)
  for i in range(robots.shape[1]):
    color_r = color_arr[i,:]
    r_label = 'robots'+str(i+1)
    rt_label = r_label+'\'s tasks'
    plt.scatter(robots[0][i], robots[1][i], marker='*', color=color_r,s=200,label=r_label)
    # plt.scatter(tasks[:,robots_tasks[i]][0,:],tasks[:,robots_tasks[i]][1,:],marker='o',color = color_r, label=rt_label)
  plt.legend(loc='center left',bbox_to_anchor=(0.9, 0.5))
	# plt.show()

# NFE to different cities
num_robots = 10 # tasks equal to num of robots
num_tasks = 300
search_range = [10,10]
num_seeds = 10
max_NFE = 10000
robots = assign_robots(search_range,num_robots)
tasks = get_random_tasks(search_range,num_tasks)

# robots_tasks = raw_allocation(robots,tasks,search_range)
MRTA_tour, MRTA_EXE = MRTA_Solver(robots,tasks,num_seeds,max_NFE,search_range)
exect_map_show(MRTA_tour,robots,tasks, search_range)


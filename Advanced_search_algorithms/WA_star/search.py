# Basic searching algorithms

def st_goal(start,goal):
    if start==goal:
       return True
'''
def cost(n_r,n_c,start):
    return abs(n_r-start[0])+abs(n_c-start[1])
'''


def cost_tot_fun(rr,cc,start,goal,parent):
    eps=0
    return eps*(abs(rr-goal[0])+abs(cc-goal[1]))+actualCost([rr,cc],start,parent)


def actualCost(index,start,parentNode):
    cost = 0
    current = index
    while current[0] != start[0] and current[1] != start[1]:
        cost = cost + 1
        current = parentNode[current[0]][current[1]]
    return cost

def astar(grid, start, goal):
    '''Return a path found by A* alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> astar_path, astar_steps = astar(grid, start, goal)
    It takes 7 steps to find a path using A*
    >>> astar_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False  #initially no goal reached
    R=C=len(grid)  #imagining the grid is square

    #define total cost function(h(x)+g(x)) and a cost queue to save all the cost values and node coordinates correspondingly
    cost_tot=0
    cost_fun_queue=[[cost_tot,start[0],start[1]]]

    # queue that maintains all the visited nodes
    visited=[]

    # direction vectors for up,right,down and left of row and column
    d_r=[0,1,0,-1]  
    d_c=[1,0,-1,0]


    visited.append(start)
    # queue that maintains the parent node information for backtracking of path. Size is equal to number of nodes
    parent=[[None for i in range(R)]for j in range(C)]
    parent[start[0]][start[1]]=[0,0]

    #special condition
    if st_goal(start,goal):
       found=True
       print("That was easy! goal is same as start")

    while len(cost_fun_queue)>0:
        cost_fun_queue.sort()
        [cost_to_go,row,col]=cost_fun_queue.pop(0)
        steps+=1

        if [row,col]==goal:
           found=True
           break
  
        for i in range(len(d_r)):
           n_row=row+d_r[i]
           n_col=col+d_c[i]
           #boundary conditions check
           if n_row<0 or n_col<0: continue
           if n_row>= R or n_col >=C: continue
           ##visited node check
           if [n_row,n_col] in visited: continue
           #obstacle check
           if grid[n_row][n_col] ==1: continue
 
           visited.append([n_row,n_col])
           parent[n_row][n_col]=[row,col]
           cost_fun_queue.append([cost_tot_fun(n_row,n_col,start,goal,parent),n_row,n_col])
           if [n_row,n_col]== goal:
              found=True
              break
    if found:
       path.append(goal)
       parent_pr=parent[goal[0]][goal[1]]
       path.append(parent_pr)
       while parent_pr != start:
           parent_pr=parent[parent_pr[0]][parent_pr[1]]    #backtracking path through parents
           path.append(parent_pr)
       path.reverse()
    if found:
        print(f"It takes {steps} steps to find a path using A*")
    else:
        print("No path found")
    return path, steps


# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()

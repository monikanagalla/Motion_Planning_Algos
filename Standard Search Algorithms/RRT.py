# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
from bresenham import bresenham
import math

# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag
        

    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)

    
    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        weight= np.linalg.norm(np.array((node1.row,node1.col)) - np.array((node2.row,node2.col)))        
        return weight

    
    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        pts=list(bresenham(node1.row, node1.col, node2.row, node2.col))
        coll=False
        for p in pts:
            if self.map_array[int(p[0]),int(p[1])]==0:
               coll=True
        return coll


    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###
        rows=self.size_row
        cols=self.size_col
        if goal_bias==1:
            new_pt=self.goal
        else:    
            new_pt=Node(np.random.randint(rows),np.random.randint(cols))

        return new_pt

    
    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        weights=[]
        for j in range(np.size(self.vertices)):
            #print(j)
            weights.append(self.dis(self.vertices[j], point))
        #print(weights)
        index=np.argmin(weights)
        #print(index)
        return self.vertices[index]

    def get_new_node(self, n_node,point,step):

        [x1,y1]=[n_node.row,n_node.col]
        [x2,y2]=[point.row,point.col]
        angle=math.atan2((y2-y1),(x2-x1))
        distance=self.dis(n_node,point)
        if distance<=step:
           new_node=point
        else:
            new_node=Node(int(x1+step*math.cos(angle)),int(y1+step*math.sin(angle)))
        
        return new_node


    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance 
        '''
        ### YOUR CODE HERE ###
        neighbours=[]
        for i in range(len(self.vertices)):
            expand_node=self.vertices[i]
            distance=self.dis(new_node,expand_node)
            if distance<=neighbor_size:
                neighbours.append(expand_node)

        return neighbours




    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###
        costs=[]
        free_neighbours=[]
        for i in range(len(neighbors)):
           
            cost=self.dis(neighbors[i],new_node)+neighbors[i].cost
            
            costs.append(cost)
        
        sort_costs=np.argsort(cost)
        # now that we have costs of all the neighbors lets check for valid free neighbors
        for p in sort_costs:
            if not self.check_collision(new_node,neighbors[p]):
                free_neighbours.append(neighbors[p])
                new_node.parent=neighbors[p]
                new_node.cost=costs[p]
                self.vertices.append(new_node)
                break
        #Once we have valid neighbors we rewire the neighbors when a low cost path is found to that neighbor through our new_node
        #this loop rewires our free_neighbors to new parent accordring to teh costs found 
        for k in range(np.size(free_neighbours)):
            intial_cost=free_neighbours[k].cost
            distance_cost=self.dis(new_node,free_neighbours[k])

            if intial_cost>(distance_cost+new_node.cost):
                free_neighbors[k].parent=new_node
                free_neighbors[k].cost=distance_cost+new_node.cost

    
    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')
        
        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col and cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()


    def RRT(self, n_pts=1000):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###
        goal_bias=0.05
        step=10
        #rows=self.size_row
        cols=self.size_col
        #print(self.vertices[0])
        for i in range(n_pts): 
            print(i)         
            goal_b=np.random.choice(2, 1, p=[1-goal_bias, goal_bias])
            new_pt=self.get_new_point(goal_b)
            print("new poit")
            print(new_pt.row,new_pt.col)
            near_node=self.get_nearest_node(new_pt)

            new_node=self.get_new_node(near_node,new_pt,step)

            if new_node.row<=0  or new_node.col>=cols: continue
            if self.map_array[new_node.row,new_node.col]==0:continue
            if self.check_collision(near_node,new_node):continue
            else:
                #print("new node found")
                new_node.parent=near_node
                new_node.cost=self.dis(near_node,new_node)+new_node.parent.cost
                self.vertices.append(new_node)

                goal_dis=self.dis(new_node,self.goal)
                if goal_dis<step:
                    goal_coll=self.check_collision(new_node,self.goal)
                    if goal_coll==False:
                        self.goal.parent=new_node
                        self.goal.cost=goal_dis+new_node.cost
                        self.vertices.append(self.goal)
                        self.found=True
                        break 
        '''            
        for p in self.vertices:
            print(p.row,p.col)
        print(len(self.vertices))    
        print(np.size(self.vertices))
        '''
  

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, check if reach the neighbor region of the goal.

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")
        
        # Draw result
        self.draw_map()


    def RRT_star(self, n_pts=1000, neighbor_size=20):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            neighbor_size - the neighbor distance
        
        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###

        # In each step,
        # get a new point, 
        # get its nearest node, 
        # extend the node and check collision to decide whether to add or drop,
        # if added, rewire the node and its neighbors,
        # and check if reach the neighbor region of the goal if the path is not found.


        # Output
        steps=10
        goal_neighbors=[]
        for i in range(n_pts):
            new_point=self.get_new_point(0.1)
            nearest_node=self.get_nearest_node(new_point)
            new_node=self.get_new_node(nearest_node,new_point,steps)
            if not self.check_collision(new_node,nearest_node):
                neighbors=self.get_neighbors(new_node,neighbor_size)
                self.rewire(new_node,neighbors)
            goal_dis=self.dis(new_node,self.goal)

            if goal_dis<steps:
                if not self.check_collision(new_node,self.goal):
                   goal_neighbors.append([new_node,goal_dis])
                   self.goal.parent=new_node

        x=[]
        
        for h in range(len(goal_neighbors)):
            x.append(goal_neighbors[h][0].cost+goal_neighbors[h][1])
        y=x.index(min(x))
        self.goal.cost=x[y]
        self.goal.parent=goal_neighbors[y][0] 
        self.vertices.append(self.goal) 
        #print(self.goal.parent)
        self.found=True 

        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()

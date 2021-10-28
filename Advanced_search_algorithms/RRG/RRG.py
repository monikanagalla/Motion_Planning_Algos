
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from PIL import Image
from numpy import random
import math
from scipy import spatial
from bresenham import bresenham
import time

# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parents = []   # parent nodes list
        self.costs = []       # cost of parent nodes list

# Class for RRT
class RRG:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = [self.start]                    # list of nodes
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

    def rrgwire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###
        self.start.costs=[0]
        cost_through_neighbor = []
        for i in range(len(neighbors)):
            collision = self.check_collision(neighbors[i],new_node)
            if collision == False:
                 # check the path cost to new_node through neighbor[i], take only minimum path through neighbor[i], use sorted array and first element
                path_cost=neighbors[i].costs[0] + self.dis(neighbors[i],new_node)
                #append the new path found through neighbour[i] and its costs to a list which can be again used to find shortest path and rewire the parents
                cost_through_neighbor.append((neighbors[i],path_cost))
         #sort the costs to find the best path through neighbors   
        cost_through_neighbor=sorted(cost_through_neighbor,key=lambda cost: cost[1])

        #wire the parents of new_node as neigbors, through their least cost path found above
        for i in range(len(cost_through_neighbor)): 
            #append neighbors into parents list of new_node
            new_node.parents.append(cost_through_neighbor[i][0])
            #append the particular path cost into costs list of new_ndoe
            new_node.costs.append(cost_through_neighbor[i][1])
            # find if there exits a better path to neighbor through the new_node
            b_cost=new_node.costs[0]+self.dis(neighbors[i],new_node)
            if b_cost<neighbors[i].costs[0]:
                #if better path found then insert the new_node to neighbor[i]'s parents list as first item, because this is the low cost path
                neighbors[i].costs.insert(0,b_cost)
                neighbors[i].parents.insert(0,new_node)
            
            



    
    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:None]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            for i in range(len(node.parents)):
                plt.plot([node.col, node.parents[i].col], [node.row, node.parents[i].row], color='y')

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col and cur.row != self.start.row:
                plt.plot([cur.col, cur.parents[0].col], [cur.row, cur.parents[0].row], color='b')
                cur = cur.parents[0]
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()
       

    def RRG(self, n_pts=1000, neighbor_size=20):
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
        cols=self.size_col
        neighbor_size=20
        goal_bias = 0.05
        step_size=10
        goal_neighbors=[]
        for i in range(n_pts): 
            goal_b=np.random.choice(2, 1, p=[1-goal_bias, goal_bias])
            new_pt=self.get_new_point(goal_b)
            print(new_pt.row,new_pt.col)
            near_node=self.get_nearest_node(new_pt)
            new_node=self.get_new_node(near_node,new_pt,step_size)
            

            if new_node.row<=0  or new_node.col>=cols: continue
            if self.map_array[new_node.row,new_node.col]==0:continue
            if self.check_collision(near_node,new_node):continue

            neighbors = self.get_neighbors(new_node,neighbor_size) 
            self.rrgwire(new_node,neighbors)
            self.vertices.append(new_node)
            goal_dis=self.dis(new_node,self.goal)

            if goal_dis<=step_size:
                if not self.check_collision(new_node,self.goal):
                   self.found=True
                   goal_neighbors.append([new_node,goal_dis])
                   self.goal.parents.append(new_node)
                   self.goal.costs.append(new_node.costs[0] + self.dis(new_node,self.goal))
                   self.vertices.append(new_node)
                   #end the search when a path is found
                   break
        x=[]
        
        for h in range(len(goal_neighbors)):
            x.append(goal_neighbors[h][0].costs[0]+goal_neighbors[h][1])
        y=x.index(min(x))
        self.goal.cost=x[y]
        self.goal.parents[0]=goal_neighbors[y][0]             

        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.costs[0]
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()

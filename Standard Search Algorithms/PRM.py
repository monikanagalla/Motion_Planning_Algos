# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import spatial
from bresenham import bresenham


# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path


    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        ### YOUR CODE HERE ###
        pts=list(bresenham(p1[0], p1[1], p2[0], p2[1]))
        coll=False
        for p in pts:
            if self.map_array[int(p[0]),int(p[1])]==0:
               coll=True
        return coll
                


    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        #print(np.array((point1[0],point1[1])) )
        #print( np.array(point2[0],point2[1]))
        weight= np.linalg.norm(np.array((point1[0],point1[1])) - np.array((point2[0],point2[1])))

        #weight=np.sqrt(np.sum(np.power(point1[0]-point2[0],2),np.power(point1[1]-point2[1],2)))
        return weight


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        #self.samples.append((0, 0))
        '''
        We want our samples to be equally spaced in image. Therefore we first calculate sqrt(n_pts) since our image is a 2D image, take this as step_len. 
        Then divide #rows and #cols with step_len which gives us the individual step legnths of rows and cols
        '''

        rows=self.size_row
        cols=self.size_col
        step_len=int(np.sqrt(n_pts))
        print(step_len)
        #ratio=rows/cols
        step_len_row= int(rows/(step_len))
        step_len_col=int(cols/(step_len))
        #print(step_len_row,step_len_col)
        pts_x,pts_y=[np.arange(0,rows,step_len_row),np.arange(0,cols,step_len_col)]
        print(pts_x,pts_y)
        for i in pts_x:
            for j in pts_y:
                if self.map_array[i,j] != 0:
                    self.samples.append((i,j))  
        #print(np.size(self.samples))


    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        #self.samples.append((0, 0))
        '''Here we just choose random points as samples. 
         We use random.randint function of numpy python to generate random samples in range(#rows,#cols).'''

        rows=self.size_row
        cols=self.size_col
        arr=[(np.random.randint(rows),np.random.randint(cols)) for j in range(n_pts)]
        for i in range(n_pts):
            if self.map_array[arr[i]]==1:
               self.samples.append(arr[i])


    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        #self.samples.append((0, 0))
        '''First generate random samples of length n_pts then select a point from this set and take a guassian sample of this point.
        Check if both the points are either in free space or in collision. If so then drop the points.
        If not then append the free space sample to our samples queue.
        '''
        rows=self.size_row
        cols=self.size_col
        arr=[(np.random.randint(rows),np.random.randint(cols)) for j in range(n_pts)]
        sigma=20
        for j in range(n_pts):
            mu_x=arr[j][0]
            mu_y=arr[j][1]
            print(arr[j])
            g_s=(int(np.random.normal(mu_x, sigma)),int(np.random.normal(mu_y, sigma)))
            if g_s[0]<=0  or g_s[1]<=0 or g_s[1]>=cols or g_s[0]>=rows: continue
            print(g_s)
            print(self.map_array[g_s],self.map_array[arr[j]])
            if self.map_array[arr[j]] ==1 and self.map_array[g_s]==1: continue
            if self.map_array[arr[j]] ==0 and self.map_array[g_s]==0: continue
            if self.map_array[arr[j]]==1: self.samples.append(arr[j])
            if self.map_array[arr[j]]==0: self.samples.append(g_s)
            print(self.samples)

        


    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        #self.samples.append((0, 0))
        '''
        This is more like gaussian sampling except that the updation condition is different. 
        Here our criteria is both of the point and its gaussian pt are in collision then consider the midpoint of the line between point and the gaussian pt. 
        Check if this mid point is in collision if not then add it to samples
        '''
        rows=self.size_row
        cols=self.size_col
        arr=[(np.random.randint(rows),np.random.randint(cols)) for j in range(n_pts)]
        sigma=20
        for j in range(n_pts):
            mu_x=arr[j][0]
            mu_y=arr[j][1]
            #print(arr[j])
            if self.map_array[arr[j]] ==1:continue      # we need to consider q1 that is only in collision if not then pass
            g_s=(int(np.random.normal(mu_x, sigma)),int(np.random.normal(mu_y, sigma))) # gaussian sample of q1
            if g_s[0]<=0  or g_s[1]<=0 or g_s[1]>=cols or g_s[0]>=rows: continue #check that it is in bounds
            #print(g_s)
            #print(self.map_array[g_s],self.map_array[arr[j]])
            if self.map_array[g_s]==1: continue
            elif self.map_array[int((arr[j][0]+g_s[0])/2),int((arr[j][1]+g_s[1])/2)]==1:
                mid_pt=(int((arr[j][0]+g_s[0])/2),int((arr[j][1]+g_s[1])/2))
                self.samples.append(mid_pt)



    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)

        ### YOUR CODE HERE ###
        pairs = []
        pairs_val=[]
        r = 20
        #self.samples.append(self.start)
        #self.samples.append(self.goal)
        positions = np.array(self.samples)
        kdtree = spatial.KDTree(positions)
        pairs_kd= kdtree.query_pairs(r)
        
        #print(np.size(self.samples))
        #print(pairs_kd)
        for p in pairs_kd:
            #print(p)
            #print(self.samples[p[0]],self.samples[p[1]])
            if self.check_collision(self.samples[p[0]],self.samples[p[1]]):continue
            
            else:
                #print("no collision")
                pairs_val.append(p[0])
                #print("appended")
                pairs_val.append(p[1])
                #print("appended second")
                weight=self.dis(self.samples[p[0]],self.samples[p[1]])
                #print(weight)
                pairs.append((p[0],p[1],weight))
                #print("final appended")
        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02), 
        #          (p_id1, p_id2, weight_12) ...]


        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01), 
        #                                     (p_id0, p_id2, weight_02), 
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of 
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from(pairs_val)
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))
        #return self.samples


    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, we        
        #                (start_id, p_id2, weight_s2) ...]
        start_pairs = []
        goal_pairs = []
        pairs_kd_start=[]
        pairs_kd_goal=[]

        #take rgoal and rstart by trial and error
        #else there is a pssibility of no path found
        rstart = 40
        rgoal=80
        positions = np.array(self.samples)
        kdtree = spatial.KDTree(positions)
        pairs_kd_start = spatial.KDTree.query_ball_point(kdtree,start,rstart)
        pairs_kd_goal = spatial.KDTree.query_ball_point(kdtree,goal,rgoal)

        

        # find nearest neighbours of start node
        for p in pairs_kd_start:
            print (p)
            if self.check_collision(self.samples[p],start):continue
            
            else:
                weight=self.dis(self.samples[p],start)
                start_pairs.append(('start',p,weight))


        # find nearest neighbours of goal node       
        for p in pairs_kd_goal:
            #print(p)
            #print(self.samples[p[0]],self.samples[p[1]])
            if self.check_collision(self.samples[p],goal):continue
            
            else:

                weight=self.dis(self.samples[p],goal)
                goal_pairs.append(('goal',p,weight))


        print(goal_pairs,start_pairs)         
        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        
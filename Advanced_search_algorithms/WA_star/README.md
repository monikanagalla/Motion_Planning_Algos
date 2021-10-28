# RBE 550 - Basic Search Algorithms Implementation

## Overview

In this assignment, you are going to implement **BFS**, **DFS**, **Dijkstra** and **A*** algorithms. These are the basic algorithms for discrete planning, which also share a lot of similarities. This template is provided to you as a start point. After you finish coding, you would be able to create your own map to test different algorithms, and visualize the path found by them.

Files included:

**search.py** is the file where you will implement your algorithms.

**main.py** is the scrip that provides helper functions that load the map from csv files and visualize the map and path. You are not required to modify anything but you are encouraged to understand the code.

**map.csv** is the map file you could modify to create your own map.

**test_map.csv** restores a test map for doc test purpose only. Do not modify this file.

Please finish reading all the instructions and rubrics below before starting actual coding.

## Instruction

Before starting any coding, please run the code first:

`python search.py`

When running **search.py** as a main function, it will run a doc test for all the algorithms. It loads **test_map.csv** as the map for testing.

As you haven't written anything yet, you would see you fail all the doc test. In the future, after implementing each algorithm, you should run this file again and make sure to pass the doc test (you will see nothing if you pass the test).

For visualization, please run:

`python main.py`

There should be 4 maps shown representing the result of 4 algorithms. As said before, there should be no path shown in the graph as you haven't implemented anything yet. The **main.py** loads the map file **map.csv**, which you are free to modify to create your own map.

Please keep in mind that, the coordinate system used here is **[row, col]**, which is different from [x, y] in Cartesian coordinates. 

Now, read the algorithm description in **seach.py** and make sure you understand the input/arguments and required output/return of the algorithms.

---

Until now, I hope you have a basic understanding of the template code and what to do next. 

One thing you may notice is that, as not all students come with a CS or programming background, the data structure used is the simplest python list, no class defined or any fancy stuff. If you are an experienced programmer and think the structure is not 'beautiful', please feel free to modify anything you want as long as you have the algorithms run well. You are encouraged to do so.

## Rubrics

- (2 pts) Your algorithms pass the basic doc test.

  After implementation, as mentioned before, you could run the doc test by:

  `python search.py`

  If you see nothing, it means you pass the test.

  For Dijkstra and A*, the cost to move each step is set to be 1. The heuristic for each node is its Manhattan distance to the goal.

  When you explore the nearby nodes in the map, please follow this order **"right, down, left, up"**, which means "[0, +1], [+1, 0], [0, -1], [-1, 0]" in coordinates. There is nothing wrong using other exploring orders. It is just that the test result was gotten by algorithms using this order. A different order may result in different results, which could let you fail the doc test.

  Please also keep in mind that the test map is stored in **test_map.csv**, so please do **NOT** modify this file. Also, for the output of the function, steps and path should include and count the first / start position and the goal / end position.

  ---

- (2 pts) Your algorithms pass all the tests for grading.

  There will be other test cases for grading. Please think of different possible test cases yourself, modify **test.csv** to create your own map and see the result. It's a good practice to think of boundary cases such as, the start and end are the same position, a 1x1 map, a map that does not have a valid path, etc.

  ---

- (1 pts) A clear structure without duplicating your code.

  Please figure out the similarity of these four algorithms. As they are somewhat similar, if you implement them separately, you would end up writing a lot of duplicated codes. It's not a good practice to do so. Please find out the similarity and construct your code in a way that you would not duplicate the codes.

  ---

- (3 pts) Documentation

  Besides the code, you should also include a documentation with the following content:

  - Code and algorithm explanation

    You should briefly explain how these four algorithms work. How they are different and similar. A pseudocode would be helpful for explanation. You also need to briefly explain your code structure.

  - Test example, test result and explanation

    Run your code with `python main.py`. You could use the map **map.csv** I provided, but I encourage you to create your own map. Run the test and see the paths you found and the steps/time it takes for these algorithms to find them. Try to explain the reason why different algorithm gives different or same result.

  - Reference paper and resources if any

  Include the documentation as a pdf file, or if you prefer, a md file.

  
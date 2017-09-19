import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
from random import choice, shuffle

# read in the images into greyscale mats
def read_sample(name):
    data = misc.imread(name, False, "L")
    return data

# see if one pixel in the image is something that we are interested in
# preprocessing for the transformation
def is_intereting_point(val):
    return val < 100

# transform the data from greyscale image to coordinates
# before it can be fed into the clustering process
def transform_data(raw):
    w = len(raw)
    h = len(raw[0])
    res = []
    for i in range(w):
        for j in range(h):
            if (is_intereting_point(raw[i][j])):
                res.append([float(i), float(j)])
    return res

# generate a blank image with the same resolution
def gen_cpy(raw):
    w = len(raw)
    h = len(raw[0])
    res = []
    for i in range(w):
        temp = []
        for j in range(h):
            temp.append(255)
        res.append(temp)
    return res


def TFKMeansCluster(vectors, noofclusters):
    """
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    """

    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)

    #Find out the dimensionality
    dim = len(vectors[0])

    #Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)

    #GRAPH OF COMPUTATION
    #We initialize a new graph and set it as the default during each run
    #of this algorithm. This ensures that as this function is called
    #multiple times, the default graph doesn't keep getting crowded with
    #unused ops and Variables from previous function calls.

    graph = tf.Graph()

    with graph.as_default():

        #SESSION OF COMPUTATION

        sess = tf.Session()

        ##CONSTRUCTING THE ELEMENTS OF COMPUTATION

        ##First lets ensure we have a Variable vector for each centroid,
        ##initialized to one of the vectors from the available data points
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]
        ##These nodes will assign the centroid Variables the appropriate
        ##values
        centroid_value = tf.placeholder("float", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))

        ##Variables for cluster assignments of individual vectors(initialized
        ##to 0 at first)
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        ##These nodes will assign an assignment Variable the appropriate
        ##value
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))

        ##Now lets construct the node that will compute the mean
        #The placeholder for the input
        mean_input = tf.placeholder("float", [None, dim])
        #The Node/op takes the input and computes a mean along the 0th
        #dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)

        ##Node for computing Euclidean distances
        #Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(
            v1, v2), 2)))

        ##This node will figure out which cluster to assign a vector to,
        ##based on Euclidean distances of the vector from the centroids.
        #Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

        ##INITIALIZING STATE VARIABLES

        ##This will help initialization of all Variables defined with respect
        ##to the graph. The Variable-initializer should be defined after
        ##all the Variables have been constructed, so that each of them
        ##will be included in the initialization.
        init_op = tf.global_variables_initializer()

        #Initialize all variables
        sess.run(init_op)

        ##CLUSTERING ITERATIONS

        #Now perform the Expectation-Maximization steps of K-Means clustering
        #iterations. To keep things simple, we will only do a set number of
        #iterations, instead of using a Stopping Criterion.
        noofiterations = 20
        for iteration_n in range(noofiterations):
            print("executing iteration ", iteration_n, " of ", noofiterations)
            ##EXPECTATION STEP
            ##Based on the centroid locations till last iteration, compute
            ##the _expected_ centroid assignments.
            #Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                #Compute Euclidean distance between this vector and each
                #centroid. Remember that this list cannot be named
                #'centroid_distances', since that is the input to the
                #cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                #Now use the cluster assignment node, with the distances
                #as the input
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                #Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})

            ##MAXIMIZATION STEP
            #Based on the expected state computed from the Expectation Step,
            #compute the locations of the centroids so as to maximize the
            #overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                #Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                #Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: np.array(assigned_vects)})
                #Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})

        #Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments

def main():
    raw = read_sample("img/sample1.png")
    data = transform_data(raw)
    noofclusters = 15
    centroids, assignments = TFKMeansCluster(data, noofclusters)
    print("centroids: ")
    print(centroids)
    print("length of assignments: ", len(assignments))
    print("length of data: ", len(data))
    size = len(data)
    res = gen_cpy(raw)
    for i in range(size):
        res[int(data[i][0])][int(data[i][1])] = (200 / noofclusters) * (assignments[i] + 1)
    plt.imshow(res)
    plt.show()


if __name__ == "__main__":
    main()

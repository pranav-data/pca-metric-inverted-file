#!/usr/local/bin/python3

# %% codecell - Implementing PCA algorithm along the Principal component with highest explained variance

# Nomenclature: object = row in dataset, reference = subset of objects to aid in querying
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


# Importing all objects as a DataFrame from the csv file
all_objects = pd.read_csv('D:\Github\pca-mif\cleaned_data.csv')

# Converting to numpy array for compute speed
my_objects = np.array(all_objects.iloc[:])

pca1 = PCA(n_components=1)
data = my_objects
# Projecting the objects along the major principal component. 'y' is the projection of each object
y = np.array(np.round(pca1.fit_transform(data), 1))
y1 = pca1.explained_variance_ratio_

# Printing the explained variance in % from major principal component alone
print(f'Explained variance in %: {y1*100}')

# Selecting 1000 targets along the range of major principal component (rounded to one decimal point)
principal_axis_targets = np.array(np.round(np.linspace(min(y), max(y), 1000), 1))

# List to store indices of objects closest to each target on the major principal component
closest_obj_index = []
for ref in range(1000):
    # Duplicating each target for each object to apply accelerated matrix numpy operations
    # Selecting the closest object index position for each of the 1000 targets
    B = np.argmin(np.abs(principal_axis_targets[ref]-y[:]), axis=0)
    closest_obj_index.append(B)
    # Reassigning each closest object away from transformed dataset to obtain 1000 unique objects close to targets
    y[B] = 1000000000
    if (np.mod(ref, 100) == 0):
        print(f'ref#{ref} computed')  # Progress tracker
print(f'Unique references generated:{len(np.unique(closest_obj_index))}')  # Rechecking uniqueness

# *******************************************************************************


# %% codecell - References  1000 targets & Objects - remaining objects
my_refs = np.array(all_objects.iloc[[x[0] for x in closest_obj_index]])
all_objects.drop([x[0] for x in closest_obj_index], axis=0, inplace=True)
my_objects = np.array(all_objects.iloc[:])

# *******************************************************************************
# %% codecell - Generating Metric Inverted file

# Function to evaluate spearman footrule distance


def compute_d_sfd(gdf_object, gdf_ref):
    # columnwise ie feature-wise estimation of distance of object to reference
    dist = np.sum(np.abs(gdf_object[:]-gdf_ref[:]), axis=1)
    # Rounding distances to one decimal point
    return np.round(dist, 1)


# List to store 10 closest references with their corresponding ranks (0 to 9) for each object
# also known in literature as L~OR
obj_to_ref = []


for obj in range(len(my_objects)):

    # Columnwise sum after matrix operation A - B , on duplicating each object to length of reference list for parallelized computing
    distance_ref = compute_d_sfd([my_objects[obj][:]
                                  for i in range(my_refs.shape[0])], my_refs)

    # Partial sorting and slicing top 10 ranks through argpartition function
    top_ten_unsorted_indexes = np.argpartition(distance_ref, 10)[:10]

    top_ten_unsorted_values = []
    for x in top_ten_unsorted_indexes:
        # Finding the actual values of d_sfd from the top_ten_unsorted_indexes
        top_ten_unsorted_values.append(distance_ref[x])

    # Sorting only the top ten indexes by using zip function, with spearman footrule distances computed as key
    sorted_top_ten_tuples = sorted(
        zip(top_ten_unsorted_indexes, top_ten_unsorted_values), key=lambda x: x[1])

    # Generating ranks array for each object
    ranks = [j[0] for j in sorted_top_ten_tuples]

    # Append the ranks to a new list. I found dictionaries proved slow, as they arent as well optimized as numpy arrays for computation and manipulation
    obj_to_ref.append(np.array(ranks))

    # Monitor progress for every 10,000 objects
    if(np.mod(obj, 10000) == 0):
        print(f'LOR computed for {obj}')

# Convert entire list of ranks for each object as numpy array
obj_to_ref = np.array(obj_to_ref)

# Open file to output Metric inverted file aka MIF, as an appendable state of .txt file
f = open("output full MIF - PCA.txt", "a")

# Indexed metric inverted file to generate from L~OR
MIF_file = []

# List of tuples -(object number,corresponding rank) for each reference derived from obj_to_ref
for reference in range(len(my_refs)):

    # Try block to handle references that are not close to any object, and continue appending the text file without freezing the execution of the loop
    try:

        # Returns the indices of format [row,col] in obj_to_ref where boolean condition is true
        y = np.argwhere(obj_to_ref == reference)
        tup = [(int(z[0]), int(z[1])) for z in y]

        # Labelling each reference prior to printing the MIF corresponding to it
        print(str('ref_no')+' '+str(reference)+':'+'\t', tup, file=f)
        MIF_file.append(tup)

        if(np.mod(reference, 50) == 0):
            print(f'ref#{reference} data appended')  # Progress tracker

    except:

        continue

f.close()

# *******************************************************************************

# %% codecell
# User input to determine number of queries to find similar Objects
# trials = input('enter the number of random queries among the list of objects to test:')

trials = 20


# %% codecell
# Querying for similar objects when the query is a subset of object list

no_correct = 0
# Seeding the random number generator to compare the performance with and without PCA to select references
np.random.seed(1000000)

# Selecting 'trials' number of objects to query for similar objects
query_index_array = np.random.randint(0, len(all_objects), int(trials))
print(query_index_array)

# Function to compute spearman footrule distance for all axes


def compute_d_sfd2(gdf_object, gdf_ref):

    dist = np.sum(np.abs(gdf_object[:]-gdf_ref[:]))
    return dist


for query_index in query_index_array:

    query_object = np.array(my_objects[query_index][:])
    dist_closest_ref = compute_d_sfd([query_object[:]
                                      for x in range(len(my_refs))], my_refs)

    # Selecting the closest 10 references to query in unsorted form
    index_top10unsorted_ref = np.argpartition(dist_closest_ref, 10)[:10]
    val_top10unsorted_ref = []
    for x in index_top10unsorted_ref:
        val_top10unsorted_ref.append(dist_closest_ref[x])

    # Sorting tuples based on distance of query to each reference
    sorted_tuples = sorted(zip(index_top10unsorted_ref, val_top10unsorted_ref), key=lambda x: x[1])
    # return the (index_of_reference,position) as a list and save as ranks1

    ranks1 = [(j[0], i) for i, j in enumerate(sorted_tuples)]
    # Sort the list based on index_of_reference to compute the difference in rank,thereby establishing similarity
    ranks2 = sorted(ranks1, key=lambda x: x[0])
    ranks2 = np.array([y[1] for y in ranks2])
    # Initiating an accumulator to decide which objects are closest to query as per the algorithm
    accum = []
    for obj in range(len(my_objects)):

        if(np.mod(obj, 100000) == 0):
            print(f'iterated for obj# {obj}')  # Progress tracker

        # If an object has the same references in the top 10 closest references as the query, then compute accumulator
        if(np.isin(index_top10unsorted_ref, obj_to_ref[obj]).all()):

            # Using enumerate function to find the position to compute the accumulator

            one = np.array([(i, j) for i, j in enumerate(obj_to_ref[obj])])
            two = np.array([y[0] for y in sorted(one, key=lambda x:x[1])])
            # Appending a tuple of object number and spearman footrule distance of similar Objects
            accum.append((obj, np.sum(compute_d_sfd2(two, ranks2))))

    # Final sorted tuples of object id and corresponding accumulator
    # Closest in terms of ranks will be determined as most similar object as per the algorithm
    final = np.array(sorted(accum, key=lambda x: x[1]))

    # If all accumulators in 20 similarity objects are equal then recompute distance of all objects with equal accumulator and sort top 20
    if(np.equal([x[1] for x in final[:20]], [final[0][1]]*len(final[:20])).all()):

        selected_objs = [x[0] for x in final if (x[1] == final[0][1])]
        object_ids = []

        # Recomputing spearman footrule distance to find most similar objects to the query (extension to basic algorithm)
        dist_query = compute_d_sfd([my_objects[query_index][:] for x in range(
            len(selected_objs))], my_objects[selected_objs][:])

        # Handling exceptions when there is only one object that is similar
        try:
            top_20_indexes = np.argpartition(dist_query, 20)[:20]
            for z in top_20_indexes:
                object_ids.append(selected_objs[z])
        except:
            object_ids = final

        print('\nAccumulator is same minimum value for more than 20 objects!')
        print(
            f'\naccumulator array for {query_index} after double checking distance is {object_ids}')
        if (np.isin(query_index, object_ids)):
            no_correct += 1

    else:

        # Return twenty similar objects as computed by original algorithm
        print(
            f'\naccumulator array for {query_index} is {final[:20]}, where first number in ordered pair is object id and second number is accumulator\n ')
        final_objects = [x[0] for x in final[:20]]

        # List of distances of similar objects relative to the query object
        Average_dist = []

        # Printing the closeness of similar objects, just for comparing with randomly selecting objects
        distance_query = compute_d_sfd([my_objects[query_index][:] for x in range(
            len(final_objects))], my_objects[final_objects][:])
        print([x for x in zip(final_objects, distance_query)])
        print(f'\nAverage distance of similar 20 objects: {np.mean(distance_query)}\n')
        Average_dist.append(np.mean(distance_query))
        if (np.isin(query_index, [x[0] for x in final])):
            # If query object is one of the similarity objects, Then add to number correct
            no_correct += 1

# Segment to compute accuracy after all queries are run

print(f'\nPercentage accuracy of search:{no_correct*100/int(trials)}\n')
print(
    f'Average distance of top 20 similar objects through PCA reference selection is {np.mean(Average_dist)}\n')

###############################################################################
# %% codecell
# segment to generate foreign objects based on the range for each column in the dataset

#trials = input('enter the number of random queries among the list of objects to test:')
trials = 10

# Finding the range of values for each column aka feature in the dataframe
columns_min = pd.DataFrame.min(all_objects, axis=0)
columns_max = pd.DataFrame.max(all_objects, axis=0)

# Printing the range of each column as tuple (min,max)
print(f'\nRange of each column in the dataset represented as a tuple (column_minimum, column_maximum) is:')
print([x for x in zip(columns_min, columns_max)])
print('\n\n')

# query_list holds a list of lists, where each list is a single query
query_list = []
for trial in range(int(trials)):
    query_list.append([int(np.random.randint(x[0], x[1], 1))
                       for x in zip(columns_min, columns_max)])
print(query_list)

###############################################################################
# %% codecell
# Searching the index for similar objects against foreign objects that were randomly generated in the previous block
no_correct = 0


for query in query_list:

    query_object = np.array(query[:])
    dist_closest_ref = compute_d_sfd([query_object[:]
                                      for x in range(len(my_refs))], my_refs)

    # Selecting the closest 10 references to query in unsorted form
    index_top10unsorted_ref = np.argpartition(dist_closest_ref, 10)[:10]
    val_top10unsorted_ref = []
    for x in index_top10unsorted_ref:
        val_top10unsorted_ref.append(dist_closest_ref[x])

    # Sorting tuples based on distance of query to each reference
    sorted_tuples = sorted(
        zip(index_top10unsorted_ref, val_top10unsorted_ref), key=lambda x: x[1])

    # return the (index_of_reference,position) as a list and save as ranks1
    ranks1 = [(j[0], i) for i, j in enumerate(sorted_tuples)]

    # Sort the list based on index_of_reference to compute the difference in rank,thereby establishing similarity
    ranks2 = sorted(ranks1, key=lambda x: x[0])
    ranks2 = np.array([y[1] for y in ranks2])
    accum = []
    for obj in range(len(my_objects)):

        if(np.mod(obj, 100000) == 0):
            print(f'iterated for obj# {obj}')

        # If an object has the same references in the top 10 closest references as the query, then compute accumulator
        if(np.isin(index_top10unsorted_ref, obj_to_ref[obj]).all()):

            # Using enumerate function tofind position to compute the accumulator

            one = np.array([(i, j) for i, j in enumerate(obj_to_ref[obj])])
            two = np.array([y[0] for y in sorted(one, key=lambda x:x[1])])
            accum.append((obj, np.sum(compute_d_sfd2(two, ranks2))))

    # Final sorted tuples of object id and corresponding accumulator
    final = np.array(sorted(accum, key=lambda x: x[1]))

    # Handling the case where there are no similar objects as the query, then look for a match for closest 3 references instead of 10
    if len(final) == 0:
        index_top10unsorted_ref = np.argpartition(dist_closest_ref, 3)[:3]
        val_top10unsorted_ref = []
        for x in index_top10unsorted_ref:
            val_top10unsorted_ref.append(dist_closest_ref[x])

        # Sorting tuples based on distance of query to each reference
        sorted_tuples = sorted(
            zip(index_top10unsorted_ref, val_top10unsorted_ref), key=lambda x: x[1])

        # return the (index_of_reference,position) as a list and save as ranks1
        ranks1 = [(j[0], i) for i, j in enumerate(sorted_tuples)]
        # Sort the list based on index_of_reference to compute the difference in rank,thereby establishing similarity
        ranks2 = sorted(ranks1, key=lambda x: x[0])
        ranks2 = np.array([y[1] for y in ranks2])
        # Initiating an accumulator to decide which objects are closest to query as per the algorithm
        accum = []
        for obj in range(len(my_objects)):

            if(np.mod(obj, 100000) == 0):
                print(f'iterated for obj# {obj}')

        # If an object has the same references in the top 3 closest references as the query, then compute accumulator
            if(np.isin(index_top10unsorted_ref, obj_to_ref[obj][:3]).all()):

                # Using enumerate function tofind position to compute the accumulator

                one = np.array([(i, j) for i, j in enumerate(obj_to_ref[obj][:3])])
                two = np.array([y[0] for y in sorted(one, key=lambda x: x[1])])
                accum.append((obj, np.sum(compute_d_sfd2(two, ranks2))))

        final = np.array(sorted(accum, key=lambda x: x[1]))

    # Return twenty similar objects as computed by original algorithm
    print(f'(obj_id,accumulator) for query  is {final[:20]}')
    print(f'query array is: {query[:]}')
    obj_ids_final = [x[0] for x in final[:20]]
    print(f'similar object arrays are:')
    print(my_objects[obj_ids_final])

###############################################################################
# %% codecell

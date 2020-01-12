#!/usr/local/bin/python3

# %% codecell
"""Implementing PCA algorithm along the Principal component
with highest explained variance"""

# Nomenclature: object = row in dataset, reference = subset of objects to aid in querying
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


# Importing all objects as a DataFrame from the csv file
ALL_OBJECTS = pd.read_csv('D:\Github\pca-mif\cleaned_data.csv')

# Converting to numpy array for compute speed
MY_OBJECTS = np.array(ALL_OBJECTS.iloc[:])


def implement_pca_targets(gdf_objects, no_components):
    """
    This function implements Principal Component Analysis and determines
    1000 targets along the range of the projections of data points along this
    axis, therby capturing maximum variance.

    We do a check of explained variance ratio for the dataset to gauge the
    effectiveness of our approach.

    """
    pca1 = PCA(n_components=no_components)
    data = gdf_objects
    # Projecting the objects along the major principal component.
    # 'y' is the projection of each object
    t_data = np.array(np.round(pca1.fit_transform(data), 1))
    exp_ratio = pca1.explained_variance_ratio_

    # Printing the explained variance in % from major principal component alone
    print(f'Explained variance in %: {exp_ratio*100}')

    # Selecting 1000 targets along the range of major principal component
    # (rounded to one decimal point)
    principal_axis_targets = np.array(np.round(np.linspace(min(t_data), max(t_data), 1000), 1))

    # List to store indices of objects closest to each target on the major principal component
    closest_obj_index = []
    for ref in range(1000):
        # Duplicating each target for each object to apply accelerated matrix numpy operations
        # Selecting the closest object index position for each of the 1000 targets
        min_pos = np.argmin(np.abs(principal_axis_targets[ref]-t_data[:]), axis=0)
        closest_obj_index.append(min_pos)
        # Reassigning each closest object away from transformed dataset
        # to obtain 1000 unique objects close to targets
        t_data[min_pos] = 1000000000
        if np.mod(ref, 100) == 0:
            print(f'ref#{ref} computed')  # Progress tracker
    # Rechecking uniqueness
    print(f'Unique references generated:{len(np.unique(closest_obj_index))}')

    pca_refs = np.array(gdf_objects.iloc[[x[0] for x in closest_obj_index]])
    gdf_objects.drop([x[0] for x in closest_obj_index], axis=0, inplace=True)
    pca_objects = np.array(gdf_objects.iloc[:])

    return pca_objects, pca_refs
# *****************************************************************************


MY_OBJECTS, MY_REFS = implement_pca_targets(MY_OBJECTS, 1)

# *******************************************************************************
# %% codecell - Generating Metric Inverted file


def compute_d_sfd(gdf_object, gdf_ref):
    """Function to evaluate spearman footrule distance"""
    # columnwise ie feature-wise estimation of distance of object to reference
    dist = np.sum(np.abs(gdf_object[:]-gdf_ref[:]), axis=1)
    # Rounding distances to one decimal point
    return np.round(dist, 1)


def compute_lor_mif(my_objects, my_refs):
    """Compute L tilda OR and metric inverted file"""
    # List to store 10 closest references with their corresponding ranks (0 to 9) for each object
    # also known in literature as L~OR
    obj_to_ref = []

    for each_obj in range(len(my_objects)):

        # Columnwise sum after matrix operation A - B , on duplicating each object
        # to length of reference list for parallelized computing
        distance_ref = compute_d_sfd([my_objects[each_obj][:]
                                      for i in range(my_refs.shape[0])], my_refs)

        # Partial sorting and slicing top 10 ranks through argpartition function
        top_ten_unsorted_indexes = np.argpartition(distance_ref, 10)[:10]

        top_ten_unsorted_values = []
        for index in top_ten_unsorted_indexes:
            # Finding the actual values of d_sfd from the top_ten_unsorted_indexes
            top_ten_unsorted_values.append(distance_ref[index])

        # Sorting only the top ten indexes by using zip function,
        # with spearman footrule distances computed as key
        sorted_top_ten_tuples = sorted(
            zip(top_ten_unsorted_indexes, top_ten_unsorted_values), key=lambda x: x[1])

        # Generating ranks array for each object
        ranks = [j[0] for j in sorted_top_ten_tuples]

        # Append the ranks to a new list. I found dictionaries proved slow,
        # as they arent as well optimized as numpy arrays for computation and manipulation
        obj_to_ref.append(np.array(ranks))

        # Monitor progress for every 10,000 objects
        if np.mod(each_obj, 10000) == 0:
            print(f'LOR computed for {each_obj}')

    # Convert entire list of ranks for each object as numpy array
    obj_to_ref = np.array(obj_to_ref)

    # Open file to output Metric inverted file aka MIF, as an appendable state of .txt file
    file = open("output full MIF - PCA.txt", "a")

    # Indexed metric inverted file to generate from L~OR
    mif_file = []

    # List of tuples -(object number,corresponding rank) for each reference derived from obj_to_ref
    for reference in range(len(my_refs)):

        # Try block to handle references that are not close to any object,
        # and continue appending the text file without freezing the execution of the loop
        try:

            # Returns the indices of format [row,col] in obj_to_ref where boolean condition is true
            matched_refs = np.argwhere(obj_to_ref == reference)
            tup = [(int(z[0]), int(z[1])) for z in matched_refs]

            # Labelling each reference prior to printing the MIF corresponding to it
            print(str('ref_no')+' '+str(reference)+':'+'\t', tup, file=file)
            mif_file.append(tup)

            if np.mod(reference, 50) == 0:
                print(f'ref#{reference} data appended')  # Progress tracker

        except IndexError():
            print(f'index out of bounds error - skipped ref#{reference}')
            continue
        except ValueError():
            print(f'Value error - skipped ref#{reference}')
            continue
        except TypeError():
            print(f'Type error - skipped ref#{reference}')
            continue
    file.close()
    return obj_to_ref


OBJ_TO_REF = compute_lor_mif(MY_OBJECTS, MY_REFS)

# *****************************************************************************

# %% codecell
# User input to determine number of queries to find similar Objects
# TRIALS = input('enter the number of random queries among the list of objects to test:')
TRIALS = 20


# %% codecell
# Querying for similar objects when the query is a subset of object list

NO_CORRECT = 0
# Seeding the random number generator to compare the performance with
# and without PCA to select references
np.random.seed(1000000)

# Selecting 'TRIALS' number of objects to query for similar objects
QUERY_INDEX_ARRAY = np.random.randint(0, len(ALL_OBJECTS), int(TRIALS))
print(QUERY_INDEX_ARRAY)


def compute_d_sfd2(gdf_object, gdf_ref):
    """Function to compute spearman footrule distance for all axes"""
    dist = np.sum(np.abs(gdf_object[:]-gdf_ref[:]))
    return dist


def finalist(query, my_objects, my_refs, closest_refs):
    """Function to return 'final' list of similar objects
    for each query, without exception handling"""

    if isinstance(query, int):
        query_object = np.array(my_objects[query][:])
    else:
        query_object = np.array(query[:])
    dist_closest_ref = compute_d_sfd([query_object[:]
                                      for x in range(len(my_refs))], my_refs)

    # Selecting the closest 10 references to query in unsorted form
    index_topunsorted_ref = np.argpartition(dist_closest_ref, closest_refs)[:closest_refs]
    val_topunsorted_ref = []
    for index1 in index_topunsorted_ref:
        val_topunsorted_ref.append(dist_closest_ref[index1])

        # Sorting tuples based on distance of query to each reference
        sorted_tuples = sorted(
            zip(index_topunsorted_ref, val_topunsorted_ref), key=lambda x: x[1])
        # return the (index_of_reference,position) as a list

        # Sort the list based on index_of_reference to compute the difference in rank,
        # thereby establishing similarity
        ranks2 = sorted([(j[0], i) for i, j in enumerate(sorted_tuples)], key=lambda x: x[0])
        ranks2 = np.array([y[1] for y in ranks2])

        # Initiating an accumulator to decide which objects are closest to query
        accum = []
        for obj in range(len(my_objects)):

            if np.mod(obj, 100000) == 0:
                print(f'iterated for obj# {obj}')  # Progress tracker

            # If an object has the same references in the top closest references as the query,
            # then compute accumulator
            if np.isin(index_topunsorted_ref, OBJ_TO_REF[obj][:closest_refs]).all():

                # Using enumerate function to find the position to compute the accumulator
                one = np.array([(i, j) for i, j in enumerate(OBJ_TO_REF[obj])])
                two = np.array([y[0] for y in sorted(one, key=lambda x: x[1])])
                # Appending a tuple of object number and spearman footrule
                # distance of similar Objects
                accum.append((obj, np.sum(compute_d_sfd2(two, ranks2))))

        # Final sorted tuples of object id and corresponding accumulator
        # Closest in terms of ranks will be determined as most similar object as per the algorithm
        return np.array(sorted(accum, key=lambda x: x[1]))


def search_index_array(final, query, my_objects, no_similar_objects_returned):
    """If all accumulators in 'no_similar_objects_returned' are equal
    then recompute distance of all objects with equal accumulator and sort
    top 'no_similar_objects_returned"""
    if np.equal([x[1] for x in final[:no_similar_objects_returned]],
                [final[0][1]]*len(final[:no_similar_objects_returned])).all():

        selected_objs = [x[0] for x in final if x[1] == final[0][1]]
        object_ids = []

    # Recomputing spearman footrule distance to find most similar objects
    # to the query (extension to basic algorithm)
        dist_query = compute_d_sfd([my_objects[query][:]
                                    for x in range(len(selected_objs))],
                                   my_objects[selected_objs][:])

        try:
            top_20_indexes = np.argpartition(dist_query, no_similar_objects_returned)[
                :no_similar_objects_returned]
            for index2 in top_20_indexes:
                object_ids.append(selected_objs[index2])
        except ValueError():
            # Handling exceptions when there is only one object that is similar
            print('ValueError noted!')
            object_ids = final
        except IndexError():
            print('IndexError noted!')
            object_ids = final

        print('\nAccumulator is same minimum value for more than 20 objects!')
        print(
            f'\naccumulator array for {query} after double checking distance is {object_ids}')
        distance_query = compute_d_sfd([my_objects[query][:] for x in range(
            len(object_ids))], my_objects[object_ids][:])
        print(f'\nAverage distance of similar objects: {np.mean(distance_query)}\n')
    else:

            # Return twenty similar objects as computed by original algorithm
        print(f'\naccumulator array for {query} is {final[:20]}) where')
        print(' first number in ordered pair is object id and second number is accumulator\n')
        final_objects = [x[0] for x in final[:20]]

        # Printing the closeness of similar objects, just for comparing
        # with randomly selecting objects
        distance_query = compute_d_sfd([my_objects[query][:] for x in range(
            len(final_objects))], my_objects[final_objects][:])
        print([x for x in zip(final_objects, distance_query)])
        print(f'\nAverage distance of similar 20 objects: {np.mean(distance_query)}\n')

    return object_ids


###############################################################################
for q1 in QUERY_INDEX_ARRAY:
    FINAL = finalist(q1, MY_OBJECTS, MY_REFS, 10)
    OBJECT_IDS = search_index_array(FINAL, q1, MY_OBJECTS, 20)
    # Segment to compute accuracy after all queries are run
    if (np.isin(q1, [x[0] for x in FINAL]) or np.isin(q1, OBJECT_IDS)):
        # If query object is one of the similarity objects, Then add to number correct
        NO_CORRECT += 1
print(f'\nPERCENTAGE ACCURACY OF SEARCH:{NO_CORRECT*100/int(TRIALS)}\n')

###############################################################################
# %% codecell

# TRIALS = input('enter the number of random queries among the list of objects to test:')
TRIALS = 10


def query_generator(trials):
    """segment to generate foreign objects based on
    the range for each column in the dataset"""
    # Finding the range of values for each column aka feature in the dataframe
    columns_min = pd.DataFrame.min(ALL_OBJECTS, axis=0)
    columns_max = pd.DataFrame.max(ALL_OBJECTS, axis=0)

    # Printing the range of each column as tuple (min,max)
    print(f'\nRange of each column in the dataset represented as a tuple is:\n')
    print([x for x in zip(columns_min, columns_max)])
    print('\n\n')

    np.random.seed(1000)

    # query_list holds a list of lists, where each list is a single query
    query_list = [int(np.random.randint(x[0], x[1], 1))
                  for x in zip(columns_min, columns_max) for trial in range(trials)]
    print(query_list)
    return query_list


###############################################################################
# %% codecell
# Searching the index for similar objects against foreign objects that were
# randomly generated in the previous block
NO_CORRECT = 0
QUERY_LIST = query_generator(TRIALS)

for q2 in QUERY_LIST:

    FINAL = finalist(q2, MY_OBJECTS, MY_REFS, 10)
    if not FINAL:
        # Handling the case where there are no similar objects as the query,
        # then look for a match for closest 3 references instead of 10
        FINAL = finalist(q2, MY_OBJECTS, MY_REFS, 3)

    parameter = int(min(len(FINAL), 20))

    # Return twenty similar objects as computed by original algorithm

    print(f'(obj_id,accumulator) for query is {FINAL[:parameter]}')
    print(f'query array is: {q2[:]}')
    obj_ids_final = [x[0] for x in FINAL[:parameter]]
    print(f'similar object arrays are:')
    print(MY_OBJECTS[obj_ids_final])

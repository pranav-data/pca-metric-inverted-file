# %% codecell - Implementing PCA algorithm along the Principal component with highest explained variance

#Nomenclature: object = row in dataset, reference = subset of objects to aid in querying
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np
import random

#Importing all objects as a DataFrame from the csv file
all_objects = pd.read_csv('my_df_cleaned8.csv')

#Converting to numpy array for compute speed
my_objects_small = np.array(all_objects.iloc[:])

pca1 = PCA(n_components=1)
data=my_objects_small
y=np.array(np.round(pca1.fit_transform(data),1))
y1=pca1.explained_variance_ratio_

# Printing the explained variance in % from major principal component alone
print(f'Explained variance in %: {y1*100}')

#Selecting 1000 targets along the range of major principal component (rounded to one decimal point)
principal_axis_targets = np.array(np.round(np.linspace(min(y),max(y),1000),1))


closest_obj_index = []
for ref in range(1000):
    # Duplicating each target for each object to apply accelerated matrix numpy operations
    array_principal_axis_targets = [principal_axis_targets[ref] for x in range(y.shape[0])]
    # Selecting the closest object index position for each of the 1000 targets
    B = np.argmin(np.abs(array_principal_axis_targets-y),axis=0)
    closest_obj_index.append(B)
    y[B]=1000000 # Reassigning each closest object away from transformed dataset to obtain 1000 unique objects close to targets
    if (np.mod(ref,10)==0):
        print(f'ref#{ref} computed') #Progress tracker
print(f'Unique references generated:{len(np.unique(closest_obj_index))}') #Rechecking uniqueness

#*******************************************************************************


# %% codecell - References  1000 targets & Objects - remaining objects
my_ref_small = np.array(all_objects.iloc[[x[0] for x in closest_obj_index]])
all_objects.drop([x[0] for x in closest_obj_index],axis=0,inplace=True)
my_objects_small=np.array(all_objects.iloc[:])

#*******************************************************************************
# %% codecell - Generating Metric Inverted file

# Function to evaluate spearman footrule distance
def compute_d_sfd(gdf_object,gdf_ref):
    #columnwise ie feature-wise estimation of distance of object to reference
    dist=np.sum(np.abs(gdf_object[:]-gdf_ref[:]),axis=1)
    return dist

# List to store 10 closest references with their corresponding ranks (0 to 9) for each object
# also known in literature as L~OR
obj_to_ref = []


for obj in range(len(my_objects_small)):

    # Columnwise sum after matrix operation A - B , on duplication each object to length of reference list for parallelized computing
    distance_ref=compute_d_sfd([my_objects_small[obj][:] for i in range(my_ref_small.shape[0])],my_ref_small)

    #Partial sorting and slicing top 10 ranks through argpartition function
    top_ten_unsorted_indexes = np.argpartition(distance_ref,10)[:10]

    top_ten_unsorted_values=[]
    for x in top_ten_unsorted_indexes:
        #Finding the actual values of d_sfd from the top_ten_unsorted_indexes
        top_ten_unsorted_values.append(distance_ref[x])

    #Sorting only the top ten indexes by using zip function, with distances computed as key
    sorted_top_ten_tuples = sorted(zip(top_ten_unsorted_indexes,top_ten_unsorted_values),key=lambda x:x[1])

    #Generating ranks array for each object
    ranks = [j[0] for j in sorted_top_ten_tuples]

    #Append the ranks to a new list. I found dictionaries proved slow, as they arent as well optimized as numpy arrays for computation and manipulation
    obj_to_ref.append(np.array(ranks))

    #Monitor progress for every 10,000 objects
    if(np.mod(obj,10000)==0):
        print(f'LOR computed for {obj}')

#Convert entire list of ranks for each object as numpy array
obj_to_ref = np.array(obj_to_ref)

#Open file to output Metric inverted file aka MIF, as an appendable state of .txt file
f = open("output full MIF - PCA.txt", "a")
MIF_file=[]

# List of tuples -(object number,corresponding rank) for each reference derived from obj_to_ref
for reference in range(len(my_ref_small)):

    # Try block to handle references that are not close to any object, and continue appending the text file
    try:

        y=np.argwhere(obj_to_ref==reference)
        tup=[(int(z[0]),int(z[1])) for z in y]

        print(str('ref_no')+' '+str(reference)+':'+'\t',tup,file=f)
        MIF_file.append(tup)

        print(f'ref#{reference} data appended') #Progress tracker


    except:

        continue

f.close()

#*******************************************************************************

# %% codecell
trials=input('enter the number of random queries among the list of objects to test:')
# %% codecell
# Slides algorithm LOR method - Existing objects query

import numpy as np
no_correct=0
np.random.seed(1000000)

query_index_array = np.random.randint(0,len(all_objects),int(trials))
print(query_index_array)

def compute_d_sfd2(gdf_object,gdf_ref):
    #columnwise ie feature-wise estimation of distance of object to reference
    dist=np.sum(np.abs(gdf_object[:]-gdf_ref[:]))
    return dist

for query_index in query_index_array:

    query_object = np.array(my_objects_small[query_index][:])
    dist_closest_ref = compute_d_sfd([query_object[:] for x in range(len(my_ref_small))],my_ref_small)

    #Selecting the closest 10 references to query in unsorted form
    index_top10unsorted_ref = np.argpartition(dist_closest_ref,10)[:10]
    val_top10unsorted_ref=[]
    for x in index_top10unsorted_ref:
        val_top10unsorted_ref.append(dist_closest_ref[x])

    #Sorting tuples based on distance of query to each reference
    sorted_tuples = sorted(zip(index_top10unsorted_ref,val_top10unsorted_ref),key=lambda x:x[1])
    #return the (index_of_reference,position) as a list and save as ranks1

    ranks1 = [(j[0],i) for i,j in enumerate(sorted_tuples)]
    #Sort the list based on index_of_reference to compute the difference in rank,thereby establishing similarity
    ranks2 = sorted(ranks1,key=lambda x:x[0])
    ranks2 = np.array([y[1] for y in ranks2])
    accum=[]
    for obj in range(len(my_objects_small)):

        if(np.mod(obj,100000)==0):
            print(f'iterated for obj# {obj}')

        #If an object has the same references in the top 10 closest references as the query, then compute accumulator
        if(np.isin(index_top10unsorted_ref,obj_to_ref[obj]).all()):

            #Using enumerate function tofind position to compute the accumulator

            one=np.array([(i,j) for i,j in enumerate(obj_to_ref[obj])])
            two=np.array([y[0] for y in sorted(one,key=lambda x:x[1])])
            accum.append((obj,np.sum(compute_d_sfd2(two,ranks2))))

    #Final sorted tuples of object id and corresponding accumulator
    final=np.array(sorted(accum,key=lambda x:x[1]))


    #If all accumulators in 20 similarity objects are equal then recompute distance of all objects with equal accumulator and sort top 20
    if(np.equal([x[1] for x in final[:20]],[final[0][1]]*len(final[:20])).all()):

        selected_objs = [x[0] for x in final if (x[1]==final[0][1])]
        object_ids = []

        #change line for external object
        dist_query=compute_d_sfd([my_objects_small[query_index][:] for x in range(len(selected_objs))],my_objects_small[selected_objs][:])

        #Handling exceptions when there is only one object
        try:
            top_20_indexes = np.argpartition(dist_query,20)[:20]
            for z in top_20_indexes:
                object_ids.append(selected_objs[z])
        except:
            object_ids =  final



        print('Accumulator is same minimum value for more than 20 objects!')
        print(f'accumulator array for {query_index} after double checking distance is {object_ids}')
        if (np.isin(query_index,object_ids)):
            no_correct+=1

    else:

    # Return twenty similar objects as computed by original algorithm
        print(f'accumulator array for {query_index} is {final[:20]}, where first number in ordered pair is object id and second number is accumulator ')
        final_objects = [x[0] for x in final[:20]]

        Average_dist=[]
        #Printing the closeness of similar objects, just for comparing with randomly selecting objects
        distance_query = compute_d_sfd([my_objects_small[query_index][:] for x in range(len(final_objects))],my_objects_small[final_objects][:])
        print([x for x in zip(final_objects,distance_query)])
        print(f'Average distance of similar 20 objects: {np.mean(distance_query)}')
        Average_dist.append(np.mean(distance_query))
        if (np.isin(query_index,[x[0] for x in final])):
            no_correct+=1

# If query object is one of the similarity objects, Then add to number correct
print(f'Percentage accuracy of search:{no_correct*100/int(trials)}')
print(f'Average distanceof top 20 similar objects through PCA reference selection is {np.mean(Average_dist)}')

# %% codecell
trials=input('enter the number of random queries among the list of objects to test:')

import pandas as pd
import numpy as np
import random

columns_min=pd.DataFrame.min(all_objects,axis=0)
columns_max=pd.DataFrame.max(all_objects,axis=0)

print([x for x in zip(columns_min,columns_max)])
query_list=[]
for trial in range(int(trials)):
    query_list.append( [int(np.random.randint(x[0],x[1],1)) for x in zip(columns_min,columns_max)])
print(query_list)
# %% codecell



#outside objects
import numpy as np
no_correct=0


def compute_d_sfd2(gdf_object,gdf_ref):
    #columnwise ie feature-wise estimation of distance of object to reference
    dist=np.sum(np.abs(gdf_object[:]-gdf_ref[:]))
    return dist

for query in query_list:

    query_object = np.array(query[:])
    dist_closest_ref = compute_d_sfd([query_object[:] for x in range(len(my_ref_small))],my_ref_small)

    #Selecting the closest 10 references to query in unsorted form
    index_top10unsorted_ref = np.argpartition(dist_closest_ref,10)[:10]
    val_top10unsorted_ref=[]
    for x in index_top10unsorted_ref:
        val_top10unsorted_ref.append(dist_closest_ref[x])

    #Sorting tuples based on distance of query to each reference
    sorted_tuples = sorted(zip(index_top10unsorted_ref,val_top10unsorted_ref),key=lambda x:x[1])
    #return the (index_of_reference,position) as a list and save as ranks1

    ranks1 = [(j[0],i) for i,j in enumerate(sorted_tuples)]
    #Sort the list based on index_of_reference to compute the difference in rank,thereby establishing similarity
    ranks2 = sorted(ranks1,key=lambda x:x[0])
    ranks2 = np.array([y[1] for y in ranks2])
    accum=[]
    for obj in range(len(my_objects_small)):

        if(np.mod(obj,100000)==0):
            print(f'iterated for obj# {obj}')

        #If an object has the same references in the top 10 closest references as the query, then compute accumulator
        if(np.isin(index_top10unsorted_ref,obj_to_ref[obj]).all()):

            #Using enumerate function tofind position to compute the accumulator

            one=np.array([(i,j) for i,j in enumerate(obj_to_ref[obj])])
            two=np.array([y[0] for y in sorted(one,key=lambda x:x[1])])
            accum.append((obj,np.sum(compute_d_sfd2(two,ranks2))))

    #Final sorted tuples of object id and corresponding accumulator
    final=np.array(sorted(accum,key=lambda x:x[1]))

    if len(final)==0:
        index_top10unsorted_ref = np.argpartition(dist_closest_ref,3)[:3]
        val_top10unsorted_ref=[]
        for x in index_top10unsorted_ref:
            val_top10unsorted_ref.append(dist_closest_ref[x])

    #Sorting tuples based on distance of query to each reference
        sorted_tuples = sorted(zip(index_top10unsorted_ref,val_top10unsorted_ref),key=lambda x:x[1])
    #return the (index_of_reference,position) as a list and save as ranks1

        ranks1 = [(j[0],i) for i,j in enumerate(sorted_tuples)]
    #Sort the list based on index_of_reference to compute the difference in rank,thereby establishing similarity
        ranks2 = sorted(ranks1,key=lambda x:x[0])
        ranks2 = np.array([y[1] for y in ranks2])
        accum=[]
        for obj in range(len(my_objects_small)):

            if(np.mod(obj,100000)==0):
                print(f'iterated for obj# {obj}')

        #If an object has the same references in the top 10 closest references as the query, then compute accumulator
            if(np.isin(index_top10unsorted_ref,obj_to_ref[obj][:3]).all()):

                #Using enumerate function tofind position to compute the accumulator

                one=np.array([(i,j) for i,j in enumerate(obj_to_ref[obj][:3])])
                two=np.array([y[0] for y in sorted(one,key=lambda x:x[1])])
                accum.append((obj,np.sum(compute_d_sfd2(two,ranks2))))

        final=np.array(sorted(accum,key=lambda x:x[1]))





    # Return twenty similar objects as computed by original algorithm
    print(f'(obj_id,accumulator) for query  is {final[:20]}')
    print(f'query array is: {query[:]}')
    obj_ids_final = [x[0] for x in final[:20]]
    print(f'similar object arrays are:')
    print(my_objects_small[obj_ids_final])




# %% codecell

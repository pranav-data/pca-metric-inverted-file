# pca-metric-inverted-file
Created an improved algorithm to index documents on top of traditional Metric Inverted file

Strategy:

I found through my research, that utilizing numpy arrives which are written using Cython (parallelized C libraries to implement functions in wrapper python objects) were the best way to achieve speed of indexing and writing to an MIF File.
My current code was able to index 1.022 million objects with 1000 references with 21columns in about 20mins on 16GB RAM 7th Gen intel i7 processor.
Some of the features I implemented was using ‘argpartition’ function in numpy which can be used to partially segregate the ten closest references from 1000, and internally rank the 10 closest references from there.
The ‘argwhere’ function was used as a single step to generate the MIF from the LOR tilda.

To improve the effectiveness of similarity search, I have created 2 additional problem statements.
1.	Use principal component analysis to determine the references to be used along PCA-1 that maximizes the variance of the data set along the 21 dimensional space. To implement this I found the range of object projections along PCA1 and evaluated the explained variance ratio for the data, to ensure it is above 50%. The range was broken down into 1000 target points and the projection of closest object was assigned as a reference. To ensure 1000 unique references, that object had to be dropped in further selection of references in the ‘for loop’.
The hope with this approach is that similarity search performs better for outlier queries outside the existing object list. 
2.	Use normalized data to reduce the skew towards certain columns and improve effectiveness of similarity search. This approach had troubles finding similarity objects as there were very few data points whose references were identical in the 10 closest references. This was not ideal to conduct similarity search as there were a lot of collisions and exceptions thrown. On handling these exceptions, many queries that were generated randomly, returned only the same object as similar object as a result.
I tried to reduce the benchmark to 3 closest references to compute the accumulator, but this approach did not really improve the returned similarity objects.

For 10 exactly same queries I ran within existing object set, with the same seed on random number generator, I got 105.33 for the average distance of 20 similar objects through randomized references and 76.33 for the average objects through PCA allocation of references.
Also, The PCA  allocation of references also allowed more similar objects to be retrieved than through random allocation. There were cases in random allocation where the only similar object was the object itself, which means the 10 closest references for no other object matched with the query.

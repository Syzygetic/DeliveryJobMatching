# Retrieve latitude and longitude using postal code
for public jobs V in V_all:
    call function getcoordinates() using V postal code, store results
for private jobs J in J_all:
    call function getcoordinates() using J postal code, store results

# Create frozen adjacency matrices
Initiate a matrix of zeros with number of rows and columns equals to total data
for row in range(len(frozen adjacency matrix)):
    for column in range(len(frozen adjacency matrix)):
        if row and column is the same job:
            continue to the next iteration
        else:
            Add an edge between rows and columns of private jobs with the same company id
            Add an edge between rows and columns,
            of all public and private job pairs that have the same frozen feature

# Create capacity adjacency matrices
Initiate a matrix of zeros with number of rows and columns equals to total data
for row in range(len(capacity adjacency matrix)):
    for column in range(len(capacity adjacency matrix)):
        if row and column is the same job:
            continue to the next iteration
        else:
            Add an edge between rows and columns of private jobs with the same company id
            Add an edge between rows and columns,
            of all public and private job pairs that have the same frozen feature

# Create distance adjacency matrices
Initiate a matrix of zeros with number of rows and columns equals to total data
for row in range(len(distance adjacency matrix)):
    for column in range(len(distance adjacency matrix)):
        if row and column is the same job:
            continue to the next iteration
        else:
            Calculate the distances of all possible routes,
            between public and private job pairs at the row and column,
            using haversine_distance() function
            Calculate gaussian_similarity() using the shortest route distance if the public job is urgent,
            else using longest route distance if the public job is not urgent
            add the gaussian similarity score as the edge value between the row and column

# Create composite adjacency matrix
Add an appropriate weight to each of the frozen, capacity, and distance adjacency matrices
Sum up the frozen, capacity, distance adjacency matrices and averaging these matrices into a single composite matrix

# Perform spectral clustering
Pass the composite adjacency matrix into the Spectral Clustering Algorithm
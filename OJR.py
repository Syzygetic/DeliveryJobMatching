import requests
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import networkx as nx

# ----------------------------------------------------------- Functions -----------------------------------------------------------

# Function to retrieve latitude, longitude using postal code
def getcoordinates(address):
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']
    else:
        pass

# Function to create frozen adjacency matrix
def create_frozen_adjacency_matrix(company_datapoints, job_datapoints, features_adjacency_matrix):
    datapoints_num = len(company_datapoints) + len(job_datapoints)
    alljobs_datapoints = pd.concat([company_datapoints, job_datapoints]).to_dict('records')
    frozen_adjacency_matrix = np.zeros((datapoints_num, datapoints_num))

    for i in range(len(frozen_adjacency_matrix)):
        for j in range(len(frozen_adjacency_matrix)):
            if i == j:
                continue
            else:
                if i < len(company_datapoints) and j < len(company_datapoints):
                    if company_datapoints['Company Id'][i] == company_datapoints['Company Id'][j]:
                        frozen_adjacency_matrix[i, j] = 3.0 # Optimal 4.0, previous optimal 5.0, new 15.0
                        features_adjacency_matrix[i, j] = "CompanyId " + str(company_datapoints['Company Id'][i])
                else:
                    if (i < len(company_datapoints) and j >= len(company_datapoints)) or (i >= len(company_datapoints) and j < len(company_datapoints)):
                        if alljobs_datapoints[i]['Frozen'] == "Yes" and alljobs_datapoints[j]['Frozen'] == "Yes":
                            frozen_adjacency_matrix[i, j] = 2.0 # Optimal 1.0, new 3.0
                            features_adjacency_matrix[i, j] = "Frozen"
                        
    return frozen_adjacency_matrix

# Function to create capacity adjacency matrix
def create_capacity_adjacency_matrix(company_datapoints, job_datapoints, features_adjacency_matrix):
    datapoints_num = len(company_datapoints) + len(job_datapoints)
    alljobs_datapoints = pd.concat([company_datapoints, job_datapoints]).to_dict('records')
    capacity_adjacency_matrix = np.zeros((datapoints_num, datapoints_num))

    for i in range(len(capacity_adjacency_matrix)):
        for j in range(len(capacity_adjacency_matrix)):
            if i == j:
                continue
            else:
                if i < len(company_datapoints) and j < len(company_datapoints):
                    if company_datapoints['Company Id'][i] == company_datapoints['Company Id'][j]:
                        capacity_adjacency_matrix[i, j] = 1.0 # Optimal 4.0, previous optimal 5.0, new 15.0
                else:
                    if (i < len(company_datapoints) and j >= len(company_datapoints)) or (i >= len(company_datapoints) and j < len(company_datapoints)):
                        if alljobs_datapoints[i]['Capacity'] == alljobs_datapoints[j]['Capacity']:
                            capacity_adjacency_matrix[i, j] = 1.0 # Optimal 1.0, new 2.0

                            if features_adjacency_matrix[i, j] != None:
                                features_adjacency_matrix[i, j] += ", " + alljobs_datapoints[i]['Capacity']
                            else:
                                features_adjacency_matrix[i, j] = alljobs_datapoints[i]['Capacity']
                        
    return capacity_adjacency_matrix

# Function to calculate haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius = 6371  # Radius of the Earth in kilometers. Use 3956 for miles.
    distance = radius * c

    return distance

# Function to calculate gaussian kernel / gaussian similarity
def gaussian_similarity(distance, urgency_type, sigma):
    if urgency_type == "Urgent":
        # Use the standard gaussian similarity function (smaller distance results in higher similarity value)
        similarity = math.exp(-distance**2 / (2 * sigma**2))
    elif urgency_type == "Not Urgent":
        # Use the complementary of the gaussian similarity function, the gaussian function is subtracted from 1 (bigger distance results in higher similarity value)
        similarity = 1 - math.exp(-distance**2 / (2 * sigma**2))
    return similarity

# Function to create distance adjacency matrix
def create_distance_adjacency_matrix(company_datapoints, job_datapoints):
    datapoints_num = len(company_datapoints) + len(job_datapoints)
    alljobs_datapoints = pd.concat([company_datapoints, job_datapoints]).to_dict('records')
    distance_adjacency_matrix = np.zeros((datapoints_num, datapoints_num))
    all_jobs_lat_long = pd.concat([company_datapoints[['Pickup Lat Long', 'Delivery Lat Long']].copy(), job_datapoints[['Pickup Lat Long', 'Delivery Lat Long']].copy()], ignore_index=True)
    all_jobs_similarity = []

    for i in range(len(distance_adjacency_matrix)):
        for j in range(len(distance_adjacency_matrix)):
            if i == j:
                continue
            else:
                if i < len(company_datapoints) and j < len(company_datapoints):
                    if company_datapoints['Company Id'][i] == company_datapoints['Company Id'][j]:
                        distance_adjacency_matrix[i, j] = 1.0
                else:
                    if (i < len(company_datapoints) and j >= len(company_datapoints)) or (i >= len(company_datapoints) and j < len(company_datapoints)):
                        # This approach calculates the gaussian similarity using the shortest possible route between 2 jobs
                        possible_route_permutations_dist_list = []

                        i_pickup_lat, i_pickup_long = all_jobs_lat_long['Pickup Lat Long'][i].split(", ")
                        i_delivery_lat, i_delivery_long = all_jobs_lat_long['Delivery Lat Long'][i].split(", ")
                        j_pickup_lat, j_pickup_long = all_jobs_lat_long['Pickup Lat Long'][j].split(", ")
                        j_delivery_lat, j_delivery_long = all_jobs_lat_long['Delivery Lat Long'][j].split(", ")

                        # Route 1: pu_location_1 -> do_location_1 -> pu_location_2 -> do_location_2
                        possible_route_permutations_dist_list.append(
                            haversine_distance(float(i_pickup_lat), float(i_pickup_long), float(i_delivery_lat), float(i_delivery_long)) +
                            haversine_distance(float(i_delivery_lat), float(i_delivery_long), float(j_pickup_lat), float(j_pickup_long)) +
                            haversine_distance(float(j_pickup_lat), float(j_pickup_long), float(j_delivery_lat), float(j_delivery_long))
                        )
                        # Route 2: pu_location_1 -> pu_location_2 -> do_location_1 -> do_location_2
                        possible_route_permutations_dist_list.append(
                            haversine_distance(float(i_pickup_lat), float(i_pickup_long), float(j_pickup_lat), float(j_pickup_long)) +
                            haversine_distance(float(j_pickup_lat), float(j_pickup_long), float(i_delivery_lat), float(i_delivery_long)) +
                            haversine_distance(float(i_delivery_lat), float(i_delivery_long), float(j_delivery_lat), float(j_delivery_long))
                        )
                        # Route 3: pu_location_1 -> pu_location_2 -> do_location_2 -> do_location_1
                        possible_route_permutations_dist_list.append(
                            haversine_distance(float(i_pickup_lat), float(i_pickup_long), float(j_pickup_lat), float(j_pickup_long)) +
                            haversine_distance(float(j_pickup_lat), float(j_pickup_long), float(j_delivery_lat), float(j_delivery_long)) +
                            haversine_distance(float(j_delivery_lat), float(j_delivery_long), float(i_delivery_lat), float(i_delivery_long))
                        )
                        # Route 4: pu_location_2 -> do_location_2 -> pu_location_1 -> do_location_1
                        possible_route_permutations_dist_list.append(
                            haversine_distance(float(j_pickup_lat), float(j_pickup_long), float(j_delivery_lat), float(j_delivery_long)) +
                            haversine_distance(float(j_delivery_lat), float(j_delivery_long), float(i_pickup_lat), float(i_pickup_long)) +
                            haversine_distance(float(i_pickup_lat), float(i_pickup_long), float(i_delivery_lat), float(i_delivery_long))
                        )
                        # Route 5: pu_location_2 -> pu_location_1 -> do_location_2 -> do_location_1
                        possible_route_permutations_dist_list.append(
                            haversine_distance(float(j_pickup_lat), float(j_pickup_long), float(i_pickup_lat), float(i_pickup_long)) +
                            haversine_distance(float(i_pickup_lat), float(i_pickup_long), float(j_delivery_lat), float(j_delivery_long)) +
                            haversine_distance(float(j_delivery_lat), float(j_delivery_long), float(i_delivery_lat), float(i_delivery_long))
                        )
                        # Route 6: pu_location_2 -> pu_location_1 -> do_location_1 -> do_location_2
                        possible_route_permutations_dist_list.append(
                            haversine_distance(float(j_pickup_lat), float(j_pickup_long), float(i_pickup_lat), float(i_pickup_long)) +
                            haversine_distance(float(i_pickup_lat), float(i_pickup_long), float(i_delivery_lat), float(i_delivery_long)) +
                            haversine_distance(float(i_delivery_lat), float(i_delivery_long), float(j_delivery_lat), float(j_delivery_long))
                        )

                        if i >= len(company_datapoints):
                            if alljobs_datapoints[i]['Urgency'] == "Urgent":
                                shortestroutedist = min(possible_route_permutations_dist_list)

                                # Calculate gaussian similarity using the shortest possible route between the 2 jobs
                                similarity = gaussian_similarity(shortestroutedist, alljobs_datapoints[i]['Urgency'], 10)
                            elif alljobs_datapoints[i]['Urgency'] == "Not Urgent":
                                longestroutedist = max(possible_route_permutations_dist_list)

                                # Calculate gaussian similarity using the longest possible route between the 2 jobs
                                similarity = gaussian_similarity(longestroutedist, alljobs_datapoints[i]['Urgency'], 35)
                        elif j >= len(company_datapoints):
                            if alljobs_datapoints[j]['Urgency'] == "Urgent":
                                shortestroutedist = min(possible_route_permutations_dist_list)

                                # Calculate gaussian similarity using the shortest possible route between the 2 jobs
                                similarity = gaussian_similarity(shortestroutedist, alljobs_datapoints[j]['Urgency'], 10)
                            elif alljobs_datapoints[j]['Urgency'] == "Not Urgent":
                                longestroutedist = max(possible_route_permutations_dist_list)

                                # Calculate gaussian similarity using the longest possible route between the 2 jobs
                                similarity = gaussian_similarity(longestroutedist, alljobs_datapoints[j]['Urgency'], 35)

                        all_jobs_similarity.append(similarity)

                        # Add the gaussian similarity as the distance adjacency matrix value between the 2 jobs
                        distance_adjacency_matrix[i, j] = similarity

    # Plotting the histogram
    plt.hist(all_jobs_similarity, bins=50)  # Number of bins can be adjusted as needed
    plt.xlabel('Gaussian Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gaussian Similarity')
    plt.show()

    return distance_adjacency_matrix

# Function to create an array to store all job's company id (Jobs that does not belong to any company will be temporarily assigned company id 0)
def create_companyid_array(company_datapoints, job_datapoints):
    datapoints_num = len(company_datapoints) + len(job_datapoints)
    companyid_array = np.zeros(datapoints_num, dtype=int)

    for i in range(len(company_datapoints)):
        companyid_array[i] = company_datapoints['Company Id'][i]

    return companyid_array

def check_matching_clusters(cluster_labels, company_ids, company_to_cluster_mapping):
    # Populate the dictionary with the values from the arrays
    for idx, company_id in enumerate(company_ids):
        # Only add to the dictionary if the company ID is not already present
        if company_id not in company_to_cluster_mapping:
            cluster_label = cluster_labels[idx]
            company_to_cluster_mapping[company_id] = cluster_label

    # Check if all items in the arrays match as expected
    for idx, company_id in enumerate(company_ids):
        cluster_label = cluster_labels[idx]
        if company_to_cluster_mapping.get(company_id) != cluster_label:
            return False

    return True

# Function to create an array to store all job's order IDs
def create_orderid_array(company_datapoints, job_datapoints):
    orderid_array = np.concatenate((company_datapoints['Order Id'], job_datapoints['Order Id']))

    return orderid_array

def assign_jobs_to_companies(cluster_labels, job_ids, company_to_cluster_mapping):
    # Create a dictionary to store the recommended jobs' IDs for each company
    company_job_ids = {}

    # Iterate through the cluster labels and job order IDs arrays
    for cluster_label, job_id in zip(cluster_labels, job_ids):
        # Get the company ID based on the cluster label using the provided dictionary
        company_id = [company_id for company_id, cluster in company_to_cluster_mapping.items() if cluster == cluster_label][0]

        # Add the job order ID to the list of job order IDs for the corresponding company
        if company_id in company_job_ids:
            company_job_ids[company_id].append(job_id)
        else:
            company_job_ids[company_id] = [job_id]

    return company_job_ids

# Function to get common features between two jobs
def get_common_features(job1_index, job2_index):
    return features_adjacency_matrix[job1_index, job2_index]

# Function to plot and display all clusters with it's nodes and their features relationship
def display_all_clusters_nodes_features_relationship(company_datapoints, job_datapoints):
    # Store the jobs in their respective predicted cluster
    clusters_dict = {}

    orderid_array = create_orderid_array(company_datapoints, job_datapoints)

    for i, cluster_num in enumerate(predicted_jobs_clusters):
        cluster_key = 'Cluster ' + str(cluster_num)
        job = "OrderID " + str(orderid_array[i])
        
        if cluster_key not in clusters_dict:
            clusters_dict[cluster_key] = []
        
        clusters_dict[cluster_key].append(job)

    # Create an empty graph
    graph = nx.Graph()

    # Iterate over the clusters dictionary, adding nodes and edges to the graph
    for cluster, jobs in clusters_dict.items():
        # Add nodes (jobs) to the graph
        graph.add_nodes_from(jobs)

        # Add edges between jobs that share common features
        for j, job1 in enumerate(jobs):
            for job2 in jobs[j+1:]:
                job1_index = ""
                job2_index = ""
                for i, orderid in enumerate(orderid_array):
                    if orderid == int(job1[8:]):
                        job1_index = i
                    if orderid == int(job2[8:]):
                        job2_index = i
                    if job1_index != "" and job2_index != "":
                        # Retrieve common features between jobs (e.g., CompanyID, Frozen, Capacity)
                        common_features = get_common_features(job1_index, job2_index)
                        # Only add edge if there is common features between the jobs
                        if common_features != None:
                            # Add an edge along with the common attributes
                            graph.add_edge(job1, job2, attributes=common_features)
                        break

    # Visualization of the graph
    pos = nx.spring_layout(graph, k=0.3)
    plt.figure(figsize=(300, 100))
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold', node_size=200, width=1, font_size=5)
    edge_labels = nx.get_edge_attributes(graph, 'attributes')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=5)
    plt.title("Delivery Jobs Clustering Features Relationship Graph")
    plt.tight_layout()
    plt.show()

# Function to plot and display a specific cluster with it's nodes and their features relationship
def display_specific_clusters_nodes_features_relationship(company_datapoints, job_datapoints, cluster):
    # Store the jobs of the specific cluster in a list
    cluster_list = []

    orderid_array = create_orderid_array(company_datapoints, job_datapoints)

    for i, cluster_num in enumerate(predicted_jobs_clusters):
        if cluster_num == cluster:
            job = "OrderID " + str(orderid_array[i])
            cluster_list.append(job)

    # Create an empty graph
    graph = nx.Graph()

    # Add nodes (jobs) to the graph
    graph.add_nodes_from(cluster_list)

    # Add edges between jobs that share common features
    for j, job1 in enumerate(cluster_list):
        for job2 in cluster_list[j+1:]:
            job1_index = ""
            job2_index = ""
            for i, orderid in enumerate(orderid_array):
                if orderid == int(job1[8:]):
                    job1_index = i
                if orderid == int(job2[8:]):
                    job2_index = i
                if job1_index != "" and job2_index != "":
                    # Retrieve common features between jobs (e.g., CompanyID, Frozen, Capacity)
                    common_features = get_common_features(job1_index, job2_index)
                    # Only add edge if there is common features between the jobs
                    if common_features != None:
                        # Add an edge along with the common attributes
                        graph.add_edge(job1, job2, attributes=common_features)
                    break

    # Visualization of the graph
    pos = nx.spring_layout(graph, k=150)
    plt.figure(figsize=(300, 100))
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', font_weight='bold', node_size=200, width=1, font_size=5)
    edge_labels = nx.get_edge_attributes(graph, 'attributes')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=5)
    title = "Cluster " + str(cluster) + " Features Relationship Graph"
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------- Load & Clean data -------------------------------------------------------

allcompanyjobsdf = pd.read_csv('./CSV_Files/Company-Jobs_2022-07-26.csv')
allcompanyjobsdf = allcompanyjobsdf.drop(columns=['Job Price'])

listofallcompanyjobs = allcompanyjobsdf.to_dict('records')

for job in listofallcompanyjobs:
    if len(str(job['Pickup Postal Code'])) < 6:
        job['Pickup Postal Code'] = str("0") + str(job['Pickup Postal Code'])
    lat, long = getcoordinates(str(job['Pickup Postal Code']))
    job['Pickup Lat Long'] = str(lat) + ", " + str(long)

    if len(str(job['Delivery Postal Code'])) < 6:
        job['Delivery Postal Code'] = str("0") + str(job['Delivery Postal Code'])
    lat, long = getcoordinates(str(job['Delivery Postal Code']))
    job['Delivery Lat Long'] = str(lat) + ", " + str(long)

allcompanyjobsdf = pd.DataFrame.from_dict(listofallcompanyjobs)

alljobtradingjobsdf = pd.read_csv('./CSV_Files/100-Job-Trading-Jobs_2022-07-26.csv')
alljobtradingjobsdf = alljobtradingjobsdf.drop(columns=['Job Price'])
# Get the first N number of rows of the DataFrame
alljobtradingjobsdf = alljobtradingjobsdf.iloc[:100]

listofalljobtradingjobs = alljobtradingjobsdf.to_dict('records')

for job in listofalljobtradingjobs:
    if len(str(job['Pickup Postal Code'])) < 6:
        job['Pickup Postal Code'] = str("0") + str(job['Pickup Postal Code'])
    lat, long = getcoordinates(str(job['Pickup Postal Code']))
    job['Pickup Lat Long'] = str(lat) + ", " + str(long)

    if len(str(job['Delivery Postal Code'])) < 6:
        job['Delivery Postal Code'] = str("0") + str(job['Delivery Postal Code'])
    lat, long = getcoordinates(str(job['Delivery Postal Code']))
    job['Delivery Lat Long'] = str(lat) + ", " + str(long)

alljobtradingjobsdf = pd.DataFrame.from_dict(listofalljobtradingjobs)

# ---------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------- Initial Data Visualisation --------------------------------------------------

# https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972

street_map = gpd.read_file('./Shape_Files/singapore-shapefile')

crs = {'init':'epsg:4326'}

# Split the Pickup Lat Long and Delivery Lat Long columns into separate Latitude and Longitude columns
allcompanyjobsdf[['Pickup Latitude', 'Pickup Longitude']] = allcompanyjobsdf['Pickup Lat Long'].str.split(', ', expand=True).astype(float)
allcompanyjobsdf[['Delivery Latitude', 'Delivery Longitude']] = allcompanyjobsdf['Delivery Lat Long'].str.split(', ', expand=True).astype(float)
alljobtradingjobsdf[['Pickup Latitude', 'Pickup Longitude']] = alljobtradingjobsdf['Pickup Lat Long'].str.split(', ', expand=True).astype(float)
alljobtradingjobsdf[['Delivery Latitude', 'Delivery Longitude']] = alljobtradingjobsdf['Delivery Lat Long'].str.split(', ', expand=True).astype(float)

# Create Point objects for Pickup and Delivery locations
allcompanyjobsdf['Pickup Point'] = allcompanyjobsdf.apply(lambda row: Point(row['Pickup Longitude'], row['Pickup Latitude']), axis=1)
allcompanyjobsdf['Delivery Point'] = allcompanyjobsdf.apply(lambda row: Point(row['Delivery Longitude'], row['Delivery Latitude']), axis=1)
alljobtradingjobsdf['Pickup Point'] = alljobtradingjobsdf.apply(lambda row: Point(row['Pickup Longitude'], row['Pickup Latitude']), axis=1)
alljobtradingjobsdf['Delivery Point'] = alljobtradingjobsdf.apply(lambda row: Point(row['Delivery Longitude'], row['Delivery Latitude']), axis=1)

# Create a GeoDataFrame from the DataFrame with geometry columns
geometry = [MultiPoint([(row['Pickup Point'].x, row['Pickup Point'].y), (row['Delivery Point'].x, row['Delivery Point'].y)]) for _, row in allcompanyjobsdf.iterrows()]
allcompanyjobsgdf = gpd.GeoDataFrame(allcompanyjobsdf, #specify our data
                                     crs=crs, #specify our coordinate reference system
                                     geometry=geometry)
geometry = [MultiPoint([(row['Pickup Point'].x, row['Pickup Point'].y), (row['Delivery Point'].x, row['Delivery Point'].y)]) for _, row in alljobtradingjobsdf.iterrows()]
alljobtradingjobsgdf = gpd.GeoDataFrame(alljobtradingjobsdf, #specify our data
                                     crs=crs, #specify our coordinate reference system
                                     geometry=geometry)

fig, ax = plt.subplots(figsize=(15,15))
street_map.plot(ax=ax, alpha=0.4, color='grey')

# Generate a colormap with a sufficient number of distinct colors
num_colors = len(allcompanyjobsgdf['Company Id'].unique())
cmap = plt.get_cmap('tab20', num_colors)

# Create a dictionary to map company IDs to colors
company_id_colors = {}
for i, company_id in enumerate(allcompanyjobsgdf['Company Id'].unique()):
    company_id_colors[company_id] = cmap(i)

# Plot all companies' jobs data using distinct colors for each company ID
for company_id, color in company_id_colors.items():
    company_jobs = allcompanyjobsgdf[allcompanyjobsgdf['Company Id'] == company_id]
    company_jobs.plot(ax=ax, markersize=20, color=color, marker='o', label=f'Company {company_id}')

# Plot all job trading jobs data
alljobtradingjobsgdf.plot(ax=ax, markersize=20, color='black', marker='^', label='Public Jobs')

# Add legend and show the plot
plt.legend(prop={'size': 10}, loc='lower right')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------ Running Clustering -------------------------------------------------------

# Features Adjacency Matrix:
datapoints_num = len(allcompanyjobsdf) + len(alljobtradingjobsdf)
features_adjacency_matrix = np.empty((datapoints_num, datapoints_num), dtype=object)

# Frozen Adjacency Matrix:
frozen_adjacency_matrix = create_frozen_adjacency_matrix(allcompanyjobsdf, alljobtradingjobsdf, features_adjacency_matrix)

# Capacity Adjacency Matrix:
capacity_adjacency_matrix = create_capacity_adjacency_matrix(allcompanyjobsdf, alljobtradingjobsdf, features_adjacency_matrix)

# Distance Adjacency Matrix:
distance_adjacency_matrix = create_distance_adjacency_matrix(allcompanyjobsdf, alljobtradingjobsdf)

# Composite Adjacency Matrix:
composite_adjacency_matrix = np.mean([frozen_adjacency_matrix, capacity_adjacency_matrix, distance_adjacency_matrix], axis=0)

# Perform spectral clustering
n_clusters = len(np.unique(allcompanyjobsdf['Company Id']))
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                n_init=10, random_state=0)
clustering.fit(composite_adjacency_matrix)

# Get the predicted cluster labels
predicted_jobs_clusters = clustering.labels_

companyid_array = create_companyid_array(allcompanyjobsdf, alljobtradingjobsdf)

# # Display the predicted jobs clusters and the company IDs the job belongs to
# for i in range(len(predicted_jobs_clusters)):
#     print(f"Job {i+1}: Predicted Cluster {predicted_jobs_clusters[i]}, Company ID {companyid_array[i]}")

# Create a dictionary to store the mapping between company IDs and cluster labels
company_to_cluster_mapping = {}
is_private_jobs_retained = check_matching_clusters(predicted_jobs_clusters[:len(allcompanyjobsdf)], companyid_array[:len(allcompanyjobsdf)], company_to_cluster_mapping)
if is_private_jobs_retained == True:
    print("\nDelivery Job Matching Successful")
    orderid_array = create_orderid_array(allcompanyjobsdf, alljobtradingjobsdf)
    recommended_jobs_by_company = assign_jobs_to_companies(predicted_jobs_clusters[len(allcompanyjobsdf):], orderid_array, company_to_cluster_mapping)
    for company in sorted(recommended_jobs_by_company.keys()):
        jobs = recommended_jobs_by_company[company]
        print(f"Recommended jobs for Company ID {company}: {jobs}")
else:
    print("\nDelivery Job Matching Unsuccessful")

# ---------------------------------------------------------------------------------------------------------------------------------



# ---------------------------------------------- After Clustering Data Visualisation ----------------------------------------------

# Display features relationship graphs
display_all_clusters_nodes_features_relationship(allcompanyjobsdf, alljobtradingjobsdf)
display_specific_clusters_nodes_features_relationship(allcompanyjobsdf, alljobtradingjobsdf, 4)

# Visualise clustering results
# https://towardsdatascience.com/geopandas-101-plot-any-data-with-a-latitude-and-longitude-on-a-map-98e01944b972

street_map = gpd.read_file('./Shape_Files/singapore-shapefile')

crs = {'init':'epsg:4326'}

# Concatenate the allcompanyjobsgdf and alljobtradingjobsgdf GeoPandas DataFrames
alljobsgdf = pd.concat([allcompanyjobsgdf, alljobtradingjobsgdf], ignore_index=True)

# Reset the index of the combined GeoDataFrame
alljobsgdf.reset_index(drop=True, inplace=True)

# Add the jobs clustering results to the combined GeoDataFrame
alljobsgdf['Job Cluster'] = predicted_jobs_clusters

fig, ax = plt.subplots(figsize=(15,15))
street_map.plot(ax=ax, alpha=0.4, color='grey')

# Generate a colormap with a sufficient number of distinct colors
num_colors = len(alljobsgdf['Job Cluster'].unique())
cmap = plt.get_cmap('tab20', num_colors)

# Create a dictionary to map clusters to colors
cluster_colors = {}
for i, cluster in enumerate(alljobsgdf['Job Cluster'].unique()):
    cluster_colors[cluster] = cmap(i)

# Plot all jobs data using distinct colors for each cluster
for cluster, color in cluster_colors.items():
    cluster_jobs = alljobsgdf[alljobsgdf['Job Cluster'] == cluster]
    cluster_jobs.plot(ax=ax, markersize=20, color=color, marker='o', label=f'Cluster {cluster}')

# Add legend and show the plot
plt.legend(prop={'size': 10}, loc='lower right')
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------
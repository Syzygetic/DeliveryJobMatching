import requests
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from scipy.stats import entropy

# ----------------------------------------------------------- Functions -----------------------------------------------------------

# Function to retrieve latitude, longitude using postal code
def getcoordinates(address):
    req = requests.get('https://developers.onemap.sg/commonapi/search?searchVal='+address+'&returnGeom=Y&getAddrDetails=Y&pageNum=1')
    resultsdict = eval(req.text)
    if len(resultsdict['results'])>0:
        return resultsdict['results'][0]['LATITUDE'], resultsdict['results'][0]['LONGITUDE']
    else:
        pass

# Function to initiate the centroids
def initiate_centroids(k, dset):
    '''
    Select k data points as centroids
    k: number of centroids
    dset: pandas dataframe
    '''
    centroids = dset.sample(k)
    return centroids

# Function to calculate distance (Mean Squared Error)
def rsserr(a,b):
    '''
    Calculate the root of sum of squared errors. 
    a and b are numpy arrays
    '''
    return np.square(np.sum((a-b)**2))

# Function to assign centroids
def centroid_assignation(dset, centroids):
    '''
    Given a dataframe `dset` and a set of `centroids`, we assign each
    data point in `dset` to a centroid. 
    - dset - pandas dataframe with observations
    - centroids - pandas dataframe with centroids
    '''
    k = centroids.shape[0]
    n = dset.shape[0]
    assignation = []
    assign_errors = []

    for obs in range(n):
        # Estimate error
        all_errors = np.array([])
        for centroid in range(k):
            err = rsserr(centroids.iloc[centroid,:], dset.iloc[obs,:])
            all_errors = np.append(all_errors, err)

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors

# Function to create an array to store all job's company id (Jobs that does not belong to any company will be temporarily assigned company id 0)
def create_companyid_array(company_datapoints, job_datapoints):
    datapoints_num = len(company_datapoints) + len(job_datapoints)
    companyid_array = np.zeros(datapoints_num, dtype=int)

    for i in range(len(company_datapoints)):
        companyid_array[i] = company_datapoints['Company Id'][i]

    return companyid_array

def cluster_entropy(cluster_labels):
    cluster_counts = np.bincount(cluster_labels)
    cluster_probs = cluster_counts / len(cluster_labels)
    return entropy(cluster_probs)

# ---------------------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------- Load & Clean data -------------------------------------------------------

# Getting Started (Data Preparations)
start = time.time()
allcompanyjobs = []
newallcompanyjobsdf = []

alljobtradingjobs = []
newalljobtradingjobsdf = []

allcompanytotalerror = {}

allcompanyjobsdf = pd.read_csv('../CSV_Files/Company-Jobs_2022-07-26.csv')
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

alljobtradingjobsdf = pd.read_csv('../CSV_Files/100-Job-Trading-Jobs_2022-07-26.csv')
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



# ------------------------------------------------------ Running Clustering -------------------------------------------------------

# Split the Pickup Lat Long and Delivery Lat Long columns into separate Latitude and Longitude columns
allcompanyjobsdf[['Pickup Latitude', 'Pickup Longitude']] = allcompanyjobsdf['Pickup Lat Long'].str.split(', ', expand=True).astype(float)
allcompanyjobsdf[['Delivery Latitude', 'Delivery Longitude']] = allcompanyjobsdf['Delivery Lat Long'].str.split(', ', expand=True).astype(float)
alljobtradingjobsdf[['Pickup Latitude', 'Pickup Longitude']] = alljobtradingjobsdf['Pickup Lat Long'].str.split(', ', expand=True).astype(float)
alljobtradingjobsdf[['Delivery Latitude', 'Delivery Longitude']] = alljobtradingjobsdf['Delivery Lat Long'].str.split(', ', expand=True).astype(float)

# Group the dataframe by "Company Id"
grouped = allcompanyjobsdf.groupby("Company Id")

# Initialize an empty list to store the results
allcompanyjobs = []

# Iterate over the groups
for company_id, group in grouped:
    # Retrieve the latitude and longitude values for the current group
    frozen = []
    capacity = []
    pickups = []
    deliveries = []

    
    # Iterate over each row in the current group
    for index, row in group.iterrows():
        # Append the values to the respective lists
        if row["Frozen"] == "Yes":
            frozen.append(1)
        else:
            frozen.append(0)

        if row["Capacity"] == "Document":
            capacity.append(0)
        elif row["Capacity"] == "<60 cm (H+L+W) & max 1kg":
            capacity.append(1)
        elif row["Capacity"] == "<80 cm (H+L+W) & max 5kg":
            capacity.append(2)
        elif row["Capacity"] == "<100 cm (H+L+W) & max 8kg":
            capacity.append(3)
        elif row["Capacity"] == "<120 cm (H+L+W) & max 10kg":
            capacity.append(4)
        elif row["Capacity"] == "<140 cm (H+L+W) & max 15kg":
            capacity.append(5)
        elif row["Capacity"] == "<160 cm (H+L+W) & max 20kg":
            capacity.append(6)
        elif row["Capacity"] == "<200 cm (H+L+W) & max 25kg":
            capacity.append(7)
        elif row["Capacity"] == "1.7m Van":
            capacity.append(8)
        elif row["Capacity"] == "2.4m Van":
            capacity.append(9)
        elif row["Capacity"] == "10Ft Lorry":
            capacity.append(10)
        elif row["Capacity"] == "14Ft Lorry":
            capacity.append(11)

        pickups.append((row["Pickup Latitude"] + row["Pickup Longitude"]))
        deliveries.append((row["Delivery Latitude"] + row["Delivery Longitude"]))
    
    # Create a dictionary for the current company and append it to the list
    company_data = {
        "frozen": frozen,
        "capacity": capacity,
        "pickup": pickups,
        "delivery": deliveries
    }
    allcompanyjobs.append(company_data)

for onecompanyjobs in allcompanyjobs:
    newallcompanyjobsdf.append(pd.DataFrame(onecompanyjobs))

# Initialize an empty list to store the results
alljobtradingjobs = []

for i in range(len(alljobtradingjobsdf)):
    # Retrieve the latitude and longitude values for the current group
    frozen = []
    capacity = []
    pickups = []
    deliveries = []

    # Append the values to the respective lists
    if alljobtradingjobsdf["Frozen"][i] == "Yes":
        frozen.append(1)
    else:
        frozen.append(0)

    if alljobtradingjobsdf["Capacity"][i] == "Document":
        capacity.append(0)
    elif alljobtradingjobsdf["Capacity"][i] == "<60 cm (H+L+W) & max 1kg":
        capacity.append(1)
    elif alljobtradingjobsdf["Capacity"][i] == "<80 cm (H+L+W) & max 5kg":
        capacity.append(2)
    elif alljobtradingjobsdf["Capacity"][i] == "<100 cm (H+L+W) & max 8kg":
        capacity.append(3)
    elif alljobtradingjobsdf["Capacity"][i] == "<120 cm (H+L+W) & max 10kg":
        capacity.append(4)
    elif alljobtradingjobsdf["Capacity"][i] == "<140 cm (H+L+W) & max 15kg":
        capacity.append(5)
    elif alljobtradingjobsdf["Capacity"][i] == "<160 cm (H+L+W) & max 20kg":
        capacity.append(6)
    elif alljobtradingjobsdf["Capacity"][i] == "<200 cm (H+L+W) & max 25kg":
        capacity.append(7)
    elif alljobtradingjobsdf["Capacity"][i] == "1.7m Van":
        capacity.append(8)
    elif alljobtradingjobsdf["Capacity"][i] == "2.4m Van":
        capacity.append(9)
    elif alljobtradingjobsdf["Capacity"][i] == "10Ft Lorry":
        capacity.append(10)
    elif alljobtradingjobsdf["Capacity"][i] == "14Ft Lorry":
        capacity.append(11)

    pickups.append((alljobtradingjobsdf["Pickup Latitude"][i] + alljobtradingjobsdf["Pickup Longitude"][i]))
    deliveries.append((alljobtradingjobsdf["Delivery Latitude"][i] + alljobtradingjobsdf["Delivery Longitude"][i]))

    # Create a dictionary for the current job and append it to the list
    job_data = {
        "frozen": frozen,
        "capacity": capacity,
        "pickup": pickups,
        "delivery": deliveries
    }
    alljobtradingjobs.append(job_data)

for onejobtradingjob in alljobtradingjobs:
    newalljobtradingjobsdf.append(pd.DataFrame(onejobtradingjob))

allcompanyids = allcompanyjobsdf['Company Id'].unique()
predicted_jobs_clusters = []

# For each Job Trading Job
for onejobtradingjobdf in newalljobtradingjobsdf:
    # For each Company:
    companycount = 0
    for onecompanyjobsdf in newallcompanyjobsdf:
        temponejobtradingjobdf = onejobtradingjobdf

        # Steps 1 and 2 - Define K and initiate the centroids
        np.random.seed(42)
        # Set k to the total number of pickup + dropoff points a company holds
        k=len(onecompanyjobsdf['frozen'])
        # Set all the pickup and dropoff points a company holds to be centroids
        centroids = initiate_centroids(k, onecompanyjobsdf)

        # Step 3 - Temporarily assign the current Job Trading Job's Pickup & Dropoff Points 
        # to the nearest centroids (Pickup/Dropoff points in the current company's list of jobs)
        # then calculating the total error/distance between the Job Trading Job's points and the Company Jobs' points
        # Allowing us to find the least total error/distance from the company existing jobs' points to the Job Trading Job's points 
        # if we were to assign the Job Trading Job to the current company
        temponejobtradingjobdf['centroid'], temponejobtradingjobdf['error'] = centroid_assignation(temponejobtradingjobdf, centroids)

        companytotalerror = temponejobtradingjobdf['error'].sum()
        # print("The total error is {}".format(companytotalerror))

        # Save the company ID as key along with the company total error as value in a dictionary
        allcompanytotalerror[allcompanyids[companycount]] = companytotalerror

        companycount += 1

    # Find the least total error/distance among all the companies for the Job Trading Job
    # mincompanytotalerror = min(allcompanytotalerror.values())
    # print("\nLEAST TOTAL ERROR AMONG ALL COMPANIES:")
    # print(mincompanytotalerror)
    # Find the company with the least total error/distance and recommend the Job Trading Job to them
    recommendjobtocompanyids = min(allcompanytotalerror, key=allcompanytotalerror.get)
    # print("\nRECOMMEND JOB TRADING JOB TO COMPANY IDS:")
    # print(str(recommendjobtocompanyids) + "\n")
    predicted_jobs_clusters.append(recommendjobtocompanyids)

companyid_array = create_companyid_array(allcompanyjobsdf, alljobtradingjobsdf)

# Display the predicted jobs clusters and the company IDs the job belongs to
for i in range(len(predicted_jobs_clusters)):
    print(f"Job {i+1}: Predicted Cluster {predicted_jobs_clusters[i]}, Company ID {(companyid_array[len(allcompanyjobsdf):])[i]}")

kmeans_entropy = cluster_entropy(predicted_jobs_clusters)

print("K-means Cluster Entropy:", kmeans_entropy)
end = time.time()
print("K-means Run Time:", end - start)

# ---------------------------------------------------------------------------------------------------------------------------------
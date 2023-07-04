# DeliveryJobMatching
This repository serves as a showcase for my capstone project, which revolves around leveraging Spectral Clustering to recommend and match publicly available delivery jobs that have not been assigned to any specific company. The goal is to assign these jobs to compatible and suitable companies based on their current portfolio of delivery jobs.

## Steps to follow to get started
1) Create an virtual environment, and run "pip install -r requirements.txt" to download required dependencies
2) Refer to the sample CSV files in the "CSV_Files" folder and if required prepare your own CSV files to use for clustering by the "deliveryjobmatching.py" program. One CSV file to store the jobs with company labels and another CSV file to store the jobs that requires matching/assignment to a company. 
3) If you have prepared your own CSV files in Step 2 to use your CSV files do the following, in "deliveryjobmatching.py" change line 327 to the name of your CSV file storing the jobs with company labels and change line 345 to the name of your CSV file storing the jobs that requires matching/assignment to a company.
4) If you require to change the shape file, you can search for resources online and change lines 373 and 473 to the name of your Shape files folder.
5) Last but not least, just run "deliveryjobmatching.py" to see the clustering results!

# DeliveryJobMatching
This repository serves as a showcase for my capstone project, which revolves around leveraging Spectral Clustering to recommend and match publicly available delivery jobs that have not been assigned to any specific company. The goal is to assign these jobs to compatible and suitable companies based on their current portfolio of delivery jobs.

## Steps to follow to get started
1) Create an virtual environment, and run "pip install -r requirements.txt" to download required dependencies
2) Both code files, "delivery_job_matching.py" and "delivery_job_matching_no_visualization.py," perform the same task. The only distinction is that "delivery_job_matching_no_visualization.py" omits visualizations for easier testing. Throughout this document, we will refer to the file "delivery_job_matching.py" for simplicity, but it is important to note that both files are interchangeable.
3) Refer to the sample CSV files in the "CSV_Files" folder and if required prepare your own CSV files to use for clustering by the "delivery_job_matching.py" program. One CSV file to store the jobs with company labels and another CSV file to store the jobs that requires matching/assignment to a company. 
4) If you have prepared your own CSV files in Step 2 to use your CSV files do the following, in "delivery_job_matching.py" change line 327 to the name of your CSV file storing the jobs with company labels and change line 345 to the name of your CSV file storing the jobs that requires matching/assignment to a company.
5) If you require to change the shape file, you can search for resources online and change lines 373 and 473 to the name of your Shape files folder.
6) Last but not least, just run "delivery_job_matching.py" to see the clustering results!

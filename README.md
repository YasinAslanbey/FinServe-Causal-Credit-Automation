FinServe: Automated Credit Risk & Causal Intervention System
This project is a Proof-of-Concept (PoC) designed for FinServe to modernize and automate their manual credit assessment pipeline. It transitions the workflow from manual data entry to an event-driven automation system that utilizes both Machine Learning and Causal AI to not only predict risk but also provide strategic mitigation advice.

System Architecture
The automation is built on a "Monitor-Predict-Intervene" logic:

Monitoring (Otomation.py): The system uses the watchdog library to perform real-time monitoring of the Json_Input folder. The moment a structured JSON file (loan application) is dropped into this directory, the entire pipeline is triggered automatically.

Risk Prediction & Causal Analysis (ML_CASUAL.py): This is the core engine of the project. It integrates two distinct AI approaches:

Machine Learning: A Random Forest model evaluates the applicant's data against historical records to predict default probability.

Causal AI (DoWhy): This is the most critical part of the project. While standard ML only finds correlations, the Causal AI engine performs "interventions." It mathematically calculates how changing specific loan parameters (like reducing the loan amount or shortening the term) would actually cause the risk level to drop. This allows FinServe to offer alternative terms to "Manual Review" candidates instead of a flat rejection.

Data Structure
Company_Database/full_credit_database.csv: This is the primary dataset used for training the model. It represents FinServe’s historical credit records.

Company_Database/processed_applications.csv: The automated output of the system. Every application processed via the monitoring tool is logged here with its final decision and the calculated causal strategies.

Json_Input/: The designated landing zone for new application files in JSON format.

Setup and Execution
Environment: Ensure Python 3.10+ is installed.

Dependencies: Install the required libraries via pip:

Bash
pip install pandas numpy scikit-learn dowhy watchdog
Running the System: Execute the main automation script:

Bash
python Otomation.py
Testing: Place a valid application JSON into the Json_Input folder. The system will log the analysis in the terminal and save the results to the processed_applications.csv file.

Why Causal AI?
In financial services, knowing that a customer is high-risk is not enough. FinServe needs to know why and how to make them low-risk. By implementing Causal Inference, this system provides actionable intelligence—telling the credit officer exactly how much the loan amount needs to be adjusted to bring the risk within acceptable limits. This directly reduces customer churn and improves the approval rate without increasing default risk.

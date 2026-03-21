# FINSERVE: AUTOMATED CREDIT RISK & CAUSAL INTERVENTION SYSTEM
## 1. PROJECT PURPOSE & PROBLEM STATEMENT
At FinServe, the current credit assessment process is entirely manual, leading to significant bottlenecks, high operational costs, and human error. This Proof-of-Concept (PoC) transitions FinServe from a slow, manual workflow to a robust, AI-driven decision engine.

By leveraging a Machine Learning model trained on the company’s historical data (full_credit_database.csv) and integrating Causal AI, this system provides a high-integrity decision mechanism.

### KEY OBJECTIVES:
Automation of Low-Risk Approvals: Automatically identify and approve low-risk applicants, significantly speeding up the registration and processing timeline.

Intelligent Triage: By filtering out clear approvals and rejections, the system ensures that human credit officers only focus on "Manual Reviews" for complex, borderline cases.

Strategic Risk Mitigation: For applicants flagged as risky, the Causal AI engine suggests specific adjustments (interventions) to make the loan viable, reducing rejection rates without increasing default risk.

[!IMPORTANT]
PATH CONFIGURATION: Before running the scripts, ensure that the file paths in Otomation.py and ML_CASUAL.py are updated to match your local directory structure. The system relies on absolute or correct relative paths to monitor folders and access datasets.

## 2. SYSTEM ARCHITECTURE
The automation is built on a "Monitor-Predict-Intervene" logic:

MONITORING (Otomation.py): The system uses the watchdog library for real-time monitoring of the Json_input folder. When a structured JSON file (loan application) is detected, the entire pipeline triggers automatically.

RISK PREDICTION & CAUSAL ANALYSIS (ML_CASUAL.py): This is the core engine integrating two AI approaches:

MACHINE LEARNING: A Random Forest model evaluates applicant data against historical records to predict default probability.

CAUSAL AI (DOWHY): This is the critical differentiator. While standard ML finds correlations, the Causal AI engine performs "interventions." It mathematically calculates how changing specific loan parameters (like reducing the amount or shortening the term) causes the risk level to drop.

## 3. DATA STRUCTURE
Company_Database/full_credit_database.csv: Primary dataset used for model training, representing historical credit records.

Company_Database/processed_applications.csv: Automated output log. Every application processed via the monitoring tool is saved here with its final decision and causal strategies.

Json_input/: Designated landing zone for new application files in JSON format.

## 4. SETUP AND EXECUTION
1. ENVIRONMENT: Ensure Python 3.10+ is installed.

2. DEPENDENCIES: Install required libraries via pip:
pip install pandas numpy scikit-learn dowhy watchdog


3. PATH SETUP: Open Otomation.py and verify that the WATCH_DIRECTORY and other file paths point to your local project folders.

4. RUNNING THE SYSTEM: Execute the main automation script:
python Otomation.py

5. TESTING: Place a valid application JSON into the Json_input folder. The system will log the analysis in the terminal and update processed_applications.csv.

## 5. WHY CAUSAL AI?
In financial services, knowing that a customer is high-risk is not enough. FinServe needs to know why and how to make them low-risk. By implementing Causal Inference, this system provides actionable intelligence—telling the credit officer exactly how much a loan amount needs to be adjusted to bring risk within acceptable limits. This reduces customer churn and improves approval rates without increasing default risk.
TESTING: Place a valid application JSON into the Json_input folder. The system will log the analysis in the terminal and update processed_applications.csv.

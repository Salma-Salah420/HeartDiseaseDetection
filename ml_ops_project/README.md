## MLFLOW PROJECT
``` bash
# Get conda channels
conda config --show channels

# Build a MLFlow project, if you use one entry point with name (main)
mlflow run . --experiment-name <exp-name> # here it is {chrun-detection}

# If you have multiple entry points
mlflow run -e forest . --experiment-name churn-detection
mlflow run -e logistic . --experiment-name churn-detection
mlflow run -e xgboost . --experiment-name churn-detection

# If you want some params instead of default values

mlflow run -e logistic . --experiment-name churn-detection -P c=3.5 -P p="l2"
mlflow run -e xgboost . --experiment-name churn-detection -P n=250 -P lr=0.15 -P d=22

```

```
## MLFLOW Models
``` bash
# serve the model via REST
mlflow models serve -m "path" --port 8000 --env-manager=local

mlflow models serve -m "file:///E:/courses/ai_depi_round_3/technical/project/final_project/ml_ops/mlruns/194489145900410023/models/m-9cd419fc238646248d7d87bf154a7713/artifacts" --port 8000 --env-manager=local


# it will open in this link
http://localhost:8000/invocations
```

``` python
# exmaple of data to be sent


## multiple samples
{
  "dataframe_split": {
    "columns": [
      "Unnamed: 0",
      "BMI",
      "Smoking",
      "AlcoholDrinking",
      "Stroke",
      "PhysicalHealth",
      "MentalHealth",
      "DiffWalking",
      "Sex",
      "AgeCategory",
      "PhysicalActivity",
      "GenHealth",
      "SleepTime",
      "Asthma",
      "KidneyDisease",
      "SkinCancer",
      "Race_Asian",
      "Race_Black",
      "Race_Hispanic",
      "Race_Other",
      "Race_White",
      "Diabetic_No, borderline diabetes",
      "Diabetic_Yes",
      "Diabetic_Yes (during pregnancy)"
    ],
    "data": [
      [
        0,           // Unnamed: 0
        28.5,        // BMI
        0,           // Smoking
        0,           // AlcoholDrinking
        0,           // Stroke
        2,           // PhysicalHealth
        5,           // MentalHealth
        0,           // DiffWalking
        1,           // Sex (1=Male, 0=Female or vice versa)
        7,           // AgeCategory (encoded number if it was encoded)
        1,           // PhysicalActivity
        3,           // GenHealth (numeric encoding of e.g. 'Good')
        7,           // SleepTime
        0,           // Asthma
        0,           // KidneyDisease
        0,           // SkinCancer
        0,           // Race_Asian
        0,           // Race_Black
        0,           // Race_Hispanic
        0,           // Race_Other
        1,           // Race_White
        1,           // Diabetic_No, borderline diabetes
        0,           // Diabetic_Yes
        0            // Diabetic_Yes (during pregnancy)
      ]
    ]
  }
}


```

``` bash 
# if you want to use curl

curl -X POST \
  http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
        "columns": [
            "Age",
            "CreditScore",
            "Balance",
            "EstimatedSalary",
            "Gender_Male",
            "Geography_Germany",
            "Geography_Spain",
            "HasCrCard",
            "Tenure",
            "IsActiveMember",
            "NumOfProducts"
        ],
        "data": [
            [-0.7541830079917924, 0.5780143566720919, 0.11375998165198585, -0.14673040749854463, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0],
            [-0.5605884106597949, 0.753908347743766, 0.7003528882054108, 1.6923927520037099, 0.0, 1.0, 0.0, 1.0, 9.0, 1.0, 1.0],
            [0.11699268000219652, -0.3221490094005933, 0.5222180917013974, -0.8721429873346316, 1.0, 1.0, 0.0, 1.0, 5.0, 0.0, 2.0],
            [0.6977764719981892, -0.7256705183297281, -1.2170740485175422, 0.07677206232885857, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0]
        ]
    }
}'


# if you want to use Powershell
Invoke-RestMethod -Uri "http://localhost:8000/invocations" -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{
    "dataframe_split": {
        "columns": [
            "Age",
            "CreditScore",
            "Balance",
            "EstimatedSalary",
            "Gender_Male",
            "Geography_Germany",
            "Geography_Spain",
            "HasCrCard",
            "Tenure",
            "IsActiveMember",
            "NumOfProducts"
        ],
        "data": [
            [-0.7541830079917924, 0.5780143566720919, 0.11375998165198585, -0.14673040749854463, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0],
            [-0.5605884106597949, 0.753908347743766, 0.7003528882054108, 1.6923927520037099, 0.0, 1.0, 0.0, 1.0, 9.0, 1.0, 1.0],
            [0.11699268000219652, -0.3221490094005933, 0.5222180917013974, -0.8721429873346316, 1.0, 1.0, 0.0, 1.0, 5.0, 0.0, 2.0],
            [0.6977764719981892, -0.7256705183297281, -1.2170740485175422, 0.07677206232885857, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0]
        ]
    }
}'

```

```
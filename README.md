# 10Academy-Kifiya-Week-4

### **Report: Empowering Sales Forecasting with Machine Learning, Deep Learning, and MLflow**

In today's fast-paced retail landscape, accurate sales forecasting is more critical than ever. Store managers face the challenge of making predictions based on a variety of factors, including promotions, holidays, seasonality, and locality. With large amounts of transactional data and rapid advancements in artificial intelligence, businesses have an unprecedented opportunity to leverage data-driven models to enhance decision-making. 

In this blog, I will share my journey in building a robust sales forecasting system for Rossmann Pharmaceuticals using historical sales records and store-specific data. By integrating deep learning, a comprehensive exploratory analysis, and MLflow for model management, this analysis aimed to unlock hidden insights and push the boundaries of predictive accuracy.

---

### **Overview**

The goal was to predict daily sales for over 1,000 stores operated by Rossmann Pharmaceuticals, ensuring store managers could adequately plan for inventory, staffing, and promotions. To achieve this, the project was divided into structured tasks:

1. Exploration of customer purchasing behavior (EDA).
2. Feature engineering and preprocessing for sales prediction.
3. Development of machine learning and deep learning models.
4. Deployment and model monitoring via MLflow.

This structured approach ensured modularity, maintainability, and scalability as the project evolved.

---

### **1. Exploratory Data Analysis**


# Insights from Data Analysis

In today’s competitive retail landscape, store managers often rely on their experience and judgment when making predictions about sales and customer behavior. However, leveraging data for more accurate forecasts can significantly enhance decision-making processes. This report delves into the findings from the data analysis, focusing on sales predictions, promotional strategies, and the impact of seasonal factors.

### Key Features of the Analysis

The dataset includes essential features such as:

- **Store**: Unique identifier for each store
- **Sales**: Daily turnover (Target variable)
- **StateHoliday** & **SchoolHoliday**: Indicators of public events
- **CompetitionDistance**: Proximity to competitor stores
- **Promo** & **Promo2**: Promotional campaign indicators

### Data Preparation

To ensure the accuracy of the analysis, several critical steps were undertaken:

- Removed irrelevant columns (like 'Id')
- Standardized the 'StateHoliday' data type
- Unified the date format to datetime
- Handled missing values using mean and median strategies

---

A strong foundation was laid through rigorous exploratory data analysis (EDA), where the following were examined:

1. **Seasonal Trends**:
   - **Weekly**
      - Stronger sales were observed on weekdays than on weekends.

         <img src="image-1.png" width="600" height="300">

   - **Monthly**
      - Sales data indicated a significant rise from November to July, peaking in March and late June. 
      - The summer months also showed a steady increase in sales, attributed to the vacation season. 
   - Leveraging these peak seasons with targeted marketing campaigns is recommended.

2. **Store-Type Insights**:
   - Different store types exhibited different sales.
      
      <img src="image-2.png" width="600" height="300">
   
   - Different store types also exhibited unique customer behaviors.
      
      <img src="image.png" width="600" height="300">

3. **Promotional Influence**:
   - Promotions play a vital role in driving sales and customer engagement. The analysis showed that:
      - **Significant Increase in Sales**: Daily sales surged during promotional events, with higher customer turnout.
      - **Optimized Promotions**: Tailoring promotions by store type can enhance their effectiveness.
      
         <img src="image-11.png" width="800" height="400">

   - Stores with active promotions consistently outperformed their counterparts, emphasizing the power of well-timed campaigns. This shows the impact of promotions on customer purchasing behavior.

      <img src="image-7.png" width="800" height="400">

   - Peaks in sales around specific holidays(Christmas, Independence Day, ...) showcased the impact of promotional strategies.

4. **Holiday Effect**:
   - School holidays do not seem to impact sales significantly. Sales at stores during holidays are consistent with regular days.

      <img src="image-4.png" width="800" height="400"> 

   - However, sales on state holidays are very low.

      <img src="image-5.png" width="800" height="400">

   - On the contrary, sales at specific holidays, such as Christmas and Independence Day, tend to be higher than usual, indicating a significant impact of these particular holiday on sales.

      <img src="image-9.png" width="800" height="400">

5. **Customer Behavior Insights**:

   - The findings demonstrated a strong correlation between customer traffic and sales:
      - **More Customers = Higher Sales**: Stores that attract more customers generally experience higher sales.
      - **Store Type Analysis**: Store type 'd' drives fewer customers but yields higher sales per customer. It's crucial to prioritize high-value promotions targeting these types.

6. **Competitive Landscape**:

   - Sales tend to decline with decreasing competition distance. While new competitor openings temporarily reduce sales, the impact diminishes over time.
   - Therefore, it’s essential to focus on competitive strategies and prepare marketing initiatives for anticipated competitor openings.

      <img src="image-10.png" width="800" height="400">

## Insights Gained from Data

From the observation, sales remained relatively constant over time, indicating current strategies may need revision to enhance sales performance. Notably, the analysis of state holidays showed a decline in average sales across amost all holiday types, indicating the need for tailored strategies to address these downturns.

## Challenges and Considerations

Despite the insights gained, challenges remain:

- Missing values in competitor distance and promotional intervals
- Outliers in sales and customer data


The results of EDA shaped the modeling strategy, particularly feature selection and scaling techniques, ensuring the data fed into models was meaningful and actionable.

---

### **2. Feature Engineering and Data Preprocessing**

Transforming raw data into model-ready features was a crucial step:

- **Temporal Features**:
  Extracted day, month, year, and weekday from dates to capture cyclic patterns.
- **Lag Variables**:
  Introduced lagged sales figures to incorporate short-term memory effects.
- **Derived Features**:
  - Derived features from sales, customers, and competitors indicators for better prediction accuracy.
  - Aggregated various sales insights at store and store type levels for enhanced prediction.
- **External Features**:
  Incorporated holiday data for more accurate prediction.

The preprocessing pipeline was modularized for seamless experimentation and ensured that future enhancements to the dataset required minimal changes in downstream processes.

---

### **3. Deep Learning for Sales Prediction**

While traditional models like Random Forest and Gradient Boosting provided solid baselines, a more sophisticated solution to capture long-term dependencies in sales data is demanded. Enter **Recurrent Neural Networks (RNNs)**, with a focus on **Long Short-Term Memory (LSTM)** networks.

#### **Why LSTMs?**
Sales data is sequential, with temporal dependencies that traditional models cannot fully capture. LSTMs, with their ability to learn from both short- and long-term sequences, emerged as the ideal choice.

#### **Model Architecture**
The LSTM-based model comprised:
- **Input Layer**: Temporal features and one-hot-encoded categorical data.
- **LSTM Layers**: Two stacked layers of LSTM units with dropout for regularization.
- **Dense Layers**: Fully connected layers for mapping LSTM outputs to predicted sales.
- **Output Layer**: Single neuron for regression output (daily sales).

#### **Model Compilation**
The model was compiled with the Adam optimizer, minimizing Mean Squared Error (MSE) as the loss function and tracking Mean Absolute Error (MAE) as a metric.

```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
```

#### **Training Results**
- Training Duration: 10 epochs with early stopping for generalization.
- Achieved Validation MAE: 728.34 units.
- Clear convergence of training and validation losses indicated robust learning.

---

### **4. Integrating MLflow for Experiment Tracking**

To streamline experimentation and track the lifecycle of models, MLflow was seamlessly integrated into the project:

1. **Tracking URI**: A local SQLite database served as backend, ensuring reproducibility and ease of sharing.
2. **Experiment Logging**:
   - Parameters like learning rate, batch size, and architecture type were logged.
   - Metrics such as MAE and MSE provided a quantitative measure of performance.
3. **Model Registry**:
   - The final LSTM model was registered with MLflow's model registry.
   - Automated transition to "Staging" for deployment after rigorous testing.

This integration not only improved transparency but also made scaling to more experiments straightforward.

---

### **5. Model Serving and Deployment**

Deploying a model is as crucial as training it. Leveraging MLflow's `models serve`, the LSTM model was deployed as a RESTful API. The setup allowed real-time inference with just a simple HTTP POST request. Alternatively, FastAPI + Uvicorn was also used to deploy the models.

#### **Testing the Endpoint**
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"columns": [...], "data": [...]}' \
http://127.0.0.1:5000/invocations
```

Results were served in milliseconds, paving the way for integration into production systems.

---

### **6. Observations**

#### **Key Insights**:
- Deep learning unlocked nuanced temporal patterns, outperforming traditional models.
- EDA and feature engineering were pivotal in enriching the dataset and improving model performance.
- MLflow brought a professional layer of accountability, simplifying experiment management and future reproducibility.
- Comprehensive logging of processes is also necessary for reproducibility and ongoing improvement.

---

### **Conclusion**

This project highlighted the transformative power of combining deep learning and modern software development practices. With LSTMs at the core and MLflow ensuring traceability, a reliable sales forecasting model tailored to Rossmann Pharmaceuticals' needs has been delivered. Beyond retail, this framework sets the stage for solving a myriad of sequential prediction problems in diverse domains.

In addition, the data analysis has also highlighted several key insights for optimizing store performance. By understanding customer behavior, leveraging seasonal trends, and implementing targeted promotional strategies, sales can significantly be enhanced.

The possibilities for AI in decision-making are endless, and my journey with this project is only the beginning. Stay tuned for more innovations in forecasting and model deployment!

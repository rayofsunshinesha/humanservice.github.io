## Welcome to Human Services Project!

Our project has two parts: validation of an existing measure for self-sufficiency and the creation of a model to predict self-sufficiency. 

### Background information:

In the late 1990s, the Illinois Department of Human Services, as part of welfare reform, was tasked with the goal of assististing TANF (Temporary Assistance for Needy Families) recipients to obtain employment and increased earnings to help move clients toward self-sufficiency. One way that this progress was measured was by the use of TANF Cases Cancelled Due to Earnings, a metric which recorded which cases earned suffiienct income to move them above the income eligibility threshold for TANF benefits.

Over the past 20 years, the size of the TANF population has fallen precipitously and the size of the SNAP (Supplemental Nutritional Assistance Program) has grown much larger. Recently, a similar metric was constructed for SNAP clients, SNAP Cases Cancelled Due to Earnings, also measuring when cases cancel due to an income increase above the eligibilty threshold for SNAP. One aim of this metric is to measure the attainment of self-sufficiency, where a client is able to maintain sufficient income to not require or qualify for benefits any longer. (credit to Chris Pecaut, Matt Coyne, Illinois Department of Human Services)

### Part I Evaluation:
SNAP Case Cancelled Due to Earnings (SCDTE) and assess how well it captures self-sufficiency. This has several components: comparing the wage records of the with the DHS internal measure of reported earned income, assessing whether the Cases Cancelled return to benefits after cancellation, and comparing the Cases Cancelled group with the SNAP population as a whole.

- Use spells data to look in the future return rate and speed
- Link wage records and spell data to explore income and type of jobs at the time both before and after a spell is canceled 

### Part II Machine Learning Models:
Our second part of project is to predict a risk score at the time spell cancels. With the prediction score, we can achieve two goals: one is to develop a new measure to evaluate self-efficiency; second, to provide intervention to the spells who are most at risk at the time spell cancels. 

- Predict the risk of return at a moment spell is canceled 
- Develop a new measure that substitute current measure 

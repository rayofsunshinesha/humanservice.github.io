## Welcome to Human Services Project!

Our project has two parts: validation of an existing measure for self-sufficiency and the creation of a model to predict self-sufficiency. 

### Background information:

In the late 1990s, the Illinois Department of Human Services, as part of welfare reform, was tasked with the goal of assisting TANF (Temporary Assistance for Needy Families) recipients to obtain employment and increased earnings to help move clients toward self-sufficiency. One way that this progress was measured was by using TANF Cases Cancelled Due to Earnings, a metric which recorded which cases earned sufficient income to move them above the income eligibility threshold for TANF benefits.

Over the past 20 years, the size of the TANF population has fallen precipitously and the size of the SNAP (Supplemental Nutritional Assistance Program) has grown much larger. Recently, a similar metric was constructed for SNAP clients, SNAP Cases Cancelled Due to Earnings, also measuring when cases cancel due to an income increase above the eligibility threshold for SNAP. This metric measures the attainment of self-sufficiency, indicating if a client can maintain sufficient income and does not require or no longer qualifies for benefits. (credit to Chris Pecaut, Matt Coyne, Illinois Department of Human Services)

### Part I Evaluation:
SNAP Case Cancelled Due to Earnings (SCDTE) assess how well it captures self-sufficiency. This has several components: comparing the wage records with the DHS internal measure of reported earned income, assessing whether the Cases Cancelled return to benefits after cancellation, and comparing the Cases Cancelled group with the SNAP population.

- Use spells data to look at the future return and time span
- Link wage records and spell data to explore income and the type of jobs before and after a spell is cancelled 

### Part II Machine Learning Models:
Our second part of project is to predict a risk score at the time the spell is cancelled. With the prediction score, we can achieve two goals: 1. to develop a new measure to evaluate self-efficiency; 2. to provide intervention to the spells who are most at risk at the time the spell is cancelled. 

- Predict the risk of return at a moment the spell is cancelled 
- Develop a new measure that substitutes current measure

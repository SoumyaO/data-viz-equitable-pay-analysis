# Data Visualisation: Human Resource Data
[Code](https://github.com/SoumyaO/data-viz-equitable-pay-analysis/blob/main/Group%206%20_SMM635_code.py) | [Data](https://github.com/SoumyaO/data-viz-equitable-pay-analysis/tree/main/dataset) | [Report](https://github.com/SoumyaO/data-viz-equitable-pay-analysis/blob/main/Report.pdf)

### Contributors
[Soumya Ogoti](https://github.com/SoumyaO)  
[Linh Nguyen](https://github.com/jill-data)  
[Wenxu Tian](https://github.com/Wayne599)  
[Aparna Viswanathan](https://github.com/aparnav97)  

## Overview
This project consisted of generating a set of five different visualisations from employee data using the python library Matplotlib to gain insights into the organization. These were
1. How do employees’ performance scores change within and across
managers?  
2. What is the overall diversity profile of the organization?  
3. What are our best recruiting sources if we want to ensure a diverse
organization?  
4. What are the most frequent factors for job termination?  
5. Are there areas of the company where pay is not equitable?

## My work
I worked on the question of understanding if the pay was equitable within the company and to see if there was any discrimination based on gender, ethnicity or race for each department in the company.

### Visualisation Design
I designed a customised lollipop chart to show if there is any pay inequality with respect to
gender, race and ethnicity as these are often the factors resulting in bias. I generated two charts, one to check the combined effect of gender and race on salary; another one to check the combined effect of gender and ethnicity on salary.

For the two charts, 
- The X-axis is divided grouped by position
- The Y-axis is the mean salary
- Horizontal lines depict the mean salaries of the job positions with more than ten employees. 
- Multiple lollipops are plotted on the horizontal lines for each position, one for each race.
- For each lollipop  
  * Colour of the lolllipos indicates race or ethnicity  
  * Gender is indicated by the style of the tip of the lollipop - filled for men, unfilled for women  
  * Size of the tips indicates the count of employees  
  * Triangle indicates the mean salary of that race  

### Design Features
The lollipop chart was chosen over a grouped bar chart to avoid cluttering and the Moiré effect. This chart is apt here to compare one numerical variable (salary) with two categorical variables (race/ethnicity, gender) simultaneously. It is easy to observe the effect of interactions between multiple factors – Salary with race/ethnicity and gender concisely with less data ink. Mean salary has been considered to capture the central tendency of each sub-category. In order to draw meaningful conclusions, only job positions having more than ten employees are shown in the chart. To guide the observer towards sub-categories with a sizeable number of employees, the tips have been scaled according to the number of employees.

### Key Insights
From this visualisation the following key insights could be inferred.
- For each job position, the mean salary for each race lies close to the mean salary of the position indicating no bias. 
- When comparing salary with race and gender, there is no specific pattern that indicates a bias towards one gender. 
- Also, for each race, the deviation from the mean caused due to gender is small. 
- The trend is similar also for ethnicity and gender (no bias). 
- Additionally, regression analysis with all relevant factors on salary did not show any pay inequity.

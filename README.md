This is my capstone project for my bachelor's degree at WGU. It consists of a rudimentary supervised regression machine-learning program that takes in a variety of behavioral factors from a student and uses them to predict the student's exam score. The program with developed with Jupyter notebook and Python and uses the CatBoostRegressor model to mix categorical and numeric inputs in order to provide a singular numeric answer. It also includes MSE and r2 score testing as well as visualizations of some of the most important aspects of the data.

- <b>[Video Walkthrough](https://youtu.be/nITm1tEzeyo)</b>

This program runs on a website known as mybinder. The steps to running the program are as follows:
1. Open a web browser and visit [mybinder.org](https://www.mybinder.org/)
2. Type or paste the [link](https://github.com/blackswan-03/C964_Capstone) to this repository into the "GitHub repository name or URL" box and click the "Launch" button (NOTE: The program may take some time to load. You can click the "Show" button in the "Build Logs" section to see what the build process is doing. If the build fails, simply refresh the page and try again.)
  <img src="Screenshots/Figure%205.png">
3. After the next page loads, go to the left side of the screen and double-click "student_exam_score_predictor.ipynb"
  <img src="Screenshots/Figure%206.png"> 
4. A new tab should open containing the application. Go to the toolbar and click on the button with the two play symbols on the left side of the "Download" button
  <img src="Screenshots/Figure%207.png">
5. When the "Restart Kernel?" pop-up window appears, click "Restart"
  <img src="Screenshots/Figure%208.png">
6. Wait a few seconds for each of the cells of the notebook to run, and then scroll down to the bottom of the page As you scroll down, you should see: (1) a small table with actual and predicted values between steps 5 and 6; (2) a pie chart between steps 6 and 7; (3) a scatter plot between steps 7 and 8; (4) a histogram between steps 8 and 9; and (5) an interactive interface with sliders and multiple-choice textboxes after step 9. (NOTE: If you do not see these things, click on the button with the two play icons and click "Restart" on the pop-up again. If these things still do not load, refresh the page and click the aforementioned buttons again.)
  <img src="Screenshots/Figure%201.png">
  <img src="Screenshots/Figure%202.png">
  <img src="Screenshots/Figure%203.png">
  <img src="Screenshots/Figure%204.png">
  <img src="Screenshots/Figure%209.png">
7. Use the sliders and multiple-choice textboxes to simulate behaviors for a student. Once you have satisfactorily fine-tuned your values, click on the button at the bottom that says "Run Interact". You should see text that says "Predicted Exam Score: (your predicted score here)". For example, a student with the following behavioral factors (age: 22; gender: Male; study_hours: 5.50; social_media_hours: 3.50; netflix_hours: 1.00; part_time_job: Yes; attendance: 90.30; sleep_hours: 7.00; diet_quality: Good; exercise_frequency: 4; parental_education: Bachelor; internet_quality: Good; mental_health_rating: 8; extracurriculars: No) should return "Predicted Exam Score: 95.48"

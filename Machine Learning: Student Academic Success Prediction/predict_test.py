import requests ## to use the POST method we use a library named requests

url = 'http://localhost:9696/predict'

student = {"marital_status": 1.0,
 "application_mode": 17.0,
 "application_order": 2.0,
 "course": 8014.0,
 "daytime/evening_attendance": 0.0,
 "previous_qualification": 1.0,
 "previous_qualification_(grade)": 133.0,
 "nacionality": 1.0,
 "mother's_qualification": 19.0,
 "father's_qualification": 1.0,
 "mother's_occupation": 5.0,
 "father's_occupation": 5.0,
 "admission_grade": 119.7,
 "displaced": 1.0,
 "educational_special_needs": 0.0,
 "debtor": 0.0,
 "tuition_fees_up_to_date": 1.0,
 "gender": 0.0,
 "scholarship_holder": 1.0,
 "age_at_enrollment": 19.0,
 "international": 0.0,
 "curricular_units_1st_sem_(credited)": 0.0,
 "curricular_units_1st_sem_(enrolled)": 6.0,
 "curricular_units_1st_sem_(evaluations)": 10.0,
 "curricular_units_1st_sem_(approved)": 4.0,
 "curricular_units_1st_sem_(grade)": 11.5,
 "curricular_units_1st_sem_(without_evaluations)": 0.0,
 "curricular_units_2nd_sem_(credited)": 0.0,
 "curricular_units_2nd_sem_(enrolled)": 6.0,
 "curricular_units_2nd_sem_(evaluations)": 11.0,
 "curricular_units_2nd_sem_(approved)": 6.0,
 "curricular_units_2nd_sem_(grade)": 11.5,
 "curricular_units_2nd_sem_(without_evaluations)": 0.0,
 "unemployment_rate": 9.4,
 "inflation_rate": -0.8,
 "gdp": -3.12}


url = 'http://localhost:9696/predict' ## this is the route we made for prediction
response = requests.post(url, json=student).json() ## post the student information in json format
print(response)
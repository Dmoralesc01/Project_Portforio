# Description of the problem
### Overview
The dataset at hand is derived from a compilation of various disjoint databases from a higher education institution. It pertains to students enrolled in diverse undergraduate programs, ranging from agronomy, design, education, nursing, journalism, management, social service, to technologies. The data captures essential information recorded at the time of student enrollment, encompassing academic paths, demographic details, and socioeconomic factors. Additionally, the dataset includes comprehensive records of the students' academic performances at the end of their initial and subsequent semesters. The primary objective is to develop robust classification models that can predict the likelihood of student dropout and academic success.

### Purpose of Dataset Creation
The creation of this dataset serves a pivotal role in a broader initiative focused on mitigating academic dropout and failure within the domain of higher education. Leveraging advanced machine learning techniques, the dataset aims to identify students at risk at an early stage of their academic journey. This preemptive identification enables the institution to implement tailored strategies and support systems to assist students in overcoming potential obstacles and challenges.

### Funders
The development of this dataset has been made possible through the support of the program SATDAP - Capacitação da Administração Pública, funded under the grant POCI-05-5762-FSE-000191, in Portugal.

### Classification Task and Data Splits
The fundamental challenge posed by this dataset is formulated as a three-category classification task, where the classes are characterized as 'dropout,' 'enrolled,' and 'graduate' status at the conclusion of the regular duration of the course.

## Instructions on how to run de project
In order to run this project in your local machine, download the contents of this repository. 
Install docker on your local machine if you don't have it.
Run `docker run -it --rm -p 9696:9696 midterm-project` on the terminal.
Once the container is running, open a new terminar window and run `python predict_test.py`
# Suicide Risk Severity Assessment

The implementation of the method in our WWW 2019 paper "Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention".

## Resources
[Paper](http://knoesis.org/sites/default/files/Suicide_Paper.pdf) and [Poster](https://drive.google.com/file/d/1vea_R09nvY2hRAE_PSy0Ez-8EUY8n7pa/preview)

## Data
Annotated Data mentioned in the paper is available in the ./Data folder (obtained from https://github.com/manasgaur/Knowledge-aware-Assessment-of-Severity-of-Suicide-Risk-for-Early-Intervention).


## Source Tree

    .
    ├── Data
    │   └── 500_Reddit_users_posts_labels.csv   : Anonymized annotated reddit dataset
    │
    └── models
        ├── 5-Label_Classification.py     : 5-Label classification {'Supportive', 'Indicator', 'Ideation', 'Behavior', 'Attempt'}
        ├── 4-Label_Classification.py     : 4-Label classification {Indicator', 'Ideation', 'Behavior', 'Attempt'}
        └── 3+1-Label_Classification.py   : 3+1-Label classification {('Supportive', 'Indicator'), 'Ideation', 'Behavior', 'Attempt'}


## How to use

- Clone the repository to your local machine:
- ```sh
    git@github.com:jpsain/Suicide-Severity.git
    ```
- Download the ConceptNet term vectors ("English-only") from [https://github.com/commonsense/conceptnet-numberbatch)
- Obtain external features for each reddit post in the input file ("Data/500_Reddit_users_posts_labels.csv") and save it as "Data/External_Features.csv".
    - External_Features.csv: "User", "Features"
- In the models -> 5-Label_Classification.py / 4-Label_Classification.py / 3+1-Label_Classification.py, modify the parameters as you desire and then run the code.
    ```sh
    python 5-Label_Classification.py
    ```
- The results will be save as "Result_5-Label_Classification.tsv".

## Licenses ##

This work is licensed under GPL-3.0 license. A copy of the first license can be found in this repository.

<p float="left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/9/93/GPLv3_Logo.svg" alt="GPLv3 Logo" width="60" />
</p>

## Citing

If you do make use of dataset or models or any of its components please cite the following publication:

    Manas Gaur, Amanuel Alambo, Joy Prakash Sain, Ugur Kursuncu, Krishnaprasad Thirunarayan, Ramakanth Kavuluru, Amit Sheth, Randy Welton, and Jyotishman Pathak. Knowledge-aware assessment of severity of suicide risk for early intervention. In The World Wide Web Conference, pp. 514-525. ACM, 2019.

    @inproceedings{gaur2019knowledge,
       title={Knowledge-aware assessment of severity of suicide risk for early intervention},
       author={Gaur, Manas
                and Alambo, Amanuel
                and Sain, Joy Prakash
                and Kursuncu, Ugur
                and Thirunarayan, Krishnaprasad
                and Kavuluru, Ramakanth
                and Sheth, Amit
                and Welton, Randy
                and Pathak, Jyotishman},
       booktitle={The World Wide Web Conference},
       pages={514--525},
       year={2019},
       organization={ACM}
    }


We would also be very happy if you provide a link to the github repository:

    ... Suicide Risk Severity Assessment\footnote{
        \url{https://github.com/jpsain/Suicide-Severity}
    }
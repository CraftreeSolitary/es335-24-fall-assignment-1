Given dataset contains six activities:
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING

For each of these, in the inertial signals folder, we've got:
- Triaxial angular velocity
- Triaxial total acceleration (with g)
- Triaxial body acceleration (without g)

Now for each of these physical variables we've got a training dataframe of shape (7352, 128) and a testing dataframe (2947, 128).

This data is split over 30 subjects. 
Train data is taken from subjects: [1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
Test data is taken from subjects: [2,  4,  9, 10, 12, 13, 18, 20, 24]

It is essential that we understand this data. 

As given in the README of the dataset, the sensor signals are sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). This means that the last reading of the first window will be the same as the middle reading of the second window.

Let us consider `total_acc_x_train.txt`, `subject_train.txt` and `y_train.txt`. It is clear that the first 27 rows of `total_acc_x_train.txt` correspond to Subject-1 Standing.
This means that we have got a recording of 2.56*(27+1)/2=35.84 seconds

`CombineScript.py` takes this data for each subject for each activity and stacks the last 64 readings of each window to undo the overlap. This is done for each of accx, accy, accz and made into a csv file for that subject for that activity as shown in the directory structure below:

Combined
    Test
        LAYING
            Subject_2.csv
            Subject_4.csv
            ...
        SITTING
            Subject_2.csv
            Subject_4.csv
            ...
    Train
        LAYING
            Subject_1.csv
            Subject_3.csv
            ...
        SITTING
            Subject_1.csv
            Subject_3.csv
            ...
        STANDING
        ...

Now it is natural for the length of each subject performing each task to vary and this is reflected in the difference in number of rows in say subject_1 laying (3201) and subject_3 laying (3969).

This natural difference is accounted for by `MakeDataset.py`.

Here for each of test and train, we go into each activity folder and append a 500 x 3 matrix into `X_train` and `X_test` respectively which consists rows from 100 to 599. Thus each of `X_train` and `X_test` will have `6*21=126` and `6*9=54` such matrices. `y_train` and `y_test` are populated with labels corresponding to the matrices.

At the end of `MakeDataset.py`, `X_train` and `X_test`, `y_train` and `y_test` are combined and again split randomly using `sklearn.model_selection.train_test_split` for good measure.
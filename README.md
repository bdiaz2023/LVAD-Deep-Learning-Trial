This model implements a series of completed computational fluid dynamics datasets focused on evaluating the hemodynamics in the left ventricle as a result of the angulation of the left ventricular assist device implant. Due to the expansive nature of the datasets we must implement a series of data preparation to ensure we can use the set in a deep learning model

To achieve this I implemented the individual reporting files from time steps 1000 to 2000 at an interval of 25 time steps (or approximately 1 cardiac cycle after tossing the initial 0 to 950 timesteps due to undeveloped fluid behaviors) which I then converted into a more readable .csv format for python using a basic matlab code to tabulate and convert the data into a .csv file

From there, I implemented a grid system to isolate the velocity components for each time step and angle case such that after gridding we have a 32x32x32x4 where the dimensions are defined as (Vx, Vy, Vz, Dim) where dimensions are defined by (Radial distance function, X, Y, Z)

From there we adjust the dimensions of our dataset to exclued the other 3 dimensions in our channels such that we have our shape defined as (Vx, Vy, Vz, RDF) where the rdf acts as our mask to define the cloudpoint space in a more confined manner and we now have a datset of 32x32x32x1

We then derived our mass flow rate for the inlet to act as an additional input value such that we concatenate the scalar inflow rate for an individual timepoint so for each time point we now have a 32x32x32x2

From there we concatenate our time points dropping a further 6 additional time steps at the end due to the near zero inlet rate at these points to avoid potential overfitting. Concatenating the total time steps we get a 15x32x32x32x2 grid for each angle case

To finalize our dataset we the concatenate the total time steps sets for each angle case together to get a 13x15x32x32x32x2 where our dimensions are defined by (angle, time, Vx, Vy, Vz, RDF/Inlet Rate)

From there we load them into the model where we will be attempting to implement a u-net model. To do this though we must reduce our 6d array to a 5d using the reshape feature to compress out 6d array into a 5d then implement a 60/20/20 trian/test/val split

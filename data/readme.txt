These folders contain the data required for and generated by the code base. As these are too big for GitHub, each folder contains a readme which explains what is required in the folders.

The following folders require external data before the code can run:

alternative_datasets    - requires the human plausibility ratings as for example generated by Kruszewski et. al.
glove                   - requires the vectors to build the density matrices. We utilize GloVe vectors

The following folders will be filled during execution:

density_matrices        - this folder will contain the density matrices generated by matrix-generation/build_density_matrices.py
output                  - this folder will contain the plausibility ratings generated from the negation framework when
                            running main.py. main.py requires the density matrices to be generated.
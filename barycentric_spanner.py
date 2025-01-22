import numpy as np

class BarycentricSpanner():

    def __init__(self , params , arms ,  C):

        # TODO: implement constructor for params
        self.arms = arms
        self.dim = params["dim_arms"]

        self.C = C
        self.spanning_set= np.identity(self.dim)
        self.construct_spanner()
        self.verify_spanner()

    def construct_spanner(self):
        '''
        construct the spanning set for the Barycentric Spanner
        '''
        # construct a basis contained in S
        print(f"Starting Phase 1 of finding the Barycentric Spanner")
        for i in range(self.dim):
            max_det = -np.inf
            for arm in self.arms:
                temp_matrix = self.spanning_set.copy()
                temp_matrix[: , i] = arm
                if np.abs(np.linalg.det(temp_matrix)) > max_det:
                        max_det = np.abs(np.linalg.det(temp_matrix))
                        self.spanning_set = temp_matrix.copy()

        # transform the basis into an approximate spanning set
        print(f"Starting Phase 2 of finding the Barycentric Spanner")
        current_dim = 0
        while True:
            for arm in self.arms:
                temp_matrix = self.spanning_set.copy()
                temp_matrix[: , current_dim] = arm
                if np.linalg.det(temp_matrix) > self.C * np.linalg.det(self.spanning_set):
                    self.spanning_set = temp_matrix.copy()
                    current_dim = 0
                    break
            current_dim += 1
            if current_dim == self.dim:
                break

    def verify_spanner(self):
        '''
        verifies the coefficients of the linear combination of spanning set are in [-C,C]
        '''
        print(f"Verifying the Barycentric Spanner...")
        for arm in self.arms:
            soln = np.linalg.solve(self.spanning_set, arm)
            assert soln.all() >= -self.C and soln.all() <= self.C , f"Barycentric Spanner failed for arm {arm}"
        print(f"Barycentric Spanner has been verified")
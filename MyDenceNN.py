import numpy as np
import time


class MyDenceNN:
    
    def __init__(self, hidden_units=None, activations={'1': "sigmoid"}, 
                 cost_fun="logistic_regression", alpha=0.1, epochs=1, display=False):  
        # Initialise Parameters
        self.hidden_units = hidden_units
        self.activations = activations
        self.cost_fun = cost_fun
        self.alpha = alpha
        self.epochs = epochs
        self.display = display
        
        self.input_units = None
        self.output_units = None
        
    #######################
    # Adds input features #
    #######################
    def add_inputs(self, X):
        try:
            self.X = np.c_[self.X, X]
            assert(self.X.shape[0] == X.shape[0])
        except:
            self.X = X
            self.input_units = X.shape[0]
        
        if self.display:
            print("Input:\n--------- \nX = \n{}\n".format(self.X))
            print("X has {} featurs\n\n".format(self.input_units))
        
    
    ########################
    # Adds target features #
    ########################
    def add_outputs(self, Y):
        try:
            self.Y = np.c_[self.Y, Y]
            assert(self.Y.shape[0] == Y.shape[0])
        except:
            self.Y = Y
            self.output_units = Y.shape[0]
        
        if self.display:
            print("Output:\n--------- \nY = \n{}\n".format(self.Y))
            print("Y has {} featurs\n\n".format(self.output_units))
        
    
    ################
    # Add features #
    ################
    def add_data(self, X, Y):
        if self.display:
            print("#################\nCurrent Features")
            print("-----------------\n-----------------\n")
        self.add_inputs(X) # Adds input features
        self.add_outputs(Y) # Adds target features
    
    
    ##############
    # Add layers #
    ##############
    def add_units(self, hidden_units):
        try:
            self.hidden_units.update(hidden_units)
        except:
            self.hidden_units = hidden_units
            
        n = self.hidden_units
        if self.display:
            print("#################\nUnits per layer")
            for l in range(1, len(n)+1):
                print("----------------")
                print("n[{}] = {}".format(l, n[str(l)]))

            print("----------------")
            print("----------------")
        
    
    #########################
    # Create Neural Network #
    #########################
    def build(self):
        units = self.hidden_units
        h = len(self.hidden_units)
        
        units["0"] = self.input_units
        units[str(h+1)] = self.output_units
        
        self.units = units
        self.initialise_params(units)
    
    
    #################################
    # Initiallise Weight and Biases #
    #################################
    def initialise_params(self, units):
        """
        ## Generates units in each layer of the network ##
        Inputs: number of units in each layer dims = [n0, n1, n2, ..., nL]
        output: W = {"W1": W1, ..., "WL": WL}, b = {"b1": b1, ..., "bL": bL}
        """
        # Number of layers
        L = len(units) - 1
        # Dictionary of parameters
        self.W = {}
        self.b = {}
        
        if self.display:
            print("#################\nWights and Biasses")
            
        for l in range(1, L+1):
            self.W[str(l)] = np.random.randn(units[str(l)], units[str(l-1)])
            self.b[str(l)] = np.zeros((units[str(l)], 1))
            
            if self.display:
                print("\n----------------")
                print("----------------")
                print("Layer {}:".format(l))
                print("----------------")
                print("W{} = \n{}".format(l, self.W[str(l)]))
                print("----------------")
                print("b{} = \n{}".format(l, self.b[str(l)]))
                print("----------------")
                print("----------------\n\n")
        
        
    ###############################
    # Forward linear Calculations #
    ###############################
    def forward_linear(self, A_prev, W, b):
        """
        ## Calculate linear equation for forward pass ##
        Inputs: A = previous layer activation, W = current weights, b = current biases
        Output: Z = W•A + b
        """
        # Linear Equation
        Z = np.dot(W, A_prev) + b
        return Z
    
    
    ########################
    # Activation Functions #
    ########################
    
    # Sigmodal Activation
    def Sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))

    # Swish Activation
    def Swish(self, Z):
        return Z/(1 + np.exp(-Z))

    # Hyperbolic Tangent
    def Tanh(self, Z):
        return np.tanh(Z)

    # Rectified Linear Unit
    def ReLU(self, Z):
        return np.maximum(0, Z)

    # Leaky Rectified Linear Unit
    def L_ReLU(self, Z):
        return np.maximum(0.01*Z, Z)

    # # Parametric Rectified Linear Unit
    # def P_ReLU(Z):
    #     return np.maximum(0.05*Z, Z)
    #-----------------------------------
    #-----------------------------------
    

    ############################
    # Forward pass calculation #
    ############################
    def forward_activation(self, Z, g):
        """ 
        ## Applies specified activation function to Z ##
        Inputs: Z = linear forward pass, g = activation function
        Output: A = g(z) 
        """
        # Dictionary of activations functions
        activations = {
            'sigmoid': self.Sigmoid,
            'swish'  : self.Swish,
            'tanh'   : self.Tanh,
            'relu'   : self.ReLU,
            'l_relu' : self.L_ReLU
        }
        # Call activation on Z
        try:
            A = activations[g](Z)
            return A
        except:
            display(print("\n----------------\nInvalid activation function\n----------------\n"))
            display(print(g))
            display(print("----------------\nValid activations:\n----------------\n"))
            for key in activations.keys():
                display(print(key))
            pass


    ##################
    # Cost functions #
    ##################

    # Logistic Regression
    def LOG(self, AL, Y):
        J = -1/Y.shape[1]*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL), axis=1, keepdims=True)
        return J

    # Mean Squared Error
    def MSE(self, AL, Y):
        error = Y - AL
        J = -1/Y.shape[1]*np.sum(error*error, axis=1, keepdims=True)
        return J

    # Mean Absolute Error
    def MAE(self, AL, Y):
        J = -1/Y.shape[1]*np.sum(np.absolute(Y, AL), axis=1, keepdims=True)
        return J
    #------------------
    #------------------

            
    ##################
    # Calculate Cost #
    ##################
    def cost(self, AL):
        """
        Calculates the current cost (error) of the models out puts AL
        Inputs: AL = Model Predication, Y = Expected (True) values, 
                cost_fun = Specified Cost Function
        Output: J = cost
        """
        # List of available functions
        functions = {## Add more functionlater
            'logistic_regression': self.LOG, # Logistic Regression
            'mean_squared_error' : self.MSE, # Mean Squared Error
            'mean_absolute_error': self.MAE  # Mean Absolute Error
        }
        try:
            J = functions[self.cost_fun](AL, self.Y)
            print("\n-------------------")
            print("-------------------")
            print("Cost: J(AL, Y) = {}".format(J))
            print("-------------------")
            print("-------------------\n")
            return J
        except:
            print("\n----------------\n----------------")
            print("Invalid Cost Function:\n----------------")
            print(self.cost_fun)
            print("\n\n----------------\nCost Functions available:\n----------------\n")
            for key in functions.keys():
                display(print(key))
            pass
            print("----------------\n----------------")
    
    
    ########################
    # Forward Propergation #
    ########################
    def forward_propagation(self):
        """
        ## Completes a full forward pass through Network ##
        Inputs: X = Full Feature Set, W = Weights in Network, 
                b = Biases in Network, g = All Activation Functions
        Output: A, Z
        """
        X = self.X
        W = self.W
        b = self.b
        g = self.activations
        #assert(W.shape[0] == b.shape[0]

        L = len(self.units) - 1
        A_prev = X
        self.A = {"0": X}
        self.Z = {}

        if self.display:
            print("###################")
            print("Forward Propagation")
            print("###################\n")
        for l in range(1, L+1):
            # Parameters
            bl = b[str(l)]
            Wl = W[str(l)]
            # Current activation 
            try:
                gl = g[str(l)]
            except:
                gl = "sigmoid"
            if self.display:
                print("Layer: {} using {} activation".format(l, gl))

            # Forward Linear Pass
            Zl = self.forward_linear(A_prev, Wl, bl)
            Al = self.forward_activation(Zl, gl)

            # Update and Chase Values
            A_prev = Al  
            self.A[str(l)] = Al
            self.Z[str(l)] = Zl
        
        self.J = self.cost(self.A[str(L)])
    
    
    ###############################################
    ###############################################
    
    
    ##########################
    # Activation Derivatives #
    ##########################
    def Sigmoid_prime(self, A):
        return A * (1-A)

    def Tanh_prime(self, A):
        return 1 - np.power(A, 2)

    def ReLU_prime(self, A):
        A[A<=0] = 0
        A[A>0] = 1
        return A

    def L_ReLU_prime(self, A, alpha):
        return 1 if A > 0 else alpha

    def initial_dAL(self, AL, Y):
        dAL = - (Y/AL - (1 - Y)/(1 - AL))
        return dAL


    #########################
    # Bsckwards Avtivations #
    #########################
    def backward_activation(self, Wl, bl, Al, dAl, A_prev, gl):
        # Dictionary of activations functions
        derivatives = {
            'sigmoid': self.Sigmoid_prime,
            'tanh'   : self.Tanh,
            'relu'   : self.ReLU_prime,
            'l_relu' : self.L_ReLU_prime
        }
        m = Al.shape[1]
        gl_prime = derivatives[gl](Al)

        dZl = dAl*gl_prime
        dWl = 1/m*np.dot(dZl, A_prev.T)
        dbl = 1/m*np.sum(dZl, axis=1, keepdims=True)

        # Calculate dA[l-1]
        dA_prev = np.dot(Wl.T, dZl)
        return dA_prev, dWl, dbl
    
    
    ####################
    # Backwards Linear #
    ####################
    def backward_linear(self, dW, db, l, alpha=1):
        self.W[str(l)] -= alpha*dW
        self.b[str(l)] -= alpha*db
    
    
    #########################
    # Bsckwards Propagation #
    #########################
    def backward_propagation(self):
        # Initialoise
        W = self.W
        b = self.b
        Y = self.Y
        A = self.A
        g = self.activations
        alpha = self.alpha
        
        L = len(self.units) - 1
        A_prev = A[str(L)]
        dA_prev = self.initial_dAL(A_prev, Y)
        # Backpropegate
        for l in range(L, 0, -1):
            # Current activation 
            try:
                gl = g[str(l)]
            except:
                gl = "sigmoid"
            if self.display:
                print("Layer: {} using {} activation".format(l, gl))
            # Get Variable and Parameters
            Wl, bl = W[str(l)], b[str(l)]
            Al, dAl = A_prev, dA_prev
            A_prev  = A[str(l-1)]
            # Calculate derivates
            dA_prev, dWl, dbl = self.backward_activation(Wl, bl, Al, dAl, A_prev, gl)
            # Update current Parameters
            self.backward_linear(dWl, dbl, l, alpha)

    
    #######################################
    #######################################
    
    
    ######################################
    # Dence Neural Network Model Builder #
    ######################################
    def solve(self, X=None, Y=None, add_hidden_units=None, add_activations=None, 
                 cost_fun="logistic_regression", alpha=0.1, epochs=1):
        # Initialise Parameters
        self.add_data(X, Y)
        self.add_units(add_hidden_units)
        self.activations.update(add_activations)
        self.cost_fun = cost_fun
        self.alpha = alpha
        self.epochs = epochs
        
        self.costs = {}
        self.forward_prop_time = {}
        self.backward_prop_time = {}
        
        if self.display:
            print("######################################")
            print("######################################")
            print("\t\t  Optimising Model")
            print("######################################")
            print("######################################\n\n")
        
        # Initialise Parameters
        self.build()
        
        # For epoch in epochs
        for epoch in range(1, epochs+1):
            print("\n----------------------------------------")
#             print("----------------------------------------")
            print("\t\t  Epoch {}".format(epoch))
            print("----------------------------------------")
#             print("----------------------------------------\n")
            
            ## Forwardpropagation
            tic = time.time()
            self.forward_propagation()
            toc = time.time()
            ### Save Cost and Run-time
            self.costs["epoch"+str(epoch)] = self.J
            self.forward_prop_time["epoch"+str(epoch)] = toc - tic
            
            L = str(len(self.units) - 1)
#             print("\n----------------------------------------")
#             print("AL = \n{}".format(self.A[L]))
#             print("W = \n {}\n\nb = \n{}".format(self.W[L],self.b[L]))
#             print("----------------------------------------")
            
#             print("\n----------------------------------------")
#             print("----------------------------------------")
#             print("Forward Prop Run-time:\t\t{}".format(toc-tic))
#             print("----------------------------------------")
#             print("----------------------------------------\n")

            ## Backwardpropagation
            tic = time.time()
            self.backward_propagation()
            toc = time.time()
            ### Save W and b
            self.backward_prop_time["epoch"+str(epoch)] = toc - tic
            
#             print("\n----------------------------------------")
#             print("----------------------------------------")
#             print("Backward Prop Run-time:\t\t{}".format(toc-tic))
#             print("----------------------------------------")
#             print("----------------------------------------\n")
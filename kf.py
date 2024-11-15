import numpy as np 

class QinKalmanFilter:
    def __init__(self, dt, a, x_init, A, B, C, H, P_init, Q, R):
        """
        Initialize the Kalman Filter for tank inflow estimation.
        
        Parameters:
        -----------
        dt : float
            Time step
        a : float
            Tank cross-sectional area
        x_init : ndarray, shape (2,), optional
            Initial state estimate [h, qin]
        P_init : ndarray, shape (2,2), optional
            Initial error covariance matrix
        Q : ndarray, shape (2,2), optional
            Process noise covariance matrix
        R : float, optional
            Measurement noise variance
        """
        self.dt = dt
        self.a = a
        
        # System matrices
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.C = C
        self.P = P_init
        
        # Initialize state
        self.x = x_init if x_init is not None else None
        

    def initialize_state(self, h_measured):
        """Initialize state if not done in constructor."""
        if self.x is None:
            self.x = np.array([h_measured, 0.0])
            
    def predict(self, qout):
        """
        Prediction step of the Kalman filter.
        
        Parameters:
        -----------
        qout : float
            Controlled outflow rate
        """
        # Predict state
        self.x_pred = self.A @ self.x + self.B.flatten() * qout
        self.P_pred = self.A @ self.P @ self.A.T + self.Q
        
    def update(self, h_measured):
        """
        Update step of the Kalman filter.
        
        Parameters:
        -----------
        h_measured : float
            Measured level in the tank
        """
        # Kalman gain
        K = self.P_pred @ self.H.T @ np.linalg.inv(self.H @ self.P_pred @ self.H.T + self.R)
        
        # Update state
        innovation = h_measured - self.H @ self.x_pred
        self.x = self.x_pred + K.flatten() * innovation
        self.x = np.array([self.x[0], self.x[1]], dtype=np.float64)
        self.P = (np.eye(2) - K @ self.H) @ self.P_pred
        
    def step(self, h_measured, qout):
        """
        Perform one complete step of the Kalman filter.
        
        Parameters:
        -----------
        h_measured : float
            Measured level in the tank
        qout : float
            Controlled outflow rate
            
        Returns:
        --------
        x : ndarray
            Updated state estimate [h, qin]
        P : ndarray
            Updated error covariance matrix
        """
        # Initialize state if needed
        if self.x is None:
            self.initialize_state(h_measured)
            
        # Predict and update
        self.predict(qout)
        self.update(h_measured)
        
        return self.x, self.P
    
    @property
    def state(self):
        """Get current state estimate."""
        return self.x
    
    @property
    def covariance(self):
        """Get current error covariance."""
        return self.P
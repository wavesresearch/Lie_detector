import numpy as np
from scipy.linalg import eigh

class CSP:
    def __init__(self, n_components=4, regularization=0.2):
        self.n_components = n_components
        self.regularization = regularization  # alpha
        self.filters_ = None

    def fit(self, X, y):
        """
        Fit CSP spatial filters to data with regularized covariance.
        
        Parameters:
        X : ndarray of shape (n_trials, n_channels, n_samples)
        y : ndarray of shape (n_trials,)
        """
        X1 = X[y == 0]
        X2 = X[y == 1]

        cov1 = self._calc_cov_matrix(X1)
        cov2 = self._calc_cov_matrix(X2)

        
        cov1 = self._regularize_cov(cov1)
        cov2 = self._regularize_cov(cov2)

        composite_cov = cov1 + cov2
        composite_cov = np.nan_to_num(composite_cov, nan=0.0, posinf=1e10, neginf=-1e10)

        eigenvalues, eigenvectors = eigh(composite_cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        whitening_transform = np.dot(np.diag(1.0 / np.sqrt(eigenvalues + 1e-10)), eigenvectors.T)

        whiten_cov1 = whitening_transform.dot(cov1).dot(whitening_transform.T)
        eigenvalues, eigenvectors = eigh(whiten_cov1)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        self.filters_ = np.dot(eigenvectors.T, whitening_transform)

        self.filters_ = np.vstack((
            self.filters_[:self.n_components // 2],
            self.filters_[-self.n_components // 2:]
        ))

    def transform(self, X):
        if self.filters_ is None:
            raise ValueError("CSP filters not fitted yet. Call fit() first.")
        return np.array([np.dot(self.filters_, trial) for trial in X])

    def _calc_cov_matrix(self, X):
        n_trials = X.shape[0]
        cov = np.zeros((X.shape[1], X.shape[1]))
        epsilon = 1e-10

        for trial in X:
            trial_cov = np.cov(trial)
            trace = np.trace(trial_cov)
            trace = trace if trace > epsilon else epsilon
            cov += trial_cov / trace

        cov /= (n_trials + epsilon)
        cov = np.nan_to_num(cov, nan=0.0, posinf=1e10, neginf=-1e10)
        return cov

    def _regularize_cov(self, cov):
        """
        Apply Tikhonov regularization (shrinkage toward identity).
        """
        alpha = self.regularization
        identity = np.eye(cov.shape[0])
        return (1 - alpha) * cov + alpha * np.trace(cov) / cov.shape[0] * identity



class CSP_L1:
    def __init__(self, n_components=4, max_iter=100, eta=0.01, tol=1e-6):
        """
        Initialize CSP-L1 algorithm
        
        Parameters:
        n_components : int (default=3)
            Number of spatial filters to find for each class
        max_iter : int (default=50)
            Maximum number of iterations for optimization
        eta : float (default=0.01)
            Learning rate for gradient ascent
        tol : float (default=1e-6)
            Tolerance for convergence
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol
        self.filters_ = None
        self.filters_inv_ = None
    
    def fit(self, X, y):
        """
        Fit CSP-L1 spatial filters to data
        
        Parameters:
        X : ndarray of shape (n_trials, n_channels, n_samples)
            EEG data for two classes
        y : ndarray of shape (n_trials,)
            Labels for each trial (0 for class 1, 1 for class 2)
        """
        if not np.any(y == 0) or not np.any(y == 1):
            raise ValueError("Both classes (0 and 1) must be present in y for CSP.")
        # Separate data by class
        X1 = X[y == 0]
        X2 = X[y == 1]
        print(f"Number of trials for class 1: {len(X1)}, class 2: {len(X2)}")
        
        # Reshape data to (n_channels, n_samples * n_trials)
        X1_flat = np.concatenate(X1, axis=1)
        X2_flat = np.concatenate(X2, axis=1)
        
        # Find spatial filters that maximize X1 / X2 ratio
        self.filters_ = self._find_filters(X1_flat, X2_flat)
        
        # Find spatial filters that maximize X2 / X1 ratio (inverse)
        self.filters_inv_ = self._find_filters(X2_flat, X1_flat)
    
    def _find_filters(self, X, Y):
        """
        Find spatial filters that maximize ||w^T X||_1 / ||w^T Y||_1
        
        Parameters:
        X : ndarray of shape (n_channels, m)
            EEG data for first class (all trials concatenated)
        Y : ndarray of shape (n_channels, n)
            EEG data for second class (all trials concatenated)
            
        Returns:
        ndarray of shape (n_components, n_channels)
            Spatial filters (rows are filters)
        """
        n_channels = X.shape[0]
        filters = []
        
        # Initialize with standard CSP solution as suggested in paper
        Cx = X @ X.T / X.shape[1]
        Cy = Y @ Y.T / Y.shape[1]
        # Regularize covariance matrices
        reg = 1e-6
        Cx += reg * np.eye(Cx.shape[0])
        Cy += reg * np.eye(Cy.shape[0])
        eigvals, eigvecs = eigh(Cx, Cy)
        idx = eigvals.argsort()[::-1]  # Sort descending
        initial_w = eigvecs[:, idx[0]]
        
        for _ in range(self.n_components):
            if len(filters) > 0:
                # Project to orthogonal complement of previous filters
                W = np.array(filters).T
                P = np.eye(n_channels) - W @ W.T
                X_proj = P @ X
                Y_proj = P @ Y
                initial_w = eigvecs[:, idx[len(filters)]]
                initial_w = P @ initial_w
            else:
                X_proj = X
                Y_proj = Y
            
            # Optimize CSP-L1 objective
            w = self._optimize_csp_l1(X_proj, Y_proj, initial_w)
            
            # Normalize filter
            w = w / np.linalg.norm(w)
            filters.append(w)
        
        return np.array(filters)
    
    def _optimize_csp_l1(self, X, Y, initial_w):
        """
        Optimize CSP-L1 objective for single filter
        
        Parameters:
        X : ndarray of shape (n_channels, m)
            EEG data for first class
        Y : ndarray of shape (n_channels, n)
            EEG data for second class
        initial_w : ndarray of shape (n_channels,)
            Initial filter weights
            
        Returns:
        ndarray of shape (n_channels,)
            Optimized spatial filter
        """
        w = initial_w.copy()
        w = w / np.linalg.norm(w)  # Normalize
        
        prev_obj = -np.inf
        for _ in range(self.max_iter):
            # Compute polarities (signs)
            p = np.sign(w.T @ X)
            q = np.sign(w.T @ Y)
            
            # Compute L1 dispersions
            X_disp = np.sum(np.abs(w.T @ X))
            Y_disp = np.sum(np.abs(w.T @ Y))
            
            # Compute gradient direction (eq. 7 in paper)
            d = (X @ p.T) / X_disp - (Y @ q.T) / Y_disp
            
            # Update filter (eq. 8 in paper)
            new_w = w + self.eta * d
            new_w = new_w / np.linalg.norm(new_w)
            
            # Compute new objective value
            new_X_disp = np.sum(np.abs(new_w.T @ X))
            new_Y_disp = np.sum(np.abs(new_w.T @ Y))
            new_obj = new_X_disp / new_Y_disp
            
            # Check for convergence
            if new_obj - prev_obj < self.tol:
                break
                
            w = new_w
            prev_obj = new_obj
        
        return w
    
    def transform(self, X):
        """
        Extract features from EEG trials using learned CSP-L1 filters
        
        Parameters:
        X : ndarray of shape (n_trials, n_channels, n_samples)
            EEG data to transform
            
        Returns:
        ndarray of shape (n_trials, 2 * n_components)
            Extracted features (L1 norms of filtered signals)
        """
        if self.filters_ is None or self.filters_inv_ is None:
            raise ValueError("CSP-L1 filters not fitted yet. Call fit() first.")
            
        features = []
        for trial in X:
            # Features from filters that maximize class1/class2
            trial_features = [np.sum(np.abs(w @ trial)) for w in self.filters_]
            
            # Features from filters that maximize class2/class1
            trial_features += [np.sum(np.abs(w @ trial)) for w in self.filters_inv_]
            
            features.append(trial_features)
        
        return np.array(features)



"""
HealthML-Toolkit Test Suite - Imputation Methods

Tests for data quality module imputation techniques.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class TestSimpleImputation:
    """Test mean/mode/median imputation methods."""
    
    def test_mean_imputation(self):
        """Test mean imputation on numeric data."""
        # Arrange
        data = np.array([[1, 2], [np.nan, 4], [5, 6]])
        expected_mean = 3.0  # (1 + 5) / 2
        
        # Act
        imputer = SimpleImputer(strategy='mean')
        result = imputer.fit_transform(data)
        
        # Assert
        assert not np.isnan(result).any(), "Result contains NaN values"
        assert result[1, 0] == expected_mean, "Mean imputation incorrect"
    
    def test_mode_imputation(self):
        """Test mode imputation on categorical data."""
        # Arrange
        data = pd.DataFrame({
            'Gender': ['M', 'F', np.nan, 'M', 'F'],
            'BloodType': ['A', 'B', 'A', np.nan, 'A']
        })
        
        # Act
        imputer = SimpleImputer(strategy='most_frequent')
        result = pd.DataFrame(
            imputer.fit_transform(data),
            columns=data.columns
        )
        
        # Assert
        assert result.isnull().sum().sum() == 0, "Result contains NaN"
        assert result.loc[2, 'Gender'] == 'M', "Mode imputation failed"
    
    def test_median_imputation(self):
        """Test median imputation (robust to outliers)."""
        # Arrange
        data = np.array([[1], [2], [np.nan], [100]])  # Outlier: 100
        expected_median = 2.0
        
        # Act
        imputer = SimpleImputer(strategy='median')
        result = imputer.fit_transform(data)
        
        # Assert
        assert result[2, 0] == expected_median, "Median imputation incorrect"


class TestKNNImputation:
    """Test K-nearest neighbors imputation."""
    
    def test_knn_basic(self):
        """Test basic KNN imputation."""
        # Arrange
        data = np.array([
            [1, 2],
            [np.nan, 4],
            [5, 6]
        ])
        
        # Act
        imputer = KNNImputer(n_neighbors=1)
        result = imputer.fit_transform(data)
        
        # Assert
        assert not np.isnan(result).any(), "KNN result contains NaN"
        # Missing value should be close to nearest neighbor
        assert 1 <= result[1, 0] <= 5, "KNN imputation out of range"
    
    def test_knn_multiple_neighbors(self):
        """Test KNN with k=3."""
        # Arrange
        data = np.array([
            [1, 10],
            [2, 20],
            [3, 30],
            [np.nan, 40],
            [5, 50]
        ])
        
        # Act
        imputer = KNNImputer(n_neighbors=3)
        result = imputer.fit_transform(data)
        
        # Assert
        assert not np.isnan(result).any()
        # Should be average of 3 nearest neighbors
        assert 2 <= result[3, 0] <= 4, "KNN k=3 imputation incorrect"


class TestMICEImputation:
    """Test Multivariate Imputation by Chained Equations (MICE)."""
    
    def test_mice_basic(self):
        """Test MICE on correlated features."""
        # Arrange
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data[10:20, 0] = np.nan  # Introduce 10% missing
        data[30:35, 1] = np.nan
        
        # Act
        imputer = IterativeImputer(max_iter=10, random_state=42)
        result = imputer.fit_transform(data)
        
        # Assert
        assert not np.isnan(result).any(), "MICE result contains NaN"
        assert result.shape == data.shape, "MICE changed data shape"
    
    def test_mice_convergence(self):
        """Test MICE convergence with max iterations."""
        # Arrange
        data = np.array([
            [1, 2, 3],
            [4, np.nan, 6],
            [7, 8, np.nan]
        ])
        
        # Act
        imputer = IterativeImputer(max_iter=50, random_state=42)
        result = imputer.fit_transform(data)
        
        # Assert
        assert not np.isnan(result).any()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_all_missing_column(self):
        """Test handling of completely missing columns."""
        # Arrange
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        # Act & Assert - should handle gracefully
        imputer = SimpleImputer(strategy='mean')
        with pytest.raises(ValueError):
            # sklearn should raise error for all-NaN column
            imputer.fit_transform(data)
    
    def test_no_missing_values(self):
        """Test imputation on complete data (no-op)."""
        # Arrange
        data = np.array([[1, 2], [3, 4], [5, 6]])
        
        # Act
        imputer = SimpleImputer(strategy='mean')
        result = imputer.fit_transform(data)
        
        # Assert
        np.testing.assert_array_equal(result, data)
    
    def test_single_row_data(self):
        """Test imputation on single row (edge case)."""
        # Arrange
        data = np.array([[np.nan, 2]])
        
        # Act & Assert
        imputer = SimpleImputer(strategy='mean')
        with pytest.raises(ValueError):
            # Cannot compute mean from single value with NaN
            imputer.fit_transform(data)


class TestImputationQuality:
    """Test imputation quality using ML evaluation."""
    
    def test_knn_vs_mean_accuracy(self):
        """Compare KNN vs Mean imputation on classification task."""
        # Arrange
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Introduce 10% missing values randomly
        np.random.seed(42)
        mask = np.random.rand(*X.shape) < 0.1
        X_missing = X.copy()
        X_missing[mask] = np.nan
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_missing, y, test_size=0.2, random_state=42
        )
        
        # Mean imputation
        mean_imputer = SimpleImputer(strategy='mean')
        X_train_mean = mean_imputer.fit_transform(X_train)
        X_test_mean = mean_imputer.transform(X_test)
        
        clf_mean = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_mean.fit(X_train_mean, y_train)
        acc_mean = accuracy_score(y_test, clf_mean.predict(X_test_mean))
        
        # KNN imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        X_train_knn = knn_imputer.fit_transform(X_train)
        X_test_knn = knn_imputer.transform(X_test)
        
        clf_knn = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_knn.fit(X_train_knn, y_train)
        acc_knn = accuracy_score(y_test, clf_knn.predict(X_test_knn))
        
        # Assert
        assert acc_mean > 0.8, "Mean imputation accuracy too low"
        assert acc_knn > 0.8, "KNN imputation accuracy too low"
        # KNN should generally perform better or equal
        print(f"Mean accuracy: {acc_mean:.4f}")
        print(f"KNN accuracy: {acc_knn:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

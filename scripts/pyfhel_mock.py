"""
Mock implementation of Pyfhel for testing purposes when the library cannot be installed.
This provides basic functionality to allow the project to run without full HE capabilities.
"""

import numpy as np
import warnings

class MockPyfhel:
    """Mock Pyfhel class that provides basic functionality without actual homomorphic encryption."""
    
    def __init__(self):
        self.scheme = "ckks"
        self.n = 16384
        self.scale = 2**40
        warnings.warn("Using mock Pyfhel implementation. No actual homomorphic encryption is performed.", UserWarning)
    
    def contextGen(self, scheme="ckks", n=16384, scale=2**40, qi_sizes=None):
        """Generate context (mock implementation)."""
        self.scheme = scheme
        self.n = n
        self.scale = scale
        return True
    
    def keyGen(self):
        """Generate keys (mock implementation)."""
        return True
    
    def relinKeyGen(self):
        """Generate relinearization keys (mock implementation)."""
        return True
    
    def rotateKeyGen(self):
        """Generate rotation keys (mock implementation).""" 
        return True
    
    def encodeFrac(self, value):
        """Mock encoding that returns a MockPyPtxt."""
        if isinstance(value, (list, np.ndarray)):
            return MockPyPtxt(np.array(value, dtype=float))
        else:
            return MockPyPtxt(np.array([float(value)]))
    
    def encryptPtxt(self, ptxt):
        """Mock encryption that returns a MockPyCtxt."""
        return MockPyCtxt(ptxt.data)
    
    def encryptFrac(self, value):
        """Mock encryption that returns a MockPyCtxt."""
        if isinstance(value, (list, np.ndarray)):
            return MockPyCtxt(np.array(value, dtype=float))
        else:
            return MockPyCtxt(np.array([float(value)]))
    
    def decryptFrac(self, ctxt):
        """Mock decryption that returns the original value."""
        return ctxt.data

class MockPyCtxt:
    """Mock ciphertext class."""
    
    def __init__(self, data):
        self.data = np.array(data, dtype=float)
    
    def __add__(self, other):
        """Addition operation."""
        if isinstance(other, MockPyCtxt):
            return MockPyCtxt(self.data + other.data)
        elif isinstance(other, MockPyPtxt):
            return MockPyCtxt(self.data + other.data)
        else:
            return MockPyCtxt(self.data + float(other))
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        """Multiplication operation."""
        if isinstance(other, MockPyCtxt):
            return MockPyCtxt(self.data * other.data)
        elif isinstance(other, MockPyPtxt):
            return MockPyCtxt(self.data * other.data)
        else:
            return MockPyCtxt(self.data * float(other))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        """Subtraction operation."""
        if isinstance(other, MockPyCtxt):
            return MockPyCtxt(self.data - other.data)
        else:
            return MockPyCtxt(self.data - float(other))
    
    def __rsub__(self, other):
        if isinstance(other, MockPyCtxt):
            return MockPyCtxt(other.data - self.data)
        else:
            return MockPyCtxt(float(other) - self.data)

class MockPyPtxt:
    """Mock plaintext class."""
    
    def __init__(self, data):
        self.data = np.array(data, dtype=float)

# Create mock instances to replace the real Pyfhel imports
Pyfhel = MockPyfhel
PyCtxt = MockPyCtxt
PyPtxt = MockPyPtxt
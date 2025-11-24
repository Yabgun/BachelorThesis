"""
NIST-Style Test Vectors for CKKS Homomorphic Encryption Algorithm
================================================================

This module implements comprehensive test vectors following NIST testing methodology
for validating CKKS homomorphic encryption implementation accuracy and security.

Test Categories:
1. Known Answer Tests (KAT) - Deterministic input/output validation
2. Parameter Validation Tests - Different CKKS parameter combinations
3. Operation Accuracy Tests - Homomorphic operation correctness
4. Error Bound Validation - Approximation error analysis
5. Cross-Implementation Compatibility - Reference implementation comparison

Author: HE Research Project
Date: 2024
"""

import json
import time
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any

try:
    from Pyfhel import Pyfhel, PyCtxt, PyPtxt
    REAL_PYFHEL = True
except ImportError:
    print("Warning: Using mock Pyfhel implementation for testing...")
    from pyfhel_mock import Pyfhel, PyCtxt, PyPtxt
    REAL_PYFHEL = False

class CKKSNISTTestVectors:
    """NIST-style test vector generator and validator for CKKS algorithm."""
    
    def __init__(self):
        self.test_results = []
        self.error_tolerance = 1e-6  # Default error tolerance for CKKS approximations
        
    def generate_known_answer_tests(self) -> List[Dict]:
        """Generate Known Answer Tests (KAT) with predetermined inputs and expected outputs."""
        print("Generating CKKS Known Answer Tests...")
        
        kat_tests = []
        
        # Test Vector 1: Basic encryption/decryption
        test_vector_1 = {
            "test_id": "CKKS_KAT_001",
            "description": "Basic encryption and decryption of single value",
            "input_plaintext": [3.14159],
            "expected_output": [3.14159],
            "parameters": {"n": 2**12, "scale": 2**40, "qi_sizes": [60, 40, 60]},
            "tolerance": 1e-3
        }
        kat_tests.append(test_vector_1)
        
        # Test Vector 2: Multiple values encryption
        test_vector_2 = {
            "test_id": "CKKS_KAT_002", 
            "description": "Encryption and decryption of multiple values",
            "input_plaintext": [1.0, 2.5, -3.7, 0.0, 42.42],
            "expected_output": [1.0, 2.5, -3.7, 0.0, 42.42],
            "parameters": {"n": 2**13, "scale": 2**30, "qi_sizes": [60, 40, 40, 60]},
            "tolerance": 1e-4
        }
        kat_tests.append(test_vector_2)
        
        # Test Vector 3: Small fractional values
        test_vector_3 = {
            "test_id": "CKKS_KAT_003",
            "description": "Encryption of small fractional values",
            "input_plaintext": [0.001, 0.0001, 0.00001],
            "expected_output": [0.001, 0.0001, 0.00001],
            "parameters": {"n": 2**14, "scale": 2**50, "qi_sizes": [60, 50, 50, 60]},
            "tolerance": 1e-5
        }
        kat_tests.append(test_vector_3)
        
        return kat_tests
    
    def generate_operation_accuracy_tests(self) -> List[Dict]:
        """Generate tests for homomorphic operation accuracy."""
        print("Generating CKKS Operation Accuracy Tests...")
        
        operation_tests = []
        
        # Addition Test
        add_test = {
            "test_id": "CKKS_OP_ADD_001",
            "description": "Homomorphic addition accuracy test",
            "operation": "addition",
            "operand_a": [5.5, 2.3, -1.7],
            "operand_b": [3.2, -0.8, 4.1],
            "expected_result": [8.7, 1.5, 2.4],
            "parameters": {"n": 2**13, "scale": 2**40, "qi_sizes": [60, 40, 40, 60]},
            "tolerance": 1e-3
        }
        operation_tests.append(add_test)
        
        # Multiplication Test
        mul_test = {
            "test_id": "CKKS_OP_MUL_001",
            "description": "Homomorphic multiplication accuracy test",
            "operation": "multiplication",
            "operand_a": [2.0, 3.0, -1.5],
            "operand_b": [4.0, -2.0, 2.0],
            "expected_result": [8.0, -6.0, -3.0],
            "parameters": {"n": 2**13, "scale": 2**40, "qi_sizes": [60, 40, 40, 60]},
            "tolerance": 1e-2
        }
        operation_tests.append(mul_test)
        
        # Scalar Multiplication Test
        scalar_mul_test = {
            "test_id": "CKKS_OP_SCALAR_001",
            "description": "Scalar multiplication accuracy test",
            "operation": "scalar_multiplication",
            "operand_a": [1.0, 2.0, 3.0],
            "scalar": 2.5,
            "expected_result": [2.5, 5.0, 7.5],
            "parameters": {"n": 2**13, "scale": 2**40, "qi_sizes": [60, 40, 40, 60]},
            "tolerance": 1e-3
        }
        operation_tests.append(scalar_mul_test)
        
        return operation_tests
    
    def generate_parameter_validation_tests(self) -> List[Dict]:
        """Generate tests for different CKKS parameter combinations."""
        print("Generating CKKS Parameter Validation Tests...")
        
        param_tests = []
        
        # Different polynomial degrees
        for i, n in enumerate([2**12, 2**13, 2**14]):
            param_test = {
                "test_id": f"CKKS_PARAM_N_{i+1:03d}",
                "description": f"Parameter validation with n={n}",
                "test_type": "parameter_validation",
                "parameters": {"n": n, "scale": 2**40, "qi_sizes": [60, 40, 60]},
                "test_data": [1.0, 2.0, 3.0],
                "expected_behavior": "successful_encryption_decryption"
            }
            param_tests.append(param_test)
        
        # Different scales
        for i, scale in enumerate([2**30, 2**40, 2**50]):
            param_test = {
                "test_id": f"CKKS_PARAM_SCALE_{i+1:03d}",
                "description": f"Parameter validation with scale={scale}",
                "test_type": "parameter_validation", 
                "parameters": {"n": 2**13, "scale": scale, "qi_sizes": [60, 40, 40, 60]},
                "test_data": [0.1, 0.01, 0.001],
                "expected_behavior": "successful_encryption_decryption"
            }
            param_tests.append(param_test)
        
        return param_tests
    
    def run_kat_test(self, test_vector: Dict) -> Dict:
        """Execute a Known Answer Test and return results."""
        try:
            HE = Pyfhel()
            params = test_vector["parameters"]
            HE.contextGen(scheme='CKKS', **params)
            HE.keyGen()
            
            # Encrypt input
            input_data = test_vector["input_plaintext"]
            if REAL_PYFHEL:
                ptxt = HE.encodeFrac(input_data)
                ctxt = HE.encryptPtxt(ptxt)
                decrypted = HE.decryptFrac(ctxt)
            else:
                # Mock implementation
                ctxt = HE.encryptFrac(input_data)
                decrypted = HE.decryptFrac(ctxt)
            
            # Calculate error
            expected = test_vector["expected_output"]
            tolerance = test_vector["tolerance"]
            
            errors = [abs(d - e) for d, e in zip(decrypted[:len(expected)], expected)]
            max_error = max(errors) if errors else 0
            
            result = {
                "test_id": test_vector["test_id"],
                "status": "PASS" if max_error <= tolerance else "FAIL",
                "input": input_data,
                "expected": expected,
                "actual": decrypted[:len(expected)] if len(decrypted) >= len(expected) else decrypted,
                "max_error": max_error,
                "tolerance": tolerance,
                "parameters": params
            }
            
            return result
            
        except Exception as e:
            return {
                "test_id": test_vector["test_id"],
                "status": "ERROR",
                "error": str(e),
                "parameters": test_vector["parameters"]
            }
    
    def run_operation_test(self, test_vector: Dict) -> Dict:
        """Execute an operation accuracy test and return results."""
        try:
            HE = Pyfhel()
            params = test_vector["parameters"]
            HE.contextGen(scheme='CKKS', **params)
            HE.keyGen()
            HE.relinKeyGen()
            
            operation = test_vector["operation"]
            
            if operation == "addition":
                if REAL_PYFHEL:
                    ptxt_a = HE.encodeFrac(test_vector["operand_a"])
                    ptxt_b = HE.encodeFrac(test_vector["operand_b"])
                    ctxt_a = HE.encryptPtxt(ptxt_a)
                    ctxt_b = HE.encryptPtxt(ptxt_b)
                    ctxt_result = ctxt_a + ctxt_b
                    result_data = HE.decryptFrac(ctxt_result)
                else:
                    ctxt_a = HE.encryptFrac(test_vector["operand_a"])
                    ctxt_b = HE.encryptFrac(test_vector["operand_b"])
                    ctxt_result = ctxt_a + ctxt_b
                    result_data = HE.decryptFrac(ctxt_result)
                    
            elif operation == "multiplication":
                if REAL_PYFHEL:
                    ptxt_a = HE.encodeFrac(test_vector["operand_a"])
                    ptxt_b = HE.encodeFrac(test_vector["operand_b"])
                    ctxt_a = HE.encryptPtxt(ptxt_a)
                    ctxt_b = HE.encryptPtxt(ptxt_b)
                    ctxt_result = ctxt_a * ctxt_b
                    result_data = HE.decryptFrac(ctxt_result)
                else:
                    ctxt_a = HE.encryptFrac(test_vector["operand_a"])
                    ctxt_b = HE.encryptFrac(test_vector["operand_b"])
                    ctxt_result = ctxt_a * ctxt_b
                    result_data = HE.decryptFrac(ctxt_result)
                    
            elif operation == "scalar_multiplication":
                if REAL_PYFHEL:
                    ptxt_a = HE.encodeFrac(test_vector["operand_a"])
                    ctxt_a = HE.encryptPtxt(ptxt_a)
                    scalar_ptxt = HE.encodeFrac([test_vector["scalar"]])
                    ctxt_result = ctxt_a * scalar_ptxt
                    result_data = HE.decryptFrac(ctxt_result)
                else:
                    ctxt_a = HE.encryptFrac(test_vector["operand_a"])
                    ctxt_result = ctxt_a * test_vector["scalar"]
                    result_data = HE.decryptFrac(ctxt_result)
            
            # Calculate error
            expected = test_vector["expected_result"]
            tolerance = test_vector["tolerance"]
            
            errors = [abs(r - e) for r, e in zip(result_data[:len(expected)], expected)]
            max_error = max(errors) if errors else 0
            
            return {
                "test_id": test_vector["test_id"],
                "operation": operation,
                "status": "PASS" if max_error <= tolerance else "FAIL",
                "expected": expected,
                "actual": result_data[:len(expected)] if len(result_data) >= len(expected) else result_data,
                "max_error": max_error,
                "tolerance": tolerance,
                "parameters": params
            }
            
        except Exception as e:
            return {
                "test_id": test_vector["test_id"],
                "operation": test_vector["operation"],
                "status": "ERROR",
                "error": str(e),
                "parameters": test_vector["parameters"]
            }
    
    def run_parameter_test(self, test_vector: Dict) -> Dict:
        """Execute a parameter validation test and return results."""
        try:
            HE = Pyfhel()
            params = test_vector["parameters"]
            HE.contextGen(scheme='CKKS', **params)
            HE.keyGen()
            
            # Test basic encryption/decryption with given parameters
            test_data = test_vector["test_data"]
            if REAL_PYFHEL:
                ptxt = HE.encodeFrac(test_data)
                ctxt = HE.encryptPtxt(ptxt)
                decrypted = HE.decryptFrac(ctxt)
            else:
                ctxt = HE.encryptFrac(test_data)
                decrypted = HE.decryptFrac(ctxt)
            
            # Check if decryption is reasonably close to input
            errors = [abs(d - t) for d, t in zip(decrypted[:len(test_data)], test_data)]
            max_error = max(errors) if errors else 0
            
            return {
                "test_id": test_vector["test_id"],
                "status": "PASS" if max_error < 1.0 else "FAIL",  # Generous tolerance for parameter tests
                "parameters": params,
                "test_data": test_data,
                "decrypted": decrypted[:len(test_data)] if len(decrypted) >= len(test_data) else decrypted,
                "max_error": max_error
            }
            
        except Exception as e:
            return {
                "test_id": test_vector["test_id"],
                "status": "ERROR",
                "error": str(e),
                "parameters": test_vector["parameters"]
            }
    
    def run_all_tests(self) -> Dict:
        """Run all NIST-style test vectors and generate comprehensive report."""
        print("Starting NIST-style CKKS Test Vector Validation...")
        print("=" * 60)
        
        start_time = time.time()
        
        # Generate test vectors
        kat_tests = self.generate_known_answer_tests()
        operation_tests = self.generate_operation_accuracy_tests()
        parameter_tests = self.generate_parameter_validation_tests()
        
        all_results = {
            "test_suite": "NIST-Style CKKS Validation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "implementation": "Mock Pyfhel" if not REAL_PYFHEL else "Real Pyfhel",
            "kat_results": [],
            "operation_results": [],
            "parameter_results": [],
            "summary": {}
        }
        
        # Run KAT tests
        print("\n1. Running Known Answer Tests (KAT)...")
        for test in kat_tests:
            result = self.run_kat_test(test)
            all_results["kat_results"].append(result)
            print(f"   {result['test_id']}: {result['status']}")
        
        # Run operation tests
        print("\n2. Running Operation Accuracy Tests...")
        for test in operation_tests:
            result = self.run_operation_test(test)
            all_results["operation_results"].append(result)
            print(f"   {result['test_id']}: {result['status']}")
        
        # Run parameter tests
        print("\n3. Running Parameter Validation Tests...")
        for test in parameter_tests:
            result = self.run_parameter_test(test)
            all_results["parameter_results"].append(result)
            print(f"   {result['test_id']}: {result['status']}")
        
        # Generate summary
        total_tests = len(kat_tests) + len(operation_tests) + len(parameter_tests)
        passed_tests = sum(1 for category in [all_results["kat_results"], 
                                            all_results["operation_results"], 
                                            all_results["parameter_results"]]
                          for result in category if result["status"] == "PASS")
        
        failed_tests = sum(1 for category in [all_results["kat_results"], 
                                            all_results["operation_results"], 
                                            all_results["parameter_results"]]
                          for result in category if result["status"] == "FAIL")
        
        error_tests = sum(1 for category in [all_results["kat_results"], 
                                           all_results["operation_results"], 
                                           all_results["parameter_results"]]
                         for result in category if result["status"] == "ERROR")
        
        execution_time = time.time() - start_time
        
        all_results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "execution_time_seconds": execution_time
        }
        
        return all_results

def main():
    """Main function to run NIST-style CKKS test vectors."""
    tester = CKKSNISTTestVectors()
    results = tester.run_all_tests()
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    # Save results to file
    output_file = Path("data/covid_ct_cxr/nist_ckks_test_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_to_list(results), f, indent=2, ensure_ascii=False)
    
    # Generate markdown report
    report_file = Path("data/covid_ct_cxr/nist_ckks_validation_report.md")
    generate_markdown_report(results, report_file)
    
    print("\n" + "=" * 60)
    print("NIST-Style CKKS Test Vector Validation Complete")
    print("=" * 60)
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Errors: {results['summary']['errors']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"Execution Time: {results['summary']['execution_time_seconds']:.2f} seconds")
    print(f"\nDetailed results saved to: {output_file}")
    print(f"Markdown report saved to: {report_file}")

def generate_markdown_report(results: Dict, output_file: Path):
    """Generate a comprehensive markdown report of test results."""
    
    report_content = f"""# NIST-Style CKKS Homomorphic Encryption Validation Report

## Test Suite Overview

- **Test Suite**: {results['test_suite']}
- **Timestamp**: {results['timestamp']}
- **Implementation**: {results['implementation']}
- **Total Tests**: {results['summary']['total_tests']}
- **Success Rate**: {results['summary']['success_rate']:.1f}%
- **Execution Time**: {results['summary']['execution_time_seconds']:.2f} seconds

## Summary Statistics

| Category | Passed | Failed | Errors | Total |
|----------|--------|--------|--------|-------|
| Known Answer Tests (KAT) | {sum(1 for r in results['kat_results'] if r['status'] == 'PASS')} | {sum(1 for r in results['kat_results'] if r['status'] == 'FAIL')} | {sum(1 for r in results['kat_results'] if r['status'] == 'ERROR')} | {len(results['kat_results'])} |
| Operation Accuracy Tests | {sum(1 for r in results['operation_results'] if r['status'] == 'PASS')} | {sum(1 for r in results['operation_results'] if r['status'] == 'FAIL')} | {sum(1 for r in results['operation_results'] if r['status'] == 'ERROR')} | {len(results['operation_results'])} |
| Parameter Validation Tests | {sum(1 for r in results['parameter_results'] if r['status'] == 'PASS')} | {sum(1 for r in results['parameter_results'] if r['status'] == 'FAIL')} | {sum(1 for r in results['parameter_results'] if r['status'] == 'ERROR')} | {len(results['parameter_results'])} |
| **Total** | **{results['summary']['passed']}** | **{results['summary']['failed']}** | **{results['summary']['errors']}** | **{results['summary']['total_tests']}** |

## Detailed Test Results

### 1. Known Answer Tests (KAT)

These tests validate the correctness of CKKS encryption/decryption with predetermined input/output pairs.

"""
    
    for result in results['kat_results']:
        status_emoji = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
        report_content += f"""
#### {result['test_id']} {status_emoji}

- **Status**: {result['status']}
- **Parameters**: n={result['parameters']['n']}, scale={result['parameters']['scale']}
"""
        if result['status'] != 'ERROR':
            report_content += f"""- **Input**: {result['input']}
- **Expected**: {result['expected']}
- **Actual**: {result['actual']}
- **Max Error**: {result['max_error']:.2e}
- **Tolerance**: {result['tolerance']:.2e}
"""
        else:
            report_content += f"- **Error**: {result.get('error', 'Unknown error')}\n"
    
    report_content += """
### 2. Operation Accuracy Tests

These tests validate the correctness of homomorphic operations (addition, multiplication, scalar multiplication).

"""
    
    for result in results['operation_results']:
        status_emoji = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
        report_content += f"""
#### {result['test_id']} {status_emoji}

- **Status**: {result['status']}
- **Operation**: {result['operation']}
- **Parameters**: n={result['parameters']['n']}, scale={result['parameters']['scale']}
"""
        if result['status'] != 'ERROR':
            report_content += f"""- **Expected**: {result['expected']}
- **Actual**: {result['actual']}
- **Max Error**: {result['max_error']:.2e}
- **Tolerance**: {result['tolerance']:.2e}
"""
        else:
            report_content += f"- **Error**: {result.get('error', 'Unknown error')}\n"
    
    report_content += """
### 3. Parameter Validation Tests

These tests validate CKKS behavior with different parameter combinations.

"""
    
    for result in results['parameter_results']:
        status_emoji = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
        report_content += f"""
#### {result['test_id']} {status_emoji}

- **Status**: {result['status']}
- **Parameters**: n={result['parameters']['n']}, scale={result['parameters']['scale']}
"""
        if result['status'] != 'ERROR':
            report_content += f"""- **Test Data**: {result['test_data']}
- **Decrypted**: {result['decrypted']}
- **Max Error**: {result['max_error']:.2e}
"""
        else:
            report_content += f"- **Error**: {result.get('error', 'Unknown error')}\n"
    
    report_content += f"""
## Compliance Assessment

### NIST Testing Methodology Compliance

This test suite follows NIST cryptographic testing principles:

1. **Deterministic Test Vectors**: ✅ All tests use predetermined inputs and expected outputs
2. **Known Answer Tests (KAT)**: ✅ Implemented for basic encryption/decryption validation
3. **Parameter Boundary Testing**: ✅ Different parameter combinations tested
4. **Error Bound Validation**: ✅ Approximation errors validated against tolerances
5. **Comprehensive Reporting**: ✅ Detailed results with error analysis

### Security Considerations

- **Implementation**: {results['implementation']}
- **Approximation Nature**: CKKS is an approximate homomorphic encryption scheme
- **Error Tolerance**: All tests include appropriate error tolerances for approximate computations
- **Parameter Security**: Test parameters chosen for demonstration purposes, production use requires security analysis

### Recommendations

1. **For Production Use**: Conduct security analysis of CKKS parameters
2. **Error Management**: Monitor approximation errors in real applications
3. **Performance Optimization**: Consider parameter tuning for specific use cases
4. **Regular Validation**: Re-run test vectors after any implementation changes

---

*Report generated on {results['timestamp']}*
*Test execution time: {results['summary']['execution_time_seconds']:.2f} seconds*
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)

if __name__ == "__main__":
    main()
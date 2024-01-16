1. replace `cat_enc\Lib\site-packages\statsmodels\regression\mixed_linear_model.py` as `mixed_linear_model.py` to avoid the error. 
2. replace `cat_enc\Lib\site-packages\scipy\linalg\_decomp_svd.py` as `_decomp_svd.py` to avoid the error.

# 1
-StudentPerformance-GLMM
`numpy.linalg.LinAlgError: singular matrix` can be fixed by using the pseudo-inverse

```python
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            raise ValueError("Cannot predict random effects from " +
                             "singular covariance structure.")
```
=>
```python
        try:
            cov_re_inv = np.linalg.inv(self.cov_re)
        except np.linalg.LinAlgError:
            # raise ValueError("Cannot predict random effects from " +
            #                  "singular covariance structure.")
            warnings.warn(
                "Cannot predict random effects from singular covariance structure."
                " Using pseudo-inverse."
            )
            cov_re_inv = np.linalg.pinv(self.cov_re)
```

# 2
5-UkAir-Similarity-RidgeCV, 
```python
if info > 0:
        raise LinAlgError("SVD did not converge")
```
=>
```python
import warnings 
        warnings.warn("numpy.linalg.LinAlgError: SVD did not converge."
                    "Tring to use general rectangular approach (``'gesvd'``)")
        return svd(a, full_matrices=full_matrices, compute_uv=compute_uv, overwrite_a=overwrite_a,
        check_finite=check_finite, lapack_driver='gesvd')
        # raise LinAlgError("SVD did not converge")
```
# Machine Learning Module - Developer Guide

## Module Overview

The `src/machineLearning/` module provides a unified, production-ready machine learning pipeline supporting both classification and regression tasks. It abstracts dataset conversion, feature scaling, model training, evaluation, and artifact persistence through a clean, extensible architecture built on PyTorch and scikit-learn.

**Core Philosophy**: Type-safe, validated data flows through composable pipeline stages with automatic task detection and standardized artifact storage.

## Architecture & Components

### 1. Dataset Layer (`dataset.py`)

**MLCompatibleDataset (Pydantic Model)**
- Schema-validated container for ML data with mandatory `data` (2D array) and `target` (1D array)
- Optional metadata: `feature_names`, `target_names`, `descr`
- Automatic coercion: `data` → float32 (2D), `target` → 1D with shape validation
- Validation: NaN/infinite checks, length consistency between features/targets

**MLDatasetConverter (Static Factory)**
Three conversion pathways:
1. `from_sklearn_bunch()`: Direct sklearn dataset integration (iris, diabetes, etc.)
2. `from_dataframe()`: Pandas DataFrame with `data`/`target` columns (array elements stacked → 2D)
3. `from_csv()`: File-based loading with feature/target column specification + optional NA handling
4. `convert()`: Auto-dispatch based on source type (unified entry point)

**Usage Pattern**:
```python
from machineLearning.dataset import MLDatasetConverter
from sklearn.datasets import load_iris

# From sklearn
ds = MLDatasetConverter.convert(load_iris())

# From CSV
ds = MLDatasetConverter.convert(
    "data.csv", 
    features=["sepal_length", "sepal_width"], 
    target="species"
)
```

### 2. Model Layer (`models/`)

**BaseMLModel (Abstract Base Class)**
Shared preprocessing infrastructure:
- Data splitting with stratification support (classification only)
- Feature scaling via `StandardScaler` (fit on train, transform on test)
- Scaler persistence to experiment directory
- TensorDataset creation and DataLoader generation
- Abstract methods: `train_model()`, `evaluate_model()`, `convert_to_tensor_datasets()`

**ClassificationMLModel**
- Automatic class count detection from `target`
- Loss: `CrossEntropyLoss`, Optimizer: SGD (lr=0.1, momentum=0.9)
- Metrics: Accuracy per epoch, test accuracy
- Target dtype: `torch.long` for class indices

**RegressionMLModel**
- Output dimension: 1 (single continuous value)
- Loss: `MSELoss`, Optimizer: SGD (lr=0.01, momentum=0.9) with gradient clipping
- Metrics: R² score (both train and test)
- Target dtype: `torch.float32` with shape (N, 1)

**Task Detection Heuristic** (`pipeline._decide_task_type`):
- Classification: ≤20 unique integer values in target
- Regression: >10 unique values or non-integer dtype

### 3. Neural Network (`simple_nn.py`)

**SimpleNeuralNetwork (PyTorch Module)**
```
Input (N features) → Linear → ReLU → Linear → Output (K classes or 1 value)
```
- Configurable: `input_dim`, `hidden_dim`, `output_dim`
- Shared across classification/regression (output interpretation differs)
- Default: 4 → 16 → 3 (Iris dataset dimensions)

### 4. Pipeline Orchestration (`pipeline.py`)

**execute_machine_learning_pipeline()**
End-to-end training workflow:
1. Task type auto-detection → model instantiation
2. Train/test split (stratified for classification)
3. Feature scaling + scaler persistence
4. TensorDataset conversion + DataLoader creation
5. Training loop with history tracking
6. Returns: `(model_wrapper, nn.Module, acc_history, loss_history, experiment_name)`

**train_and_save_pipeline()**
Combines execution + persistence:
- Calls `execute_machine_learning_pipeline()`
- Invokes `store_model_and_learning_logs()` for artifact storage
- Single function for complete train-to-disk workflow

**Parameters**:
- `dataset`: MLCompatibleDataset
- `dataset_name`: Human-readable identifier (e.g., "iris", "diabetes")
- `epochs`: Training iterations (default: 5)
- `learning_rate`: Optional LR override
- `experiment_name`: Optional custom folder name (default: timestamp)
- `log_dirs_root`: Optional output directory (default: `PATHS.ml_outputs`)

### 5. Artifact Management (`save_util.py`)

**store_model_and_learning_logs()**
Creates experiment directory with:
```
{log_dirs_root}/{experiment_name}/
├── model_param.pth          # state_dict
├── scaler.joblib            # StandardScaler
├── loss_curve.png           # Training loss visualization
├── acc_curve.png            # Accuracy/R² visualization
├── loss.csv                 # Epoch-by-epoch loss
└── acc.csv                  # Epoch-by-epoch accuracy/R²
```

Also appends to `{log_dirs_root}/train_log/trained_model.csv`:
- `dataset_name`, `epochs`, `experiment_name`, `timestamp`
- Central registry for all training runs

**Error Handling**: Fails gracefully per artifact type (logs warnings, only raises if all saves fail)

### 6. Batch Inference (`eval_batch.py`)

**evaluate_iris_batch()**
Production inference function (despite name, handles both classification/regression):
- Loads `state_dict` + scaler from disk
- Auto-detects model architecture from layer shapes
- Validates input: feature count, numeric types
- Returns:
  - **Classification**: List of class names (from `class_names` or defaults)
  - **Regression**: List of float strings (6 decimal places)

**Input Validation**:
- Non-empty data, correct feature count (matches scaler)
- Numeric values only (int/float)
- Model/scaler file existence

**Usage in API Layer** (`services/iris_service.py`):
```python
from machineLearning.eval_batch import evaluate_iris_batch

predictions = evaluate_iris_batch(
    input_data_list=[[5.1, 3.5, 1.4, 0.2]],
    model_path=PATHS.ml_outputs / "experiment/model_param.pth",
    scaler_path=PATHS.ml_outputs / "experiment/scaler.joblib"
)
# Returns: ["Iris setosa"]
```

## Directory Structure

```
machineLearning/
├── __init__.py                # Empty module marker
├── dataset.py                 # MLCompatibleDataset + converters
├── pipeline.py                # Training orchestration
├── save_util.py               # Artifact persistence
├── eval_batch.py              # Inference endpoint
├── simple_nn.py               # PyTorch network definition
├── models/
│   ├── base_model.py         # Abstract base class
│   ├── classification_model.py
│   └── regression_model.py
├── examples/
│   ├── train_iris.py         # Classification demo
│   ├── train_diabetes.py     # Regression demo
│   └── ml_dataset_from_csv_sample.py
└── README.md                 # Legacy documentation
```

## Integration Points

### Flask API Endpoints (`src/server/api/iris.py`)

**`/api/iris` (POST)**
Single prediction:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "sepal_length_cm": 1.4,
  "sepal_width_cm": 0.2
}
```
Response: `{"prediction": "Iris setosa"}`

**`/api/iris-batch` (POST)**
Batch prediction:
```json
{
  "data": [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5]
  ]
}
```
Response: `{"predictions": ["Iris setosa", "Iris versicolor"]}`

### Service Layer (`src/services/iris_service.py`)
- LRU-cached wrappers around `evaluate_iris_batch()`
- Caches single/batch predictions to avoid redundant model loads
- Tuple conversion for hashable cache keys

## Development Patterns

### Adding New Datasets

1. **Prepare Data Source**:
   - sklearn Bunch / CSV / DataFrame format
   - Ensure numeric features, valid target column

2. **Convert to MLCompatibleDataset**:
   ```python
   from machineLearning.dataset import MLDatasetConverter
   
   dataset = MLDatasetConverter.convert(
       source="path/to/data.csv",
       features=["col1", "col2"],
       target="label"
   )
   ```

3. **Train Using Pipeline**:
   ```python
   from machineLearning.pipeline import train_and_save_pipeline
   
   model, net, hist, loss, name = train_and_save_pipeline(
       dataset=dataset,
       dataset_name="my_dataset",
       epochs=20
   )
   ```

4. **Inference**:
   ```python
   from machineLearning.eval_batch import evaluate_iris_batch
   
   predictions = evaluate_iris_batch(
       input_data_list=[[1.0, 2.0]],
       model_path=f"outputs/machineLearning/{name}/model_param.pth",
       scaler_path=f"outputs/machineLearning/{name}/scaler.joblib"
   )
   ```

### Custom Model Architectures

To use different neural networks:
1. Create new PyTorch `nn.Module` in `models/`
2. Override `BaseMLModel` or classification/regression models
3. Update `_build_model()` in `pipeline.py` if task detection needs changes

**Current Limitation**: `SimpleNeuralNetwork` is hardcoded in model classes. Refactor to accept custom architectures if needed.

### Extending Metrics

Classification: Add metrics to `ClassificationMLModel.train_model()`:
```python
# Example: Add precision/recall
from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_true, y_pred, average='macro')
```

Regression: Similar approach in `RegressionMLModel.train_model()`:
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true_np, y_pred_np)
```

## Testing

**Test Coverage** (`tests/machineLearning/`):
- `test_dataset_converter.py`: Dataset conversion validation
- Future: Add tests for pipeline, models, inference

**Running Tests**:
```bash
uv run pytest tests/machineLearning/ -v
```

**Key Test Scenarios**:
1. Dataset conversion from all source types
2. Invalid data handling (empty, NaN, shape mismatches)
3. End-to-end pipeline execution (small epochs)
4. Inference with mismatched feature counts
5. Scaler/model file loading

## Common Issues & Solutions

### Issue: "Length mismatch: len(target)=X vs n_samples (data)=Y"
**Cause**: Inconsistent row counts between features and target
**Solution**: Check CSV/DataFrame preprocessing; ensure no accidental filtering

### Issue: "data must be 2D array-like"
**Cause**: DataFrame column contains scalars instead of arrays
**Solution**: For DataFrame input, ensure `data` column contains list/array elements:
```python
df["data"] = df[["col1", "col2"]].values.tolist()
```

### Issue: Gradient explosion in regression
**Symptoms**: Loss → NaN, predictions → Inf
**Solution**: Already mitigated via:
- Lower learning rate (0.01 vs 0.1)
- Gradient clipping (`max_norm=5.0`)
- Consider target normalization if range is extreme

### Issue: Model/scaler file not found during inference
**Cause**: Incorrect path construction or experiment name mismatch
**Solution**: 
- Check `outputs/machineLearning/{experiment_name}/` structure
- Verify paths passed to `evaluate_iris_batch()`
- Use `PATHS.ml_outputs` from `config.paths` for consistency

## Future Enhancements

1. **Model Registry**: Centralized metadata store (experiment params, metrics)
2. **Custom Architectures**: Plugin system for user-defined networks
3. **Hyperparameter Tuning**: Grid search / Optuna integration
4. **Advanced Metrics**: Confusion matrices, ROC curves for classification
5. **Multi-output Support**: Vector regression, multi-label classification
6. **Deployment**: ONNX export for production inference optimization

## Related Modules

- `src/config/paths.py`: Defines `PATHS.ml_outputs` for artifact storage
- `src/server/api/iris.py`: Flask endpoints consuming this module
- `src/services/iris_service.py`: Caching layer for predictions
- `src/util.csv_plot/csv_util.py`: CSV reading utilities used by converter

## Key Dependencies

- **PyTorch**: Neural network training/inference
- **scikit-learn**: Preprocessing (StandardScaler), model selection (train_test_split), datasets
- **Pydantic**: Schema validation for `MLCompatibleDataset`
- **loguru**: Structured logging throughout pipeline
- **joblib**: Scaler serialization
- **matplotlib**: Learning curve visualization

## Coding Standards

All code follows project-wide standards:
- **Type hints**: All function signatures annotated
- **Path handling**: `pathlib.Path` exclusively (no `os.path`)
- **Logging**: `loguru.logger` (not `print` or built-in `logging`)
- **Validation**: Pydantic models for data structures
- **Error handling**: Explicit validation with clear error messages
- **Testing**: Prefer `pytest` with descriptive test names

See `.github/instructions/modern.instructions.md` for comprehensive Python patterns.

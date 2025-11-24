# Next-Location Prediction - Research Findings

## Dataset Analysis
- **Geolife**: 1167 unique locations, 45 users, avg seq length 18
- **Test-Train Distribution Mismatch**: Critical issue identified
  - Validation Acc@1: ~35-41%
  - Test Acc@1: ~25-32% (9-10% drop!)
  - Indicates significant distributional shift between validation and test sets

## Key Problem: Long-Tail Distribution
- Location 1 appears in 22.79% of test samples but model NEVER predicts it
- Model predicts only 42 unique locations vs 315 unique targets in test
- High frequency bias - model heavily biased toward frequent training locations

## Approaches Attempted

### 1. Basic Transformer (471K params)
- d_model=96, 3 layers
- Result: 31.87% test Acc@1
- Issue: Severe overfitting (74% train vs 32% test)

### 2. Improved Transformer with Attention Pooling (472K params)  
- Added frequency features
- Better temporal encoding
- Attention-based pooling
- Result: 32.18% test Acc@1
- Issue: Still severe overfitting (87% train vs 32% test)

### 3. Simplified Architecture (464K params)
- Reduced complexity, learnable positional embeddings
- Recency-weighted pooling
- Result: 29.73% test Acc@1
- Issue: Worse performance, model too simple

### 4. Class-Balanced Focal Loss
- Effective number of samples weighting
- Result: 0% test Acc@1 (complete collapse)
- Issue: Too aggressive reweighting

### 5. Weighted Cross-Entropy (sqrt inverse frequency)
- Softer class weighting
- Result: 25.16% test Acc@1  
- Issue: Still unable to predict rare classes

## Root Causes
1. **Extreme class imbalance** with long-tail distribution
2. **Test set contains many rare locations** not well-represented in training
3. **User-specific patterns** may differ significantly between splits
4. **Model learns frequency bias** rather than sequential patterns

## Recommended Next Steps
1. Investigate user-level cross-validation splits
2. Implement mixed objective: combine cross-entropy with ranking loss
3. Use contrastive learning to better separate location embeddings
4. Consider hierarchical models (coarse-to-fine location prediction)
5. Ensemble with frequency-based baseline

## Parameter Constraints Met
- Geolife: All models <500K parameters ✓
- DIY: Target <1M parameters (not yet trained)

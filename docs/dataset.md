# <div align="center">Wine Quality Dataset Documentation</div>

<div align="justify">

## Overview

This dataset contains physicochemical properties and quality ratings for red wine samples. It's commonly used for regression and classification tasks in machine learning to predict wine quality based on various chemical characteristics.

## Dataset Information

- **Dataset Name**: Wine Quality - Red Wine
- **File**: `winequality-red.csv`
- **Format**: CSV (Comma-Separated Values)
- **Total Records**: 1,599 samples (excluding header)
- **Total Features**: 12 (11 input features + 1 target variable)
- **Missing Values**: None
- **Data Type**: Numerical (continuous and discrete)

## Source and Origin

This dataset is part of the UCI Machine Learning Repository's Wine Quality datasets, originally created by:

- **Creators**: Paulo Cortez, University of Minho, Guimarães, Portugal
- **Contributors**: A. Cerdeira, F. Almeida, T. Matos, J. Reis
- **Publication**: "Modeling wine preferences by data mining from physicochemical properties" (2009)
- **Original Source**: UCI Machine Learning Repository

## Feature Descriptions

### Input Features (11)

1. **Fixed Acidity** (g/L)

   - Most acids involved with wine that are fixed or non-volatile
   - Range: Typically 4.6 - 15.9
   - Unit: grams per liter (tartaric acid)
   - Impact: Affects taste, preservation, and color stability

2. **Volatile Acidity** (g/L)

   - Amount of acetic acid in wine
   - Range: Typically 0.12 - 1.58
   - Unit: grams per liter (acetic acid)
   - Impact: High levels can lead to unpleasant vinegar taste

3. **Citric Acid** (g/L)

   - Found in small quantities, adds freshness and flavor
   - Range: Typically 0.00 - 1.00
   - Unit: grams per liter
   - Impact: Can add freshness and flavor to wines

4. **Residual Sugar** (g/L)

   - Amount of sugar remaining after fermentation
   - Range: Typically 0.9 - 15.5
   - Unit: grams per liter
   - Impact: Determines sweetness level; most wines are dry (< 4 g/L)

5. **Chlorides** (g/L)

   - Amount of salt in the wine
   - Range: Typically 0.012 - 0.611
   - Unit: grams per liter (sodium chloride)
   - Impact: High levels can make wine taste salty

6. **Free Sulfur Dioxide** (mg/L)

   - Free form of SO2 in equilibrium between molecular SO2 and bisulfite ion
   - Range: Typically 1 - 72
   - Unit: milligrams per liter
   - Impact: Prevents microbial growth and oxidation

7. **Total Sulfur Dioxide** (mg/L)

   - Amount of free and bound forms of sulfur dioxide
   - Range: Typically 6 - 289
   - Unit: milligrams per liter
   - Impact: Low concentrations are undetectable; high concentrations are evident

8. **Density** (g/cm³)

   - Density of the wine
   - Range: Typically 0.990 - 1.004
   - Unit: grams per cubic centimeter
   - Impact: Close to water density depending on alcohol and sugar content

9. **pH**

   - Describes acidity/alkalinity on a scale of 0-14
   - Range: Typically 2.74 - 4.01
   - Unit: pH scale (logarithmic)
   - Impact: Most wines are between 3-4 on the pH scale

10. **Sulphates** (g/L)

    - Wine additive that contributes to sulfur dioxide levels
    - Range: Typically 0.33 - 2.00
    - Unit: grams per liter (potassium sulphate)
    - Impact: Acts as antimicrobial and antioxidant

11. **Alcohol** (% vol)
    - Alcohol content of the wine
    - Range: Typically 8.4 - 14.9
    - Unit: percentage by volume
    - Impact: Major factor in wine character and quality perception

### Target Variable

**Quality** (Score)

- Sensory quality rating based on expert evaluation
- Range: 3 - 8 (in this dataset)
- Scale: 0 (very bad) to 10 (very excellent)
- Type: Discrete integer values
- Distribution: Most wines rated between 5-6 (median quality)

## Data Distribution Analysis

### Quality Distribution

- **Quality 3**: Rare (lowest quality in dataset)
- **Quality 4**: Low frequency
- **Quality 5**: High frequency (most common)
- **Quality 6**: High frequency (second most common)
- **Quality 7**: Moderate frequency
- **Quality 8**: Low frequency (highest quality in dataset)

### Statistical Summary

- The dataset is imbalanced with most wines rated as average quality (5-6)
- No wines rated below 3 or above 8 in this particular dataset
- Most features follow approximately normal distributions
- Some features may have outliers that require preprocessing

## Data Quality Assessment

### Strengths

- **Complete Data**: No missing values
- **Consistent Format**: All numerical features with consistent units
- **Real-world Data**: Based on actual Portuguese "Vinho Verde" wine samples
- **Expert Evaluation**: Quality ratings from wine experts
- **Sufficient Size**: 1,599 samples provide good statistical power

### Limitations

- **Imbalanced Classes**: Uneven distribution of quality ratings
- **Limited Range**: Quality scores only span 3-8 (not full 0-10 scale)
- **Single Wine Type**: Only red wine variants
- **Geographic Limitation**: Only Portuguese wines
- **Temporal Scope**: Data from a specific time period

## Use Cases and Applications

### Machine Learning Tasks

1. **Regression**: Predict exact quality score (3-8)
2. **Classification**: Classify wines into quality categories (low/medium/high)
3. **Feature Analysis**: Identify most important factors affecting quality
4. **Outlier Detection**: Find unusual wine samples
5. **Clustering**: Group wines by similar characteristics

### Business Applications

- **Quality Control**: Automated wine quality assessment
- **Production Optimization**: Adjust chemical composition for better quality
- **Market Positioning**: Price wines based on predicted quality
- **Inventory Management**: Prioritize high-quality wine storage

## Preprocessing Recommendations

### Feature Engineering

- **Normalization/Standardization**: Scale features to similar ranges
- **Outlier Treatment**: Handle extreme values in chemical properties
- **Feature Interactions**: Create ratio features (e.g., free/total sulfur dioxide)
- **Binning**: Convert continuous quality scores to categorical classes

### Data Splits

- **Training Set**: 70-80% for model development
- **Validation Set**: 10-15% for hyperparameter tuning
- **Test Set**: 10-15% for final evaluation
- **Stratification**: Maintain quality distribution across splits

### Class Imbalance Handling

- **Oversampling**: SMOTE for minority classes
- **Undersampling**: Random sampling of majority classes
- **Class Weights**: Adjust model weights for imbalanced classes
- **Ensemble Methods**: Use balanced random forests

## Related Datasets

- **White Wine Quality Dataset**: Companion dataset with white wine samples
- **Wine Recognition Dataset**: Different wine classification task
- **Wine Reviews Dataset**: Text-based wine reviews and ratings

## Citation

If using this dataset in research or publications, please cite:

```
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
```

## License and Usage Rights

This dataset is publicly available through the UCI Machine Learning Repository and can be used for:

- Educational purposes
- Research and academic publications
- Commercial applications (check specific terms)
- Open-source projects

## Technical Notes

### File Format Details

- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Header Row**: Yes (feature names)
- **Decimal Separator**: Period (.)
- **Line Endings**: CRLF (Windows style)

### Data Types

- All features: Floating-point numbers
- Target variable: Integer (quality scores)
- No categorical variables in original dataset

## Contact and Support

For questions about this dataset documentation or analysis:

- Check the original UCI repository for source information
- Refer to the cited research paper for methodological details
- Review project README.md for implementation-specific information

</div>

---

<div align="center">

_Last Updated: **July 2025**_ - _Dataset Version: **Original UCI Repository Version**_

</div>

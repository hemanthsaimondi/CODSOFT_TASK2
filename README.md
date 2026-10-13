# Task 2: Movie Rating Prediction

Regression model that predicts a movie's IMDb rating from genre, director, actors, duration and year.
Dataset: IMDb Movies India (15,509 rows, 7,919 with a rating).

## Approach
1. **Cleaning**: parsed Year "(2019)", Duration "109 min", Votes "1,234" into numbers; dropped rows with no rating
2. **Features**: multi-hot encoding of the 15 most common genres, number of genres, duration, year; director and 3 actors encoded with sklearn's cross-fitted `TargetEncoder` (avoids target leakage)
3. **Models compared** (5-fold CV, R²): Ridge, Random Forest, Gradient Boosting
4. **Two setups**: without Votes (only information known before release) and with Votes (popularity signal)

## Results (Gradient Boosting, 20% hold-out test set)
| Setup | MAE | RMSE | R² |
|---|---|---|---|
| Without Votes | 0.858 | 1.105 | 0.344 |
| With Votes | 0.777 | 1.024 | 0.436 |

Baseline (always predict the mean rating): MAE 1.107.

## Run
```
pip install pandas numpy scikit-learn matplotlib seaborn
python movie_rating_prediction.py
```

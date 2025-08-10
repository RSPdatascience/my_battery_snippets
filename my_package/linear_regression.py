from sklearn.linear_model import LinearRegression

def add_linear_regression_predictions(df, x_field, y_field):
    # Extract features and target variable
    X = df[[x_field]]  # Independent variable
    y = df[y_field]  # Dependent variable

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X, y)

    # Get the slope (coefficient) and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    # Predicting values
    predictions = model.predict(X)

    # Add the predictions to the DataFrame
    df[f'{y_field} predicted'] = predictions

    # Print the equation
    print(f"The linear equation is: {y_field} = {slope:.2f} * {x_field} + {intercept:.2f}")

    return df, intercept, slope

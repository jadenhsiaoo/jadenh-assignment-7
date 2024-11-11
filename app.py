from flask import Flask, render_template, request, session, redirect, flash, url_for
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats
import uuid

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

# Dictionary to store large session data in memory
session_data_store = {}


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.rand(N)

    # Generate random dataset Y using specified beta0, beta1, mu, and sigma2
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # Fit a linear regression model to X and Y
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, color="blue", label="Data Points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.5, label="Simulated Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, label="Simulated Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed
    slope_more_extreme = np.mean(np.abs(slopes) >= abs(slope))
    intercept_more_extreme = np.mean(np.abs(intercepts) >= abs(intercept))

    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_more_extreme,
        slopes,
        intercepts,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return generate()
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    session["session_id"] = session_id

    # Get user input from the form
    N = int(request.form["N"])
    mu = float(request.form["mu"])
    sigma2 = float(request.form["sigma2"])
    beta0 = float(request.form["beta0"])
    beta1 = float(request.form["beta1"])
    S = int(request.form["S"])

    # Generate data and initial plots
    (
        X,
        Y,
        slope,
        intercept,
        plot1,
        plot2,
        slope_more_extreme,
        intercept_more_extreme,
        slopes,
        intercepts,
    ) = generate_data(N, mu, beta0, beta1, sigma2, S)

    # Store only essential small values in session cookies, use in-memory storage for large data
    session_data_store[session_id] = {
        "X": X,
        "Y": Y,
        "slope": slope,
        "intercept": intercept,
        "slopes": slopes,
        "intercepts": intercepts,
        "slope_more_extreme": slope_more_extreme,
        "intercept_more_extreme": intercept_more_extreme,
        "N": N,
        "mu": mu,
        "sigma2": sigma2,
        "beta0": beta0,
        "beta1": beta1,
        "S": S
    }

    return render_template(
        "index.html",
        plot1=plot1,
        plot2=plot2,
        slope_more_extreme=slope_more_extreme,
        intercept_more_extreme=intercept_more_extreme,
        slope=slope,
        intercept=intercept,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve session_id from the session and corresponding data from session_data_store
    session_id = session.get("session_id")
    if session_id not in session_data_store:
        flash("Please generate data before running hypothesis testing.")
        return redirect(url_for("index"))

    data = session_data_store[session_id]
    slope = data["slope"]
    intercept = data["intercept"]
    slopes = data["slopes"]
    intercepts = data["intercepts"]
    beta0 = data["beta0"]
    beta1 = data["beta1"]

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:  # Not equal to
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))

    # Fun message for small p-value
    fun_message = None
    if p_value <= 0.0001:
        fun_message = "This is a rare event with p-value <= 0.0001!"

    # Plot histogram of simulated statistics
    plt.figure()
    plt.hist(simulated_stats, bins=20, alpha=0.5)
    plt.axvline(observed_stat, color="red", linestyle="dashed", linewidth=2, label="Observed Stat")
    plt.axvline(hypothesized_value, color="blue", linestyle="dotted", linewidth=2, label="Hypothesized Value")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value,
        fun_message=fun_message,
        slope=slope,
        intercept=intercept,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve session_id from the session and corresponding data from session_data_store
    session_id = session.get("session_id")
    if session_id not in session_data_store:
        flash("Please generate data before calculating confidence intervals.")
        return redirect(url_for("index"))

    data = session_data_store[session_id]
    slopes = data["slopes"]
    intercepts = data["intercepts"]
    beta0 = data["beta0"]
    beta1 = data["beta1"]

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = data["slope"]
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = data["intercept"]
        true_param = beta0

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)

    # Calculate confidence interval for the parameter estimate
    ci_margin = std_estimate * scipy.stats.t.ppf((1 + confidence_level / 100) / 2, len(estimates) - 1)
    ci_lower = mean_estimate - ci_margin
    ci_upper = mean_estimate + ci_margin

    # Check if confidence interval includes true parameter
    includes_true = ci_lower <= true_param <= ci_upper

    # Plot the individual estimates and confidence interval
    plt.figure()
    plt.scatter(range(len(estimates)), estimates, color="gray", alpha=0.5)
    plt.axhline(mean_estimate, color="blue", label="Mean Estimate")
    plt.axhline(ci_lower, color="green", linestyle="dashed", label="Confidence Interval")
    plt.axhline(ci_upper, color="green", linestyle="dashed")
    plt.axhline(true_param, color="red", linestyle="dotted", label="True Parameter")
    plt.xlabel("Simulation Index")
    plt.ylabel("Estimate")
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
    )


if __name__ == "__main__":
    app.run(debug=True)

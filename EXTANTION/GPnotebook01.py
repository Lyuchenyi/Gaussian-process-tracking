from stonesoup.types.groundtruth import GroundTruthState
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from gaussianprocess import GaussianProcess
from GaussianProcessPredictor import GaussianProcessPredictor
from Generator import generate_groundtruth  # Import generate_groundtruth function

# Step 1: Define kernel function
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Radial Basis Function (RBF) Kernel
    """
    sqdist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Step 2: Configure parameters
kernel = rbf_kernel
num_points = 100  # Number of data points
win_size = 30  # Sliding window size
groundtruth_noise_std = 1
measurement_noise_std = 10 
selected_dim = "x,y"  # Options: "x", "y", "x,y"
start_time = datetime.now()

# Step 3: Generate high-density data
groundtruth_x, groundtruth_y = generate_groundtruth(num_points, mode="S1", noise_std = groundtruth_noise_std)  # Noise-free Ground Truth

# Add measurement noise to create measurements
 # Measurement noise standard deviation
measurement_x = groundtruth_x + np.random.normal(0, measurement_noise_std, num_points)
measurement_y = groundtruth_y + np.random.normal(0, measurement_noise_std, num_points)

timestamps = [start_time + timedelta(seconds=i) for i in range(num_points)]

# Wrap Ground Truth and Measurement data in GroundTruthState objects
groundtruth_states = [
    GroundTruthState(np.array([x, y]).reshape(-1, 1), timestamp=t)
    for x, y, t in zip(groundtruth_x, groundtruth_y, timestamps)
]

measurements = [
    GroundTruthState(np.array([x, y]).reshape(-1, 1), timestamp=t)
    for x, y, t in zip(measurement_x, measurement_y, timestamps)
]

# Step 4: Sliding window logic and prediction
gp = GaussianProcess(kernel=kernel)
predictions = []
prediction_timestamps = []

for i in range(5, len(timestamps)):  # Start from the 5th data point
    # Define sliding window range
    start_index = max(0, i - win_size)

    # Use measurements for training
    window_states = measurements[start_index:i]

    # Train on current window
    gp.train(window_states, win_size)

    # Predict for the next timestamp
    predictor = GaussianProcessPredictor(gp)
    next_timestamp = timestamps[i]
    test_prediction = predictor.predict([next_timestamp], selected_dim)
    predictions.append((test_prediction[0]["mean"].flatten()[0], test_prediction[1]["mean"].flatten()[0]))
    prediction_timestamps.append(next_timestamp)

# Step 5: Animation
fig, ax = plt.subplots(figsize=(8, 8))

# Initialize plots
gt_line, = ax.plot([], [], 'k--', label="Ground Truth")
pred_line, = ax.plot([], [], 'ro', label="Predictions")
measurement_scatter = ax.scatter([], [], c='blue', alpha=0.6, label="Measurements")

# Configure plot
ax.set_xlim(min(groundtruth_x) - 1, max(groundtruth_x) + 1)
ax.set_ylim(min(groundtruth_y) - 1, max(groundtruth_y) + 1)
ax.set_xlabel("State X")
ax.set_ylabel("State Y")
ax.set_title("Gaussian Process Sliding Window Prediction")
ax.legend()

# Animation update function
def update(frame):
    if frame < len(timestamps):  # Normal animation frames
        gt_line.set_data(groundtruth_x[:frame], groundtruth_y[:frame])

        if frame >= 5:  # Only show predictions after the 5th data point
            pred_x = [p[0] for p in predictions[:frame - 5]]
            pred_y = [p[1] for p in predictions[:frame - 5]]
            pred_line.set_data(pred_x, pred_y)
        
        measurement_scatter.set_offsets(np.c_[measurement_x[:frame], measurement_y[:frame]])
    else:  # Final frame: show all data
        gt_line.set_data(groundtruth_x, groundtruth_y)
        pred_x = [p[0] for p in predictions]
        pred_y = [p[1] for p in predictions]
        pred_line.set_data(pred_x, pred_y)
        measurement_scatter.set_offsets(np.c_[measurement_x, measurement_y])
    return gt_line, pred_line, measurement_scatter

# Include an extra frame for the final "snapshot" effect
ani = animation.FuncAnimation(fig, update, frames=len(timestamps) + 1, interval=100, blit=True, repeat=False)

# Step 6: Add play/pause button
class AnimationControl:
    def __init__(self, animation):
        self.animation = animation
        self.running = True

    def toggle_animation(self, event):
        if self.running:
            self.animation.event_source.stop()
        else:
            self.animation.event_source.start()
        self.running = not self.running

# Add a button to the plot
button_ax = plt.axes([0.8, 0.02, 0.1, 0.04])  # x, y, width, height
button = Button(button_ax, "Play/Pause")
animation_control = AnimationControl(ani)
button.on_clicked(animation_control.toggle_animation)

plt.show()

# Step 7: Calculate RMSE
# Compute RMSE for measurements
rmse_measurement_x = np.sqrt(np.mean((measurement_x[5:] - groundtruth_x[5:])**2))
rmse_measurement_y = np.sqrt(np.mean((measurement_y[5:] - groundtruth_y[5:])**2))

# Compute RMSE for predictions
rmse_prediction_x = np.sqrt(np.mean((np.array([p[0] for p in predictions]) - groundtruth_x[5:])**2))
rmse_prediction_y = np.sqrt(np.mean((np.array([p[1] for p in predictions]) - groundtruth_y[5:])**2))

print(f"RMSE for Measurements (X): {rmse_measurement_x}")
print(f"RMSE for Measurements (Y): {rmse_measurement_y}")
print(f"RMSE for Predictions (X): {rmse_prediction_x}")
print(f"RMSE for Predictions (Y): {rmse_prediction_y}")

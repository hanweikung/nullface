import numpy as np
import matplotlib.pyplot as plt


# Function to sample a vector within a specified angle with a set seed value, maintaining the magnitude
def random_vector_within_angle(vec, max_angle_deg, seed=None):
    """
    Sample a random vector within a specified angle from the given vector's
    direction and maintains the same magnitude, with an optional random seed.

    Parameters:
    vec (ndarray): The input vector to base the direction and magnitude on.
    max_angle_deg (float): The maximum allowed angle (in degrees) from the vector's direction.
    seed (int, optional): Seed for the random number generator.

    Returns:
    ndarray: A randomly sampled vector within the specified angle with the same magnitude.
    """
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility

    magnitude = np.linalg.norm(vec)  # Get the magnitude of the input vector
    vec = vec / magnitude  # Normalize the input vector to a unit vector

    # Convert the maximum angle to radians
    max_angle_rad = np.radians(max_angle_deg)

    # Sample a random angle within the max angle
    random_angle = np.random.uniform(0, max_angle_rad)

    # Sample a random direction in the plane perpendicular to `vec`
    random_direction = np.random.randn(len(vec))
    random_direction -= random_direction.dot(vec) * vec  # Orthogonalize
    random_direction = random_direction / np.linalg.norm(random_direction)  # Normalize

    # Create the random vector by rotating `vec` towards `random_direction`
    sampled_vec = np.cos(random_angle) * vec + np.sin(random_angle) * random_direction

    # Scale back to the original magnitude
    sampled_vec *= magnitude

    return sampled_vec


# Function to plot and save the 3D vectors as a PDF
def plot_and_save_vector_sampling(
    vec, random_vec, filename="vector_sampling_visualization.pdf"
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original vector (blue)
    ax.quiver(
        0,
        0,
        0,
        vec[0],
        vec[1],
        vec[2],
        color="b",
        label="Original Vector",
        arrow_length_ratio=0.1,
    )

    # Plot the random sampled vector (green)
    ax.quiver(
        0,
        0,
        0,
        random_vec[0],
        random_vec[1],
        random_vec[2],
        color="g",
        label="Random Sampled Vector",
        arrow_length_ratio=0.1,
    )

    # Set plot limits and labels
    ax.set_xlim([-1.5 * np.linalg.norm(vec), 1.5 * np.linalg.norm(vec)])
    ax.set_ylim([-1.5 * np.linalg.norm(vec), 1.5 * np.linalg.norm(vec)])
    ax.set_zlim([-1.5 * np.linalg.norm(vec), 1.5 * np.linalg.norm(vec)])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

    # Save the figure as a PDF
    fig.savefig(filename, format="pdf")
    plt.close(fig)


if __name__ == "__main__":
    # Example usage with seed:
    vec = np.array([1, 0, 0])  # Example vector
    max_angle = 30  # degrees
    seed = 42  # Set a seed for reproducibility

    # Generate a random vector within the specified angle
    random_vec = random_vector_within_angle(vec, max_angle, seed)

    # Plot and save the result
    plot_and_save_vector_sampling(
        vec, random_vec, filename="vector_sampling_with_seed.pdf"
    )

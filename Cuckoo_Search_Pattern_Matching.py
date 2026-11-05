
"""
Cuckoo Search Based Image Matching
Inspired by:
Zhang et al. (2017),
"The Study on Image Matching Method Based on Cuckoo Search"

Educational implementation, not an exact reproduction of the authors' MATLAB code.

Install:
    pip install numpy opencv-python scikit-image scipy matplotlib
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from scipy.special import gamma


# ============================================================
# CONFIGURATION
# ============================================================

TEMPLATE_PATH = "template.jpg"
SEARCH_IMAGE_PATH = "search_image.jpg"

N_NESTS = 250
MAX_ITERATIONS = 100
DISCOVERY_PROBABILITY = 0.5
LEVY_SCALE = 0.5
RANDOM_SEED = 42

# Exhaustive baseline checks every N pixels.
BASELINE_STRIDE = 4

# Set to (x, y) if the true top-left position is known.
# Otherwise leave as None.
GROUND_TRUTH = None


# ============================================================
# OBJECTIVE 1: LOAD IMAGES
# ============================================================

def load_image(path):
    """Load an image as grayscale."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Image not found: {path}\n"
            "Put the image in this folder or update its path."
        )

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


# ============================================================
# OBJECTIVE 2: EXTRACT HOG FEATURES
# ============================================================

def extract_hog_features(image):
    """
    Extract Histogram of Oriented Gradients features.

    Fixed parameters ensure the template and candidate image
    patches have feature vectors of the same length.
    """
    image = np.asarray(image, dtype=np.uint8)

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True
    )

    return features.astype(np.float64)


# ============================================================
# OBJECTIVE 3: SIMILARITY / FITNESS FUNCTION
# ============================================================

def correlation_similarity(features_a, features_b):
    """
    Pearson correlation between two HOG feature vectors.

    Returns a score in [-1, 1].
    Higher positive values indicate greater similarity.
    """
    a = features_a - np.mean(features_a)
    b = features_b - np.mean(features_b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator < 1e-12:
        # Avoid division by zero for constant feature vectors.
        return 1.0 if np.allclose(features_a, features_b) else 0.0

    return float(np.dot(a, b) / denominator)


# ============================================================
# OBJECTIVE 4: CANDIDATE NESTS
# Each nest represents a possible (x, y) location.
# ============================================================

class ImageMatchingProblem:

    def __init__(self, search_image, template):
        self.image = search_image
        self.template = template

        self.template_h, self.template_w = template.shape
        self.image_h, self.image_w = search_image.shape

        if (self.template_h > self.image_h or
                self.template_w > self.image_w):
            raise ValueError(
                "The template must be smaller than the search image."
            )

        # Inclusive coordinate bounds.
        self.max_x = self.image_w - self.template_w
        self.max_y = self.image_h - self.template_h

        self.template_features = extract_hog_features(template)

        # Cache repeated coordinate evaluations.
        self.cache = {}

        # Count actual fitness evaluations.
        self.evaluation_count = 0

    def clamp_position(self, position):
        """Round and clip a position to valid image coordinates."""
        x = int(np.clip(np.rint(position[0]), 0, self.max_x))
        y = int(np.clip(np.rint(position[1]), 0, self.max_y))
        return np.array([x, y], dtype=int)

    def fitness(self, position):
        """Evaluate the HOG similarity at a candidate location."""
        position = self.clamp_position(position)
        x, y = int(position[0]), int(position[1])
        key = (x, y)

        if key in self.cache:
            return self.cache[key]

        patch = self.image[
            y:y + self.template_h,
            x:x + self.template_w
        ]

        patch_features = extract_hog_features(patch)

        score = correlation_similarity(
            self.template_features,
            patch_features
        )

        self.cache[key] = score
        self.evaluation_count += 1

        return score


# ============================================================
# OBJECTIVE 5: LEVY FLIGHT
# Mantegna's method for generating Levy-distributed steps.
# ============================================================

def levy_flight(beta=1.5, size=2, rng=None):
    """Generate Levy-flight step values using Mantegna's method."""
    if rng is None:
        rng = np.random.default_rng()

    sigma_u = (
        gamma(1 + beta) * np.sin(np.pi * beta / 2)
        / (
            gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = rng.normal(0, sigma_u, size=size)
    v = rng.normal(0, 1, size=size)

    return u / (np.abs(v) ** (1 / beta) + 1e-12)


# ============================================================
# OBJECTIVE 6: CUCKOO SEARCH OPTIMIZATION
# ============================================================

def cuckoo_search(problem,
                  n_nests=N_NESTS,
                  max_iterations=MAX_ITERATIONS,
                  pa=DISCOVERY_PROBABILITY,
                  alpha=LEVY_SCALE,
                  seed=RANDOM_SEED):
    """
    Search for the target location using Cuckoo Search.

    Maximizes HOG correlation similarity.
    """
    rng = np.random.default_rng(seed)

    lower = np.array([0.0, 0.0])
    upper = np.array([problem.max_x, problem.max_y])

    # Randomly initialize nests.
    nests = rng.uniform(
        lower, upper, size=(n_nests, 2)
    )
    nests = np.array([
        problem.clamp_position(nest) for nest in nests
    ])

    scores = np.array([
        problem.fitness(nest) for nest in nests
    ])

    best_index = int(np.argmax(scores))
    best_position = nests[best_index].copy()
    best_score = float(scores[best_index])

    history = [best_score]

    for iteration in range(max_iterations):

        # Phase A: Levy-flight-inspired exploration.
        for i in range(n_nests):
            current = nests[i].astype(float)

            step = levy_flight(
                beta=1.5, size=2, rng=rng
            )

            # Relative movement toward/away from current best.
            candidate = (
                current
                + alpha * step * (best_position - current)
            )

            # Occasional independent jump improves exploration.
            if rng.random() < 0.15:
                candidate += (
                    rng.normal(0, 0.08, size=2)
                    * (upper - lower + 1)
                )

            candidate = problem.clamp_position(candidate)
            candidate_score = problem.fitness(candidate)

            # Greedy nest replacement.
            if candidate_score > scores[i]:
                nests[i] = candidate
                scores[i] = candidate_score

        # Phase B: discovery and replacement of nests.
        for i in range(n_nests):
            if rng.random() > pa:
                j, k = rng.choice(n_nests, size=2, replace=False)

                r = rng.random()

                candidate = nests[i].astype(float) + r * (
                    nests[j].astype(float)
                    - nests[k].astype(float)
                )

                candidate = problem.clamp_position(candidate)
                candidate_score = problem.fitness(candidate)

                if candidate_score > scores[i]:
                    nests[i] = candidate
                    scores[i] = candidate_score

        # Preserve the best result found so far.
        current_best = int(np.argmax(scores))

        if scores[current_best] > best_score:
            best_score = float(scores[current_best])
            best_position = nests[current_best].copy()

        history.append(best_score)

        if (iteration + 1) % 10 == 0:
            print(
                f"Iteration {iteration + 1:3d}/"
                f"{max_iterations}: "
                f"best score = {best_score:.5f}, "
                f"location = ({best_position[0]}, "
                f"{best_position[1]})"
            )

    return best_position.astype(int), best_score, history


# ============================================================
# OBJECTIVE 7: EXHAUSTIVE SLIDING-WINDOW BASELINE
# ============================================================

def exhaustive_matching(problem, stride=4):
    """
    Compare CS with a grid-based exhaustive search.

    A stride greater than one reduces runtime but means this
    is a coarse grid search rather than every possible pixel.
    """
    start = time.perf_counter()

    best_position = np.array([0, 0])
    best_score = -np.inf

    for y in range(0, problem.max_y + 1, stride):
        for x in range(0, problem.max_x + 1, stride):
            score = problem.fitness((x, y))

            if score > best_score:
                best_score = score
                best_position = np.array([x, y])

    # Also check the bottom-right boundary.
    score = problem.fitness((problem.max_x, problem.max_y))

    if score > best_score:
        best_score = score
        best_position = np.array([problem.max_x, problem.max_y])

    elapsed = time.perf_counter() - start

    return best_position, float(best_score), elapsed


# ============================================================
# OBJECTIVE 8: EVALUATION METRICS
# ============================================================

def localization_error(predicted, ground_truth):
    """Euclidean distance between predicted and true top-left."""
    return float(np.linalg.norm(
        np.asarray(predicted, dtype=float)
        - np.asarray(ground_truth, dtype=float)
    ))


def draw_match(image, template, position, title):
    """Display the detected region."""
    x, y = map(int, position)

    output = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    cv2.rectangle(
        output,
        (x, y),
        (x + template.shape[1] - 1,
         y + template.shape[0] - 1),
        (255, 0, 0),
        2
    )

    plt.imshow(output)
    plt.title(title)
    plt.axis("off")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    search_image = load_image(SEARCH_IMAGE_PATH)
    template = load_image(TEMPLATE_PATH)

    print("\n--- IMAGE MATCHING CONFIGURATION ---")
    print("Search image size:", search_image.shape)
    print("Template size:", template.shape)
    print("Number of nests:", N_NESTS)
    print("Maximum iterations:", MAX_ITERATIONS)
    print("Discovery probability:", DISCOVERY_PROBABILITY)
    print("Levy scale:", LEVY_SCALE)

    # Independent problem instances keep evaluation counts
    # and caches separate for the two algorithms.
    cs_problem = ImageMatchingProblem(search_image, template)

    # Run Cuckoo Search.
    start = time.perf_counter()

    cs_position, cs_score, history = cuckoo_search(cs_problem)

    cs_time = time.perf_counter() - start
    cs_evaluations = cs_problem.evaluation_count

    print("\n--- CUCKOO SEARCH RESULT ---")
    print("Best location (x, y):", tuple(cs_position))
    print(f"Best HOG correlation: {cs_score:.6f}")
    print(f"Execution time: {cs_time:.4f} seconds")
    print("Unique fitness evaluations:", cs_evaluations)

    # Run coarse exhaustive baseline.
    baseline_problem = ImageMatchingProblem(search_image, template)

    baseline_position, baseline_score, baseline_time = (
        exhaustive_matching(
            baseline_problem,
            stride=BASELINE_STRIDE
        )
    )

    print("\n--- EXHAUSTIVE BASELINE RESULT ---")
    print("Best location (x, y):", tuple(baseline_position))
    print(f"Best HOG correlation: {baseline_score:.6f}")
    print(f"Execution time: {baseline_time:.4f} seconds")
    print(
        "Unique fitness evaluations:",
        baseline_problem.evaluation_count
    )

    # Ground-truth localization accuracy, if known.
    if GROUND_TRUTH is not None:
        cs_error = localization_error(
            cs_position, GROUND_TRUTH
        )
        baseline_error = localization_error(
            baseline_position, GROUND_TRUTH
        )

        print("\n--- LOCALIZATION ACCURACY ---")
        print("Ground truth (x, y):", GROUND_TRUTH)
        print(f"CS localization error: {cs_error:.2f} pixels")
        print(
            f"Baseline localization error: "
            f"{baseline_error:.2f} pixels"
        )

    # Visualize the results.
    plt.figure(figsize=(15, 7))

    plt.subplot(2, 2, 1)
    plt.imshow(search_image, cmap="gray")
    plt.title("Original Search Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(template, cmap="gray")
    plt.title("Template Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    draw_match(
        search_image,
        template,
        cs_position,
        f"Cuckoo Search (score={cs_score:.3f})"
    )

    plt.subplot(2, 2, 4)
    draw_match(
        search_image,
        template,
        baseline_position,
        f"Grid Baseline (score={baseline_score:.3f})"
    )

    plt.tight_layout()
    plt.show()

    # Plot convergence history.
    plt.figure(figsize=(8, 4))
    plt.plot(history, color="darkblue")
    plt.xlabel("Iteration")
    plt.ylabel("Best HOG correlation")
    plt.title("Cuckoo Search Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n--- SUMMARY ---")
    print("CS score:", round(cs_score, 6))
    print("Baseline score:", round(baseline_score, 6))
    print("CS time (s):", round(cs_time, 4))
    print("Baseline time (s):", round(baseline_time, 4))
    print("\nMatching completed.")


if __name__ == "__main__":
    main()

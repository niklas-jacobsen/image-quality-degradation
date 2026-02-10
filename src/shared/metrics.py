import cv2 as cv
import numpy as np

def get_brightness(img_gray: np.ndarray) -> float:
    """Average pixel intensity (0-255). Lower is darker."""
    return float(np.mean(img_gray))

def get_contrast(img_gray: np.ndarray) -> float:
    """RMS contrast (Standard Deviation). Lower is washed out."""
    return float(img_gray.std())

def get_sharpness(img_gray: np.ndarray) -> float:
    """Laplacian Variance. Lower is blurrier."""
    return float(cv.Laplacian(img_gray, cv.CV_64F).var())

def get_entropy(img_gray: np.ndarray) -> float:
    """Shannon Entropy. Measures information contennt. lower is less detail (e.g. fog/snow)."""
    #calculate histogram
    hist = cv.calcHist([img_gray], [0], None, [256], [0, 256])
    #normalize
    hist_norm = hist.ravel() / hist.sum()
    #filter non-zero values to avoid log(0)
    hist_norm = hist_norm[hist_norm > 0]
    #entropy formula
    return float(-np.sum(hist_norm * np.log2(hist_norm)))

def get_overexposure(img_gray: np.ndarray) -> float:
    """Fraction of pixels that are pure white (clipped)."""
    #count pixels > 250 (near white)
    white_pixels = np.count_nonzero(img_gray > 250)
    return float(white_pixels / img_gray.size)

def measure_all(img: np.ndarray) -> dict:
    """Runs all metrics on an image and returns a profile."""
    #convert to grayscale once to save some time
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    return {
        "brightness": get_brightness(gray),
        "contrast": get_contrast(gray),
        "sharpness": get_sharpness(gray),
        "entropy": get_entropy(gray),
        "overexposure": get_overexposure(gray)
    }



import numpy as np
import cv2 as cv
import albumentations as A
from dataclasses import dataclass
from typing import Protocol
import random

class ModifierStrategy(Protocol):
    """
    Strategy interface for image modifications.
    Implementations must provide apply(img, pass_index) -> numpy array
    """

    name: str

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        ...

@dataclass
class ResolutionModifier:
    """
    Downscale by a step size per pass.
    step_percent: positive value, percent reduction per pass, e.g. 10 for 10 percent smaller per pass.
    min_scale: minimum allowed scale factor, e.g. 0.1
    """
    name: str = "Resolution"
    step_percent: float = 10.0
    min_scale: float = 0.05

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        """
        Compute scale factor for this pass, then downscale and upscale back to original size,
        returning a same-sized image but degraded.
        pass_index starts at 0 for the original image, but by requirement we will consider passes from 1..N.
        """
        if pass_index <= 0:
            return img.copy()
        factor = max(self.min_scale, 1.0 - (self.step_percent / 100.0) * pass_index)
        if factor >= 0.9999:
            return img.copy()
        h, w = img.shape[:2]
        new_w = max(1, int(round(w * factor)))
        new_h = max(1, int(round(h * factor)))
        # downscale then upscale
        small = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
        degraded = cv.resize(small, (w, h), interpolation=cv.INTER_LINEAR)
        return degraded
    
@dataclass
class LuminanceModifier:
    name: str = "Luminance"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        """
        Apply brightness reduction based on step_percent per pass.
        Uses OpenCV-compatible image arrays (numpy BGR uint8).
        """
        if pass_index <= 0:
            return img.copy()

        factor = max(0.1, 1.0 - (self.step_percent / 100.0) * pass_index)

        bright_img = cv.convertScaleAbs(img, alpha=factor, beta=0)
        return bright_img


@dataclass
class BlurModifier:
    name: str = "Blur"
    max_ksize: int = 45
    step_percent: float = 10
    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        """
        Apply Gaussian blur whose kernel size increases with pass_index and step_percent.
        The kernel size is always odd and capped at max_ksize.
        """
        if pass_index <= 0:
            return img.copy()

        # compute kernel size as a percentage of max_ksize, scaled by pass_index
        k = min(3 + pass_index * 3, self.max_ksize)
        # ensure odd and at least 3 for visible blur
        k = max(3, k | 1)  
        k = min(k, self.max_ksize)  

        return cv.GaussianBlur(img, (k, k), 0)


@dataclass
class ContrastModifier:
    name: str = "Contrast"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        """
        Reduces contrast by interpolating towards a middle grey (128).
        pass_index 0 = Original Image
        pass_index 10 = Solid Grey Image (if step is 10%)
        """
        if pass_index <= 0:
            return img.copy()

        alpha = max(0.0, 1.0 - (self.step_percent / 100.0) * pass_index)
        
        # create a grey image (128 is the middle of 0-255)
        grey_img = np.full_like(img, 128)
    
        low_contrast = cv.addWeighted(img, alpha, grey_img, 1.0 - alpha, 0)
        
        return low_contrast
    
    def label_for_pass(self, pass_index: int) -> str:
        factor = max(0.0, 1.0 - (self.step_percent / 100.0) * pass_index)
        return f"Contrast: {factor:.2f}x"

@dataclass
class NoiseModifier:
    name: str = "Noise"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()
        # increase variance with pass
        base_sigma = (self.step_percent * pass_index) * 2.0
        sigma = abs(base_sigma)

        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

@dataclass
class DesaturationModifier:
    name: str = "Saturation"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
        #reduce S channel
        factor = max(0.0, 1.0 - (self.step_percent / 100.0) * pass_index)
        hsv[:, :, 1] *= factor
        return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

# SEMANTIC MODIFIERS (albumentations wrappers)

@dataclass
class MotionBlurModifier:
    name: str = "MotionBlur"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()
        
        #force determinism
        random.seed(42 + pass_index)
        np.random.seed(42 + pass_index)

        intensity = (self.step_percent / 100.0) * pass_index
        k_size = int(3 + (30 * intensity))
        if k_size % 2 == 0: k_size += 1
        
        transform = A.MotionBlur(blur_limit=(k_size, k_size), p=1.0)
        return transform(image=img)["image"]

    def label_for_pass(self, pass_index: int) -> str:
        return f"Motion Blur Lvl {pass_index}"

@dataclass
class RainModifier:
    name: str = "Rain"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()
        
        random.seed(42 + pass_index)
        np.random.seed(42 + pass_index)
        
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        intensity = (self.step_percent / 100.0) * pass_index
        bright_coeff = max(0.5, 1.0 - (0.4 * intensity))
        
        transform = A.RandomRain(
            brightness_coefficient=bright_coeff,
            drop_width=1, 
            blur_value=2, 
            p=1.0
        )
        aug = transform(image=img_rgb)["image"]
        return cv.cvtColor(aug, cv.COLOR_RGB2BGR)

    def label_for_pass(self, pass_index: int) -> str:
        return f"Rain Lvl {pass_index}"

@dataclass
class FogModifier:
    name: str = "Fog"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()

        random.seed(42 + pass_index)
        np.random.seed(42 + pass_index)
        
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        intensity = min(1.0, (self.step_percent / 100.0) * pass_index)
        fog_coef_lower = min(1.0, 0.1 * intensity)
        fog_coef_upper = min(1.0, 1.0 * intensity)
        alpha_coef = min(1.0, 0.08 * intensity)

        transform = A.RandomFog(
            fog_coef_lower=fog_coef_lower, 
            fog_coef_upper=fog_coef_upper, 
            alpha_coef=alpha_coef, 
            p=1.0
        )
        aug = transform(image=img_rgb)["image"]
        return cv.cvtColor(aug, cv.COLOR_RGB2BGR)

    def label_for_pass(self, pass_index: int) -> str:
        return f"Fog Lvl {pass_index}"

@dataclass
class ShadowModifier:
    name: str = "Shadow"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()
        
        random.seed(42 + pass_index)
        np.random.seed(42 + pass_index)
        
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        intensity = (self.step_percent / 100.0) * pass_index
        num_shadows = int(1 + (5 * intensity))
        
        transform = A.RandomShadow(
            num_shadows_lower=num_shadows, 
            num_shadows_upper=num_shadows + 1, 
            shadow_dimension=5, 
            shadow_roi=(0, 0.5, 1, 1), 
            p=1.0
        )
        aug = transform(image=img_rgb)["image"]
        return cv.cvtColor(aug, cv.COLOR_RGB2BGR)

    def label_for_pass(self, pass_index: int) -> str:
        return f"Shadow Lvl {pass_index}"

@dataclass
class SnowModifier:
    name: str = "Snow"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()

        random.seed(42 + pass_index)
        np.random.seed(42 + pass_index)
        
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        intensity = (self.step_percent / 100.0) * pass_index
        snow_val = 0.1 + (0.4 * intensity)
        
        snow_point_lower = min(1.0, snow_val)
        snow_point_upper = min(1.0, snow_val + 0.1)
        
        transform = A.RandomSnow(
            snow_point_range=(snow_point_lower, snow_point_upper), 
            brightness_coeff=2.5, 
            p=1.0
        )
        aug = transform(image=img_rgb)["image"]
        return cv.cvtColor(aug, cv.COLOR_RGB2BGR)

    def label_for_pass(self, pass_index: int) -> str:
        return f"Snow Lvl {pass_index}"

@dataclass
class ChromaticAberrationModifier:
    name: str = "ChromAberr"
    step_percent: float = 10.0

    def apply(self, img: np.ndarray, pass_index: int) -> np.ndarray:
        if pass_index <= 0: return img.copy()
        
        random.seed(42 + pass_index)
        np.random.seed(42 + pass_index)
        
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        intensity = (self.step_percent / 100.0) * pass_index
        
        transform = A.ChromaticAberration(
            primary_distortion_limit=0.5 * intensity,
            secondary_distortion_limit=0.1 * intensity,
            p=1.0
        )
        aug = transform(image=img_rgb)["image"]
        return cv.cvtColor(aug, cv.COLOR_RGB2BGR)

    def label_for_pass(self, pass_index: int) -> str:
        return f"ChromAberr Lvl {pass_index}"

MODIFIER_REGISTRY = {
    "resolution": ResolutionModifier,
    "luminance": LuminanceModifier,
    "contrast": ContrastModifier,
    "blur": BlurModifier,
    "noise": NoiseModifier,
    "saturation": DesaturationModifier,
    "motion_blur": MotionBlurModifier,
    "rain": RainModifier,
    "fog": FogModifier,
    "shadow": ShadowModifier,
    "snow": SnowModifier,
    "chromatic_aberration": ChromaticAberrationModifier
}

SYNTACTIC_GROUP = ["resolution", "luminance", "contrast", "blur", "noise", "saturation"]
SEMANTIC_GROUP = ["motion_blur", "rain", "fog", "shadow", "snow", "chromatic_aberration"]

def resolve_modifier_list(selection):
    """
    Accepts a single string (e.g. 'blur', 'all', 'syntactic') 
    or a list of strings (['blur', 'noise']).
    Returns a LIST of modifier names (strings).
    """
    if isinstance(selection, str):
        if selection == "all":
            return list(MODIFIER_REGISTRY.keys())
        elif selection == "syntactic":
            return SYNTACTIC_GROUP
        elif selection == "semantic":
            return SEMANTIC_GROUP
        else:
            #single specific modifier
            return [selection]
    elif isinstance(selection, list):
        resolved = []
        for item in selection:
            resolved.extend(resolve_modifier_list(item))
        return sorted(list(set(resolved))) # dedupe
    return []

# BSS scale disambiguation algorithms
# Copyright (C) 2020  Robin Scheibler
from .algorithms import projection_back, minimum_distortion

algorithms = {
    "projection_back": projection_back,
    "minimum_distortion": minimum_distortion,
}

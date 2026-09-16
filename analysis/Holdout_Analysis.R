library(tidyverse)
library(readxl)
library(broom)

holdout_files = list.files(
  path = "../results",
  pattern = "holdout",
  full.names = TRUE
)
frozen_models = list.files(
  path = "../results",
  pattern = "frozen",
  full.names = TRUE
)
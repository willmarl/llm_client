- smoketest lite (full can come later)

  make all the tests in root, then make test_menu.py file to ask which tests to run (find menu lib like how vite does it)
  move all to llm_client, maybe rename folder from tests to smoketest

- add to config, toggle debug print texts
- may want to make loaders.py give error if trying to load images
- make my chroma client accept images
- touch up docs and make it pretty and consistent
- (TORCH) if wanting to use local CLIP then enable it in env via (amd, or nvidia)
  - sanity check first.
    IF AMD
    see if `rocminfo` prints
    IF NVIDIA
    see if `nvidia-smi` prints

  go to this repo and in separate env, test torch can detect and use ur gpu
  repo AMD
  repo NVIDIA

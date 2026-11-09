#!/usr/bin/env bash
nvidia_libs=/cache/robot-baselines/system-libs/nvidia
export LD_LIBRARY_PATH=$nvidia_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=$nvidia_libs/usr/share/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=$nvidia_libs/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

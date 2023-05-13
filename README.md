# Weeping Angel Script

The Weeping Angels are one of the most iconic and terrifying creatures in the Doctor Who universe. They are known for their ability to move only when they are not being observed. Similarly, this script detects blinks, symbolizing the Weeping Angels freezing when someone looks at them. So, watch out for those blinks, or you might find yourself surrounded by these quantum-locked creatures!

Remember, don't blink! Blink and you're... well, maybe not sent back in time, but you might miss something important!

## Script

This script detects blinks in a live video stream or a video file by analyzing eye landmarks. It uses the dlib library for face and landmark detection, OpenCV for video rendering, and scipy for calculating the Eye Aspect Ratio (EAR).

> **Why did I create this script?**
> As a Doctor Who fan, I couldn't help but be fascinated by the eerie and mysterious nature of the Weeping Angels. Inspired by these iconic villains, I decided to create a script that detects blinks, just like the Weeping Angels freeze when you're looking at them. But don't worry, this script won't send you back in time!

## Installation

Activate the conda environment

```
conda env create -n weeping --file environment.yml
```

## Usage

Once you have set up the conda environment, you can use the script to detect blinks. Follow the steps below to run the script:

Place the video file you want to analyze in the same directory as the script or provide the full path to the video file.

In the terminal, make sure the conda environment is activated:
```
python angel.py
```

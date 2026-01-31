## Motion Detection System in Video Streams

### Problem Description

The designed system is intended for automatic motion detection in video streams
captured by a surveillance camera.
Here, motion is defined as changes in the observed scene associated with the
appearance, movement, or disappearance of objects in the frame.

The system is considered as an autonomous engineering module that processes
video data and generates a signal indicating the presence or absence of motion
over time.

Task type: classification (motion / no motion).

Additionally, the system may perform motion event attribution by determining
the presumed type of motion initiator (e.g., human / vehicle / animal / unknown)
for each detected activity interval. Attribution is used for event prioritization
and notification filtering and does not aim at precise spatial localization of
objects as a primary output.

---

### System Input

The system input is a video stream or a video file.

Input data characteristics:
* data type: video, 
* format: standard formats (.mp4, .avi, .mov), 
* frame rate: from 5 to 30 frames per second (NTSC standard, typical values), 
* resolution: arbitrary, 
* color space: color or monochrome images.

Input data constraints:
* the camera is assumed to be predominantly static, 
* noise, compression artifacts, illumination changes, and similar effects are possible, 
* weather effects such as rain or snow are allowed.

---

### System Output

The system output represents information about motion presence over time.

Output data format:
* a binary motion presence indicator for each frame or temporal window, 
* a list of activity intervals in the format (start time, end time), 
* the type of motion initiator for each activity interval.

Output interpretation:
* the signal is used to trigger event recording and activate video capture, 
* absence of motion is interpreted as a static scene state, 
* initiator classification is used to reduce false alarms and prioritize events.

---

### Application Context

The system is intended for use in video surveillance and video analytics tasks, 
including:
* preliminary filtering of video streams to eliminate empty scenes, 
* automatic activity detection for security and monitoring systems, 
* activation of subsequent analytical modules only when motion is detected.

The system can operate both in offline video analysis mode and in near-real-time
processing scenarios.

---

### Critical and Acceptable Errors

The following error types are considered critical:
* false positive detections leading to frequent unjustified alarms, 
* unstable system behavior where motion signals appear and disappear erratically
  in the absence of real scene changes, 
* incorrect attribution of the motion initiator, resulting in improper event
  handling when the system is used for security or notification purposes.

Acceptable errors include:
* missing subtle or very slow motion, 
* motion detection delay of several frames, 
* errors caused by abrupt illumination changes (e.g., lights turning on), 
* assigning an “unknown” initiator class when confidence is insufficient, 
  provided that the motion event itself is correctly detected.

---

### System Constraints

At the design stage, the following system constraints are assumed:
* operation on limited computational resources (CPU), 
* minimal latency between motion onset and signal generation, 
* lack of semantic understanding of objects in the scene, 
* requirement to adjust system sensitivity for a specific scene, 
* robustness to moderate noise and minor changes in capture conditions, 
* motion initiator attribution is performed only when motion is detected and
  is not guaranteed to be reliable under strong occlusions, low image quality, 
  nighttime conditions, or when objects occupy a small portion of the frame, 
* the system is not intended for personal identification or recognition of
  specific objects by unique features; attribution is performed only at the
  level of initiator categories.

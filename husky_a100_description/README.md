# husky_a100_description

Parametric URDF of the Clearpath Husky A100 (2010): six wheels, three per side, chained together. No official A100 URDF exists: the published `husky_description` packages describe the four-wheel A200. This package takes the wheel mesh, the wheel macro, the materials and the "extras" mechanism from the `noetic-devel` branch of `husky/husky_description` (BSD license, see `meshes/LICENSE.husky_description`), and builds the body from primitives.

All dimensions are in `config/a100_dimensions.yaml`, each with its source: the user guide, measurements on its scaled drawing, or measurements on the real robot.

## Frames

```
base_footprint                                   on the ground, under the centre of the robot
  └ base_link                                    at axle height
      ├ {front,middle,rear}_{left,right}_wheel_link
      ├ {left,right}_rail_link
      ├ {front,rear}_bumper_link
      └ top_chassis_link                         centre of the top plate, where sensors are mounted
```

The wheel joint names match the ones published by the Clearpath driver, so the wheels turn in RViz with its `joint_states`. The middle wheels mimic the front ones.

## Use

```bash
ros2 launch husky_a100_description display.launch.py
```

`dimensions_file:=` loads another dimensions file, and `urdf_extras:=` adds a xacro file with sensors attached to `top_chassis_link`. `urdf/sensors.urdf.xacro` holds the sensors of our robot.

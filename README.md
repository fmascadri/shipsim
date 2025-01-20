# Simulation of ship dynamics using strip theory

This project simulates ship motion dynamics using strip theory. 

This is still a work in progress: so far only the slicer has been implemented.

## Slicer

The slicer takes an .stl file of a ship (or any other object) as an input and generates contours of equally spaced plane intersections.

### Usage
`python3 slicer.py [filename] [number of planes] [number of decimal places accuracy for vertex comparisons]`

For example

`python3 slicer.py ./examples/models/TugBoat.stl 20 4`

### Examples
#### Sphere
![Sphere mesh](https://github.com/user-attachments/assets/82dc6098-161b-42bf-bff0-187f3b92e304)
![Sphere 5 slices](https://github.com/user-attachments/assets/41901d4c-22fb-4e43-a653-3a1398eba5c9)
![Sphere 10 slices](https://github.com/user-attachments/assets/2213f9f8-c57e-4251-9b4d-ce5bddc8c7f7)
![Sphere 15 slices](https://github.com/user-attachments/assets/642a68b9-a814-4204-8b8c-c15e59d905b5)
![Sphere 20 slices](https://github.com/user-attachments/assets/92d0c273-2459-49fd-8590-370ad698cd37)

#### Tugboat
![Tugboat mesh](https://github.com/user-attachments/assets/12c7c18f-b2b5-419f-8aae-a85189019a06)
![Tugboat 5 slices](https://github.com/user-attachments/assets/db8f6252-3ecd-4261-949a-8366f78a2127)
![Tugboat 10 slices](https://github.com/user-attachments/assets/92998849-2ee2-4d4d-b842-600510623f9b)
![Tugboat 15 slices](https://github.com/user-attachments/assets/974e4a20-389f-4cfe-820d-c1fefbc2139f)
![Tugboat 20 slices](https://github.com/user-attachments/assets/c7dc525b-7331-4bc0-b6a2-8048df48cf35)






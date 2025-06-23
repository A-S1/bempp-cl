// Parameters
Length     = 1.0;
Separation = 0.1;
lc         = 0.1;  // global mesh size

// Define corner points
Point(1) = {0,           0,          0};
Point(2) = {1.0,      0,          0};
Point(3) = {1.0,      0.01, 0};
Point(4) = {0, 0.01, 0};

// Define lines: bottom wire, right connector, top wire, left connector
Line(1) = {1, 2};  // bottom wire
Line(2) = {2, 3};  // right connector
Line(3) = {3, 4};  // top wire
Line(4) = {4, 1};  // left connector

// Transfinite mesh: 20 segments along the parallel wires, 1 along each connector
Transfinite Line {1} = 21;
Transfinite Line {3} = 21;
Transfinite Line {2} = 2;
Transfinite Line {4} = 2;

// Define physical groups for clarity
Physical Curve("Wires")     = {1, 3};
Physical Curve("Connectors") = {2, 4};

// Generate a 1D mesh
Mesh 1;
